#![allow(clippy::expect_used)]

mod support;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;
use std::{fs, io};

use hyper_util::rt::TokioIo;
use nexa_memoryd::db::Store;
use nexa_memoryd::service::{BoundSocket, MemoryRpc};
use nexa_proto::v1::memory_service_client::MemoryServiceClient;
use nexa_proto::v1::*;
use proptest::prelude::*;
use tempfile::{Builder, TempDir};
use tokio::net::UnixStream;
use tokio::sync::oneshot;
use tonic::{
    Code,
    transport::{Channel, Endpoint},
};
use tower::service_fn;

use support::*;

fn tempdir() -> TempDir {
    Builder::new()
        .prefix("nexa-memory-")
        .tempdir_in("/tmp")
        .expect("short tempdir for UDS")
}

type Client = MemoryServiceClient<Channel>;

async fn connect(path: &Path) -> Client {
    let path = path.to_owned();
    let channel = Endpoint::from_static("http://[::]:50051")
        .timeout(Duration::from_secs(10))
        .connect_with_connector(service_fn(move |_| {
            let path = path.clone();
            async move { UnixStream::connect(path).await.map(TokioIo::new) }
        }))
        .await
        .expect("connect over UDS");
    Client::new(channel)
}

struct EndpointFixture {
    client: Client,
    shutdown: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<Result<(), tonic::transport::Error>>,
    socket_path: PathBuf,
}

impl EndpointFixture {
    async fn start(dir: &Path, global: bool) -> Self {
        let mut store = Store::open(&dir.join("data")).expect("open disk databases");
        store
            .configure_voice_attribution(global)
            .expect("configure global voice switch");
        let socket = BoundSocket::bind(&dir.join("run")).expect("bind memory.sock");
        let socket_path = socket.path().to_owned();
        let (shutdown, signal) = oneshot::channel();
        let task = tokio::spawn(socket.serve(MemoryRpc::new(store, true), async {
            let _ = signal.await;
        }));
        let client = connect(&socket_path).await;
        Self {
            client,
            shutdown,
            task,
            socket_path,
        }
    }

    async fn stop(self) {
        drop(self.client);
        self.shutdown.send(()).expect("send shutdown");
        tokio::time::timeout(Duration::from_secs(10), self.task)
            .await
            .expect("shutdown timeout")
            .expect("join daemon")
            .expect("serve result");
        assert!(!self.socket_path.exists());
    }
}

#[tokio::test]
async fn delete_fact_scrubs_plaintext_from_database_and_wal_before_ack() {
    const MARKER: &str = "synthetic-delete-fact-plaintext-marker";
    let dir = tempdir();
    let mut endpoint = EndpointFixture::start(dir.path(), false).await;
    let mut request = fact("source", "alice", 0);
    request.content = MARKER.into();
    let stored = endpoint
        .client
        .store_fact(request)
        .await
        .expect("store fact")
        .into_inner();
    let database = dir.path().join("data/memory.db");
    let wal = dir.path().join("data/memory.db-wal");
    {
        // Materialize the fact in the database while retaining its WAL frames.
        let reader = rusqlite::Connection::open(&database).expect("open database for checkpoint");
        let busy: i64 = reader
            .query_row("PRAGMA wal_checkpoint(FULL)", [], |row| row.get(0))
            .expect("checkpoint fact");
        assert_eq!(busy, 0, "checkpoint completed");
    }
    // Put the fact page back in the WAL and keep the same vec0 chunk live.
    // Dropping an empty chunk could cross SQLite's auto-checkpoint threshold.
    endpoint
        .client
        .store_fact(fact("keep", "alice", 1))
        .await
        .expect("store unrelated fact in the same owner partition");
    let contains_marker = |bytes: &[u8]| {
        bytes
            .windows(MARKER.len())
            .any(|window| window == MARKER.as_bytes())
    };
    for path in [&database, &wal] {
        let bytes = fs::read(path).expect("read seeded database file");
        assert!(
            contains_marker(&bytes),
            "missing test marker in {}",
            path.display()
        );
    }

    let receipt = endpoint
        .client
        .delete_fact(DeleteFactRequest {
            fact_id: stored.fact_id,
            actor: Some(home("alice", &[Permission::MemoryOwnWrite])),
        })
        .await
        .expect("owner deletes fact")
        .into_inner();
    let database_bytes = fs::read(&database).expect("read database immediately after DeleteFact");
    let wal_bytes = match fs::read(&wal) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
        Err(error) => panic!("read WAL immediately after DeleteFact: {error}"),
    };
    assert_eq!((receipt.facts_deleted, receipt.vectors_deleted), (1, 1));
    assert!(
        !contains_marker(&database_bytes) && !contains_marker(&wal_bytes),
        "plaintext after ACK (database: {}, WAL: {})",
        contains_marker(&database_bytes),
        contains_marker(&wal_bytes),
    );
    endpoint.stop().await;
}

#[tokio::test]
async fn uds_permissions_erase_and_restart() {
    let dir = tempdir();
    let mut endpoint = EndpointFixture::start(dir.path(), true).await;
    let metadata = endpoint.socket_path.metadata().expect("socket metadata");
    assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
    assert!(BoundSocket::bind(&dir.path().join("run")).is_err());
    assert!(Store::open(&dir.path().join("data")).is_err());

    let health = endpoint
        .client
        .get_health(actor_metadata(
            Empty {},
            &operator(&[Permission::DiagnosticsRead]),
        ))
        .await
        .expect("health")
        .into_inner();
    assert_eq!(health.state, HealthState::Ok as i32);
    assert_eq!(
        endpoint
            .client
            .get_health(Empty {})
            .await
            .expect_err("no UDS actor")
            .code(),
        Code::Unauthenticated
    );
    endpoint
        .client
        .store_fact(fact("source", "", 0))
        .await
        .expect("store fact");
    endpoint
        .client
        .store_fact(fact("keep", "", 1))
        .await
        .expect("store unrelated fact");
    endpoint
        .client
        .record_voice_observation(observation("source", 0))
        .await
        .expect("capture");

    let receipt = endpoint
        .client
        .erase_conversation(erase("source"))
        .await
        .expect("erase")
        .into_inner();
    assert!(receipt.tombstone_recorded);
    assert_eq!(
        (
            receipt.facts_deleted,
            receipt.vectors_deleted,
            receipt.captures_deleted
        ),
        (1, 2, 1)
    );
    endpoint.stop().await;

    let mut endpoint = EndpointFixture::start(dir.path(), true).await;
    assert_eq!(
        endpoint
            .client
            .erase_conversation(erase("source"))
            .await
            .expect("retry erase")
            .into_inner(),
        receipt
    );
    assert_eq!(
        endpoint
            .client
            .store_fact(fact("source", "", 0))
            .await
            .expect_err("erased fact source")
            .code(),
        Code::FailedPrecondition
    );
    assert_eq!(
        endpoint
            .client
            .record_voice_observation(observation("source", 0))
            .await
            .expect_err("erased capture")
            .code(),
        Code::FailedPrecondition
    );
    let results = endpoint
        .client
        .query_memory(query("", true, 10))
        .await
        .expect("query after restart")
        .into_inner();
    assert_eq!(results.hits.len(), 1);
    assert_eq!(
        results.hits[0].fact.as_ref().expect("fact").conversation_id,
        "keep"
    );
    assert!(
        endpoint
            .client
            .list_voice_profiles(list_profiles())
            .await
            .expect("profiles")
            .into_inner()
            .profiles
            .is_empty()
    );
    endpoint.stop().await;
}

#[derive(Clone, Debug)]
enum Action {
    ChangeLink(bool),
    StaleLink,
    SetConsent(bool),
    Erase,
    Replay,
}

fn action() -> impl Strategy<Value = Action> {
    prop_oneof![
        3 => any::<bool>().prop_map(Action::ChangeLink),
        2 => Just(Action::StaleLink),
        2 => any::<bool>().prop_map(Action::SetConsent),
        1 => Just(Action::Erase),
        3 => Just(Action::Replay),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn link_unlink_consent_erase_interleavings_never_resurrect(actions in prop::collection::vec(action(), 2..25)) {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("runtime");
        runtime.block_on(async move {
            let dir = tempdir();
            let mut endpoint = EndpointFixture::start(dir.path(), true).await;
            let client = &mut endpoint.client;
            let original = client.record_voice_observation(observation("source", 0)).await.expect("seed voice");
            client.store_fact(fact("source", "", 0)).await.expect("seed fact");
            let mut profile = client.list_voice_profiles(list_profiles()).await.expect("seed profile").into_inner().profiles.remove(0);
            let mut linked = false;
            let mut opted_in = false;
            let mut erased = false;
            for (index, action) in actions.into_iter().enumerate() {
                match action {
                    Action::ChangeLink(new_linked) => {
                        let user = if new_linked { "alice" } else { "" };
                        let result = client.link_voice_profile_to_person(link(&profile, user)).await;
                        if erased {
                            prop_assert_eq!(result.expect_err("erased profile").code(), Code::NotFound);
                            continue;
                        }
                        let next = result.expect("CAS write").into_inner();
                        let expected = profile.link_revision + u64::from(linked != new_linked);
                        prop_assert_eq!(next.link_revision, expected);
                        if linked != new_linked { opted_in = false; }
                        linked = new_linked;
                        profile = next;
                    }
                    Action::StaleLink => {
                        let mut stale = link(&profile, "alice");
                        stale.expected_link_revision += 1;
                        let error = client.link_voice_profile_to_person(stale).await.expect_err("stale CAS");
                        prop_assert_eq!(error.code(), if erased { Code::NotFound } else { Code::Aborted });
                    }
                    Action::SetConsent(granted) => {
                        let result = client.set_voice_consent(consent("alice", granted)).await;
                        if granted && (!linked || erased) {
                            prop_assert_eq!(result.expect_err("unlinked cannot opt in").code(), Code::FailedPrecondition);
                        } else {
                            result.expect("write consent");
                            opted_in = granted;
                        }
                    }
                    Action::Erase => {
                        let receipt = client.erase_conversation(erase("source")).await.expect("erasure").into_inner();
                        prop_assert!(receipt.tombstone_recorded);
                        prop_assert_eq!((receipt.facts_deleted, receipt.vectors_deleted, receipt.captures_deleted), (1, 2, 1));
                        erased = true;
                        opted_in = false;
                        linked = false;
                    }
                    Action::Replay => {
                        let result = client.record_voice_observation(observation("source", 0)).await;
                        if erased {
                            prop_assert_eq!(result.expect_err("erased utterance cannot replay").code(), Code::FailedPrecondition);
                            let error = client.store_fact(fact("source", "", 0)).await.expect_err("tombstoned fact source");
                            prop_assert_eq!(error.code(), Code::FailedPrecondition);
                        } else {
                            prop_assert_eq!(&result.expect("idempotent replay").into_inner().capture_id, &original.get_ref().capture_id);
                        }
                    }
                }
                let probe = client.record_voice_observation(observation(&format!("probe-{index}"), 0))
                    .await.expect("probe").into_inner();
                prop_assert_eq!(probe.matched_profile_id.as_str(), if opted_in { profile.profile_id.as_str() } else { "" });
            }
            endpoint.stop().await;
            Ok(())
        })?;
    }
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn binary_notifies_systemd_after_schema_and_private_socket_ready() {
    use std::os::unix::net::UnixDatagram;
    use std::process::Command;

    let dir = tempdir();
    let runtime = dir.path().join("run");
    let data = dir.path().join("data");
    let config = include_str!("../../../config/nexa/nexa.toml")
        .replace("/var/lib/nexa", data.to_str().expect("test path"));
    fs::write(dir.path().join("nexa.toml"), config).expect("test config");
    let notify_path = dir.path().join("notify.sock");
    let notify = UnixDatagram::bind(&notify_path).expect("notify socket");
    notify
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("notification timeout");
    let mut daemon = Command::new(env!("CARGO_BIN_EXE_nexa-memoryd"))
        .arg("--config")
        .arg(dir.path())
        .env("RUNTIME_DIRECTORY", &runtime)
        .env("NOTIFY_SOCKET", &notify_path)
        .spawn()
        .expect("spawn binary");
    let mut bytes = [0; 128];
    let received = notify.recv(&mut bytes);
    let check = async move {
        let bytes = &bytes[..received.expect("systemd notification")];
        assert_eq!(bytes, b"READY=1\n");
        assert!(data.join("memory/memory.db").exists());
        assert!(data.join("memory/ambient.db").exists());
        let path = runtime.join("memory.sock");
        assert_eq!(
            path.metadata().expect("socket").permissions().mode() & 0o777,
            0o600
        );
        let mut client = connect(&path).await;
        assert_eq!(
            client
                .get_health(actor_metadata(
                    Empty {},
                    &operator(&[Permission::DiagnosticsRead])
                ))
                .await
                .expect("health after readiness")
                .into_inner()
                .state,
            HealthState::Ok as i32
        );
    };
    // Reap the child even when notification fails or an assertion panics.
    let result = tokio::spawn(check).await;
    daemon.kill().expect("kill test daemon");
    daemon.wait().expect("reap test daemon");
    result.expect("readiness assertions");
}

#[tokio::test]
async fn concurrent_uds_links_allow_only_one_compare_and_swap() {
    let dir = tempdir();
    let mut endpoint = EndpointFixture::start(dir.path(), true).await;
    endpoint
        .client
        .record_voice_observation(observation("seed", 0))
        .await
        .expect("capture");
    let profile = endpoint
        .client
        .list_voice_profiles(list_profiles())
        .await
        .expect("profile")
        .into_inner()
        .profiles
        .remove(0);
    let mut competitor = endpoint.client.clone();
    let (one, other) = tokio::join!(
        endpoint
            .client
            .link_voice_profile_to_person(link(&profile, "one")),
        competitor.link_voice_profile_to_person(link(&profile, "other")),
    );
    let (winner, rejected) = if one.is_ok() {
        (one, other)
    } else {
        (other, one)
    };
    let winner = winner.expect("one link wins").into_inner();
    assert_eq!(winner.link_revision, 1);
    assert_eq!(
        rejected.expect_err("stale competitor").code(),
        Code::Aborted
    );
    let profiles = endpoint
        .client
        .list_voice_profiles(list_profiles())
        .await
        .expect("current state");
    assert_eq!(profiles.into_inner().profiles, [winner]);
    drop(competitor);
    endpoint.stop().await;
}
