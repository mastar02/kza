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

fn overview(view: &str, cursor: &str, limit: u32, actor: ActorContext) -> MemoryOverviewRequest {
    MemoryOverviewRequest {
        view: view.into(),
        cursor: cursor.into(),
        limit,
        actor: Some(actor),
    }
}

fn mutation(
    operation: &str,
    target: &str,
    payload: serde_json::Value,
    actor: ActorContext,
) -> MemoryMutationRequest {
    MemoryMutationRequest {
        operation: operation.into(),
        target_id: target.into(),
        payload_json: payload.to_string(),
        actor: Some(actor),
    }
}

fn json_payload(value: &str) -> serde_json::Value {
    serde_json::from_str(value).expect("valid JSON payload")
}

fn checkpoint(path: &Path) {
    let conn = rusqlite::Connection::open(path).expect("open sqlite for checkpoint");
    let busy: i64 = conn
        .query_row("PRAGMA wal_checkpoint(FULL)", [], |row| row.get(0))
        .expect("checkpoint seeded data");
    assert_eq!(busy, 0);
}

fn needle_in_db_or_wal(path: &Path, needle: &[u8]) -> bool {
    [path.to_owned(), path.with_extension("db-wal")]
        .into_iter()
        .map(|file| match fs::read(file) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(error) => panic!("read database files: {error}"),
        })
        .any(|bytes| bytes.windows(needle.len()).any(|window| window == needle))
}

#[tokio::test]
async fn overview_and_fact_mutations_enforce_scopes_and_scrub_proposals() {
    use Permission::*;
    use serde_json::json;

    const MARKER: &str = "synthetic-rejected-proposal-forensic-marker";
    let dir = tempdir();
    let mut endpoint = EndpointFixture::start(dir.path(), false).await;
    let client = &mut endpoint.client;
    let reader = operator(&[MemoryRead]);
    let reviewer = operator(&[MemoryReview]);
    let assigner = operator(&[MemoryRead, MemoryAssign]);
    let alice = home("alice", &[MemoryOwnRead, MemoryOwnWrite]);
    let bob = home("bob", &[MemoryOwnRead, MemoryOwnWrite]);
    for view in ["mine", "voice_mine"] {
        assert_eq!(
            client
                .list_memory_overview(overview(
                    view,
                    "",
                    1,
                    operator(&[MemoryOwnRead, MemoryOwnVoice]),
                ))
                .await
                .expect_err("operator cannot use personal view")
                .code(),
            Code::PermissionDenied
        );
        assert_eq!(
            client
                .list_memory_overview(overview(
                    view,
                    "",
                    1,
                    home("", &[MemoryOwnRead, MemoryOwnVoice])
                ))
                .await
                .expect_err("HOME requires bound user ID")
                .code(),
            Code::PermissionDenied
        );
    }
    assert_eq!(
        client
            .list_memory_overview(overview("returned", "", 1, reader.clone()))
            .await
            .expect_err("review queue requires assign")
            .code(),
        Code::PermissionDenied
    );

    let mut ids = Vec::new();
    for (source, owner) in [
        ("keep", ""),
        ("personal-a", "alice"),
        ("personal-b", "alice"),
        ("unrelated", "bob"),
        ("reject", ""),
        ("confirm", ""),
    ] {
        let mut request = fact(source, owner, 0);
        if source == "reject" {
            request.content = MARKER.into();
        }
        ids.push(
            client
                .store_fact(request)
                .await
                .expect("store fact")
                .into_inner()
                .fact_id,
        );
    }
    let db = dir.path().join("data/memory.db");
    {
        // StoreFact only admits confirmed facts. Seed two pending rows using the
        // persisted proposal representation while the daemon is idle.
        let conn = rusqlite::Connection::open(&db).expect("open proposal fixtures");
        conn.execute(
            "UPDATE facts SET confirmed=0 WHERE fact_id IN (?1, ?2)",
            [&ids[4], &ids[5]],
        )
        .expect("seed pending proposals");
    }
    checkpoint(&db);
    assert!(needle_in_db_or_wal(&db, MARKER.as_bytes()));
    let summary = client
        .list_memory_overview(overview("summary", "", 1, reader.clone()))
        .await
        .expect("read summary")
        .into_inner();
    let summary = json_payload(&summary.payload_json);
    assert_eq!(summary["facts"].as_array().expect("facts").len(), 1);
    assert_eq!(summary["proposals"].as_array().expect("proposals").len(), 1);

    for (view, key, actor) in [
        ("facts", "facts", reader.clone()),
        ("proposals", "proposals", reader.clone()),
        ("mine", "facts", alice.clone()),
    ] {
        let request = overview(view, "", 1, actor);
        let first = client
            .list_memory_overview(request.clone())
            .await
            .expect("first page")
            .into_inner();
        let first = json_payload(&first.payload_json);
        let second = client
            .list_memory_overview(MemoryOverviewRequest {
                cursor: first["next"].as_str().expect("cursor").into(),
                ..request
            })
            .await
            .expect("second page")
            .into_inner();
        let second = json_payload(&second.payload_json);
        assert_eq!(first[key].as_array().expect("first items").len(), 1);
        assert_ne!(first[key][0]["fact_id"], second[key][0]["fact_id"]);
        if view == "mine" {
            for row in [&first[key][0], &second[key][0]] {
                assert_eq!(row["owner_user_id"], "alice");
                assert!(row.get("conversation_id").is_none());
            }
        }
        if view != "facts" {
            assert!(second["next"].is_null());
        }
    }

    let reject = mutation("reject_proposal", &ids[4], json!({}), reviewer.clone());
    assert_eq!(
        client
            .mutate_memory_item(MemoryMutationRequest {
                actor: Some(reader.clone()),
                ..reject.clone()
            })
            .await
            .expect_err("requires review")
            .code(),
        Code::PermissionDenied
    );
    client
        .mutate_memory_item(reject.clone())
        .await
        .expect("erase proposal");
    assert!(!needle_in_db_or_wal(&db, MARKER.as_bytes()));
    client
        .mutate_memory_item(reject)
        .await
        .expect("retry rejection");
    assert_eq!(
        client
            .mutate_memory_item(mutation(
                "reject_proposal",
                &ids[0],
                json!({}),
                reviewer.clone()
            ))
            .await
            .expect_err("cannot reject confirmed fact")
            .code(),
        Code::FailedPrecondition
    );
    let result = client
        .mutate_memory_item(mutation("confirm_proposal", &ids[5], json!({}), reviewer))
        .await
        .expect("confirm pending")
        .into_inner();
    assert_eq!(
        json_payload(&result.payload_json)["fact"]["confirmed"],
        true
    );

    for actor in [operator(&[MemoryOwnWrite, MemoryReview]), bob.clone()] {
        let code = if actor.role == Role::Home as i32 {
            Code::NotFound
        } else {
            Code::PermissionDenied
        };
        assert_eq!(
            client
                .mutate_memory_item(mutation("return_fact", &ids[1], json!({}), actor))
                .await
                .expect_err("cannot return someone else's fact")
                .code(),
            code
        );
    }
    client
        .mutate_memory_item(mutation("return_fact", &ids[1], json!({}), alice))
        .await
        .expect("return personal fact");
    let returned = client
        .list_memory_overview(overview("returned", "", 5, assigner.clone()))
        .await
        .expect("read returned queue")
        .into_inner();
    assert_eq!(
        json_payload(&returned.payload_json)["facts"][0]["fact_id"],
        ids[1]
    );
    let query_result = client
        .query_memory(query("", true, 10))
        .await
        .expect("shared search")
        .into_inner();
    assert!(
        query_result
            .hits
            .iter()
            .all(|hit| hit.fact.as_ref().expect("fact").fact_id != ids[1])
    );

    let assign = mutation(
        "assign_owner",
        &ids[1],
        json!({"person_id": "bob", "expected_owner_revision": 2}),
        assigner.clone(),
    );
    assert_eq!(
        client
            .mutate_memory_item(MemoryMutationRequest {
                actor: Some(reader.clone()),
                ..assign.clone()
            })
            .await
            .expect_err("requires assign")
            .code(),
        Code::PermissionDenied
    );
    client
        .mutate_memory_item(assign)
        .await
        .expect("reassign returned fact");
    let query_result = client
        .query_memory(query("bob", false, 10))
        .await
        .expect("owner search")
        .into_inner();
    assert!(
        query_result
            .hits
            .iter()
            .any(|hit| hit.fact.as_ref().expect("fact").fact_id == ids[1])
    );
    let counts = client
        .list_memory_overview(overview("owners_counts", "", 1, assigner.clone()))
        .await
        .expect("owner counts")
        .into_inner();
    assert_eq!(
        json_payload(&counts.payload_json)["counts"],
        json!({"alice": 1, "bob": 2})
    );

    client
        .erase_conversation(erase("personal-b"))
        .await
        .expect("erase source");
    client
        .purge_retired_user(purge("bob"))
        .await
        .expect("retire Bob");
    assert_eq!(
        client
            .mutate_memory_item(mutation(
                "assign_owner",
                &ids[0],
                json!({"owner_user_id": "bob"}),
                assigner.clone()
            ))
            .await
            .expect_err("retired target")
            .code(),
        Code::FailedPrecondition
    );
    let all = client
        .list_memory_overview(overview("facts", "", 100, reader))
        .await
        .expect("live facts")
        .into_inner();
    let all = json_payload(&all.payload_json);
    assert_eq!(all["facts"].as_array().expect("remaining facts").len(), 2);
    let counts = client
        .list_memory_overview(overview("owners_counts", "", 1, assigner))
        .await
        .expect("post-purge counts")
        .into_inner();
    assert_eq!(
        json_payload(&counts.payload_json),
        json!({"counts": {}, "retired_owner_prune_pending": false})
    );
    endpoint.stop().await;
}

#[tokio::test]
async fn voice_switch_and_personal_consent_use_cas_and_persist_revocation() {
    use Permission::*;
    use serde_json::json;

    let dir = tempdir();
    let mut endpoint = EndpointFixture::start(dir.path(), true).await;
    let client = &mut endpoint.client;
    let switch_reader = ActorRequest {
        actor: Some(operator(&[MemoryRead, MemoryAssign])),
    };
    assert_eq!(
        client
            .get_voice_global_switch(ActorRequest {
                actor: Some(operator(&[MemoryRead])),
            })
            .await
            .expect_err("switch read also requires assign")
            .code(),
        Code::PermissionDenied
    );
    let enabled = client
        .get_voice_global_switch(switch_reader.clone())
        .await
        .expect("switch state")
        .into_inner();
    assert!(enabled.enabled);
    let disable = SetVoiceGlobalSwitchRequest {
        enabled: false,
        expected_revision: enabled.revision,
        actor: Some(operator(&[MemoryAssign, AdminManage])),
    };
    assert_eq!(
        client
            .set_voice_global_switch(SetVoiceGlobalSwitchRequest {
                actor: Some(operator(&[MemoryAssign])),
                ..disable.clone()
            })
            .await
            .expect_err("switch change also requires admin")
            .code(),
        Code::PermissionDenied
    );

    client
        .record_voice_observation(observation("voice-consent", 0))
        .await
        .expect("create profile");
    let profile = client
        .list_voice_profiles(list_profiles())
        .await
        .expect("read profile")
        .into_inner()
        .profiles
        .remove(0);
    client
        .link_voice_profile_to_person(link(&profile, "alice"))
        .await
        .expect("link profile");
    let alice = home("alice", &[MemoryOwnVoice]);
    let read_mine = overview("voice_mine", "", 10, alice.clone());
    let mine = client
        .list_memory_overview(read_mine.clone())
        .await
        .expect("personal voice status")
        .into_inner();
    let mine = json_payload(&mine.payload_json);
    assert_eq!(mine["profile_id"], profile.profile_id);
    assert_eq!(mine["granted"], false);

    let grant = mutation(
        "voice_consent_grant",
        "alice",
        json!({"expected_revision": mine["revision"]}),
        alice.clone(),
    );
    assert_eq!(
        client
            .mutate_memory_item(MemoryMutationRequest {
                target_id: "bob".into(),
                ..grant.clone()
            })
            .await
            .expect_err("cannot grant for Bob")
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        client
            .mutate_memory_item(MemoryMutationRequest {
                payload_json: json!({"user_id": "bob"}).to_string(),
                ..grant.clone()
            })
            .await
            .expect_err("cannot spoof payload identity")
            .code(),
        Code::PermissionDenied
    );
    let receipt = client
        .mutate_memory_item(grant.clone())
        .await
        .expect("explicit opt-in")
        .into_inner();
    assert_eq!(json_payload(&receipt.payload_json)["granted"], true);
    assert_eq!(
        client
            .mutate_memory_item(grant)
            .await
            .expect_err("stale consent CAS")
            .code(),
        Code::Aborted
    );

    let mut contender = connect(&endpoint.socket_path).await;
    let (first, second) = tokio::join!(
        client.set_voice_global_switch(disable.clone()),
        contender.set_voice_global_switch(disable),
    );
    let off = match (first, second) {
        (Ok(current), Err(stale)) | (Err(stale), Ok(current)) => {
            assert_eq!(stale.code(), Code::Aborted);
            current.into_inner()
        }
        _ => panic!("exactly one CAS should commit"),
    };
    assert!(!off.enabled);
    assert_eq!(off.revision, enabled.revision + 1);
    let list = overview(
        "voice_consents",
        "",
        10,
        operator(&[MemoryRead, MemoryAssign]),
    );
    let consents = client
        .list_memory_overview(list)
        .await
        .expect("operator consent view")
        .into_inner();
    let consents = json_payload(&consents.payload_json);
    assert_eq!(consents["consents"][0]["granted"], false);
    assert_eq!(consents["consents"][0]["user_id"], "alice");
    let stored: (bool, Option<String>, Option<u64>, Option<u64>) =
        rusqlite::Connection::open(dir.path().join("data/memory.db"))
            .expect("open revocation record")
            .query_row(
                "SELECT granted, profile_id, profile_revision, link_revision
                        FROM voice_attribution_consents WHERE person_id='alice'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .expect("read durable consent row");
    assert_eq!(stored, (false, None, None, None));
    drop(contender);
    endpoint.stop().await;

    let mut endpoint = EndpointFixture::start(dir.path(), true).await;
    let client = &mut endpoint.client;
    assert_eq!(
        client
            .get_voice_global_switch(switch_reader)
            .await
            .expect("read after restart")
            .into_inner(),
        off
    );
    client
        .set_voice_global_switch(SetVoiceGlobalSwitchRequest {
            enabled: true,
            expected_revision: off.revision,
            actor: Some(operator(&[MemoryAssign, AdminManage])),
        })
        .await
        .expect("re-enable switch");
    let mine = client
        .list_memory_overview(read_mine.clone())
        .await
        .expect("read consent after re-enable")
        .into_inner();
    assert_eq!(json_payload(&mine.payload_json)["granted"], false);
    for _ in 0..3 {
        client
            .mutate_memory_item(mutation(
                "voice_verification",
                "",
                json!({"outcome": "failed"}),
                alice.clone(),
            ))
            .await
            .expect("record verification attempt");
    }
    let locked = client
        .list_memory_overview(read_mine)
        .await
        .expect("read verification lock")
        .into_inner();
    let locked = json_payload(&locked.payload_json);
    assert_eq!(locked["verification_locked"], true);
    assert_eq!(
        client
            .mutate_memory_item(mutation(
                "voice_consent_grant",
                "",
                json!({}),
                alice.clone()
            ))
            .await
            .expect_err("locked account cannot grant")
            .code(),
        Code::FailedPrecondition
    );
    let reopened = client
        .mutate_memory_item(mutation(
            "reopen_voice_consent",
            "alice",
            json!({"expected_revision": locked["revision"]}),
            operator(&[MemoryAssign]),
        ))
        .await
        .expect("clear person's verification lock")
        .into_inner();
    let reopened = json_payload(&reopened.payload_json);
    assert_eq!(reopened["verification_locked"], false);
    assert_eq!(reopened["granted"], false);
    client
        .mutate_memory_item(mutation(
            "voice_consent_grant",
            "",
            json!({"expected_revision": reopened["revision"]}),
            alice.clone(),
        ))
        .await
        .expect("fresh opt-in after reopen");
    client
        .mutate_memory_item(mutation("voice_consent_revoke", "", json!({}), alice))
        .await
        .expect("personal revoke");
    endpoint.stop().await;
}

#[tokio::test]
async fn capture_and_evaluation_projections_filter_and_fence_erasure() {
    use Permission::*;
    use serde_json::json;

    let dir = tempdir();
    let mut endpoint = EndpointFixture::start(dir.path(), false).await;
    let client = &mut endpoint.client;
    let review_actor = operator(&[VoicesRead, VoicesReview]);
    let evaluator = ActorContext {
        role: Role::Evaluator as i32,
        subject: "evaluator".into(),
        permissions: vec![EvaluationRead as i32],
        ..Default::default()
    };
    let mut ids = Vec::new();
    for (source, room) in [
        ("erased", "hall"),
        ("second", "hall"),
        ("retired", "kitchen"),
    ] {
        let capture = observation(source, 0).into_inner();
        let request = actor_metadata(
            RecordVoiceObservationRequest {
                room_id: room.into(),
                ..capture
            },
            &operator(&[VoicesReview]),
        );
        ids.push(
            client
                .record_voice_observation(request)
                .await
                .expect("capture")
                .into_inner()
                .capture_id,
        );
    }

    let review = VoiceCaptureReviewRequest {
        fingerprint_id: ids[0].clone(),
        include_transcript: false,
        actor: Some(review_actor.clone()),
    };
    assert_eq!(
        client
            .get_voice_capture_review(VoiceCaptureReviewRequest {
                actor: Some(operator(&[VoicesRead])),
                ..review.clone()
            })
            .await
            .expect_err("requires both voice permissions")
            .code(),
        Code::PermissionDenied
    );
    let private = client
        .get_voice_capture_review(review.clone())
        .await
        .expect("read capture")
        .into_inner();
    let private = json_payload(&private.payload_json);
    assert!(private.get("transcript_preview").is_none());
    assert!(private.get("embedding").is_none());
    let with_text = client
        .get_voice_capture_review(VoiceCaptureReviewRequest {
            include_transcript: true,
            ..review.clone()
        })
        .await
        .expect("explicit preview")
        .into_inner();
    assert_eq!(
        json_payload(&with_text.payload_json)["transcript_preview"],
        "synthetic speech"
    );

    let query = VoiceCaptureQueryRequest {
        query_json: json!({"room": "hall", "limit": 1, "review_status": "pending"}).to_string(),
        actor: Some(review_actor.clone()),
    };
    assert_eq!(
        client
            .query_voice_captures(VoiceCaptureQueryRequest {
                actor: Some(home("alice", &[VoicesRead, VoicesReview])),
                ..query.clone()
            })
            .await
            .expect_err("operator role required")
            .code(),
        Code::PermissionDenied
    );
    let first = client
        .query_voice_captures(query.clone())
        .await
        .expect("first hall page")
        .into_inner();
    let first = json_payload(&first.payload_json);
    let second = client
        .query_voice_captures(VoiceCaptureQueryRequest {
            query_json:
                json!({"room":"hall", "limit":1, "review_status":"pending", "cursor":first["next"]})
                    .to_string(),
            ..query.clone()
        })
        .await
        .expect("second hall page")
        .into_inner();
    let second = json_payload(&second.payload_json);
    assert_ne!(
        first["captures"][0]["fingerprint_id"],
        second["captures"][0]["fingerprint_id"]
    );
    assert!(second["next"].is_null());
    for page in [&first, &second] {
        assert_eq!(page["captures"][0]["room_id"], "hall");
        assert!(page["captures"][0].get("transcript_preview").is_none());
    }

    let evaluation = EvaluationSampleRequest {
        fingerprint_id: ids[0].clone(),
        actor: Some(evaluator.clone()),
    };
    for actor in [
        operator(&[EvaluationRead]),
        home("alice", &[EvaluationRead]),
        ActorContext {
            permissions: vec![],
            ..evaluator.clone()
        },
    ] {
        assert_eq!(
            client
                .get_evaluation_sample(EvaluationSampleRequest {
                    actor: Some(actor),
                    ..evaluation.clone()
                })
                .await
                .expect_err("requires evaluator role and permission")
                .code(),
            Code::PermissionDenied
        );
    }
    let sample = client
        .get_evaluation_sample(evaluation.clone())
        .await
        .expect("evaluation sample")
        .into_inner();
    let sample = json_payload(&sample.payload_json);
    assert_eq!(sample["fingerprint_id"], ids[0]);
    for key in [
        "transcript_preview",
        "embedding",
        "audio_path",
        "room_id",
        "profile_id",
        "utterance_id",
    ] {
        assert!(
            sample.get(key).is_none(),
            "evaluation projection contains {key}"
        );
    }
    {
        let db = dir.path().join("data/memory.db");
        let conn = rusqlite::Connection::open(db).expect("open tombstone fixture");
        conn.execute(
            "INSERT INTO conversation_tombstones VALUES ('erased', 1)",
            [],
        )
        .expect("seed a fence while residual capture rows still exist");
    }
    assert_eq!(
        client
            .get_evaluation_sample(evaluation.clone())
            .await
            .expect_err("residual capture cannot escape the fence")
            .code(),
        Code::NotFound
    );
    assert_eq!(
        client
            .get_voice_capture_review(review.clone())
            .await
            .expect_err("residual review cannot escape the fence")
            .code(),
        Code::NotFound
    );
    let retired_capture = client
        .get_voice_capture_review(VoiceCaptureReviewRequest {
            fingerprint_id: ids[2].clone(),
            ..review.clone()
        })
        .await
        .expect("find profile to retire")
        .into_inner();
    let retired_capture = json_payload(&retired_capture.payload_json);
    let profile_id = retired_capture["profile_id"].as_str().expect("profile id");
    let profile = client
        .list_voice_profiles(list_profiles())
        .await
        .expect("profiles")
        .into_inner()
        .profiles
        .into_iter()
        .find(|profile| profile.profile_id == profile_id)
        .expect("capture profile");
    client
        .link_voice_profile_to_person(link(&profile, "retired-user"))
        .await
        .expect("link retired-user profile");
    client
        .erase_conversation(erase("erased"))
        .await
        .expect("erase source");
    client
        .purge_retired_user(purge("retired-user"))
        .await
        .expect("retire capture owner");

    for fingerprint_id in [&ids[0], &ids[2]] {
        assert_eq!(
            client
                .get_evaluation_sample(EvaluationSampleRequest {
                    fingerprint_id: fingerprint_id.clone(),
                    ..evaluation.clone()
                })
                .await
                .expect_err("erased sample unavailable")
                .code(),
            Code::NotFound
        );
        assert_eq!(
            client
                .get_voice_capture_review(VoiceCaptureReviewRequest {
                    fingerprint_id: fingerprint_id.clone(),
                    ..review.clone()
                })
                .await
                .expect_err("erased review unavailable")
                .code(),
            Code::NotFound
        );
    }
    let remaining = client
        .query_voice_captures(VoiceCaptureQueryRequest {
            query_json: "{}".into(),
            ..query
        })
        .await
        .expect("only live captures")
        .into_inner();
    let remaining = json_payload(&remaining.payload_json);
    assert_eq!(
        remaining["captures"]
            .as_array()
            .expect("remaining captures")
            .len(),
        1
    );
    assert_eq!(remaining["captures"][0]["fingerprint_id"], ids[1]);
    assert_eq!(
        client
            .record_voice_observation(observation("erased", 0))
            .await
            .expect_err("source replay fenced")
            .code(),
        Code::FailedPrecondition
    );
    assert_eq!(
        client
            .record_voice_observation(observation("retired", 0))
            .await
            .expect_err("retired sample replay fenced")
            .code(),
        Code::FailedPrecondition
    );
    endpoint.stop().await;
}

#[tokio::test]
async fn reviewed_voice_profile_erasure_scrubs_sources_vectors_links_and_consent() {
    use Permission::*;
    use serde_json::json;

    const PREVIEW: &str = "synthetic-voice-delete-forensic-marker";
    let dir = tempdir();
    let mut endpoint = EndpointFixture::start(dir.path(), true).await;
    let client = &mut endpoint.client;
    let reviewer = operator(&[VoicesReview]);
    let mut seed = observation("delete-profile-source", 0).into_inner();
    seed.transcript_preview = PREVIEW.into();
    seed.embedding[0] = 0.3;
    seed.embedding[1] = 0.4;
    let capture = client
        .record_voice_observation(actor_metadata(seed, &reviewer))
        .await
        .expect("sample to review")
        .into_inner();
    client
        .record_voice_observation(observation("keep-profile-source", 2))
        .await
        .expect("unrelated sample");
    let confirm = mutation(
        "confirm_voice_profile",
        "",
        json!({"fingerprint_ids": [capture.capture_id]}),
        reviewer.clone(),
    );
    let confirmed = client
        .mutate_memory_item(confirm.clone())
        .await
        .expect("confirm sample")
        .into_inner();
    let confirmed = json_payload(&confirmed.payload_json);
    let retry = client
        .mutate_memory_item(confirm)
        .await
        .expect("confirm retry")
        .into_inner();
    assert_eq!(json_payload(&retry.payload_json), confirmed);
    let profile_id = confirmed["profile_id"]
        .as_str()
        .expect("confirmed profile id");
    let accepted = client
        .query_voice_captures(VoiceCaptureQueryRequest {
            query_json: json!({"review_status": "accepted"}).to_string(),
            actor: Some(operator(&[VoicesRead, VoicesReview])),
        })
        .await
        .expect("accepted samples")
        .into_inner();
    assert_eq!(
        json_payload(&accepted.payload_json)["captures"][0]["fingerprint_id"],
        capture.capture_id
    );

    let grouped = client
        .mutate_memory_item(mutation(
            "correct_voice_group",
            "manual-group",
            json!({"profile_ids": [profile_id], "expected_revision": 0}),
            reviewer.clone(),
        ))
        .await
        .expect("persist group correction")
        .into_inner();
    assert_eq!(json_payload(&grouped.payload_json)["updated"], 1);
    let profile = client
        .list_voice_profiles(list_profiles())
        .await
        .expect("profiles")
        .into_inner()
        .profiles
        .into_iter()
        .find(|profile| profile.profile_id == profile_id)
        .expect("confirmed profile");
    let linked = client
        .link_voice_profile_to_person(link(&profile, "alice"))
        .await
        .expect("link confirmed profile")
        .into_inner();
    client
        .set_voice_consent(consent("alice", true))
        .await
        .expect("consent");

    let memory = dir.path().join("data/memory.db");
    let ambient = dir.path().join("data/ambient.db");
    let embedding: Vec<u8> =
        {
            let conn = rusqlite::Connection::open(&memory).expect("open SQLite");
            let revision: u64 = conn.query_row(
            "SELECT g.revision FROM voice_groups g JOIN voice_group_members m USING (group_id)
             WHERE g.group_id='manual-group' AND m.profile_id=?1",
            [profile_id],
            |row| row.get(0),
        ).expect("persisted group and membership");
            assert_eq!(revision, 1);
            let marker = conn
                .query_row(
                    "SELECT embedding FROM voice_fingerprints WHERE capture_id=?1",
                    [&capture.capture_id],
                    |row| row.get(0),
                )
                .expect("load exact embedding bytes");
            conn.execute(
                "INSERT INTO voice_attribution_suspensions VALUES (?1, 1, 1)",
                [profile_id],
            )
            .expect("seed attribution suspension while daemon is idle");
            marker
        };
    let resume = mutation(
        "resume_attribution",
        profile_id,
        json!({"expected_revision": linked.profile_revision}),
        operator(&[VoicesReview, AdminManage, MemoryAssign]),
    );
    assert_eq!(
        client
            .mutate_memory_item(MemoryMutationRequest {
                actor: Some(operator(&[VoicesReview, AdminManage])),
                ..resume.clone()
            })
            .await
            .expect_err("resume requires all three permissions")
            .code(),
        Code::PermissionDenied
    );
    let resumed = client
        .mutate_memory_item(resume)
        .await
        .expect("explicit resume")
        .into_inner();
    let resumed = json_payload(&resumed.payload_json);
    let mine = overview("voice_mine", "", 1, home("alice", &[MemoryOwnVoice]));
    let unconsented = client
        .list_memory_overview(mine.clone())
        .await
        .expect("after resume")
        .into_inner();
    assert_eq!(json_payload(&unconsented.payload_json)["granted"], false);
    client
        .set_voice_consent(consent("alice", true))
        .await
        .expect("fresh consent");

    checkpoint(&memory);
    checkpoint(&ambient);
    assert!(needle_in_db_or_wal(&memory, &embedding));
    assert!(needle_in_db_or_wal(&ambient, PREVIEW.as_bytes()));
    let deletion = mutation(
        "delete_voice_profile",
        profile_id,
        json!({"expected_revision": resumed["revision"]}),
        reviewer,
    );
    assert_eq!(
        client
            .mutate_memory_item(MemoryMutationRequest {
                payload_json: json!({"expected_revision": linked.profile_revision}).to_string(),
                ..deletion.clone()
            })
            .await
            .expect_err("stale profile delete")
            .code(),
        Code::Aborted
    );
    let erased = client
        .mutate_memory_item(deletion)
        .await
        .expect("erase reviewed profile")
        .into_inner();
    assert_eq!(json_payload(&erased.payload_json)["captures_deleted"], 1);
    assert!(!needle_in_db_or_wal(&memory, &embedding));
    assert!(!needle_in_db_or_wal(&ambient, PREVIEW.as_bytes()));
    assert!(!needle_in_db_or_wal(&memory, profile_id.as_bytes()));
    let mine = client
        .list_memory_overview(mine)
        .await
        .expect("status after deletion")
        .into_inner();
    let mine = json_payload(&mine.payload_json);
    assert_eq!(mine["linked"], false);
    assert_eq!(mine["granted"], false);
    assert_eq!(
        client
            .record_voice_observation(observation("delete-profile-source", 0))
            .await
            .expect_err("erased source cannot recreate profile")
            .code(),
        Code::FailedPrecondition
    );
    let remaining = client
        .list_voice_profiles(list_profiles())
        .await
        .expect("remaining profiles")
        .into_inner();
    assert_eq!(remaining.profiles.len(), 1, "unrelated profile preserved");
    endpoint.stop().await;
}

#[tokio::test]
async fn voice_profile_cursor_is_deterministic_over_uds() {
    let dir = tempdir();
    let mut endpoint = EndpointFixture::start(dir.path(), false).await;
    let client = &mut endpoint.client;
    for i in 0..51 {
        client
            .record_voice_observation(observation(&format!("page-voice-{i}"), i))
            .await
            .expect("seed profile");
    }
    let first = client
        .list_voice_profiles(list_profiles())
        .await
        .expect("first page")
        .into_inner();
    assert_eq!(first.profiles.len(), 50);
    assert_eq!(
        first.next_cursor,
        first.profiles.last().expect("last profile").profile_id
    );
    let second = client
        .list_voice_profiles(ListVoiceProfilesRequest {
            cursor: first.next_cursor.clone(),
            ..list_profiles()
        })
        .await
        .expect("second page")
        .into_inner();
    assert_eq!(second.profiles.len(), 1);
    assert!(second.next_cursor.is_empty());
    assert!(second.profiles[0].profile_id > first.next_cursor);
    let repeated = client
        .list_voice_profiles(list_profiles())
        .await
        .expect("repeat page")
        .into_inner();
    assert_eq!(first, repeated);
    endpoint.stop().await;
}
