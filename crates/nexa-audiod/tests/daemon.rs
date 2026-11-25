#![allow(clippy::unwrap_used)]

use std::fs;
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::path::Path;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use hyper_util::rt::TokioIo;
use nexa_audiod::capture_seal::CapturePhase;
use nexa_audiod::socket::AudioSocket;
use nexa_proto::v1::audio_service_client::AudioServiceClient;
use nexa_proto::v1::{Empty, HealthState, SubscribeUtterancesRequest};
use rustix::process::{Pid, Signal, kill_process};
use tokio::net::{UnixDatagram, UnixListener, UnixStream};
use tokio::time::{sleep, timeout};
use tonic::transport::Endpoint;
use tower::service_fn;

const CORE: &str = include_str!("../../../config/nexa/nexa.toml");
const ROOMS: &str = include_str!("../../../config/nexa/rooms.example.toml");

struct TestProcess(Child);

impl Drop for TestProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

async fn receive(notify: &UnixDatagram) -> String {
    let mut bytes = [0; 256];
    let count = timeout(Duration::from_secs(4), notify.recv(&mut bytes))
        .await
        .unwrap()
        .unwrap();
    std::str::from_utf8(&bytes[..count]).unwrap().to_owned()
}

fn write_config(root: &Path) {
    let core = CORE
        .replace("/var/lib/nexa", root.join("data").to_str().unwrap())
        .replace(
            "/run/user/nexa/nexa",
            root.join("runtime").to_str().unwrap(),
        );
    fs::write(root.join("nexa.toml"), core).unwrap();
    fs::write(root.join("rooms.toml"), ROOMS).unwrap();
}

async fn stop_process(child: &mut TestProcess, notify: &UnixDatagram, signal: Signal) {
    let start = Instant::now();
    let pid = Pid::from_raw(child.0.id().try_into().unwrap()).unwrap();
    kill_process(pid, signal).unwrap();
    let mut stopped = false;
    let result = timeout(Duration::from_secs(3), async {
        loop {
            match receive(notify).await.as_str() {
                "STOPPING=1" => break,
                "WATCHDOG=1" => (),
                other => panic!("unexpected systemd notification {other:?}"),
            }
        }
        stopped = true;
        loop {
            if let Some(status) = child.0.try_wait().unwrap() {
                assert!(status.success(), "daemon exited with {status}");
                break;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(
        result.is_ok(),
        "daemon took {:?} to stop (STOPPING received: {stopped})",
        start.elapsed()
    );
    assert!(start.elapsed() < Duration::from_secs(3));
}

#[tokio::test]
async fn simulation_daemon_notifies_serves_private_uds_and_stops_on_sigterm() {
    let root = tempfile::Builder::new()
        .prefix("nad-")
        .tempdir_in("/tmp")
        .unwrap();
    write_config(root.path());
    let notify_path = root.path().join("notify");
    let notify = UnixDatagram::bind(&notify_path).unwrap();
    let socket = root.path().join("custom/audio.sock");
    let mut child = TestProcess(
        Command::new(env!("CARGO_BIN_EXE_nexa-audiod"))
            .arg("--simulation")
            .arg("--socket")
            .arg(&socket)
            .env("NEXA_CONFIG_DIR", root.path())
            .env("NOTIFY_SOCKET", &notify_path)
            .env("WATCHDOG_USEC", "100000")
            .env_remove("WATCHDOG_PID")
            .spawn()
            .unwrap(),
    );

    assert_eq!(receive(&notify).await, "READY=1");
    let metadata = socket.symlink_metadata().unwrap();
    assert!(metadata.file_type().is_socket());
    assert_eq!(metadata.mode() & 0o777, 0o600);
    assert_eq!(
        socket.parent().unwrap().metadata().unwrap().mode() & 0o777,
        0o700
    );
    let address = socket.clone();
    let channel = Endpoint::from_static("http://[::]:50051")
        .timeout(Duration::from_secs(2))
        .connect_with_connector(service_fn(move |_| {
            let address = address.clone();
            async move { UnixStream::connect(address).await.map(TokioIo::new) }
        }))
        .await
        .unwrap();
    let mut client = AudioServiceClient::new(channel);
    let health = client.get_health(Empty {}).await.unwrap().into_inner();
    assert_eq!(health.service, "nexa-audiod");
    assert_eq!(health.state(), HealthState::Ok);
    let capture = client
        .get_capture_control(Empty {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(capture.phase(), CapturePhase::Running);
    assert!(capture.supported && !capture.ingress_sealed);
    assert_eq!(receive(&notify).await, "WATCHDOG=1");

    // An open streaming RPC must not block socket cleanup or process exit.
    let _stream = client
        .subscribe_utterances(SubscribeUtterancesRequest::default())
        .await
        .unwrap()
        .into_inner();
    stop_process(&mut child, &notify, Signal::TERM).await;
    assert!(!socket.exists());
}

#[tokio::test]
async fn simulation_accepts_config_flag_and_stops_on_sigint() {
    let root = tempfile::Builder::new()
        .prefix("nai-")
        .tempdir_in("/tmp")
        .unwrap();
    write_config(root.path());
    let notify_path = root.path().join("notify");
    let notify = UnixDatagram::bind(&notify_path).unwrap();
    let socket = root.path().join("xdg/nexa/audio.sock");
    let mut child = TestProcess(
        Command::new(env!("CARGO_BIN_EXE_nexa-audiod"))
            .arg("--simulation")
            .env_remove("RUNTIME_DIRECTORY")
            .env("XDG_RUNTIME_DIR", root.path().join("xdg"))
            .arg("--config")
            .arg(root.path())
            .env("NEXA_CONFIG_DIR", root.path().join("unused"))
            .env("NOTIFY_SOCKET", &notify_path)
            .env_remove("WATCHDOG_USEC")
            .spawn()
            .unwrap(),
    );
    assert_eq!(receive(&notify).await, "READY=1");
    assert!(socket.symlink_metadata().unwrap().file_type().is_socket());
    stop_process(&mut child, &notify, Signal::INT).await;
    assert!(!socket.exists());
}

#[tokio::test]
async fn socket_binding_replaces_only_a_stale_socket() {
    let root = tempfile::Builder::new()
        .prefix("nas-")
        .tempdir_in("/tmp")
        .unwrap();
    let path = root.path().join("runtime/audio.sock");
    let original = AudioSocket::bind(&path).await.unwrap();
    assert!(AudioSocket::bind(&path).await.is_err());
    drop(original);
    let foreign = UnixListener::bind(&path).unwrap();
    assert!(AudioSocket::bind(&path).await.is_err());
    drop(foreign);
    let recovered = AudioSocket::bind(&path).await.unwrap();
    assert_eq!(path.metadata().unwrap().mode() & 0o777, 0o600);
    drop(recovered);

    fs::write(&path, "unrelated").unwrap();
    assert!(AudioSocket::bind(&path).await.is_err());
    assert_eq!(fs::read_to_string(&path).unwrap(), "unrelated");
}
