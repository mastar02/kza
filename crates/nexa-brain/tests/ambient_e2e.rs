//! The ambient pipeline talks to a real nexa-memoryd over its private socket.
//! For a package-only run, build nexa-memoryd in the same target directory
//! first, or set NEXA_MEMORYD_BIN to the binary under test.

#[allow(dead_code)]
mod support;

use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Arc,
    time::Duration,
};

use nexa_brain::{
    Brain, Dependencies,
    clients::{AudioClient, MemoryClient, MlClient, StreamOf, UdsMemory, channel},
};
use nexa_household::SystemClock;
use nexa_proto::v1::{
    self as v1, Permission, Role, SttMode, UtteranceKind,
    memory_service_client::MemoryServiceClient,
};
use prost::Message;
use serde_json::Value;
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Status, metadata::MetadataValue};

use support::{Result, actor, bundle, receive};

const ENCODER: &str = "speechbrain/spkrec-ecapa-voxceleb";
type RpcResult<T> = std::result::Result<T, Status>;
type CaptureReceiver = mpsc::Receiver<RpcResult<v1::AudioUtterance>>;

#[derive(Debug)]
struct Audio(Mutex<Option<CaptureReceiver>>);

#[tonic::async_trait]
impl AudioClient for Audio {
    async fn subscribe(
        &self,
        include_ambient: bool,
    ) -> RpcResult<StreamOf<RpcResult<v1::AudioUtterance>>> {
        assert!(include_ambient);
        let rx = self
            .0
            .lock()
            .await
            .take()
            .ok_or_else(|| Status::unavailable("test stream closed"))?;
        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn play(&self, _: ReceiverStream<v1::PlayAudioChunk>) -> RpcResult<v1::PlayAudioReceipt> {
        panic!("ambient audio must never play a reply");
    }

    async fn cancel(&self, _: v1::CancelPlaybackRequest) -> RpcResult<()> {
        panic!("ambient audio must never cancel a reply");
    }
}

#[derive(Debug)]
enum Call {
    Transcribe(v1::TranscribeRequest),
    Embed(v1::EmbedSpeakerRequest),
    Observe(v1::RecordVoiceObservationRequest),
}

#[derive(Debug)]
struct Ml(mpsc::UnboundedSender<Call>);

#[tonic::async_trait]
impl MlClient for Ml {
    async fn transcribe(
        &self,
        request: v1::TranscribeRequest,
    ) -> RpcResult<v1::TranscribeResponse> {
        self.0
            .send(Call::Transcribe(request))
            .map_err(|_| Status::cancelled("test ended"))?;
        Ok(v1::TranscribeResponse {
            text: "synthetic ambient speech".into(),
            ..Default::default()
        })
    }

    async fn embed_speaker(
        &self,
        request: v1::EmbedSpeakerRequest,
    ) -> RpcResult<v1::SpeakerEmbeddingResponse> {
        self.0
            .send(Call::Embed(request))
            .map_err(|_| Status::cancelled("test ended"))?;
        let mut vector = vec![0.0; 192];
        vector[0] = 1.0;
        Ok(v1::SpeakerEmbeddingResponse {
            vector,
            encoder_model_key: ENCODER.into(),
            ..Default::default()
        })
    }

    async fn synthesize(
        &self,
        _: v1::SynthesizeRequest,
    ) -> RpcResult<StreamOf<RpcResult<v1::TtsAudioChunk>>> {
        panic!("ambient audio must never reach TTS");
    }

    async fn embed_text(&self, _: v1::EmbedTextRequest) -> RpcResult<v1::BatchEmbeddingResponse> {
        panic!("ambient audio must never reach fact embeddings");
    }
}

#[derive(Debug)]
struct ObservedMemory(UdsMemory, mpsc::UnboundedSender<Call>);

#[tonic::async_trait]
impl MemoryClient for ObservedMemory {
    async fn query(&self, request: v1::QueryMemoryRequest) -> RpcResult<v1::QueryMemoryResponse> {
        self.0.query(request).await
    }

    async fn observe_voice(
        &self,
        request: v1::RecordVoiceObservationRequest,
    ) -> RpcResult<v1::VoiceObservationReceipt> {
        self.1
            .send(Call::Observe(request.clone()))
            .map_err(|_| Status::cancelled("test ended"))?;
        self.0.observe_voice(request).await
    }
}

struct MemoryDaemon(Child);

impl MemoryDaemon {
    fn start(config: &Path, runtime: &Path) -> Result<Self> {
        let executable = match std::env::var_os("NEXA_MEMORYD_BIN") {
            Some(path) => PathBuf::from(path),
            None => Path::new(env!("CARGO_BIN_EXE_nexa-brain")).with_file_name("nexa-memoryd"),
        };
        if !executable.is_file() {
            return Err(format!(
                "{} is missing. Build nexa-memoryd or set NEXA_MEMORYD_BIN.",
                executable.display()
            )
            .into());
        }
        let child = Command::new(executable)
            .arg("--config")
            .arg(config)
            .env("RUNTIME_DIRECTORY", runtime)
            .env_remove("NOTIFY_SOCKET")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .spawn()?;
        Ok(Self(child))
    }
}

impl Drop for MemoryDaemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct LogWriter(mpsc::UnboundedSender<String>);

impl Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let line = String::from_utf8_lossy(bytes);
        if line.contains("Ambient observation") {
            let _ = self.0.send(line.into_owned());
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn control_request() -> Request<v1::Empty> {
    let operator = actor(Role::Operator, &[Permission::AmbientControl]);
    let mut request = Request::new(v1::Empty {});
    request.metadata_mut().insert_bin(
        "actor-context-bin",
        MetadataValue::from_bytes(&operator.encode_to_vec()),
    );
    request
}

async fn expect_status(
    logs: &mut mpsc::UnboundedReceiver<String>,
    id: &str,
    status: &str,
) -> Result {
    let line = receive(logs).await?;
    assert!(line.contains(&format!("utterance_id={id} ")), "{line}");
    assert!(line.contains(&format!("status=\"{status}\"")), "{line}");
    assert!(
        line.contains("room_id=living") && line.contains("latency_ms="),
        "{line}"
    );
    assert!(!line.contains("synthetic ambient speech"), "{line}");
    Ok(())
}

#[tokio::test]
async fn live_memoryd_pause_prevents_inference_and_persistence() -> Result {
    let (log, mut logs) = mpsc::unbounded_channel();
    tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_writer(move || LogWriter(log.clone()))
        .try_init()?;

    let dir = tempfile::tempdir_in("/tmp")?;
    let config = bundle(dir.path())?;
    let runtime = &config.core.paths.runtime_dir;
    let core_toml = include_str!("../../../config/nexa/nexa.toml").replace(
        "/var/lib/nexa",
        config.core.paths.data_dir.to_str().ok_or("data path")?,
    );
    tokio::fs::write(dir.path().join("nexa.toml"), core_toml).await?;
    let mut daemon = MemoryDaemon::start(dir.path(), runtime)?;
    let socket = runtime.join(nexa_proto::sockets::MEMORY);
    tokio::time::timeout(Duration::from_secs(5), async {
        while !socket.exists() {
            if let Some(status) = daemon.0.try_wait()? {
                return Err(io::Error::other(format!(
                    "memoryd exited before binding: {status}"
                )));
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok::<_, io::Error>(())
    })
    .await??;
    let mut memory = MemoryServiceClient::new(channel(&socket));
    let initial = memory
        .get_ambient_control(control_request())
        .await?
        .into_inner();
    assert!(!initial.paused);

    let (capture, frames) = mpsc::channel(4);
    let (calls, mut received) = mpsc::unbounded_channel();
    let brain = Brain::open(
        config,
        Dependencies {
            audio: Arc::new(Audio(Mutex::new(Some(frames)))),
            ml: Arc::new(Ml(calls.clone())),
            memory: Some(Arc::new(ObservedMemory(UdsMemory(memory.clone()), calls))),
            home: None,
            clock: Arc::new(SystemClock::default()),
        },
    )
    .await?;
    let worker = tokio::spawn(Arc::clone(&brain).run_voice());
    let pcm = 0.04_f32.to_le_bytes().repeat(12_800);
    let utterance = |id: &str| v1::AudioUtterance {
        utterance_id: id.into(),
        room_id: "living".into(),
        pcm_f32le: pcm.clone(),
        sample_rate: 16_000,
        captured_at_unix_ms: 1_800_000_000_000,
        kind: UtteranceKind::Ambient.into(),
        ..Default::default()
    };

    capture.send(Ok(utterance("ambient-active"))).await?;
    expect_status(&mut logs, "ambient-active", "observed").await?;
    // STT and ECAPA run concurrently. Observe must follow both.
    let first = receive(&mut received).await?;
    let second = receive(&mut received).await?;
    let (transcript, speaker) = match (first, second) {
        (Call::Transcribe(stt), Call::Embed(speaker))
        | (Call::Embed(speaker), Call::Transcribe(stt)) => (stt, speaker),
        other => panic!("unexpected inference calls: {other:?}"),
    };
    assert_eq!(transcript.mode(), SttMode::Ambient);
    assert_eq!(transcript.pcm_f32le, pcm);
    assert_eq!(speaker.pcm_f32le, pcm);
    assert_eq!(speaker.sample_rate, 16_000);
    let Call::Observe(observation) = receive(&mut received).await? else {
        return Err("expected a voice observation after both inference calls".into());
    };
    assert_eq!(observation.utterance_id, "ambient-active");
    assert_eq!(observation.room_id, "living");
    assert_eq!(observation.embedding.len(), 192);
    assert_eq!(observation.encoder_model_key, ENCODER);

    let paused = memory
        .set_ambient_pause(v1::SetAmbientPauseRequest {
            paused: true,
            expected_revision: initial.revision,
            actor: Some(actor(Role::Operator, &[Permission::AmbientControl])),
        })
        .await?
        .into_inner();
    assert!(paused.paused);
    capture.send(Ok(utterance("ambient-paused"))).await?;
    expect_status(&mut logs, "ambient-paused", "paused").await?;
    assert!(matches!(
        received.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));

    let captures = memory
        .query_voice_captures(v1::VoiceCaptureQueryRequest {
            query_json: "{}".into(),
            actor: Some(actor(
                Role::Operator,
                &[Permission::VoicesRead, Permission::VoicesReview],
            )),
        })
        .await?
        .into_inner();
    let page: Value = serde_json::from_str(&captures.payload_json)?;
    let persisted = page["captures"].as_array().ok_or("missing capture list")?;
    assert_eq!(persisted.len(), 1);
    assert_eq!(persisted[0]["utterance_id"], "ambient-active");
    assert_eq!(persisted[0]["encoder_model_key"], ENCODER);

    drop(daemon);
    capture.send(Ok(utterance("ambient-unavailable"))).await?;
    expect_status(&mut logs, "ambient-unavailable", "failed_closed").await?;
    assert!(matches!(
        received.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    brain.shutdown();
    worker.await?;
    Ok(())
}
