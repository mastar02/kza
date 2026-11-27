#![allow(clippy::expect_used)]

#[allow(dead_code)]
mod support;

use std::{
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Arc,
    time::Duration,
};

use nexa_brain::{
    Brain, Dependencies,
    clients::{AudioClient, MemoryClient, MlClient, StreamOf, UdsMemory, channel},
    control::Control,
};
use nexa_household::SystemClock;
use nexa_proto::v1::{
    self as v1, Permission, Role, SttMode, UtteranceKind,
    brain_control_service_server::BrainControlService, memory_service_client::MemoryServiceClient,
};
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Code, Request, Status};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

use support::{Result, actor, bundle, receive};

const ENCODER: &str = "speechbrain/spkrec-ecapa-voxceleb";
type RpcResult<T> = std::result::Result<T, Status>;
type CaptureReceiver = mpsc::Receiver<RpcResult<v1::AudioUtterance>>;

#[derive(Debug)]
struct Audio(Mutex<Option<CaptureReceiver>>, mpsc::UnboundedSender<()>);

#[tonic::async_trait]
impl AudioClient for Audio {
    async fn subscribe(
        &self,
        _include_ambient: bool,
    ) -> RpcResult<StreamOf<RpcResult<v1::AudioUtterance>>> {
        let rx = self
            .0
            .lock()
            .await
            .take()
            .ok_or_else(|| Status::unavailable("test stream closed"))?;
        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn play(
        &self,
        mut chunks: ReceiverStream<v1::PlayAudioChunk>,
    ) -> RpcResult<v1::PlayAudioReceipt> {
        use futures_util::StreamExt;
        let mut playback_id = String::new();
        while let Some(chunk) = chunks.next().await {
            if playback_id.is_empty() && !chunk.playback_id.is_empty() {
                playback_id = chunk.playback_id;
            }
            if chunk.is_final {
                break;
            }
        }
        let _ = self.1.send(());
        Ok(v1::PlayAudioReceipt {
            playback_id,
            completed: true,
            ..Default::default()
        })
    }

    async fn cancel(&self, _req: v1::CancelPlaybackRequest) -> RpcResult<()> {
        Ok(())
    }
}

#[derive(Debug)]
struct TestMl {
    transcribed_tx: mpsc::UnboundedSender<SttMode>,
    spoken_tx: mpsc::UnboundedSender<String>,
    empty_ambient: bool,
}

impl TestMl {
    fn new(
        empty_ambient: bool,
    ) -> (
        Self,
        mpsc::UnboundedReceiver<SttMode>,
        mpsc::UnboundedReceiver<String>,
    ) {
        let (transcribed_tx, stt_rx) = mpsc::unbounded_channel();
        let (spoken_tx, spoken_rx) = mpsc::unbounded_channel();
        (
            Self {
                transcribed_tx,
                spoken_tx,
                empty_ambient,
            },
            stt_rx,
            spoken_rx,
        )
    }
}

#[tonic::async_trait]
impl MlClient for TestMl {
    async fn transcribe(
        &self,
        request: v1::TranscribeRequest,
    ) -> RpcResult<v1::TranscribeResponse> {
        let _ = self.transcribed_tx.send(request.mode());
        let mut bytes = request.pcm_f32le;
        while bytes.last() == Some(&0) {
            bytes.pop();
        }
        let text = if self.empty_ambient && bytes.is_empty() {
            String::new()
        } else {
            String::from_utf8(bytes)
                .map_err(|_| Status::invalid_argument("fixture text not UTF-8"))?
        };
        Ok(v1::TranscribeResponse {
            text,
            avg_logprob: -0.2,
            no_speech_prob: 0.02,
            compression_ratio: 1.1,
            ..Default::default()
        })
    }

    async fn synthesize(
        &self,
        request: v1::SynthesizeRequest,
    ) -> RpcResult<StreamOf<RpcResult<v1::TtsAudioChunk>>> {
        let _ = self.spoken_tx.send(request.text);
        let stream = futures_util::stream::iter([Ok(v1::TtsAudioChunk {
            sequence: 0,
            sample_rate: 24_000,
            pcm_f32le: 0.25_f32.to_le_bytes().to_vec(),
            is_final: true,
        })]);
        Ok(Box::pin(stream))
    }

    async fn embed_speaker(
        &self,
        _request: v1::EmbedSpeakerRequest,
    ) -> RpcResult<v1::SpeakerEmbeddingResponse> {
        let mut vector = vec![0.0; 192];
        vector[0] = 1.0;
        Ok(v1::SpeakerEmbeddingResponse {
            vector,
            encoder_model_key: ENCODER.into(),
            latency_ms: 0,
        })
    }

    async fn embed_text(
        &self,
        request: v1::EmbedTextRequest,
    ) -> RpcResult<v1::BatchEmbeddingResponse> {
        let vectors = request
            .texts
            .iter()
            .map(|_| {
                let mut values = vec![0.0_f32; 1024];
                values[0] = 1.0;
                v1::EmbeddingVector { values }
            })
            .collect();
        Ok(v1::BatchEmbeddingResponse {
            vectors,
            model_id: "BAAI/bge-m3".into(),
            latency_ms: 0,
        })
    }
}

#[derive(Debug)]
struct ObservedMemory {
    inner: UdsMemory,
    observed: mpsc::UnboundedSender<v1::RecordVoiceObservationRequest>,
}

#[tonic::async_trait]
impl MemoryClient for ObservedMemory {
    async fn query(&self, request: v1::QueryMemoryRequest) -> RpcResult<v1::QueryMemoryResponse> {
        self.inner.query(request).await
    }

    async fn observe_voice(
        &self,
        request: v1::RecordVoiceObservationRequest,
    ) -> RpcResult<v1::VoiceObservationReceipt> {
        let receipt = self.inner.observe_voice(request.clone()).await?;
        let _ = self.observed.send(request);
        Ok(receipt)
    }

    async fn erase_conversation(
        &self,
        request: v1::EraseConversationRequest,
    ) -> RpcResult<v1::ErasureReceipt> {
        self.inner.erase_conversation(request).await
    }

    async fn overview(
        &self,
        request: v1::MemoryOverviewRequest,
    ) -> RpcResult<v1::MemoryOverviewResponse> {
        self.inner.overview(request).await
    }

    async fn mutate(
        &self,
        request: v1::MemoryMutationRequest,
    ) -> RpcResult<v1::MemoryMutationResponse> {
        self.inner.mutate(request).await
    }

    async fn ambient_control(&self) -> RpcResult<v1::AmbientControlStatus> {
        self.inner.ambient_control().await
    }

    async fn store_fact(&self, request: v1::StoreFactRequest) -> RpcResult<v1::MemoryFact> {
        self.inner.store_fact(request).await
    }

    async fn identify_voice(
        &self,
        request: v1::IdentifyVoiceRequest,
    ) -> RpcResult<v1::IdentifyVoiceResponse> {
        self.inner.identify_voice(request).await
    }

    async fn upsert_voice_personality(
        &self,
        request: v1::UpsertVoicePersonalityRequest,
    ) -> RpcResult<v1::VoicePersonality> {
        self.inner.upsert_voice_personality(request).await
    }

    async fn get_voice_personality(
        &self,
        request: v1::GetVoicePersonalityRequest,
    ) -> RpcResult<v1::VoicePersonality> {
        self.inner.get_voice_personality(request).await
    }
}

struct MemoryDaemon(Child);

impl MemoryDaemon {
    async fn start(
        config_dir: &Path,
        runtime: &Path,
    ) -> Result<(Self, MemoryServiceClient<tonic::transport::Channel>)> {
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
            .arg(config_dir)
            .env("RUNTIME_DIRECTORY", runtime)
            .env_remove("NOTIFY_SOCKET")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .spawn()?;
        let guard = Self(child);
        let socket = runtime.join(nexa_proto::sockets::MEMORY);
        tokio::time::timeout(Duration::from_secs(5), async {
            while !socket.exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await?;
        let client = MemoryServiceClient::new(channel(&socket));
        Ok((guard, client))
    }
}

impl Drop for MemoryDaemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn padded_text_pcm(text: &str) -> Vec<u8> {
    let mut bytes = text.as_bytes().to_vec();
    while !bytes.len().is_multiple_of(4) || bytes.is_empty() {
        bytes.push(0);
    }
    bytes
}

fn sse_reply(content: &str) -> ResponseTemplate {
    let payload = json!({"choices": [{"delta": {"content": content}}]});
    ResponseTemplate::new(200)
        .insert_header("Content-Type", "text/event-stream")
        .set_body_string(format!("data: {payload}\n\ndata: [DONE]\n\n"))
}

async fn seed_consented_alice(
    memory: &mut MemoryServiceClient<tonic::transport::Channel>,
) -> Result<String> {
    memory
        .set_voice_global_switch(v1::SetVoiceGlobalSwitchRequest {
            enabled: true,
            expected_revision: 0,
            actor: Some(actor(
                Role::Operator,
                &[Permission::MemoryAssign, Permission::AdminManage],
            )),
        })
        .await?;
    let uds = UdsMemory(memory.clone());
    let mut emb = vec![0.0_f32; 192];
    emb[0] = 1.0;
    uds.observe_voice(v1::RecordVoiceObservationRequest {
        utterance_id: "seed-alice-voice".into(),
        room_id: "living".into(),
        embedding: emb,
        encoder_model_key: ENCODER.into(),
        transcript_preview: "hola soy alice".into(),
        captured_at_unix_ms: 1_799_000_000_000,
        conversation_id: "seed-conv-alice".into(),
    })
    .await?;
    let profiles = memory
        .list_voice_profiles(v1::ListVoiceProfilesRequest {
            include_person: true,
            actor: Some(actor(
                Role::Operator,
                &[
                    Permission::VoicesRead,
                    Permission::VoicesReview,
                    Permission::AdminManage,
                ],
            )),
            cursor: String::new(),
        })
        .await?
        .into_inner()
        .profiles;
    let profile = &profiles[0];
    memory
        .link_voice_profile_to_person(v1::LinkVoiceProfileRequest {
            profile_id: profile.profile_id.clone(),
            person_id: "alice".into(),
            expected_profile_revision: profile.profile_revision,
            expected_link_revision: profile.link_revision,
            actor: Some(actor(
                Role::Operator,
                &[Permission::VoicesReview, Permission::AdminManage],
            )),
        })
        .await?;
    memory
        .set_voice_consent(v1::SetVoiceConsentRequest {
            user_id: "alice".into(),
            granted: true,
            actor: Some(v1::ActorContext {
                subject: "alice".into(),
                role: Role::Home as i32,
                user_id: "alice".into(),
                permissions: vec![Permission::MemoryOwnVoice as i32],
                ..Default::default()
            }),
        })
        .await?;
    Ok(profile.profile_id.clone())
}

#[tokio::test]
async fn ambient_conversation_distills_proposed_facts_and_personality_and_erases_cleanly() -> Result
{
    let local_llm = MockServer::start().await;
    let cloud_llm = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&cloud_llm)
        .await;

    let dir = tempfile::tempdir_in("/tmp")?;
    let mut config = bundle(dir.path())?;
    config.core.llm.fast_router_url = format!("{}/v1", local_llm.uri());
    config.core.llm.cloud_gateway_url = format!("{}/v1", cloud_llm.uri());
    config.core.llm.cloud_consent = false;
    config.core.memory.enabled = true;
    config.core.memory.ambient_enabled = true;
    config.core.memory.voice_proposals_enabled = true;

    let core_toml = include_str!("../../../config/nexa/nexa.toml").replace(
        "/var/lib/nexa",
        config.core.paths.data_dir.to_str().ok_or("data path")?,
    );
    tokio::fs::write(dir.path().join("nexa.toml"), core_toml).await?;
    let (_daemon, mut memory) =
        MemoryDaemon::start(dir.path(), &config.core.paths.runtime_dir).await?;
    let alice_profile = seed_consented_alice(&mut memory).await?;

    let distilled_json = json!({
        "facts": [{"content": "Prefiere cafe sin azucar", "category": "preferencia"}],
        "personalities": [{"profile_id": alice_profile, "summary": {"tone": "calido", "topics": ["cafe"]}}],
    })
    .to_string();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_reply(&distilled_json))
        .expect(1)
        .mount(&local_llm)
        .await;

    let (capture_tx, capture_rx) = mpsc::channel(8);
    let (obs_tx, mut obs_rx) = mpsc::unbounded_channel();
    let (play_tx, _play_rx) = mpsc::unbounded_channel();
    let (ml, _stt_rx, _spoken_rx) = TestMl::new(false);
    let brain = Brain::open(
        config,
        Dependencies {
            audio: Arc::new(Audio(Mutex::new(Some(capture_rx)), play_tx)),
            ml: Arc::new(ml),
            memory: Some(Arc::new(ObservedMemory {
                inner: UdsMemory(memory.clone()),
                observed: obs_tx,
            })),
            home: None,
            clock: Arc::new(SystemClock::default()),
        },
    )
    .await?;
    let worker = tokio::spawn(Arc::clone(&brain).run_voice());

    let ambient_utt = |id: &str, text: &str, at_ms: u64| v1::AudioUtterance {
        utterance_id: id.into(),
        room_id: "living".into(),
        kind: UtteranceKind::Ambient as i32,
        sample_rate: 16_000,
        pcm_f32le: padded_text_pcm(text),
        speech_ratio: 0.65,
        captured_at_unix_ms: at_ms,
        ..Default::default()
    };

    capture_tx
        .send(Ok(ambient_utt(
            "conv-u1",
            "A la manana tomo siempre cafe sin azucar",
            1_800_000_000_000,
        )))
        .await?;
    let obs1 = receive(&mut obs_rx).await?;
    capture_tx
        .send(Ok(ambient_utt(
            "conv-u2",
            "Me encanta el cafe bien cargado",
            1_800_000_010_000,
        )))
        .await?;
    let obs2 = receive(&mut obs_rx).await?;
    assert!(!obs1.conversation_id.is_empty());
    assert_eq!(obs1.conversation_id, obs2.conversation_id);

    // Gap > 45 s closes session 1 and triggers async local-only distillation in the background.
    capture_tx
        .send(Ok(ambient_utt(
            "conv-u3",
            "Nuevo tema mas tarde",
            1_800_000_080_000,
        )))
        .await?;
    let obs3 = receive(&mut obs_rx).await?;
    assert_ne!(obs3.conversation_id, obs1.conversation_id);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    let proposals = loop {
        let proposals_resp = memory
            .list_memory_overview(v1::MemoryOverviewRequest {
                view: "proposals".into(),
                cursor: String::new(),
                limit: 20,
                actor: Some(actor(Role::Operator, &[Permission::MemoryRead])),
            })
            .await?
            .into_inner();
        let proposals_doc: Value = serde_json::from_str(&proposals_resp.payload_json)?;
        let list = proposals_doc["proposals"]
            .as_array()
            .ok_or("missing proposals")?
            .clone();
        if list.iter().any(|p| {
            p["content"] == "Prefiere cafe sin azucar"
                && p["conversation_id"] == obs1.conversation_id
                && p["confirmed"] == false
        }) {
            break list;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for async distilled fact in {proposals_doc}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    assert!(!proposals.is_empty());

    let hits = memory
        .query_memory(v1::QueryMemoryRequest {
            query_embedding: vec![1.0; 1].into_iter().chain(vec![0.0; 1023]).collect(),
            top_k: 5,
            owner_user_id: String::new(),
            include_shared: true,
            actor: Some(actor(Role::Operator, &[Permission::MemoryRead])),
        })
        .await?
        .into_inner()
        .hits;
    assert!(
        hits.is_empty(),
        "unconfirmed proposed fact must not be returned by query_memory"
    );

    let personality = loop {
        match memory
            .get_voice_personality(v1::GetVoicePersonalityRequest {
                profile_id: alice_profile.clone(),
                actor: Some(actor(Role::Operator, &[Permission::MemoryRead])),
            })
            .await
        {
            Ok(resp) => break resp.into_inner(),
            Err(status) if tokio::time::Instant::now() < deadline => {
                let _ = status;
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
            Err(err) => return Err(err.into()),
        }
    };
    let summary: Value = serde_json::from_str(&personality.summary_json)?;
    assert_eq!(summary["tone"], "calido");

    Control(Arc::clone(&brain))
        .mutate_memory_item(Request::new(v1::MemoryMutationRequest {
            operation: "erase_conversation".into(),
            target_id: obs1.conversation_id.clone(),
            payload_json: "{}".into(),
            actor: Some(actor(
                Role::Operator,
                &[
                    Permission::ConversationsRead,
                    Permission::AdminManage,
                    Permission::MemoryReview,
                ],
            )),
        }))
        .await?;

    assert_eq!(
        memory
            .get_voice_personality(v1::GetVoicePersonalityRequest {
                profile_id: alice_profile,
                actor: Some(actor(Role::Operator, &[Permission::MemoryRead])),
            })
            .await
            .expect_err("personality erased with source conversation")
            .code(),
        Code::NotFound
    );

    brain.shutdown();
    worker.await?;
    Ok(())
}

#[tokio::test]
async fn media_tv_ambient_stream_is_filtered_without_observing_or_distilling() -> Result {
    let local_llm = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&local_llm)
        .await;

    let dir = tempfile::tempdir_in("/tmp")?;
    let mut config = bundle(dir.path())?;
    config.core.llm.fast_router_url = format!("{}/v1", local_llm.uri());
    config.core.memory.ambient_enabled = true;

    let core_toml = include_str!("../../../config/nexa/nexa.toml").replace(
        "/var/lib/nexa",
        config.core.paths.data_dir.to_str().ok_or("data path")?,
    );
    tokio::fs::write(dir.path().join("nexa.toml"), core_toml).await?;
    let (_daemon, memory) = MemoryDaemon::start(dir.path(), &config.core.paths.runtime_dir).await?;

    let (capture_tx, capture_rx) = mpsc::channel(8);
    let (obs_tx, mut obs_rx) = mpsc::unbounded_channel();
    let (play_tx, _play_rx) = mpsc::unbounded_channel();
    let (ml, mut stt_rx, _spoken_rx) = TestMl::new(true);
    let brain = Brain::open(
        config,
        Dependencies {
            audio: Arc::new(Audio(Mutex::new(Some(capture_rx)), play_tx)),
            ml: Arc::new(ml),
            memory: Some(Arc::new(ObservedMemory {
                inner: UdsMemory(memory.clone()),
                observed: obs_tx,
            })),
            home: None,
            clock: Arc::new(SystemClock::default()),
        },
    )
    .await?;
    let worker = tokio::spawn(Arc::clone(&brain).run_voice());

    let tv_utt = |id: &str, text: &str| v1::AudioUtterance {
        utterance_id: id.into(),
        room_id: "living".into(),
        kind: UtteranceKind::Ambient as i32,
        sample_rate: 16_000,
        pcm_f32le: padded_text_pcm(text),
        has_doa: true,
        doa_azimuth_rad: 1.1,
        doa_stability: 0.97,
        speech_ratio: 0.93,
        captured_at_unix_ms: 1_800_000_000_000,
        ..Default::default()
    };

    for (id, text) in [
        ("tv-1", "Bienvenidos al programa de la tarde"),
        ("tv-2", "Seguimos en vivo con noticias del dia"),
        ("tv-3", "En instantes llega el analisis completo"),
    ] {
        capture_tx.send(Ok(tv_utt(id, text))).await?;
        assert_eq!(receive(&mut stt_rx).await?, SttMode::Ambient);
        assert_eq!(receive(&mut obs_rx).await?.utterance_id, id);
    }

    capture_tx
        .send(Ok(tv_utt(
            "tv-4",
            "Continuamos con el informe meteorologico completo del dia",
        )))
        .await?;
    assert_eq!(receive(&mut stt_rx).await?, SttMode::Ambient);
    // Sentinel noise frame guarantees tv-4 completed before asserting obs_rx is empty.
    capture_tx
        .send(Ok(v1::AudioUtterance {
            utterance_id: "noise-sentinel".into(),
            room_id: "living".into(),
            kind: UtteranceKind::Ambient as i32,
            sample_rate: 16_000,
            pcm_f32le: vec![0; 4],
            captured_at_unix_ms: 1_800_000_005_000,
            ..Default::default()
        }))
        .await?;
    assert_eq!(receive(&mut stt_rx).await?, SttMode::Ambient);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        obs_rx.try_recv().is_err(),
        "static-DoA high-density tv-4 must be filtered as MediaTv without observe_voice"
    );

    brain.shutdown();
    worker.await?;
    Ok(())
}

#[tokio::test]
async fn command_turns_inject_short_term_history_and_personality_only_locally_never_cloud() -> Result
{
    let local_llm = MockServer::start().await;
    let cloud_llm = MockServer::start().await;

    let dir = tempfile::tempdir_in("/tmp")?;
    let mut config = bundle(dir.path())?;
    config.core.llm.fast_router_url = format!("{}/v1", local_llm.uri());
    config.core.llm.cloud_gateway_url = format!("{}/v1", cloud_llm.uri());
    config.core.llm.cloud_consent = true;
    config.core.memory.enabled = true;
    config.core.memory.short_term_turns = 10;

    let core_toml = include_str!("../../../config/nexa/nexa.toml").replace(
        "/var/lib/nexa",
        config.core.paths.data_dir.to_str().ok_or("data path")?,
    );
    tokio::fs::write(dir.path().join("nexa.toml"), core_toml).await?;
    let (_daemon, mut memory) =
        MemoryDaemon::start(dir.path(), &config.core.paths.runtime_dir).await?;
    let alice_profile = seed_consented_alice(&mut memory).await?;
    memory
        .upsert_voice_personality(v1::UpsertVoicePersonalityRequest {
            profile_id: alice_profile.clone(),
            summary_json: r#"{"tone":"directo"}"#.into(),
            source_conversation_id: "seed-conv-alice".into(),
            actor: Some(actor(
                Role::Operator,
                &[Permission::MemoryReview, Permission::VoicesReview],
            )),
        })
        .await?;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(wiremock::matchers::body_string_contains(
            "contame algo breve",
        ))
        .respond_with(sse_reply("Todo bien por aca."))
        .expect(1)
        .up_to_n_times(1)
        .mount(&local_llm)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(wiremock::matchers::body_string_contains("decime mas"))
        .respond_with(sse_reply("Seguimos charlando."))
        .expect(1)
        .up_to_n_times(1)
        .mount(&local_llm)
        .await;

    let (capture_tx, capture_rx) = mpsc::channel(8);
    let (obs_tx, _obs_rx) = mpsc::unbounded_channel();
    let (play_tx, mut play_rx) = mpsc::unbounded_channel();
    let (ml, _stt_rx, mut spoken_rx) = TestMl::new(false);
    let brain = Brain::open(
        config,
        Dependencies {
            audio: Arc::new(Audio(Mutex::new(Some(capture_rx)), play_tx)),
            ml: Arc::new(ml),
            memory: Some(Arc::new(ObservedMemory {
                inner: UdsMemory(memory),
                observed: obs_tx,
            })),
            home: None,
            clock: Arc::new(SystemClock::default()),
        },
    )
    .await?;
    let worker = tokio::spawn(Arc::clone(&brain).run_voice());

    let cmd = |id: &str, text: &str| v1::AudioUtterance {
        utterance_id: id.into(),
        room_id: "living".into(),
        kind: UtteranceKind::Command as i32,
        sample_rate: 16_000,
        pcm_f32le: padded_text_pcm(text),
        wake_score: 0.99,
        rms: 0.2,
        ..Default::default()
    };

    capture_tx
        .send(Ok(cmd("cmd-1", "Nexa, contame algo breve")))
        .await?;
    assert_eq!(receive(&mut spoken_rx).await?, "Todo bien por aca.");
    receive(&mut play_rx).await?;
    tokio::time::sleep(Duration::from_millis(20)).await;

    capture_tx
        .send(Ok(cmd("cmd-2", "Nexa, decime mas")))
        .await?;
    assert_eq!(receive(&mut spoken_rx).await?, "Seguimos charlando.");
    receive(&mut play_rx).await?;
    tokio::time::sleep(Duration::from_millis(20)).await;

    let local_reqs = local_llm.received_requests().await.ok_or("no recording")?;
    assert_eq!(local_reqs.len(), 2);
    let second_body: Value = serde_json::from_slice(&local_reqs[1].body)?;
    let msgs = second_body["messages"]
        .as_array()
        .ok_or("missing messages")?;
    assert!(
        msgs[0]["content"]
            .as_str()
            .is_some_and(|s| s.contains("Personalidad observada:") && s.contains("directo")),
        "expected personality in local system prompt: {}",
        msgs[0]
    );
    assert_eq!(msgs[1]["role"], "user");
    assert!(
        msgs[1]["content"]
            .as_str()
            .is_some_and(|s| s.contains("contame algo breve")),
        "unexpected history turn 1: {}",
        msgs[1]
    );
    assert_eq!(msgs[2]["role"], "assistant");
    assert_eq!(msgs[2]["content"], "Todo bien por aca.");
    assert_eq!(msgs[3]["role"], "user");
    assert!(
        msgs[3]["content"]
            .as_str()
            .is_some_and(|s| s.contains("decime mas")),
        "unexpected current user prompt: {}",
        msgs[3]
    );

    local_llm.reset().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .expect(1)
        .mount(&local_llm)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_reply("Estrella de plasma."))
        .expect(1)
        .mount(&cloud_llm)
        .await;

    capture_tx
        .send(Ok(cmd("cmd-3", "Nexa, explica que es una estrella")))
        .await?;
    assert_eq!(receive(&mut spoken_rx).await?, "Estrella de plasma.");
    receive(&mut play_rx).await?;

    let cloud_reqs = cloud_llm.received_requests().await.ok_or("no recording")?;
    assert_eq!(cloud_reqs.len(), 1);
    let cloud_text = String::from_utf8_lossy(&cloud_reqs[0].body);
    let cloud_body: Value = serde_json::from_slice(&cloud_reqs[0].body)?;
    let cloud_msgs = cloud_body["messages"]
        .as_array()
        .ok_or("missing cloud messages")?;
    assert_eq!(cloud_msgs.len(), 2);
    assert_eq!(cloud_msgs[1]["role"], "user");
    assert!(
        cloud_msgs[1]["content"]
            .as_str()
            .is_some_and(|s| s.contains("explica que es una estrella")),
        "unexpected cloud user prompt: {}",
        cloud_msgs[1]
    );
    for leak in ["directo", "Personalidad", "contame", "Todo bien"] {
        assert!(
            !cloud_text.contains(leak),
            "cloud payload leaked local context ({leak}): {cloud_text}"
        );
    }

    brain.shutdown();
    worker.await?;
    Ok(())
}

#[tokio::test]
async fn consented_speaker_long_static_monologue_is_not_filtered_as_media_tv() -> Result {
    let local_llm = MockServer::start().await;
    let dir = tempfile::tempdir_in("/tmp")?;
    let mut config = bundle(dir.path())?;
    config.core.llm.fast_router_url = format!("{}/v1", local_llm.uri());
    config.core.memory.enabled = true;
    config.core.memory.ambient_enabled = true;

    let core_toml = include_str!("../../../config/nexa/nexa.toml").replace(
        "/var/lib/nexa",
        config.core.paths.data_dir.to_str().ok_or("data path")?,
    );
    tokio::fs::write(dir.path().join("nexa.toml"), core_toml).await?;
    let (_daemon, mut memory) =
        MemoryDaemon::start(dir.path(), &config.core.paths.runtime_dir).await?;
    let _alice_profile = seed_consented_alice(&mut memory).await?;

    let (capture_tx, capture_rx) = mpsc::channel(16);
    let (obs_tx, mut obs_rx) = mpsc::unbounded_channel();
    let (play_tx, _play_rx) = mpsc::unbounded_channel();
    let (ml, _stt_rx, _spoken_rx) = TestMl::new(false);
    let brain = Brain::open(
        config,
        Dependencies {
            audio: Arc::new(Audio(Mutex::new(Some(capture_rx)), play_tx)),
            ml: Arc::new(ml),
            memory: Some(Arc::new(ObservedMemory {
                inner: UdsMemory(memory),
                observed: obs_tx,
            })),
            home: None,
            clock: Arc::new(SystemClock::default()),
        },
    )
    .await?;
    let worker = tokio::spawn(Arc::clone(&brain).run_voice());

    for idx in 0..5 {
        let id = format!("mono-{idx}");
        capture_tx
            .send(Ok(v1::AudioUtterance {
                utterance_id: id.clone(),
                room_id: "living".into(),
                kind: UtteranceKind::Ambient as i32,
                sample_rate: 16_000,
                pcm_f32le: padded_text_pcm(&format!(
                    "Segmento {idx} de mi monologo sentado quieto frente al escritorio"
                )),
                has_doa: true,
                doa_azimuth_rad: 1.1,
                doa_stability: 0.98,
                speech_ratio: 0.96,
                captured_at_unix_ms: 1_800_000_000_000 + (idx as u64) * 12_000,
                ..Default::default()
            }))
            .await?;
        let obs = receive(&mut obs_rx).await?;
        assert_eq!(obs.utterance_id, id);
    }

    brain.shutdown();
    worker.await?;
    Ok(())
}

#[tokio::test]
async fn one_hour_continuous_conversation_preserves_full_transcripts_single_id_rolling_distillation_and_atomic_erasure()
-> Result {
    let local_llm = MockServer::start().await;
    let distilled_json = json!({
        "facts": [{"content": "Planificación semanal de compras y agenda familiar en marcha", "category": "rutina"}],
        "personalities": [],
    })
    .to_string();
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(sse_reply(&distilled_json))
        .mount(&local_llm)
        .await;

    let dir = tempfile::tempdir_in("/tmp")?;
    let mut config = bundle(dir.path())?;
    config.core.llm.fast_router_url = format!("{}/v1", local_llm.uri());
    config.core.memory.enabled = true;
    config.core.memory.ambient_enabled = true;
    config.core.memory.voice_proposals_enabled = true;

    let core_toml = include_str!("../../../config/nexa/nexa.toml").replace(
        "/var/lib/nexa",
        config.core.paths.data_dir.to_str().ok_or("data path")?,
    );
    tokio::fs::write(dir.path().join("nexa.toml"), core_toml).await?;
    let (_daemon, mut memory) =
        MemoryDaemon::start(dir.path(), &config.core.paths.runtime_dir).await?;

    let (capture_tx, capture_rx) = mpsc::channel(64);
    let (obs_tx, mut obs_rx) = mpsc::unbounded_channel();
    let (play_tx, _play_rx) = mpsc::unbounded_channel();
    let (ml, _stt_rx, _spoken_rx) = TestMl::new(false);
    let brain = Brain::open(
        config,
        Dependencies {
            audio: Arc::new(Audio(Mutex::new(Some(capture_rx)), play_tx)),
            ml: Arc::new(ml),
            memory: Some(Arc::new(ObservedMemory {
                inner: UdsMemory(memory.clone()),
                observed: obs_tx,
            })),
            home: None,
            clock: Arc::new(SystemClock::default()),
        },
    )
    .await?;
    let worker = tokio::spawn(Arc::clone(&brain).run_voice());

    let mut shared_conv_id: Option<String> = None;
    let mut first_expected_text = String::new();
    let base_ts = 1_800_000_000_000_u64;
    // 300 consecutive 12-second segments = 3,600,000 ms (1 full hour).
    for idx in 0..300_usize {
        let continues = idx % 2 == 1;
        let text = if !continues {
            format!(
                "Bloque {idx:03} de la conversación continua sobre logística del hogar, música, educación, compras de la semana, recetas saludables y preparación de la reunión del próximo viernes con mucha calma para organizar"
            )
        } else {
            format!(
                "para organizar mejor cada detalle pendiente del cronograma semanal número {idx:03}, revisando entregas, presupuestos domésticos, turnos médicos y listas compartidas en perfecto español."
            )
        };
        if idx == 0 {
            first_expected_text = text.clone();
            assert!(first_expected_text.chars().count() > 200);
        }
        capture_tx
            .send(Ok(v1::AudioUtterance {
                utterance_id: format!("hour-u-{idx:03}"),
                room_id: "living".into(),
                kind: UtteranceKind::Ambient as i32,
                sample_rate: 16_000,
                pcm_f32le: padded_text_pcm(&text),
                speech_ratio: 0.65,
                captured_at_unix_ms: base_ts + (idx as u64) * 12_000,
                continues_segment: continues,
                ..Default::default()
            }))
            .await?;
        let obs = receive(&mut obs_rx).await?;
        match &shared_conv_id {
            Some(expected) => assert_eq!(
                &obs.conversation_id, expected,
                "1h conversation split unexpectedly at segment {idx}"
            ),
            None => shared_conv_id = Some(obs.conversation_id),
        }
    }
    let conversation_id = shared_conv_id.ok_or("missing conversation_id")?;

    let review_permissions = &[Permission::VoicesRead, Permission::VoicesReview];
    let mut cursor = String::new();
    let first_capture_id = loop {
        let captures_resp = memory
            .query_voice_captures(v1::VoiceCaptureQueryRequest {
                query_json: json!({"limit": 100, "cursor": cursor}).to_string(),
                actor: Some(actor(Role::Operator, review_permissions)),
            })
            .await?
            .into_inner();
        let captures_doc: Value = serde_json::from_str(&captures_resp.payload_json)?;
        if let Some(found) = captures_doc["captures"]
            .as_array()
            .and_then(|arr| arr.iter().find(|c| c["utterance_id"] == "hour-u-000"))
        {
            break found["capture_id"]
                .as_str()
                .ok_or("missing capture_id")?
                .to_owned();
        }
        let Some(next) = captures_doc["next"].as_str() else {
            return Err("missing hour-u-000 capture across paginated query".into());
        };
        cursor = next.to_owned();
    };
    let review_resp = memory
        .get_voice_capture_review(v1::VoiceCaptureReviewRequest {
            fingerprint_id: first_capture_id.clone(),
            include_transcript: true,
            actor: Some(actor(
                Role::Operator,
                &[
                    Permission::VoicesRead,
                    Permission::VoicesReview,
                    Permission::ConversationsRead,
                ],
            )),
        })
        .await?
        .into_inner();
    let review_doc: Value = serde_json::from_str(&review_resp.payload_json)?;
    assert_eq!(
        review_doc["transcript_preview"].as_str(),
        Some(first_expected_text.as_str()),
        "stored transcript must not be truncated at 160 chars"
    );

    let deadline = tokio::time::Instant::now() + Duration::from_secs(6);
    let llm_reqs = loop {
        let reqs = local_llm.received_requests().await.ok_or("no recording")?;
        let proposals_resp = memory
            .list_memory_overview(v1::MemoryOverviewRequest {
                view: "proposals".into(),
                cursor: String::new(),
                limit: 50,
                actor: Some(actor(Role::Operator, &[Permission::MemoryRead])),
            })
            .await?
            .into_inner();
        let proposals_doc: Value = serde_json::from_str(&proposals_resp.payload_json)?;
        let has_fact = proposals_doc["proposals"]
            .as_array()
            .is_some_and(|arr| arr.iter().any(|p| p["conversation_id"] == conversation_id));
        if reqs.len() >= 4 && has_fact {
            break reqs;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out waiting for rolling 1h distillation (reqs={}, docs={proposals_doc})",
            reqs.len()
        );
        tokio::time::sleep(Duration::from_millis(40)).await;
    };
    let req0_body = String::from_utf8_lossy(&llm_reqs[0].body);
    assert!(req0_body.contains("Bloque 000"));
    assert!(
        !req0_body.contains("para organizar para organizar"),
        "boundary overlap was not deduplicated: {req0_body}"
    );
    let has_rolling_context = llm_reqs[1..].iter().any(|req| {
        String::from_utf8_lossy(&req.body).contains("Contexto previo de esta misma conversación")
    });
    assert!(
        has_rolling_context,
        "later rolling windows did not include prior context"
    );

    Control(Arc::clone(&brain))
        .mutate_memory_item(Request::new(v1::MemoryMutationRequest {
            operation: "erase_conversation".into(),
            target_id: conversation_id.clone(),
            payload_json: "{}".into(),
            actor: Some(actor(
                Role::Operator,
                &[
                    Permission::ConversationsRead,
                    Permission::AdminManage,
                    Permission::MemoryReview,
                ],
            )),
        }))
        .await?;

    let proposals_after = memory
        .list_memory_overview(v1::MemoryOverviewRequest {
            view: "proposals".into(),
            cursor: String::new(),
            limit: 50,
            actor: Some(actor(Role::Operator, &[Permission::MemoryRead])),
        })
        .await?
        .into_inner();
    let after_doc: Value = serde_json::from_str(&proposals_after.payload_json)?;
    assert!(
        !after_doc["proposals"]
            .as_array()
            .is_some_and(|arr| arr.iter().any(|p| p["conversation_id"] == conversation_id)),
        "expected atomic erasure of 1h proposed facts: {after_doc}"
    );
    assert_eq!(
        memory
            .get_voice_capture_review(v1::VoiceCaptureReviewRequest {
                fingerprint_id: first_capture_id,
                include_transcript: false,
                actor: Some(actor(Role::Operator, review_permissions)),
            })
            .await
            .expect_err("capture should be erased with 1h conversation")
            .code(),
        Code::NotFound
    );

    brain.shutdown();
    worker.await?;
    Ok(())
}
