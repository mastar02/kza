use std::{
    error::Error,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicI64, Ordering},
    },
    time::Duration,
};

use nexa_brain::{
    Brain, Dependencies, HomeDependencies,
    clients::{StreamOf, UdsAudio, UdsMemory, UdsMl, channel},
    control::BoundControl,
};
use nexa_config::{LoadedBundle, NexaConfig};
use nexa_household::Clock;
use nexa_llm::CancellationToken;
use nexa_proto::v1::{
    self as v1,
    audio_service_server::{AudioService, AudioServiceServer},
    brain_control_service_client::BrainControlServiceClient,
    memory_service_server::{MemoryService, MemoryServiceServer},
    ml_service_server::{MlService, MlServiceServer},
    *,
};
use tempfile::TempDir;
use tokio::{
    net::UnixListener,
    sync::{Mutex, mpsc},
    task::JoinSet,
};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::{
    Request, Response, Status, Streaming,
    transport::{Channel, Server},
};

pub type Result<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
type Reply<T> = std::result::Result<Response<T>, Status>;
type Stream<T> = StreamOf<std::result::Result<T, Status>>;

#[derive(Debug)]
pub struct TestClock(AtomicI64);

impl TestClock {
    pub fn advance(&self, seconds: i64) {
        self.0.fetch_add(seconds * 1000, Ordering::SeqCst);
    }
}

impl Clock for TestClock {
    fn now(&self) -> i64 {
        self.0.load(Ordering::SeqCst)
    }

    fn monotonic(&self) -> Duration {
        Duration::from_millis(u64::try_from(self.now() - 1_800_000_000_000).unwrap_or_default())
    }
}

#[derive(Debug)]
pub struct Audio {
    capture: Mutex<Option<mpsc::Receiver<std::result::Result<AudioUtterance, Status>>>>,
    played: mpsc::UnboundedSender<PlayAudioChunk>,
    cancelled: mpsc::UnboundedSender<CancelPlaybackRequest>,
}

#[tonic::async_trait]
impl AudioService for Audio {
    type SubscribeUtterancesStream = Stream<AudioUtterance>;

    async fn subscribe_utterances(
        &self,
        _: Request<SubscribeUtterancesRequest>,
    ) -> Reply<Self::SubscribeUtterancesStream> {
        let capture = self
            .capture
            .lock()
            .await
            .take()
            .ok_or_else(|| Status::failed_precondition("already subscribed"))?;
        Ok(Response::new(Box::pin(ReceiverStream::new(capture))))
    }

    async fn play_audio_stream(
        &self,
        request: Request<Streaming<PlayAudioChunk>>,
    ) -> Reply<PlayAudioReceipt> {
        let mut stream = request.into_inner();
        let mut id = None;
        let mut sent = false;
        while let Some(chunk) = stream.message().await? {
            if chunk.zone_id != "zone_living"
                || chunk.sample_rate != 24000
                || chunk.barge_in_token != chunk.playback_id
            {
                return Err(Status::invalid_argument("inconsistent playback metadata"));
            }
            if id.get_or_insert_with(|| chunk.playback_id.clone()) != &chunk.playback_id {
                return Err(Status::invalid_argument("playback ID changed"));
            }
            sent |= !chunk.pcm_f32le.is_empty();
            self.played
                .send(chunk.clone())
                .map_err(|_| Status::cancelled("test done"))?;
            if chunk.is_final && sent {
                return Ok(Response::new(PlayAudioReceipt {
                    playback_id: chunk.playback_id,
                    completed: true,
                    ..PlayAudioReceipt::default()
                }));
            }
        }
        Err(Status::cancelled("playback stream cancelled"))
    }

    async fn cancel_playback(&self, request: Request<CancelPlaybackRequest>) -> Reply<Empty> {
        self.cancelled
            .send(request.into_inner())
            .map_err(|_| Status::cancelled("test done"))?;
        Ok(Response::new(Empty {}))
    }
    async fn get_health(&self, _: Request<Empty>) -> Reply<HealthResponse> {
        Err(Status::unimplemented("get_health"))
    }
    async fn get_microphones(&self, _: Request<Empty>) -> Reply<MicrophonesSnapshot> {
        Err(Status::unimplemented("get_microphones"))
    }
    async fn get_zones(&self, _: Request<Empty>) -> Reply<ZonesSnapshot> {
        Err(Status::unimplemented("get_zones"))
    }
    async fn set_zone_volume(&self, _: Request<SetZoneVolumeRequest>) -> Reply<ZoneState> {
        Err(Status::unimplemented("set_zone_volume"))
    }
    async fn get_capture_control(&self, _: Request<Empty>) -> Reply<CaptureControlStatus> {
        Err(Status::unimplemented("get_capture_control"))
    }
    async fn set_capture_mute(
        &self,
        _: Request<SetCaptureMuteRequest>,
    ) -> Reply<CaptureControlStatus> {
        Err(Status::unimplemented("set_capture_mute"))
    }
    async fn forget_carried_owners(
        &self,
        _: Request<ForgetCarriedOwnersRequest>,
    ) -> Reply<ForgetCarriedOwnersResponse> {
        Err(Status::unimplemented("forget_carried_owners"))
    }
}

#[derive(Debug)]
pub struct Ml {
    pub synthesis: mpsc::UnboundedSender<String>,
    pub transcription: mpsc::UnboundedSender<SttMode>,
}

#[tonic::async_trait]
impl MlService for Ml {
    type SynthesizeStreamStream = Stream<TtsAudioChunk>;

    async fn transcribe(&self, request: Request<TranscribeRequest>) -> Reply<TranscribeResponse> {
        let request = request.into_inner();
        self.transcription
            .send(request.mode())
            .map_err(|_| Status::cancelled("test done"))?;
        let mut bytes = request.pcm_f32le;
        while bytes.last() == Some(&0) {
            bytes.pop();
        }
        let text = String::from_utf8(bytes)
            .map_err(|_| Status::invalid_argument("fixture text not UTF-8"))?;
        Ok(Response::new(TranscribeResponse {
            text,
            avg_logprob: -0.1,
            no_speech_prob: 0.01,
            compression_ratio: 1.1,
            ..TranscribeResponse::default()
        }))
    }

    async fn synthesize_stream(
        &self,
        request: Request<SynthesizeRequest>,
    ) -> Reply<Self::SynthesizeStreamStream> {
        self.synthesis
            .send(request.into_inner().text)
            .map_err(|_| Status::cancelled("test done"))?;
        let stream = futures_util::stream::iter([Ok(TtsAudioChunk {
            sequence: 0,
            sample_rate: 24000,
            pcm_f32le: 0.25_f32.to_le_bytes().to_vec(),
            is_final: true,
        })]);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn embed_speaker(
        &self,
        request: Request<EmbedSpeakerRequest>,
    ) -> Reply<SpeakerEmbeddingResponse> {
        if request.get_ref().sample_rate != 16000 {
            return Err(Status::invalid_argument("unexpected sample rate"));
        }
        Ok(Response::new(SpeakerEmbeddingResponse {
            vector: vec![0.01; 192],
            encoder_model_key: "test-encoder".into(),
            latency_ms: 0,
        }))
    }

    async fn embed_text(&self, _: Request<EmbedTextRequest>) -> Reply<BatchEmbeddingResponse> {
        Ok(Response::new(BatchEmbeddingResponse {
            vectors: vec![EmbeddingVector {
                values: vec![0.1; 4],
            }],
            model_id: "test-embedding".into(),
            latency_ms: 0,
        }))
    }

    async fn get_health(&self, _: Request<Empty>) -> Reply<MlHealthResponse> {
        Err(Status::unimplemented("get_health"))
    }

    async fn reload_model(&self, _: Request<ReloadModelRequest>) -> Reply<HealthResponse> {
        Err(Status::unimplemented("reload_model"))
    }
}

#[derive(Debug)]
pub struct Memory {
    queries: mpsc::UnboundedSender<QueryMemoryRequest>,
    observations: mpsc::UnboundedSender<RecordVoiceObservationRequest>,
}

#[tonic::async_trait]
impl MemoryService for Memory {
    async fn query_memory(
        &self,
        request: Request<QueryMemoryRequest>,
    ) -> Reply<QueryMemoryResponse> {
        self.queries
            .send(request.into_inner())
            .map_err(|_| Status::cancelled("test done"))?;
        Ok(Response::new(QueryMemoryResponse::default()))
    }

    async fn record_voice_observation(
        &self,
        request: Request<RecordVoiceObservationRequest>,
    ) -> Reply<VoiceObservationReceipt> {
        self.observations
            .send(request.into_inner())
            .map_err(|_| Status::cancelled("test done"))?;
        Ok(Response::new(VoiceObservationReceipt {
            capture_id: "cap".into(),
            matched_profile_id: "acoustic-profile".into(),
            similarity_score: 0.99,
        }))
    }

    async fn erase_conversation(
        &self,
        request: Request<EraseConversationRequest>,
    ) -> Reply<ErasureReceipt> {
        if request.get_ref().actor.is_none() || request.get_ref().conversation_id.is_empty() {
            return Err(Status::permission_denied("missing erasure scope"));
        }
        Ok(Response::new(ErasureReceipt {
            erasure_id: "erased".into(),
            tombstone_recorded: true,
            ..ErasureReceipt::default()
        }))
    }
    async fn get_health(&self, _: Request<Empty>) -> Reply<HealthResponse> {
        Err(Status::unimplemented("get_health"))
    }
    async fn store_fact(&self, _: Request<StoreFactRequest>) -> Reply<MemoryFact> {
        Err(Status::unimplemented("store_fact"))
    }
    async fn delete_fact(&self, _: Request<DeleteFactRequest>) -> Reply<ErasureReceipt> {
        Err(Status::unimplemented("delete_fact"))
    }
    async fn list_voice_profiles(
        &self,
        _: Request<ListVoiceProfilesRequest>,
    ) -> Reply<ListVoiceProfilesResponse> {
        Err(Status::unimplemented("list_voice_profiles"))
    }
    async fn link_voice_profile_to_person(
        &self,
        _: Request<LinkVoiceProfileRequest>,
    ) -> Reply<VoiceProfile> {
        Err(Status::unimplemented("link_voice_profile_to_person"))
    }
    async fn set_voice_consent(
        &self,
        _: Request<SetVoiceConsentRequest>,
    ) -> Reply<VoiceConsentStatus> {
        Err(Status::unimplemented("set_voice_consent"))
    }
    async fn purge_retired_user(
        &self,
        _: Request<PurgeRetiredUserRequest>,
    ) -> Reply<ErasureReceipt> {
        Err(Status::unimplemented("purge_retired_user"))
    }
    async fn get_ambient_control(&self, _: Request<Empty>) -> Reply<AmbientControlStatus> {
        Err(Status::unimplemented("get_ambient_control"))
    }
    async fn set_ambient_pause(
        &self,
        _: Request<SetAmbientPauseRequest>,
    ) -> Reply<AmbientControlStatus> {
        Err(Status::unimplemented("set_ambient_pause"))
    }
}

pub fn bundle(dir: &Path) -> Result<LoadedBundle> {
    let mut core = NexaConfig::from_toml_str(include_str!("../../../../config/nexa/nexa.toml"))?;
    core.paths.data_dir = dir.join("data");
    core.paths.runtime_dir = dir.join("r");
    core.paths.secrets_dir = dir.join("secrets");
    Ok(LoadedBundle {
        core,
        rooms: nexa_config::RoomsConfig::load_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../config/nexa/rooms.example.toml"
        ))?,
        ha_controls: nexa_config::HaControlPolicyConfig {
            entities: [(
                "light.grupo_living".to_owned(),
                vec!["turn_on".to_owned(), "turn_off".to_owned()],
            )]
            .into(),
        },
    })
}

pub struct Harness {
    pub dir: TempDir,
    pub config: LoadedBundle,
    pub brain: Arc<Brain>,
    pub client: BrainControlServiceClient<Channel>,
    pub clock: Arc<TestClock>,
    pub spoken: mpsc::UnboundedReceiver<String>,
    pub played: mpsc::UnboundedReceiver<PlayAudioChunk>,
    pub cancelled: mpsc::UnboundedReceiver<CancelPlaybackRequest>,
    pub modes: mpsc::UnboundedReceiver<SttMode>,
    pub queries: mpsc::UnboundedReceiver<QueryMemoryRequest>,
    pub observations: mpsc::UnboundedReceiver<RecordVoiceObservationRequest>,
    capture: mpsc::Sender<std::result::Result<AudioUtterance, Status>>,
    shutdown: CancellationToken,
    jobs: JoinSet<Result>,
}

impl Harness {
    pub async fn start(configure: impl FnOnce(&mut LoadedBundle), ha: bool) -> Result<Self> {
        // The short base keeps UDS paths within sun_path on both macOS and Linux.
        let dir = tempfile::tempdir_in("/tmp")?;
        let mut config = bundle(dir.path())?;
        configure(&mut config);
        let runtime = &config.core.paths.runtime_dir;
        let control = BoundControl::bind(&runtime.join("brain.sock"))?;
        let (capture, from_capture) = mpsc::channel(12);
        let (play, played) = mpsc::unbounded_channel();
        let (cancel, cancelled) = mpsc::unbounded_channel();
        let (synthesis, spoken) = mpsc::unbounded_channel();
        let (transcription, modes) = mpsc::unbounded_channel();
        let (query, queries) = mpsc::unbounded_channel();
        let (observe, observations) = mpsc::unbounded_channel();
        let audio = Audio {
            capture: Mutex::new(Some(from_capture)),
            played: play,
            cancelled: cancel,
        };
        let ml = Ml {
            synthesis,
            transcription,
        };
        let memory = Memory {
            queries: query,
            observations: observe,
        };
        let shutdown = CancellationToken::new();
        let mut jobs = JoinSet::new();
        let listener = UnixListener::bind(runtime.join("audio.sock"))?;
        let stop = shutdown.clone();
        jobs.spawn(async move {
            Server::builder()
                .add_service(AudioServiceServer::new(audio))
                .serve_with_incoming_shutdown(
                    UnixListenerStream::new(listener),
                    stop.cancelled_owned(),
                )
                .await?;
            Ok(())
        });
        let listener = UnixListener::bind(runtime.join("ml.sock"))?;
        let stop = shutdown.clone();
        jobs.spawn(async move {
            Server::builder()
                .add_service(MlServiceServer::new(ml))
                .serve_with_incoming_shutdown(
                    UnixListenerStream::new(listener),
                    stop.cancelled_owned(),
                )
                .await?;
            Ok(())
        });
        let listener = UnixListener::bind(runtime.join("memory.sock"))?;
        let stop = shutdown.clone();
        jobs.spawn(async move {
            Server::builder()
                .add_service(MemoryServiceServer::new(memory))
                .serve_with_incoming_shutdown(
                    UnixListenerStream::new(listener),
                    stop.cancelled_owned(),
                )
                .await?;
            Ok(())
        });
        let clock = Arc::new(TestClock(AtomicI64::new(1_800_000_000_000)));
        let deps = Dependencies {
            audio: Arc::new(UdsAudio(v1::audio_service_client::AudioServiceClient::new(
                channel(&runtime.join("audio.sock")),
            ))),
            ml: Arc::new(UdsMl(v1::ml_service_client::MlServiceClient::new(channel(
                &runtime.join("ml.sock"),
            )))),
            memory: Some(Arc::new(UdsMemory(
                v1::memory_service_client::MemoryServiceClient::new(channel(
                    &runtime.join("memory.sock"),
                )),
            ))),
            home: if ha {
                Some(HomeDependencies::new(&config, "ha-test-token")?)
            } else {
                None
            },
            clock: Arc::clone(&clock) as _,
        };
        let brain = Brain::open(config.clone(), deps).await?;
        jobs.spawn(control.serve(Arc::clone(&brain)));
        let voice = Arc::clone(&brain);
        jobs.spawn(async move {
            voice.run_voice().await;
            Ok(())
        });
        Ok(Self {
            client: BrainControlServiceClient::new(channel(&runtime.join("brain.sock"))),
            brain,
            config,
            dir,
            clock,
            spoken,
            played,
            cancelled,
            modes,
            queries,
            observations,
            capture,
            shutdown,
            jobs,
        })
    }

    pub async fn utterance(&self, id: &str, text: &str, kind: UtteranceKind) -> Result {
        let mut pcm = text.as_bytes().to_vec();
        while !pcm.len().is_multiple_of(4) {
            pcm.push(0);
        }
        self.capture
            .send(Ok(AudioUtterance {
                utterance_id: id.into(),
                room_id: "living".into(),
                kind: kind.into(),
                sample_rate: 16000,
                pcm_f32le: pcm,
                wake_score: if kind == UtteranceKind::Command {
                    0.99
                } else {
                    0.0
                },
                rms: 0.3,
                parakeet_shadow_text: text.into(),
                captured_at_unix_ms: u64::try_from(self.clock.now())?,
                ..AudioUtterance::default()
            }))
            .await?;
        Ok(())
    }

    pub async fn playback(&mut self) -> Result<Vec<PlayAudioChunk>> {
        let mut chunks = Vec::new();
        loop {
            let chunk = receive(&mut self.played).await?;
            let final_chunk = chunk.is_final;
            chunks.push(chunk);
            if final_chunk {
                return Ok(chunks);
            }
        }
    }

    pub async fn stop(mut self) -> Result {
        self.brain.shutdown();
        self.shutdown.cancel();
        drop(self.capture);
        tokio::time::timeout(Duration::from_secs(3), async {
            while let Some(job) = self.jobs.join_next().await {
                job??;
            }
            Ok::<_, Box<dyn Error + Send + Sync>>(())
        })
        .await??;
        Ok(())
    }
}

pub async fn receive<T>(rx: &mut mpsc::UnboundedReceiver<T>) -> Result<T> {
    tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await?
        .ok_or_else(|| "test channel closed".into())
}

pub fn actor(role: Role, permissions: &[Permission]) -> ActorContext {
    ActorContext {
        subject: "test-actor".into(),
        role: role.into(),
        permissions: permissions
            .iter()
            .map(|permission| *permission as i32)
            .collect(),
        lease_id: "lease".into(),
        lease_epoch: 1,
        ..ActorContext::default()
    }
}

pub async fn event(
    stream: &mut Streaming<LiveRuntimeEvent>,
    kind: LiveEventKind,
    turn: &str,
) -> Result<LiveRuntimeEvent> {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let event = stream.message().await?.ok_or("event stream ended")?;
            if event.kind() == kind && event.conversation_id == turn {
                return Ok(event);
            }
        }
    })
    .await?
}
