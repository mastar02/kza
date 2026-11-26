use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use nexa_llm::CancellationToken;
use nexa_proto::v1::{
    self, AudioUtterance, LiveEventKind, SttMode, TranscribeRequest, TranscribeResponse,
    UtteranceKind, memory_service_client::MemoryServiceClient,
};
use prost::Message;
use serde_json::json;
use tokio::{sync::OnceCell, task::JoinSet};
use tonic::{Request, Status, metadata::MetadataValue, transport::Channel};

use crate::Brain;

const AMBIENT_CONTROL_TIMEOUT: Duration = Duration::from_secs(1);
const COMMAND_STT_HOTWORDS: &str =
    "Nexa prendé apagá subí bajá poné escritorio living cocina baño hall";

async fn ambient_control(
    client: &mut MemoryServiceClient<Channel>,
) -> Result<v1::AmbientControlStatus, Status> {
    let actor = v1::ActorContext {
        subject: "nexa-brain".into(),
        role: v1::Role::Operator.into(),
        permissions: vec![v1::Permission::AmbientControl.into()],
        ..Default::default()
    };
    let mut request = Request::new(v1::Empty {});
    request.metadata_mut().insert_bin(
        "actor-context-bin",
        MetadataValue::from_bytes(&actor.encode_to_vec()),
    );
    tokio::time::timeout(AMBIENT_CONTROL_TIMEOUT, client.get_ambient_control(request))
        .await
        .map_err(|_| Status::deadline_exceeded("ambient control lookup timed out"))?
        .map(tonic::Response::into_inner)
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ActiveTurns {
    zones: Arc<Mutex<HashMap<String, Lease>>>,
    followups: Arc<Mutex<HashMap<String, FollowUpWindow>>>,
}

#[derive(Debug)]
struct FollowUpWindow {
    capture_epoch: u64,
    expires: Duration,
}

#[derive(Debug, Clone)]
pub(super) struct Lease {
    pub(super) id: String,
    pub(super) cancel: CancellationToken,
    pub(super) cancel_rpc: Arc<OnceCell<()>>,
}

#[derive(Debug)]
pub(crate) struct Turn {
    active: ActiveTurns,
    pub(crate) zone: String,
    pub(super) lease: Lease,
}

impl ActiveTurns {
    pub(super) fn start(
        &self,
        zone: &str,
        parent: &CancellationToken,
        replace: bool,
    ) -> Option<(Turn, Option<Lease>)> {
        let mut zones = self
            .zones
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !replace && zones.contains_key(zone) {
            return None;
        }
        let lease = Lease {
            id: crate::id("play"),
            cancel: parent.child_token(),
            cancel_rpc: Arc::default(),
        };
        let previous = zones.insert(zone.to_owned(), lease.clone());
        if let Some(old) = &previous {
            old.cancel.cancel();
        }
        Some((
            Turn {
                active: self.clone(),
                zone: zone.to_owned(),
                lease,
            },
            previous,
        ))
    }

    pub(super) fn open_followup(&self, room: &str, epoch: u64, expires: Duration) {
        self.followups
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                room.to_owned(),
                FollowUpWindow {
                    capture_epoch: epoch,
                    expires,
                },
            );
    }

    pub(super) fn can_follow_up(&self, room: &str, epoch: u64, now: Duration) -> bool {
        self.followups
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(room)
            .is_some_and(|window| window.capture_epoch == epoch && now < window.expires)
    }

    pub(super) fn busy(&self, zone: &str) -> bool {
        self.zones
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains_key(zone)
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        let mut zones = self
            .active
            .zones
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if zones
            .get(&self.zone)
            .is_some_and(|lease| lease.id == self.lease.id)
        {
            zones.remove(&self.zone);
        }
    }
}

impl Brain {
    /// Subscribe and reconnect until shutdown. Capture work remains bounded by room count.
    pub async fn run_voice(self: Arc<Self>) {
        let mut jobs = JoinSet::new();
        let mut ambient_jobs = HashSet::new();
        let ambient_control = (self.config.core.memory.ambient_enabled && self.memory.is_some())
            .then(|| {
                // Match the daemon's runtime directory resolution for the private memory socket.
                let runtime = std::env::var_os("RUNTIME_DIRECTORY")
                    .map(PathBuf::from)
                    .or_else(|| {
                        std::env::var_os("XDG_RUNTIME_DIR").map(|dir| Path::new(&dir).join("nexa"))
                    })
                    .unwrap_or_else(|| self.config.core.paths.runtime_dir.clone());
                MemoryServiceClient::new(crate::clients::channel(
                    &runtime.join(nexa_proto::sockets::MEMORY),
                ))
            });
        while !self.shutdown.is_cancelled() {
            let result = tokio::select! {
                _ = self.shutdown.cancelled() => break,
                result = self.audio.subscribe(ambient_control.is_some()) => result,
            };
            if let Ok(mut stream) = result {
                loop {
                    tokio::select! {
                        biased;
                        _ = self.shutdown.cancelled() => break,
                        Some(done) = jobs.join_next(), if !jobs.is_empty() => {
                            if let Ok(Some(room)) = done { ambient_jobs.remove(&room); }
                        }
                        item = stream.next() => {
                            match item {
                                Some(Ok(utterance)) => self.accept(utterance, &mut jobs, &mut ambient_jobs, ambient_control.as_ref()),
                                _ => break,
                            }
                        }
                    }
                }
            }
            tokio::select! {
                _ = self.shutdown.cancelled() => break,
                _ = tokio::time::sleep(Duration::from_millis(250)) => {}
            }
        }
        let grace = async { while jobs.join_next().await.is_some() {} };
        if tokio::time::timeout(Duration::from_millis(600), grace)
            .await
            .is_err()
        {
            jobs.shutdown().await;
        }
    }

    fn accept(
        self: &Arc<Self>,
        utterance: AudioUtterance,
        jobs: &mut JoinSet<Option<String>>,
        ambient_jobs: &mut HashSet<String>,
        ambient_control: Option<&MemoryServiceClient<Channel>>,
    ) {
        if !self.zone(&utterance.room_id)
            || utterance.utterance_id.is_empty()
            || utterance.utterance_id.len() > 128
            || utterance.sample_rate != self.config.core.audio.capture_sample_rate
            || utterance.pcm_f32le.is_empty()
            || !utterance.pcm_f32le.len().is_multiple_of(4)
            || utterance.pcm_f32le.len() > 4 * 1024 * 1024
        {
            return;
        }
        let brain = Arc::clone(self);
        if utterance.kind() == UtteranceKind::Ambient {
            let Some(control) = ambient_control.cloned() else {
                return;
            };
            if ambient_jobs.insert(utterance.room_id.clone()) {
                jobs.spawn(async move {
                    let started = Instant::now();
                    let result = tokio::select! {
                        biased;
                        _ = brain.shutdown.cancelled() => Ok(Ok("cancelled")),
                        result = tokio::time::timeout(Duration::from_secs(15), brain.ambient(&utterance, control)) => result,
                    };
                    let room_id = &utterance.room_id;
                    let utterance_id = &utterance.utterance_id;
                    let latency_ms = started.elapsed().as_millis();
                    match result {
                        Ok(Ok(status)) => tracing::info!(%room_id, %utterance_id, latency_ms, status, "Ambient observation"),
                        Ok(Err(error)) => tracing::warn!(%room_id, %utterance_id, latency_ms,
                            status = "failed_closed", code = ?error.code(), "Ambient observation"),
                        Err(_) => tracing::warn!(%room_id, %utterance_id, latency_ms,
                            status = "timed_out", code = ?tonic::Code::DeadlineExceeded, "Ambient observation"),
                    }
                    Some(utterance.room_id)
                });
            }
        } else if matches!(
            utterance.kind(),
            UtteranceKind::Command | UtteranceKind::FollowUp
        ) {
            if utterance.kind() == UtteranceKind::FollowUp
                && !self.active.can_follow_up(
                    &utterance.room_id,
                    utterance.capture_epoch,
                    self.clock.monotonic(),
                )
            {
                return;
            }
            if jobs.len() >= self.config.rooms.rooms.len().saturating_mul(4) {
                return;
            }
            let Some((turn, previous)) =
                self.active.start(&utterance.room_id, &self.shutdown, true)
            else {
                return;
            };
            jobs.spawn(async move {
                brain.command(utterance, turn, previous).await;
                None
            });
        }
    }

    pub(super) async fn transcribe(
        &self,
        utterance: &AudioUtterance,
        mode: SttMode,
    ) -> Result<TranscribeResponse, Status> {
        let hotwords = if mode == SttMode::Command {
            COMMAND_STT_HOTWORDS.to_owned()
        } else {
            String::new()
        };
        let response = self
            .ml
            .transcribe(TranscribeRequest {
                pcm_f32le: utterance.pcm_f32le.clone(),
                sample_rate: utterance.sample_rate,
                language: self.config.core.nlu.language.clone(),
                mode: mode.into(),
                hotwords,
                ..TranscribeRequest::default()
            })
            .await?;
        if response.text.len() > 4096 {
            return Err(Status::resource_exhausted("transcript too long"));
        }
        Ok(response)
    }

    async fn ambient(
        &self,
        utterance: &AudioUtterance,
        mut control: MemoryServiceClient<Channel>,
    ) -> Result<&'static str, Status> {
        let Some(memory) = &self.memory else {
            return Ok("memory_disabled");
        };
        let before = ambient_control(&mut control).await?;
        if before.paused {
            return Ok("paused");
        }
        let transcript = self.transcribe(utterance, SttMode::Ambient);
        let embedding = self.ml.embed_speaker(v1::EmbedSpeakerRequest {
            pcm_f32le: utterance.pcm_f32le.clone(),
            sample_rate: utterance.sample_rate,
        });
        let (transcript, embedding) = tokio::try_join!(transcript, embedding)?;
        if embedding.vector.len() != 192
            || !embedding.vector.iter().all(|value| value.is_finite())
            || embedding.encoder_model_key.trim().is_empty()
        {
            return Err(Status::data_loss("invalid speaker embedding"));
        }
        let after = ambient_control(&mut control).await?;
        if after.paused || after.revision != before.revision {
            return Ok("control_changed");
        }
        // The observation receipt is acoustic metadata. It does not enter command admission.
        memory
            .observe_voice(v1::RecordVoiceObservationRequest {
                utterance_id: utterance.utterance_id.clone(),
                room_id: utterance.room_id.clone(),
                embedding: embedding.vector,
                encoder_model_key: embedding.encoder_model_key,
                transcript_preview: transcript.text.chars().take(160).collect(),
                captured_at_unix_ms: utterance.captured_at_unix_ms,
            })
            .await?;
        Ok("observed")
    }

    async fn command(&self, utterance: AudioUtterance, turn: Turn, previous: Option<Lease>) {
        let room = &utterance.room_id;
        if let Some(previous) = previous {
            self.cancel_playback(room, &previous).await;
        }
        let cancel = &turn.lease.cancel;
        let result = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                self.cancel_playback(room, &turn.lease).await;
                tracing::info!(room_id = %room, utterance_id = %utterance.utterance_id,
                    code = ?tonic::Code::Cancelled, "Voice turn cancelled");
                self.event(LiveEventKind::Tts, room, &utterance.utterance_id, json!({"status": "cancelled"}));
                return;
            }
            result = tokio::time::timeout(Duration::from_secs(75), self.route_and_respond(&utterance, &turn)) => result,
        };
        let code = match result {
            Ok(Ok(())) => return,
            Ok(Err(error)) => error.code(),
            Err(_) => tonic::Code::DeadlineExceeded,
        };
        tracing::warn!(room_id = %room, utterance_id = %utterance.utterance_id, ?code, "Voice turn failed");
        cancel.cancel();
        self.cancel_playback(room, &turn.lease).await;
        self.event(
            LiveEventKind::Turn,
            room,
            &utterance.utterance_id,
            json!({"status": "failed"}),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BrainError, Dependencies,
        clients::{UdsAudio, UdsMl, channel},
    };
    use nexa_config::{LoadedBundle, NexaConfig, RoomsConfig};
    use nexa_household::SystemClock;
    use nexa_proto::v1::{
        audio_service_client::AudioServiceClient, ml_service_client::MlServiceClient,
    };

    #[tokio::test]
    async fn saturated_command_queue_keeps_the_active_turn() -> Result<(), BrainError> {
        let dir = tempfile::tempdir_in("/tmp")?;
        let mut core =
            NexaConfig::from_toml_str(include_str!("../../../../config/nexa/nexa.toml"))?;
        core.paths.data_dir = dir.path().join("data");
        let utterance = AudioUtterance {
            room_id: "living".into(),
            utterance_id: "incoming".into(),
            sample_rate: core.audio.capture_sample_rate,
            pcm_f32le: vec![0; 4],
            kind: UtteranceKind::Command.into(),
            ..Default::default()
        };
        let config = LoadedBundle {
            core,
            rooms: RoomsConfig::load_file(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../config/nexa/rooms.example.toml"
            ))?,
            ha_controls: Default::default(),
        };
        let channel = channel(&dir.path().join("unused.sock"));
        let brain = Brain::open(
            config,
            Dependencies {
                audio: Arc::new(UdsAudio(AudioServiceClient::new(channel.clone()))),
                ml: Arc::new(UdsMl(MlServiceClient::new(channel))),
                memory: None,
                home: None,
                clock: Arc::new(SystemClock::default()),
            },
        )
        .await?;
        let (active, _) = brain
            .active
            .start(&utterance.room_id, &brain.shutdown, false)
            .ok_or("could not start active turn")?;
        let mut ambient_jobs = HashSet::new();
        let mut jobs = JoinSet::new();
        let capacity = brain.config.rooms.rooms.len() * 4;
        for _ in 0..capacity {
            jobs.spawn(std::future::pending());
        }
        brain.accept(utterance.clone(), &mut jobs, &mut ambient_jobs, None);
        assert_eq!(jobs.len(), capacity);
        assert!(!active.lease.cancel.is_cancelled());
        assert!(brain.active.busy(&utterance.room_id));

        jobs.shutdown().await;
        brain.accept(utterance, &mut jobs, &mut ambient_jobs, None);
        assert_eq!(jobs.len(), 1);
        assert!(active.lease.cancel.is_cancelled());
        jobs.shutdown().await;
        Ok(())
    }

    #[tokio::test]
    async fn whisper_confidence_runs_in_shadow_while_compression_ratio_enforces()
    -> Result<(), BrainError> {
        use nexa_nlu::{
            gate::{GateContext, GateReason, SttConfidence},
            router_engine::{Route, RouteContext},
        };

        let dir = tempfile::tempdir_in("/tmp")?;
        let mut core =
            NexaConfig::from_toml_str(include_str!("../../../../config/nexa/nexa.toml"))?;
        core.paths.data_dir = dir.path().join("data");
        let config = LoadedBundle {
            core,
            rooms: RoomsConfig::load_file(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../config/nexa/rooms.example.toml"
            ))?,
            ha_controls: Default::default(),
        };
        let channel = channel(&dir.path().join("unused.sock"));
        let brain = Brain::open(
            config,
            Dependencies {
                audio: Arc::new(UdsAudio(AudioServiceClient::new(channel.clone()))),
                ml: Arc::new(UdsMl(MlServiceClient::new(channel))),
                memory: None,
                home: None,
                clock: Arc::new(SystemClock::default()),
            },
        )
        .await?;

        let low_conf = brain.router.route(
            "Nexa, prendé la luz",
            RouteContext {
                gate: GateContext {
                    confidence: SttConfidence {
                        no_speech_prob: Some(0.85),
                        avg_logprob: Some(-1.8),
                        compression_ratio: Some(1.1),
                    },
                    ..GateContext::default()
                },
                acoustic_wake: true,
                source_room: Some("escritorio"),
                wake_score: 0.54,
                rms: 0.012,
                ..RouteContext::default()
            },
        );
        assert!(low_conf.acceptance.accept);
        assert!(matches!(low_conf.route, Route::Domotics(_)));
        assert_eq!(
            low_conf.acceptance.would_reject,
            vec![GateReason::LowConfidence]
        );

        let high_comp = brain.router.route(
            "Nexa, prendé la luz",
            RouteContext {
                gate: GateContext {
                    confidence: SttConfidence {
                        no_speech_prob: Some(0.0),
                        avg_logprob: Some(-0.2),
                        compression_ratio: Some(2.5),
                    },
                    ..GateContext::default()
                },
                acoustic_wake: true,
                source_room: Some("escritorio"),
                wake_score: 0.54,
                rms: 0.012,
                ..RouteContext::default()
            },
        );
        assert!(!high_comp.acceptance.accept);
        assert_eq!(high_comp.acceptance.reason, GateReason::HighCompression);
        assert!(matches!(high_comp.route, Route::Rejected));
        Ok(())
    }
}
