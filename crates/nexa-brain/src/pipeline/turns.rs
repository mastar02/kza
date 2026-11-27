use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use nexa_llm::CancellationToken;
use nexa_proto::v1::{
    self, AudioUtterance, LiveEventKind, SttMode, TranscribeRequest, TranscribeResponse,
    UtteranceKind,
};
use serde_json::json;
use tokio::{
    sync::{OnceCell, mpsc},
    task::JoinSet,
};
use tonic::Status;

use crate::{
    Brain,
    ambient::{AmbientTurn, ClosedSession, SourceKind},
};

const AMBIENT_CONTROL_TIMEOUT: Duration = Duration::from_secs(1);
const AMBIENT_ROOM_QUEUE_CAPACITY: usize = 32;
const DISTILL_QUEUE_CAPACITY: usize = 32;
const AMBIENT_IDLE_SWEEP_INTERVAL: Duration = Duration::from_secs(5);
const MAX_OBSERVATION_PREVIEW_BYTES: usize = 4096;
const VOICE_ID_TIMEOUT: Duration = Duration::from_millis(200);

fn truncate_utf8_bytes(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_owned();
    }
    let mut end = 0;
    for (idx, ch) in text.char_indices() {
        let next = idx + ch.len_utf8();
        if next > max_bytes {
            break;
        }
        end = next;
    }
    text[..end].to_owned()
}

fn enqueue_distillation(tx: &mpsc::Sender<ClosedSession>, session: ClosedSession) {
    if let Err(mpsc::error::TrySendError::Full(dropped)) = tx.try_send(session) {
        tracing::warn!(
            conversation_id = %dropped.conversation_id,
            room_id = %dropped.room_id,
            "Ambient distiller queue full; dropping closed session"
        );
    }
}

async fn ambient_control(
    memory: &dyn crate::clients::MemoryClient,
) -> Result<v1::AmbientControlStatus, Status> {
    tokio::time::timeout(AMBIENT_CONTROL_TIMEOUT, memory.ambient_control())
        .await
        .map_err(|_| Status::deadline_exceeded("ambient control lookup timed out"))?
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
        let mut ambient_rooms: HashMap<String, mpsc::Sender<AudioUtterance>> = HashMap::new();
        let (distill_tx, mut distill_rx) = mpsc::channel::<ClosedSession>(DISTILL_QUEUE_CAPACITY);
        let distill_brain = Arc::clone(&self);
        jobs.spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = distill_brain.shutdown.cancelled() => break,
                    maybe = distill_rx.recv() => {
                        let Some(session) = maybe else { break };
                        distill_brain.distill_closed_session(session).await;
                    }
                }
            }
        });
        let ambient_enabled = self.config.core.memory.ambient_enabled && self.memory.is_some();
        while !self.shutdown.is_cancelled() {
            let result = tokio::select! {
                _ = self.shutdown.cancelled() => break,
                result = self.audio.subscribe(ambient_enabled) => result,
            };
            if let Ok(mut stream) = result {
                loop {
                    tokio::select! {
                        biased;
                        _ = self.shutdown.cancelled() => break,
                        Some(_) = jobs.join_next(), if !jobs.is_empty() => {}
                        item = stream.next() => {
                            match item {
                                Some(Ok(utterance)) => self.accept(utterance, &mut jobs, &mut ambient_rooms, &distill_tx, ambient_enabled),
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
        ambient_rooms.clear();
        drop(distill_tx);
        let grace = async { while jobs.join_next().await.is_some() {} };
        if tokio::time::timeout(Duration::from_millis(600), grace)
            .await
            .is_err()
        {
            jobs.shutdown().await;
        }
    }

    fn spawn_ambient_room_worker(
        self: &Arc<Self>,
        room_id: String,
        distill_tx: mpsc::Sender<ClosedSession>,
        jobs: &mut JoinSet<()>,
    ) -> mpsc::Sender<AudioUtterance> {
        let (tx, mut rx) = mpsc::channel::<AudioUtterance>(AMBIENT_ROOM_QUEUE_CAPACITY);
        let brain = Arc::clone(self);
        jobs.spawn(async move {
            let mut sweep = tokio::time::interval(AMBIENT_IDLE_SWEEP_INTERVAL);
            sweep.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            sweep.tick().await;
            loop {
                tokio::select! {
                    biased;
                    _ = brain.shutdown.cancelled() => break,
                    maybe = rx.recv() => {
                        let Some(utterance) = maybe else { break };
                        let started = Instant::now();
                        let result = tokio::select! {
                            biased;
                            _ = brain.shutdown.cancelled() => Ok(Ok("cancelled")),
                            result = tokio::time::timeout(Duration::from_secs(15), brain.ambient(&utterance, &distill_tx)) => result,
                        };
                        let utterance_id = &utterance.utterance_id;
                        let latency_ms = started.elapsed().as_millis();
                        match result {
                            Ok(Ok(status)) => tracing::info!(%room_id, %utterance_id, latency_ms, status, "Ambient observation"),
                            Ok(Err(error)) => tracing::warn!(%room_id, %utterance_id, latency_ms,
                                status = "failed_closed", code = ?error.code(), "Ambient observation"),
                            Err(_) => tracing::warn!(%room_id, %utterance_id, latency_ms,
                                status = "timed_out", code = ?tonic::Code::DeadlineExceeded, "Ambient observation"),
                        }
                    }
                    _ = sweep.tick() => {
                        if let Some(closed) = brain.ambient.take_idle(&room_id, brain.now()) {
                            enqueue_distillation(&distill_tx, closed);
                        }
                    }
                }
            }
        });
        tx
    }

    fn accept(
        self: &Arc<Self>,
        utterance: AudioUtterance,
        jobs: &mut JoinSet<()>,
        ambient_rooms: &mut HashMap<String, mpsc::Sender<AudioUtterance>>,
        distill_tx: &mpsc::Sender<ClosedSession>,
        ambient_enabled: bool,
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
            if !ambient_enabled {
                return;
            }
            let room_id = utterance.room_id.clone();
            let sender = ambient_rooms.entry(room_id.clone()).or_insert_with(|| {
                self.spawn_ambient_room_worker(room_id.clone(), distill_tx.clone(), jobs)
            });
            match sender.try_send(utterance) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Closed(returned)) => {
                    let fresh =
                        self.spawn_ambient_room_worker(room_id.clone(), distill_tx.clone(), jobs);
                    if fresh.try_send(returned).is_err() {
                        tracing::warn!(%room_id, "ambient room worker queue unavailable");
                    }
                    ambient_rooms.insert(room_id, fresh);
                }
                Err(mpsc::error::TrySendError::Full(dropped)) => {
                    tracing::warn!(
                        room_id = %dropped.room_id,
                        utterance_id = %dropped.utterance_id,
                        "Ambient room worker queue full; dropping utterance"
                    );
                }
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
            });
        }
    }

    pub(super) async fn transcribe(
        &self,
        utterance: &AudioUtterance,
        mode: SttMode,
    ) -> Result<TranscribeResponse, Status> {
        let (beam_size, initial_prompt, hotwords) = if mode == SttMode::Command {
            (
                crate::COMMAND_STT_BEAM_SIZE,
                crate::COMMAND_STT_INITIAL_PROMPT.to_owned(),
                crate::COMMAND_STT_HOTWORDS.to_owned(),
            )
        } else {
            (0, String::new(), String::new())
        };
        let response = self
            .ml
            .transcribe(TranscribeRequest {
                pcm_f32le: utterance.pcm_f32le.clone(),
                sample_rate: utterance.sample_rate,
                language: self.config.core.nlu.language.clone(),
                beam_size,
                initial_prompt,
                hotwords,
                mode: mode.into(),
            })
            .await?;
        if response.text.len() > 4096 {
            return Err(Status::resource_exhausted("transcript too long"));
        }
        Ok(response)
    }

    fn room_media_playing(&self, room_id: &str) -> bool {
        self.catalog().entities.into_iter().any(|entity| {
            entity.room_id == room_id
                && entity.domain == "media_player"
                && entity.state == "playing"
        })
    }

    async fn ambient(
        &self,
        utterance: &AudioUtterance,
        distill_tx: &mpsc::Sender<ClosedSession>,
    ) -> Result<&'static str, Status> {
        let Some(memory) = &self.memory else {
            return Ok("memory_disabled");
        };
        let before = ambient_control(memory.as_ref()).await?;
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
        let after = ambient_control(memory.as_ref()).await?;
        if after.paused || after.revision != before.revision {
            return Ok("control_changed");
        }
        let media_playing = self.room_media_playing(&utterance.room_id);
        match self.ambient.classify(utterance, &transcript, media_playing) {
            SourceKind::Noise => return Ok("filtered_noise"),
            SourceKind::MediaTv {
                confirmed_by_ha: true,
            } => return Ok("filtered_media_tv"),
            SourceKind::MediaTv {
                confirmed_by_ha: false,
            } => {
                let identified = tokio::time::timeout(
                    VOICE_ID_TIMEOUT,
                    memory.identify_voice(v1::IdentifyVoiceRequest {
                        embedding: embedding.vector.clone(),
                        encoder_model_key: embedding.encoder_model_key.clone(),
                        room_id: utterance.room_id.clone(),
                        actor: Some(v1::ActorContext {
                            subject: "nexa-brain".into(),
                            role: v1::Role::Operator as i32,
                            permissions: vec![v1::Permission::VoicesRead as i32],
                            ..Default::default()
                        }),
                    }),
                )
                .await;
                if !matches!(identified, Ok(Ok(ref res)) if res.consented && !res.profile_id.is_empty())
                {
                    return Ok("filtered_media_tv");
                }
            }
            SourceKind::Live => {}
        }
        let (conversation_id, idle_closed) = self
            .ambient
            .prepare_ambient(&utterance.room_id, utterance.captured_at_unix_ms);
        if let Some(closed) = idle_closed {
            enqueue_distillation(distill_tx, closed);
        }
        // The observation receipt is acoustic metadata. It does not enter command admission.
        let receipt = memory
            .observe_voice(v1::RecordVoiceObservationRequest {
                utterance_id: utterance.utterance_id.clone(),
                room_id: utterance.room_id.clone(),
                embedding: embedding.vector,
                encoder_model_key: embedding.encoder_model_key,
                transcript_preview: truncate_utf8_bytes(
                    transcript.text.trim(),
                    MAX_OBSERVATION_PREVIEW_BYTES,
                ),
                captured_at_unix_ms: utterance.captured_at_unix_ms,
                conversation_id: conversation_id.clone(),
            })
            .await?;
        if let Some(window_closed) = self.ambient.finish_ambient(
            &utterance.room_id,
            &conversation_id,
            AmbientTurn {
                text: transcript.text,
                matched_profile_id: (!receipt.matched_profile_id.is_empty())
                    .then_some(receipt.matched_profile_id),
                captured_at_unix_ms: utterance.captured_at_unix_ms,
                continues_segment: utterance.continues_segment,
            },
        ) {
            enqueue_distillation(distill_tx, window_closed);
        }
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
        let mut ambient_rooms = HashMap::new();
        let (distill_tx, _distill_rx) = mpsc::channel(1);
        let mut jobs = JoinSet::new();
        let capacity = brain.config.rooms.rooms.len() * 4;
        for _ in 0..capacity {
            jobs.spawn(std::future::pending());
        }
        brain.accept(
            utterance.clone(),
            &mut jobs,
            &mut ambient_rooms,
            &distill_tx,
            false,
        );
        assert_eq!(jobs.len(), capacity);
        assert!(!active.lease.cancel.is_cancelled());
        assert!(brain.active.busy(&utterance.room_id));

        jobs.shutdown().await;
        brain.accept(utterance, &mut jobs, &mut ambient_rooms, &distill_tx, false);
        assert_eq!(jobs.len(), 1);
        assert!(active.lease.cancel.is_cancelled());
        jobs.shutdown().await;
        Ok(())
    }

    #[test]
    fn truncate_utf8_bytes_preserves_multibyte_boundaries() {
        let sample = "Configuración y canción con ñ y ¿¡acentos!";
        let truncated = truncate_utf8_bytes(sample, 10);
        assert!(truncated.len() <= 10);
        assert!(sample.starts_with(&truncated));
        assert_eq!(truncate_utf8_bytes("á", 1), "");
        assert_eq!(truncate_utf8_bytes("á", 2), "á");
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
