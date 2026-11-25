//! Voice turns, per-zone cancellation, and streaming playback.

#![allow(
    clippy::result_large_err,
    reason = "Tonic status carries gRPC metadata and codes across this boundary"
)]

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use futures_util::StreamExt;
use nexa_llm::CancellationToken;
use nexa_nlu::{
    gate::{GateContext, SttConfidence},
    normalize::words,
    router_engine::{Route, RouteContext},
    world,
};
use nexa_proto::v1::{
    self, AudioUtterance, CancelPlaybackRequest, LiveEventKind, PlayAudioChunk, SttMode,
    SynthesizeRequest, TranscribeRequest, TranscribeResponse, UtteranceKind,
};
use serde_json::json;
use tokio::{
    sync::{OnceCell, mpsc},
    task::JoinSet,
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;

use crate::{Brain, clients::StreamOf};

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
struct Lease {
    id: String,
    cancel: CancellationToken,
    cancel_rpc: Arc<OnceCell<()>>,
}

#[derive(Debug)]
pub(crate) struct Turn {
    active: ActiveTurns,
    pub(crate) zone: String,
    lease: Lease,
}

impl ActiveTurns {
    fn start(
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

    fn open_followup(&self, room: &str, epoch: u64, expires: Duration) {
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

    fn can_follow_up(&self, room: &str, epoch: u64, now: Duration) -> bool {
        self.followups
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(room)
            .is_some_and(|window| window.capture_epoch == epoch && now < window.expires)
    }

    fn busy(&self, zone: &str) -> bool {
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
        while !self.shutdown.is_cancelled() {
            let ambient = self.config.core.memory.ambient_enabled && self.memory.is_some();
            let result = tokio::select! {
                _ = self.shutdown.cancelled() => break,
                result = self.audio.subscribe(ambient) => result,
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
                                Some(Ok(utterance)) => self.accept(utterance, &mut jobs, &mut ambient_jobs),
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
            if self.config.core.memory.ambient_enabled
                && self.memory.is_some()
                && ambient_jobs.insert(utterance.room_id.clone())
            {
                jobs.spawn(async move {
                    tokio::select! {
                        _ = brain.shutdown.cancelled() => {}
                        result = tokio::time::timeout(Duration::from_secs(15), brain.ambient(&utterance)) => {
                            if !matches!(result, Ok(Ok(()))) {
                                tracing::warn!("ambient observation failed");
                            }
                        }
                    }
                    Some(utterance.room_id)
                });
            }
        } else if matches!(
            utterance.kind(),
            UtteranceKind::Command | UtteranceKind::FollowUp
        ) {
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

    async fn transcribe(
        &self,
        utterance: &AudioUtterance,
        mode: SttMode,
    ) -> Result<TranscribeResponse, Status> {
        let response = self
            .ml
            .transcribe(TranscribeRequest {
                pcm_f32le: utterance.pcm_f32le.clone(),
                sample_rate: utterance.sample_rate,
                language: self.config.core.nlu.language.clone(),
                mode: mode.into(),
                ..TranscribeRequest::default()
            })
            .await?;
        if response.text.len() > 4096 {
            return Err(Status::resource_exhausted("transcript too long"));
        }
        Ok(response)
    }

    async fn ambient(&self, utterance: &AudioUtterance) -> Result<(), Status> {
        let Some(memory) = &self.memory else {
            return Ok(());
        };
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
        Ok(())
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
                self.event(LiveEventKind::Tts, room, &utterance.utterance_id, json!({"status": "cancelled"}));
                return;
            }
            result = tokio::time::timeout(Duration::from_secs(75), self.route_and_respond(&utterance, &turn)) => result,
        };
        match result {
            Ok(Ok(())) => {}
            _ => {
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
    }

    async fn cancel_playback(&self, zone: &str, lease: &Lease) {
        lease
            .cancel_rpc
            .get_or_init(|| async {
                let cancelled = tokio::time::timeout(
                    Duration::from_millis(350),
                    self.audio.cancel(CancelPlaybackRequest {
                        zone_id: self.output_zone(zone),
                        barge_in_token: lease.id.clone(),
                    }),
                )
                .await;
                if !matches!(cancelled, Ok(Ok(()))) {
                    tracing::warn!(zone, "playback cancellation was not acknowledged");
                }
            })
            .await;
    }

    async fn route_and_respond(
        &self,
        utterance: &AudioUtterance,
        turn: &Turn,
    ) -> Result<(), Status> {
        let room = &utterance.room_id;
        let conversation = &utterance.utterance_id;
        let wake = utterance.kind() == UtteranceKind::Command
            && utterance.wake_score.is_finite()
            && utterance.wake_score >= self.config.core.audio.wake_threshold;
        let monotonic = self.clock.monotonic();
        let follow_up = utterance.kind() == UtteranceKind::FollowUp
            && self
                .active
                .can_follow_up(room, utterance.capture_epoch, monotonic);
        if wake {
            self.event(
                LiveEventKind::Wake,
                room,
                conversation,
                json!({"event": "acoustic_wake"}),
            );
        }
        let stt = self.transcribe(utterance, SttMode::Command).await?;
        let valid_confidence = [stt.no_speech_prob, stt.avg_logprob, stt.compression_ratio]
            .iter()
            .all(|value| value.is_finite());
        let ring_room = room.to_owned();
        let ring_open = self
            .household
            .call(move |db| Ok(!db.ringing_timers(&ring_room)?.is_empty()))
            .await
            .map_err(crate::tasks::household_status)?;
        let routed = self.router.route(
            &stt.text,
            RouteContext {
                gate: GateContext {
                    confidence: SttConfidence {
                        no_speech_prob: Some(stt.no_speech_prob),
                        avg_logprob: Some(stt.avg_logprob),
                        compression_ratio: Some(stt.compression_ratio),
                    },
                    stt_vetoed: !valid_confidence,
                    ring_stop_open: ring_open,
                    ..GateContext::default()
                },
                acoustic_wake: wake,
                source_room: Some(room),
                wake_score: utterance.wake_score,
                rms: utterance.rms,
                ..RouteContext::default()
            },
        );
        let shadow = words(&utterance.parakeet_shadow_text);
        let corroborated = !shadow.is_empty() && shadow == words(&stt.text);
        let admitted = (wake || follow_up || routed.acceptance.ring_stop)
            && (!(routed.requires_veto || follow_up) || corroborated);
        self.event(
            LiveEventKind::Turn,
            room,
            conversation,
            json!({
                "path": routed.decision.path.as_str_name(),
                "reason": routed.decision.reason,
                "admitted": admitted && !matches!(routed.route, Route::Rejected),
            }),
        );
        if !admitted {
            if !wake && !follow_up {
                return Ok(());
            }
            return self
                .say(turn, conversation, "No entendí con seguridad.")
                .await
                .map(|_| ());
        }
        if wake && !matches!(routed.route, Route::Rejected | Route::RefusalOnly) {
            let window =
                Duration::from_secs(self.config.core.household.voice_choice_window_s.into());
            self.active.open_followup(
                room,
                utterance.capture_epoch,
                monotonic.saturating_add(window),
            );
        }
        let reply = match routed.route {
            Route::Rejected if routed.decision.reason.as_deref() == Some("security_order") => {
                crate::tasks::SECURITY_REFUSAL.into()
            }
            Route::Rejected if routed.earcon_eligible => "No entendí.".into(),
            Route::Rejected => return Ok(()),
            Route::RefusalOnly => "No puedo ejecutar esa acción por voz.".into(),
            Route::Domotics(command) => self.voice_action(&command, room).await,
            Route::Compound(intent) => {
                let mut answers = Vec::new();
                for part in intent.parts {
                    if part.security_refusal {
                        answers.push(crate::tasks::SECURITY_REFUSAL.into());
                    } else if let Some(command) = part.command {
                        answers.push(self.voice_action(&command, room).await);
                    } else if let Some(intent) = part.intent {
                        answers.push(
                            self.voice_household(&intent, &part.segment, turn, conversation)
                                .await?,
                        );
                    }
                }
                answers.join(" ")
            }
            Route::Household(intent) => {
                self.voice_household(&intent, &stt.text, turn, conversation)
                    .await?
            }
            Route::Media(_) => "Ese control no está disponible.".into(),
            Route::Time(query) => world::describe_clock(query, &self.local_now()),
            Route::Weather(horizon) => self.weather(horizon).await,
            Route::SlowLlm => {
                return self
                    .slow_llm(turn, conversation, &routed.decision.normalized_text)
                    .await;
            }
        };
        self.say(turn, conversation, reply).await.map(|_| ())
    }

    async fn voice_action(&self, command: &nexa_nlu::grammar::HomeCommand, room: &str) -> String {
        let Some(action) = self.resolve_voice_action(command, room) else {
            return "¿A qué dispositivo te referís?".into();
        };
        let result = self.execute_home(crate::id("voice"), action).await;
        match result.as_ref().map(|value| value.status.as_str()) {
            Ok("accepted") => "Mandé la orden.".into(),
            Ok("uncertain" | "partial") => "No pude confirmar la orden.".into(),
            _ => "No pude enviar la orden.".into(),
        }
    }

    async fn memory_context(&self, text: &str) -> Result<Vec<String>, Status> {
        let Some(memory) = &self.memory else {
            return Ok(Vec::new());
        };
        if !self.config.core.memory.enabled {
            return Ok(Vec::new());
        }
        let response = self
            .ml
            .embed_text(v1::EmbedTextRequest {
                texts: vec![text.into()],
            })
            .await?;
        let mut vectors = response.vectors.into_iter();
        let Some(vector) = vectors.next() else {
            return Err(Status::data_loss("missing query embedding"));
        };
        if vector.values.is_empty()
            || vector.values.len() > 4096
            || !vector.values.iter().all(|value| value.is_finite())
        {
            return Err(Status::data_loss("invalid query embedding"));
        }
        let result = memory
            .query(v1::QueryMemoryRequest {
                query_embedding: vector.values,
                top_k: 4,
                include_shared: true,
                ..v1::QueryMemoryRequest::default()
            })
            .await?;
        Ok(result
            .hits
            .into_iter()
            .take(4)
            .filter_map(|hit| hit.fact)
            .filter(|fact| fact.owner_user_id.is_empty() && fact.confirmed)
            .map(|fact| fact.content.chars().take(256).collect())
            .collect())
    }

    async fn slow_llm(&self, turn: &Turn, conversation: &str, text: &str) -> Result<(), Status> {
        let context =
            match tokio::time::timeout(Duration::from_millis(350), self.memory_context(text)).await
            {
                Ok(Ok(context)) => context,
                Ok(Err(_)) | Err(_) => {
                    tracing::warn!("shared memory lookup unavailable");
                    Vec::new()
                }
            };
        let output_text = if context.is_empty() {
            text.to_owned()
        } else {
            // Memory context can go only to the local model. Only the original
            // user input can be retried across the cloud consent boundary.
            format!(
                "Contexto local de la casa: {}\nConsulta: {text}",
                json!(context)
            )
        };
        let (tx, rx) = mpsc::channel(64);
        let cancel = turn.lease.cancel.child_token();
        let generate = async {
            let mut segment = String::new();
            let mut truncated = false;
            let mut started = false;
            let mut length = 0usize;
            let mut emit =
                |chunk: &str, segment: &mut String, started: &mut bool, truncated: &mut bool| {
                    if *truncated {
                        return;
                    }
                    length = length.saturating_add(chunk.len());
                    if length > 32 * 1024 {
                        *truncated = true;
                        cancel.cancel();
                        return;
                    }
                    for character in chunk.chars() {
                        segment.push(character);
                        if (matches!(character, '.' | '!' | '?' | '\n') && segment.len() > 24)
                            || segment.len() >= 256 && character.is_whitespace()
                            || segment.len() >= 1024
                        {
                            let value = std::mem::take(segment);
                            if tx.try_send(value).is_err() {
                                *truncated = true;
                                cancel.cancel();
                                return;
                            }
                            *started = true;
                        }
                    }
                };
            let mut result = if context.is_empty() {
                self.llm
                    .stream(text, 256, &cancel, |chunk| {
                        emit(chunk, &mut segment, &mut started, &mut truncated)
                    })
                    .await
                    .map(|_| ())
            } else {
                self.stream_local_with_context(&output_text, &cancel, |chunk| {
                    emit(chunk, &mut segment, &mut started, &mut truncated)
                })
                .await
            };
            if result.is_err()
                && !context.is_empty()
                && !started
                && segment.is_empty()
                && !cancel.is_cancelled()
            {
                result = self
                    .llm
                    .stream(text, 256, &cancel, |chunk| {
                        emit(chunk, &mut segment, &mut started, &mut truncated)
                    })
                    .await
                    .map(|_| ());
            }
            if !segment.trim().is_empty() && !truncated {
                tx.send(segment)
                    .await
                    .map_err(|_| Status::cancelled("speech stopped"))?;
            } else if !started && result.is_err() && !cancel.is_cancelled() {
                tx.send("No pude responder en este momento.".into())
                    .await
                    .map_err(|_| Status::cancelled("speech stopped"))?;
            }
            drop(tx);
            for endpoint in &self.endpoints {
                let remaining = self.llm.cooldown_remaining(&endpoint.id).await;
                if !remaining.is_zero() {
                    self.event(
                        LiveEventKind::Cooldown,
                        "",
                        "",
                        json!({
                            "endpoint_id": endpoint.id, "remaining_ms": remaining.as_millis()
                        }),
                    );
                }
            }
            if truncated {
                return Err(Status::resource_exhausted("speech queue full"));
            }
            Ok(())
        };
        let play = self.speak(turn, conversation, Box::pin(ReceiverStream::new(rx)));
        tokio::try_join!(generate, play)?;
        Ok(())
    }

    async fn stream_local_with_context(
        &self,
        prompt: &str,
        cancel: &CancellationToken,
        emit: impl FnMut(&str),
    ) -> Result<(), nexa_llm::LlmError> {
        let endpoint = &self.endpoints[0];
        if !self.llm.cooldown_remaining(&endpoint.id).await.is_zero() {
            return Err(nexa_llm::LlmError::Unavailable);
        }
        let client = nexa_llm::OpenAiClient::new(Arc::new(nexa_llm::CloudConsentGate::new(false)))?;
        client
            .stream(endpoint, prompt, 256, cancel, emit)
            .await
            .map(|_| ())
    }

    pub(crate) async fn say(
        &self,
        turn: &Turn,
        conversation: &str,
        text: impl Into<String>,
    ) -> Result<v1::PlayAudioReceipt, Status> {
        let text = text.into();
        self.speak(
            turn,
            conversation,
            Box::pin(futures_util::stream::iter([text])),
        )
        .await
    }

    async fn speak(
        &self,
        turn: &Turn,
        conversation: &str,
        mut texts: StreamOf<String>,
    ) -> Result<v1::PlayAudioReceipt, Status> {
        let Some(first) = texts.next().await else {
            return Err(Status::failed_precondition("no speech generated"));
        };
        let (tx, rx) = mpsc::channel(8);
        let sample_rate = self.config.core.audio.playback_sample_rate;
        let zone = &turn.zone;
        let chunk = |pcm, is_final| PlayAudioChunk {
            playback_id: turn.lease.id.clone(),
            zone_id: self.output_zone(zone),
            sample_rate,
            pcm_f32le: pcm,
            priority: 1,
            barge_in_token: turn.lease.id.clone(),
            is_final,
        };
        let produce = async {
            let mut texts = futures_util::stream::iter([first]).chain(texts);
            let mut sent = false;
            while let Some(text) = texts.next().await {
                if text.trim().is_empty() {
                    continue;
                }
                let mut audio = self
                    .ml
                    .synthesize(SynthesizeRequest {
                        text,
                        voice: self.config.core.ml.tts_voice.clone(),
                        lang_code: self.config.core.ml.tts_lang_code.clone(),
                        speed: 1.0,
                    })
                    .await?;
                let mut finished = false;
                while let Some(frame) = audio.next().await {
                    let frame = frame?;
                    if frame.sample_rate != sample_rate || !frame.pcm_f32le.len().is_multiple_of(4)
                    {
                        return Err(Status::data_loss("invalid TTS audio format"));
                    }
                    if !frame.pcm_f32le.is_empty() {
                        tx.send(chunk(frame.pcm_f32le, false))
                            .await
                            .map_err(|_| Status::cancelled("playback stopped"))?;
                        sent = true;
                    }
                    if frame.is_final {
                        finished = true;
                        break;
                    }
                }
                if !finished {
                    return Err(Status::data_loss("incomplete TTS audio"));
                }
            }
            if !sent {
                return Err(Status::data_loss("TTS generated no audio"));
            }
            tx.send(chunk(Vec::new(), true))
                .await
                .map_err(|_| Status::cancelled("playback stopped"))?;
            Ok(())
        };
        self.event(
            LiveEventKind::Tts,
            zone,
            conversation,
            json!({"status": "started"}),
        );
        let play = async {
            let receipt = self.audio.play(ReceiverStream::new(rx)).await?;
            if receipt.playback_id != turn.lease.id {
                return Err(Status::data_loss("playback receipt ID mismatch"));
            }
            if receipt.interrupted_by_barge_in {
                turn.lease.cancel.cancel();
                return Err(Status::cancelled("playback interrupted by barge-in"));
            }
            Ok(receipt)
        };
        let result = tokio::try_join!(produce, play).map(|(_, receipt)| receipt);
        let status = match &result {
            Ok(receipt) if receipt.completed => "completed",
            Err(error) if error.code() == tonic::Code::Cancelled => "cancelled",
            _ => "unconfirmed",
        };
        self.event(
            LiveEventKind::Tts,
            zone,
            conversation,
            json!({"status": status}),
        );
        result
    }

    pub(crate) fn output_busy(&self, zone: &str) -> bool {
        self.active.busy(zone)
    }

    pub(crate) async fn announce(
        &self,
        zone: &str,
        message: &str,
    ) -> nexa_household::intercom::Playback {
        use nexa_household::intercom::Playback;
        if !self.zone(zone) {
            return Playback::NotStarted;
        }
        let Some((turn, _)) = self.active.start(zone, &self.shutdown, false) else {
            return Playback::NotStarted;
        };
        let result = tokio::select! {
            biased;
            _ = turn.lease.cancel.cancelled() => None,
            result = tokio::time::timeout(Duration::from_secs(10), self.say(&turn, "", message)) => result.ok(),
        };
        match result {
            Some(Ok(receipt)) if receipt.completed => Playback::Confirmed,
            _ => {
                self.cancel_playback(zone, &turn.lease).await;
                Playback::Uncertain
            }
        }
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
        let mut core = NexaConfig::from_toml_str(include_str!("../../../config/nexa/nexa.toml"))?;
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
        brain.accept(utterance.clone(), &mut jobs, &mut ambient_jobs);
        assert_eq!(jobs.len(), capacity);
        assert!(!active.lease.cancel.is_cancelled());
        assert!(brain.active.busy(&utterance.room_id));

        jobs.shutdown().await;
        brain.accept(utterance, &mut jobs, &mut ambient_jobs);
        assert_eq!(jobs.len(), 1);
        assert!(active.lease.cancel.is_cancelled());
        jobs.shutdown().await;
        Ok(())
    }
}
