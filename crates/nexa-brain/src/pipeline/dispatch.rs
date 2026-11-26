use std::time::{Duration, Instant};

use nexa_nlu::{
    gate::{GateContext, SttConfidence},
    household::HouseholdIntent,
    normalize::words,
    router_engine::{Route, RouteContext},
    world,
};
use nexa_proto::v1::{AudioUtterance, LiveEventKind, SttMode, UtteranceKind};
use serde_json::json;
use tonic::Status;

use crate::Brain;

use super::Turn;

impl Brain {
    pub(super) async fn route_and_respond(
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
            tracing::info!(room_id = %room, utterance_id = %conversation, wake_score = utterance.wake_score, "Wake received");
            self.event(
                LiveEventKind::Wake,
                room,
                conversation,
                json!({"event": "acoustic_wake"}),
            );
        }
        let stt_started = Instant::now();
        let stt = self.transcribe(utterance, SttMode::Command).await?;
        let valid_confidence = [stt.no_speech_prob, stt.avg_logprob, stt.compression_ratio]
            .iter()
            .all(|value| value.is_finite());
        tracing::info!(
            room_id = %room,
            utterance_id = %conversation,
            latency_ms = stt_started.elapsed().as_millis(),
            word_count = stt.text.split_whitespace().count(),
            vetoed = !valid_confidence,
            "STT ready"
        );
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
        let path = routed.decision.path.as_str_name();
        let admitted_route = admitted && !matches!(routed.route, Route::Rejected);
        tracing::info!(
            room_id = %room,
            utterance_id = %conversation,
            path,
            admitted = admitted_route,
            reason = routed.decision.reason.as_deref().unwrap_or("none"),
            corroborated,
            "NLU route resolved"
        );
        self.event(
            LiveEventKind::Turn,
            room,
            conversation,
            json!({
                "path": path,
                "reason": routed.decision.reason,
                "admitted": admitted_route,
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
            Route::Domotics(command) => self.voice_action(&command, room, conversation).await,
            Route::Compound(intent) => {
                let mut answers = Vec::new();
                for part in intent.parts {
                    if part.security_refusal {
                        answers.push(crate::tasks::SECURITY_REFUSAL.into());
                    } else if let Some(command) = part.command {
                        answers.push(self.voice_action(&command, room, conversation).await);
                    } else if let Some(intent) = part.intent {
                        answers.push(
                            self.dispatch_household(&intent, &part.segment, turn, conversation)
                                .await?,
                        );
                    }
                }
                answers.join(" ")
            }
            Route::Household(intent) => {
                self.dispatch_household(&intent, &stt.text, turn, conversation)
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

    async fn voice_action(
        &self,
        command: &nexa_nlu::grammar::HomeCommand,
        room: &str,
        conversation: &str,
    ) -> String {
        let started = Instant::now();
        let Some(action) = self.resolve_voice_action(command, room) else {
            tracing::info!(route = "ha", room_id = %room, utterance_id = %conversation,
                latency_ms = started.elapsed().as_millis(), status = "unresolved", "Voice route executed");
            return "¿A qué dispositivo te referís?".into();
        };
        let result = self.execute_home(crate::id("voice"), action).await;
        let latency_ms = started.elapsed().as_millis();
        match &result {
            Ok(receipt) => {
                tracing::info!(route = "ha", room_id = %room, utterance_id = %conversation,
                latency_ms, status = %receipt.status, "Voice route executed")
            }
            Err(error) => {
                tracing::warn!(route = "ha", room_id = %room, utterance_id = %conversation,
                latency_ms, status = ?error.code(), "Voice route failed")
            }
        }
        match result.as_ref().map(|value| value.status.as_str()) {
            Ok("accepted") => "Mandé la orden.".into(),
            Ok("uncertain" | "partial") => "No pude confirmar la orden.".into(),
            _ => "No pude enviar la orden.".into(),
        }
    }

    async fn dispatch_household(
        &self,
        intent: &HouseholdIntent,
        input: &str,
        turn: &Turn,
        conversation: &str,
    ) -> Result<String, Status> {
        let started = Instant::now();
        let result = self
            .voice_household(intent, input, turn, conversation)
            .await;
        let latency_ms = started.elapsed().as_millis();
        match &result {
            Ok(_) => {
                tracing::info!(route = "household", room_id = %turn.zone, utterance_id = %conversation,
                latency_ms, status = "ok", "Voice route executed")
            }
            Err(error) => {
                tracing::warn!(route = "household", room_id = %turn.zone, utterance_id = %conversation,
                latency_ms, status = ?error.code(), "Voice route failed")
            }
        }
        result
    }
}
