use std::sync::OnceLock;
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

/// Minimum Whisper avg_logprob for uncorroborated SlowLlm transcripts lacking a wake word.
/// Live captures showed real commands at [-0.69, -0.34] and non-speech hallucinations at <= -1.20.
const MIN_UNCORROBORATED_LLM_LOGPROB: f32 = -1.10;

fn contains_wake_word(text: &str) -> bool {
    words(text).split_whitespace().any(|word| {
        nexa_nlu::normalize::WAKE_WORDS.contains(&word) || matches!(word, "neska" | "nexca")
    })
}

fn should_drop_low_confidence_llm(route: &Route, avg_logprob: f32, stt_text: &str) -> bool {
    matches!(route, Route::SlowLlm)
        && avg_logprob.is_finite()
        && avg_logprob < MIN_UNCORROBORATED_LLM_LOGPROB
        && !contains_wake_word(stt_text)
}

/// Opt-in operator diagnostics flag (off by default) to log voice transcripts and LLM decisions.
pub(super) fn debug_voice_transcripts() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("NEXA_DEBUG_VOICE_TRANSCRIPTS")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
    })
}

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
            no_speech_prob = stt.no_speech_prob,
            avg_logprob = stt.avg_logprob,
            compression_ratio = stt.compression_ratio,
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
        let shadow_veto_ok = shadow.is_empty() || corroborated;
        let admitted = (wake || (follow_up && shadow_veto_ok) || routed.acceptance.ring_stop)
            && (!routed.requires_veto || shadow_veto_ok);
        let path = routed.decision.path.as_str_name();
        let low_confidence_llm =
            should_drop_low_confidence_llm(&routed.route, stt.avg_logprob, &stt.text);
        let admitted_route =
            admitted && !matches!(routed.route, Route::Rejected) && !low_confidence_llm;
        let reason = if low_confidence_llm {
            "low_confidence_llm"
        } else {
            routed.decision.reason.as_deref().unwrap_or("none")
        };
        tracing::info!(
            room_id = %room,
            utterance_id = %conversation,
            path,
            admitted = admitted_route,
            reason,
            corroborated,
            "NLU route resolved"
        );
        let transcript_preview: String = stt.text.chars().take(320).collect();
        if debug_voice_transcripts() {
            tracing::info!(
                debug_voice = true,
                room_id = %room,
                utterance_id = %conversation,
                kind = utterance.kind().as_str_name(),
                stt_text = %transcript_preview,
                normalized = %routed.decision.normalized_text,
                path,
                admitted = admitted_route,
                reason,
                "Voice transcript trace"
            );
        }
        self.event(
            LiveEventKind::Turn,
            room,
            conversation,
            json!({
                "kind": utterance.kind().as_str_name(),
                "transcript": transcript_preview,
                "normalized": routed.decision.normalized_text,
                "avg_logprob": stt.avg_logprob,
                "compression_ratio": stt.compression_ratio,
                "path": path,
                "reason": if low_confidence_llm { Some("low_confidence_llm") } else { routed.decision.reason.as_deref() },
                "admitted": admitted_route,
            }),
        );
        if low_confidence_llm {
            return Ok(());
        }
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
        self.events.update_turn(
            conversation,
            json!({
                "reply": reply.chars().take(320).collect::<String>(),
            }),
        );
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
            self.events
                .update_turn(conversation, json!({"executed_status": "unresolved"}));
            return "¿A qué dispositivo te referís?".into();
        };
        let entity_id = action.entity_id.clone();
        let operation = action.operation.name();
        let result = self.execute_home(crate::id("voice"), action).await;
        let latency_ms = started.elapsed().as_millis();
        match &result {
            Ok(receipt) => {
                tracing::info!(route = "ha", room_id = %room, utterance_id = %conversation,
                entity_id = %entity_id, operation, latency_ms, status = %receipt.status, "Voice route executed");
                self.events.update_turn(
                    conversation,
                    json!({
                        "executed_entity": entity_id,
                        "executed_operation": operation,
                        "executed_status": receipt.status,
                    }),
                );
            }
            Err(error) => {
                tracing::warn!(route = "ha", room_id = %room, utterance_id = %conversation,
                entity_id = %entity_id, operation, latency_ms, status = ?error.code(), "Voice route failed");
                self.events.update_turn(
                    conversation,
                    json!({
                        "executed_entity": entity_id,
                        "executed_operation": operation,
                        "executed_status": format!("{:?}", error.code()),
                    }),
                );
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

#[cfg(test)]
mod tests {
    use super::*;
    use nexa_nlu::grammar::{Domain, HomeCommand, Operation, Slots};

    #[test]
    fn drops_low_confidence_uncorroborated_llm_hallucinations_only() {
        assert!(should_drop_low_confidence_llm(
            &Route::SlowLlm,
            -1.205,
            "Cuidado."
        ));
        assert!(should_drop_low_confidence_llm(
            &Route::SlowLlm,
            -1.358,
            "Cuidado por la noche Cuidado por la noche"
        ));
        assert!(!should_drop_low_confidence_llm(
            &Route::SlowLlm,
            -0.494,
            "Nexa apagá levítorio."
        ));
        assert!(!should_drop_low_confidence_llm(
            &Route::SlowLlm,
            -1.25,
            "Nexa tengo poca luz acá"
        ));
        let domotics = Route::Domotics(HomeCommand {
            domain: Domain::Light,
            operation: Operation::On,
            target: String::new(),
            room: Some("escritorio".into()),
            slots: Slots::default(),
        });
        assert!(!should_drop_low_confidence_llm(
            &domotics,
            -1.30,
            "prendé la luz"
        ));
    }
}
