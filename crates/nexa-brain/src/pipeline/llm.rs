use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use nexa_ha::EffectIntent;
use nexa_llm::{CancellationToken, LlmError};
use nexa_proto::v1::{self, LiveEventKind};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;

use crate::Brain;

use super::Turn;

const VOICE_MAX_TOKENS: u32 = 120;
const HOME_ROUTER_MAX_TOKENS: u32 = 140;
const HOME_ROUTER_TEMPERATURE: f32 = 0.15;
const MAX_CATALOG_ENTITIES: usize = 32;

const HOME_ROUTER_SYSTEM_PROMPT: &str = "Sos Nexa, asistente doméstico rioplatense. Respondé SOLO JSON válido sin markdown ni razonamiento:\n{\"action\": {\"entity_id\": \"...\", \"operation\": \"...\", \"params\": {}} | null, \"confidence\": 0.0, \"reply\": \"<=12 palabras en español rioplatense\"}\nReglas:\n1. Usá \"action\" SOLO con entity_id y operation exactos de la lista de dispositivos.\n2. El texto viene de STT de voz y puede tener errores fonéticos menores (ej: 'aprende' por 'prendé', o una palabra rara en vez de 'escritorio'/'luz'). Si el usuario claramente pide prender/apagar/regular sin nombrar otra habitación válida, actuá sobre la luz principal de source_room.\n3. Operaciones y params: turn_on ({}), turn_off ({}), activate_scene ({}), open_cover ({}), close_cover ({}), stop_cover ({}), set_cover_position ({\"position\": \"0..100\"}), brightness ({\"brightness_pct\": \"1..100\"}), color_temp_kelvin ({\"color_temp_kelvin\": \"2000..6500\"}), set_temperature ({\"temperature\": \"16..30\"}), set_hvac_mode ({\"hvac_mode\": \"off|cool|heat|dry|fan_only|auto\"}).\n4. Nunca inventes entidades ni controles cerraduras o alarmas.\n5. Si es pregunta de estado o charla general, devolvé \"action\": null, \"confidence\": 0.0 y \"reply\" breve.";

#[derive(Debug, Deserialize)]
struct LlmHomeDecision {
    #[serde(default)]
    action: Option<LlmHomeAction>,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    reply: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LlmHomeAction {
    entity_id: String,
    operation: String,
    #[serde(default)]
    params: HashMap<String, Value>,
}

fn strip_json_fences(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        let body = rest
            .strip_prefix("json")
            .or_else(|| rest.strip_prefix("JSON"))
            .unwrap_or(rest)
            .trim_start();
        if let Some(inner) = body.strip_suffix("```") {
            return inner.trim();
        }
    }
    trimmed
}

fn parse_llm_home_decision(raw: &str) -> Option<LlmHomeDecision> {
    let clean = strip_json_fences(raw);
    if let Ok(parsed) = serde_json::from_str::<LlmHomeDecision>(clean) {
        return Some(parsed);
    }
    let start = clean.find('{')?;
    let end = clean.rfind('}')?;
    (end > start)
        .then(|| serde_json::from_str::<LlmHomeDecision>(&clean[start..=end]).ok())
        .flatten()
}

fn flatten_params(raw: &HashMap<String, Value>) -> Option<HashMap<String, String>> {
    let mut out = HashMap::with_capacity(raw.len());
    for (key, value) in raw {
        let text = match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => return None,
        };
        out.insert(key.clone(), text);
    }
    Some(out)
}

fn is_disallowed_security_entity(entity_id: &str) -> bool {
    matches!(
        entity_id.split_once('.').map(|(domain, _)| domain),
        Some("lock" | "alarm_control_panel")
    )
}

fn is_readonly_state_query(text: &str) -> bool {
    let folded = nexa_nlu::normalize::words(text);
    let tokens: Vec<&str> = folded.split_whitespace().collect();
    let Some(&first) = tokens.first() else {
        return false;
    };
    let asks_status_verb = matches!(
        first,
        "esta" | "estan" | "sigue" | "siguen" | "quedo" | "quedaron"
    );
    let has_state_adjective = tokens.iter().any(|word| {
        matches!(
            *word,
            "prendida"
                | "prendidas"
                | "prendido"
                | "prendidos"
                | "encendida"
                | "encendidas"
                | "encendido"
                | "encendidos"
                | "apagada"
                | "apagadas"
                | "apagado"
                | "apagados"
                | "abierta"
                | "abiertas"
                | "abierto"
                | "abiertos"
                | "cerrada"
                | "cerradas"
                | "cerrado"
                | "cerrados"
        )
    });
    if asks_status_verb && has_state_adjective {
        return true;
    }
    folded.starts_with("como esta ")
        || folded.starts_with("como estan ")
        || folded.starts_with("a cuanto esta ")
        || folded.starts_with("que temperatura ")
}

impl Brain {
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
                actor: Some(v1::ActorContext {
                    subject: "nexa-brain".into(),
                    role: v1::Role::Operator as i32,
                    permissions: vec![v1::Permission::MemoryRead as i32],
                    ..Default::default()
                }),
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

    fn allowlisted_home_catalog(&self, source_room: &str) -> Vec<String> {
        let other_rooms: Vec<String> = self
            .config
            .rooms
            .rooms
            .iter()
            .map(|room| room.room_id.clone())
            .filter(|id| id != source_room)
            .collect();
        let mut entities: Vec<_> = self
            .catalog()
            .entities
            .into_iter()
            .filter(|entity| {
                if entity.capabilities.is_empty()
                    || is_disallowed_security_entity(&entity.entity_id)
                {
                    return false;
                }
                if let Some(scene_slug) = entity.entity_id.strip_prefix("scene.") {
                    // Keep global scenes and source_room scenes; omit scenes scoped to other rooms.
                    return !other_rooms
                        .iter()
                        .any(|room| scene_slug.starts_with(&format!("{room}_")));
                }
                true
            })
            .collect();
        // Prioritize devices in the room where the user spoke, then non-scene devices, then scenes.
        entities.sort_by(|a, b| {
            (
                a.room_id != source_room,
                a.entity_id.starts_with("scene."),
                &a.entity_id,
            )
                .cmp(&(
                    b.room_id != source_room,
                    b.entity_id.starts_with("scene."),
                    &b.entity_id,
                ))
        });
        entities
            .into_iter()
            .take(MAX_CATALOG_ENTITIES)
            .map(|entity| {
                let room = if entity.room_id.is_empty() {
                    "-"
                } else {
                    entity.room_id.as_str()
                };
                format!(
                    "{}|room={}|state={}|ops={}",
                    entity.entity_id,
                    room,
                    entity.state,
                    entity.capabilities.join(",")
                )
            })
            .collect()
    }

    async fn try_local_home_router(
        &self,
        turn: &Turn,
        conversation: &str,
        text: &str,
        context: &[String],
    ) -> Result<Option<String>, Status> {
        if self.home.deps.is_none() || self.config.ha_controls.entities.is_empty() {
            return Ok(None);
        }
        let catalog = self.allowlisted_home_catalog(&turn.zone);
        if catalog.is_empty() {
            return Ok(None);
        }
        // Keep stable device list first so llama-server reuses its KV prefix cache across turns.
        let mut user_prompt = format!("Dispositivos permitidos:\n{}\n", catalog.join("\n"));
        if !context.is_empty() {
            user_prompt.push_str("Memoria local: ");
            user_prompt.push_str(&json!(context).to_string());
            user_prompt.push('\n');
        }
        user_prompt.push_str(&format!("source_room: {}\nutterance: {}", turn.zone, text));
        let started = Instant::now();
        let cancel = turn.lease.cancel.child_token();
        let (endpoint_id, raw) = match self
            .llm
            .complete_local_with_system(
                HOME_ROUTER_SYSTEM_PROMPT,
                &user_prompt,
                HOME_ROUTER_MAX_TOKENS,
                HOME_ROUTER_TEMPERATURE,
                &cancel,
            )
            .await
        {
            Ok(result) => result,
            Err(LlmError::Cancelled) => return Err(Status::cancelled("turn cancelled")),
            Err(error) => {
                tracing::warn!(
                    route = "llm_ha",
                    room_id = %turn.zone,
                    utterance_id = %conversation,
                    latency_ms = started.elapsed().as_millis(),
                    status = error.code(),
                    "Local HA LLM router failed; falling back to streaming reply"
                );
                return Ok(None);
            }
        };
        let llm_ms = started.elapsed().as_millis();
        let Some(decision) = parse_llm_home_decision(&raw) else {
            tracing::info!(
                route = "llm_ha",
                room_id = %turn.zone,
                utterance_id = %conversation,
                endpoint_id,
                latency_ms = llm_ms,
                status = "non_json_fallback",
                "Local HA LLM router returned non-JSON"
            );
            let fallback = raw.trim();
            let looks_like_broken_json = fallback.starts_with('{') || fallback.starts_with("```");
            return Ok((!fallback.is_empty() && !looks_like_broken_json)
                .then(|| fallback.chars().take(320).collect()));
        };

        let min_conf = self.config.core.nlu.min_llm_command_confidence;
        let readonly_query = is_readonly_state_query(text);
        if let Some(action) = decision.action
            && !readonly_query
            && decision.confidence.is_finite()
            && decision.confidence >= min_conf
        {
            let allowed = !is_disallowed_security_entity(&action.entity_id)
                && self
                    .config
                    .ha_controls
                    .entities
                    .contains_key(&action.entity_id);
            let intent = if allowed {
                flatten_params(&action.params)
                    .and_then(|params| {
                        crate::home::typed_operation(&action.operation, &params).ok()
                    })
                    .map(|operation| EffectIntent {
                        entity_id: action.entity_id.clone(),
                        operation,
                    })
            } else {
                None
            };
            if let Some(intent) = intent {
                let result = self.execute_home(crate::id("llm-voice"), intent).await;
                let total_ms = started.elapsed().as_millis();
                match &result {
                    Ok(receipt) => tracing::info!(
                        route = "llm_ha",
                        room_id = %turn.zone,
                        utterance_id = %conversation,
                        endpoint_id,
                        confidence = decision.confidence,
                        latency_ms = total_ms,
                        status = %receipt.status,
                        "Voice route executed"
                    ),
                    Err(error) => tracing::warn!(
                        route = "llm_ha",
                        room_id = %turn.zone,
                        utterance_id = %conversation,
                        endpoint_id,
                        confidence = decision.confidence,
                        latency_ms = total_ms,
                        status = ?error.code(),
                        "Voice route failed"
                    ),
                }
                let spoken = match result.as_ref().map(|receipt| receipt.status.as_str()) {
                    Ok("accepted") => decision
                        .reply
                        .filter(|reply| !reply.trim().is_empty() && reply.chars().count() <= 180)
                        .unwrap_or_else(|| "Listo.".into()),
                    Ok("uncertain" | "partial") => "No pude confirmar la orden.".into(),
                    _ => "No pude enviar la orden.".into(),
                };
                return Ok(Some(spoken));
            }
            tracing::warn!(
                route = "llm_ha",
                room_id = %turn.zone,
                utterance_id = %conversation,
                endpoint_id,
                confidence = decision.confidence,
                latency_ms = llm_ms,
                status = "denied_action",
                "Local HA LLM proposed a disallowed or invalid action"
            );
            return Ok(Some("No puedo ejecutar esa acción por voz.".into()));
        }

        tracing::info!(
            route = "llm",
            room_id = %turn.zone,
            utterance_id = %conversation,
            endpoint_id,
            latency_ms = llm_ms,
            local_only = true,
            status = "ok",
            "Voice route executed"
        );
        let reply = decision
            .reply
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        Ok(reply)
    }

    pub(super) async fn slow_llm(
        &self,
        turn: &Turn,
        conversation: &str,
        text: &str,
    ) -> Result<(), Status> {
        let context = match tokio::time::timeout(
            Duration::from_millis(1500),
            self.memory_context(text),
        )
        .await
        {
            Ok(Ok(context)) => context,
            Ok(Err(error)) => {
                tracing::warn!(room_id = %turn.zone, utterance_id = %conversation, code = ?error.code(),
                    "Shared memory lookup failed");
                Vec::new()
            }
            Err(_) => {
                tracing::warn!(room_id = %turn.zone, utterance_id = %conversation, code = ?tonic::Code::DeadlineExceeded,
                    "Shared memory lookup timed out");
                Vec::new()
            }
        };
        if let Some(reply) = self
            .try_local_home_router(turn, conversation, text, &context)
            .await?
        {
            return self.say(turn, conversation, reply).await.map(|_| ());
        }
        let local_prompt = (!context.is_empty()).then(|| {
            // Memory context can go only to the local model. Only the original
            // user input can be retried across the cloud consent boundary.
            format!(
                "Contexto local de la casa: {}\nConsulta: {text}",
                json!(context)
            )
        });
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
            let mut result = self
                .stream_reply(
                    turn,
                    conversation,
                    local_prompt.as_deref().unwrap_or(text),
                    local_prompt.is_some(),
                    &cancel,
                    |chunk| emit(chunk, &mut segment, &mut started, &mut truncated),
                )
                .await;
            if result.is_err()
                && local_prompt.is_some()
                && !started
                && segment.is_empty()
                && !cancel.is_cancelled()
            {
                result = self
                    .stream_reply(turn, conversation, text, false, &cancel, |chunk| {
                        emit(chunk, &mut segment, &mut started, &mut truncated)
                    })
                    .await;
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

    async fn stream_reply(
        &self,
        turn: &Turn,
        conversation: &str,
        prompt: &str,
        local_only: bool,
        cancel: &CancellationToken,
        emit: impl FnMut(&str),
    ) -> Result<(), LlmError> {
        let started = Instant::now();
        let result = if local_only {
            self.llm
                .stream_local_only(prompt, VOICE_MAX_TOKENS, cancel, emit)
                .await
        } else {
            self.llm
                .stream(prompt, VOICE_MAX_TOKENS, cancel, emit)
                .await
        };
        let latency_ms = started.elapsed().as_millis();
        match &result {
            Ok(endpoint_id) => {
                tracing::info!(route = "llm", room_id = %turn.zone, utterance_id = %conversation,
                latency_ms, endpoint_id, local_only, status = "ok", "Voice route executed")
            }
            Err(error) => {
                let http_status = match error {
                    LlmError::Status(status) => Some(status.as_u16()),
                    _ => None,
                };
                tracing::warn!(route = "llm", room_id = %turn.zone, utterance_id = %conversation,
                    latency_ms, local_only, status = error.code(), http_status, "Voice route failed");
            }
        }
        result.map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_and_fenced_llm_home_decisions() {
        let fenced = "```json\n{\"action\":{\"entity_id\":\"light.grupo_escritorio\",\"operation\":\"brightness\",\"params\":{\"brightness_pct\":50}},\"confidence\":0.92,\"reply\":\"Puse la luz al 50%.\"}\n```";
        let decision = parse_llm_home_decision(fenced).expect("fenced JSON must parse");
        let action = decision.action.expect("action present");
        assert_eq!(action.entity_id, "light.grupo_escritorio");
        assert_eq!(action.operation, "brightness");
        let params = flatten_params(&action.params).expect("params flatten");
        assert_eq!(params.get("brightness_pct").map(String::as_str), Some("50"));
        assert!(crate::home::typed_operation(&action.operation, &params).is_ok());
        assert!((decision.confidence - 0.92).abs() < 1e-6);
        assert_eq!(decision.reply.as_deref(), Some("Puse la luz al 50%."));
    }

    #[test]
    fn blocks_security_domains_even_if_proposed_by_llm() {
        assert!(is_disallowed_security_entity("lock.puerta_entrada"));
        assert!(is_disallowed_security_entity("alarm_control_panel.casa"));
        assert!(!is_disallowed_security_entity("light.grupo_escritorio"));
    }

    #[test]
    fn distinguishes_readonly_state_queries_from_natural_requests() {
        assert!(is_readonly_state_query("¿Está encendida la luz?"));
        assert!(is_readonly_state_query("quedo prendida la luz del living"));
        assert!(is_readonly_state_query("¿Cómo está el aire?"));
        assert!(!is_readonly_state_query("Tengo poca luz acá"));
        assert!(!is_readonly_state_query(
            "¿Podés prender la luz del escritorio?"
        ));
    }
}
