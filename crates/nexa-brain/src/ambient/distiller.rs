use std::time::Duration;

use nexa_proto::v1::{self, Permission};
use serde::Deserialize;
use serde_json::Value;
use tracing::warn;

use crate::{Brain, ambient::ClosedSession, pipeline::llm::strip_json_fences};

const DISTILL_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_FACTS_PER_SESSION: usize = 5;
const MAX_PERSONALITIES_PER_SESSION: usize = 4;
const MAX_FACT_CHARS: usize = 256;
const MAX_SUMMARY_BYTES: usize = 4096;

const DISTILLER_SYSTEM_PROMPT: &str = "Sos un destilador de memoria doméstica local. Analizás una conversación ambiental en español y devolvés EXCLUSIVAMENTE JSON válido sin markdown:\n{\"facts\":[{\"content\":\"hecho concreto y duradero en <=256 caracteres\",\"category\":\"preferencia|hogar|rutina|recordatorio|general\"}],\"personalities\":[{\"profile_id\":\"id exacto si aparece como [perfil:<id>]\",\"summary\":{\"tone\":\"...\",\"style\":\"...\",\"interests\":[\"...\"]}}]}\nReglas:\n1. Máximo 5 facts relevantes y no triviales (no guardes saludos ni ruido).\n2. Máximo 4 personalities, SOLO para profile_id explícitamente vistos en [perfil:<id>] (nunca inventes IDs ni incluyas hablantes anónimos).\n3. Si no hay hechos ni rasgos claros, devolvé {\"facts\":[],\"personalities\":[]}.";

#[derive(Debug, Deserialize)]
struct DistilledPayload {
    #[serde(default)]
    facts: Vec<DistilledFact>,
    #[serde(default)]
    personalities: Vec<DistilledPersonality>,
}

#[derive(Debug, Deserialize)]
struct DistilledFact {
    content: String,
    #[serde(default = "default_category")]
    category: String,
}

fn default_category() -> String {
    "general".to_owned()
}

#[derive(Debug, Deserialize)]
struct DistilledPersonality {
    profile_id: String,
    summary: Value,
}

fn parse_distilled_payload(raw: &str) -> Option<DistilledPayload> {
    let clean = strip_json_fences(raw);
    if let Ok(parsed) = serde_json::from_str::<DistilledPayload>(clean) {
        return Some(parsed);
    }
    let start = clean.find('{')?;
    let end = clean.rfind('}')?;
    (end > start)
        .then(|| serde_json::from_str::<DistilledPayload>(&clean[start..=end]).ok())
        .flatten()
}

fn operator_actor(permissions: &[Permission]) -> v1::ActorContext {
    v1::ActorContext {
        subject: "nexa-brain".into(),
        role: v1::Role::Operator as i32,
        permissions: permissions.iter().map(|p| *p as i32).collect(),
        ..Default::default()
    }
}

pub(crate) fn should_distill(session: &ClosedSession) -> bool {
    let total_words: usize = session
        .turns
        .iter()
        .map(|turn| turn.text.split_whitespace().count())
        .sum();
    session.turns.len() >= 2 || total_words >= 20
}

impl Brain {
    pub(crate) async fn distill_closed_session(&self, session: ClosedSession) {
        if !should_distill(&session) || self.ambient.is_erased(&session.conversation_id) {
            return;
        }
        let Some(memory) = self.memory.as_deref() else {
            return;
        };
        match tokio::time::timeout(Duration::from_secs(1), memory.ambient_control()).await {
            Ok(Ok(ctrl)) if !ctrl.paused => {}
            _ => return,
        }

        let transcript: String = session
            .turns
            .iter()
            .take(24)
            .map(|turn| {
                let speaker = turn
                    .matched_profile_id
                    .as_deref()
                    .map_or_else(|| "anon".to_owned(), |id| format!("perfil:{id}"));
                let text: String = turn.text.trim().chars().take(320).collect();
                format!("[{speaker}] {text}")
            })
            .collect::<Vec<_>>()
            .join("\n");
        if transcript.trim().is_empty() {
            return;
        }
        let prompt = format!(
            "Habitación: {}\nConversación:\n{transcript}",
            session.room_id
        );
        let cancel = self.shutdown.child_token();
        let llm_future = self.llm.complete_local_with_system(
            DISTILLER_SYSTEM_PROMPT,
            &prompt,
            512,
            0.1,
            &cancel,
        );
        let raw = match tokio::time::timeout(DISTILL_TIMEOUT, llm_future).await {
            Ok(Ok((_, raw))) => raw,
            Ok(Err(error)) => {
                warn!(
                    conversation_id = %session.conversation_id,
                    room_id = %session.room_id,
                    status = error.code(),
                    "Ambient distiller local LLM failed"
                );
                return;
            }
            Err(_) => {
                warn!(
                    conversation_id = %session.conversation_id,
                    room_id = %session.room_id,
                    "Ambient distiller timed out"
                );
                return;
            }
        };
        if self.ambient.is_erased(&session.conversation_id) {
            return;
        }
        let Some(payload) = parse_distilled_payload(&raw) else {
            warn!(
                conversation_id = %session.conversation_id,
                room_id = %session.room_id,
                "Ambient distiller returned invalid JSON"
            );
            return;
        };

        let facts: Vec<(String, String)> = payload
            .facts
            .into_iter()
            .filter_map(|fact| {
                let content: String = fact.content.trim().chars().take(MAX_FACT_CHARS).collect();
                if content.is_empty() {
                    return None;
                }
                let category: String = fact.category.trim().chars().take(64).collect::<String>();
                let category = if category.is_empty() {
                    default_category()
                } else {
                    category
                };
                Some((content, category))
            })
            .take(MAX_FACTS_PER_SESSION)
            .collect();

        if !facts.is_empty() {
            let texts: Vec<String> = facts.iter().map(|(content, _)| content.clone()).collect();
            match self.ml.embed_text(v1::EmbedTextRequest { texts }).await {
                Ok(batch) if batch.vectors.len() == facts.len() => {
                    for ((content, category), vector) in facts.into_iter().zip(batch.vectors) {
                        if self.ambient.is_erased(&session.conversation_id) {
                            return;
                        }
                        if vector.values.is_empty()
                            || vector.values.len() > 4096
                            || !vector.values.iter().all(|v| v.is_finite())
                        {
                            continue;
                        }
                        let req = v1::StoreFactRequest {
                            conversation_id: session.conversation_id.clone(),
                            owner_user_id: String::new(),
                            content,
                            category,
                            embedding: vector.values,
                            actor: Some(operator_actor(&[Permission::MemoryReview])),
                            proposed: true,
                        };
                        if let Err(error) = memory.store_fact(req).await {
                            warn!(
                                conversation_id = %session.conversation_id,
                                code = ?error.code(),
                                "Failed to store distilled proposed fact"
                            );
                        }
                    }
                }
                Ok(_) => warn!(
                    conversation_id = %session.conversation_id,
                    "Ambient distiller embedding count mismatch"
                ),
                Err(error) => warn!(
                    conversation_id = %session.conversation_id,
                    code = ?error.code(),
                    "Ambient distiller embed_text failed"
                ),
            }
        }

        for item in payload
            .personalities
            .into_iter()
            .filter(|entry| {
                session.matched_profiles.contains(&entry.profile_id) && entry.summary.is_object()
            })
            .take(MAX_PERSONALITIES_PER_SESSION)
        {
            if self.ambient.is_erased(&session.conversation_id) {
                return;
            }
            let summary_json = item.summary.to_string();
            if summary_json.len() > MAX_SUMMARY_BYTES {
                continue;
            }
            let req = v1::UpsertVoicePersonalityRequest {
                profile_id: item.profile_id,
                summary_json,
                source_conversation_id: session.conversation_id.clone(),
                actor: Some(operator_actor(&[
                    Permission::MemoryReview,
                    Permission::VoicesReview,
                ])),
            };
            if let Err(error) = memory.upsert_voice_personality(req).await {
                warn!(
                    conversation_id = %session.conversation_id,
                    code = ?error.code(),
                    "Failed to upsert distilled voice personality"
                );
            }
        }
    }
}
