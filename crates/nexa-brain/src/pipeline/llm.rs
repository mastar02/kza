use std::time::{Duration, Instant};

use nexa_llm::{CancellationToken, LlmError};
use nexa_proto::v1::{self, LiveEventKind};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;

use crate::Brain;

use super::Turn;

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
            self.llm.stream_local_only(prompt, 256, cancel, emit).await
        } else {
            self.llm.stream(prompt, 256, cancel, emit).await
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
