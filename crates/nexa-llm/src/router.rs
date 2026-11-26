use crate::{
    CancellationToken, CloudConsentGate, Endpoint, LlmError, Metrics, OpenAiClient,
    health::{Cooldowns, Samples},
};
use nexa_config::LlmConfig;
use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};
use tokio::{sync::Mutex, time::Instant};

#[derive(Debug)]
pub struct FailoverRouter {
    endpoints: Vec<Endpoint>,
    client: OpenAiClient,
    health: Cooldowns,
    metrics: Mutex<HashMap<String, Samples>>,
}

impl FailoverRouter {
    pub async fn from_config(config: &LlmConfig, path: Option<PathBuf>) -> Result<Self, LlmError> {
        Self::new(
            Endpoint::from_config(config).into(),
            Arc::new(CloudConsentGate::new(config.cloud_consent)),
            path,
        )
        .await
    }

    pub async fn new(
        mut endpoints: Vec<Endpoint>,
        gate: Arc<CloudConsentGate>,
        path: Option<PathBuf>,
    ) -> Result<Self, LlmError> {
        endpoints.sort_by_key(|e| e.priority);
        Ok(Self {
            endpoints,
            client: OpenAiClient::new(gate)?,
            health: Cooldowns::load(path).await?,
            metrics: Mutex::default(),
        })
    }

    pub fn consent(&self) -> &CloudConsentGate {
        &self.client.gate
    }

    pub async fn clear_cooldown(&self, id: &str) -> Result<(), LlmError> {
        Ok(self.health.clear(id).await?)
    }

    pub async fn cooldown_remaining(&self, id: &str) -> Duration {
        self.health.remaining(id).await
    }

    /// Returns the failure streak for this endpoint, including after the
    /// cooldown expires. A success or clear resets it to zero.
    pub async fn consecutive_errors(&self, id: &str) -> u32 {
        self.health.consecutive_errors(id).await
    }

    pub async fn metrics(&self, id: &str) -> Option<Metrics> {
        self.metrics
            .lock()
            .await
            .get(id)
            .and_then(Samples::snapshot)
    }

    /// Returns the successful endpoint id. A failure after visible output is
    /// terminal for this stream, so consumers never receive duplicated prefixes.
    pub async fn stream(
        &self,
        prompt: &str,
        max_tokens: u32,
        cancel: &CancellationToken,
        emit: impl FnMut(&str),
    ) -> Result<&str, LlmError> {
        self.stream_endpoints(None, prompt, max_tokens, None, cancel, false, emit)
            .await
    }

    /// Generate with local endpoints only, even when cloud egress is authorized.
    pub async fn stream_local_only(
        &self,
        prompt: &str,
        max_tokens: u32,
        cancel: &CancellationToken,
        emit: impl FnMut(&str),
    ) -> Result<&str, LlmError> {
        self.stream_endpoints(None, prompt, max_tokens, None, cancel, true, emit)
            .await
    }

    /// Run a bounded local-only completion with a custom system prompt and temperature.
    pub async fn complete_local_with_system(
        &self,
        system_prompt: &str,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        cancel: &CancellationToken,
    ) -> Result<(&str, String), LlmError> {
        let mut output = String::new();
        let endpoint = self
            .stream_endpoints(
                Some(system_prompt),
                prompt,
                max_tokens,
                Some(temperature),
                cancel,
                true,
                |chunk| output.push_str(chunk),
            )
            .await?;
        Ok((endpoint, output))
    }

    #[allow(clippy::too_many_arguments)]
    async fn stream_endpoints(
        &self,
        system_prompt: Option<&str>,
        prompt: &str,
        max_tokens: u32,
        temperature: Option<f32>,
        cancel: &CancellationToken,
        local_only: bool,
        mut emit: impl FnMut(&str),
    ) -> Result<&str, LlmError> {
        if cancel.is_cancelled() {
            return Err(LlmError::Cancelled);
        }
        let mut last = LlmError::Unavailable;
        for ep in self.endpoints.iter().filter(|ep| !local_only || !ep.cloud) {
            if !self.health.remaining(&ep.id).await.is_zero() {
                continue;
            }
            let start = Instant::now();
            // Filtered reasoning can establish endpoint progress before the caller sees text.
            let mut visible = false;
            let (result, first_token) = if let Some(system_prompt) = system_prompt {
                self.client
                    .stream_attempt_with(
                        ep,
                        system_prompt,
                        prompt,
                        max_tokens,
                        temperature,
                        cancel,
                        |text| {
                            visible = true;
                            emit(text);
                        },
                    )
                    .await
            } else {
                self.client
                    .stream_attempt(ep, prompt, max_tokens, cancel, |text| {
                        visible = true;
                        emit(text);
                    })
                    .await
            };
            if cancel.is_cancelled() {
                return Err(LlmError::Cancelled);
            }
            match result {
                Ok(tokens) => {
                    let elapsed = start.elapsed();
                    Self::log_health_write(self.health.clear(&ep.id).await);
                    self.metrics
                        .lock()
                        .await
                        .entry(ep.id.clone())
                        .or_default()
                        .record(elapsed, tokens);
                    return Ok(&ep.id);
                }
                Err(e) => {
                    if !matches!(e, LlmError::Consent) {
                        let http_status = match &e {
                            LlmError::Status(status) => Some(status.as_u16()),
                            _ => None,
                        };
                        tracing::warn!(endpoint_id = %ep.id, latency_ms = start.elapsed().as_millis(),
                            status = e.code(), http_status, first_token, "LLM endpoint failed");
                    }
                    if !first_token && e.affects_cooldown() {
                        Self::log_health_write(self.health.record_failure(&ep.id, cancel).await);
                    }
                    if visible || !(e.retryable() || matches!(e, LlmError::Consent)) {
                        return Err(e);
                    }
                    last = e;
                }
            }
        }
        if cancel.is_cancelled() {
            Err(LlmError::Cancelled)
        } else {
            Err(last)
        }
    }

    fn log_health_write(result: std::io::Result<()>) {
        if let Err(error) = result {
            tracing::warn!(kind = ?error.kind(), "Could not persist LLM cooldown state");
        }
    }
}
