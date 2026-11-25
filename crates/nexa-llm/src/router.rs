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
        Ok(self.health.update(id, false).await?)
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
        mut emit: impl FnMut(&str),
    ) -> Result<&str, LlmError> {
        let mut last = LlmError::Unavailable;
        for ep in &self.endpoints {
            if cancel.is_cancelled() {
                return Err(LlmError::Cancelled);
            }
            if !self.health.remaining(&ep.id).await.is_zero() {
                continue;
            }
            let start = Instant::now();
            let mut visible = false;
            let result = self
                .client
                .stream(ep, prompt, max_tokens, cancel, |text| {
                    visible = true;
                    emit(text);
                })
                .await;
            match result {
                Ok(tokens) => {
                    let elapsed = start.elapsed();
                    self.record_health(&ep.id, false).await;
                    self.metrics
                        .lock()
                        .await
                        .entry(ep.id.clone())
                        .or_default()
                        .record(elapsed, tokens);
                    return Ok(&ep.id);
                }
                Err(e) => {
                    if e.retryable() {
                        self.record_health(&ep.id, true).await;
                    } else if !matches!(e, LlmError::Consent) {
                        return Err(e);
                    }
                    if visible {
                        return Err(e);
                    }
                    last = e;
                }
            }
        }
        Err(last)
    }

    async fn record_health(&self, id: &str, failed: bool) {
        if let Err(error) = self.health.update(id, failed).await {
            tracing::warn!(kind = ?error.kind(), "Could not persist LLM cooldown state");
        }
    }
}
