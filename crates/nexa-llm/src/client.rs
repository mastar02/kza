use std::{io, sync::Arc, time::Duration};

use eventsource_stream::{EventStreamError, Eventsource};
use futures_util::StreamExt;
use nexa_config::LlmConfig;
use serde::Deserialize;
use serde_json::json;
use tokio::time::timeout;

use crate::{CancellationToken, CloudConsentGate, ThinkTagStripper};

#[derive(Clone)]
pub struct ApiKey(String);

impl From<String> for ApiKey {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

#[derive(Clone, Debug)]
pub struct Endpoint {
    pub id: String,
    /// Full OpenAI-compatible completions or chat/completions URL.
    pub url: String,
    pub api_key: Option<ApiKey>,
    pub model: String,
    pub cloud: bool,
    pub priority: u32,
    pub timeout: Duration,
    pub idle_timeout: Duration,
}

impl Endpoint {
    pub fn from_config(config: &LlmConfig) -> [Self; 2] {
        [
            Self {
                id: "fast_router_7b".into(),
                url: format!(
                    "{}/completions",
                    config.fast_router_url.trim_end_matches('/')
                ),
                api_key: None,
                model: config.fast_router_model.clone(),
                cloud: false,
                priority: 1,
                timeout: Duration::from_millis(config.fast_router_timeout_ms),
                idle_timeout: Duration::from_secs(3),
            },
            Self {
                id: "reasoner_cloud".into(),
                url: format!(
                    "{}/chat/completions",
                    config.cloud_gateway_url.trim_end_matches('/')
                ),
                api_key: None,
                model: config.cloud_model.clone(),
                cloud: true,
                priority: 2,
                timeout: Duration::from_millis(config.cloud_timeout_ms),
                idle_timeout: Duration::from_secs(25),
            },
        ]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("cloud consent denied")]
    Consent,
    #[error("generation cancelled")]
    Cancelled,
    #[error("endpoint deadline exceeded")]
    Timeout,
    #[error("stream idle deadline exceeded")]
    Idle,
    #[error("endpoint returned HTTP {0}")]
    Status(reqwest::StatusCode),
    #[error("HTTP transport failure")]
    Http(#[from] reqwest::Error),
    #[error("invalid or incomplete completion stream")]
    Protocol,
    #[error("no available endpoint")]
    Unavailable,
    #[error("I/O failure: {0}")]
    Io(#[from] io::Error),
}

impl LlmError {
    pub(crate) fn retryable(&self) -> bool {
        match self {
            Self::Status(s) => s.is_server_error() || matches!(s.as_u16(), 408 | 429),
            Self::Http(e) => !e.is_builder(),
            Self::Timeout | Self::Idle | Self::Protocol => true,
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct OpenAiClient {
    http: reqwest::Client,
    pub(crate) gate: Arc<CloudConsentGate>,
}

#[derive(Deserialize)]
struct Frame {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}
#[derive(Deserialize)]
struct Usage {
    completion_tokens: u64,
}
#[derive(Deserialize)]
struct Choice {
    delta: Option<Delta>,
    text: Option<String>,
}
#[derive(Deserialize)]
struct Delta {
    content: Option<String>,
}

impl OpenAiClient {
    pub fn new(gate: Arc<CloudConsentGate>) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .no_proxy()
            .build()?;
        Ok(Self { http, gate })
    }

    /// Deliver visible deltas as they arrive. Returns completion tokens if reported.
    pub async fn stream(
        &self,
        endpoint: &Endpoint,
        prompt: &str,
        max_tokens: u32,
        cancel: &CancellationToken,
        mut emit: impl FnMut(&str),
    ) -> Result<Option<u64>, LlmError> {
        let check = || {
            if cancel.is_cancelled() {
                Err(LlmError::Cancelled)
            } else if endpoint.cloud && !self.gate.is_allowed() {
                Err(LlmError::Consent)
            } else {
                Ok(())
            }
        };
        check()?;
        let request = async {
            let mut body = json!({"model":endpoint.model,"stream":true,"max_tokens":max_tokens,
                "stream_options":{"include_usage":true}});
            if endpoint.url.ends_with("/chat/completions") {
                body["messages"] = json!([{"role":"user","content":prompt}]);
            } else {
                body["prompt"] = json!(prompt);
            }
            let mut post = self.http.post(&endpoint.url).json(&body);
            if let Some(key) = &endpoint.api_key
                && !key.0.is_empty()
            {
                post = post.bearer_auth(&key.0);
            }
            let send = post.send();
            tokio::pin!(send);
            let mut response = std::future::poll_fn(|cx| {
                let _admission = endpoint.cloud.then(|| self.gate.admit()).transpose()?;
                send.as_mut().poll(cx).map_err(LlmError::Http)
            })
            .await?;
            if !response.status().is_success() {
                return Err(LlmError::Status(response.status()));
            }
            let chunks = async_stream::try_stream! {
                while let Some(chunk) = timeout(endpoint.idle_timeout, response.chunk())
                    .await.map_err(|_| LlmError::Idle)?? {
                    yield chunk;
                }
            };
            let events = chunks.eventsource();
            tokio::pin!(events);
            let mut stripper = ThinkTagStripper::default();
            let mut tokens = None;
            while let Some(event) = events.next().await {
                check()?;
                let data = event
                    .map_err(|error| match error {
                        EventStreamError::Transport(error) => error,
                        _ => LlmError::Protocol,
                    })?
                    .data;
                if data.trim() == "[DONE]" {
                    let tail = stripper.finish();
                    if !tail.is_empty() {
                        emit(&tail);
                    }
                    return Ok(tokens);
                }
                let frame: Frame = serde_json::from_str(&data).map_err(|_| LlmError::Protocol)?;
                if let Some(usage) = frame.usage {
                    tokens = Some(usage.completion_tokens);
                }
                if let Some(delta) = frame
                    .choices
                    .into_iter()
                    .next()
                    .and_then(|c| c.text.or_else(|| c.delta.and_then(|d| d.content)))
                {
                    let visible = stripper.push(&delta);
                    if !visible.is_empty() {
                        emit(&visible);
                    }
                }
            }
            Err(LlmError::Protocol)
        };
        tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(LlmError::Cancelled),
            _ = self.gate.withdrawn.cancelled(), if endpoint.cloud => Err(LlmError::Consent),
            result = timeout(endpoint.timeout, request) => result.map_err(|_| LlmError::Timeout)?,
        }
    }
}
