use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use eventsource_stream::{EventStreamError, Eventsource};
use futures_util::StreamExt;
use nexa_config::LlmConfig;
use serde::Deserialize;
use serde_json::json;
use tokio::time::{Instant, timeout, timeout_at};

use crate::{CancellationToken, CloudConsentGate, ThinkTagStripper};

const SYSTEM_PROMPT: &str =
    "Sos Nexa, un asistente doméstico. Respondé en español, con claridad y brevedad.";
const LOCAL_STREAM_BUDGET: Duration = Duration::from_secs(25);
const CLOUD_STREAM_BUDGET: Duration = Duration::from_secs(60);
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(5);

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
    /// Full OpenAI-compatible chat/completions URL.
    pub url: String,
    pub api_key: Option<ApiKey>,
    pub model: String,
    pub cloud: bool,
    pub priority: u32,
    /// Time from starting the request to receiving the first generated content.
    pub timeout: Duration,
    pub idle_timeout: Duration,
}

impl Endpoint {
    pub fn from_config(config: &LlmConfig) -> [Self; 2] {
        [
            Self {
                id: "fast_router_7b".into(),
                url: format!(
                    "{}/chat/completions",
                    config.fast_router_url.trim_end_matches('/')
                ),
                api_key: None,
                model: config.fast_router_model.clone(),
                cloud: false,
                priority: 1,
                timeout: Duration::from_millis(config.fast_router_timeout_ms),
                idle_timeout: STREAM_IDLE_TIMEOUT,
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
                idle_timeout: STREAM_IDLE_TIMEOUT,
            },
        ]
    }

    fn stream_budget(&self) -> Duration {
        self.timeout.max(if self.cloud {
            CLOUD_STREAM_BUDGET
        } else {
            LOCAL_STREAM_BUDGET
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("cloud consent denied")]
    Consent,
    #[error("generation cancelled")]
    Cancelled,
    #[error("first token deadline exceeded")]
    Timeout,
    #[error("total stream deadline exceeded")]
    StreamTimeout,
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
            Self::Timeout | Self::StreamTimeout | Self::Idle | Self::Protocol => true,
            _ => false,
        }
    }

    pub(crate) fn affects_cooldown(&self) -> bool {
        match self {
            Self::Status(status) => status.is_server_error(),
            Self::Http(error) => error.is_connect(),
            Self::Timeout => true,
            _ => false,
        }
    }

    /// A bounded status label safe to log without including provider details.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Consent => "consent_denied",
            Self::Cancelled => "cancelled",
            Self::Timeout => "first_token_timeout",
            Self::StreamTimeout => "stream_timeout",
            Self::Idle => "idle_timeout",
            Self::Status(_) => "http_status",
            Self::Http(_) => "http_transport",
            Self::Protocol => "protocol",
            Self::Unavailable => "unavailable",
            Self::Io(_) => "io",
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
        emit: impl FnMut(&str),
    ) -> Result<Option<u64>, LlmError> {
        let (result, _) = self
            .stream_attempt(endpoint, prompt, max_tokens, cancel, emit)
            .await;
        result
    }

    pub(crate) async fn stream_attempt(
        &self,
        endpoint: &Endpoint,
        prompt: &str,
        max_tokens: u32,
        cancel: &CancellationToken,
        emit: impl FnMut(&str),
    ) -> (Result<Option<u64>, LlmError>, bool) {
        self.stream_attempt_with(
            endpoint,
            SYSTEM_PROMPT,
            prompt,
            max_tokens,
            None,
            cancel,
            emit,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn stream_attempt_with(
        &self,
        endpoint: &Endpoint,
        system_prompt: &str,
        prompt: &str,
        max_tokens: u32,
        temperature: Option<f32>,
        cancel: &CancellationToken,
        mut emit: impl FnMut(&str),
    ) -> (Result<Option<u64>, LlmError>, bool) {
        let check = || {
            if cancel.is_cancelled() {
                Err(LlmError::Cancelled)
            } else if endpoint.cloud && !self.gate.is_allowed() {
                Err(LlmError::Consent)
            } else {
                Ok(())
            }
        };
        // Shared by the HTTP chunk reader and SSE decoder within this attempt.
        let first_token = AtomicBool::new(false);
        let started = Instant::now();
        let first_token_deadline = started + endpoint.timeout;
        let request = async {
            check()?;
            let mut body = json!({
                "model": endpoint.model,
                "stream": true,
                "max_tokens": max_tokens,
                "stream_options": {"include_usage": true},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            });
            if let Some(temp) = temperature {
                body["temperature"] = json!(temp);
            }
            if !endpoint.cloud {
                // Disable slow reasoning blocks on Qwen3 / Qwen3.5 for real-time voice turns.
                body["chat_template_kwargs"] = json!({"enable_thinking": false});
            }
            let mut post = self.http.post(&endpoint.url).json(&body);
            if let Some(key) = &endpoint.api_key
                && !key.0.is_empty()
            {
                post = post.bearer_auth(&key.0);
            }
            let send = post.send();
            tokio::pin!(send);
            let gated_send = std::future::poll_fn(|cx| {
                let _admission = endpoint.cloud.then(|| self.gate.admit()).transpose()?;
                send.as_mut().poll(cx).map_err(LlmError::Http)
            });
            let mut response = timeout_at(first_token_deadline, gated_send)
                .await
                .map_err(|_| LlmError::Timeout)??;
            if !response.status().is_success() {
                return Err(LlmError::Status(response.status()));
            }
            let chunk_has_started = &first_token;
            let chunks = async_stream::try_stream! {
                loop {
                    let chunk = if chunk_has_started.load(Ordering::Relaxed) {
                        timeout(endpoint.idle_timeout, response.chunk())
                            .await.map_err(|_| LlmError::Idle)??
                    } else {
                        response.chunk().await?
                    };
                    let Some(chunk) = chunk else { break };
                    yield chunk;
                }
            };
            let events = chunks.eventsource();
            tokio::pin!(events);
            let mut stripper = ThinkTagStripper::default();
            let mut tokens = None;
            loop {
                let event = if first_token.load(Ordering::Relaxed) {
                    events.next().await
                } else {
                    timeout_at(first_token_deadline, events.next())
                        .await
                        .map_err(|_| LlmError::Timeout)?
                };
                let Some(event) = event else {
                    return Err(LlmError::Protocol);
                };
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
                    .and_then(|choice| choice.delta?.content)
                    .filter(|content| !content.is_empty())
                {
                    first_token.store(true, Ordering::Relaxed);
                    let visible = stripper.push(&delta);
                    if !visible.is_empty() {
                        emit(&visible);
                    }
                }
            }
        };
        let result = tokio::select! {
            biased;
            _ = cancel.cancelled() => Err(LlmError::Cancelled),
            _ = self.gate.withdrawn.cancelled(), if endpoint.cloud => Err(LlmError::Consent),
            result = timeout_at(started + endpoint.stream_budget(), request) => {
                result.unwrap_or_else(|_| {
                    if first_token.load(Ordering::Relaxed) {
                        Err(LlmError::StreamTimeout)
                    } else {
                        Err(LlmError::Timeout)
                    }
                })
            },
        };
        (result, first_token.load(Ordering::Relaxed))
    }
}
