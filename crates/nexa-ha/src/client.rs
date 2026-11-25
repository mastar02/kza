use std::{fmt, sync::Arc, time::Duration};

use arc_swap::ArcSwap;
use nexa_config::HomeAssistantConfig;
use reqwest::{Client, RequestBuilder, Response, StatusCode, Url, header::HeaderValue};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use thiserror::Error;
use tokio::{sync::Semaphore, time::Instant};

use crate::state::{EntityState, Observation, StateFrame, is_fresh};

pub(crate) const MAX_STATE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum HaError {
    #[error("invalid Home Assistant client configuration")]
    InvalidConfiguration,
    #[error("Home Assistant transport failed or timed out")]
    Transport,
    #[error("Home Assistant returned HTTP {0}")]
    Http(StatusCode),
    #[error("invalid Home Assistant response")]
    InvalidResponse,
    #[error("Home Assistant state sync is already running")]
    AlreadySyncing,
}

/// Read transport and atomically published state projection. Clones share the frame.
#[derive(Clone)]
pub struct HaClient {
    pub(crate) inner: Arc<ClientInner>,
}

pub(crate) struct ClientInner {
    http: Client,
    base: Url,
    authorization: HeaderValue,
    pub ws: Url,
    pub token: String,
    pub timeout: Duration,
    pub refresh_interval: Duration,
    pub unavailable_precheck: bool,
    pub frame: ArcSwap<StateFrame>,
    pub sync_slot: Semaphore,
}

impl fmt::Debug for HaClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("HaClient").finish_non_exhaustive()
    }
}

impl HaClient {
    pub fn new(
        config: &HomeAssistantConfig,
        access_token: impl Into<String>,
    ) -> Result<Self, HaError> {
        let mut base = Url::parse(&config.url).map_err(|_| HaError::InvalidConfiguration)?;
        if !matches!(base.scheme(), "http" | "https")
            || base.host_str().is_none()
            || !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
            || config.timeout_ms == 0
            || config.full_refresh_interval_s == 0
            || config.cache_max_staleness_s == 0
        {
            return Err(HaError::InvalidConfiguration);
        }
        // Keep a trusted reverse-proxy prefix, if one was configured.
        if !base.path().ends_with('/') {
            base.path_segments_mut()
                .map_err(|_| HaError::InvalidConfiguration)?
                .push("");
        }
        let mut ws = base
            .join("api/websocket")
            .map_err(|_| HaError::InvalidConfiguration)?;
        ws.set_scheme(if base.scheme() == "https" {
            "wss"
        } else {
            "ws"
        })
        .map_err(|_| HaError::InvalidConfiguration)?;
        let token = access_token.into();
        if token.is_empty() || token.bytes().any(|byte| !byte.is_ascii_graphic()) {
            return Err(HaError::InvalidConfiguration);
        }
        let mut authorization = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|_| HaError::InvalidConfiguration)?;
        authorization.set_sensitive(true);
        let timeout = Duration::from_millis(config.timeout_ms);
        let refresh_interval = Duration::from_secs(config.full_refresh_interval_s);
        if Instant::now().checked_add(refresh_interval).is_none() {
            return Err(HaError::InvalidConfiguration);
        }
        let http = Client::builder()
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .build()
            .map_err(|_| HaError::InvalidConfiguration)?;
        Ok(Self {
            inner: Arc::new(ClientInner {
                http,
                base,
                authorization,
                ws,
                token,
                timeout,
                refresh_interval,
                unavailable_precheck: config.unavailable_precheck_enabled,
                frame: ArcSwap::from_pointee(StateFrame::empty(Duration::from_secs(
                    config.cache_max_staleness_s,
                ))),
                sync_slot: Semaphore::new(1),
            }),
        })
    }

    pub fn frame(&self) -> Arc<StateFrame> {
        self.inner.frame.load_full()
    }

    fn request(&self, method: reqwest::Method, path: &str) -> Result<RequestBuilder, HaError> {
        let url = self
            .inner
            .base
            .join(path)
            .map_err(|_| HaError::InvalidConfiguration)?;
        Ok(self.inner.http.request(method, url).header(
            reqwest::header::AUTHORIZATION,
            self.inner.authorization.clone(),
        ))
    }

    async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, HaError> {
        let response = send(self.request(reqwest::Method::GET, path)?).await?;
        let mut response = response;
        let mut data = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| HaError::Transport)? {
            if chunk.len() > MAX_STATE_BYTES - data.len() {
                return Err(HaError::InvalidResponse);
            }
            data.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&data).map_err(|_| HaError::InvalidResponse)
    }

    pub(crate) async fn snapshot(&self, as_of: Instant) -> Result<StateFrame, HaError> {
        let rows: Vec<EntityState> = self.get("api/states").await?;
        let mut frame = StateFrame::empty(self.frame().max_staleness());
        frame.snapshot_at = Some(as_of);
        for state in rows {
            if !state.valid() {
                return Err(HaError::InvalidResponse);
            }
            let previous = frame.entries.insert(
                state.entity_id().to_owned(),
                Observation {
                    state: Arc::new(state),
                    as_of,
                },
            );
            if previous.is_some() {
                return Err(HaError::InvalidResponse);
            }
        }
        Ok(frame)
    }

    pub(crate) async fn current_entity(&self, entity_id: &str) -> Result<Observation, HaError> {
        let frame = self.frame();
        // Disabling the cached precheck forces point reads. A fresh HA
        // unavailable response still prevents dispatch.
        if self.inner.unavailable_precheck
            && let Ok(observation) = frame.current(entity_id, Instant::now())
        {
            return Ok(observation.clone());
        }
        let as_of = Instant::now();
        let state: EntityState = self.get(&format!("api/states/{entity_id}")).await?;
        if !state.valid()
            || state.entity_id() != entity_id
            || !is_fresh(as_of, Instant::now(), frame.max_staleness())
        {
            return Err(HaError::InvalidResponse);
        }
        // Prefer state changes observed while the point read was in flight.
        let latest = self.frame();
        if !Arc::ptr_eq(&latest, &frame) {
            if latest
                .deletions
                .get(entity_id)
                .is_some_and(|deleted_at| *deleted_at >= as_of)
                || (latest
                    .snapshot_at
                    .is_some_and(|snapshot_at| snapshot_at > as_of)
                    && !latest.entries.contains_key(entity_id))
            {
                return Err(HaError::Http(StatusCode::NOT_FOUND));
            }
            if let Ok(current) = latest.current(entity_id, Instant::now())
                && current.as_of >= as_of
                && frame
                    .entries
                    .get(entity_id)
                    .is_none_or(|previous| !Arc::ptr_eq(&previous.state, &current.state))
            {
                return Ok(current.clone());
            }
        }
        Ok(Observation {
            state: Arc::new(state),
            as_of,
        })
    }

    pub(crate) async fn temperature_is_celsius(&self) -> Result<bool, HaError> {
        let data: Value = self.get("api/config").await?;
        Ok(data
            .pointer("/unit_system/temperature")
            .and_then(Value::as_str)
            == Some("°C"))
    }

    pub(crate) async fn call_service(
        &self,
        domain: &str,
        service: &str,
        body: &Map<String, Value>,
    ) -> Result<(), HaError> {
        let request = self.request(
            reqwest::Method::POST,
            &format!("api/services/{domain}/{service}"),
        )?;
        send(request.json(body)).await?;
        Ok(())
    }
}

async fn send(request: RequestBuilder) -> Result<Response, HaError> {
    let response = request.send().await.map_err(|_| HaError::Transport)?;
    if !response.status().is_success() {
        return Err(HaError::Http(response.status()));
    }
    Ok(response)
}
