use std::{collections::HashMap, sync::Arc, time::Duration};

use chrono::{DateTime, FixedOffset};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::{
    net::TcpStream,
    sync::mpsc,
    time::{self, Instant, MissedTickBehavior},
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async_with_config,
    tungstenite::{Message, protocol::WebSocketConfig},
};

use crate::{
    client::{HaClient, HaError, MAX_STATE_BYTES},
    state::{Delta, EntityState, Observation, domain},
};

const SUBSCRIPTION_ID: u64 = 1;
const FIRST_PING_ID: u64 = SUBSCRIPTION_ID + 1;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

enum Update {
    Reconcile,
    Changed(Delta),
}

#[derive(Deserialize)]
struct StateChanged {
    event_type: String,
    time_fired: Option<DateTime<FixedOffset>>,
    data: StateChangedData,
}

#[derive(Deserialize)]
struct StateChangedData {
    entity_id: String,
    new_state: NewState,
}

#[derive(Deserialize)]
#[serde(transparent)]
struct NewState(Option<EntityState>);

impl HaClient {
    /// Drive periodic snapshots and reconnecting event delivery until cancelled.
    /// Dropping the future also drops the subscription and in-flight snapshot.
    pub async fn run_state_sync(&self) -> Result<(), HaError> {
        let _slot = self
            .inner
            .sync_slot
            .try_acquire()
            .map_err(|_| HaError::AlreadySyncing)?;
        tracing::info!(
            refresh_interval_s = self.inner.refresh_interval.as_secs(),
            "HA state sync started"
        );
        let (events, receiver) = mpsc::channel(128);
        tokio::join!(self.websocket_loop(events), self.publish_loop(receiver));
        Ok(())
    }

    async fn publish_loop(&self, mut updates: mpsc::Receiver<Update>) {
        let period = self.inner.refresh_interval;
        let mut refresh = time::interval_at(Instant::now() + period, period);
        refresh.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let mut reconcile = true;
        loop {
            if !reconcile {
                tokio::select! {
                    _ = refresh.tick() => {}
                    update = updates.recv() => {
                        match update {
                            Some(Update::Changed(delta)) => {
                                self.publish_delta(&delta);
                                continue;
                            }
                            Some(Update::Reconcile) => {}
                            None => return,
                        }
                    }
                }
            }
            reconcile = false;
            refresh.reset();
            let mut changes = HashMap::new();
            let fetch = self.snapshot(Instant::now());
            tokio::pin!(fetch);
            let result = loop {
                tokio::select! {
                    result = &mut fetch => break result,
                    update = updates.recv() => match update {
                        Some(Update::Changed(delta)) => {
                            if self.publish_delta(&delta) {
                                changes.insert(delta.entity_id.clone(), delta);
                            }
                        }
                        Some(Update::Reconcile) => reconcile = true,
                        None => return,
                    }
                }
            };
            match result {
                Ok(mut frame) => {
                    // An in-flight snapshot must preserve newer upserts and deletions.
                    for delta in changes.into_values() {
                        delta.apply(&mut frame);
                    }
                    let entity_count = frame.entries.len();
                    self.inner.frame.store(Arc::new(frame));
                    tracing::info!(entity_count, "HA full state refresh published");
                }
                Err(error) => tracing::warn!(%error, "HA state snapshot failed"),
            }
        }
    }

    fn publish_delta(&self, delta: &Delta) -> bool {
        let mut frame = (*self.frame()).clone();
        if !delta.apply(&mut frame) {
            return false;
        }
        self.inner.frame.store(Arc::new(frame));
        true
    }

    async fn websocket_loop(&self, updates: mpsc::Sender<Update>) {
        let mut backoff = INITIAL_BACKOFF;
        loop {
            match time::timeout(self.inner.timeout, self.subscribe()).await {
                Ok(Ok(mut socket)) => {
                    tracing::info!(event = "subscribed", "HA event stream connected");
                    if updates.send(Update::Reconcile).await.is_err() {
                        return;
                    }
                    backoff = INITIAL_BACKOFF;
                    if let Err(error) = read_events(&mut socket, &updates, self.inner.timeout).await
                    {
                        tracing::warn!(%error, "HA event stream disconnected");
                    }
                    if updates.send(Update::Reconcile).await.is_err() {
                        return;
                    }
                }
                Ok(Err(error)) => tracing::warn!(%error, "HA WebSocket subscription failed"),
                Err(_) => tracing::warn!(
                    timeout_ms = self.inner.timeout.as_millis(),
                    "HA WebSocket subscription timed out"
                ),
            }
            tracing::info!(
                retry_delay_ms = backoff.as_millis(),
                "HA WebSocket reconnect scheduled"
            );
            time::sleep(backoff).await;
            backoff = (backoff * 2).min(MAX_BACKOFF);
        }
    }

    async fn subscribe(&self) -> Result<Socket, HaError> {
        let config = WebSocketConfig::default()
            .max_message_size(Some(MAX_STATE_BYTES))
            .max_frame_size(Some(MAX_STATE_BYTES));
        let (mut socket, _) = connect_async_with_config(self.inner.ws.as_str(), Some(config), true)
            .await
            .map_err(|_| HaError::Transport)?;
        let greeting = read_message(&mut socket).await?;
        if greeting["type"] != "auth_required" {
            return Err(HaError::InvalidResponse);
        }
        write_message(
            &mut socket,
            json!({"type": "auth", "access_token": self.inner.token}),
        )
        .await?;
        let auth = read_message(&mut socket).await?;
        if auth["type"] != "auth_ok" {
            return Err(HaError::InvalidResponse);
        }
        write_message(
            &mut socket,
            json!({"id": SUBSCRIPTION_ID, "type": "subscribe_events", "event_type": "state_changed"}),
        ).await?;
        let subscription = read_message(&mut socket).await?;
        if subscription["type"] != "result"
            || subscription["id"] != SUBSCRIPTION_ID
            || subscription["success"] != true
        {
            return Err(HaError::InvalidResponse);
        }
        Ok(socket)
    }
}

async fn read_events(
    socket: &mut Socket,
    updates: &mpsc::Sender<Update>,
    timeout: Duration,
) -> Result<(), HaError> {
    let mut heartbeat = time::interval_at(Instant::now() + HEARTBEAT_INTERVAL, HEARTBEAT_INTERVAL);
    heartbeat.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut ping_id = FIRST_PING_ID;
    let mut pong_deadline: Option<std::pin::Pin<Box<time::Sleep>>> = None;
    loop {
        tokio::select! {
            _ = heartbeat.tick(), if pong_deadline.is_none() => {
                time::timeout(timeout, write_message(socket, json!({"id": ping_id, "type": "ping"})))
                    .await
                    .map_err(|_| HaError::Transport)??;
                pong_deadline = Some(Box::pin(time::sleep(timeout)));
            }
            _ = async {
                if let Some(deadline) = &mut pong_deadline {
                    deadline.as_mut().await;
                }
            }, if pong_deadline.is_some() => return Err(HaError::Transport),
            message = read_message(socket) => {
                let message = message?;
                if message["type"] == "pong" && message["id"] == ping_id && pong_deadline.is_some() {
                    pong_deadline = None;
                    ping_id = ping_id.checked_add(1).ok_or(HaError::InvalidResponse)?;
                } else if message["type"] == "event" && message["id"] == SUBSCRIPTION_ID {
                    let as_of = Instant::now();
                    let event: StateChanged = serde_json::from_value(message["event"].clone())
                        .map_err(|_| HaError::InvalidResponse)?;
                    if event.event_type != "state_changed" || domain(&event.data.entity_id).is_none() {
                        return Err(HaError::InvalidResponse);
                    }
                    let state = event.data.new_state.0.map(|state| Observation { state: Arc::new(state), as_of });
                    if state.as_ref().is_some_and(|observation| {
                        !observation.state.valid() || observation.state.entity_id() != event.data.entity_id
                    }) {
                        return Err(HaError::InvalidResponse);
                    }
                    let delta = Delta { entity_id: event.data.entity_id, state, fired_at: event.time_fired, as_of };
                    updates.send(Update::Changed(delta)).await.map_err(|_| HaError::Transport)?;
                }
            }
        }
    }
}

async fn read_message(socket: &mut Socket) -> Result<Value, HaError> {
    loop {
        match socket.next().await {
            Some(Ok(Message::Text(text))) => {
                return serde_json::from_str(&text).map_err(|_| HaError::InvalidResponse);
            }
            Some(Ok(Message::Ping(_))) => socket.flush().await.map_err(|_| HaError::Transport)?,
            Some(Ok(Message::Pong(_))) => {}
            Some(Ok(Message::Close(_)) | Err(_)) | None => return Err(HaError::Transport),
            Some(Ok(_)) => return Err(HaError::InvalidResponse),
        }
    }
}

async fn write_message(socket: &mut Socket, message: Value) -> Result<(), HaError> {
    socket
        .send(Message::Text(message.to_string().into()))
        .await
        .map_err(|_| HaError::Transport)
}
