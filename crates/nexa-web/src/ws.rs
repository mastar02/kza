//! Browser protocol v3. Private frames and REST receipts share a current lease.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    Router,
    extract::{
        State, WebSocketUpgrade,
        ws::{CloseFrame, Message, WebSocket},
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
};
use futures_util::StreamExt;
use nexa_proto::v1::{
    LiveEventKind, LiveRuntimeEvent,
    Permission::{self, *},
    Role, SubscribeLiveEventsRequest,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::{
    sync::watch,
    time::{Instant, interval, sleep_until, timeout},
};
use uuid::Uuid;

use crate::{
    AppState,
    auth::{self, AuthError, Lease, Presented},
    routes::ApiError,
};

const MAX_CONNECTIONS: usize = 32;
const MAX_RECEIPTS: usize = 128;
const HANDSHAKE: Duration = Duration::from_secs(5);
const SEND_DEADLINE: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Channel {
    Conversations,
    VoiceCaptures,
    VoiceProfiles,
    VoiceReview,
    Evaluation,
}

impl Channel {
    fn name(self) -> &'static str {
        match self {
            Self::Conversations => "conversations",
            Self::VoiceCaptures => "voices:captures",
            Self::VoiceProfiles => "voices:profiles",
            Self::VoiceReview => "voices:review",
            Self::Evaluation => "evaluation:sample",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Live,
    Voices,
    Evaluation,
}

impl Mode {
    fn permissions(self) -> &'static [Permission] {
        match self {
            Self::Live => &[ConversationsRead],
            Self::Voices => &[VoicesRead, VoicesReview],
            Self::Evaluation => &[EvaluationRead],
        }
    }

    fn permits(self, channel: Channel) -> bool {
        match channel {
            Channel::Conversations => self == Self::Live,
            Channel::Evaluation => self == Self::Evaluation,
            _ => self != Self::Evaluation,
        }
    }

    fn matches_role(self, lease: &Lease) -> bool {
        self != Self::Evaluation || lease.actor().role == Role::Evaluator as i32
    }

    fn ready_message(self, connection: Uuid) -> Value {
        match self {
            Self::Live => {
                json!({"type":"rest_ready","connection_id":connection.to_string(),"ts":""})
            }
            Self::Voices => json!({"type":"rest_ready","connection_id":connection.to_string(),
                "channels":["voices:captures","voices:profiles","voices:review"],"ts":""}),
            Self::Evaluation => json!({"type":"rest_ready","connection_id":connection.to_string(),
                "channels":["evaluation:sample"],"ts":""}),
        }
    }
}

struct Receipt {
    ticket: Uuid,
    nonce: Uuid,
    event: Uuid,
    channel: Channel,
    rest: bool,
    acknowledged: bool,
}

impl Receipt {
    fn new(channel: Channel, rest: bool) -> Self {
        Self {
            ticket: Uuid::new_v4(),
            nonce: Uuid::new_v4(),
            event: Uuid::new_v4(),
            channel,
            rest,
            acknowledged: false,
        }
    }

    fn public(&self) -> Value {
        json!({"ticket":self.ticket.to_string(),"nonce":self.nonce.to_string()})
    }
}

struct Connection {
    lease_id: String,
    epoch: u64,
    mode: Mode,
    ready: bool,
    retired: bool,
    fenced: bool,
    receipts: Vec<Receipt>,
}

impl Connection {
    fn bound(&self, lease: &Lease) -> bool {
        let actor = lease.actor();
        self.lease_id == actor.lease_id && self.epoch == actor.lease_epoch
    }
}

#[derive(Clone)]
pub(crate) struct Connections {
    states: Arc<Mutex<HashMap<Uuid, Connection>>>,
    fences: watch::Sender<()>,
}

impl Default for Connections {
    fn default() -> Self {
        Self {
            states: Arc::default(),
            fences: watch::channel(()).0,
        }
    }
}

impl Connections {
    pub(crate) fn fence(&self) {
        if let Ok(mut states) = self.states.lock() {
            for state in states.values_mut() {
                state.retired = true;
                state.fenced = true;
            }
            self.fences.send_replace(());
        }
    }

    fn open(&self, mode: Mode, lease: &Lease) -> Option<Registration> {
        let mut states = self.states.lock().ok()?;
        if states.len() >= MAX_CONNECTIONS {
            return None;
        }
        let id = Uuid::new_v4();
        let actor = lease.actor();
        states.insert(
            id,
            Connection {
                lease_id: actor.lease_id,
                epoch: actor.lease_epoch,
                mode,
                ready: false,
                retired: false,
                fenced: false,
                receipts: Vec::new(),
            },
        );
        Some(Registration {
            id,
            states: self.clone(),
        })
    }

    fn with<T>(&self, id: Uuid, work: impl FnOnce(&mut Connection) -> Option<T>) -> Option<T> {
        let mut states = self.states.lock().ok()?;
        work(states.get_mut(&id)?)
    }

    pub(crate) fn authorize(
        &self,
        headers: &HeaderMap,
        lease: &Lease,
        channel: Channel,
    ) -> Result<Uuid, ApiError> {
        let mut values = headers.get_all("x-kza-live-connection").iter();
        let id = values
            .next()
            .and_then(|value| value.to_str().ok())
            .and_then(canonical_uuid)
            .ok_or(ApiError(
                StatusCode::PRECONDITION_REQUIRED,
                "live_receipts_required",
            ))?;
        if values.next().is_some() {
            return Err(ApiError(
                StatusCode::PRECONDITION_REQUIRED,
                "live_receipts_required",
            ));
        }
        self.with(id, |state| {
            (state.ready && !state.retired && state.bound(lease) && state.mode.permits(channel))
                .then_some(id)
        })
        .ok_or(ApiError(
            StatusCode::PRECONDITION_REQUIRED,
            "live_receipts_required",
        ))
    }

    pub(crate) fn projection(
        &self,
        id: Uuid,
        channel: Channel,
        value: Value,
    ) -> Result<Value, ApiError> {
        let (rows, next_cursor) = if value.is_array() {
            (value, Value::Null)
        } else {
            let rows = value
                .get("rows")
                .filter(|rows| rows.is_array())
                .cloned()
                .ok_or(ApiError(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "live_projection_unavailable",
                ))?;
            (rows, value["next_cursor"].clone())
        };
        let limit = match channel {
            Channel::Conversations => 200,
            Channel::VoiceReview | Channel::Evaluation => 1,
            _ => 50,
        };
        if rows.as_array().is_none_or(|rows| rows.len() > limit) {
            return Err(ApiError(
                StatusCode::SERVICE_UNAVAILABLE,
                "live_projection_unavailable",
            ));
        }
        self.with(id, |state| {
            if !state.ready || state.retired || !state.mode.permits(channel) {
                return None;
            }
            let snapshot = if rows.as_array().is_some_and(Vec::is_empty) {
                Value::Null
            } else {
                let receipt = Receipt::new(channel, true);
                let result = json!({"event_id":receipt.event.to_string(),"observed_at":auth::now(),
                    "expires_at":null,"receipt":receipt.public(),"channel":channel.name(),
                    "rows":rows,"next_cursor":next_cursor});
                if state.receipts.len() >= MAX_RECEIPTS {
                    state.retired = true;
                    return None;
                }
                state.receipts.push(receipt);
                result
            };
            Some(json!({"connection_id":id.to_string(),"snapshot":snapshot,"channel":channel.name(),"next_cursor":next_cursor}))
        }).ok_or(ApiError(StatusCode::PRECONDITION_REQUIRED,"live_receipts_required"))
    }
}

struct Registration {
    id: Uuid,
    states: Connections,
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Ok(mut states) = self.states.states.lock() {
            states.remove(&self.id);
        }
    }
}

impl Registration {
    fn fenced(&self) -> bool {
        self.states
            .with(self.id, |state| Some(state.fenced))
            .unwrap_or(false)
    }

    fn current(&self, lease: &Lease) -> bool {
        self.states
            .with(self.id, |state| Some(!state.retired && state.bound(lease)))
            .unwrap_or(false)
    }

    fn ready(&self) -> bool {
        self.states
            .with(self.id, |state| Some(state.ready && !state.retired))
            .unwrap_or(false)
    }

    fn retire(&self) -> Vec<Value> {
        self.states.with(self.id, |state| {
            state.retired = true;
            let mut frames = Vec::new();
            for (channel, rest) in [
                (Channel::Conversations, false), (Channel::Conversations, true),
                (Channel::VoiceCaptures, true), (Channel::VoiceProfiles, true),
                (Channel::VoiceReview, true), (Channel::Evaluation, true),
            ] {
                let matching: Vec<_> = state.receipts.iter()
                    .filter(|receipt| receipt.channel == channel && receipt.rest == rest)
                    .collect();
                for batch in matching.chunks(32) {
                    let ids: Vec<_> = batch.iter().map(|receipt| receipt.event.to_string()).collect();
                    let receipts: Vec<_> = batch.iter().filter(|receipt| !receipt.acknowledged).map(|receipt| {
                        json!({"event_id":receipt.event.to_string(),"ticket":receipt.ticket.to_string(),"nonce":receipt.nonce.to_string()})
                    }).collect();
                    let payload = json!({"event_ids":ids,"receipts":receipts});
                    let frame = if rest {
                        json!({"type":"rest_retire","channel":channel.name(),"ts":"","payload":payload})
                    } else {
                        json!({"type":"retire","ts":"","payload":payload})
                    };
                    frames.push(frame);
                }
            }
            // A reset is sent even when no receipt was issued. It clears in-flight
            // responses and local views before a policy close.
            frames.push(reset_frame(Uuid::new_v4(),Uuid::new_v4()));
            Some(frames)
        }).unwrap_or_default()
    }
}

pub(crate) fn router(state: AppState) -> Router {
    Router::new()
        .route(
            "/ws/live",
            get(
                |State(state): State<AppState>, headers: HeaderMap, upgrade: WebSocketUpgrade| {
                    upgrade_socket(state, headers, upgrade, Mode::Live)
                },
            ),
        )
        .route(
            "/ws/voices",
            get(
                |State(state): State<AppState>, headers: HeaderMap, upgrade: WebSocketUpgrade| {
                    upgrade_socket(state, headers, upgrade, Mode::Voices)
                },
            ),
        )
        .route(
            "/ws/evaluation",
            get(
                |State(state): State<AppState>, headers: HeaderMap, upgrade: WebSocketUpgrade| {
                    upgrade_socket(state, headers, upgrade, Mode::Evaluation)
                },
            ),
        )
        .with_state(state)
}

async fn upgrade_socket(
    state: AppState,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
    mode: Mode,
) -> Result<Response, ApiError> {
    state.config.origin.check(&headers)?;
    let proof = auth::presented(&headers).ok_or(AuthError::Unauthorized)?;
    let changes = state.auth.changes();
    let lease = state.auth.validate(&proof, mode.permissions()).await?;
    if !mode.matches_role(&lease) {
        return Err(AuthError::Forbidden.into());
    }
    let registration = state
        .live
        .open(mode, &lease)
        .ok_or(ApiError(StatusCode::SERVICE_UNAVAILABLE, "live_capacity"))?;
    Ok(upgrade
        .max_message_size(8192)
        .max_frame_size(8192)
        .on_upgrade(move |socket| session(socket, state, proof, registration, mode, changes))
        .into_response())
}

async fn valid(
    state: &AppState,
    proof: &Presented,
    mode: Mode,
    registration: &Registration,
) -> Option<Lease> {
    let lease = state.auth.validate(proof, mode.permissions()).await.ok()?;
    (mode.matches_role(&lease) && registration.current(&lease)).then_some(lease)
}

async fn session(
    mut socket: WebSocket,
    state: AppState,
    proof: Presented,
    registration: Registration,
    mode: Mode,
    mut changes: tokio::sync::watch::Receiver<u64>,
) {
    let mut fences = state.live.fences.subscribe();
    let Some(lease) = valid(&state, &proof, mode, &registration).await else {
        retire(&mut socket, &registration, "lease_retired").await;
        return;
    };
    let connection = state
        .clients
        .brain
        .subscribe_live_events(SubscribeLiveEventsRequest {
            actor: Some(lease.actor()),
        });
    tokio::pin!(connection);
    let mut periodic = interval(Duration::from_millis(200));
    let mut stream = loop {
        tokio::select! {
            biased;
            _ = fences.changed() => {},
            _ = changes.changed() => {},
            _ = periodic.tick() => {},
            result = &mut connection => match result {
                Ok(stream) => break stream,
                Err(_) => {
                    if valid(&state,&proof,mode,&registration).await.is_none() {
                        retire(&mut socket,&registration,"lease_retired").await;
                    } else {
                        close(&mut socket,1013,"live_upstream_unavailable").await;
                    }
                    return;
                }
            },
        }
        if valid(&state, &proof, mode, &registration).await.is_none() {
            retire(&mut socket, &registration, "lease_retired").await;
            return;
        }
    };

    let mut phase = Phase::Hello;
    let deadline = Instant::now() + HANDSHAKE;
    let expires = Instant::now()
        + Duration::from_secs_f64((lease.expires_at() - auth::now()).clamp(0.0, 2_592_000.0));
    let end = loop {
        tokio::select! {
            biased;
            _ = fences.changed() => {
                if registration.fenced() {
                    break "erasure_fence";
                }
            },
            _ = sleep_until(expires) => break "lease_retired",
            _ = changes.changed() => {
                if valid(&state,&proof,mode,&registration).await.is_none() {
                    break "lease_retired";
                }
            },
            _ = periodic.tick() => {
                if valid(&state,&proof,mode,&registration).await.is_none() {
                    break "lease_retired";
                }
            },
            _ = sleep_until(deadline), if !matches!(phase,Phase::Ready) => break "handshake_timeout",
            event = stream.next() => {
                if valid(&state,&proof,mode,&registration).await.is_none() {
                    break "lease_retired";
                }
                let Some(Ok(event)) = event else {
                    break "upstream_retired";
                };
                if event.kind == LiveEventKind::ErasureFence as i32 {
                    state.live.fence();
                    break "erasure_fence";
                }
                if mode == Mode::Live && registration.ready()
                    && let Some(frame) = live_event(&registration,event)
                    && !send(&mut socket,frame).await
                {
                    return;
                }
            },
            frame = socket.recv() => {
                if valid(&state,&proof,mode,&registration).await.is_none() {
                    break "lease_retired";
                }
                match frame {
                    Some(Ok(Message::Text(text))) => {
                        let Ok(command) = serde_json::from_str::<Command>(&text) else {
                            break "invalid_live_protocol";
                        };
                        match handle(&mut phase,command,mode,&registration) {
                            Ok(Some(frame)) => {
                                if valid(&state,&proof,mode,&registration).await.is_none() {
                                    break "lease_retired";
                                }
                                if !send(&mut socket,frame).await {
                                    return;
                                }
                            }
                            Ok(None) => {},
                            Err(()) => break "invalid_live_protocol",
                        }
                    }
                    Some(Ok(Message::Ping(_) | Message::Pong(_))) => {},
                    Some(Ok(Message::Binary(_))) => break "invalid_live_protocol",
                    _ => return,
                }
            }
        }
    };
    retire(&mut socket, &registration, end).await;
}

enum Phase {
    Hello,
    Reset { id: Uuid, nonce: Uuid },
    Ready,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Ack {
    ticket: String,
    nonce: String,
}

#[derive(Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
enum Command {
    #[serde(rename = "live_hello")]
    Hello { version: u8, client_id: String },
    #[serde(rename = "live_reset_ack")]
    ResetAck { id: String, nonce: String },
    #[serde(rename = "live_ack")]
    Ack { receipts: Vec<Ack> },
}

fn canonical_uuid(raw: &str) -> Option<Uuid> {
    let id = Uuid::parse_str(raw).ok()?;
    (id.to_string() == raw && !id.is_nil()).then_some(id)
}

fn handle(
    phase: &mut Phase,
    command: Command,
    mode: Mode,
    registration: &Registration,
) -> Result<Option<Value>, ()> {
    match (&*phase, command) {
        (
            Phase::Hello,
            Command::Hello {
                version: 3,
                client_id,
            },
        ) if canonical_uuid(&client_id).is_some() => {
            let (id, nonce) = (Uuid::new_v4(), Uuid::new_v4());
            *phase = Phase::Reset { id, nonce };
            Ok(Some(reset_frame(id, nonce)))
        }
        (
            Phase::Reset { id, nonce },
            Command::ResetAck {
                id: reply,
                nonce: proof,
            },
        ) if canonical_uuid(&reply) == Some(*id) && canonical_uuid(&proof) == Some(*nonce) => {
            registration
                .states
                .with(registration.id, |state| {
                    if state.retired {
                        return None;
                    }
                    state.ready = true;
                    Some(())
                })
                .ok_or(())?;
            *phase = Phase::Ready;
            Ok(Some(mode.ready_message(registration.id)))
        }
        (Phase::Ready, Command::Ack { receipts }) if (1..=16).contains(&receipts.len()) => {
            registration
                .states
                .with(registration.id, |state| {
                    let mut indices = Vec::new();
                    for ack in receipts {
                        let ticket = canonical_uuid(&ack.ticket)?;
                        let nonce = canonical_uuid(&ack.nonce)?;
                        let index = state
                            .receipts
                            .iter()
                            .position(|known| known.ticket == ticket && known.nonce == nonce)?;
                        if indices.contains(&index) {
                            return None;
                        }
                        indices.push(index);
                    }
                    for index in indices {
                        state.receipts[index].acknowledged = true;
                    }
                    Some(())
                })
                .ok_or(())?;
            Ok(None)
        }
        _ => Err(()),
    }
}

fn reset_frame(id: Uuid, nonce: Uuid) -> Value {
    json!({"type":"live_reset","payload":{"id":id.to_string(),"nonce":nonce.to_string()},"ts":""})
}

fn live_event(registration: &Registration, event: LiveRuntimeEvent) -> Option<Value> {
    let kind = match LiveEventKind::try_from(event.kind).ok()? {
        LiveEventKind::Turn => "turn",
        LiveEventKind::Wake => "wake",
        LiveEventKind::Alert => "alert",
        LiveEventKind::Tts => "tts",
        LiveEventKind::Cooldown => "cooldown",
        LiveEventKind::HaState => "ha_state",
        _ => return None,
    };
    let payload: Value = serde_json::from_str(&event.payload_json).ok()?;
    if !payload.is_object() {
        return None;
    }
    let receipt = Receipt::new(Channel::Conversations, false);
    let result = json!({"type":kind,"event_id":receipt.event.to_string(),
        "payload":payload,"receipt":receipt.public(),"ts":event.emitted_at_unix_ms});
    registration.states.with(registration.id, |state| {
        if state.retired || state.receipts.len() >= MAX_RECEIPTS {
            state.retired = true;
            return None;
        }
        state.receipts.push(receipt);
        Some(result)
    })
}

async fn send(socket: &mut WebSocket, frame: Value) -> bool {
    timeout(
        SEND_DEADLINE,
        socket.send(Message::Text(frame.to_string().into())),
    )
    .await
    .is_ok_and(|result| result.is_ok())
}

async fn close(socket: &mut WebSocket, code: u16, reason: &'static str) {
    let _ = timeout(
        SEND_DEADLINE,
        socket.send(Message::Close(Some(CloseFrame {
            code,
            reason: reason.into(),
        }))),
    )
    .await;
}

async fn retire(socket: &mut WebSocket, registration: &Registration, reason: &'static str) {
    let frames = registration.retire();
    let _ = timeout(SEND_DEADLINE, async {
        for frame in frames {
            socket.send(Message::Text(frame.to_string().into())).await?;
        }
        Ok::<(), axum::Error>(())
    })
    .await;
    let reason = if registration.fenced() {
        "erasure_fence"
    } else {
        reason
    };
    close(socket, 1008, reason).await;
}
