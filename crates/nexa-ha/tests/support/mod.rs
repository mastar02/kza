use std::{
    future::Future,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};

use futures_util::{SinkExt, StreamExt};
use nexa_config::HomeAssistantConfig;
use nexa_ha::{HaClient, HaError};
use serde_json::{Value, json};
use tokio::{
    io,
    net::{TcpListener, TcpStream},
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_tungstenite::{WebSocketStream, accept_async, tungstenite::Message};
use wiremock::{
    Mock, MockBuilder, MockServer, ResponseTemplate,
    matchers::{header, method, path},
};

pub const TOKEN: &str = "synthetic-ha-test-token";
pub const UPDATED: &str = "2026-11-25T00:00:00Z";
const WALL_TEST_LIMIT: Duration = Duration::from_secs(12);

pub fn config(url: String) -> HomeAssistantConfig {
    HomeAssistantConfig {
        url,
        timeout_ms: 2_000,
        state_confirmation_timeout_ms: 2_000,
        cache_max_staleness_s: 5,
        full_refresh_interval_s: 300,
        unavailable_precheck_enabled: true,
        weather_entity: "weather.test".into(),
    }
}

pub fn entity(id: &str, state: &str, attributes: Value) -> Value {
    json!({"entity_id": id, "state": state, "attributes": attributes, "last_updated": UPDATED})
}

pub fn response(status: u16) -> ResponseTemplate {
    ResponseTemplate::new(status).insert_header("connection", "close")
}

pub fn authenticated(verb: &str, route: &str) -> MockBuilder {
    Mock::given(method(verb))
        .and(path(route))
        .and(header("authorization", format!("Bearer {TOKEN}")))
}

// Keep the virtual clock from auto-advancing while a real socket is waiting
// on the OS or wiremock's server thread. Tests advance it explicitly.
pub async fn paused(case: impl Future<Output = ()>) {
    tokio::time::pause();
    let start = std::time::Instant::now();
    tokio::select! {
        () = case => {}
        () = async {
            while start.elapsed() < WALL_TEST_LIMIT {
                tokio::task::yield_now().await;
            }
        } => panic!("loopback integration test did not progress")
    }
}

pub async fn wait_for(mut ready: impl FnMut() -> bool) {
    while !ready() {
        tokio::task::yield_now().await;
    }
}

pub struct StateSync(JoinHandle<Result<(), HaError>>);

impl StateSync {
    pub fn start(client: &HaClient) -> Self {
        let client = client.clone();
        Self(tokio::spawn(async move { client.run_state_sync().await }))
    }
}

impl Drop for StateSync {
    fn drop(&mut self) {
        self.0.abort();
    }
}

struct Hold {
    route: String,
    reached: oneshot::Sender<()>,
    resume: oneshot::Receiver<()>,
}

type Peer = WebSocketStream<TcpStream>;

/// Single loopback origin. HTTP traffic goes to wiremock and WebSocket traffic
/// uses a real tungstenite server. A gate can pause one HTTP response in transit.
pub struct HaOrigin {
    pub http: MockServer,
    pub url: String,
    pub sessions: mpsc::UnboundedReceiver<Peer>,
    holds: Arc<Mutex<Vec<Hold>>>,
    listener: JoinHandle<()>,
}

impl HaOrigin {
    pub async fn start() -> Self {
        let http = MockServer::start().await;
        let backend = *http.address();
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("loopback listener");
        let url = format!(
            "http://{}",
            listener.local_addr().expect("listener address")
        );
        let (connected, sessions) = mpsc::unbounded_channel();
        let holds = Arc::new(Mutex::new(Vec::new()));
        let pending_holds = Arc::clone(&holds);
        let listener = tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                tokio::spawn(route(
                    socket,
                    backend,
                    connected.clone(),
                    Arc::clone(&pending_holds),
                ));
            }
        });
        Self {
            http,
            url,
            sessions,
            holds,
            listener,
        }
    }

    pub fn hold_next(&self, route: &str) -> (oneshot::Receiver<()>, oneshot::Sender<()>) {
        let (reached, arrived) = oneshot::channel();
        let (release, resume) = oneshot::channel();
        self.holds.lock().expect("hold lock").push(Hold {
            route: route.to_owned(),
            reached,
            resume,
        });
        (arrived, release)
    }
}

impl Drop for HaOrigin {
    fn drop(&mut self) {
        self.listener.abort();
    }
}

async fn route(
    mut socket: TcpStream,
    backend: SocketAddr,
    connected: mpsc::UnboundedSender<Peer>,
    holds: Arc<Mutex<Vec<Hold>>>,
) -> io::Result<()> {
    let mut prefix = [0; 1024];
    let target = loop {
        let size = socket.peek(&mut prefix).await?;
        if size == 0 {
            return Ok(());
        }
        if let Some(end) = prefix[..size].iter().position(|byte| *byte == b'\n') {
            let line = String::from_utf8_lossy(&prefix[..end]);
            break line
                .split_whitespace()
                .nth(1)
                .unwrap_or_default()
                .to_owned();
        }
        assert!(size < prefix.len(), "request line must fit");
        tokio::task::yield_now().await;
    };
    if target == "/api/websocket" {
        let mut ws = accept_async(socket).await.expect("WebSocket handshake");
        send(&mut ws, json!({"type": "auth_required"})).await;
        assert_eq!(
            recv(&mut ws).await,
            json!({"type": "auth", "access_token": TOKEN})
        );
        send(&mut ws, json!({"type": "auth_ok"})).await;
        let subscribe = recv(&mut ws).await;
        assert_eq!(subscribe["type"], "subscribe_events");
        assert_eq!(subscribe["event_type"], "state_changed");
        send(
            &mut ws,
            json!({"id": subscribe["id"], "type": "result", "success": true}),
        )
        .await;
        connected.send(ws).expect("session receiver");
        return Ok(());
    }
    let mut upstream = TcpStream::connect(backend).await?;
    let hold = {
        let mut holds = holds.lock().expect("hold lock");
        holds
            .iter()
            .position(|hold| hold.route == target)
            .map(|index| holds.remove(index))
    };
    if let Some(hold) = hold {
        let (mut from_client, mut to_client) = socket.split();
        let (mut from_backend, mut to_backend) = upstream.split();
        let upload = io::copy(&mut from_client, &mut to_backend);
        let download = async {
            let _ = hold.reached.send(());
            let _ = hold.resume.await;
            io::copy(&mut from_backend, &mut to_client).await
        };
        tokio::select! {
            result = upload => { result?; }
            result = download => { result?; }
        }
    } else {
        io::copy_bidirectional(&mut socket, &mut upstream).await?;
    }
    Ok(())
}

pub async fn delta(ws: &mut Peer, entity_id: &str, state: Value) {
    send(
        ws,
        json!({
            "id": 1,
            "type": "event",
            "event": {
                "event_type": "state_changed", "time_fired": "2026-11-25T00:00:01Z",
                "data": {"entity_id": entity_id, "new_state": state}
            }
        }),
    )
    .await;
}

pub async fn send(ws: &mut Peer, value: Value) {
    ws.send(Message::Text(value.to_string().into()))
        .await
        .expect("send websocket JSON");
}

pub async fn recv(ws: &mut Peer) -> Value {
    let message = ws
        .next()
        .await
        .expect("websocket message")
        .expect("valid websocket message");
    serde_json::from_str(message.to_text().expect("text message")).expect("JSON message")
}
