use std::{error::Error, path::PathBuf, sync::Arc, time::Duration};

use nexa_llm::{
    CancellationToken, CloudConsentGate, Endpoint, FailoverRouter, LlmError, OpenAiClient,
    ThinkTagStripper,
};
use proptest::prelude::*;
use serde_json::json;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    time::{advance, pause},
};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{BodyPartialJsonMatcher, body_partial_json, method, path},
};

type Result<T = ()> = std::result::Result<T, Box<dyn Error>>;
const SEC: Duration = Duration::from_secs(1);

fn endpoint(base: &str, cloud: bool) -> Endpoint {
    Endpoint {
        id: if cloud { "cloud" } else { "fast" }.into(),
        url: format!("{base}/v1/chat/completions"),
        api_key: None,
        model: "test".into(),
        cloud,
        priority: if cloud { 2 } else { 1 },
        timeout: 10 * SEC,
        idle_timeout: 3 * SEC,
    }
}

#[tokio::test]
async fn optional_bearer_is_sensitive_in_debug_and_sent_only_when_present() -> Result {
    let client = OpenAiClient::new(Arc::new(CloudConsentGate::new(true)))?;
    for token in [Some("wiremock-secret"), None, Some("")] {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw("data: [DONE]\n\n", "text/event-stream"),
            )
            .expect(1)
            .mount(&server)
            .await;
        let mut ep = endpoint(&server.uri(), false);
        ep.api_key = token.map(|value| value.to_owned().into());
        if token.is_some() {
            let debug = format!("{ep:?}");
            assert!(debug.contains("<redacted>"));
            assert!(!debug.contains("wiremock-secret"));
        }
        client
            .stream(&ep, "hola", 1, &CancellationToken::new(), |_| {})
            .await?;
        let requests = server
            .received_requests()
            .await
            .ok_or("recording disabled")?;
        let auth = requests[0].headers.get("authorization");
        assert_eq!(
            auth.map(|value| value.to_str()).transpose()?,
            token
                .filter(|key| !key.is_empty())
                .map(|_| "Bearer wiremock-secret")
        );
    }
    Ok(())
}

async fn router(eps: Vec<Endpoint>, path: Option<PathBuf>) -> Result<FailoverRouter> {
    Ok(FailoverRouter::new(eps, Arc::new(CloudConsentGate::new(true)), path).await?)
}

fn chat_request(prompt: &str) -> BodyPartialJsonMatcher {
    body_partial_json(json!({
        "model": "test",
        "stream": true,
        "stream_options": {"include_usage": true},
        "messages": [
            {"role": "system", "content": "Sos Nexa, un asistente doméstico. Respondé en español, con claridad y brevedad."},
            {"role": "user", "content": prompt},
        ],
    }))
}

async fn cloud_success(server: &MockServer) {
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
        .and(chat_request("hola"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "data: {\"choices\":[{\"delta\":{\"content\":\"sí<th\"}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"ink>secreto</think>!\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":7}}\n\ndata: [DONE]\n\n", "text/event-stream"))
        .expect(1).mount(server).await;
}

// Real loopback sockets are serviced on an independent runtime by wiremock.
// Keep Tokio from auto-advancing our deadlines while waiting for socket readiness.
fn manual_clock() {
    pause();
    tokio::spawn(async {
        loop {
            tokio::task::yield_now().await;
        }
    });
}

#[tokio::test]
async fn priority_503_failover_then_metrics_expire() -> Result {
    manual_clock();
    let fast = MockServer::start().await;
    let cloud = MockServer::start().await;
    Mock::given(path("/v1/chat/completions"))
        .and(chat_request("hola"))
        .respond_with(ResponseTemplate::new(503))
        .expect(1)
        .mount(&fast)
        .await;
    cloud_success(&cloud).await;
    let router = router(
        vec![endpoint(&cloud.uri(), true), endpoint(&fast.uri(), false)],
        None,
    )
    .await?;
    let mut pieces = Vec::new();
    assert_eq!(
        router
            .stream("hola", 12, &CancellationToken::new(), |s| pieces
                .push(s.to_owned()))
            .await?,
        "cloud"
    );
    assert_eq!(pieces, ["sí", "!"]);
    assert_eq!(router.cooldown_remaining("fast").await, 60 * SEC);
    assert_eq!(
        router
            .metrics("cloud")
            .await
            .ok_or("missing metrics")?
            .completion_tokens,
        7
    );
    advance(301 * SEC).await;
    assert!(router.metrics("cloud").await.is_none());
    Ok(())
}

#[tokio::test]
async fn exponential_cooldown_persistence_and_clear() -> Result {
    manual_clock();
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503))
        .expect(6)
        .mount(&mock)
        .await;
    let tmp = tempfile::tempdir()?;
    let path = Some(tmp.path().join("state/cooldowns.json"));
    let ep = endpoint(&mock.uri(), false);
    let r = router(vec![ep.clone()], path.clone()).await?;
    let cancel = CancellationToken::new();
    assert_eq!(r.consecutive_errors("fast").await, 0);
    assert_eq!(r.consecutive_errors("unknown").await, 0);
    for (failures, seconds) in [(1, 60), (2, 300), (3, 1500), (4, 3600), (5, 3600)] {
        assert!(matches!(
            r.stream("", 1, &cancel, |_| {}).await,
            Err(LlmError::Status(_))
        ));
        assert_eq!(r.consecutive_errors("fast").await, failures);
        assert_eq!(r.cooldown_remaining("fast").await, seconds * SEC);
        assert!(matches!(
            r.stream("", 1, &cancel, |_| {}).await,
            Err(LlmError::Unavailable)
        ));
        advance(seconds * SEC).await;
        assert_eq!(r.consecutive_errors("fast").await, failures);
    }
    drop(r);
    let r = router(vec![ep.clone()], path.clone()).await?;
    assert!(r.cooldown_remaining("fast").await > 3500 * SEC);
    assert_eq!(r.consecutive_errors("fast").await, 5);
    r.clear_cooldown("fast").await?;
    assert_eq!(r.consecutive_errors("fast").await, 0);
    drop(r);
    let r = router(vec![ep], path).await?;
    assert_eq!(r.cooldown_remaining("fast").await, Duration::ZERO);
    assert_eq!(r.consecutive_errors("fast").await, 0);
    assert!(matches!(
        r.stream("", 1, &cancel, |_| {}).await,
        Err(LlmError::Status(_))
    ));
    assert_eq!(r.cooldown_remaining("fast").await, 60 * SEC);
    assert_eq!(r.consecutive_errors("fast").await, 1);
    Ok(())
}

#[tokio::test]
async fn persistence_failure_preserves_failover_and_success() -> Result {
    manual_clock();
    let fast = MockServer::start().await;
    let cloud = MockServer::start().await;
    let failure = Mock::given(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(503))
        .expect(1)
        .mount_as_scoped(&fast)
        .await;
    cloud_success(&cloud).await;
    let tmp = tempfile::tempdir()?;
    let cooldown_path = tmp.path().join("cooldowns.json");
    let router = router(
        vec![endpoint(&fast.uri(), false), endpoint(&cloud.uri(), true)],
        Some(cooldown_path.clone()),
    )
    .await?;
    // Atomic replacement of a directory by the JSON file fails even as root.
    std::fs::create_dir(&cooldown_path)?;
    let cancel = CancellationToken::new();
    let mut answer = String::new();
    assert_eq!(
        router
            .stream("hola", 12, &cancel, |text| answer.push_str(text))
            .await?,
        "cloud"
    );
    assert_eq!(answer, "sí!");
    assert_eq!(router.consecutive_errors("fast").await, 1);
    assert_eq!(router.cooldown_remaining("fast").await, 60 * SEC);

    drop(failure);
    Mock::given(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "data: {\"choices\":[{\"delta\":{\"content\":\"local\"}}]}\n\ndata: [DONE]\n\n",
            "text/event-stream",
        ))
        .expect(2)
        .mount(&fast)
        .await;
    advance(60 * SEC).await;
    answer.clear();
    assert_eq!(
        router
            .stream("hola", 12, &cancel, |text| answer.push_str(text))
            .await?,
        "fast"
    );
    assert_eq!(answer, "local");
    assert_eq!(router.consecutive_errors("fast").await, 0);
    assert_eq!(
        router.metrics("fast").await.ok_or("missing metrics")?.calls,
        1
    );
    assert!(cooldown_path.is_dir());

    std::fs::remove_dir(&cooldown_path)?;
    assert_eq!(router.stream("hola", 12, &cancel, |_| {}).await?, "fast");
    assert_eq!(std::fs::read_to_string(&cooldown_path)?, "{}");
    Ok(())
}

#[tokio::test]
async fn corrupt_cooldowns_are_quarantined_on_startup() -> Result {
    for corrupt in ["{", r#"{"fast":{"failures":"invalid"}}"#] {
        let tmp = tempfile::tempdir()?;
        let path = tmp.path().join("cooldowns.json");
        std::fs::write(&path, corrupt)?;
        let r = router(vec![], Some(path.clone())).await?;
        assert_eq!(r.consecutive_errors("fast").await, 0);
        assert_eq!(r.cooldown_remaining("fast").await, Duration::ZERO);
        assert!(!path.exists());
        assert_eq!(
            std::fs::read_to_string(tmp.path().join("cooldowns.json.corrupt"))?,
            corrupt
        );
    }
    Ok(())
}

#[tokio::test]
async fn corrupt_cooldowns_allow_startup_when_quarantine_fails() -> Result {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("cooldowns.json");
    std::fs::write(&path, "{")?;
    std::fs::create_dir(tmp.path().join("cooldowns.json.corrupt"))?;
    let r = router(vec![], Some(path.clone())).await?;
    assert_eq!(r.consecutive_errors("fast").await, 0);
    assert!(path.is_file());
    Ok(())
}

#[tokio::test]
async fn denied_and_withdrawn_cloud_send_zero_requests() -> Result {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&mock)
        .await;
    for initial in [true, false] {
        let gate = Arc::new(CloudConsentGate::new(initial));
        let client = OpenAiClient::new(gate.clone())?;
        if initial {
            gate.withdraw();
        }
        let error = client
            .stream(
                &endpoint(&mock.uri(), true),
                "private",
                1,
                &CancellationToken::new(),
                |_| {},
            )
            .await;
        assert!(matches!(error, Err(LlmError::Consent)));
    }
    Ok(())
}

#[tokio::test]
async fn ttft_deadline_includes_wait_for_http_headers() -> Result {
    manual_clock();
    let socket = TcpListener::bind(("127.0.0.1", 0)).await?;
    let ep = endpoint(&format!("http://{}", socket.local_addr()?), false);
    let cloud = MockServer::start().await;
    cloud_success(&cloud).await;
    let r = router(vec![ep, endpoint(&cloud.uri(), true)], None).await?;
    let stalled = async {
        let (mut client, _) = socket.accept().await?;
        client.read_u8().await?;
        advance(11 * SEC).await;
        Ok::<_, std::io::Error>(client)
    };
    let cancel = CancellationToken::new();
    let (open, result) = tokio::join!(stalled, r.stream("hola", 12, &cancel, |_| {}));
    let _socket = open?;
    assert_eq!(result?, "cloud");
    assert_eq!(r.consecutive_errors("fast").await, 1);
    Ok(())
}

#[tokio::test]
async fn idle_after_visible_delta_preserves_partial_answer() -> Result {
    manual_clock();
    let socket = TcpListener::bind(("127.0.0.1", 0)).await?;
    let fast = endpoint(&format!("http://{}", socket.local_addr()?), false);
    let cloud = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&cloud)
        .await;
    let r = router(vec![fast, endpoint(&cloud.uri(), true)], None).await?;
    let (seen, delta) = tokio::sync::oneshot::channel();
    let stalled = async {
        let (mut client, _) = socket.accept().await?;
        client.read_u8().await?;
        client.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n").await?;
        delta.await?;
        advance(4 * SEC).await;
        Result::Ok(client)
    };
    let mut seen = Some(seen);
    let cancel = CancellationToken::new();
    let (open, result) = tokio::join!(
        stalled,
        r.stream("", 1, &cancel, |s| {
            assert_eq!(s, "ok");
            if let Some(signal) = seen.take() {
                let _ = signal.send(());
            }
        })
    );
    let _socket = open?;
    assert!(matches!(result, Err(LlmError::Idle)));
    assert_eq!(r.consecutive_errors("fast").await, 0);
    assert_eq!(r.cooldown_remaining("fast").await, Duration::ZERO);
    Ok(())
}

async fn open_sse(listener: &TcpListener) -> std::io::Result<TcpStream> {
    let (mut socket, _) = listener.accept().await?;
    socket.read_u8().await?;
    socket
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
        )
        .await?;
    Ok(socket)
}

async fn send_delta(socket: &mut TcpStream, text: &str) -> std::io::Result<()> {
    let body = json!({"choices": [{"delta": {"content": text}}]});
    socket
        .write_all(format!("data: {body}\n\n").as_bytes())
        .await
}

#[tokio::test]
async fn role_only_frames_do_not_satisfy_first_token_deadline() -> Result {
    manual_clock();
    let socket = TcpListener::bind(("127.0.0.1", 0)).await?;
    let fast = endpoint(&format!("http://{}", socket.local_addr()?), false);
    let r = router(vec![fast], None).await?;
    let server = async {
        let mut connection = open_sse(&socket).await?;
        connection.write_all(b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\n").await?;
        advance(9 * SEC).await;
        assert_eq!(r.consecutive_errors("fast").await, 0);
        advance(2 * SEC).await;
        Result::Ok(connection)
    };
    let cancel = CancellationToken::new();
    let (connection, result) = tokio::join!(server, r.stream("hola", 16, &cancel, |_| {}));
    let _connection = connection?;
    assert!(matches!(result, Err(LlmError::Timeout)));
    assert_eq!(r.consecutive_errors("fast").await, 1);
    Ok(())
}

#[tokio::test]
async fn generation_can_finish_after_first_token_deadline() -> Result {
    manual_clock();
    let socket = TcpListener::bind(("127.0.0.1", 0)).await?;
    let mut fast = endpoint(&format!("http://{}", socket.local_addr()?), false);
    fast.timeout = SEC;
    let r = router(vec![fast], None).await?;
    let (seen, mut delta) = tokio::sync::mpsc::unbounded_channel();
    let server = async {
        let mut connection = open_sse(&socket).await?;
        send_delta(&mut connection, "hola").await?;
        assert_eq!(delta.recv().await.as_deref(), Some("hola"));
        advance(2 * SEC).await;
        send_delta(&mut connection, " mundo").await?;
        connection.write_all(b"data: [DONE]\n\n").await?;
        Result::Ok(connection)
    };
    let cancel = CancellationToken::new();
    let (connection, result) = tokio::join!(
        server,
        r.stream("hola", 16, &cancel, move |text| {
            let _ = seen.send(text.to_owned());
        })
    );
    let _connection = connection?;
    assert_eq!(result?, "fast");
    assert_eq!(delta.recv().await.as_deref(), Some(" mundo"));
    assert_eq!(r.consecutive_errors("fast").await, 0);
    Ok(())
}

#[tokio::test]
async fn total_stream_budget_caps_active_stream_without_cooldown() -> Result {
    manual_clock();
    let socket = TcpListener::bind(("127.0.0.1", 0)).await?;
    let mut fast = endpoint(&format!("http://{}", socket.local_addr()?), false);
    fast.timeout = SEC;
    fast.idle_timeout = 5 * SEC;
    let r = router(vec![fast], None).await?;
    let (seen, mut delta) = tokio::sync::mpsc::unbounded_channel();
    let server = async {
        let mut connection = open_sse(&socket).await?;
        for _ in 0..7 {
            send_delta(&mut connection, "hola").await?;
            assert_eq!(delta.recv().await.as_deref(), Some("hola"));
            advance(4 * SEC).await;
        }
        Result::Ok(connection)
    };
    let cancel = CancellationToken::new();
    let (connection, result) = tokio::join!(
        server,
        r.stream("hola", 16, &cancel, move |text| {
            let _ = seen.send(text.to_owned());
        })
    );
    let _connection = connection?;
    assert!(matches!(result, Err(LlmError::StreamTimeout)));
    assert_eq!(r.consecutive_errors("fast").await, 0);
    assert_eq!(r.cooldown_remaining("fast").await, Duration::ZERO);
    Ok(())
}

#[tokio::test]
async fn cancellation_before_or_after_first_token_never_penalizes_health() -> Result {
    for after_token in [false, true] {
        let socket = TcpListener::bind(("127.0.0.1", 0)).await?;
        let fast = endpoint(&format!("http://{}", socket.local_addr()?), false);
        let r = router(vec![fast], None).await?;
        let cancel = CancellationToken::new();
        let server = async {
            let mut connection = open_sse(&socket).await?;
            if after_token {
                send_delta(&mut connection, "hola").await?;
            } else {
                cancel.cancel();
            }
            Result::Ok(connection)
        };
        let (connection, result) = tokio::join!(
            server,
            r.stream("hola", 16, &cancel, |_| {
                assert!(after_token);
                cancel.cancel();
            })
        );
        let _connection = connection?;
        assert!(matches!(result, Err(LlmError::Cancelled)));
        assert_eq!(r.consecutive_errors("fast").await, 0);
        assert_eq!(r.cooldown_remaining("fast").await, Duration::ZERO);
    }
    Ok(())
}

#[tokio::test]
async fn local_only_never_calls_cloud_and_shares_cooldown() -> Result {
    let local = MockServer::start().await;
    Mock::given(path("/v1/chat/completions"))
        .and(chat_request("contexto privado"))
        .respond_with(ResponseTemplate::new(503))
        .expect(1)
        .mount(&local)
        .await;
    let cloud = MockServer::start().await;
    cloud_success(&cloud).await;
    let r = router(
        vec![endpoint(&local.uri(), false), endpoint(&cloud.uri(), true)],
        None,
    )
    .await?;
    assert!(r.consent().is_allowed());
    let cancel = CancellationToken::new();
    assert!(matches!(
        r.stream_local_only("contexto privado", 12, &cancel, |_| {})
            .await,
        Err(LlmError::Status(reqwest::StatusCode::SERVICE_UNAVAILABLE))
    ));
    assert_eq!(r.consecutive_errors("fast").await, 1);
    assert!(matches!(
        r.stream_local_only("contexto privado", 12, &cancel, |_| {})
            .await,
        Err(LlmError::Unavailable)
    ));
    assert_eq!(r.stream("hola", 12, &cancel, |_| {}).await?, "cloud");
    assert_eq!(
        local
            .received_requests()
            .await
            .ok_or("recording disabled")?
            .len(),
        1
    );
    assert_eq!(
        cloud
            .received_requests()
            .await
            .ok_or("recording disabled")?
            .len(),
        1
    );
    Ok(())
}

#[tokio::test]
async fn retryable_http_429_does_not_penalize_health() -> Result {
    let local = MockServer::start().await;
    Mock::given(path("/v1/chat/completions"))
        .and(chat_request("hola"))
        .respond_with(ResponseTemplate::new(429))
        .expect(1)
        .mount(&local)
        .await;
    let cloud = MockServer::start().await;
    cloud_success(&cloud).await;
    let r = router(
        vec![endpoint(&local.uri(), false), endpoint(&cloud.uri(), true)],
        None,
    )
    .await?;
    assert_eq!(
        r.stream("hola", 12, &CancellationToken::new(), |_| {})
            .await?,
        "cloud"
    );
    assert_eq!(r.consecutive_errors("fast").await, 0);
    Ok(())
}

#[test]
fn ordinary_text_is_emitted_immediately() {
    let mut f = ThinkTagStripper::default();
    assert_eq!(f.push("voz 🙂<thi"), "voz 🙂");
    assert_eq!(f.push("x<"), "<thix");
    assert_eq!(f.finish(), "<");
}

proptest! {
    #[test]
    fn arbitrary_chunk_partitions_match_whole_text(
        parts in prop::collection::vec(prop_oneof!["[^<>]{0,12}",Just("<th".to_owned()),Just("<think>".to_owned()),Just("</think>".to_owned()),Just("<".to_owned()),Just(">".to_owned())],0..48),
        cuts in prop::collection::vec(any::<bool>(),1..64)
    ) {
        let input = parts.concat();
        let mut reference = String::new();
        let mut rest = input.as_str();
        while let Some((visible,hidden)) = rest.split_once("<think>") {
            reference.push_str(visible);
            rest = hidden.split_once("</think>").map_or("",|(_,tail)| tail);
        }
        reference.push_str(rest);
        let mut f = ThinkTagStripper::default();
        let mut actual = String::new();
        let mut chunk = String::new();
        for (c,cut) in input.chars().zip(cuts.iter().cycle()) {
            chunk.push(c);
            if *cut {actual += &f.push(&chunk); chunk.clear();}
        }
        actual += &f.push(&chunk);
        actual += &f.finish();
        prop_assert_eq!(actual, reference);
    }
}

#[tokio::test]
async fn shipped_config_selects_fast_then_cloud_and_starts_closed() -> Result {
    let config = nexa_config::NexaConfig::load_file(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/nexa/nexa.toml"),
    )?;
    let [fast, cloud] = Endpoint::from_config(&config.llm);
    assert_eq!(fast.url, "http://127.0.0.1:8101/v1/chat/completions");
    assert_eq!(cloud.url, "http://127.0.0.1:8200/v1/chat/completions");
    assert!(fast.priority < cloud.priority && !fast.cloud && cloud.cloud);
    assert_eq!(
        fast.timeout,
        Duration::from_millis(config.llm.fast_router_timeout_ms)
    );
    assert_eq!(
        cloud.timeout,
        Duration::from_millis(config.llm.cloud_timeout_ms)
    );
    let router = FailoverRouter::from_config(&config.llm, None).await?;
    assert!(!router.consent().is_allowed());
    Ok(())
}

#[tokio::test]
async fn withdrawn_gateway_skipped_without_penalizing_health() -> Result {
    manual_clock();
    let cloud = MockServer::start().await;
    let local = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&cloud)
        .await;
    Mock::given(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(
            "data: {\"choices\":[{\"delta\":{\"content\":\"local\"}}]}\n\ndata: [DONE]\n\n",
            "text/event-stream",
        ))
        .expect(1)
        .mount(&local)
        .await;
    let mut first = endpoint(&cloud.uri(), true);
    first.priority = 0;
    let r = router(vec![first, endpoint(&local.uri(), false)], None).await?;
    r.consent().withdraw();
    let mut text = String::new();
    assert_eq!(
        r.stream("hola", 12, &CancellationToken::new(), |s| text.push_str(s))
            .await?,
        "fast"
    );
    assert_eq!(text, "local");
    assert_eq!(r.cooldown_remaining("cloud").await, Duration::ZERO);
    Ok(())
}
