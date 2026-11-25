#[path = "e2e/pre_release.rs"]
mod pre_release;
mod support;

use std::{os::unix::fs::PermissionsExt, time::Duration};

use nexa_brain::control::ACTOR_METADATA;
use nexa_proto::v1::{
    self as v1, LiveEventKind as Event, Permission as P, Role, SttMode, UtteranceKind as Kind,
};
use prost::Message;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tonic::{Code, Request, metadata::MetadataValue};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

use support::{Harness, Result, actor, event, receive};

#[tokio::test]
async fn command_compound_security_and_control_permissions() -> Result {
    let ha = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/states/light.grupo_living"))
        .and(header("authorization", "Bearer ha-test-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "entity_id": "light.grupo_living", "state": "off", "attributes": {}
        })))
        .mount(&ha)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/services/light/turn_on"))
        .and(header("authorization", "Bearer ha-test-token"))
        .and(body_json(json!({"entity_id": "light.grupo_living"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
        .expect(1)
        .mount(&ha)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/services/light/turn_off"))
        .and(body_json(json!({"entity_id": "light.grupo_living"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
        .expect(1)
        .mount(&ha)
        .await;
    let mut h = Harness::start(|cfg| cfg.core.home_assistant.url = ha.uri(), true).await?;
    let permissions = actor(
        Role::Operator,
        &[
            P::HomeRead,
            P::ConversationsRead,
            P::DiagnosticsRead,
            P::AdminManage,
        ],
    );
    let mut events = h
        .client
        .subscribe_live_events(v1::SubscribeLiveEventsRequest {
            actor: Some(permissions.clone()),
        })
        .await?
        .into_inner();

    h.utterance("action", "Nexa, prendé la luz", Kind::Command)
        .await?;
    assert_eq!(receive(&mut h.modes).await?, SttMode::Command);
    assert_eq!(receive(&mut h.spoken).await?, "Mandé la orden.");
    let chunks = h.playback().await?;
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].pcm_f32le, 0.25_f32.to_le_bytes());
    assert_eq!(chunks[0].barge_in_token, chunks[1].barge_in_token);
    let routed = event(&mut events, Event::Turn, "action").await?;
    assert_eq!(
        serde_json::from_str::<Value>(&routed.payload_json)?["path"],
        "PATH_TYPE_FAST_DOMOTICS"
    );
    let _ = event(&mut events, Event::Tts, "action").await?;

    h.utterance(
        "compound",
        "Apagá la luz y cerrá la puerta con llave",
        Kind::Command,
    )
    .await?;
    let response = receive(&mut h.spoken).await?;
    assert!(response.contains("Mandé la orden."), "{response}");
    assert!(response.contains("cerraduras"), "{response}");
    let _ = h.playback().await?;
    let compound = event(&mut events, Event::Turn, "compound").await?;
    assert_eq!(
        serde_json::from_str::<Value>(&compound.payload_json)?["path"],
        "PATH_TYPE_COMPOUND"
    );

    h.utterance("alarm", "Activá la alarma de seguridad", Kind::Command)
        .await?;
    assert!(receive(&mut h.spoken).await?.contains("cerraduras"));
    let _ = h.playback().await?;

    let write = v1::HomeActionRequest {
        request_id: "missing".into(),
        entity_id: "light.grupo_living".into(),
        operation: "turn_on".into(),
        ..v1::HomeActionRequest::default()
    };
    let failure = h
        .client
        .execute_home_action(write.clone())
        .await
        .err()
        .ok_or("missing actor was allowed")?;
    assert_eq!(failure.code(), Code::Unauthenticated);
    let failure = h
        .client
        .execute_home_action(v1::HomeActionRequest {
            actor: Some(permissions.clone()),
            ..write.clone()
        })
        .await
        .err()
        .ok_or("missing HomeControl was allowed")?;
    assert_eq!(failure.code(), Code::PermissionDenied);
    let failure = h
        .client
        .execute_home_action(v1::HomeActionRequest {
            actor: Some(actor(Role::Evaluator, &[P::EvaluationRead, P::HomeControl])),
            ..write
        })
        .await
        .err()
        .ok_or("evaluator claimed HomeControl")?;
    assert_eq!(failure.code(), Code::PermissionDenied);
    assert_eq!(
        h.client
            .get_cloud_consent(v1::Empty {})
            .await
            .err()
            .ok_or("missing metadata was allowed")?
            .code(),
        Code::Unauthenticated
    );

    h.client
        .mutate_memory_item(v1::MemoryMutationRequest {
            actor: Some(permissions.clone()),
            operation: "erase_conversation".into(),
            target_id: "action".into(),
            payload_json: String::new(),
        })
        .await?;
    let _ = event(&mut events, Event::ErasureFence, "action").await?;
    let conversations = h
        .client
        .list_conversations(v1::ListConversationsRequest {
            actor: Some(permissions),
            ..v1::ListConversationsRequest::default()
        })
        .await?
        .into_inner();
    assert!(!conversations.payload_json.contains("\"action\""));

    let posts = ha
        .received_requests()
        .await
        .ok_or("wiremock recording disabled")?
        .into_iter()
        .filter(|request| request.method == wiremock::http::Method::POST)
        .collect::<Vec<_>>();
    assert_eq!(posts.len(), 2, "{posts:?}");
    let socket = h.config.core.paths.runtime_dir.join("brain.sock");
    assert_eq!(
        std::fs::metadata(&socket)?.permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(
        std::fs::metadata(&h.config.core.paths.runtime_dir)?
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    drop(events);
    h.stop().await
}

#[tokio::test]
async fn streaming_barge_in_and_irreversible_cloud_withdrawal() -> Result {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let addr = listener.local_addr()?;
    let cloud = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&cloud)
        .await;
    let (disconnected, closed) = tokio::sync::oneshot::channel();
    let (retried, tried) = tokio::sync::oneshot::channel();
    let backend = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await?;
        // Read the complete POST body so the next read reflects cancellation.
        read_http_request(&mut socket).await?;
        socket.write_all(concat!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            "data: {\"choices\":[{\"text\":\"Esta oración llega antes de que termine el modelo.\"}]}\n\n"
        ).as_bytes()).await?;
        let mut byte = [0];
        let result = socket.read(&mut byte).await;
        assert!(matches!(result, Ok(0)) || result.is_err());
        let _ = disconnected.send(());
        let (mut socket, _) = listener.accept().await?;
        read_http_request(&mut socket).await?;
        socket.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await?;
        let _ = retried.send(());
        Ok::<_, std::io::Error>(())
    });
    let mut h = Harness::start(
        |cfg| {
            cfg.core.llm.fast_router_url = format!("http://{addr}/v1");
            cfg.core.llm.cloud_gateway_url = format!("{}/v1", cloud.uri());
            cfg.core.llm.cloud_consent = true;
        },
        false,
    )
    .await?;
    let operator = actor(
        Role::Operator,
        &[P::AdminManage, P::DiagnosticsRead, P::ConversationsRead],
    );
    let mut events = h
        .client
        .subscribe_live_events(v1::SubscribeLiveEventsRequest {
            actor: Some(operator.clone()),
        })
        .await?
        .into_inner();

    h.utterance("stream", "Por qué es azul el cielo", Kind::Command)
        .await?;
    let query = receive(&mut h.queries).await?;
    assert!(query.include_shared && query.owner_user_id.is_empty());
    let first = receive(&mut h.played).await?;
    assert!(!first.is_final);
    assert!(
        receive(&mut h.spoken)
            .await?
            .contains("antes de que termine")
    );
    h.utterance("barge", "Qué hora es", Kind::FollowUp).await?;
    let cancelled = receive(&mut h.cancelled).await?;
    assert_eq!(cancelled.barge_in_token, first.barge_in_token);
    tokio::time::timeout(Duration::from_secs(5), closed).await??;
    let after = h.playback().await?;
    assert!(
        after
            .iter()
            .all(|chunk| chunk.barge_in_token != first.barge_in_token)
    );
    assert!(receive(&mut h.spoken).await?.contains("Son las"));

    let withdrawn = h
        .client
        .withdraw_cloud_consent(v1::WithdrawCloudConsentRequest {
            actor: Some(operator.clone()),
        })
        .await?
        .into_inner();
    assert!(!withdrawn.consent_granted && withdrawn.withdrawn_latched);
    assert!(
        event(&mut events, Event::Cooldown, "")
            .await?
            .payload_json
            .contains("cloud_consent_withdrawn")
    );
    let mut request = Request::new(v1::Empty {});
    request.metadata_mut().insert_bin(
        ACTOR_METADATA,
        MetadataValue::from_bytes(&operator.encode_to_vec()),
    );
    assert_eq!(
        h.client.get_cloud_consent(request).await?.into_inner(),
        withdrawn
    );
    h.utterance(
        "after-withdrawal",
        "Contame por qué existe el arcoíris",
        Kind::Command,
    )
    .await?;
    tokio::time::timeout(Duration::from_secs(5), tried).await??;
    assert_eq!(
        receive(&mut h.spoken).await?,
        "No pude responder en este momento."
    );
    let _ = h.playback().await?;
    assert!(
        cloud
            .received_requests()
            .await
            .ok_or("wiremock recording disabled")?
            .is_empty()
    );
    assert!(
        h.config
            .core
            .paths
            .data_dir
            .join("cloud-consent-withdrawn")
            .is_file()
    );
    backend.await??;
    drop(events);
    h.stop().await
}

async fn read_http_request(socket: &mut tokio::net::TcpStream) -> std::io::Result<()> {
    let mut header = Vec::new();
    while !header.ends_with(b"\r\n\r\n") {
        header.push(socket.read_u8().await?);
        if header.len() > 8192 {
            return Err(std::io::Error::other("oversized fixture header"));
        }
    }
    let headers = String::from_utf8_lossy(&header);
    let length = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, length)| length.trim().parse::<usize>().ok())
        .ok_or_else(|| std::io::Error::other("missing content-length"))?;
    let mut body = vec![0; length.min(8192)];
    if body.len() != length {
        return Err(std::io::Error::other("oversized fixture body"));
    }
    socket.read_exact(&mut body).await?;
    Ok(())
}

#[tokio::test]
async fn household_worker_reminders_routines_timers_and_ambient() -> Result {
    let mut h = Harness::start(|_| {}, false).await?;
    let operator = actor(
        Role::Operator,
        &[
            P::TasksRead,
            P::TasksWrite,
            P::HomeRead,
            P::HomeControl,
            P::RoutinesManage,
            P::ConversationsRead,
        ],
    );
    let mut person = actor(Role::Home, &[P::TasksRead, P::TasksWrite, P::TasksPersonal]);
    person.user_id = "private-owner".into();
    let due = u64::try_from(nexa_household::Clock::now(&*h.clock))? + 120_000;
    let private = h.client.mutate_reminder(v1::ReminderMutationRequest {
        operation: "create".into(), actor: Some(person.clone()),
        payload_json: json!({"personal": true, "title": "pagar patente", "room": "living", "due_at_unix_ms": due }).to_string(), ..v1::ReminderMutationRequest::default()
    }).await?.into_inner();
    let shared = h
        .client
        .mutate_reminder(v1::ReminderMutationRequest {
            operation: "create".into(),
            actor: Some(operator.clone()),
            payload_json:
                json!({"title": "regar plantas", "room": "living", "due_at_unix_ms": due})
                    .to_string(),
            ..v1::ReminderMutationRequest::default()
        })
        .await?
        .into_inner();
    let list = h
        .client
        .list_reminders(v1::ListRemindersRequest {
            actor: Some(operator.clone()),
            ..v1::ListRemindersRequest::default()
        })
        .await?
        .into_inner();
    assert_eq!(list.reminders, [shared]);

    h.utterance("ambient", "una conversación ambiental", Kind::Ambient)
        .await?;
    assert_eq!(receive(&mut h.modes).await?, SttMode::Ambient);
    let obs = receive(&mut h.observations).await?;
    assert_eq!(obs.embedding.len(), 192);
    assert_eq!(obs.utterance_id, "ambient");
    assert_eq!(obs.room_id, "living");

    h.utterance("read-reminders", "Qué recordatorios tengo", Kind::Command)
        .await?;
    let read = receive(&mut h.spoken).await?;
    assert!(
        read.contains("regar plantas") && !read.contains("pagar patente"),
        "{read}"
    );
    let _ = h.playback().await?;
    h.utterance(
        "cancel-private",
        "Cancelá el recordatorio pagar patente",
        Kind::Command,
    )
    .await?;
    assert!(receive(&mut h.spoken).await?.contains("No encontré"));
    let _ = h.playback().await?;
    let reminders = h
        .client
        .list_reminders(v1::ListRemindersRequest {
            actor: Some(person),
            include_personal: true,
            ..v1::ListRemindersRequest::default()
        })
        .await?
        .into_inner();
    assert!(reminders.reminders.contains(&private));
    h.utterance(
        "cancel-shared",
        "Cancelá el recordatorio regar plantas",
        Kind::Command,
    )
    .await?;
    assert!(receive(&mut h.spoken).await?.contains("Cancelé"));
    let _ = h.playback().await?;

    let spec = json!({"name": "saludo", "enabled": true, "steps": [{"kind": "announce", "room": "living", "message": "Rutina en ejecución."}]}).to_string();
    let preview = h
        .client
        .preview_routine(v1::PreviewRoutineRequest {
            spec_json: spec.clone(),
            actor: Some(operator.clone()),
        })
        .await?
        .into_inner();
    assert!(preview.valid);
    let routine = h
        .client
        .upsert_routine(v1::UpsertRoutineRequest {
            spec_json: spec.clone(),
            actor: Some(operator.clone()),
            ..v1::UpsertRoutineRequest::default()
        })
        .await?
        .into_inner();
    let updated = h
        .client
        .upsert_routine(v1::UpsertRoutineRequest {
            spec_json: spec,
            expected_revision: routine.revision,
            routine_id: routine.routine_id.clone(),
            actor: Some(operator.clone()),
        })
        .await?
        .into_inner();
    assert!(updated.revision > routine.revision);
    let listed = h
        .client
        .list_routines(v1::ListRoutinesRequest {
            actor: Some(operator.clone()),
            enabled_only: Some(true),
        })
        .await?
        .into_inner();
    assert_eq!(listed.routines[0].routine_id, routine.routine_id);
    let run = v1::ExecuteRoutineRequest {
        routine_id: routine.routine_id.clone(),
        request_id: "execute-saludo".into(),
        actor: Some(operator.clone()),
    };
    let started = h.client.execute_routine(run.clone()).await?.into_inner();
    assert_eq!(started.status, "running");
    // The poll clock is independent of the Tokio timer and can be driven explicitly.
    h.brain.poll_household().await?;
    assert_eq!(receive(&mut h.spoken).await?, "Rutina en ejecución.");
    let _ = h.playback().await?;
    h.brain.poll_household().await?;
    let finished = h.client.execute_routine(run).await?.into_inner();
    assert_eq!(finished.run_id, started.run_id);
    assert_eq!(finished.status, "succeeded");
    h.client
        .delete_routine(v1::DeleteRoutineRequest {
            routine_id: routine.routine_id.clone(),
            expected_revision: updated.revision,
            actor: Some(operator.clone()),
        })
        .await?;
    assert_eq!(
        h.client
            .get_routine(v1::GetRoutineRequest {
                routine_id: routine.routine_id,
                actor: Some(operator.clone())
            })
            .await
            .err()
            .ok_or("deleted routine visible")?
            .code(),
        Code::NotFound
    );

    let mut events = h
        .client
        .subscribe_live_events(v1::SubscribeLiveEventsRequest {
            actor: Some(operator),
        })
        .await?
        .into_inner();
    h.utterance(
        "start-timer",
        "Poné un timer de dos segundos",
        Kind::Command,
    )
    .await?;
    assert!(receive(&mut h.spoken).await?.contains("2 segundos"));
    let _ = h.playback().await?;
    loop {
        let event = event(&mut events, Event::Tts, "start-timer").await?;
        if event.payload_json.contains("completed") {
            break;
        }
    }
    h.clock.advance(3);
    h.brain.poll_household().await?;
    assert!(
        event(&mut events, Event::Alert, "")
            .await?
            .payload_json
            .contains("timer_due")
    );
    assert!(
        receive(&mut h.spoken)
            .await?
            .contains("Terminó el temporizador")
    );
    let _ = h.playback().await?;
    assert!(h.dir.path().join("data/household/household.db").is_file());
    drop(events);
    h.stop().await
}
