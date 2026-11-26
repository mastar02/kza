use nexa_brain::control::ACTOR_METADATA;
use nexa_household::Clock;
use nexa_proto::v1::{self, Permission as P, Role, UtteranceKind};
use prost::Message;
use serde_json::{Value, json};
use tonic::{Code, Request, metadata::MetadataValue};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

use super::support::{Harness, Result, actor, receive};

#[tokio::test]
async fn memory_control_forwards_all_seven_rpcs_over_uds() -> Result {
    let mut h = Harness::start(|_| {}, false).await?;
    let operator = actor(
        Role::Operator,
        &[
            P::MemoryRead,
            P::MemoryAssign,
            P::MemoryReview,
            P::VoicesRead,
            P::VoicesReview,
            P::AdminManage,
        ],
    );
    let mut person = actor(Role::Home, &[P::MemoryOwnWrite, P::MemoryOwnVoice]);
    person.user_id = "test-person".into();
    for (view, who) in [("voice_consents", &operator), ("voice_mine", &person)] {
        let reply = h
            .client
            .list_memory_overview(v1::MemoryOverviewRequest {
                actor: Some(who.clone()),
                view: view.into(),
                limit: 9,
                ..Default::default()
            })
            .await?
            .into_inner();
        assert_eq!(
            serde_json::from_str::<Value>(&reply.payload_json)?,
            json!({"view": view, "limit": 9})
        );
    }
    let target_id = "test-person";
    for (operation, who) in [
        ("return_fact", &person),
        ("voice_consent_grant", &person),
        ("voice_consent_revoke", &person),
        ("voice_verification", &person),
        ("confirm_voice_profile", &operator),
        ("delete_voice_profile", &operator),
        ("correct_voice_group", &operator),
    ] {
        let reply = h
            .client
            .mutate_memory_item(v1::MemoryMutationRequest {
                actor: Some(who.clone()),
                operation: operation.into(),
                target_id: target_id.into(),
                payload_json: "{}".into(),
            })
            .await?
            .into_inner();
        assert_eq!(
            serde_json::from_str::<Value>(&reply.payload_json)?,
            json!({
                "operation": operation, "target_id": target_id, "user_id": who.user_id,
            })
        );
    }
    let denied = h
        .client
        .mutate_memory_item(v1::MemoryMutationRequest {
            actor: Some(operator.clone()),
            operation: "return_fact".into(),
            target_id: target_id.into(),
            payload_json: "{}".into(),
        })
        .await
        .err()
        .ok_or("operator was allowed to return a personal fact")?;
    assert_eq!(denied.code(), Code::PermissionDenied);

    let initial = h
        .client
        .get_voice_global_switch(v1::ActorRequest {
            actor: Some(operator.clone()),
        })
        .await?
        .into_inner();
    assert!(!initial.enabled);
    assert_eq!(initial.revision, 10);
    let updated = h
        .client
        .set_voice_global_switch(v1::SetVoiceGlobalSwitchRequest {
            actor: Some(operator.clone()),
            enabled: true,
            expected_revision: initial.revision,
        })
        .await?
        .into_inner();
    assert!(updated.enabled);
    assert_eq!(updated.revision, 11);
    let review = h
        .client
        .get_voice_capture_review(v1::VoiceCaptureReviewRequest {
            actor: Some(operator.clone()),
            fingerprint_id: "capture-1".into(),
            include_transcript: true,
        })
        .await?
        .into_inner();
    assert_eq!(
        serde_json::from_str::<Value>(&review.payload_json)?,
        json!({
            "review": "capture-1", "include_transcript": true,
        })
    );
    let query_json = json!({"limit": 7}).to_string();
    let captures = h
        .client
        .query_voice_captures(v1::VoiceCaptureQueryRequest {
            actor: Some(operator),
            query_json: query_json.clone(),
        })
        .await?
        .into_inner();
    assert_eq!(captures.payload_json, query_json);
    let sample = h
        .client
        .get_evaluation_sample(v1::EvaluationSampleRequest {
            actor: Some(actor(Role::Evaluator, &[P::EvaluationRead])),
            fingerprint_id: "capture-2".into(),
        })
        .await?
        .into_inner();
    assert_eq!(
        serde_json::from_str::<Value>(&sample.payload_json)?,
        json!({"sample": "capture-2"})
    );
    h.stop().await
}

#[tokio::test]
async fn scheduled_routine_uses_injected_clock_and_advances_once_per_poll() -> Result {
    let mut h = Harness::start(|_| {}, false).await?;
    let operator = actor(Role::Operator, &[P::RoutinesManage, P::HomeRead]);
    let now = h.clock.now();
    let schedule = json!({"kind": "once", "at_ms": now + 10_000});
    let mut definition = json!({
        "name": "rutina programada", "enabled": true, "schedule": schedule,
        "steps": [
            {"kind": "announce", "room": "living", "message": "Primer paso de la rutina."},
            {"kind": "announce", "room": "hall", "message": "Segundo paso de la rutina."}
        ],
    });
    let mut invalid = definition.clone();
    for schedule in [
        json!({"kind": "daily", "hour": 24, "minute": 0, "timezone": "UTC"}),
        json!({"kind": "daily", "hour": 8, "minute": 0, "timezone": "Invalid/Timezone"}),
        json!({"kind": "daily", "hour": 8, "minute": 0, "timezone": "UTC", "days": [0]}),
    ] {
        invalid["schedule"] = schedule;
        let preview = h
            .client
            .preview_routine(v1::PreviewRoutineRequest {
                spec_json: invalid.to_string(),
                actor: Some(operator.clone()),
            })
            .await?
            .into_inner();
        assert!(!preview.valid && !preview.errors.is_empty());
    }
    let preview = h
        .client
        .preview_routine(v1::PreviewRoutineRequest {
            spec_json: definition.to_string(),
            actor: Some(operator.clone()),
        })
        .await?
        .into_inner();
    assert!(preview.valid);
    assert_eq!(
        serde_json::from_str::<Value>(&preview.preview_json)?["schedule"],
        schedule
    );

    let created = h
        .client
        .upsert_routine(v1::UpsertRoutineRequest {
            spec_json: definition.to_string(),
            actor: Some(operator.clone()),
            ..Default::default()
        })
        .await?
        .into_inner();
    assert_eq!(created.updated_at_unix_ms, u64::try_from(now)?);
    assert_eq!(
        serde_json::from_str::<Value>(&created.schedule_json)?,
        schedule
    );
    h.brain.poll_household().await?;
    assert!(h.spoken.try_recv().is_err());

    h.clock.advance(1);
    definition["schedule"]["at_ms"] = json!(now + 20_000);
    let mut update = v1::UpsertRoutineRequest {
        spec_json: definition.to_string(),
        actor: Some(operator.clone()),
        routine_id: created.routine_id.clone(),
        expected_revision: created.revision,
    };
    let updated = h.client.upsert_routine(update.clone()).await?.into_inner();
    assert_eq!(updated.updated_at_unix_ms, u64::try_from(h.clock.now())?);
    let stale = h
        .client
        .upsert_routine(update.clone())
        .await
        .err()
        .ok_or("stale update committed")?;
    assert_eq!(stale.code(), Code::Aborted);

    // The web enable/disable route copies spec_json and changes enabled.
    let mut spec = serde_json::from_str::<Value>(&updated.spec_json)?;
    spec["enabled"] = json!(false);
    update.spec_json = spec.to_string();
    update.expected_revision = updated.revision;
    let disabled = h.client.upsert_routine(update.clone()).await?.into_inner();
    assert!(!disabled.enabled);
    assert_eq!(disabled.schedule_json, updated.schedule_json);
    spec["enabled"] = json!(true);
    update.spec_json = spec.to_string();
    update.expected_revision = disabled.revision;
    let enabled = h.client.upsert_routine(update).await?.into_inner();
    assert_eq!(enabled.schedule_json, updated.schedule_json);
    let read = h
        .client
        .get_routine(v1::GetRoutineRequest {
            actor: Some(operator),
            routine_id: created.routine_id,
        })
        .await?
        .into_inner();
    assert_eq!(read, enabled);

    h.clock.advance(9);
    h.brain.poll_household().await?;
    assert!(h.spoken.try_recv().is_err());
    h.clock.advance(10);
    h.brain.poll_household().await?;
    assert_eq!(receive(&mut h.spoken).await?, "Primer paso de la rutina.");
    assert!(
        h.spoken.try_recv().is_err(),
        "more than one step in the same poll"
    );
    assert_eq!(
        h.playback()
            .await?
            .last()
            .map(|chunk| chunk.zone_id.as_str()),
        Some("zone_living")
    );
    h.brain.poll_household().await?;
    assert_eq!(receive(&mut h.spoken).await?, "Segundo paso de la rutina.");
    assert_eq!(
        h.playback()
            .await?
            .last()
            .map(|chunk| chunk.zone_id.as_str()),
        Some("zone_hall")
    );
    h.brain.poll_household().await?;
    assert!(h.spoken.try_recv().is_err(), "the occurrence ran again");
    h.stop().await
}

#[tokio::test]
async fn reminder_mutations_use_persisted_revisions_and_exact_owner_scope() -> Result {
    let mut h = Harness::start(|_| {}, false).await?;
    let operator = actor(
        Role::Operator,
        &[
            P::TasksRead,
            P::TasksWrite,
            P::HomeControl,
            P::DiagnosticsRead,
        ],
    );
    let mut owner = actor(Role::Home, &[P::TasksRead, P::TasksWrite, P::TasksPersonal]);
    owner.user_id = "test-owner".into();
    let mut other = owner.clone();
    other.user_id = "other-person".into();
    let due = h.clock.now() + 60_125;
    let private = h
        .client
        .mutate_reminder(v1::ReminderMutationRequest {
            actor: Some(owner.clone()),
            operation: "create".into(),
            payload_json: json!({
                "text": "recordatorio privado", "trigger_at": due as f64 / 1000.0,
                "recurrence": null, "ha_actions": null, "delivery_room": null,
            })
            .to_string(),
            ..Default::default()
        })
        .await?
        .into_inner();
    assert_eq!(private.owner_user_id, owner.user_id);
    assert_eq!(private.due_at_unix_ms, u64::try_from(due)?);
    assert_eq!(
        serde_json::from_str::<Value>(&private.payload_json)?["delivery_room"],
        "living"
    );
    let shared = h.client.mutate_reminder(v1::ReminderMutationRequest {
        actor: Some(operator.clone()), operation: "create".into(),
        payload_json: json!({
            "text": "recordatorio compartido", "scope": "shared",
            "trigger_at": due, "delivery_room": "hall",
            "ha_actions": [{"domain": "light", "service": "turn_on", "entity_id": "light.grupo_living"}],
        }).to_string(),
        ..Default::default()
    }).await?.into_inner();
    assert_eq!(shared.due_at_unix_ms, private.due_at_unix_ms);
    assert!(shared.owner_user_id.is_empty());
    assert_eq!(
        serde_json::from_str::<Value>(&shared.payload_json)?["ha_actions"][0]["domain"],
        "light"
    );

    let edit = json!({
        "reminder_id": private.reminder_id, "expected_revision": private.revision,
        "text": "nuevo texto", "scope": "private", "delivery_room": "cocina",
        "trigger_at": due + 1000, "ha_actions": [],
    });
    let mut request = v1::ReminderMutationRequest {
        actor: Some(other.clone()),
        operation: "update".into(),
        payload_json: edit.to_string(),
        ..Default::default()
    };
    let denied = h
        .client
        .mutate_reminder(request.clone())
        .await
        .err()
        .ok_or("wrong owner edited reminder")?;
    assert_eq!(denied.code(), Code::NotFound);
    request.actor = Some(operator.clone());
    let denied = h
        .client
        .mutate_reminder(request.clone())
        .await
        .err()
        .ok_or("missing TasksPersonal was accepted")?;
    assert_eq!(denied.code(), Code::PermissionDenied);
    request.actor = Some(owner.clone());
    let mut invalid = edit.clone();
    invalid["personal"] = json!(false);
    request.payload_json = invalid.to_string();
    let denied = h
        .client
        .mutate_reminder(request.clone())
        .await
        .err()
        .ok_or("conflicting scopes were accepted")?;
    assert_eq!(denied.code(), Code::InvalidArgument);
    let mut invalid = edit.clone();
    invalid["unknown_field"] = json!(true);
    request.payload_json = invalid.to_string();
    let denied = h
        .client
        .mutate_reminder(request.clone())
        .await
        .err()
        .ok_or("unknown field was accepted")?;
    assert_eq!(denied.code(), Code::InvalidArgument);
    request.payload_json = edit.to_string();
    let updated = h
        .client
        .mutate_reminder(request.clone())
        .await?
        .into_inner();
    assert_eq!(updated.title, "nuevo texto");
    assert_eq!(updated.owner_user_id, private.owner_user_id);
    assert_eq!(updated.revision, private.revision + 1);
    assert_eq!(updated.due_at_unix_ms, private.due_at_unix_ms + 1000);
    let stale = h
        .client
        .mutate_reminder(request)
        .await
        .err()
        .ok_or("stale reminder edit committed")?;
    assert_eq!(stale.code(), Code::Aborted);

    let notices = h
        .client
        .list_reminder_notices(v1::ActorRequest {
            actor: Some(owner.clone()),
        })
        .await?
        .into_inner();
    assert_eq!(
        serde_json::from_str::<Value>(&notices.payload_json)?,
        json!({"notices": [{
            "reminder_id": shared.reminder_id, "state": "pending", "attempts": 0,
            "next_attempt_at_unix_ms": due,
        }]})
    );
    let completed = h
        .client
        .mutate_reminder(v1::ReminderMutationRequest {
            actor: Some(operator.clone()),
            operation: "dismiss_notice".into(),
            payload_json: json!({
                "reminder_id": shared.reminder_id, "trigger_at": due as f64 / 1000.0,
                "expected_revision": shared.revision,
            })
            .to_string(),
            ..Default::default()
        })
        .await?
        .into_inner();
    assert_eq!(completed.status, "completed");
    let cancelled = h
        .client
        .mutate_reminder(v1::ReminderMutationRequest {
            actor: Some(owner.clone()),
            operation: "cancel".into(),
            reminder_id: updated.reminder_id,
            expected_revision: updated.revision,
            payload_json: "{}".into(),
        })
        .await?
        .into_inner();
    assert_eq!(cancelled.status, "cancelled");
    assert_eq!(cancelled.revision, updated.revision + 1);

    let mut list = v1::ListRemindersRequest {
        actor: Some(owner),
        include_personal: true,
        include_completed: false,
    };
    assert!(
        h.client
            .list_reminders(list.clone())
            .await?
            .into_inner()
            .reminders
            .is_empty()
    );
    list.include_completed = true;
    assert_eq!(
        h.client
            .list_reminders(list.clone())
            .await?
            .into_inner()
            .reminders,
        [completed.clone(), cancelled]
    );
    list.actor = Some(other);
    assert_eq!(
        h.client.list_reminders(list).await?.into_inner().reminders,
        [completed]
    );

    let status = h
        .client
        .get_reminder_status(v1::ActorRequest {
            actor: Some(operator),
        })
        .await?
        .into_inner();
    let status: Value = serde_json::from_str(&status.payload_json)?;
    assert_eq!(status["pending"], 0);
    assert_eq!(status["completed"], 1);
    h.stop().await
}

#[tokio::test]
async fn list_sync_reports_shared_list_count_without_todo_configuration() -> Result {
    let mut h = Harness::start(|_| {}, false).await?;
    let mut person = actor(Role::Home, &[P::TasksWrite, P::TasksPersonal]);
    person.user_id = "test-person".into();
    for (name, personal) in [("compras compartidas", false), ("compras privadas", true)] {
        h.client
            .mutate_household_list(v1::HouseholdListMutationRequest {
                actor: Some(person.clone()),
                operation: "create_list".into(),
                payload_json: json!({"name": name, "personal": personal}).to_string(),
                ..Default::default()
            })
            .await?;
    }
    let sync = h
        .client
        .sync_household_lists(v1::ActorRequest {
            actor: Some(person),
        })
        .await?
        .into_inner();
    assert_eq!(
        serde_json::from_str::<Value>(&sync.payload_json)?,
        json!({
            "lists": 1, "synced": 0, "ha_todo_configured": false, "status": "not_configured",
        })
    );
    h.stop().await
}

#[tokio::test]
async fn shared_memory_goes_to_the_local_endpoint_only() -> Result {
    voice_with_memory(false).await
}

#[tokio::test]
async fn local_failure_falls_back_without_memory_and_reports_error_streak() -> Result {
    voice_with_memory(true).await
}

async fn voice_with_memory(local_fails: bool) -> Result {
    const SHARED_FACT: &str = "Marcador privado de la casa verificado";
    const QUESTION: &str = "por que es azul el cielo";
    let local = MockServer::start().await;
    let cloud = MockServer::start().await;
    let reply = |text: &str| {
        ResponseTemplate::new(200)
            .insert_header("Content-Type", "text/event-stream")
            .set_body_string(format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({"choices": [{"text": text}]}),
            ))
    };
    let enriched = format!(
        "Contexto local de la casa: {}\nConsulta: {QUESTION}",
        json!([SHARED_FACT])
    );
    Mock::given(method("POST"))
        .and(path("/v1/completions"))
        .and(move |req: &wiremock::Request| {
            serde_json::from_slice::<Value>(&req.body).is_ok_and(|body| body["prompt"] == enriched)
        })
        .respond_with(if local_fails {
            ResponseTemplate::new(503)
        } else {
            reply("Respuesta local.")
        })
        .expect(1)
        .mount(&local)
        .await;
    if local_fails {
        Mock::given(method("POST"))
            .and(path("/v1/completions"))
            .and(|req: &wiremock::Request| {
                serde_json::from_slice::<Value>(&req.body)
                    .is_ok_and(|body| body["prompt"] == QUESTION)
            })
            .respond_with(ResponseTemplate::new(503))
            .expect(1)
            .mount(&local)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(|req: &wiremock::Request| {
                serde_json::from_slice::<Value>(&req.body).is_ok_and(|body| {
                    body["messages"] == json!([{"role": "user", "content": QUESTION}])
                })
            })
            .respond_with(reply("Respuesta sin memoria."))
            .expect(1)
            .mount(&cloud)
            .await;
    }
    Mock::given(method("POST"))
        .and(move |req: &wiremock::Request| {
            !local_fails || String::from_utf8_lossy(&req.body).contains(SHARED_FACT)
        })
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&cloud)
        .await;
    let mut h = Harness::start_with_hits(
        |cfg| {
            cfg.core.llm.fast_router_url = format!("{}/v1", local.uri());
            cfg.core.llm.cloud_gateway_url = format!("{}/v1", cloud.uri());
            cfg.core.llm.cloud_consent = true;
        },
        false,
        vec![v1::MemorySearchHit {
            fact: Some(v1::MemoryFact {
                content: SHARED_FACT.into(),
                confirmed: true,
                owner_user_id: String::new(),
                ..Default::default()
            }),
            score: 0.99,
        }],
    )
    .await?;
    h.utterance("shared-memory", QUESTION, UtteranceKind::Command)
        .await?;
    let query = receive(&mut h.queries).await?;
    assert!(query.include_shared && query.owner_user_id.is_empty());
    assert_eq!(
        query.actor,
        Some(v1::ActorContext {
            subject: "nexa-brain".into(),
            role: Role::Operator as i32,
            permissions: vec![P::MemoryRead as i32],
            ..Default::default()
        })
    );
    let spoken = receive(&mut h.spoken).await?;
    assert_eq!(
        spoken,
        if local_fails {
            "Respuesta sin memoria."
        } else {
            "Respuesta local."
        }
    );
    let _ = h.playback().await?;

    let diagnostics = actor(Role::Operator, &[P::DiagnosticsRead]);
    let mut request = Request::new(v1::Empty {});
    request.metadata_mut().insert_bin(
        ACTOR_METADATA,
        MetadataValue::from_bytes(&diagnostics.encode_to_vec()),
    );
    let endpoints = h
        .client
        .get_llm_endpoints(request)
        .await?
        .into_inner()
        .endpoints;
    let local_status = endpoints
        .iter()
        .find(|endpoint| endpoint.kind == "local")
        .ok_or("missing local endpoint")?;
    assert_eq!(local_status.consecutive_errors > 0, local_fails);
    h.stop().await?;
    local.verify().await;
    cloud.verify().await;
    assert_eq!(
        local
            .received_requests()
            .await
            .ok_or("wiremock recording disabled")?
            .len(),
        if local_fails { 2 } else { 1 }
    );
    assert_eq!(
        cloud
            .received_requests()
            .await
            .ok_or("wiremock recording disabled")?
            .len(),
        usize::from(local_fails)
    );
    Ok(())
}
