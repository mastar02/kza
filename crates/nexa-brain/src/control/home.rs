use super::*;

pub(super) async fn get_health(
    control: &Control,
    _: Request<v1::Empty>,
) -> Reply<v1::HealthResponse> {
    ok(v1::HealthResponse {
        service: "nexa-brain".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        state: if control.0.shutdown.is_cancelled() {
            v1::HealthState::Unavailable
        } else {
            v1::HealthState::Ok
        }
        .into(),
        uptime_ms: u64::try_from(control.0.started.elapsed().as_millis()).unwrap_or(u64::MAX),
        detail: String::new(),
    })
}

pub(super) async fn get_home_catalog(
    control: &Control,
    request: Request<v1::GetHomeCatalogRequest>,
) -> Reply<v1::HomeCatalogSnapshot> {
    authorize(request.get_ref().actor.as_ref(), &[P::HomeRead])?;
    ok(control.0.catalog())
}

pub(super) async fn execute_home_action(
    control: &Control,
    request: Request<v1::HomeActionRequest>,
) -> Reply<v1::HomeActionReceipt> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::HomeControl])?;
    let operation = typed_operation(&request.operation, &request.params)?;
    let action = nexa_ha::EffectIntent {
        entity_id: request.entity_id,
        operation,
    };
    ok(control.0.execute_home(request.request_id, action).await?)
}

pub(super) async fn get_home_action_receipt(
    control: &Control,
    request: Request<v1::GetHomeActionReceiptRequest>,
) -> Reply<v1::HomeActionReceipt> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::HomeRead])?;
    ok(control
        .0
        .home_receipt(&request.request_id)
        .ok_or_else(|| Status::not_found("receipt not available"))?)
}

pub(super) async fn get_llm_endpoints(
    control: &Control,
    request: Request<v1::Empty>,
) -> Reply<v1::LlmEndpointsSnapshot> {
    authorize_metadata(&request, &[P::DiagnosticsRead])?;
    let mut endpoints = Vec::new();
    for endpoint in &control.0.endpoints {
        let remaining = control.0.llm.cooldown_remaining(&endpoint.id).await;
        let during = !remaining.is_zero();
        let until = if during {
            control
                .0
                .now()
                .saturating_add(u64::try_from(remaining.as_millis()).unwrap_or(u64::MAX))
        } else {
            0
        };
        endpoints.push(v1::LlmEndpointStatus {
            endpoint_id: endpoint.id.clone(),
            kind: if endpoint.cloud { "cloud" } else { "local" }.into(),
            priority: endpoint.priority,
            in_cooldown: during,
            cooldown_until_unix_ms: until,
            consecutive_errors: control.0.llm.consecutive_errors(&endpoint.id).await,
        });
    }
    ok(v1::LlmEndpointsSnapshot { endpoints })
}

pub(super) async fn clear_llm_cooldown(
    control: &Control,
    request: Request<v1::ClearLlmCooldownRequest>,
) -> Reply<v1::Empty> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::AdminManage])?;
    if !control
        .0
        .endpoints
        .iter()
        .any(|endpoint| endpoint.id == request.endpoint_id)
    {
        return Err(Status::not_found("unknown LLM endpoint"));
    }
    control
        .0
        .llm
        .clear_cooldown(&request.endpoint_id)
        .await
        .map_err(|_| Status::internal("cooldown update failed"))?;
    control.0.event(
        v1::LiveEventKind::Cooldown,
        "",
        "",
        json!({"event": "cleared", "endpoint_id": request.endpoint_id}),
    );
    ok(v1::Empty {})
}

pub(super) async fn get_cloud_consent(
    control: &Control,
    request: Request<v1::Empty>,
) -> Reply<v1::CloudConsentStatus> {
    authorize_metadata(&request, &[P::DiagnosticsRead])?;
    ok(control.0.cloud_status().await)
}

pub(super) async fn withdraw_cloud_consent(
    control: &Control,
    request: Request<v1::WithdrawCloudConsentRequest>,
) -> Reply<v1::CloudConsentStatus> {
    authorize(request.get_ref().actor.as_ref(), &[P::AdminManage])?;
    ok(control.0.withdraw_cloud().await?)
}

pub(super) async fn subscribe_live_events(
    control: &Control,
    request: Request<v1::SubscribeLiveEventsRequest>,
) -> Reply<StreamOf<Result<v1::LiveRuntimeEvent, Status>>> {
    let actor = authorize(request.get_ref().actor.as_ref(), &[])?;
    let conversations = has(actor, P::ConversationsRead);
    let diagnostics = has(actor, P::DiagnosticsRead);
    let home = has(actor, P::HomeRead);
    let voices = has(actor, P::VoicesRead) && has(actor, P::VoicesReview);
    let evaluation = has(actor, P::EvaluationRead);
    if !(conversations || diagnostics || home || voices || evaluation) {
        return Err(Status::permission_denied("live events permission required"));
    }
    let mut events = control.0.events.subscribe();
    let shutdown = control.0.shutdown_token();
    let stream = async_stream::try_stream! {
        loop {
            let message = tokio::select! {
                _ = shutdown.cancelled() => break,
                event = events.recv() => event,
            };
            let mut event = match message {
                Ok(event) => event,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    Err(Status::resource_exhausted("live events skipped; resubscribe"))?
                }
            };
            let visible = match event.kind() {
                v1::LiveEventKind::Turn | v1::LiveEventKind::Wake | v1::LiveEventKind::Tts => conversations,
                v1::LiveEventKind::Cooldown => diagnostics,
                v1::LiveEventKind::Alert | v1::LiveEventKind::HaState => home,
                v1::LiveEventKind::ErasureFence => {
                    if !conversations {
                        event.conversation_id.clear();
                        event.payload_json = "{}".into();
                    }
                    conversations || voices || evaluation
                }
                v1::LiveEventKind::Unspecified => false,
            };
            if visible { yield event; }
        }
    };
    ok(Box::pin(stream))
}
