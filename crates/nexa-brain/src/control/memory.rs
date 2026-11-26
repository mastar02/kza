use super::*;

impl Control {
    fn memory(&self) -> Result<&dyn MemoryClient, Status> {
        self.0
            .memory
            .as_deref()
            .ok_or_else(|| Status::unavailable("memory service unavailable"))
    }
}
pub(super) async fn list_conversations(
    control: &Control,
    request: Request<v1::ListConversationsRequest>,
) -> Reply<v1::ConversationsSnapshot> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::ConversationsRead])?;
    ok(v1::ConversationsSnapshot {
        payload_json: control
            .0
            .events
            .conversations(&request.room_id, limit(request.limit))
            .to_string(),
    })
}

pub(super) async fn list_ha_actions(
    control: &Control,
    request: Request<v1::ListHaActionsRequest>,
) -> Reply<v1::HaActionsSnapshot> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::DiagnosticsRead])?;
    ok(v1::HaActionsSnapshot {
        receipts: control.0.home.history(limit(request.limit)),
    })
}

pub(super) async fn list_memory_overview(
    control: &Control,
    request: Request<v1::MemoryOverviewRequest>,
) -> Reply<v1::MemoryOverviewResponse> {
    let mut request = request.into_inner();
    let needed: &[_] = match request.view.as_str() {
        "summary" | "facts" | "proposals" => &[P::MemoryRead],
        "returned" | "owners_counts" | "voice_consents" => &[P::MemoryRead, P::MemoryAssign],
        "mine" => &[P::MemoryOwnRead],
        "voice_mine" => &[P::MemoryOwnVoice],
        _ => {
            authorize(request.actor.as_ref(), &[P::MemoryRead])?;
            return Err(Status::invalid_argument("unknown memory view"));
        }
    };
    authorize(request.actor.as_ref(), needed)?;
    request.limit = limit(request.limit) as u32;
    ok(control.memory()?.overview(request).await?)
}

pub(super) async fn mutate_memory_item(
    control: &Control,
    request: Request<v1::MemoryMutationRequest>,
) -> Reply<v1::MemoryMutationResponse> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[])?;
    let needed: &[_] = match request.operation.as_str() {
        "assign_owner" | "reopen_voice_consent" => &[P::MemoryAssign],
        "confirm_proposal" | "reject_proposal" => &[P::MemoryReview],
        "return_fact" => &[P::MemoryOwnWrite],
        "resume_attribution" => &[P::MemoryAssign, P::VoicesReview, P::AdminManage],
        "voice_consent_grant" | "voice_consent_revoke" | "voice_verification" => {
            &[P::MemoryOwnVoice]
        }
        "confirm_voice_profile" | "delete_voice_profile" | "correct_voice_group" => {
            &[P::VoicesReview]
        }
        "erase_conversation" => &[P::ConversationsRead, P::AdminManage],
        _ => return Err(Status::invalid_argument("unknown memory operation")),
    };
    authorize(request.actor.as_ref(), needed)?;
    if request.operation == "erase_conversation" {
        if !valid_identifier(&request.target_id) {
            return Err(Status::invalid_argument("invalid conversation ID"));
        }
        let brain = Arc::clone(&control.0);
        // Finish the fence even if the requesting client disconnects.
        let result = tokio::spawn(async move {
                let result = Control(Arc::clone(&brain)).memory()?.erase_conversation(v1::EraseConversationRequest {
                    conversation_id: request.target_id.clone(), actor: request.actor,
                }).await?;
                if !result.tombstone_recorded {
                    return Err(Status::failed_precondition("memory did not confirm the erasure fence"));
                }
                brain.event(v1::LiveEventKind::ErasureFence, "", &request.target_id, json!({"erasure_id": result.erasure_id}));
                Ok(v1::MemoryMutationResponse { payload_json: json!({
                    "erasure_id": result.erasure_id, "tombstone_recorded": true,
                    "facts_deleted": result.facts_deleted, "vectors_deleted": result.vectors_deleted,
                    "captures_deleted": result.captures_deleted, "completed_at_unix_ms": result.completed_at_unix_ms,
                }).to_string() })
            }).await.map_err(|_| Status::internal("erasure worker failed"))??;
        return ok(result);
    }
    ok(control.memory()?.mutate(request).await?)
}

pub(super) async fn get_voice_global_switch(
    control: &Control,
    request: Request<v1::ActorRequest>,
) -> Reply<v1::VoiceGlobalSwitchStatus> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::MemoryRead, P::MemoryAssign])?;
    ok(control.memory()?.voice_switch(request).await?)
}

pub(super) async fn set_voice_global_switch(
    control: &Control,
    request: Request<v1::SetVoiceGlobalSwitchRequest>,
) -> Reply<v1::VoiceGlobalSwitchStatus> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::MemoryAssign, P::AdminManage])?;
    ok(control.memory()?.set_voice_switch(request).await?)
}

pub(super) async fn get_voice_capture_review(
    control: &Control,
    request: Request<v1::VoiceCaptureReviewRequest>,
) -> Reply<v1::VoiceCaptureReviewResponse> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::VoicesRead, P::VoicesReview])?;
    ok(control.memory()?.review_capture(request).await?)
}

pub(super) async fn query_voice_captures(
    control: &Control,
    request: Request<v1::VoiceCaptureQueryRequest>,
) -> Reply<v1::VoiceCaptureQueryResponse> {
    let request = request.into_inner();
    authorize(
        request.actor.as_ref(),
        &[P::VoicesRead, P::VoicesReview, P::AdminManage],
    )?;
    ok(control.memory()?.query_captures(request).await?)
}

pub(super) async fn get_evaluation_sample(
    control: &Control,
    request: Request<v1::EvaluationSampleRequest>,
) -> Reply<v1::EvaluationSampleResponse> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::EvaluationRead])?;
    ok(control.memory()?.evaluation_sample(request).await?)
}
