use super::*;

#[derive(Clone, Copy)]
pub(super) enum Action {
    Memory(&'static str),
    MemoryMutation(&'static str),
    DeleteFact,
}

pub(super) async fn execute(
    op: Action,
    state: &AppState,
    fields: &Fields,
    lease: &Lease,
) -> Result<Response> {
    use Action::*;
    let brain = &state.clients.brain;
    let memory = &state.clients.memory;
    let id = fields.id();
    let actor = || Some(lease.actor());
    let status = StatusCode::OK;
    let value = match op {
        Memory(view) => memory_overview(state, fields, view, lease.actor()).await?,
        MemoryMutation(operation) => {
            let (target, payload) = memory_mutation(fields, operation)?;
            let request = MemoryMutationRequest {
                operation: operation.into(),
                target_id: target.into(),
                payload_json: payload.to_string(),
                actor: actor(),
            };
            match operation {
                "delete_voice_profile" => {
                    let memory = memory.clone();
                    let live = state.live.clone();
                    // Complete the erasure fence even if the request is disconnected.
                    tokio::spawn(async move {
                        let doc =
                            projection(memory.mutate_memory_item(request).await?.payload_json)?;
                        if doc["deleted"] != true {
                            return Err(BAD_UPSTREAM);
                        }
                        live.fence();
                        Ok(doc)
                    })
                    .await
                    .map_err(|_| BAD_UPSTREAM)??
                }
                "voice_consent_grant"
                | "voice_consent_revoke"
                | "voice_verification"
                | "confirm_voice_profile"
                | "correct_voice_group" => {
                    projection(memory.mutate_memory_item(request).await?.payload_json)?
                }
                _ => projection(brain.mutate_memory_item(request).await?.payload_json)?,
            }
        }
        DeleteFact => {
            let receipt = memory
                .delete_fact(DeleteFactRequest {
                    fact_id: id.into(),
                    actor: actor(),
                })
                .await?;
            if !receipt.tombstone_recorded {
                return Err(BAD_UPSTREAM);
            }
            state.live.fence();
            json!({"deleted":receipt.facts_deleted > 0})
        }
    };
    Ok((status, Json(value)).into_response())
}

async fn memory_overview(
    state: &AppState,
    fields: &Fields,
    view: &str,
    actor: ActorContext,
) -> Result<Value> {
    let limit = fields.limit(50, 100)?;
    let request = |view: &str, cursor: &str| MemoryOverviewRequest {
        view: view.into(),
        limit,
        actor: Some(actor.clone()),
        cursor: cursor.into(),
    };
    let brain = &state.clients.brain;
    let query = &fields.query;
    let cursor = page_cursor(query, &["cursor", "after"])?;
    if view == "summary" {
        // MemoryService returns a next cursor per page only on the individual views.
        let cursor_for = |key| {
            if (query.contains_key("cursor") || query.contains_key("after"))
                && query.contains_key(key)
            {
                return Err(INVALID);
            }
            if query.contains_key(key) {
                page_cursor(query, &[key])
            } else {
                Ok(cursor)
            }
        };
        let (facts, proposals) = tokio::try_join!(
            brain.list_memory_overview(request("facts", cursor_for("facts_after")?)),
            brain.list_memory_overview(request("proposals", cursor_for("proposals_after")?)),
        )?;
        let mut facts = projection(facts.payload_json)?;
        let mut proposals = projection(proposals.payload_json)?;
        if !facts["facts"].is_array() || !proposals["proposals"].is_array() {
            return Err(BAD_UPSTREAM);
        }
        return Ok(json!({
            "facts": facts["facts"].take(),
            "facts_next": facts["next"].take(),
            "proposals": proposals["proposals"].take(),
            "proposals_next": proposals["next"].take(),
        }));
    }
    let alias = format!("{view}_after");
    let req = request(view, page_cursor(query, &["cursor", "after", &alias])?);
    let page = match view {
        "voice_mine" | "voice_consents" => state.clients.memory.list_memory_overview(req).await?,
        _ => brain.list_memory_overview(req).await?,
    };
    projection(page.payload_json)
}

fn memory_mutation<'a>(fields: &'a Fields, operation: &str) -> Result<(&'a str, Value)> {
    let body = &fields.body;
    let mut target = fields.id();
    let mut payload = body.clone();
    match operation {
        "voice_consent_grant" | "voice_consent_revoke" => {
            allowed_fields(
                body,
                &[
                    "expected_revision",
                    "expected_profile_revision",
                    "expected_link_revision",
                ],
            )?;
            for key in [
                "expected_revision",
                "expected_profile_revision",
                "expected_link_revision",
            ] {
                if body.get(key).is_some() {
                    unsigned(body, key)?;
                }
            }
        }
        "voice_verification" => {
            allowed_fields(body, &["expected_revision", "outcome", "room_id"])?;
            if body.get("room_id").is_some() {
                memory_id(string(body, "room_id")?)?;
                if body.get("outcome").is_some() {
                    return Err(INVALID);
                }
            } else if !matches!(string(body, "outcome")?, "passed" | "failed") {
                return Err(INVALID);
            }
            if body.get("expected_revision").is_some() {
                unsigned(body, "expected_revision")?;
            }
        }
        "confirm_voice_profile" => {
            allowed_fields(
                body,
                &[
                    "profile_id",
                    "fingerprint_ids",
                    "expected_revision",
                    "name",
                    "request_id",
                ],
            )?;
            if let Some(profile) = optional_string(body, "profile_id")?.filter(|id| !id.is_empty())
            {
                target = memory_id(profile)?;
            }
            voice_ids(body, "fingerprint_ids")?;
            if body.get("expected_revision").is_some() {
                unsigned(body, "expected_revision")?;
            }
            if body.get("name").is_some() {
                bounded_text(body, "name", 64)?;
            }
            if body.get("request_id").is_some() {
                request_id(string(body, "request_id")?)?;
            }
        }
        "delete_voice_profile" => {
            memory_id(target)?;
            allowed_fields(body, &["expected_revision"])?;
            if body.get("expected_revision").is_some()
                || fields.query.contains_key("expected_revision")
            {
                payload["expected_revision"] = json!(fields.revision()?);
            }
        }
        "correct_voice_group" => {
            allowed_fields(
                body,
                &[
                    "group_id",
                    "action",
                    "profile_ids",
                    "fingerprint_ids",
                    "expected_revision",
                    "expected_revisions",
                ],
            )?;
            if let Some(group) = optional_string(body, "group_id")?.filter(|id| !id.is_empty()) {
                target = memory_id(group)?;
            }
            let action = optional_string(body, "action")?.unwrap_or_default();
            if !matches!(action, "" | "merge" | "split" | "unknown")
                || (!target.is_empty() && matches!(action, "split" | "unknown"))
            {
                return Err(INVALID);
            }
            match (body.get("profile_ids"), body.get("fingerprint_ids")) {
                (Some(_), None) => voice_ids(body, "profile_ids")?,
                (None, Some(_)) => voice_ids(body, "fingerprint_ids")?,
                _ => return Err(INVALID),
            }
            if body.get("expected_revision").is_some() {
                unsigned(body, "expected_revision")?;
            }
            if let Some(revisions) = body.get("expected_revisions") {
                for (id, value) in revisions.as_object().ok_or(INVALID)? {
                    memory_id(id)?;
                    value.as_u64().ok_or(INVALID)?;
                }
            }
        }
        _ => {}
    }
    Ok((target, payload))
}
