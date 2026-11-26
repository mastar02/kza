use super::*;

#[derive(Clone, Copy)]
pub(super) enum Action {
    Users,
    UserWrite,
    UserRetire,
    Credentials,
    CredentialMutation(&'static str),
    CredentialAttempt,
}

pub(super) async fn execute(
    op: Action,
    state: &AppState,
    fields: &Fields,
    proof: &Presented,
    policy: Policy,
) -> Result<Response> {
    use Action::*;
    let memory = &state.clients.memory;
    let id = fields.id();
    let body = &fields.body;
    let status = StatusCode::OK;
    let value = match op {
        Users => state.auth.users().await?,
        UserWrite => {
            state
                .auth
                .write_user(
                    proof.clone(),
                    (!id.is_empty()).then(|| id.to_owned()),
                    body.clone(),
                )
                .await?
        }
        UserRetire => {
            if state.auth.retire(proof.clone(), id.into()).await? {
                let current = state.auth.validate(proof, policy.permissions).await?;
                let receipt = memory
                    .purge_retired_user(PurgeRetiredUserRequest {
                        user_id: id.into(),
                        actor: Some(current.actor()),
                    })
                    .await;
                if !receipt.is_ok_and(|value| value.tombstone_recorded) {
                    return Ok((StatusCode::SERVICE_UNAVAILABLE,Json(json!({
                        "detail":{"code":"retired_user_purge_pending","credentials_retired":true}
                    }))).into_response());
                }
                state.live.fence();
                state.auth.finish_purge(id.into()).await?;
            }
            json!({"ok":true,"user_id":id,"credentials_retired":true})
        }
        Credentials => {
            if fields
                .query
                .keys()
                .any(|key| !["limit", "after"].contains(&key.as_str()))
            {
                return Err(INVALID);
            }
            state
                .auth
                .credential_catalog(fields.query("after").into(), fields.limit(50, 50)?)
                .await?
        }
        CredentialMutation(action) => {
            if !fields.query.is_empty() {
                return Err(INVALID);
            }
            let allowed = match action {
                "issue" => &[
                    "request_id",
                    "subject",
                    "role",
                    "expires_at",
                    "user_id",
                    "selections",
                ][..],
                "revise" => &[
                    "request_id",
                    "expected_revision",
                    "role",
                    "expires_at",
                    "user_id",
                    "selections",
                ],
                _ => &["request_id", "expected_revision"],
            };
            allowed_fields(body, allowed)?;
            state
                .auth
                .change_credential(
                    proof.clone(),
                    action,
                    (!id.is_empty()).then(|| id.to_owned()),
                    body.clone(),
                )
                .await?
        }
        CredentialAttempt => {
            state
                .auth
                .credential_attempt(proof.clone(), id.into())
                .await?
        }
    };
    Ok((status, Json(value)).into_response())
}
