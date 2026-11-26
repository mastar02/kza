use super::*;

#[derive(Clone, Copy)]
pub(super) enum Action {
    Zones,
    Conversations,
    HaEntities,
    HaActions,
    Llm,
    Cooldown,
    Gpus,
    Services,
    Restart,
    Cloud,
    CloudWithdraw,
}

pub(super) async fn execute(
    op: Action,
    state: &AppState,
    fields: &Fields,
    lease: &Lease,
) -> Result<Response> {
    use Action::*;
    let brain = &state.clients.brain;
    let audio = &state.clients.audio;
    let id = fields.id();
    let actor = || Some(lease.actor());
    let status = StatusCode::OK;
    let value = match op {
        Zones => json!(
            audio
                .get_zones(Empty {})
                .await?
                .zones
                .into_iter()
                .map(zone)
                .collect::<Vec<_>>()
        ),
        Conversations => {
            let connection =
                state
                    .live
                    .authorize(&fields.headers, lease, ws::Channel::Conversations)?;
            if fields
                .query
                .keys()
                .any(|key| !matches!(key.as_str(), "limit" | "zone"))
            {
                return Err(ApiError(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "conversation_filter_unavailable",
                ));
            }
            let rows = projection(
                brain
                    .list_conversations(ListConversationsRequest {
                        limit: fields.limit(100, 200)?,
                        room_id: fields.query("zone").into(),
                        actor: actor(),
                    })
                    .await?
                    .payload_json,
            )?;
            state
                .live
                .projection(connection, ws::Channel::Conversations, rows)?
        }
        HaEntities => {
            let output = brain
                .get_home_catalog(GetHomeCatalogRequest { actor: actor() })
                .await?;
            let domain = fields.query("domain");
            let entities: Vec<_> = if output.stale {
                vec![]
            } else {
                output
                    .entities
                    .into_iter()
                    .filter(|row| domain.is_empty() || row.domain == domain)
                    .map(entity)
                    .collect()
            };
            json!(entities)
        }
        HaActions => json!(
            brain
                .list_ha_actions(ListHaActionsRequest {
                    limit: fields.limit(50, 200)?,
                    actor: actor()
                })
                .await?
                .receipts
                .into_iter()
                .map(home_receipt)
                .collect::<Vec<_>>()
        ),
        Llm => json!(
            brain
                .get_llm_endpoints(lease.actor())
                .await?
                .endpoints
                .into_iter()
                .map(llm_endpoint)
                .collect::<Vec<_>>()
        ),
        Cooldown => {
            brain
                .clear_llm_cooldown(ClearLlmCooldownRequest {
                    endpoint_id: id.into(),
                    actor: actor(),
                })
                .await?;
            json!({"ok":true,"endpoint_id":id})
        }
        Gpus => json!(state.system.gpus().await),
        Services => json!(state.system.services().await),
        Restart => {
            if !crate::sys::restartable(id) {
                return Err(MonitorError::Denied.into());
            }
            state.system.restart(id).await?;
            json!({"ok":true,"service":id,"command_completed":true})
        }
        Cloud => cloud(brain.get_cloud_consent(lease.actor()).await?, false),
        CloudWithdraw => cloud(
            brain
                .withdraw_cloud_consent(WithdrawCloudConsentRequest { actor: actor() })
                .await?,
            true,
        ),
    };
    Ok((status, Json(value)).into_response())
}
