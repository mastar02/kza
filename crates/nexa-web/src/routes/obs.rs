use super::*;

#[derive(Clone, Copy)]
pub(super) enum Action {
    Zones,
    ZoneVolume,
    Conversations,
    HaEntities,
    HaActions,
    Llm,
    Cooldown,
    Gpus,
    Services,
    Daemons,
    ReloadMlSlot,
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
        ZoneVolume => {
            exact_fields(&fields.body, &["volume_pct"])?;
            let pct = unsigned(&fields.body, "volume_pct")?;
            if pct > 100 {
                return Err(INVALID);
            }
            let updated = audio
                .set_zone_volume(SetZoneVolumeRequest {
                    zone_id: memory_id(id)?.into(),
                    volume_pct: pct as u32,
                    actor: actor(),
                })
                .await?;
            zone(updated)
        }
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
            let mut rows = projection(
                brain
                    .list_conversations(ListConversationsRequest {
                        limit: fields.limit(100, 200)?,
                        room_id: fields.query("zone").into(),
                        actor: actor(),
                    })
                    .await?
                    .payload_json,
            )?;
            normalize_conversations(&mut rows);
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
        Daemons => {
            let (audio_res, brain_res, memory_res, ml_res) = tokio::join!(
                audio.get_health(Empty {}),
                brain.get_health(Empty {}),
                state.clients.memory.get_health(lease.actor()),
                state.clients.ml.get_health(Empty {}),
            );
            let daemons = json!({
                "audio": daemon_health("nexa-audiod", audio_res),
                "brain": daemon_health("nexa-brain", brain_res),
                "memory": daemon_health("nexa-memoryd", memory_res),
                "ml": ml_daemon_health(ml_res),
            });
            json!({ "daemons": daemons })
        }
        ReloadMlSlot => {
            let slot_id = memory_id(id)?;
            let resp = state
                .clients
                .ml
                .reload_model(ReloadModelRequest {
                    slot_id: slot_id.into(),
                    actor: actor(),
                })
                .await?;
            json!({
                "ok": true,
                "slot_id": slot_id,
                "state": health_state_name(resp.state),
                "detail": resp.detail,
            })
        }
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

fn normalize_conversations(doc: &mut Value) {
    let Some(items) = doc.as_array_mut() else {
        return;
    };
    for item in items {
        let Some(row) = item.as_object_mut() else {
            continue;
        };
        if !row.contains_key("id")
            && let Some(id) = row.get("conversation_id").cloned()
        {
            row.insert("id".into(), id);
        }
        if !row.contains_key("ts") {
            let ts = row
                .get("observed_at_unix_ms")
                .and_then(Value::as_i64)
                .and_then(chrono::DateTime::from_timestamp_millis)
                .map(|dt| dt.format("%H:%M:%S UTC").to_string())
                .unwrap_or_default();
            row.insert("ts".into(), json!(ts));
        }
        if !row.contains_key("zone") {
            let zone = row
                .get("room_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            row.insert("zone".into(), json!(zone));
        }
        let routing = row.get("routing").and_then(Value::as_object).cloned();
        if !row.contains_key("user") {
            let user = routing
                .as_ref()
                .and_then(|r| {
                    r.get("speaker_id")
                        .or_else(|| r.get("user_id"))
                        .or_else(|| r.get("user"))
                        .and_then(Value::as_str)
                })
                .unwrap_or_default()
                .to_owned();
            row.insert("user".into(), json!(user));
        }
        if !row.contains_key("stt") {
            let stt = routing
                .as_ref()
                .and_then(|r| {
                    r.get("transcript")
                        .or_else(|| r.get("stt"))
                        .or_else(|| r.get("raw_text"))
                        .and_then(Value::as_str)
                })
                .unwrap_or_default()
                .to_owned();
            row.insert("stt".into(), json!(stt));
        }
        if !row.contains_key("tts") {
            let tts = routing
                .as_ref()
                .and_then(|r| {
                    r.get("spoken_reply")
                        .or_else(|| r.get("tts"))
                        .or_else(|| r.get("reply"))
                        .and_then(Value::as_str)
                })
                .unwrap_or_default()
                .to_owned();
            row.insert("tts".into(), json!(tts));
        }
    }
}
