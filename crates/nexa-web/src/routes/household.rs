use super::*;

#[derive(Clone, Copy)]
pub(super) enum Action {
    Lists,
    ListSync,
    ListItems,
    ListMutation(&'static str),
    Reminders,
    ReminderNotices,
    ReminderStatus,
    ReminderMutation(&'static str),
    Alerts,
    AlertZones,
    AlertZonesWrite,
    AlertAck,
    AlertDelete,
}

pub(super) async fn execute(
    op: Action,
    state: &AppState,
    fields: &Fields,
    lease: &Lease,
) -> Result<Response> {
    use Action::*;
    let brain = &state.clients.brain;
    let id = fields.id();
    let body = &fields.body;
    let actor = || Some(lease.actor());
    let actor_request = || ActorRequest { actor: actor() };
    let task_query = match op {
        Lists => Some(&["scope"][..]),
        Reminders => Some(&["scope", "include_completed"][..]),
        ListItems | ListSync | ReminderNotices | ReminderStatus => Some(&[][..]),
        ListMutation(_) | ReminderMutation(_) => Some(&["expected_revision"][..]),
        _ => None,
    };
    if task_query.is_some_and(|allowed| {
        fields
            .query
            .keys()
            .any(|key| !allowed.contains(&key.as_str()))
    }) {
        return Err(INVALID);
    }
    let status = StatusCode::OK;
    let value = match op {
        Lists | ListItems => {
            let scope = if matches!(op, Lists) {
                task_scope(fields.query.get("scope").map(String::as_str), lease)?
            } else {
                None
            };
            let rows = brain.list_household_lists(actor_request()).await?.lists;
            let user = lease.actor().user_id;
            if matches!(op, ListItems) {
                let row = rows
                    .into_iter()
                    .find(|row| {
                        row.list_id == id
                            && (row.owner_user_id.is_empty() || row.owner_user_id == user)
                    })
                    .ok_or(NOT_FOUND)?;
                list_items(&row)?
            } else {
                json!(
                    rows.into_iter()
                        .filter(|row| scope_matches(&row.owner_user_id, &user, scope))
                        .map(list_record)
                        .collect::<Vec<_>>()
                )
            }
        }
        ListMutation(operation) => {
            let allowed = match operation {
                "create_list" => &["name", "shared", "expected_revision"][..],
                "add_item" => &["text", "expected_revision"],
                _ => &["expected_revision"],
            };
            allowed_fields(body, allowed)?;
            if operation == "create_list" {
                bounded_text(body, "name", 200)?;
                let shared = body
                    .get("shared")
                    .map(|_| boolean(body, "shared"))
                    .transpose()?
                    .unwrap_or(false);
                if !shared {
                    lease.require(&[TasksPersonal])?;
                }
            } else if operation == "add_item" {
                bounded_text(body, "text", 500)?;
            }
            let mut payload = body.clone();
            if let Some(item) = fields.path.get("item_id") {
                payload["item_id"] = json!(item);
            }
            let record = brain
                .mutate_household_list(HouseholdListMutationRequest {
                    operation: operation.into(),
                    list_id: id.into(),
                    expected_revision: fields.optional_revision()?,
                    payload_json: payload.to_string(),
                    actor: actor(),
                })
                .await?;
            match operation {
                "delete_item" => json!({"status":"deleted","outcome":committed()}),
                "add_item" => {
                    let text = string(body, "text")?;
                    let items = list_items(&record)?;
                    let mut item = items
                        .as_array()
                        .and_then(|rows| rows.iter().rev().find(|row| row["text"] == text))
                        .cloned()
                        .ok_or(BAD_UPSTREAM)?;
                    item["outcome"] = committed();
                    item
                }
                _ => {
                    let mut result = list_record(record);
                    result["outcome"] = committed();
                    result
                }
            }
        }
        ListSync => {
            exact_fields(body, &[])?;
            projection(
                brain
                    .sync_household_lists(actor_request())
                    .await?
                    .payload_json,
            )?
        }
        ReminderNotices => projection(
            brain
                .list_reminder_notices(actor_request())
                .await?
                .payload_json,
        )?,
        ReminderStatus => projection(
            brain
                .get_reminder_status(actor_request())
                .await?
                .payload_json,
        )?,
        Reminders => {
            let scope = task_scope(fields.query.get("scope").map(String::as_str), lease)?;
            let user = lease.actor().user_id;
            let rows = brain
                .list_reminders(ListRemindersRequest {
                    include_personal: !user.is_empty() && scope != Some(false),
                    include_completed: optional_bool(&fields.query, "include_completed")?
                        .unwrap_or(false),
                    actor: actor(),
                })
                .await?
                .reminders;
            json!(
                rows.into_iter()
                    .filter(|row| scope_matches(&row.owner_user_id, &user, scope))
                    .map(reminder)
                    .collect::<Result<Vec<_>>>()?
            )
        }
        ReminderMutation(operation) => {
            let fields_allowed = match operation {
                "create" => &[
                    "text",
                    "trigger_at",
                    "scope",
                    "recurrence",
                    "ha_actions",
                    "delivery_room",
                    "expected_revision",
                ][..],
                "update" => &[
                    "text",
                    "trigger_at",
                    "scope",
                    "ha_actions",
                    "delivery_room",
                    "expected_revision",
                ],
                "dismiss_notice" => &["reminder_id", "trigger_at", "expected_revision"],
                _ => &["expected_revision"],
            };
            allowed_fields(body, fields_allowed)?;
            let mut payload = body.clone();
            if matches!(operation, "create" | "update") {
                if operation == "create" || body.get("text").is_some() {
                    bounded_text(body, "text", 500)?;
                }
                let scope = optional_string(body, "scope")?.unwrap_or("private");
                match scope {
                    "private" => lease.require(&[TasksPersonal])?,
                    "shared" => {}
                    _ => return Err(INVALID),
                }
                if operation == "update" {
                    payload["scope"] = json!(scope);
                    optional_string(body, "delivery_room")?;
                    if body
                        .get("ha_actions")
                        .is_some_and(|value| !value.is_array() && !value.is_null())
                    {
                        return Err(INVALID);
                    }
                }
            }
            if (matches!(operation, "create" | "dismiss_notice")
                || (operation == "update" && body.get("trigger_at").is_some()))
                && !body
                    .get("trigger_at")
                    .and_then(Value::as_f64)
                    .is_some_and(|time| time.is_finite() && time > 0.0)
            {
                return Err(INVALID);
            }
            let record = brain
                .mutate_reminder(ReminderMutationRequest {
                    operation: operation.into(),
                    reminder_id: id.into(),
                    expected_revision: if operation == "update" {
                        fields.revision()?
                    } else {
                        fields.optional_revision()?
                    },
                    payload_json: payload.to_string(),
                    actor: actor(),
                })
                .await?;
            if matches!(operation, "create" | "update") {
                let mut row = reminder(record)?;
                row["outcome"] = committed();
                row
            } else {
                let result = if operation == "cancel" {
                    "cancelled"
                } else {
                    &record.status
                };
                json!({"status":result,"outcome":committed()})
            }
        }
        Alerts => {
            let filter = fields.query("status");
            let alerts: Vec<_> = brain
                .list_alerts(actor_request())
                .await?
                .alerts
                .into_iter()
                .filter(|row| match filter {
                    "active" => !row.acknowledged,
                    "acked" => row.acknowledged,
                    _ => true,
                })
                .map(alert)
                .collect();
            let output = brain
                .get_alert_zones(actor_request())
                .await
                .map(alert_zones)
                .unwrap_or_else(|_| json!({"available":false}));
            json!({"alerts":alerts,"available":true,"output":output,"producers":[]})
        }
        AlertZones => alert_zones(brain.get_alert_zones(actor_request()).await?),
        AlertZonesWrite => {
            let zones = body
                .get("zones")
                .and_then(Value::as_array)
                .ok_or(INVALID)?
                .iter()
                .map(|row| row.as_str().map(str::to_owned).ok_or(INVALID))
                .collect::<Result<Vec<_>>>()?;
            alert_zones(
                brain
                    .set_alert_zones(SetAlertZonesRequest {
                        expected_revision: fields.revision()?,
                        zone_ids: zones,
                        actor: actor(),
                    })
                    .await?,
            )
        }
        AlertAck | AlertDelete => {
            let request = AlertMutationRequest {
                alert_id: id.into(),
                expected_revision: fields.optional_revision()?,
                actor: actor(),
            };
            if matches!(op, AlertAck) {
                brain.ack_alert(request).await?;
                json!({"ok":true,"alert_id":id,"acked":true})
            } else {
                brain.delete_alert(request).await?;
                json!({"ok":true,"alert_id":id,"dismissed":true})
            }
        }
    };
    Ok((status, Json(value)).into_response())
}
