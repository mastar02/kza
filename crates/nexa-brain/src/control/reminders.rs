use super::*;

pub(super) async fn list_reminders(
    control: &Control,
    request: Request<v1::ListRemindersRequest>,
) -> Reply<v1::RemindersSnapshot> {
    let request = request.into_inner();
    let actor = authorize(request.actor.as_ref(), &[P::TasksRead])?;
    let owner = if request.include_personal {
        Some(personal(actor)?)
    } else {
        None
    };
    let include_completed = request.include_completed;
    let reminders = control
        .0
        .household
        .call(move |db| {
            let mut reminders = db.list_reminders_all(None, include_completed)?;
            if let Some(owner) = owner {
                reminders.extend(db.list_reminders_all(Some(&owner), include_completed)?);
            }
            reminders.sort_by_key(|reminder| (reminder.due_at, reminder.id.0));
            reminders
                .into_iter()
                .map(|reminder| reminder_record(db, reminder))
                .collect()
        })
        .await
        .map_err(household_status)?;
    ok(v1::RemindersSnapshot { reminders })
}

pub(super) async fn mutate_reminder(
    control: &Control,
    request: Request<v1::ReminderMutationRequest>,
) -> Reply<v1::ReminderRecord> {
    let request = request.into_inner();
    let actor = authorize(request.actor.as_ref(), &[P::TasksWrite])?;
    let body: ReminderPayload = parse_json(if request.payload_json.is_empty() {
        "{}"
    } else {
        &request.payload_json
    })?;
    let operation = request.operation.as_str();
    let scope = body.is_personal()?;
    let owner = if scope.unwrap_or(body.text.is_some()) {
        Some(personal(actor)?)
    } else {
        None
    };
    let raw_id = body.reminder_id.as_deref().unwrap_or(&request.reminder_id);
    if !request.reminder_id.is_empty() && raw_id != request.reminder_id {
        return Err(Status::invalid_argument("conflicting reminder IDs"));
    }
    let raw_revision = body.expected_revision.unwrap_or(request.expected_revision);
    if request.expected_revision != 0 && raw_revision != request.expected_revision {
        return Err(Status::invalid_argument("conflicting reminder revisions"));
    }
    let expected = revision(raw_revision)?;
    let due = body.due_at()?;
    let record = match operation {
        "create" | "update" => {
            let reminder_id = if operation == "create" {
                if expected != 0 || !raw_id.is_empty() {
                    return Err(Status::invalid_argument(
                        "invalid new reminder ID or revision",
                    ));
                }
                None
            } else {
                Some(ReminderId(id(raw_id)?))
            };
            if body.recurrence.is_some() {
                return Err(Status::invalid_argument(
                    "recurring reminders are unsupported",
                ));
            }
            let message = match (body.title, body.text) {
                (Some(_), Some(_)) => {
                    return Err(Status::invalid_argument("conflicting reminder titles"));
                }
                (title, text) => title.or(text).unwrap_or_default(),
            };
            let room = body
                .room
                .or_else(|| {
                    body.trigger_at.as_ref()?;
                    control
                        .0
                        .room_for_zone(&control.0.config.core.household.default_output_zone)
                        .map(str::to_owned)
                })
                .filter(|room| control.0.zone(room))
                .ok_or_else(|| Status::invalid_argument("invalid reminder room"))?;
            let due_at = due.ok_or_else(|| Status::invalid_argument("a due date is required"))?;
            let actions = body
                .actions
                .unwrap_or_default()
                .into_iter()
                .map(ReminderHomeAction::into_action)
                .collect::<Result<Vec<_>, _>>()?;
            if !actions.is_empty() {
                authorize(Some(actor), &[P::HomeControl])?;
                for action in &actions {
                    control.0.allow_home(&scheduled_action(action)?)?;
                }
            }
            control
                .0
                .household
                .call(move |db| {
                    let owner = owner.as_deref();
                    let reminder = if let Some(id) = reminder_id {
                        db.update_reminder(
                            id,
                            owner,
                            expected,
                            &ReminderUpdate {
                                message: &message,
                                room: &room,
                                due_at,
                                actions: &actions,
                            },
                        )?
                    } else {
                        db.create_reminder(&message, owner, &room, due_at, &actions)?
                    };
                    reminder_record(db, reminder)
                })
                .await
        }
        "cancel" | "dismiss_notice" => {
            let id = ReminderId(id(raw_id)?);
            let cancel = operation == "cancel";
            // The web DELETE route supplies only an ID and revision. An
            // unscoped request may cancel a shared or the actor's own row.
            let own_cancel = (cancel && scope.is_none() && has(actor, P::TasksPersonal))
                .then(|| actor.user_id.clone());
            control
                .0
                .household
                .call(move |db| {
                    use nexa_household::Error;
                    let owner = owner.as_deref();
                    let record = match (db.reminder(id, owner)?, own_cancel.as_deref()) {
                        (None, Some(own)) => db.reminder(id, Some(own))?,
                        (record, _) => record,
                    }
                    .ok_or(Error::NotFound("reminder"))?;
                    if record.revision != expected {
                        return Err(Error::Conflict("reminder revision changed"));
                    }
                    let owner = record.owner.as_deref();
                    if cancel {
                        if !db.cancel_reminder(id, owner)? {
                            return Err(Error::Conflict("reminder cannot be cancelled"));
                        }
                    } else if owner.is_some()
                        || due.is_some_and(|due| due != record.due_at)
                        || !db.dismiss_reminder_notice(id)?
                    {
                        return Err(Error::Conflict("notice cannot be dismissed"));
                    }
                    let reminder = db.reminder(id, owner)?.ok_or(Error::NotFound("reminder"))?;
                    reminder_record(db, reminder)
                })
                .await
        }
        _ => return Err(Status::invalid_argument("unknown reminder operation")),
    }
    .map_err(household_status)?;
    ok(record)
}

pub(super) async fn list_reminder_notices(
    control: &Control,
    request: Request<v1::ActorRequest>,
) -> Reply<v1::ReminderNoticesResponse> {
    authorize(request.into_inner().actor.as_ref(), &[P::TasksRead])?;
    // Household only creates delivery notices for shared reminders.
    let notices = control
        .0
        .household
        .call(|db| db.list_reminder_notices(None))
        .await
        .map_err(household_status)?;
    let notices: Vec<_> = notices
        .into_iter()
        .map(|notice| {
            let state = match notice.receipt.state {
                NoticeState::Pending => "pending",
                NoticeState::Claimed => "claimed",
                NoticeState::Delivered => "delivered",
                NoticeState::Dismissed => "dismissed",
                NoticeState::Uncertain => "uncertain",
                NoticeState::Cancelled => "cancelled",
                NoticeState::Parked => "parked",
            };
            json!({
                "reminder_id": notice.reminder.0.to_string(),
                "state": state,
                "attempts": notice.receipt.attempts,
                "next_attempt_at_unix_ms": notice.receipt.next_attempt_at,
            })
        })
        .collect();
    ok(v1::ReminderNoticesResponse {
        payload_json: json!({ "notices": notices }).to_string(),
    })
}

pub(super) async fn get_reminder_status(
    control: &Control,
    request: Request<v1::ActorRequest>,
) -> Reply<v1::ReminderStatusResponse> {
    authorize(request.into_inner().actor.as_ref(), &[P::DiagnosticsRead])?;
    let status = control
        .0
        .household
        .call(|db| db.reminder_status())
        .await
        .map_err(household_status)?;
    ok(v1::ReminderStatusResponse {
        payload_json: serde_json::to_string(&status)
            .map_err(|_| Status::internal("could not encode reminder status"))?,
    })
}
