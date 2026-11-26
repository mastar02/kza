use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RoutinePayload {
    pub(super) name: String,
    pub(super) enabled: bool,
    pub(super) steps: Vec<RoutineStep>,
    pub(super) schedule: Option<RoutineSchedule>,
}

pub(super) fn routine_json(spec: &RoutineSpec, schedule: Option<&RoutineSchedule>) -> String {
    let mut value = json!({"name": spec.name, "enabled": spec.enabled, "steps": spec.steps});
    if let Some(schedule) = schedule {
        value["schedule"] = json!(schedule);
    }
    value.to_string()
}

pub(super) fn routine_record(routine: Routine) -> Result<v1::RoutineRecord, Status> {
    Ok(v1::RoutineRecord {
        routine_id: routine.id.0.to_string(),
        name: routine.spec.name.clone(),
        enabled: routine.spec.enabled,
        revision: unsigned(routine.revision),
        spec_json: routine_json(&routine.spec, routine.schedule.as_ref()),
        updated_at_unix_ms: unsigned(routine.updated_at_ms),
        schedule_json: routine
            .schedule
            .as_ref()
            .map(|schedule| json!(schedule).to_string())
            .unwrap_or_default(),
    })
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(super) struct ListPayload {
    pub(super) personal: bool,
    pub(super) name: String,
    pub(super) text: String,
    pub(super) item_id: String,
    pub(super) completed: Option<bool>,
}

pub(super) fn list_record(snapshot: ListSnapshot) -> v1::HouseholdListRecord {
    let items: Vec<Value> = snapshot
        .items
        .into_iter()
        .map(|item| {
            json!({
                "item_id": item.id.0.to_string(), "text": item.text, "completed": item.completed,
            })
        })
        .collect();
    v1::HouseholdListRecord {
        list_id: snapshot.list.id.0.to_string(),
        name: snapshot.list.name,
        owner_user_id: snapshot.list.owner.unwrap_or_default(),
        revision: unsigned(snapshot.list.revision),
        items_json: json!(items).to_string(),
    }
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(super) struct ReminderPayload {
    pub(super) personal: Option<bool>,
    scope: Option<ReminderScope>,
    pub(super) title: Option<String>,
    pub(super) text: Option<String>,
    #[serde(alias = "room_id", alias = "delivery_room")]
    pub(super) room: Option<String>,
    pub(super) due_at_unix_ms: Option<i64>,
    pub(super) trigger_at: Option<serde_json::Number>,
    #[serde(alias = "ha_actions")]
    pub(super) actions: Option<Vec<ReminderHomeAction>>,
    pub(super) recurrence: Option<String>,
    pub(super) expected_revision: Option<u64>,
    pub(super) reminder_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReminderScope {
    Private,
    Shared,
}

impl ReminderPayload {
    pub(super) fn is_personal(&self) -> Result<Option<bool>, Status> {
        let scope = self
            .scope
            .as_ref()
            .map(|scope| matches!(scope, ReminderScope::Private));
        if let (Some(personal), Some(scope)) = (self.personal, scope)
            && personal != scope
        {
            return Err(Status::invalid_argument("conflicting reminder scopes"));
        }
        Ok(self.personal.or(scope))
    }

    pub(super) fn due_at(&self) -> Result<Option<i64>, Status> {
        let invalid_due = || Status::invalid_argument("invalid reminder due date");
        match (self.due_at_unix_ms, &self.trigger_at) {
            (None, None) => Ok(None),
            (Some(ms), None) if ms > 0 => Ok(Some(ms)),
            (None, Some(timestamp)) => {
                // Contemporary Unix seconds have fewer than twelve digits.
                const MILLIS_THRESHOLD: i64 = 100_000_000_000;
                if let Some(ms) = timestamp.as_i64().filter(|ms| *ms >= MILLIS_THRESHOLD) {
                    return Ok(Some(ms));
                }
                let time = timestamp.as_f64().ok_or_else(invalid_due)?;
                let millis = if time < MILLIS_THRESHOLD as f64 {
                    time * 1000.0
                } else {
                    time
                }
                .round();
                if !(1.0..(i64::MAX as f64)).contains(&millis) {
                    return Err(invalid_due());
                }
                Ok(Some(millis as i64))
            }
            _ => Err(invalid_due()),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReminderHomeAction {
    domain: Option<String>,
    service: String,
    entity_id: String,
    data: Option<serde_json::Map<String, Value>>,
}

impl ReminderHomeAction {
    pub(super) fn into_action(self) -> Result<HomeAction, Status> {
        let service = match self.domain {
            Some(domain)
                if !domain.is_empty() && !domain.contains('.') && !self.service.contains('.') =>
            {
                format!("{domain}.{}", self.service)
            }
            Some(_) => {
                return Err(Status::invalid_argument(
                    "invalid reminder action domain or service",
                ));
            }
            None => self.service,
        };
        Ok(HomeAction {
            service,
            entity_id: self.entity_id,
            data: self.data.unwrap_or_default(),
        })
    }
}

pub(super) fn reminder_record(
    db: &nexa_household::Household<Arc<dyn crate::BrainClock>>,
    value: Reminder,
) -> nexa_household::Result<v1::ReminderRecord> {
    let status = match value.state {
        ReminderState::Pending => "pending",
        ReminderState::Completed => "completed",
        ReminderState::Cancelled => "cancelled",
    };
    let mut payload = json!({"room_id": value.room, "delivery_room": value.room});
    if value.owner.is_none() {
        let actions: Vec<_> = db
            .reminder_actions(value.id)?
            .into_iter()
            .map(|action| {
                let command = action.command;
                let (domain, service) = command.service.split_once('.').unwrap_or_else(|| {
                    let domain = command
                        .entity_id
                        .split_once('.')
                        .map_or("", |(domain, _)| domain);
                    (domain, &command.service)
                });
                json!({"domain": domain, "service": service, "entity_id": command.entity_id})
            })
            .collect();
        if !actions.is_empty() {
            payload["ha_actions"] = json!(actions);
        }
    }
    Ok(v1::ReminderRecord {
        reminder_id: value.id.0.to_string(),
        owner_user_id: value.owner.unwrap_or_default(),
        title: value.message,
        status: status.into(),
        due_at_unix_ms: unsigned(value.due_at),
        revision: unsigned(value.revision),
        payload_json: payload.to_string(),
    })
}

pub(super) fn alert_record(value: Alert) -> v1::AlertRecord {
    let severity = match value.priority {
        AlertPriority::Low => "low",
        AlertPriority::Medium => "medium",
        AlertPriority::High => "high",
        AlertPriority::Critical => "critical",
    };
    let status = if value.resolved_at.is_some() {
        "resolved"
    } else if value.acknowledged_at.is_some() {
        "acknowledged"
    } else {
        "active"
    };
    v1::AlertRecord {
        alert_id: value.id.0.to_string(),
        kind: value.key,
        severity: severity.into(),
        status: status.into(),
        message: value.message,
        revision: unsigned(value.revision),
        acknowledged: value.acknowledged_at.is_some(),
        created_at_unix_ms: unsigned(value.opened_at),
    }
}
