//! Permission-checked local control service and private Unix listener.

#![allow(
    clippy::result_large_err,
    reason = "Tonic status carries gRPC metadata and codes across this boundary"
)]

use std::{
    fs, io,
    os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use nexa_household::{
    HomeAction,
    alerts::{Alert, AlertId, AlertPreferences, AlertPriority},
    lists::{ItemId, ListId, ListSnapshot},
    reminders::{NoticeState, Reminder, ReminderId, ReminderState, ReminderUpdate},
    routines::{
        Occurrence, Routine, RoutineId, RoutineSchedule, RoutineSpec, RoutineStep, RunClaim,
        RunState, ScheduledRoutineSpec,
    },
};
use nexa_proto::v1::{
    self as v1, Permission as P, Role,
    brain_control_service_server::{BrainControlService, BrainControlServiceServer},
};
use prost::Message;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::net::UnixListener;
use tonic::{Request, Response, Status, transport::Server};

use crate::{
    Brain, BrainError,
    clients::{MemoryClient, StreamOf},
    home::{scheduled_action, typed_operation},
    tasks::household_status,
    unsigned,
};

/// Binary ActorContext metadata for sensitive methods whose proto body is Empty.
pub const ACTOR_METADATA: &str = "x-nexa-actor-bin";

#[derive(Debug, Clone)]
pub struct Control(pub Arc<Brain>);

type Reply<T> = Result<Response<T>, Status>;

fn ok<T>(value: T) -> Reply<T> {
    Ok(Response::new(value))
}

fn valid_identifier(value: &str) -> bool {
    (1..=128).contains(&value.len())
        && !value.chars().any(char::is_control)
        && !value.trim().is_empty()
}

fn authorize<'a>(
    actor: Option<&'a v1::ActorContext>,
    permissions: &[P],
) -> Result<&'a v1::ActorContext, Status> {
    let actor = actor
        .filter(|actor| valid_identifier(&actor.subject))
        .ok_or_else(|| Status::unauthenticated("ActorContext required"))?;
    let role = Role::try_from(actor.role)
        .ok()
        .filter(|role| *role != Role::Unspecified)
        .ok_or_else(|| Status::unauthenticated("recognized actor role required"))?;
    if !actor.user_id.is_empty() && (role != Role::Home || !valid_identifier(&actor.user_id)) {
        return Err(Status::permission_denied("invalid personal actor scope"));
    }
    for &raw in &actor.permissions {
        let permission =
            P::try_from(raw).map_err(|_| Status::permission_denied("unknown actor permission"))?;
        let personal = matches!(
            permission,
            P::TasksPersonal | P::MemoryOwnRead | P::MemoryOwnWrite | P::MemoryOwnVoice
        );
        let allowed = match role {
            Role::Operator => permission != P::Unspecified && !personal,
            Role::Home => {
                matches!(
                    permission,
                    P::HomeRead
                        | P::HomeControl
                        | P::TasksRead
                        | P::TasksWrite
                        | P::RoutinesExecute
                ) || personal && !actor.user_id.is_empty()
            }
            Role::Evaluator => permission == P::EvaluationRead,
            Role::Unspecified => false,
        };
        if !allowed {
            return Err(Status::permission_denied(
                "permission is outside the actor role",
            ));
        }
    }
    if !permissions.iter().all(|permission| has(actor, *permission)) {
        return Err(Status::permission_denied("required permission missing"));
    }
    Ok(actor)
}

fn has(actor: &v1::ActorContext, permission: P) -> bool {
    actor.permissions.contains(&(permission as i32))
}

fn authorize_metadata(request: &Request<v1::Empty>, permissions: &[P]) -> Result<(), Status> {
    let encoded = request
        .metadata()
        .get_bin(ACTOR_METADATA)
        .ok_or_else(|| Status::unauthenticated("ActorContext metadata required"))?;
    let bytes = encoded
        .to_bytes()
        .map_err(|_| Status::unauthenticated("invalid ActorContext metadata"))?;
    let actor = v1::ActorContext::decode(bytes)
        .map_err(|_| Status::unauthenticated("invalid ActorContext metadata"))?;
    authorize(Some(&actor), permissions)?;
    Ok(())
}

fn id(value: &str) -> Result<i64, Status> {
    value
        .parse::<i64>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| Status::invalid_argument("invalid household ID"))
}

fn revision(value: u64) -> Result<i64, Status> {
    i64::try_from(value).map_err(|_| Status::invalid_argument("invalid revision"))
}

fn limit(value: u32) -> usize {
    if value == 0 {
        50
    } else {
        value.min(128) as usize
    }
}

fn parse_json<T: serde::de::DeserializeOwned>(value: &str) -> Result<T, Status> {
    if value.len() > 64 * 1024 {
        return Err(Status::resource_exhausted("JSON payload exceeds limit"));
    }
    serde_json::from_str(value).map_err(|_| Status::invalid_argument("invalid JSON payload"))
}

fn personal(actor: &v1::ActorContext) -> Result<String, Status> {
    authorize(Some(actor), &[P::TasksPersonal])?;
    if actor.user_id.is_empty() {
        return Err(Status::permission_denied("person-bound actor required"));
    }
    Ok(actor.user_id.clone())
}

impl Control {
    fn memory(&self) -> Result<&dyn MemoryClient, Status> {
        self.0
            .memory
            .as_deref()
            .ok_or_else(|| Status::unavailable("memory service unavailable"))
    }

    fn routine_spec(&self, value: &str) -> Result<ScheduledRoutineSpec, Status> {
        let payload: RoutinePayload = parse_json(value)?;
        let spec = RoutineSpec {
            name: payload.name,
            enabled: payload.enabled,
            steps: payload.steps,
        };
        if spec.name.trim().is_empty()
            || spec.name.len() > 128
            || !(1..=32).contains(&spec.steps.len())
        {
            return Err(Status::invalid_argument(
                "routine requires a name and 1 to 32 steps",
            ));
        }
        for step in &spec.steps {
            match step {
                RoutineStep::Home { action } => self.0.allow_home(&scheduled_action(action)?)?,
                RoutineStep::Announce { room, message } => {
                    if !self.0.zone(room) || message.trim().is_empty() || message.len() > 1024 {
                        return Err(Status::invalid_argument("invalid routine announcement"));
                    }
                }
                RoutineStep::Delay { duration } if *duration > Duration::from_secs(86400) => {
                    return Err(Status::invalid_argument("routine delay exceeds one day"));
                }
                RoutineStep::Delay { .. } => {}
            }
        }
        if let Some(RoutineSchedule::Daily {
            hour,
            minute,
            timezone,
            days,
        }) = &payload.schedule
            && (chrono::NaiveTime::from_hms_opt(*hour, *minute, 0).is_none()
                || timezone.parse::<chrono_tz::Tz>().is_err()
                || days.iter().any(|day| !(1..=7).contains(day)))
        {
            return Err(Status::invalid_argument("invalid daily routine schedule"));
        }
        Ok(ScheduledRoutineSpec {
            spec,
            schedule: payload.schedule,
        })
    }

    fn alert_zones(&self, config: AlertPreferences) -> v1::AlertZonesConfig {
        v1::AlertZonesConfig {
            revision: unsigned(config.revision),
            zone_ids: config.zones.unwrap_or_else(|| {
                let default = &self.0.config.core.household.default_output_zone;
                if self.0.room_for_zone(default).is_some() {
                    vec![default.clone()]
                } else {
                    Vec::new()
                }
            }),
        }
    }
}

#[tonic::async_trait]
impl BrainControlService for Control {
    async fn get_health(&self, _: Request<v1::Empty>) -> Reply<v1::HealthResponse> {
        ok(v1::HealthResponse {
            service: "nexa-brain".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            state: if self.0.shutdown.is_cancelled() {
                v1::HealthState::Unavailable
            } else {
                v1::HealthState::Ok
            }
            .into(),
            uptime_ms: u64::try_from(self.0.started.elapsed().as_millis()).unwrap_or(u64::MAX),
            detail: String::new(),
        })
    }

    async fn get_home_catalog(
        &self,
        request: Request<v1::GetHomeCatalogRequest>,
    ) -> Reply<v1::HomeCatalogSnapshot> {
        authorize(request.get_ref().actor.as_ref(), &[P::HomeRead])?;
        ok(self.0.catalog())
    }

    async fn execute_home_action(
        &self,
        request: Request<v1::HomeActionRequest>,
    ) -> Reply<v1::HomeActionReceipt> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::HomeControl])?;
        let operation = typed_operation(&request.operation, &request.params)?;
        let action = nexa_ha::EffectIntent {
            entity_id: request.entity_id,
            operation,
        };
        ok(self.0.execute_home(request.request_id, action).await?)
    }

    async fn get_home_action_receipt(
        &self,
        request: Request<v1::GetHomeActionReceiptRequest>,
    ) -> Reply<v1::HomeActionReceipt> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::HomeRead])?;
        ok(self
            .0
            .home_receipt(&request.request_id)
            .ok_or_else(|| Status::not_found("receipt not available"))?)
    }

    async fn get_llm_endpoints(
        &self,
        request: Request<v1::Empty>,
    ) -> Reply<v1::LlmEndpointsSnapshot> {
        authorize_metadata(&request, &[P::DiagnosticsRead])?;
        let mut endpoints = Vec::new();
        for endpoint in &self.0.endpoints {
            let remaining = self.0.llm.cooldown_remaining(&endpoint.id).await;
            let during = !remaining.is_zero();
            let until = if during {
                self.0
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
                consecutive_errors: self.0.llm.consecutive_errors(&endpoint.id).await,
            });
        }
        ok(v1::LlmEndpointsSnapshot { endpoints })
    }

    async fn clear_llm_cooldown(
        &self,
        request: Request<v1::ClearLlmCooldownRequest>,
    ) -> Reply<v1::Empty> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::AdminManage])?;
        if !self
            .0
            .endpoints
            .iter()
            .any(|endpoint| endpoint.id == request.endpoint_id)
        {
            return Err(Status::not_found("unknown LLM endpoint"));
        }
        self.0
            .llm
            .clear_cooldown(&request.endpoint_id)
            .await
            .map_err(|_| Status::internal("cooldown update failed"))?;
        self.0.event(
            v1::LiveEventKind::Cooldown,
            "",
            "",
            json!({"event": "cleared", "endpoint_id": request.endpoint_id}),
        );
        ok(v1::Empty {})
    }

    async fn get_cloud_consent(
        &self,
        request: Request<v1::Empty>,
    ) -> Reply<v1::CloudConsentStatus> {
        authorize_metadata(&request, &[P::DiagnosticsRead])?;
        ok(self.0.cloud_status().await)
    }

    async fn withdraw_cloud_consent(
        &self,
        request: Request<v1::WithdrawCloudConsentRequest>,
    ) -> Reply<v1::CloudConsentStatus> {
        authorize(request.get_ref().actor.as_ref(), &[P::AdminManage])?;
        ok(self.0.withdraw_cloud().await?)
    }

    type SubscribeLiveEventsStream = StreamOf<Result<v1::LiveRuntimeEvent, Status>>;

    async fn subscribe_live_events(
        &self,
        request: Request<v1::SubscribeLiveEventsRequest>,
    ) -> Reply<Self::SubscribeLiveEventsStream> {
        let actor = authorize(request.get_ref().actor.as_ref(), &[])?;
        let conversations = has(actor, P::ConversationsRead);
        let diagnostics = has(actor, P::DiagnosticsRead);
        let home = has(actor, P::HomeRead);
        let voices = has(actor, P::VoicesRead) && has(actor, P::VoicesReview);
        let evaluation = has(actor, P::EvaluationRead);
        if !(conversations || diagnostics || home || voices || evaluation) {
            return Err(Status::permission_denied("live events permission required"));
        }
        let mut events = self.0.events.subscribe();
        let shutdown = self.0.shutdown_token();
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

    async fn list_routines(
        &self,
        request: Request<v1::ListRoutinesRequest>,
    ) -> Reply<v1::RoutinesSnapshot> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::HomeRead])?;
        let routines = self
            .0
            .household
            .call(|db| db.routines())
            .await
            .map_err(household_status)?;
        ok(v1::RoutinesSnapshot {
            routines: routines
                .into_iter()
                .filter(|routine| !request.enabled_only.unwrap_or(false) || routine.spec.enabled)
                .map(routine_record)
                .collect::<Result<_, _>>()?,
        })
    }

    async fn get_routine(
        &self,
        request: Request<v1::GetRoutineRequest>,
    ) -> Reply<v1::RoutineRecord> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::HomeRead])?;
        let id = RoutineId(id(&request.routine_id)?);
        let routine = self
            .0
            .household
            .call(move |db| db.routine(id))
            .await
            .map_err(household_status)?
            .ok_or_else(|| Status::not_found("routine not found"))?;
        ok(routine_record(routine)?)
    }

    async fn preview_routine(
        &self,
        request: Request<v1::PreviewRoutineRequest>,
    ) -> Reply<v1::RoutinePreviewResponse> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::RoutinesManage])?;
        match self.routine_spec(&request.spec_json) {
            Ok(spec) => ok(v1::RoutinePreviewResponse {
                valid: true,
                preview_json: routine_json(&spec.spec, spec.schedule.as_ref()),
                errors: Vec::new(),
            }),
            Err(error) => ok(v1::RoutinePreviewResponse {
                valid: false,
                preview_json: String::new(),
                errors: vec![error.message().into()],
            }),
        }
    }

    async fn upsert_routine(
        &self,
        request: Request<v1::UpsertRoutineRequest>,
    ) -> Reply<v1::RoutineRecord> {
        let request = request.into_inner();
        let actor = authorize(request.actor.as_ref(), &[P::RoutinesManage])?;
        let spec = self.routine_spec(&request.spec_json)?;
        if spec
            .spec
            .steps
            .iter()
            .any(|step| matches!(step, RoutineStep::Home { .. }))
        {
            authorize(Some(actor), &[P::HomeControl])?;
        }
        let expected = revision(request.expected_revision)?;
        let id = if request.routine_id.is_empty() {
            None
        } else {
            Some(RoutineId(id(&request.routine_id)?))
        };
        let routine = self
            .0
            .household
            .call(move |db| match id {
                None if expected == 0 => db.create_scheduled_routine(&spec),
                None => Err(nexa_household::Error::Conflict(
                    "new routine cannot have a revision",
                )),
                Some(id) => {
                    let conflict = || nexa_household::Error::Conflict("routine revision changed");
                    let current = db
                        .routine(id)?
                        .filter(|current| current.revision == expected)
                        .ok_or_else(conflict)?;
                    let updated = if current.spec.enabled != spec.spec.enabled
                        && current.spec.name == spec.spec.name
                        && current.spec.steps == spec.spec.steps
                        && current.schedule == spec.schedule
                    {
                        db.set_routine_enabled(id, expected, spec.spec.enabled)?
                    } else {
                        db.update_scheduled_routine(id, expected, &spec)?
                    };
                    updated.ok_or_else(conflict)
                }
            })
            .await
            .map_err(household_status)?;
        ok(routine_record(routine)?)
    }

    async fn delete_routine(&self, request: Request<v1::DeleteRoutineRequest>) -> Reply<v1::Empty> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::RoutinesManage])?;
        let id = RoutineId(id(&request.routine_id)?);
        let expected = revision(request.expected_revision)?;
        let changed = self
            .0
            .household
            .call(move |db| db.delete_routine(id, expected))
            .await
            .map_err(household_status)?;
        if !changed {
            return Err(Status::aborted("routine revision changed"));
        }
        ok(v1::Empty {})
    }

    async fn execute_routine(
        &self,
        request: Request<v1::ExecuteRoutineRequest>,
    ) -> Reply<v1::RoutineExecutionReceipt> {
        let request = request.into_inner();
        let actor = authorize(request.actor.as_ref(), &[P::HomeControl])?;
        if !(has(actor, P::RoutinesManage) || has(actor, P::RoutinesExecute)) {
            return Err(Status::permission_denied(
                "routine execution permission required",
            ));
        }
        if !valid_identifier(&request.request_id) {
            return Err(Status::invalid_argument("invalid routine request ID"));
        }
        let id = RoutineId(id(&request.routine_id)?);
        let result = self
            .0
            .household
            .call(move |db| db.claim_routine(id, Occurrence::Manual(&request.request_id), None))
            .await
            .map_err(household_status)?;
        let run = match result {
            RunClaim::Created(run) | RunClaim::Existing(run) => run,
        };
        let status = match run.state {
            RunState::Running => "running",
            RunState::Succeeded => "succeeded",
            RunState::Failed => "failed",
            RunState::Stopped => "stopped",
            RunState::Skipped => "skipped",
            RunState::Interrupted => "interrupted",
        };
        ok(v1::RoutineExecutionReceipt {
            run_id: run.id.0.to_string(),
            routine_id: run.routine.0.to_string(),
            status: status.into(),
            accepted_at_unix_ms: unsigned(run.started_at),
        })
    }

    async fn list_household_lists(
        &self,
        request: Request<v1::ActorRequest>,
    ) -> Reply<v1::HouseholdListsSnapshot> {
        let actor = authorize(request.get_ref().actor.as_ref(), &[P::TasksRead])?;
        let owner = if has(actor, P::TasksPersonal) {
            Some(personal(actor)?)
        } else {
            None
        };
        let lists = self
            .0
            .household
            .call(move |db| {
                let mut lists = db.lists(None)?;
                if let Some(owner) = &owner {
                    lists.extend(db.lists(Some(owner))?);
                }
                lists
                    .into_iter()
                    .map(|list| {
                        db.list_snapshot(list.id, list.owner.as_deref())?
                            .ok_or(nexa_household::Error::NotFound("list"))
                            .map(list_record)
                    })
                    .collect()
            })
            .await
            .map_err(household_status)?;
        ok(v1::HouseholdListsSnapshot { lists })
    }

    async fn mutate_household_list(
        &self,
        request: Request<v1::HouseholdListMutationRequest>,
    ) -> Reply<v1::HouseholdListRecord> {
        let request = request.into_inner();
        let actor = authorize(request.actor.as_ref(), &[P::TasksWrite])?;
        let body: ListPayload = parse_json(if request.payload_json.is_empty() {
            "{}"
        } else {
            &request.payload_json
        })?;
        let owner = if body.personal {
            Some(personal(actor)?)
        } else {
            None
        };
        let expected = revision(request.expected_revision)?;
        let operation = request.operation;
        let list_id = if operation == "create_list" {
            None
        } else {
            Some(ListId(id(&request.list_id)?))
        };
        ok(self
            .0
            .household
            .call(move |db| {
                use nexa_household::Error;
                let owner = owner.as_deref();
                let Some(id) = list_id else {
                    if expected != 0 {
                        return Err(Error::Conflict("new list cannot have a revision"));
                    }
                    let list = db.create_list(&body.name, owner, None)?;
                    return Ok(list_record(ListSnapshot {
                        list,
                        items: Vec::new(),
                    }));
                };
                let snapshot = db
                    .list_snapshot(id, owner)?
                    .ok_or(Error::NotFound("list"))?;
                if snapshot.list.revision != expected {
                    return Err(Error::Conflict("list revision changed"));
                }
                match operation.as_str() {
                    "rename_list" => {
                        db.rename_list(id, owner, &body.name)?;
                    }
                    "delete_list" => {
                        db.delete_list(id, owner)?;
                        return Ok(list_record(ListSnapshot {
                            list: snapshot.list,
                            items: Vec::new(),
                        }));
                    }
                    "add_item" => {
                        db.add_list_item(id, owner, &body.text)?;
                    }
                    "toggle_item" | "delete_item" => {
                        let item_id = body
                            .item_id
                            .parse::<i64>()
                            .ok()
                            .map(ItemId)
                            .ok_or(Error::InvalidInput("invalid item ID"))?;
                        let item = snapshot
                            .items
                            .iter()
                            .find(|item| item.id == item_id)
                            .ok_or(Error::NotFound("item"))?;
                        if operation == "toggle_item" {
                            db.update_list_item(
                                id,
                                owner,
                                item.id,
                                &item.text,
                                body.completed.unwrap_or(!item.completed),
                            )?;
                        } else {
                            db.delete_list_item(id, owner, item.id)?;
                        }
                    }
                    _ => return Err(Error::InvalidInput("unknown list operation")),
                }
                db.list_snapshot(id, owner)?
                    .map(list_record)
                    .ok_or(Error::NotFound("list"))
            })
            .await
            .map_err(household_status)?)
    }

    async fn list_reminders(
        &self,
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
        let reminders = self
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

    async fn mutate_reminder(
        &self,
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
                        self.0
                            .room_for_zone(&self.0.config.core.household.default_output_zone)
                            .map(str::to_owned)
                    })
                    .filter(|room| self.0.zone(room))
                    .ok_or_else(|| Status::invalid_argument("invalid reminder room"))?;
                let due_at =
                    due.ok_or_else(|| Status::invalid_argument("a due date is required"))?;
                let actions = body
                    .actions
                    .unwrap_or_default()
                    .into_iter()
                    .map(ReminderHomeAction::into_action)
                    .collect::<Result<Vec<_>, _>>()?;
                if !actions.is_empty() {
                    authorize(Some(actor), &[P::HomeControl])?;
                    for action in &actions {
                        self.0.allow_home(&scheduled_action(action)?)?;
                    }
                }
                self.0
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
                self.0
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
                        let reminder =
                            db.reminder(id, owner)?.ok_or(Error::NotFound("reminder"))?;
                        reminder_record(db, reminder)
                    })
                    .await
            }
            _ => return Err(Status::invalid_argument("unknown reminder operation")),
        }
        .map_err(household_status)?;
        ok(record)
    }

    async fn list_alerts(&self, request: Request<v1::ActorRequest>) -> Reply<v1::AlertsSnapshot> {
        authorize(request.get_ref().actor.as_ref(), &[P::HomeRead])?;
        let alerts = self
            .0
            .household
            .call(|db| db.active_alerts())
            .await
            .map_err(household_status)?;
        ok(v1::AlertsSnapshot {
            alerts: alerts.into_iter().map(alert_record).collect(),
        })
    }

    async fn ack_alert(
        &self,
        request: Request<v1::AlertMutationRequest>,
    ) -> Reply<v1::AlertRecord> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::AdminManage])?;
        let id = AlertId(id(&request.alert_id)?);
        let expected = revision(request.expected_revision)?;
        let value = self
            .0
            .household
            .call(move |db| {
                if !db.acknowledge_alert(id, expected)? {
                    return Err(nexa_household::Error::Conflict(
                        "alert changed or is inactive",
                    ));
                }
                db.alert(id)?
                    .ok_or(nexa_household::Error::NotFound("alert"))
            })
            .await
            .map_err(household_status)?;
        self.0.event(
            v1::LiveEventKind::Alert,
            "",
            "",
            json!({"event": "acknowledged", "alert_id": id.0}),
        );
        ok(alert_record(value))
    }

    async fn delete_alert(&self, request: Request<v1::AlertMutationRequest>) -> Reply<v1::Empty> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::AdminManage])?;
        let id = AlertId(id(&request.alert_id)?);
        let expected = revision(request.expected_revision)?;
        self.0
            .household
            .call(move |db| {
                let value = db
                    .alert(id)?
                    .ok_or(nexa_household::Error::NotFound("alert"))?;
                if value.revision != expected || !db.resolve_alert(id)? {
                    return Err(nexa_household::Error::Conflict(
                        "alert changed or is inactive",
                    ));
                }
                Ok(())
            })
            .await
            .map_err(household_status)?;
        self.0.event(
            v1::LiveEventKind::Alert,
            "",
            "",
            json!({"event": "resolved", "alert_id": id.0}),
        );
        ok(v1::Empty {})
    }

    async fn get_alert_zones(
        &self,
        request: Request<v1::ActorRequest>,
    ) -> Reply<v1::AlertZonesConfig> {
        authorize(request.get_ref().actor.as_ref(), &[P::HomeRead])?;
        let preferences = self
            .0
            .household
            .call(|db| db.alert_preferences())
            .await
            .map_err(household_status)?;
        ok(self.alert_zones(preferences))
    }

    async fn set_alert_zones(
        &self,
        request: Request<v1::SetAlertZonesRequest>,
    ) -> Reply<v1::AlertZonesConfig> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::AdminManage])?;
        let revision = revision(request.expected_revision)?;
        let intercom = self.0.intercom()?;
        let value = self
            .0
            .household
            .call(move |db| {
                let zones = request
                    .zone_ids
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                db.set_alert_zones(revision, Some(&zones), &intercom)?
                    .ok_or(nexa_household::Error::Conflict(
                        "alert zone revision changed",
                    ))
            })
            .await
            .map_err(household_status)?;
        ok(self.alert_zones(value))
    }

    async fn list_conversations(
        &self,
        request: Request<v1::ListConversationsRequest>,
    ) -> Reply<v1::ConversationsSnapshot> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::ConversationsRead])?;
        ok(v1::ConversationsSnapshot {
            payload_json: self
                .0
                .events
                .conversations(&request.room_id, limit(request.limit))
                .to_string(),
        })
    }

    async fn list_ha_actions(
        &self,
        request: Request<v1::ListHaActionsRequest>,
    ) -> Reply<v1::HaActionsSnapshot> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::DiagnosticsRead])?;
        ok(v1::HaActionsSnapshot {
            receipts: self.0.home.history(limit(request.limit)),
        })
    }

    async fn list_memory_overview(
        &self,
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
        ok(self.memory()?.overview(request).await?)
    }

    async fn mutate_memory_item(
        &self,
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
            let brain = Arc::clone(&self.0);
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
        ok(self.memory()?.mutate(request).await?)
    }

    async fn get_voice_global_switch(
        &self,
        request: Request<v1::ActorRequest>,
    ) -> Reply<v1::VoiceGlobalSwitchStatus> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::MemoryRead, P::MemoryAssign])?;
        ok(self.memory()?.voice_switch(request).await?)
    }

    async fn set_voice_global_switch(
        &self,
        request: Request<v1::SetVoiceGlobalSwitchRequest>,
    ) -> Reply<v1::VoiceGlobalSwitchStatus> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::MemoryAssign, P::AdminManage])?;
        ok(self.memory()?.set_voice_switch(request).await?)
    }

    async fn get_voice_capture_review(
        &self,
        request: Request<v1::VoiceCaptureReviewRequest>,
    ) -> Reply<v1::VoiceCaptureReviewResponse> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::VoicesRead, P::VoicesReview])?;
        ok(self.memory()?.review_capture(request).await?)
    }

    async fn query_voice_captures(
        &self,
        request: Request<v1::VoiceCaptureQueryRequest>,
    ) -> Reply<v1::VoiceCaptureQueryResponse> {
        let request = request.into_inner();
        authorize(
            request.actor.as_ref(),
            &[P::VoicesRead, P::VoicesReview, P::AdminManage],
        )?;
        ok(self.memory()?.query_captures(request).await?)
    }

    async fn get_evaluation_sample(
        &self,
        request: Request<v1::EvaluationSampleRequest>,
    ) -> Reply<v1::EvaluationSampleResponse> {
        let request = request.into_inner();
        authorize(request.actor.as_ref(), &[P::EvaluationRead])?;
        ok(self.memory()?.evaluation_sample(request).await?)
    }

    async fn sync_household_lists(
        &self,
        request: Request<v1::ActorRequest>,
    ) -> Reply<v1::ListSyncResponse> {
        authorize(request.into_inner().actor.as_ref(), &[P::TasksWrite])?;
        let lists = self
            .0
            .household
            .call(|db| db.lists(None))
            .await
            .map_err(household_status)?;
        let configured = lists.iter().any(|list| list.ha_entity.is_some());
        ok(v1::ListSyncResponse {
            payload_json: json!({
                "lists": lists.len(),
                "synced": 0,
                "ha_todo_configured": configured,
                "status": if configured { "adapter_unavailable" } else { "not_configured" },
            })
            .to_string(),
        })
    }

    async fn list_reminder_notices(
        &self,
        request: Request<v1::ActorRequest>,
    ) -> Reply<v1::ReminderNoticesResponse> {
        authorize(request.into_inner().actor.as_ref(), &[P::TasksRead])?;
        // Household only creates delivery notices for shared reminders.
        let notices = self
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

    async fn get_reminder_status(
        &self,
        request: Request<v1::ActorRequest>,
    ) -> Reply<v1::ReminderStatusResponse> {
        authorize(request.into_inner().actor.as_ref(), &[P::DiagnosticsRead])?;
        let status = self
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
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RoutinePayload {
    name: String,
    enabled: bool,
    steps: Vec<RoutineStep>,
    schedule: Option<RoutineSchedule>,
}

fn routine_json(spec: &RoutineSpec, schedule: Option<&RoutineSchedule>) -> String {
    let mut value = json!({"name": spec.name, "enabled": spec.enabled, "steps": spec.steps});
    if let Some(schedule) = schedule {
        value["schedule"] = json!(schedule);
    }
    value.to_string()
}

fn routine_record(routine: Routine) -> Result<v1::RoutineRecord, Status> {
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
struct ListPayload {
    personal: bool,
    name: String,
    text: String,
    item_id: String,
    completed: Option<bool>,
}

fn list_record(snapshot: ListSnapshot) -> v1::HouseholdListRecord {
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
struct ReminderPayload {
    personal: Option<bool>,
    scope: Option<ReminderScope>,
    title: Option<String>,
    text: Option<String>,
    #[serde(alias = "room_id", alias = "delivery_room")]
    room: Option<String>,
    due_at_unix_ms: Option<i64>,
    trigger_at: Option<serde_json::Number>,
    #[serde(alias = "ha_actions")]
    actions: Option<Vec<ReminderHomeAction>>,
    recurrence: Option<String>,
    expected_revision: Option<u64>,
    reminder_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReminderScope {
    Private,
    Shared,
}

impl ReminderPayload {
    fn is_personal(&self) -> Result<Option<bool>, Status> {
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

    fn due_at(&self) -> Result<Option<i64>, Status> {
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
struct ReminderHomeAction {
    domain: Option<String>,
    service: String,
    entity_id: String,
    data: Option<serde_json::Map<String, Value>>,
}

impl ReminderHomeAction {
    fn into_action(self) -> Result<HomeAction, Status> {
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

fn reminder_record(
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

fn alert_record(value: Alert) -> v1::AlertRecord {
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

/// A bound socket whose inode is removed when this owner exits.
#[derive(Debug)]
pub struct BoundControl {
    listener: UnixListener,
    path: PathBuf,
    inode: u64,
    device: u64,
}

impl BoundControl {
    pub fn bind(path: &Path) -> io::Result<Self> {
        let dir = path
            .parent()
            .ok_or_else(|| io::Error::other("brain socket requires a parent directory"))?;
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700).create(dir)?;
        let uid = rustix::process::geteuid().as_raw();
        let metadata = fs::symlink_metadata(dir)?;
        if !metadata.is_dir() || metadata.uid() != uid {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "runtime directory is not owned by daemon UID",
            ));
        }
        if metadata.mode() & 0o7777 != 0o700 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "runtime directory mode must be 0700",
            ));
        }
        let listener = UnixListener::bind(path)?;
        let metadata = match fs::symlink_metadata(path) {
            Ok(metadata) => metadata,
            Err(error) => {
                let _ = fs::remove_file(path);
                return Err(error);
            }
        };
        let bound = Self {
            listener,
            path: path.to_owned(),
            inode: metadata.ino(),
            device: metadata.dev(),
        };
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        Ok(bound)
    }

    pub async fn serve(self, brain: Arc<Brain>) -> Result<(), BrainError> {
        let listener = &self.listener;
        let incoming = async_stream::stream! {
            let uid = rustix::process::geteuid().as_raw();
            loop {
                match listener.accept().await {
                    Ok((stream, _)) if stream.peer_cred().is_ok_and(|peer| peer.uid() == uid) => yield Ok(stream),
                    Ok(_) => tracing::warn!("rejected brain socket peer"),
                    Err(error) => { yield Err(error); break; }
                }
            }
        };
        let stop = brain.shutdown_token();
        Server::builder()
            .add_service(BrainControlServiceServer::new(Control(brain)))
            .serve_with_incoming_shutdown(incoming, stop.cancelled_owned())
            .await?;
        Ok(())
    }
}

impl Drop for BoundControl {
    fn drop(&mut self) {
        if fs::symlink_metadata(&self.path).is_ok_and(|metadata| {
            metadata.file_type().is_socket()
                && metadata.ino() == self.inode
                && metadata.dev() == self.device
        }) {
            let _ = fs::remove_file(&self.path);
        }
    }
}
