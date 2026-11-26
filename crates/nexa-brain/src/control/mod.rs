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

mod alerts;
mod auth;
mod dto;
mod home;
mod household;
mod memory;
mod reminders;
mod routines;
mod socket;
use auth::*;
use dto::*;
pub use socket::BoundControl;
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

// Tonic requires a single service impl. Keep the wire signatures here and
// dispatch to the permission-checked handlers in their domain modules.
macro_rules! control_service {
    ($( $domain:ident :: $name:ident ($input:ty) -> $output:ty; )*) => {
        #[tonic::async_trait]
        impl BrainControlService for Control {
            type SubscribeLiveEventsStream = StreamOf<Result<v1::LiveRuntimeEvent, Status>>;
            $(async fn $name(&self, request: Request<$input>) -> Reply<$output> {
                $domain::$name(self, request).await
            })*
        }
    };
}
control_service! {
    home::get_health(v1::Empty) -> v1::HealthResponse;
    home::get_home_catalog(v1::GetHomeCatalogRequest) -> v1::HomeCatalogSnapshot;
    home::execute_home_action(v1::HomeActionRequest) -> v1::HomeActionReceipt;
    home::get_home_action_receipt(v1::GetHomeActionReceiptRequest) -> v1::HomeActionReceipt;
    home::get_llm_endpoints(v1::Empty) -> v1::LlmEndpointsSnapshot;
    home::clear_llm_cooldown(v1::ClearLlmCooldownRequest) -> v1::Empty;
    home::get_cloud_consent(v1::Empty) -> v1::CloudConsentStatus;
    home::withdraw_cloud_consent(v1::WithdrawCloudConsentRequest) -> v1::CloudConsentStatus;
    home::subscribe_live_events(v1::SubscribeLiveEventsRequest) -> Self::SubscribeLiveEventsStream;
    routines::list_routines(v1::ListRoutinesRequest) -> v1::RoutinesSnapshot;
    routines::get_routine(v1::GetRoutineRequest) -> v1::RoutineRecord;
    routines::preview_routine(v1::PreviewRoutineRequest) -> v1::RoutinePreviewResponse;
    routines::upsert_routine(v1::UpsertRoutineRequest) -> v1::RoutineRecord;
    routines::delete_routine(v1::DeleteRoutineRequest) -> v1::Empty;
    routines::execute_routine(v1::ExecuteRoutineRequest) -> v1::RoutineExecutionReceipt;
    routines::get_routine_run(v1::GetRoutineRunRequest) -> v1::RoutineRunRecord;
    routines::list_routine_runs(v1::ListRoutineRunsRequest) -> v1::RoutineRunsSnapshot;
    household::list_household_lists(v1::ActorRequest) -> v1::HouseholdListsSnapshot;
    household::mutate_household_list(v1::HouseholdListMutationRequest) -> v1::HouseholdListRecord;
    household::sync_household_lists(v1::ActorRequest) -> v1::ListSyncResponse;
    reminders::list_reminders(v1::ListRemindersRequest) -> v1::RemindersSnapshot;
    reminders::mutate_reminder(v1::ReminderMutationRequest) -> v1::ReminderRecord;
    reminders::list_reminder_notices(v1::ActorRequest) -> v1::ReminderNoticesResponse;
    reminders::get_reminder_status(v1::ActorRequest) -> v1::ReminderStatusResponse;
    alerts::list_alerts(v1::ActorRequest) -> v1::AlertsSnapshot;
    alerts::ack_alert(v1::AlertMutationRequest) -> v1::AlertRecord;
    alerts::delete_alert(v1::AlertMutationRequest) -> v1::Empty;
    alerts::get_alert_zones(v1::ActorRequest) -> v1::AlertZonesConfig;
    alerts::set_alert_zones(v1::SetAlertZonesRequest) -> v1::AlertZonesConfig;
    memory::list_conversations(v1::ListConversationsRequest) -> v1::ConversationsSnapshot;
    memory::list_ha_actions(v1::ListHaActionsRequest) -> v1::HaActionsSnapshot;
    memory::list_memory_overview(v1::MemoryOverviewRequest) -> v1::MemoryOverviewResponse;
    memory::mutate_memory_item(v1::MemoryMutationRequest) -> v1::MemoryMutationResponse;
    memory::get_voice_global_switch(v1::ActorRequest) -> v1::VoiceGlobalSwitchStatus;
    memory::set_voice_global_switch(v1::SetVoiceGlobalSwitchRequest) -> v1::VoiceGlobalSwitchStatus;
    memory::get_voice_capture_review(v1::VoiceCaptureReviewRequest) -> v1::VoiceCaptureReviewResponse;
    memory::query_voice_captures(v1::VoiceCaptureQueryRequest) -> v1::VoiceCaptureQueryResponse;
    memory::get_evaluation_sample(v1::EvaluationSampleRequest) -> v1::EvaluationSampleResponse;
}
