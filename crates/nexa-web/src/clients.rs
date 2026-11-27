//! Injected daemon services and their production Tonic transports.

use std::{os::unix::fs::MetadataExt, path::Path, pin::Pin, sync::Arc, time::Duration};

use futures_util::Stream;
use hyper_util::rt::TokioIo;
use nexa_proto::{sockets, v1::*};
use prost::Message;
use tokio::net::UnixStream;
use tonic::{
    Request, Status,
    metadata::MetadataValue,
    transport::{Channel, Endpoint},
};
use tower::service_fn;

pub type LiveEvents = Pin<Box<dyn Stream<Item = Result<LiveRuntimeEvent, Status>> + Send>>;

macro_rules! daemon {
    ($service:ident, $transport:ident, $client:path;
     $( $method:ident($request:ty $(=> $actor_metadata:literal)?) -> $response:ty $(=> $adapt:expr)?; )+) => {
        #[tonic::async_trait]
        pub trait $service: Send + Sync {
            $(
                async fn $method(&self, _request: $request) -> Result<$response, Status> {
                    Err(Status::unimplemented(stringify!($method)))
                }
            )+
        }

        #[derive(Debug)]
        struct $transport(Channel);

        #[tonic::async_trait]
        impl $service for $transport {
            $(
                async fn $method(&self, request: $request) -> Result<$response, Status> {
                    $(let request = actor_request(request, $actor_metadata);)?
                    let response = <$client>::new(self.0.clone()).$method(request).await?.into_inner();
                    $(let response = ($adapt)(response);)?
                    Ok(response)
                }
            )+
        }
    };
}

fn actor_request(actor: ActorContext, metadata_key: &'static str) -> Request<Empty> {
    let mut request = Request::new(Empty {});
    request.metadata_mut().insert_bin(
        metadata_key,
        MetadataValue::from_bytes(&actor.encode_to_vec()),
    );
    request
}

daemon! {
    Audio, TonicAudio, audio_service_client::AudioServiceClient<Channel>;
    get_health(Empty) -> HealthResponse;
    get_microphones(Empty) -> MicrophonesSnapshot;
    get_zones(Empty) -> ZonesSnapshot;
    set_zone_volume(SetZoneVolumeRequest) -> ZoneState;
    get_capture_control(Empty) -> CaptureControlStatus;
    set_capture_mute(SetCaptureMuteRequest) -> CaptureControlStatus;
    forget_carried_owners(ForgetCarriedOwnersRequest) -> ForgetCarriedOwnersResponse;
}

daemon! {
    Brain, TonicBrain, brain_control_service_client::BrainControlServiceClient<Channel>;
    get_health(Empty) -> HealthResponse;
    get_home_catalog(GetHomeCatalogRequest) -> HomeCatalogSnapshot;
    execute_home_action(HomeActionRequest) -> HomeActionReceipt;
    get_home_action_receipt(GetHomeActionReceiptRequest) -> HomeActionReceipt;
    get_llm_endpoints(ActorContext => "x-nexa-actor-bin") -> LlmEndpointsSnapshot;
    clear_llm_cooldown(ClearLlmCooldownRequest) -> Empty;
    get_cloud_consent(ActorContext => "x-nexa-actor-bin") -> CloudConsentStatus;
    withdraw_cloud_consent(WithdrawCloudConsentRequest) -> CloudConsentStatus;
    subscribe_live_events(SubscribeLiveEventsRequest) -> LiveEvents
        => |response| -> LiveEvents { Box::pin(response) };
    list_routine_runs(ListRoutineRunsRequest) -> RoutineRunsSnapshot;
    get_routine_run(GetRoutineRunRequest) -> RoutineRunRecord;
    list_routines(ListRoutinesRequest) -> RoutinesSnapshot;
    get_routine(GetRoutineRequest) -> RoutineRecord;
    preview_routine(PreviewRoutineRequest) -> RoutinePreviewResponse;
    upsert_routine(UpsertRoutineRequest) -> RoutineRecord;
    delete_routine(DeleteRoutineRequest) -> Empty;
    execute_routine(ExecuteRoutineRequest) -> RoutineExecutionReceipt;
    list_household_lists(ActorRequest) -> HouseholdListsSnapshot;
    mutate_household_list(HouseholdListMutationRequest) -> HouseholdListRecord;
    sync_household_lists(ActorRequest) -> ListSyncResponse;
    list_reminders(ListRemindersRequest) -> RemindersSnapshot;
    mutate_reminder(ReminderMutationRequest) -> ReminderRecord;
    list_reminder_notices(ActorRequest) -> ReminderNoticesResponse;
    get_reminder_status(ActorRequest) -> ReminderStatusResponse;
    list_alerts(ActorRequest) -> AlertsSnapshot;
    ack_alert(AlertMutationRequest) -> AlertRecord;
    delete_alert(AlertMutationRequest) -> Empty;
    get_alert_zones(ActorRequest) -> AlertZonesConfig;
    set_alert_zones(SetAlertZonesRequest) -> AlertZonesConfig;
    list_conversations(ListConversationsRequest) -> ConversationsSnapshot;
    list_ha_actions(ListHaActionsRequest) -> HaActionsSnapshot;
    list_memory_overview(MemoryOverviewRequest) -> MemoryOverviewResponse;
    mutate_memory_item(MemoryMutationRequest) -> MemoryMutationResponse;
    get_voice_global_switch(ActorRequest) -> VoiceGlobalSwitchStatus;
    set_voice_global_switch(SetVoiceGlobalSwitchRequest) -> VoiceGlobalSwitchStatus;
    get_voice_capture_review(VoiceCaptureReviewRequest) -> VoiceCaptureReviewResponse;
    query_voice_captures(VoiceCaptureQueryRequest) -> VoiceCaptureQueryResponse;
    get_evaluation_sample(EvaluationSampleRequest) -> EvaluationSampleResponse;
}

daemon! {
    Memory, TonicMemory, memory_service_client::MemoryServiceClient<Channel>;
    get_health(ActorContext => "actor-context-bin") -> HealthResponse;
    delete_fact(DeleteFactRequest) -> ErasureReceipt;
    list_memory_overview(MemoryOverviewRequest) -> MemoryOverviewResponse;
    mutate_memory_item(MemoryMutationRequest) -> MemoryMutationResponse;
    list_voice_profiles(ListVoiceProfilesRequest) -> ListVoiceProfilesResponse;
    link_voice_profile_to_person(LinkVoiceProfileRequest) -> VoiceProfile;
    set_voice_consent(SetVoiceConsentRequest) -> VoiceConsentStatus;
    purge_retired_user(PurgeRetiredUserRequest) -> ErasureReceipt;
    get_ambient_control(ActorContext => "actor-context-bin") -> AmbientControlStatus;
    set_ambient_pause(SetAmbientPauseRequest) -> AmbientControlStatus;
}

daemon! {
    Ml, TonicMl, ml_service_client::MlServiceClient<Channel>;
    get_health(Empty) -> MlHealthResponse;
    reload_model(ReloadModelRequest) -> HealthResponse;
}

#[derive(Clone)]
pub struct Clients {
    pub audio: Arc<dyn Audio>,
    pub brain: Arc<dyn Brain>,
    pub memory: Arc<dyn Memory>,
    pub ml: Arc<dyn Ml>,
}

impl std::fmt::Debug for Clients {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Clients").finish_non_exhaustive()
    }
}

impl Clients {
    /// Channels reconnect lazily so Axum can start before the upstream daemons.
    pub fn uds(directory: &Path) -> Self {
        Self {
            audio: Arc::new(TonicAudio(channel(directory, sockets::AUDIO))),
            brain: Arc::new(TonicBrain(channel(directory, sockets::BRAIN))),
            memory: Arc::new(TonicMemory(channel(directory, sockets::MEMORY))),
            ml: Arc::new(TonicMl(channel(directory, sockets::ML))),
        }
    }
}

fn channel(directory: &Path, name: &str) -> Channel {
    let path = directory.join(name);
    Endpoint::from_static("http://[::]:50051")
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(5))
        .connect_with_connector_lazy(service_fn(move |_| {
            let path = path.clone();
            async move {
                use std::os::unix::fs::FileTypeExt;
                let metadata = tokio::fs::symlink_metadata(&path).await?;
                if !metadata.file_type().is_socket()
                    || metadata.mode() & 0o777 != 0o600
                    || metadata.uid() != rustix::process::geteuid().as_raw()
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "unsafe upstream socket",
                    ));
                }
                let stream = UnixStream::connect(path).await?;
                if stream.peer_cred()?.uid() != rustix::process::geteuid().as_raw() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "unexpected upstream peer",
                    ));
                }
                Ok(TokioIo::new(stream))
            }
        }))
}
