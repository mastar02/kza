//! Deterministic, local-only fixtures for integration tests and browser automation.

use std::{
    io,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use axum::Router;
use nexa_proto::v1::*;
use serde_json::json;
use tempfile::TempDir;
use tokio::sync::broadcast;
use tonic::{Status, async_trait};

use crate::{
    WebConfig,
    auth::{AuthError, AuthStore, Origin},
    build_app,
    clients::{Audio, Brain, Clients, LiveEvents, Memory},
    sys::{Gpu, MonitorError, Service, SystemMonitor},
};

const OBSERVED: u64 = 1_700_000_000_000;
pub const OPERATOR_TOKEN: &str = "test-operator";
pub const HOME_TOKEN: &str = "test-home";
pub const EVALUATOR_TOKEN: &str = "test-evaluator";
pub const HOME_ID: &str = "test-home-user";

#[derive(Debug, thiserror::Error)]
pub enum TestModeError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Auth(#[from] AuthError),
}

#[derive(Debug)]
pub struct TestMode {
    root: TempDir,
    pub auth: AuthStore,
    pub clients: Clients,
    fixtures: Arc<Fixtures>,
}

impl TestMode {
    pub async fn new() -> Result<Self, TestModeError> {
        let root = tempfile::tempdir()?;
        let private = root.path().join("private");
        tokio::fs::create_dir(&private).await?;
        tokio::fs::write(private.join("casa.glb"), synthetic_model()).await?;
        tokio::fs::write(private.join("house-plan.js"),
            b"export const HOUSE_PLAN = {rooms:[{id:'test-room',name:'Test Room',polygon:[[0,0],[2,0],[2,2],[0,2]]}]};\nexport default HOUSE_PLAN;\n",
        ).await?;
        let auth = AuthStore::open(&root.path().join("auth.db"))?;
        auth.seed(
            "test-operator-credential",
            "operator",
            Role::Operator,
            None,
            OPERATOR_TOKEN,
        )
        .await?;
        auth.seed(
            "test-home-credential",
            "home",
            Role::Home,
            Some(HOME_ID),
            HOME_TOKEN,
        )
        .await?;
        auth.seed(
            "test-evaluator-credential",
            "evaluator",
            Role::Evaluator,
            None,
            EVALUATOR_TOKEN,
        )
        .await?;
        let fixtures = Arc::new(Fixtures::default());
        let clients = Clients {
            audio: fixtures.clone(),
            brain: fixtures.clone(),
            memory: fixtures.clone(),
        };
        Ok(Self {
            root,
            auth,
            clients,
            fixtures,
        })
    }

    pub fn app(
        &self,
        origin: Origin,
        allow_insecure_cookies: bool,
        static_dir: Option<PathBuf>,
    ) -> io::Result<Router> {
        build_app(
            WebConfig {
                origin,
                allow_insecure_cookies,
                static_dir,
                private_dir: Some(self.private_dir()),
            },
            self.auth.clone(),
            self.clients.clone(),
            Arc::new(FixturesMonitor),
        )
    }

    pub fn private_dir(&self) -> PathBuf {
        self.root.path().join("private")
    }

    pub fn database_path(&self) -> PathBuf {
        self.root.path().join("auth.db")
    }

    pub fn emit(&self, event: LiveRuntimeEvent) {
        let _ = self.fixtures.events.send(event);
    }
}

fn synthetic_model() -> Vec<u8> {
    // A glTF 2.0 container with a scene and no external resources.
    let mut scene = br#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[]}]}"#.to_vec();
    while !scene.len().is_multiple_of(4) {
        scene.push(b' ');
    }
    let size = u32::try_from(scene.len()).unwrap_or_default();
    let mut model = b"glTF".to_vec();
    model.extend(2_u32.to_le_bytes());
    model.extend((20 + size).to_le_bytes());
    model.extend(size.to_le_bytes());
    model.extend(b"JSON");
    model.extend(scene);
    model
}

#[derive(Debug)]
struct Fixtures {
    events: broadcast::Sender<LiveRuntimeEvent>,
    capture: Mutex<CaptureControlStatus>,
    ambient: Mutex<AmbientControlStatus>,
    consent: Mutex<CloudConsentStatus>,
}

impl Default for Fixtures {
    fn default() -> Self {
        Self {
            events: broadcast::channel(64).0,
            capture: Mutex::new(CaptureControlStatus {
                phase: CapturePhase::Running as i32,
                supported: true,
                ..Default::default()
            }),
            ambient: Mutex::new(AmbientControlStatus::default()),
            consent: Mutex::new(CloudConsentStatus {
                consent_granted: true,
                ..Default::default()
            }),
        }
    }
}

struct FixtureLockError;

impl From<FixtureLockError> for Status {
    fn from(_: FixtureLockError) -> Self {
        Status::unavailable("fixture unavailable")
    }
}

fn locked<T>(value: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, FixtureLockError> {
    value.lock().map_err(|_| FixtureLockError)
}

#[async_trait]
impl Audio for Fixtures {
    async fn get_microphones(&self, _: Empty) -> Result<MicrophonesSnapshot, Status> {
        Ok(MicrophonesSnapshot {
            observed_at_unix_ms: OBSERVED,
            stale: false,
            microphones: vec![MicrophoneRoomStatus {
                room_id: "test-room".into(),
                room_name: "Test Room".into(),
                state: MicCaptureState::Receiving as i32,
                last_frame_unix_ms: OBSERVED,
                ..Default::default()
            }],
        })
    }

    async fn get_zones(&self, _: Empty) -> Result<ZonesSnapshot, Status> {
        Ok(ZonesSnapshot {
            zones: vec![ZoneState {
                zone_id: "test-room".into(),
                room_id: "test-room".into(),
                ma1260_zone: 1,
                volume_pct: 20,
                ..Default::default()
            }],
        })
    }

    async fn get_capture_control(&self, _: Empty) -> Result<CaptureControlStatus, Status> {
        Ok(*locked(&self.capture)?)
    }

    async fn set_capture_mute(
        &self,
        request: SetCaptureMuteRequest,
    ) -> Result<CaptureControlStatus, Status> {
        let mut state = locked(&self.capture)?;
        if state.revision != request.expected_revision {
            return Err(Status::aborted("capture revision"));
        }
        state.revision += 1;
        state.phase = if request.muted {
            CapturePhase::Muted
        } else {
            CapturePhase::Running
        } as i32;
        state.persisted_muted = request.muted;
        state.ingress_sealed = request.muted;
        state.total_mute_confirmed = request.muted;
        Ok(*state)
    }
}

#[async_trait]
impl Brain for Fixtures {
    async fn get_home_catalog(
        &self,
        _: GetHomeCatalogRequest,
    ) -> Result<HomeCatalogSnapshot, Status> {
        Ok(HomeCatalogSnapshot {
            observed_at_unix_ms: OBSERVED,
            stale: false,
            rooms: vec![HomeRoom {
                room_id: "test-room".into(),
                name: "Test Room".into(),
                entity_ids: vec!["light.fixture".into()],
                default_light: "light.fixture".into(),
                ..Default::default()
            }],
            entities: vec![HomeEntity {
                entity_id: "light.fixture".into(),
                domain: "light".into(),
                room_id: "test-room".into(),
                friendly_name: "Fixture light".into(),
                state: "off".into(),
                available: true,
                capabilities: vec!["turn_on".into(), "turn_off".into()],
                observed_at_unix_ms: OBSERVED,
            }],
        })
    }

    async fn subscribe_live_events(
        &self,
        _: SubscribeLiveEventsRequest,
    ) -> Result<LiveEvents, Status> {
        let receiver = self.events.subscribe();
        Ok(Box::pin(futures_util::stream::unfold(
            receiver,
            |mut receiver| async {
                match receiver.recv().await {
                    Ok(event) => Some((Ok(event), receiver)),
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        Some((Err(Status::aborted("live event gap")), receiver))
                    }
                    Err(broadcast::error::RecvError::Closed) => None,
                }
            },
        )))
    }

    async fn list_routines(&self, _: ListRoutinesRequest) -> Result<RoutinesSnapshot, Status> {
        Ok(RoutinesSnapshot::default())
    }

    async fn list_household_lists(
        &self,
        _: ActorRequest,
    ) -> Result<HouseholdListsSnapshot, Status> {
        Ok(HouseholdListsSnapshot {
            lists: vec![HouseholdListRecord {
                list_id: "shopping".into(),
                name: "Compras".into(),
                revision: 1,
                items_json: "[]".into(),
                ..Default::default()
            }],
        })
    }

    async fn list_reminders(&self, _: ListRemindersRequest) -> Result<RemindersSnapshot, Status> {
        Ok(RemindersSnapshot::default())
    }

    async fn list_alerts(&self, _: ActorRequest) -> Result<AlertsSnapshot, Status> {
        Ok(AlertsSnapshot::default())
    }

    async fn get_alert_zones(&self, _: ActorRequest) -> Result<AlertZonesConfig, Status> {
        Ok(AlertZonesConfig {
            revision: 1,
            zone_ids: vec!["test-room".into()],
        })
    }

    async fn list_conversations(
        &self,
        _: ListConversationsRequest,
    ) -> Result<ConversationsSnapshot, Status> {
        Ok(ConversationsSnapshot {
            payload_json: "[]".into(),
        })
    }

    async fn list_ha_actions(&self, _: ListHaActionsRequest) -> Result<HaActionsSnapshot, Status> {
        Ok(HaActionsSnapshot::default())
    }

    async fn get_llm_endpoints(&self, _: Empty) -> Result<LlmEndpointsSnapshot, Status> {
        Ok(LlmEndpointsSnapshot::default())
    }

    async fn get_cloud_consent(&self, _: Empty) -> Result<CloudConsentStatus, Status> {
        Ok(*locked(&self.consent)?)
    }

    async fn withdraw_cloud_consent(
        &self,
        _: WithdrawCloudConsentRequest,
    ) -> Result<CloudConsentStatus, Status> {
        let mut consent = locked(&self.consent)?;
        consent.consent_granted = false;
        consent.withdrawn_latched = true;
        consent.updated_at_unix_ms = OBSERVED;
        Ok(*consent)
    }

    async fn list_memory_overview(
        &self,
        request: MemoryOverviewRequest,
    ) -> Result<MemoryOverviewResponse, Status> {
        let page = match request.view.as_str() {
            "proposals" => json!({"proposals":[]}),
            "owners_counts" => json!({"counts":{},"retired_owner_prune_pending":false}),
            "summary" => json!({"facts":[],"proposals":[]}),
            _ => json!({"facts":[],"next":null,"limit":request.limit}),
        };
        Ok(MemoryOverviewResponse {
            payload_json: page.to_string(),
        })
    }

    async fn query_voice_captures(
        &self,
        _: VoiceCaptureQueryRequest,
    ) -> Result<VoiceCaptureQueryResponse, Status> {
        Ok(VoiceCaptureQueryResponse {
            payload_json: "[]".into(),
        })
    }

    async fn get_voice_global_switch(
        &self,
        _: ActorRequest,
    ) -> Result<VoiceGlobalSwitchStatus, Status> {
        Ok(VoiceGlobalSwitchStatus::default())
    }
}

#[async_trait]
impl Memory for Fixtures {
    async fn list_voice_profiles(
        &self,
        _: ListVoiceProfilesRequest,
    ) -> Result<ListVoiceProfilesResponse, Status> {
        Ok(ListVoiceProfilesResponse::default())
    }

    async fn purge_retired_user(
        &self,
        request: PurgeRetiredUserRequest,
    ) -> Result<ErasureReceipt, Status> {
        Ok(ErasureReceipt {
            erasure_id: request.user_id,
            tombstone_recorded: true,
            completed_at_unix_ms: OBSERVED,
            ..Default::default()
        })
    }

    async fn get_ambient_control(&self, _: Empty) -> Result<AmbientControlStatus, Status> {
        Ok(*locked(&self.ambient)?)
    }

    async fn set_ambient_pause(
        &self,
        request: SetAmbientPauseRequest,
    ) -> Result<AmbientControlStatus, Status> {
        let mut state = locked(&self.ambient)?;
        if state.revision != request.expected_revision {
            return Err(Status::aborted("ambient revision"));
        }
        state.revision += 1;
        state.paused = request.paused;
        state.updated_at_unix_ms = OBSERVED;
        Ok(*state)
    }
}

#[derive(Debug)]
struct FixturesMonitor;

#[async_trait]
impl SystemMonitor for FixturesMonitor {
    async fn gpus(&self) -> Vec<Gpu> {
        vec![]
    }
    async fn services(&self) -> Vec<Service> {
        vec![Service {
            name: "nexa-brain".into(),
            status: "active".into(),
            substate: "running".into(),
        }]
    }
    async fn restart(&self, _: &str) -> Result<(), MonitorError> {
        Err(MonitorError::Demo)
    }
}
