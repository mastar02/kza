//! Deterministic, local-only fixtures for integration tests and browser automation.

use std::{
    collections::HashMap,
    fs, io,
    os::unix::fs::DirBuilderExt,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::Router;
use nexa_proto::v1::*;
use serde_json::{Value, json};
use tokio::sync::broadcast;
use tonic::{Status, async_trait};

use crate::{
    WebConfig,
    auth::{AuthError, AuthStore, Origin},
    build_app,
    clients::{Audio, Brain, Clients, LiveEvents, Memory},
    sys::{Gpu, MonitorError, Service, SystemMonitor},
};

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| u64::try_from(d.as_millis()).ok())
        .unwrap_or(1_700_000_000_000)
}
pub const OPERATOR_TOKEN: &str = "test-operator";
pub const HOME_TOKEN: &str = "test-home";
pub const EVALUATOR_TOKEN: &str = "test-evaluator";
pub const HOME_ID: &str = "test-home-user";
const REMINDER_ID: &str = "test-reminder";
const FIRST_PROFILE: &str = "00000000-0000-4000-8000-000000000001";
const SECOND_PROFILE: &str = "00000000-0000-4000-8000-000000000002";

#[derive(Debug, thiserror::Error)]
pub enum TestModeError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Auth(#[from] AuthError),
}

#[derive(Debug)]
pub struct TestMode {
    root: FixtureDir,
    pub auth: AuthStore,
    pub clients: Clients,
    fixtures: Arc<Fixtures>,
}

#[derive(Debug)]
struct FixtureDir(PathBuf);

impl FixtureDir {
    fn new() -> io::Result<Self> {
        let path = std::env::temp_dir().join(format!("nexa-web-{}", uuid::Uuid::new_v4()));
        fs::DirBuilder::new().mode(0o700).create(&path)?;
        Ok(Self(path))
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

impl TestMode {
    pub async fn new() -> Result<Self, TestModeError> {
        let root = FixtureDir::new()?;
        let private = root.0.join("private");
        tokio::fs::create_dir(&private).await?;
        tokio::fs::write(private.join("casa.glb"), synthetic_model()).await?;
        tokio::fs::write(private.join("house-plan.js"),
            b"export const HOUSE_PLAN = {rooms:[{id:'test-room',name:'Test Room',polygon:[[0,0],[2,0],[2,2],[0,2]]}]};\nexport default HOUSE_PLAN;\n",
        ).await?;
        let auth = AuthStore::open(&root.0.join("auth.db"))?;
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
        self.root.0.join("private")
    }

    pub fn database_path(&self) -> PathBuf {
        self.root.0.join("auth.db")
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
    voice_switch: Mutex<VoiceGlobalSwitchStatus>,
    voice_consent: Mutex<(bool, u64)>,
    reminder: Mutex<ReminderRecord>,
    voice_profiles: Mutex<Vec<VoiceProfile>>,
    light_state: Mutex<String>,
    receipts: Mutex<HashMap<String, HomeActionReceipt>>,
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
            voice_switch: Mutex::default(),
            voice_consent: Mutex::default(),
            reminder: Mutex::new(ReminderRecord {
                reminder_id: REMINDER_ID.into(),
                title: "Recordatorio de prueba".into(),
                status: "pending".into(),
                due_at_unix_ms: 1_800_000_000_000,
                revision: 1,
                ..Default::default()
            }),
            voice_profiles: Mutex::new(
                [FIRST_PROFILE, SECOND_PROFILE]
                    .into_iter()
                    .map(|id| VoiceProfile {
                        profile_id: id.into(),
                        profile_revision: 1,
                        sample_count: 3,
                        ..Default::default()
                    })
                    .collect(),
            ),
            light_state: Mutex::new("off".into()),
            receipts: Mutex::new(HashMap::new()),
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
        let observed = now_unix_ms();
        Ok(MicrophonesSnapshot {
            observed_at_unix_ms: observed,
            stale: false,
            microphones: vec![MicrophoneRoomStatus {
                room_id: "test-room".into(),
                room_name: "Test Room".into(),
                state: MicCaptureState::Receiving as i32,
                last_frame_unix_ms: observed,
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
        let observed = now_unix_ms();
        let state = locked(&self.light_state)?.clone();
        Ok(HomeCatalogSnapshot {
            observed_at_unix_ms: observed,
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
                state,
                available: true,
                capabilities: vec!["turn_on".into(), "turn_off".into()],
                observed_at_unix_ms: observed,
            }],
        })
    }

    async fn execute_home_action(
        &self,
        request: HomeActionRequest,
    ) -> Result<HomeActionReceipt, Status> {
        if request.entity_id != "light.fixture"
            || !matches!(request.operation.as_str(), "turn_on" | "turn_off")
        {
            return Err(Status::permission_denied("entity or operation denied"));
        }
        let mut receipts = locked(&self.receipts)?;
        if let Some(existing) = receipts.get(&request.request_id) {
            if existing.entity_id != request.entity_id || existing.operation != request.operation {
                return Err(Status::already_exists("conflicting request_id"));
            }
            return Ok(existing.clone());
        }
        *locked(&self.light_state)? = if request.operation == "turn_on" {
            "on".into()
        } else {
            "off".into()
        };
        let receipt = HomeActionReceipt {
            request_id: request.request_id.clone(),
            entity_id: request.entity_id,
            operation: request.operation,
            status: "confirmed".into(),
            detail: String::new(),
            completed_at_unix_ms: now_unix_ms(),
        };
        receipts.insert(request.request_id, receipt.clone());
        Ok(receipt)
    }

    async fn get_home_action_receipt(
        &self,
        request: GetHomeActionReceiptRequest,
    ) -> Result<HomeActionReceipt, Status> {
        locked(&self.receipts)?
            .get(&request.request_id)
            .cloned()
            .ok_or_else(|| Status::not_found("home action receipt not found"))
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

    async fn sync_household_lists(&self, _: ActorRequest) -> Result<ListSyncResponse, Status> {
        Ok(ListSyncResponse {
            payload_json: json!({"synced": true}).to_string(),
        })
    }

    async fn list_reminder_notices(
        &self,
        _: ActorRequest,
    ) -> Result<ReminderNoticesResponse, Status> {
        Ok(ReminderNoticesResponse {
            payload_json: json!({"source": "reminder-store", "notices": []}).to_string(),
        })
    }

    async fn get_reminder_status(&self, _: ActorRequest) -> Result<ReminderStatusResponse, Status> {
        Ok(ReminderStatusResponse {
            payload_json: json!({"source": "reminder-store"}).to_string(),
        })
    }

    async fn list_reminders(
        &self,
        request: ListRemindersRequest,
    ) -> Result<RemindersSnapshot, Status> {
        let pending = locked(&self.reminder)?.clone();
        let mut reminders = vec![pending.clone()];
        if request.include_completed {
            reminders.push(ReminderRecord {
                reminder_id: "completed-reminder".into(),
                status: "completed".into(),
                revision: 2,
                ..pending
            });
        }
        Ok(RemindersSnapshot { reminders })
    }

    async fn mutate_reminder(
        &self,
        request: ReminderMutationRequest,
    ) -> Result<ReminderRecord, Status> {
        if request.operation != "update" {
            return Err(Status::unimplemented(
                "fixture only supports reminder updates",
            ));
        }
        let mut row = locked(&self.reminder)?;
        if request.reminder_id != row.reminder_id {
            return Err(Status::not_found("reminder"));
        }
        if request.expected_revision != row.revision {
            return Err(Status::aborted("reminder revision"));
        }
        let body: Value = serde_json::from_str(&request.payload_json)
            .map_err(|_| Status::invalid_argument("reminder payload"))?;
        if let Some(title) = body["text"].as_str() {
            row.title = title.into();
        }
        if body["scope"] == "private" {
            row.owner_user_id = request
                .actor
                .ok_or_else(|| Status::unauthenticated("actor"))?
                .user_id;
        } else {
            row.owner_user_id.clear();
        }
        row.revision += 1;
        Ok(row.clone())
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
        consent.updated_at_unix_ms = now_unix_ms();
        Ok(*consent)
    }

    async fn list_memory_overview(
        &self,
        request: MemoryOverviewRequest,
    ) -> Result<MemoryOverviewResponse, Status> {
        let page = match request.view.as_str() {
            "owners_counts" => json!({"counts":{},"retired_owner_prune_pending":false}),
            view => {
                let key = if view == "proposals" {
                    "proposals"
                } else {
                    "facts"
                };
                let first = format!("{view}-a");
                let second = format!("{view}-b");
                match request.cursor.as_str() {
                    "" => json!({(key): [{"id": first}], "next": first, "limit": request.limit}),
                    cursor if cursor == first => {
                        json!({(key): [{"id": second}], "next": null, "limit": request.limit})
                    }
                    _ => return Err(Status::invalid_argument("fixture page cursor")),
                }
            }
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
        Ok(*locked(&self.voice_switch)?)
    }

    async fn set_voice_global_switch(
        &self,
        request: SetVoiceGlobalSwitchRequest,
    ) -> Result<VoiceGlobalSwitchStatus, Status> {
        let mut row = locked(&self.voice_switch)?;
        if row.revision != request.expected_revision {
            return Err(Status::aborted("voice switch revision"));
        }
        row.enabled = request.enabled;
        row.revision += 1;
        row.updated_at_unix_ms = now_unix_ms();
        Ok(*row)
    }
}

#[async_trait]
impl Memory for Fixtures {
    async fn list_memory_overview(
        &self,
        request: MemoryOverviewRequest,
    ) -> Result<MemoryOverviewResponse, Status> {
        let (granted, revision) = *locked(&self.voice_consent)?;
        let personal = json!({"user_id":HOME_ID, "granted":granted, "revision":revision});
        let page = match request.view.as_str() {
            "voice_mine" => personal,
            "voice_consents" => json!({"consents":[personal],"next":null,"limit":request.limit}),
            _ => return Err(Status::invalid_argument("fixture voice view")),
        };
        Ok(MemoryOverviewResponse {
            payload_json: page.to_string(),
        })
    }

    async fn mutate_memory_item(
        &self,
        request: MemoryMutationRequest,
    ) -> Result<MemoryMutationResponse, Status> {
        let body: Value = serde_json::from_str(&request.payload_json)
            .map_err(|_| Status::invalid_argument("voice payload"))?;
        let payload = match request.operation.as_str() {
            "voice_consent_grant" | "voice_consent_revoke" | "voice_verification" => {
                let user = request.actor.as_ref().map(|actor| actor.user_id.as_str());
                if user != Some(HOME_ID) || !request.target_id.is_empty() {
                    return Err(Status::permission_denied("fixture voice owner"));
                }
                let mut row = locked(&self.voice_consent)?;
                if body["expected_revision"]
                    .as_u64()
                    .is_some_and(|expected| expected != row.1)
                {
                    return Err(Status::aborted("consent revision"));
                }
                if request.operation == "voice_verification" {
                    if !matches!(body["outcome"].as_str(), Some("passed" | "failed")) {
                        return Err(Status::invalid_argument("verification outcome required"));
                    }
                    json!({"recorded":true, "revision":row.1})
                } else {
                    row.0 = request.operation == "voice_consent_grant";
                    row.1 += 1;
                    json!({"user_id":HOME_ID,"granted":row.0,"revision":row.1})
                }
            }
            "delete_voice_profile" => {
                let mut profiles = locked(&self.voice_profiles)?;
                let index = profiles
                    .iter()
                    .position(|row| row.profile_id == request.target_id)
                    .ok_or_else(|| Status::not_found("voice profile"))?;
                if body["expected_revision"]
                    .as_u64()
                    .is_some_and(|rev| rev != profiles[index].profile_revision)
                {
                    return Err(Status::aborted("voice profile revision"));
                }
                let deleted = profiles.remove(index);
                json!({"deleted":true,"captures_deleted":deleted.sample_count})
            }
            _ => return Err(Status::unimplemented("fixture voice mutation")),
        };
        Ok(MemoryMutationResponse {
            payload_json: payload.to_string(),
        })
    }

    async fn list_voice_profiles(
        &self,
        request: ListVoiceProfilesRequest,
    ) -> Result<ListVoiceProfilesResponse, Status> {
        let profiles = locked(&self.voice_profiles)?;
        let mut candidates = profiles
            .iter()
            .filter(|row| row.profile_id > request.cursor);
        let first = candidates.next();
        let next_cursor = match (first, candidates.next()) {
            (Some(row), Some(_)) => row.profile_id.clone(),
            _ => String::new(),
        };
        Ok(ListVoiceProfilesResponse {
            profiles: first.cloned().into_iter().collect(),
            next_cursor,
        })
    }

    async fn purge_retired_user(
        &self,
        request: PurgeRetiredUserRequest,
    ) -> Result<ErasureReceipt, Status> {
        Ok(ErasureReceipt {
            erasure_id: request.user_id,
            tombstone_recorded: true,
            completed_at_unix_ms: now_unix_ms(),
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
        state.updated_at_unix_ms = now_unix_ms();
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
