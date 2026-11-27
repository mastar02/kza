//! Voice orchestration and the local Nexa control plane.

pub mod ambient;
pub mod clients;
pub mod control;
pub mod events;
mod home;
pub mod notify;
pub mod pipeline;
mod tasks;

use std::{
    fmt::Debug,
    io,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use nexa_config::{ConfigError, LoadedBundle};
use nexa_ha::{
    ControlPolicy, ControlService, EffectIntent, HaClient, MembershipSource, ReviewedGroup,
};
use nexa_household::{Clock, HouseholdWorker, reminders::ReminderPolicy};
use nexa_llm::{ApiKey, CancellationToken, CloudConsentGate, Endpoint, FailoverRouter};
use nexa_nlu::{
    RouterEngine, gate::GateSettings, household::HouseholdIntentSettings,
    router_engine::RouterSettings,
};
use nexa_proto::v1::{CloudConsentStatus, LiveEventKind};
use serde_json::json;
use tonic::Status;

use clients::{AudioClient, MemoryClient, MlClient};
use events::Events;
use home::Home;

pub type BrainError = Box<dyn std::error::Error + Send + Sync>;
type Worker = HouseholdWorker<Arc<dyn BrainClock>>;

// Keep initial_prompt empty on whisper-v3-turbo: non-empty domain prompts inject decoder history
// that triggers prompt echoes ("Voseo, prendé, apagá.") and high-compression repetition loops on
// real speech. Domain vocabulary boosting is handled cleanly by COMMAND_STT_HOTWORDS instead.
pub(crate) const COMMAND_STT_INITIAL_PROMPT: &str = "";
pub(crate) const COMMAND_STT_HOTWORDS: &str = "Nexa prendé apagá prendelo apagalo prendela apagala subí bajá poné escritorio living cocina baño cuarto balcón escalera luz luces aire";
pub(crate) const COMMAND_STT_BEAM_SIZE: u32 = 5;

#[derive(Debug)]
pub struct Dependencies {
    pub audio: Arc<dyn AudioClient>,
    pub ml: Arc<dyn MlClient>,
    pub memory: Option<Arc<dyn MemoryClient>>,
    pub home: Option<HomeDependencies>,
    pub clock: Arc<dyn BrainClock>,
}

/// A clock shared by the voice router, control plane, and household database.
pub trait BrainClock: Clock + Debug {}
impl<T: Clock + Debug> BrainClock for T {}

/// Keep the process lock for as long as any blocking database operation can run.
#[derive(Debug)]
struct LockedClock {
    clock: Arc<dyn BrainClock>,
    _owner: std::fs::File,
}

impl Clock for LockedClock {
    fn now(&self) -> i64 {
        self.clock.now()
    }

    fn monotonic(&self) -> Duration {
        self.clock.monotonic()
    }
}

#[derive(Debug, Clone)]
pub struct HomeDependencies {
    pub client: HaClient,
    pub control: ControlService,
    forecast: ForecastReader,
}

#[derive(Clone)]
struct ForecastReader {
    client: reqwest::Client,
    url: reqwest::Url,
    token: reqwest::header::HeaderValue,
}

impl Debug for ForecastReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ForecastReader").finish_non_exhaustive()
    }
}

impl HomeDependencies {
    pub fn new(bundle: &LoadedBundle, token: &str) -> Result<Self, BrainError> {
        let grants = &bundle.ha_controls;
        grants.validate()?;
        let groups = grants
            .groups
            .iter()
            .map(|(entity_id, members)| {
                (
                    entity_id.clone(),
                    ReviewedGroup {
                        members: members.clone(),
                        source: MembershipSource::HaState,
                    },
                )
            })
            .collect();
        let scenes = grants
            .scenes
            .iter()
            .map(|(entity_id, effects)| {
                let effects = effects
                    .iter()
                    .map(|effect| {
                        Ok::<_, ConfigError>(EffectIntent {
                            entity_id: effect.entity_id.clone(),
                            operation: effect.scene_operation(entity_id)?.into(),
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((entity_id.clone(), effects))
            })
            .collect::<Result<_, ConfigError>>()?;
        let policy = ControlPolicy::new(grants.clone()).with_effects(scenes, groups);

        let config = &bundle.core.home_assistant;
        let client = HaClient::new(config, token)?;
        let control = ControlService::new(client.clone(), policy);
        let mut url = reqwest::Url::parse(&config.url)?;
        url.path_segments_mut()
            .map_err(|()| io::Error::other("invalid HA path"))?
            .pop_if_empty()
            .extend(["api", "services", "weather", "get_forecasts"]);
        url.set_query(Some("return_response=true"));
        let mut token = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))?;
        token.set_sensitive(true);
        let forecast = ForecastReader {
            client: reqwest::Client::builder()
                .timeout(Duration::from_millis(config.timeout_ms))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            url,
            token,
        };
        Ok(Self {
            client,
            control,
            forecast,
        })
    }
}

#[derive(Debug)]
pub struct Brain {
    config: LoadedBundle,
    audio: Arc<dyn AudioClient>,
    ml: Arc<dyn MlClient>,
    memory: Option<Arc<dyn MemoryClient>>,
    clock: Arc<dyn BrainClock>,
    quiet: std::sync::Mutex<std::collections::HashSet<String>>,
    started: Instant,
    household: Worker,
    llm: FailoverRouter,
    endpoints: [Endpoint; 2],
    consent_file: PathBuf,
    consent: tokio::sync::Mutex<u64>,
    router: RouterEngine,
    timezone: chrono_tz::Tz,
    home: Home,
    pub events: Events,
    pub(crate) ambient: Arc<ambient::AmbientState>,
    shutdown: CancellationToken,
    active: pipeline::ActiveTurns,
}

fn api_key(value: String) -> Option<ApiKey> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_owned().into())
}

fn env_key(names: &[&str]) -> Option<ApiKey> {
    names
        .iter()
        .find_map(|name| std::env::var(name).ok().and_then(api_key))
}

impl Brain {
    /// Prepare local engines and open the household store exactly once.
    pub async fn open(config: LoadedBundle, deps: Dependencies) -> Result<Arc<Self>, BrainError> {
        let timezone = config.core.household.timezone.parse()?;
        let data = &config.core.paths.data_dir;
        let household_dir = data.join("household");
        tokio::fs::create_dir_all(&household_dir).await?;
        let lock_path = household_dir.join("owner.lock");
        let household_lock = tokio::task::spawn_blocking(move || {
            use std::os::unix::fs::OpenOptionsExt;
            let lock = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(false)
                .mode(0o600)
                .open(lock_path)?;
            rustix::fs::flock(&lock, rustix::fs::FlockOperation::NonBlockingLockExclusive)?;
            Ok::<_, io::Error>(lock)
        })
        .await??;
        let consent_file = household_dir.join("cloud-consent-withdrawn");
        let marker = match tokio::fs::read_to_string(&consent_file).await {
            // Honor withdrawals persisted before the household directory cutover.
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                tokio::fs::read_to_string(data.join("cloud-consent-withdrawn")).await
            }
            result => result,
        };
        let withdrawal = match marker {
            Ok(value) => Some(value.trim().parse::<u64>().unwrap_or_default()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let mut endpoints = Endpoint::from_config(&config.core.llm);
        endpoints[0].api_key = match env_key(&["NEXA_FAST_LLM_API_KEY", "LLAMA_API_KEY"]) {
            Some(key) => Some(key),
            None => {
                match tokio::fs::read_to_string(config.core.paths.secrets_dir.join("llama-api-key"))
                    .await
                {
                    Ok(value) => api_key(value),
                    Err(error) if error.kind() == io::ErrorKind::NotFound => None,
                    Err(error) => return Err(error.into()),
                }
            }
        };
        endpoints[1].api_key = env_key(&[
            "NEXA_CLOUD_LLM_API_KEY",
            "LLM_GATEWAY_API_KEY",
            "LITELLM_API_KEY",
        ]);
        let llm = FailoverRouter::new(
            endpoints.to_vec(),
            Arc::new(CloudConsentGate::new(config.core.llm.cloud_consent)),
            Some(household_dir.join("llm-cooldowns.json")),
        )
        .await?;
        if withdrawal.is_some() {
            llm.consent().withdraw();
        }
        let locked_clock = LockedClock {
            clock: deps.clock.clone(),
            _owner: household_lock,
        };
        let household =
            HouseholdWorker::open_in(household_dir, Arc::new(locked_clock) as Arc<dyn BrainClock>)
                .await?;
        let policy = ReminderPolicy {
            late_action_grace: Duration::from_secs(
                config.core.household.late_action_grace_s.into(),
            ),
            voice_choice_window: Duration::from_secs(
                config.core.household.voice_choice_window_s.into(),
            ),
            ..ReminderPolicy::default()
        };
        household
            .call(move |db| db.set_reminder_policy(policy))
            .await?;
        let nlu = &config.core.nlu;
        let router = RouterEngine::new(RouterSettings {
            gate: GateSettings {
                // Keep Whisper confidence checks in shadow mode: on whisper-v3-turbo over XVF3800
                // audio, avg_logprob/no_speech_prob reject real commands while compression_ratio
                // reliably isolates repetitive hallucinations (see legacy settings.yaml).
                initial_prompt: (!COMMAND_STT_INITIAL_PROMPT.is_empty())
                    .then(|| COMMAND_STT_INITIAL_PROMPT.to_owned()),
                enforce_confidence: false,
                enforce_compression_ratio: nlu.enforce_compression_ratio,
                max_compression_ratio: nlu.max_compression_ratio,
                ..GateSettings::default()
            },
            household: HouseholdIntentSettings {
                unvetoed_max_words: usize::from(nlu.unvetoed_max_words),
                compound_max_parts: usize::from(nlu.compound_max_parts),
                compound_execute: nlu.compound_execute,
                ..HouseholdIntentSettings::default()
            },
            wake_words: vec!["neska".into(), "nexca".into()],
            indoor_rooms: config
                .rooms
                .rooms
                .iter()
                .map(|room| room.name.clone())
                .collect(),
            ..RouterSettings::default()
        });
        let short_term_turns = usize::from(config.core.memory.short_term_turns);
        Ok(Arc::new(Self {
            endpoints,
            config,
            audio: deps.audio,
            ml: deps.ml,
            memory: deps.memory,
            clock: deps.clock,
            quiet: Default::default(),
            started: Instant::now(),
            household,
            llm,
            consent_file,
            consent: tokio::sync::Mutex::new(withdrawal.unwrap_or_default()),
            router,
            timezone,
            home: Home::new(deps.home),
            events: Events::default(),
            ambient: Arc::new(ambient::AmbientState::new(short_term_turns)),
            shutdown: CancellationToken::new(),
            active: pipeline::ActiveTurns::default(),
        }))
    }

    /// Run the local voice, state, and housekeeping loops until shutdown.
    pub async fn run_background(self: Arc<Self>) {
        tokio::join!(
            Arc::clone(&self).run_voice(),
            Arc::clone(&self).run_household(),
            self.run_home_events(),
        );
    }

    pub fn shutdown(&self) {
        self.shutdown.cancel();
    }

    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    pub(crate) fn now(&self) -> u64 {
        unsigned(self.clock.now())
    }

    pub(crate) fn event(
        &self,
        kind: LiveEventKind,
        room: &str,
        turn: &str,
        payload: serde_json::Value,
    ) {
        self.events.publish(kind, room, turn, self.now(), payload);
    }

    pub(crate) async fn cloud_status(&self) -> CloudConsentStatus {
        self.cloud_status_at(*self.consent.lock().await)
    }

    fn cloud_status_at(&self, updated_at_unix_ms: u64) -> CloudConsentStatus {
        CloudConsentStatus {
            consent_granted: self.llm.consent().is_allowed(),
            withdrawn_latched: self.llm.consent().is_withdrawn(),
            updated_at_unix_ms,
        }
    }

    #[allow(
        clippy::result_large_err,
        reason = "Returns the Tonic status if the withdrawal could not be persisted"
    )]
    pub(crate) async fn withdraw_cloud(self: &Arc<Self>) -> Result<CloudConsentStatus, Status> {
        // Complete the durable withdrawal even if the client disconnects.
        let brain = Arc::clone(self);
        tokio::spawn(async move {
            let mut timestamp = brain.consent.lock().await;
            let changed = !brain.llm.consent().is_withdrawn();
            brain.llm.consent().withdraw();
            if changed {
                brain.event(
                    LiveEventKind::Cooldown,
                    "",
                    "",
                    json!({"event": "cloud_consent_withdrawn"}),
                );
            }
            if *timestamp == 0 || changed {
                let now = brain.now().max(1);
                let path = brain.consent_file.clone();
                tokio::task::spawn_blocking(move || {
                    use std::io::Write;
                    use std::os::unix::fs::OpenOptionsExt;
                    let mut marker = std::fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .truncate(false)
                        .mode(0o600)
                        .open(&path)?;
                    marker.write_all(now.to_string().as_bytes())?;
                    marker.sync_all()?;
                    if let Some(parent) = path.parent() {
                        std::fs::File::open(parent)?.sync_all()?;
                    }
                    Ok::<_, io::Error>(())
                })
                .await
                .map_err(|_| Status::internal("withdrawal persistence worker failed"))?
                .map_err(|_| Status::internal("could not persist cloud withdrawal"))?;
                *timestamp = now;
            }
            Ok(brain.cloud_status_at(*timestamp))
        })
        .await
        .map_err(|_| Status::internal("withdrawal worker failed"))?
    }

    fn output_zone(&self, room: &str) -> String {
        format!("zone_{room}")
    }

    fn room_for_zone(&self, zone: &str) -> Option<&str> {
        self.config
            .rooms
            .rooms
            .iter()
            .find(|room| zone.strip_prefix("zone_") == Some(room.room_id.as_str()))
            .map(|room| room.room_id.as_str())
    }

    pub(crate) fn zone(&self, id: &str) -> bool {
        self.config
            .rooms
            .rooms
            .iter()
            .any(|room| room.room_id == id)
    }
}

pub(crate) fn unsigned(value: i64) -> u64 {
    u64::try_from(value).unwrap_or_default()
}

pub(crate) fn id(prefix: &str) -> String {
    static SEQUENCE: AtomicU64 = AtomicU64::new(1);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{prefix}-{}-{:x}-{:x}",
        std::process::id(),
        now.as_nanos(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}
