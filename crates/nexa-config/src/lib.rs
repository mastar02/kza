//! Typed, boot-validated configuration loader for Nexa v2.
//!
//! Replaces the monolithic 2,195-line `config/settings.yaml` with three focused,
//! schema-validated TOML files (`nexa.toml`, `rooms.toml`, `ha-controls.toml`)
//! with zero hardcoded legacy `/home/kza` paths.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Reserved TCP port range for KZA/Nexa per `docs/SERVER_CONVENTIONS.md`.
pub const NEXA_PORT_MIN: u16 = 9500;
pub const NEXA_PORT_MAX: u16 = 9599;

/// Forbidden legacy path prefix that must never leak into Nexa v2 runtime configs.
const LEGACY_KZA_HOME_PREFIX: &str = "/home/kza";

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read config file {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse TOML in {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("configuration validation failed: {0}")]
    Validation(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NexaConfig {
    pub paths: PathsConfig,
    pub audio: AudioConfig,
    pub ml: MlConfig,
    pub nlu: NluConfig,
    pub home_assistant: HomeAssistantConfig,
    pub llm: LlmConfig,
    pub memory: MemoryConfig,
    pub household: HouseholdConfig,
    pub web: WebConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PathsConfig {
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
    pub runtime_dir: PathBuf,
    pub secrets_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AudioConfig {
    pub capture_sample_rate: u32,
    pub playback_sample_rate: u32,
    pub frame_ms: u16,
    pub wake_model: String,
    pub wake_threshold: f32,
    pub silero_vad_model: String,
    #[serde(default)]
    pub parakeet_model: String,
    pub ma1260_connection: Ma1260Connection,
    pub barge_in_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Ma1260Connection {
    Simulation,
    Serial { device: PathBuf, baud_rate: u32 },
    Tcp { host: String, port: u16 },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MlConfig {
    pub command_stt_model: String,
    pub command_stt_device: String,
    pub command_stt_compute_type: String,
    pub ambient_stt_model: String,
    pub ambient_stt_device: String,
    #[serde(default)]
    pub ambient_stt_compute_type: Option<String>,
    pub ambient_stt_workers: u8,
    pub tts_voice: String,
    pub tts_lang_code: String,
    pub tts_device: String,
    pub speaker_encoder_device: String,
    pub embeddings_model: String,
    pub embeddings_device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NluConfig {
    pub language: String,
    pub max_compression_ratio: f32,
    pub enforce_compression_ratio: bool,
    pub unvetoed_max_words: u8,
    pub compound_max_parts: u8,
    pub compound_execute: bool,
    pub min_llm_command_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HomeAssistantConfig {
    pub url: String,
    pub timeout_ms: u64,
    pub state_confirmation_timeout_ms: u64,
    pub cache_max_staleness_s: u64,
    pub full_refresh_interval_s: u64,
    pub unavailable_precheck_enabled: bool,
    pub weather_entity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LlmConfig {
    pub fast_router_url: String,
    pub fast_router_model: String,
    pub fast_router_timeout_ms: u64,
    pub cloud_gateway_url: String,
    pub cloud_model: String,
    pub cloud_timeout_ms: u64,
    pub cloud_consent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    pub enabled: bool,
    pub automatic_retention_days: u16,
    pub short_term_turns: u16,
    pub voice_proposals_enabled: bool,
    pub ambient_enabled: bool,
    pub ambient_keep_audio: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HouseholdConfig {
    pub timezone: String,
    pub default_output_zone: String,
    pub late_action_grace_s: u32,
    pub voice_choice_window_s: u32,
    pub default_list_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WebConfig {
    pub bind_host: String,
    pub port: u16,
    pub proxy_tls_port: u16,
    pub trusted_proxy_ip: String,
    pub home_enabled: bool,
    pub private_assets_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoomsConfig {
    pub rooms: Vec<RoomDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoomDefinition {
    pub room_id: String,
    pub name: String,
    pub mic_usb_port: Option<String>,
    pub mic_serial: Option<String>,
    pub mic_channel: u8,
    pub ma1260_zone: u8,
    pub default_light: Option<String>,
    pub default_climate: Option<String>,
    pub xvf_tuning: XvfTuning,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct XvfTuning {
    pub pp_agc_max_gain: f32,
    pub pp_attns_mode: u8,
    pub pp_attns_slope: f32,
    pub pp_min_nn: f32,
}

/// Explicit fail-closed allowlist of Home Assistant controls.
/// Empty `entities` denies every mutation by design.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct HaControlPolicyConfig {
    #[serde(default)]
    pub entities: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub groups: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub scenes: HashMap<String, Vec<SceneEffectConfig>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SceneEffectConfig {
    pub entity_id: String,
    pub operation: String,
    #[serde(default)]
    pub brightness_pct: Option<u8>,
    #[serde(default)]
    pub brightness: Option<u8>,
    #[serde(default)]
    pub color_temp_kelvin: Option<u32>,
}

/// Reviewed scene effects support only light and switch operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneOperation {
    TurnOn,
    TurnOff,
    Brightness(SceneBrightness),
    ColorTempKelvin {
        color_temp_kelvin: u32,
        brightness: Option<SceneBrightness>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneBrightness {
    Percent(u8),
    Raw(u8),
}

impl SceneEffectConfig {
    pub fn scene_operation(&self, scene: &str) -> Result<SceneOperation, ConfigError> {
        let invalid = || {
            ConfigError::Validation(format!(
                "scene '{scene}' has an unsupported operation or invalid parameters for '{}' on '{}'",
                self.operation, self.entity_id
            ))
        };
        let brightness = match (self.brightness_pct, self.brightness) {
            (Some(pct @ 0..=100), None) => Some(SceneBrightness::Percent(pct)),
            (None, Some(raw)) => Some(SceneBrightness::Raw(raw)),
            (None, None) => None,
            _ => return Err(invalid()),
        };
        match (
            ha_entity_domain(&self.entity_id),
            self.operation.as_str(),
            brightness,
            self.color_temp_kelvin,
        ) {
            (Some("light" | "switch"), "turn_on", None, None) => Ok(SceneOperation::TurnOn),
            (Some("light" | "switch"), "turn_off", None, None) => Ok(SceneOperation::TurnOff),
            (Some("light"), "brightness", Some(value), None) => {
                Ok(SceneOperation::Brightness(value))
            }
            (Some("light"), "color_temp_kelvin", brightness, Some(kelvin @ 1..))
                if !matches!(
                    brightness,
                    Some(SceneBrightness::Percent(0) | SceneBrightness::Raw(0))
                ) =>
            {
                Ok(SceneOperation::ColorTempKelvin {
                    color_temp_kelvin: kelvin,
                    brightness,
                })
            }
            _ => Err(invalid()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoadedBundle {
    pub core: NexaConfig,
    pub rooms: RoomsConfig,
    pub ha_controls: HaControlPolicyConfig,
}

impl NexaConfig {
    pub fn from_toml_str(raw: &str) -> Result<Self, ConfigError> {
        let cfg: Self = toml::from_str(raw).map_err(|source| ConfigError::Parse {
            path: PathBuf::from("<inline>"),
            source,
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path_ref = path.as_ref();
        let raw = fs::read_to_string(path_ref).map_err(|source| ConfigError::Io {
            path: path_ref.to_path_buf(),
            source,
        })?;
        let cfg: Self = toml::from_str(&raw).map_err(|source| ConfigError::Parse {
            path: path_ref.to_path_buf(),
            source,
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        for (label, path) in [
            ("paths.data_dir", &self.paths.data_dir),
            ("paths.models_dir", &self.paths.models_dir),
            ("paths.runtime_dir", &self.paths.runtime_dir),
            ("paths.secrets_dir", &self.paths.secrets_dir),
            ("web.private_assets_dir", &self.web.private_assets_dir),
        ] {
            reject_legacy_path(label, path)?;
        }
        reject_legacy_str("llm.fast_router_model", &self.llm.fast_router_model)?;
        reject_legacy_str("ml.command_stt_model", &self.ml.command_stt_model)?;
        reject_legacy_str("ml.ambient_stt_model", &self.ml.ambient_stt_model)?;

        if self.audio.capture_sample_rate != 16_000 {
            return Err(ConfigError::Validation(
                "audio.capture_sample_rate must be 16000 Hz for XVF3800/Silero/OWW/Whisper".into(),
            ));
        }
        if self.audio.frame_ms != 80 {
            return Err(ConfigError::Validation(
                "audio.frame_ms must be 80 ms for Silero VAD and OWW".into(),
            ));
        }
        if !matches!(
            self.audio.playback_sample_rate,
            16_000 | 22_050 | 24_000 | 48_000
        ) {
            return Err(ConfigError::Validation(
                "audio.playback_sample_rate must be one of 16000, 22050, 24000, 48000".into(),
            ));
        }
        if !(0.05..=0.99).contains(&self.audio.wake_threshold) {
            return Err(ConfigError::Validation(
                "audio.wake_threshold must be within [0.05, 0.99]".into(),
            ));
        }
        if self.ml.ambient_stt_workers == 0 || self.ml.ambient_stt_workers > 4 {
            return Err(ConfigError::Validation(
                "ml.ambient_stt_workers must be between 1 and 4".into(),
            ));
        }
        if let Some(compute_type) = &self.ml.ambient_stt_compute_type
            && compute_type.trim().is_empty()
        {
            return Err(ConfigError::Validation(
                "ml.ambient_stt_compute_type must not be empty".into(),
            ));
        }
        if self.nlu.compound_max_parts == 0 || self.nlu.compound_max_parts > 5 {
            return Err(ConfigError::Validation(
                "nlu.compound_max_parts must be between 1 and 5".into(),
            ));
        }
        if !(1.1..=5.0).contains(&self.nlu.max_compression_ratio) {
            return Err(ConfigError::Validation(
                "nlu.max_compression_ratio must be within [1.1, 5.0]".into(),
            ));
        }
        if self.home_assistant.timeout_ms == 0
            || self.home_assistant.state_confirmation_timeout_ms == 0
            || self.home_assistant.full_refresh_interval_s == 0
        {
            return Err(ConfigError::Validation(
                "home_assistant timeouts and refresh intervals must be > 0".into(),
            ));
        }
        if self.home_assistant.full_refresh_interval_s >= self.home_assistant.cache_max_staleness_s
        {
            return Err(ConfigError::Validation(
                "home_assistant.full_refresh_interval_s must be less than cache_max_staleness_s"
                    .into(),
            ));
        }
        if self.llm.fast_router_timeout_ms == 0 || self.llm.cloud_timeout_ms == 0 {
            return Err(ConfigError::Validation("llm timeouts must be > 0".into()));
        }
        if !(NEXA_PORT_MIN..=NEXA_PORT_MAX).contains(&self.web.port) {
            return Err(ConfigError::Validation(format!(
                "web.port ({}) must stay inside reserved KZA/Nexa range {NEXA_PORT_MIN}..={NEXA_PORT_MAX}",
                self.web.port
            )));
        }
        if !(NEXA_PORT_MIN..=NEXA_PORT_MAX).contains(&self.web.proxy_tls_port)
            || self.web.proxy_tls_port == self.web.port
        {
            return Err(ConfigError::Validation(format!(
                "web.proxy_tls_port ({}) must be distinct from web.port and within {NEXA_PORT_MIN}..={NEXA_PORT_MAX}",
                self.web.proxy_tls_port
            )));
        }
        if self.web.bind_host != "127.0.0.1" && self.web.bind_host != "::1" {
            return Err(ConfigError::Validation(
                "web.bind_host must be loopback (127.0.0.1 or ::1); LAN exposure goes through TLS proxy (R6)".into(),
            ));
        }
        if self.web.trusted_proxy_ip != "127.0.0.1" && self.web.trusted_proxy_ip != "::1" {
            return Err(ConfigError::Validation(
                "web.trusted_proxy_ip must be loopback (127.0.0.1 or ::1)".into(),
            ));
        }
        if self.household.timezone.trim().is_empty()
            || self.household.default_output_zone.trim().is_empty()
        {
            return Err(ConfigError::Validation(
                "household.timezone and household.default_output_zone must be non-empty".into(),
            ));
        }
        Ok(())
    }
}

impl RoomsConfig {
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path_ref = path.as_ref();
        let raw = fs::read_to_string(path_ref).map_err(|source| ConfigError::Io {
            path: path_ref.to_path_buf(),
            source,
        })?;
        let cfg: Self = toml::from_str(&raw).map_err(|source| ConfigError::Parse {
            path: path_ref.to_path_buf(),
            source,
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.rooms.is_empty() {
            return Err(ConfigError::Validation(
                "rooms.toml must declare at least one room".into(),
            ));
        }
        let mut seen_ids = HashSet::new();
        let mut seen_ports = HashSet::new();
        for room in &self.rooms {
            if room.room_id.trim().is_empty() || room.name.trim().is_empty() {
                return Err(ConfigError::Validation(
                    "every room requires non-empty room_id and name".into(),
                ));
            }
            if !seen_ids.insert(room.room_id.as_str()) {
                return Err(ConfigError::Validation(format!(
                    "duplicate room_id '{}'",
                    room.room_id
                )));
            }
            if let Some(port) = room.mic_usb_port.as_deref() {
                if port.trim().is_empty() {
                    return Err(ConfigError::Validation(format!(
                        "room '{}' has empty mic_usb_port",
                        room.room_id
                    )));
                }
                if !seen_ports.insert(port) {
                    return Err(ConfigError::Validation(format!(
                        "duplicate mic_usb_port '{port}'"
                    )));
                }
            }
            if room.mic_channel > 5 {
                return Err(ConfigError::Validation(format!(
                    "room '{}' mic_channel ({}) out of range 0..=5",
                    room.room_id, room.mic_channel
                )));
            }
            if room.ma1260_zone == 0 || room.ma1260_zone > 6 {
                return Err(ConfigError::Validation(format!(
                    "room '{}' ma1260_zone ({}) must be in 1..=6",
                    room.room_id, room.ma1260_zone
                )));
            }
            if !(0.0..=1.0).contains(&room.xvf_tuning.pp_min_nn) {
                return Err(ConfigError::Validation(format!(
                    "room '{}' xvf_tuning.pp_min_nn must be in [0.0, 1.0]",
                    room.room_id
                )));
            }
        }
        Ok(())
    }
}

impl HaControlPolicyConfig {
    pub fn load_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path_ref = path.as_ref();
        let raw = fs::read_to_string(path_ref).map_err(|source| ConfigError::Io {
            path: path_ref.to_path_buf(),
            source,
        })?;
        let cfg: Self = toml::from_str(&raw).map_err(|source| ConfigError::Parse {
            path: path_ref.to_path_buf(),
            source,
        })?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        for (entity_id, ops) in &self.entities {
            if ha_entity_domain(entity_id).is_none() {
                return Err(ConfigError::Validation(format!(
                    "invalid HA entity_id '{entity_id}'"
                )));
            }
            if ops.is_empty() {
                return Err(ConfigError::Validation(format!(
                    "entity '{entity_id}' in ha-controls must specify at least one allowed operation"
                )));
            }
            if let Some(op) = ops.iter().find(|op| !supports_ha_operation(entity_id, op)) {
                return Err(ConfigError::Validation(format!(
                    "unsupported HA operation '{op}' for '{entity_id}'"
                )));
            }
        }
        for (entity_id, members) in &self.groups {
            if ha_entity_domain(entity_id) != Some("light")
                || members.is_empty()
                || members
                    .iter()
                    .any(|member| ha_entity_domain(member) != Some("light"))
            {
                return Err(ConfigError::Validation(format!(
                    "group '{entity_id}' in ha-controls must be a light with at least one light member"
                )));
            }
            if members.iter().collect::<HashSet<_>>().len() != members.len() {
                return Err(ConfigError::Validation(format!(
                    "duplicate member in group '{entity_id}'"
                )));
            }
        }
        validate_group_graph(&self.groups)?;
        for (entity_id, effects) in &self.scenes {
            if ha_entity_domain(entity_id) != Some("scene") || effects.is_empty() {
                return Err(ConfigError::Validation(format!(
                    "scene '{entity_id}' in ha-controls must be a scene with at least one effect"
                )));
            }
            let mut targets = HashSet::new();
            for effect in effects {
                if !targets.insert(&effect.entity_id) {
                    return Err(ConfigError::Validation(format!(
                        "scene '{entity_id}' repeats target '{}'",
                        effect.entity_id
                    )));
                }
                effect.scene_operation(entity_id)?;
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn allows(&self, entity_id: &str, operation: &str) -> bool {
        self.entities
            .get(entity_id)
            .is_some_and(|ops| ops.iter().any(|op| op == operation))
    }
}

impl LoadedBundle {
    pub fn load_from_dir(dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let base = dir.as_ref();
        let core = NexaConfig::load_file(base.join("nexa.toml"))?;
        let rooms = RoomsConfig::load_file(base.join("rooms.toml"))?;
        let controls_path = base.join("ha-controls.toml");
        let ha_controls = if controls_path.is_file() {
            HaControlPolicyConfig::load_file(controls_path)?
        } else {
            HaControlPolicyConfig::default()
        };
        Ok(Self {
            core,
            rooms,
            ha_controls,
        })
    }
}

// Count incoming group edges and remove roots iteratively, including disconnected components.
fn validate_group_graph(groups: &HashMap<String, Vec<String>>) -> Result<(), ConfigError> {
    let mut incoming: HashMap<_, usize> = groups.keys().map(|id| (id, 0)).collect();
    for member in groups.values().flatten() {
        if let Some(count) = incoming.get_mut(member) {
            *count += 1;
        }
    }
    let mut roots: Vec<_> = incoming
        .iter()
        .filter_map(|(&id, &count)| (count == 0).then_some(id))
        .collect();
    let mut visited = 0;
    while let Some(root) = roots.pop() {
        visited += 1;
        for member in &groups[root] {
            if let Some(count) = incoming.get_mut(member) {
                *count -= 1;
                if *count == 0 {
                    roots.push(member);
                }
            }
        }
    }
    if visited != groups.len() {
        return Err(ConfigError::Validation(
            "cycle in ha-controls groups".into(),
        ));
    }
    Ok(())
}

pub fn supports_ha_operation(entity_id: &str, operation: &str) -> bool {
    match ha_entity_domain(entity_id) {
        Some("light") => matches!(
            operation,
            "turn_on" | "turn_off" | "brightness" | "color_temp_kelvin"
        ),
        Some("switch") => matches!(operation, "turn_on" | "turn_off"),
        Some("climate") => matches!(operation, "set_temperature" | "set_hvac_mode"),
        Some("cover") => matches!(
            operation,
            "open_cover" | "close_cover" | "stop_cover" | "set_cover_position"
        ),
        Some("scene") => operation == "activate_scene",
        _ => false,
    }
}

fn ha_entity_domain(entity_id: &str) -> Option<&str> {
    let (domain, name) = entity_id.split_once('.')?;
    let valid_part = |part: &str| {
        !part.is_empty()
            && part
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    };
    (valid_part(domain) && valid_part(name)).then_some(domain)
}

fn reject_legacy_path(label: &str, path: &Path) -> Result<(), ConfigError> {
    reject_legacy_str(label, &path.to_string_lossy())
}

fn reject_legacy_str(label: &str, value: &str) -> Result<(), ConfigError> {
    if value.contains(LEGACY_KZA_HOME_PREFIX) {
        return Err(ConfigError::Validation(format!(
            "{label} must not reference legacy '{LEGACY_KZA_HOME_PREFIX}'; use /var/lib/nexa, /opt/nexa, or /etc/nexa"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn sample_nexa_toml() -> &'static str {
        include_str!("../../../config/nexa/nexa.toml")
    }

    fn sample_rooms_toml() -> &'static str {
        include_str!("../../../config/nexa/rooms.example.toml")
    }

    fn sample_ha_controls_toml() -> &'static str {
        include_str!("../../../config/nexa/ha-controls.example.toml")
    }

    #[test]
    fn committed_templates_parse_and_validate() {
        let core = NexaConfig::from_toml_str(sample_nexa_toml()).expect("nexa.toml valid");
        assert_eq!(core.web.bind_host, "127.0.0.1");
        assert_eq!(core.web.port, 9500);
        assert_eq!(core.web.proxy_tls_port, 9540);
        assert!(
            core.home_assistant.full_refresh_interval_s < core.home_assistant.cache_max_staleness_s,
            "the HA cache should stay fresh between scheduled full refreshes"
        );

        let rooms: RoomsConfig =
            toml::from_str(sample_rooms_toml()).expect("rooms template parses");
        rooms.validate().expect("rooms template validates");
        assert!(rooms.rooms.len() >= 4);

        let controls: HaControlPolicyConfig =
            toml::from_str(sample_ha_controls_toml()).expect("ha-controls template parses");
        controls.validate().expect("ha-controls template validates");
        assert_eq!(controls, HaControlPolicyConfig::default());
    }

    #[test]
    fn legacy_entity_grants_load_without_reviewed_definitions() {
        let policy: HaControlPolicyConfig = toml::from_str(
            r#"[entities]
                "light.living" = ["turn_on"]
            "#,
        )
        .expect("legacy controls parse");
        policy.validate().expect("legacy controls validate");
        assert!(policy.allows("light.living", "turn_on"));
        assert!(policy.groups.is_empty());
        assert!(policy.scenes.is_empty());
    }

    #[test]
    fn reviewed_groups_and_scenes_load_without_granting_operations() {
        let policy: HaControlPolicyConfig = toml::from_str(
            r#"
            [groups]
            "light.group" = ["light.leaf"]

            [scenes]
            "scene.reading" = [
                { entity_id = "light.group", operation = "brightness", brightness_pct = 100 },
                { entity_id = "light.desk", operation = "color_temp_kelvin", color_temp_kelvin = 3000, brightness = 200 },
                { entity_id = "switch.lamp", operation = "turn_off" },
            ]
            "#,
        )
        .expect("reviewed definitions parse");
        policy.validate().expect("reviewed definitions validate");
        assert_eq!(policy.groups["light.group"], ["light.leaf"]);
        let effects = &policy.scenes["scene.reading"];
        assert_eq!(effects[0].brightness_pct, Some(100));
        assert_eq!(effects[1].brightness, Some(200));
        assert_eq!(effects[1].color_temp_kelvin, Some(3000));
        assert_eq!(effects[2].brightness, None);
        assert!(!policy.allows("scene.reading", "activate_scene"));
        assert!(!policy.allows("light.group", "brightness"));
    }

    #[test]
    fn rejects_invalid_reviewed_groups_and_scenes() {
        for definition in [
            r#"[groups]
            "switch.group" = ["light.leaf"]"#,
            r#"[groups]
            "lock.group" = ["light.leaf"]"#,
            r#"[groups]
            "light.group" = []"#,
            r#"[groups]
            "light.group" = ["switch.member"]"#,
            r#"[groups]
            "light.group" = ["alarm_control_panel.home"]"#,
            r#"[scenes]
            "light.reading" = [{ entity_id = "light.leaf", operation = "turn_on" }]"#,
            r#"[scenes]
            "scene.reading" = []"#,
            r#"[scenes]
            "scene.reading" = [{ entity_id = "lock.door", operation = "turn_on" }]"#,
            r#"[scenes]
            "scene.reading" = [{ entity_id = "alarm_control_panel.home", operation = "turn_off" }]"#,
            r#"[scenes]
            "scene.reading" = [{ entity_id = "light.", operation = "turn_on" }]"#,
            r#"[scenes]
            "scene.reading" = [{ entity_id = "light.leaf", operation = "brightness", brightness_pct = 101 }]"#,
            r#"[scenes]
            "scene.reading" = [{ entity_id = "light.leaf", operation = "brightness", brightness_pct = 40, brightness = 20 }]"#,
        ] {
            let policy: HaControlPolicyConfig =
                toml::from_str(definition).expect("definitions parse before validation");
            assert!(policy.validate().is_err(), "accepted {definition}");
        }
    }

    #[test]
    fn ha_timing_boundaries() {
        let mut cfg = NexaConfig::from_toml_str(sample_nexa_toml()).expect("template");
        cfg.home_assistant.full_refresh_interval_s = 2;
        cfg.home_assistant.cache_max_staleness_s = 3;
        cfg.home_assistant.state_confirmation_timeout_ms = 1;
        assert!(cfg.validate().is_ok());
        cfg.home_assistant.state_confirmation_timeout_ms = 0;
        assert!(cfg.validate().is_err());
        cfg.home_assistant.state_confirmation_timeout_ms = 1;
        for staleness in [1, 2] {
            cfg.home_assistant.cache_max_staleness_s = staleness;
            assert!(cfg.validate().is_err());
        }
    }

    #[test]
    fn validates_grant_operation_against_entity_domain() {
        for (id, ops) in [
            ("light.a.b", vec!["turn_on"]),
            ("Light.a", vec!["turn_on"]),
            ("light.a-b", vec!["turn_on"]),
            ("sensor.a", vec!["turn_on"]),
            ("light.a", vec![]),
            ("light.a", vec!["turn_on", ""]),
            ("switch.a", vec!["brightness"]),
            ("cover.a", vec!["turn_off"]),
            ("scene.a", vec!["turn_on"]),
            ("climate.a", vec!["activate_scene"]),
        ] {
            let policy = HaControlPolicyConfig {
                entities: HashMap::from([(id.into(), ops.into_iter().map(String::from).collect())]),
                ..Default::default()
            };
            assert!(policy.validate().is_err(), "accepted {id}");
        }
    }

    #[test]
    fn rejects_repeated_members_and_cycles_but_accepts_shared_subgroups() {
        for (members, valid) in [
            (vec!["light.leaf", "light.leaf"], false),
            (vec!["light.c"], false),
            (vec!["light.a"], false),
            (vec!["light.leaf"], true),
        ] {
            let mut policy = HaControlPolicyConfig::default();
            for (group, children) in [
                ("light.a", vec!["light.b", "light.c"]),
                ("light.b", vec!["light.c"]),
                ("light.c", members),
            ] {
                policy.groups.insert(
                    group.into(),
                    children.into_iter().map(String::from).collect(),
                );
            }
            assert_eq!(policy.validate().is_ok(), valid);
        }
    }

    #[test]
    fn rejects_invalid_scene_operation_shapes_and_repeated_targets() {
        for fields in [
            r#"operation = "toggle""#,
            r#"operation = "brightness""#,
            r#"operation = "turn_off", brightness = 1"#,
            r#"operation = "color_temp_kelvin""#,
            r#"operation = "color_temp_kelvin", color_temp_kelvin = 0"#,
            r#"operation = "color_temp_kelvin", color_temp_kelvin = 1, brightness = 0"#,
            r#"operation = "color_temp_kelvin", color_temp_kelvin = 1, brightness_pct = 0"#,
            r#"operation = "brightness", brightness = 1, color_temp_kelvin = 1"#,
        ] {
            let definition = format!(
                r#"[scenes]
            "scene.test" = [{{entity_id = "light.test", {fields}}}]"#
            );
            let policy: HaControlPolicyConfig = toml::from_str(&definition).expect("scene");
            assert!(policy.validate().is_err(), "accepted {fields}");
        }
        for effects in [
            r#"{entity_id = "switch.test", operation = "brightness", brightness_pct = 1}"#,
            r#"{entity_id = "light.test", operation = "turn_on"}, {entity_id = "light.test", operation = "turn_off"}"#,
        ] {
            let definition = format!(
                r#"[scenes]
            "scene.test" = [{effects}]"#
            );
            let policy: HaControlPolicyConfig = toml::from_str(&definition).expect("scene");
            assert!(policy.validate().is_err(), "accepted {effects}");
        }
    }

    #[test]
    fn rejects_unknown_scene_effect_fields() {
        let result = toml::from_str::<HaControlPolicyConfig>(
            r#"[scenes]
            "scene.reading" = [{ entity_id = "light.leaf", operation = "turn_on", transition = 1 }]"#,
        );
        assert!(result.is_err());
    }

    #[test]
    fn rejects_hardcoded_legacy_kza_paths_and_non_loopback_web_bind() {
        let mut cfg = NexaConfig::from_toml_str(sample_nexa_toml()).expect("valid base");
        cfg.paths.models_dir = PathBuf::from("/home/kza/kza/models");
        assert!(cfg.validate().is_err());

        let mut cfg2 = NexaConfig::from_toml_str(sample_nexa_toml()).expect("valid base");
        cfg2.web.bind_host = "0.0.0.0".into();
        assert!(cfg2.validate().is_err());
    }

    #[test]
    fn rejects_restricted_security_domains_in_ha_controls() {
        let mut policy = HaControlPolicyConfig::default();
        policy
            .entities
            .insert("lock.front_door".into(), vec!["unlock".into()]);
        assert!(policy.validate().is_err());

        let mut alarm = HaControlPolicyConfig::default();
        alarm.entities.insert(
            "alarm_control_panel.home".into(),
            vec!["alarm_disarm".into()],
        );
        assert!(alarm.validate().is_err());
    }

    #[test]
    fn rejects_non_80ms_audio_frame_and_empty_ambient_compute_type() {
        let mut bad_frame = NexaConfig::from_toml_str(sample_nexa_toml()).expect("valid base");
        bad_frame.audio.frame_ms = 40;
        assert!(bad_frame.validate().is_err());

        let mut empty_ct = NexaConfig::from_toml_str(sample_nexa_toml()).expect("valid base");
        empty_ct.ml.ambient_stt_compute_type = Some("   ".into());
        assert!(empty_ct.validate().is_err());

        let base = NexaConfig::from_toml_str(sample_nexa_toml()).expect("valid base");
        assert_eq!(base.ml.ambient_stt_compute_type.as_deref(), Some("float16"));
    }

    proptest! {
        #[test]
        fn rejects_ports_outside_reserved_nexa_range(port in 1u16..65535u16) {
            let mut cfg = NexaConfig::from_toml_str(sample_nexa_toml()).expect("valid base");
            cfg.web.port = port;
            let expected_ok = (NEXA_PORT_MIN..=NEXA_PORT_MAX).contains(&port) && port != cfg.web.proxy_tls_port;
            prop_assert_eq!(cfg.validate().is_ok(), expected_ok);
        }
    }
}
