//! Home Assistant grants, typed operation translation, state and receipts.

#![allow(
    clippy::result_large_err,
    reason = "Tonic status carries gRPC metadata and codes across this boundary"
)]

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::Mutex,
};

use nexa_ha::{
    ActionRequest, ActionStatus, Brightness, EffectIntent, HvacMode, Operation, authorize_action,
};
use nexa_household::HomeAction;
use nexa_nlu::{
    grammar,
    normalize::words,
    world::{self, CurrentWeather, ForecastEntry, WeatherHorizon},
};
use nexa_proto::v1::{HomeActionReceipt, HomeCatalogSnapshot, HomeEntity, HomeRoom, LiveEventKind};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::Instant;
use tonic::Status;

use crate::{Brain, HomeDependencies, unsigned};

#[derive(Debug)]
pub(crate) struct Home {
    pub(crate) deps: Option<HomeDependencies>,
    receipts: Mutex<VecDeque<HomeActionReceipt>>,
}

impl Home {
    pub(crate) fn new(deps: Option<HomeDependencies>) -> Self {
        Self {
            deps,
            receipts: Mutex::default(),
        }
    }

    pub(crate) fn history(&self, limit: usize) -> Vec<HomeActionReceipt> {
        self.receipts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
}

pub(crate) fn typed_operation(
    name: &str,
    params: &HashMap<String, String>,
) -> Result<Operation, Status> {
    let bad = || Status::invalid_argument("invalid HA parameters");
    let param = |key: &str| params.get(key).ok_or_else(bad);
    let byte = |key: &str| param(key)?.parse::<u8>().map_err(|_| bad());
    let brightness = || byte("brightness_pct").map(Brightness::Percent);
    let op = match (name, params.len()) {
        ("turn_on", 0) => Operation::TurnOn,
        ("turn_off", 0) => Operation::TurnOff,
        ("activate_scene", 0) => Operation::ActivateScene,
        ("open_cover", 0) => Operation::OpenCover,
        ("close_cover", 0) => Operation::CloseCover,
        ("stop_cover", 0) => Operation::StopCover,
        ("set_cover_position", 1) => Operation::SetCoverPosition {
            position: byte("position")?,
        },
        ("set_temperature", 1) => {
            let temperature: f64 = param("temperature")?.parse().map_err(|_| bad())?;
            if !temperature.is_finite() {
                return Err(bad());
            }
            Operation::SetTemperature { temperature }
        }
        ("set_hvac_mode", 1) => Operation::SetHvacMode {
            hvac_mode: serde_json::from_value(json!(param("hvac_mode")?)).map_err(|_| bad())?,
        },
        ("brightness", 1) => Operation::Brightness {
            brightness: brightness()?,
        },
        ("color_temp_kelvin", 1 | 2) => Operation::ColorTempKelvin {
            color_temp_kelvin: param("color_temp_kelvin")?.parse().map_err(|_| bad())?,
            brightness: if params.len() == 2 {
                Some(brightness()?)
            } else {
                None
            },
        },
        _ => return Err(bad()),
    };
    Ok(op)
}

pub(crate) fn scheduled_action(action: &HomeAction) -> Result<EffectIntent, Status> {
    let service = if let Some((domain, operation)) = action.service.split_once('.') {
        if action
            .entity_id
            .split_once('.')
            .map(|(entity_domain, _)| entity_domain)
            != Some(domain)
        {
            return Err(Status::invalid_argument(
                "scheduled service does not match the entity domain",
            ));
        }
        operation
    } else {
        &action.service
    };
    let params = action
        .data
        .iter()
        .map(|(key, value)| {
            let value = match value {
                Value::String(text) => text.clone(),
                Value::Number(number) => number.to_string(),
                _ => {
                    return Err(Status::invalid_argument(
                        "invalid scheduled action parameters",
                    ));
                }
            };
            Ok((key.clone(), value))
        })
        .collect::<Result<HashMap<_, _>, _>>()?;
    Ok(EffectIntent {
        entity_id: action.entity_id.clone(),
        operation: typed_operation(service, &params)?,
    })
}

fn receipt(value: &nexa_ha::ActionReceipt, completed: u64) -> HomeActionReceipt {
    let (status, detail) = match value.status {
        ActionStatus::NotSent(reason) => ("not_sent", format!("{reason:?}")),
        ActionStatus::Accepted => ("accepted", String::new()),
        ActionStatus::Rejected(code) => ("rejected", code.to_string()),
        ActionStatus::Uncertain => ("uncertain", String::new()),
        ActionStatus::Partial => ("partial", String::new()),
    };
    HomeActionReceipt {
        request_id: value.request_id.clone(),
        entity_id: value.entity_id.clone(),
        operation: value.operation.into(),
        status: status.into(),
        detail,
        completed_at_unix_ms: completed,
    }
}

impl Brain {
    pub(crate) async fn execute_home(
        &self,
        request_id: String,
        action: EffectIntent,
    ) -> Result<HomeActionReceipt, Status> {
        self.allow_home(&action)?;
        let deps = self
            .home
            .deps
            .as_ref()
            .ok_or_else(|| Status::unavailable("Home Assistant unavailable"))?;
        let received = deps
            .control
            .execute(ActionRequest { request_id, action })
            .await
            .map_err(|error| match error {
                nexa_ha::ControlError::InvalidRequestId => {
                    Status::invalid_argument("invalid action request ID")
                }
                nexa_ha::ControlError::IdempotencyConflict => {
                    Status::already_exists("action request ID already used")
                }
                nexa_ha::ControlError::ExecutionInterrupted => {
                    Status::unavailable("action outcome unavailable")
                }
            })?;
        let mut history = self
            .home
            .receipts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(existing) = history
            .iter()
            .find(|entry| entry.request_id == received.request_id)
        {
            return Ok(existing.clone());
        }
        let result = receipt(&received, self.now());
        if history.len() == 128 {
            history.pop_front();
        }
        history.push_back(result.clone());
        Ok(result)
    }

    pub(crate) fn allow_home(&self, action: &EffectIntent) -> Result<(), Status> {
        authorize_action(
            &self.config.ha_controls,
            &action.entity_id,
            action.operation.name(),
        )
        .map_err(|_| {
            Status::permission_denied("home action is outside the reviewed control policy")
        })
    }

    pub(crate) fn home_receipt(&self, id: &str) -> Option<HomeActionReceipt> {
        if let Some(value) = self
            .home
            .history(128)
            .into_iter()
            .find(|entry| entry.request_id == id)
        {
            return Some(value);
        }
        self.home
            .deps
            .as_ref()?
            .control
            .get_receipt(id)
            .map(|value| receipt(&value, 0))
    }

    pub(crate) fn catalog(&self) -> HomeCatalogSnapshot {
        let frame = self.home.deps.as_ref().map(|home| home.client.frame());
        let instant = Instant::now();
        let now_ms = self.now();
        build_catalog_from_frame(
            &self.config.rooms,
            &self.config.ha_controls,
            frame.as_deref(),
            instant,
            now_ms,
        )
    }

    pub(crate) async fn run_home_events(&self) {
        let Some(home) = &self.home.deps else { return };
        let mut tick = tokio::time::interval(std::time::Duration::from_millis(500));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut previous = HashMap::new();
        let mut sync = std::pin::pin!(home.client.run_state_sync());
        loop {
            tokio::select! {
                _ = self.shutdown.cancelled() => return,
                result = &mut sync => {
                    if result.is_err() { tracing::warn!("Home Assistant state synchronization stopped"); }
                    return;
                }
                _ = tick.tick() => {
                    for entity in self.catalog().entities {
                        let current = (entity.state.clone(), entity.available);
                        if previous.get(&entity.entity_id) != Some(&current) {
                            previous.insert(entity.entity_id.clone(), current);
                            self.event(LiveEventKind::HaState, &entity.room_id, "", json!({
                                "entity_id": entity.entity_id, "state": entity.state, "available": entity.available
                            }));
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn resolve_voice_action(
        &self,
        command: &grammar::HomeCommand,
        source_room: &str,
    ) -> Option<EffectIntent> {
        use grammar::{Domain as D, Operation as O};
        let slots = &command.slots;
        let operation = match command.operation {
            O::On if command.domain == D::Light && slots.rgb_color.is_some() => return None,
            O::On if command.domain == D::Light && slots.color_temp_kelvin.is_some() => {
                Operation::ColorTempKelvin {
                    color_temp_kelvin: slots.color_temp_kelvin?.into(),
                    brightness: slots.brightness_pct.map(Brightness::Percent),
                }
            }
            O::On if command.domain == D::Light && slots.brightness_pct.is_some() => {
                Operation::Brightness {
                    brightness: Brightness::Percent(slots.brightness_pct?),
                }
            }
            O::On => Operation::TurnOn,
            O::Off => Operation::TurnOff,
            O::SetLight if slots.rgb_color.is_some() => return None,
            O::SetLight if slots.color_temp_kelvin.is_some() => Operation::ColorTempKelvin {
                color_temp_kelvin: slots.color_temp_kelvin?.into(),
                brightness: slots.brightness_pct.map(Brightness::Percent),
            },
            O::SetLight => Operation::Brightness {
                brightness: Brightness::Percent(slots.brightness_pct?),
            },
            O::SetTemperature => Operation::SetTemperature {
                temperature: slots.temperature?,
            },
            O::SetHvacMode => Operation::SetHvacMode {
                hvac_mode: match slots.hvac_mode? {
                    grammar::HvacMode::Off => HvacMode::Off,
                    grammar::HvacMode::Cool => HvacMode::Cool,
                    grammar::HvacMode::Heat => HvacMode::Heat,
                    grammar::HvacMode::Dry => HvacMode::Dry,
                    grammar::HvacMode::FanOnly => HvacMode::FanOnly,
                    grammar::HvacMode::Auto => HvacMode::Auto,
                },
            },
            O::OpenCover => Operation::OpenCover,
            O::CloseCover => Operation::CloseCover,
            O::StopCover => Operation::StopCover,
            O::SetCoverPosition => Operation::SetCoverPosition {
                position: slots.position?,
            },
            O::ActivateScene => Operation::ActivateScene,
            O::MediaPlay | O::MediaPause | O::MediaNext => return None,
        };
        let prefix = match command.domain {
            D::Light => "light",
            D::Climate => "climate",
            D::Cover => "cover",
            D::Scene => "scene",
            D::Fan | D::MediaPlayer => return None,
        };
        let room = if command.domain == D::Scene {
            source_room
        } else {
            command.room.as_deref()?
        };
        let entity = self
            .config
            .rooms
            .rooms
            .iter()
            .find(|candidate| candidate.room_id == room || words(&candidate.name) == words(room))
            .and_then(|candidate| match command.domain {
                D::Light => candidate.default_light.as_ref(),
                D::Climate => candidate.default_climate.as_ref(),
                _ => None,
            });
        let target = words(&command.target);
        let target_id = format!("{prefix}.{}", target.replace(' ', "_"));
        let slug = words(room).replace(' ', "_");
        let room_id = format!("{prefix}.{slug}");
        let room_target = target.is_empty()
            || [
                "de la ", "de el ", "del ", "de ", "en la ", "en el ", "en ", "a la ", "al ",
            ]
            .iter()
            .find_map(|prefix| target.strip_prefix(prefix))
            .or(Some(target.as_str()))
            .is_some_and(|name| grammar::canonical_room(name) == Some(room) || name == words(room));
        let entity_id = if command.domain == D::Scene || !room_target {
            target_id
        } else if let Some(configured) = entity {
            configured.clone()
        } else if command.domain == D::Light {
            let group_id = format!("light.grupo_{slug}");
            if self.config.ha_controls.entities.contains_key(&group_id) {
                group_id
            } else {
                room_id
            }
        } else if command.domain == D::Climate {
            let climate_id = format!("climate.aire_{slug}");
            if self.config.ha_controls.entities.contains_key(&climate_id) {
                climate_id
            } else {
                room_id
            }
        } else {
            room_id
        };
        Some(EffectIntent {
            entity_id,
            operation,
        })
    }

    pub(crate) async fn weather(&self, horizon: WeatherHorizon) -> String {
        if let Some(text) = world::weather_refusal(horizon) {
            return text.into();
        }
        let Some(home) = &self.home.deps else {
            return world::describe_current(None);
        };
        let entity = &self.config.core.home_assistant.weather_entity;
        let frame = home.client.frame();
        if horizon == WeatherHorizon::Current {
            return world::describe_current(frame.entity(entity, Instant::now()).ok().map(
                |state| {
                    CurrentWeather {
                        condition: Some(state.state()),
                        temperature: state
                            .attributes()
                            .get("temperature")
                            .and_then(Value::as_f64),
                        humidity: state.attributes().get("humidity").and_then(Value::as_f64),
                    }
                },
            ));
        }
        let WeatherHorizon::Forecast(horizon) = horizon else {
            return world::describe_current(None);
        };
        let entries = home.forecast.daily(entity).await;
        let today = self.local_now().date_naive();
        world::describe_forecast(
            &entries.iter().map(ForecastRow::borrow).collect::<Vec<_>>(),
            &horizon.spoken(today),
            horizon.date(today),
            &self.timezone,
        )
        .text
    }

    pub(crate) fn local_now(&self) -> chrono::DateTime<chrono_tz::Tz> {
        chrono::DateTime::from_timestamp_millis(self.clock.now())
            .unwrap_or_default()
            .with_timezone(&self.timezone)
    }
}

#[derive(Deserialize)]
struct ForecastRow {
    datetime: Option<String>,
    condition: Option<String>,
    templow: Option<f64>,
    temperature: Option<f64>,
}

impl ForecastRow {
    fn borrow(&self) -> ForecastEntry<'_> {
        ForecastEntry {
            datetime: self.datetime.as_deref(),
            condition: self.condition.as_deref(),
            low: self.templow,
            high: self.temperature,
        }
    }
}

impl crate::ForecastReader {
    async fn daily(&self, entity: &str) -> Vec<ForecastRow> {
        if !entity.starts_with("weather.") {
            return Vec::new();
        }
        let result = async {
            let mut response = self
                .client
                .post(self.url.clone())
                .header(reqwest::header::AUTHORIZATION, self.token.clone())
                .json(&json!({"entity_id": entity, "type": "daily"}))
                .send()
                .await?
                .error_for_status()?;
            let mut body = Vec::new();
            while let Some(chunk) = response.chunk().await? {
                if body.len() + chunk.len() > 128 * 1024 {
                    return Ok::<_, reqwest::Error>(Vec::new());
                }
                body.extend_from_slice(&chunk);
            }
            Ok(body)
        }
        .await;
        let Ok(bytes) = result else { return Vec::new() };
        serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|reply| {
                serde_json::from_value(reply["service_response"][entity]["forecast"].clone()).ok()
            })
            .unwrap_or_default()
    }
}

struct RoomBuilder {
    room_id: String,
    name: String,
    entity_ids: Vec<String>,
    default_light: String,
    default_climate: String,
    aliases: Vec<String>,
}

const EXTRA_ROOMS: &[(&str, &str, &[&str])] = &[
    (
        "dormitorio",
        "Dormitorio",
        &[
            "dormitorio",
            "cuarto",
            "habitacion",
            "pieza",
            "bedroom",
            "velador",
        ],
    ),
    ("bano", "Baño", &["bano", "banio", "bathroom", "ducha", "toilet"]),
    ("balcon", "Balcón", &["balcon", "terraza", "exterior"]),
    ("lavadero", "Lavadero", &["lavadero", "lavanderia", "laundry"]),
    ("escalera", "Escalera", &["escalera", "escaleras", "stairs"]),
    (
        "pasillo",
        "Pasillo",
        &["pasillo", "corredor", "hall", "recibidor"],
    ),
];

fn configured_room_aliases(room_id: &str, name: &str) -> Vec<String> {
    let mut aliases = Vec::new();
    for raw in [room_id, name] {
        let normalized = words(raw).replace('_', " ");
        for part in normalized.split_whitespace() {
            if !part.is_empty() && !aliases.iter().any(|existing| existing == part) {
                aliases.push(part.to_owned());
            }
        }
    }
    let extra: &[&str] = match room_id {
        "living" => &["living", "sala", "comedor", "estar", "tv"],
        "cocina" => &["cocina", "kitchen", "mesada", "isla", "desayunador"],
        "escritorio" => &["escritorio", "estudio", "oficina", "office", "pc"],
        "hall" => &["hall", "pasillo", "recibidor", "entrada"],
        "dormitorio" | "cuarto" => &["dormitorio", "cuarto", "habitacion", "pieza", "velador"],
        "bano" => &["bano", "banio", "ducha", "bathroom"],
        "balcon" => &["balcon", "terraza", "exterior"],
        "lavadero" => &["lavadero", "lavanderia"],
        "escalera" => &["escalera", "escaleras"],
        _ => &[],
    };
    for alias in extra {
        if !aliases.iter().any(|existing| existing == alias) {
            aliases.push((*alias).to_owned());
        }
    }
    aliases
}

fn entity_tokens(entity_id: &str, friendly_name: &str) -> Vec<String> {
    let object = entity_id
        .split_once('.')
        .map_or(entity_id, |(_, obj)| obj);
    let combined = format!("{object} {friendly_name}");
    words(&combined)
        .replace('_', " ")
        .split_whitespace()
        .map(str::to_owned)
        .collect()
}

fn matches_aliases(tokens: &[String], aliases: &[&str]) -> bool {
    tokens
        .iter()
        .any(|token| aliases.contains(&token.as_str()))
}

fn as_u32_rounded(value: &Value) -> Option<u32> {
    value
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .or_else(|| {
            value
                .as_f64()
                .filter(|v| v.is_finite() && *v >= 0.0)
                .map(|v| v.round() as u32)
        })
}

fn mireds_to_kelvin(value: &Value) -> Option<u32> {
    value
        .as_f64()
        .filter(|m| m.is_finite() && *m > 0.0)
        .map(|m| (1_000_000.0 / m).round() as u32)
}

struct EntityAttrs {
    brightness_pct: Option<u32>,
    color_temp_kelvin: Option<u32>,
    color_temp_min: Option<u32>,
    color_temp_max: Option<u32>,
    temperature: Option<f64>,
    current_temperature: Option<f64>,
    current_humidity: Option<f64>,
    hvac_action: String,
    hvac_modes: Vec<String>,
    temperature_min: Option<f64>,
    temperature_max: Option<f64>,
    temperature_step: Option<f64>,
    position: Option<u32>,
}

fn extract_entity_attrs(
    domain: &str,
    capabilities: &[String],
    state: Option<&nexa_ha::EntityState>,
) -> EntityAttrs {
    let Some(attrs) = state.map(nexa_ha::EntityState::attributes) else {
        return EntityAttrs {
            brightness_pct: None,
            color_temp_kelvin: None,
            color_temp_min: capabilities
                .iter()
                .any(|c| c == "color_temp_kelvin")
                .then_some(2200),
            color_temp_max: capabilities
                .iter()
                .any(|c| c == "color_temp_kelvin")
                .then_some(6500),
            temperature: None,
            current_temperature: None,
            current_humidity: None,
            hvac_action: String::new(),
            hvac_modes: Vec::new(),
            temperature_min: None,
            temperature_max: None,
            temperature_step: None,
            position: None,
        };
    };

    let brightness_pct = attrs
        .get("brightness_pct")
        .and_then(as_u32_rounded)
        .map(|v| v.min(100))
        .or_else(|| {
            attrs.get("brightness").and_then(Value::as_f64).map(|b| {
                ((b * 100.0) / 255.0).round().clamp(0.0, 100.0) as u32
            })
        });

    let color_temp_kelvin = attrs
        .get("color_temp_kelvin")
        .and_then(as_u32_rounded)
        .or_else(|| attrs.get("color_temp").and_then(mireds_to_kelvin));

    let mut color_temp_min = attrs
        .get("min_color_temp_kelvin")
        .and_then(as_u32_rounded)
        .or_else(|| attrs.get("max_mireds").and_then(mireds_to_kelvin));
    let mut color_temp_max = attrs
        .get("max_color_temp_kelvin")
        .and_then(as_u32_rounded)
        .or_else(|| attrs.get("min_mireds").and_then(mireds_to_kelvin));
    if capabilities.iter().any(|c| c == "color_temp_kelvin") {
        color_temp_min.get_or_insert(2200);
        color_temp_max.get_or_insert(6500);
    }

    let temperature = attrs.get("temperature").and_then(Value::as_f64);
    let current_temperature = attrs.get("current_temperature").and_then(Value::as_f64);
    let current_humidity = attrs.get("current_humidity").and_then(Value::as_f64);
    let hvac_action = attrs
        .get("hvac_action")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let hvac_modes = attrs
        .get("hvac_modes")
        .and_then(Value::as_array)
        .map(|modes| {
            modes
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let temperature_min = attrs.get("min_temp").and_then(Value::as_f64);
    let temperature_max = attrs.get("max_temp").and_then(Value::as_f64);
    let temperature_step = attrs
        .get("target_temp_step")
        .and_then(Value::as_f64)
        .filter(|step| step.is_finite() && *step > 0.0)
        .or_else(|| {
            if domain == "climate" && temperature_min.is_some() && temperature_max.is_some() {
                Some(1.0)
            } else {
                None
            }
        });
    let position = attrs
        .get("current_position")
        .and_then(as_u32_rounded)
        .map(|v| v.min(100));

    EntityAttrs {
        brightness_pct,
        color_temp_kelvin,
        color_temp_min,
        color_temp_max,
        temperature,
        current_temperature,
        current_humidity,
        hvac_action,
        hvac_modes,
        temperature_min,
        temperature_max,
        temperature_step,
        position,
    }
}

pub(crate) fn build_catalog_from_frame(
    rooms_cfg: &nexa_config::RoomsConfig,
    controls: &nexa_config::HaControlPolicyConfig,
    frame: Option<&nexa_ha::StateFrame>,
    instant: Instant,
    now_ms: u64,
) -> HomeCatalogSnapshot {
    let mut room_map: BTreeMap<String, usize> = BTreeMap::new();
    let mut builders: Vec<RoomBuilder> = Vec::new();
    let mut entries: BTreeMap<String, String> = BTreeMap::new();

    for room in &rooms_cfg.rooms {
        let mut ids = Vec::new();
        for id in room.default_light.iter().chain(room.default_climate.iter()) {
            entries.insert(id.clone(), room.room_id.clone());
            if !ids.contains(id) {
                ids.push(id.clone());
            }
        }
        let idx = builders.len();
        room_map.insert(room.room_id.clone(), idx);
        builders.push(RoomBuilder {
            room_id: room.room_id.clone(),
            name: room.name.clone(),
            entity_ids: ids,
            default_light: room.default_light.clone().unwrap_or_default(),
            default_climate: room.default_climate.clone().unwrap_or_default(),
            aliases: configured_room_aliases(&room.room_id, &room.name),
        });
    }

    for id in controls.entities.keys() {
        entries.entry(id.clone()).or_default();
    }

    let frame_fresh = frame.is_some_and(|f| f.snapshot_is_fresh(instant));
    let mut observed_count = 0usize;
    let mut max_reported_at = 0u64;

    let mut entities: Vec<HomeEntity> = entries
        .into_iter()
        .map(|(id, mut room_id)| {
            let state = frame.and_then(|f| f.entity(&id, instant).ok());
            if state.is_some() {
                observed_count += 1;
            }
            let at = state
                .and_then(nexa_ha::EntityState::reported_at)
                .map_or(0, |time| unsigned(time.timestamp_millis()));
            max_reported_at = max_reported_at.max(at);

            let domain = id.split_once('.').map_or("", |(domain, _)| domain).to_owned();
            let friendly_name = state
                .and_then(|s| s.attributes().get("friendly_name"))
                .and_then(Value::as_str)
                .unwrap_or(&id)
                .to_owned();
            let capabilities = controls.entities.get(&id).cloned().unwrap_or_default();

            if room_id.is_empty() {
                let tokens = entity_tokens(&id, &friendly_name);
                if let Some(configured) = builders.iter_mut().find(|room| {
                    let alias_refs: Vec<&str> = room.aliases.iter().map(String::as_str).collect();
                    matches_aliases(&tokens, &alias_refs)
                }) {
                    room_id = configured.room_id.clone();
                    if !configured.entity_ids.contains(&id) {
                        configured.entity_ids.push(id.clone());
                    }
                    if configured.default_light.is_empty() && domain == "light" {
                        configured.default_light = id.clone();
                    } else if configured.default_climate.is_empty() && domain == "climate" {
                        configured.default_climate = id.clone();
                    }
                } else if let Some(&(extra_id, extra_name, extra_aliases)) = EXTRA_ROOMS
                    .iter()
                    .find(|&&(_, _, aliases)| matches_aliases(&tokens, aliases))
                {
                    room_id = extra_id.to_owned();
                    let idx = if let Some(&idx) = room_map.get(extra_id) {
                        idx
                    } else {
                        let idx = builders.len();
                        room_map.insert(extra_id.to_owned(), idx);
                        builders.push(RoomBuilder {
                            room_id: extra_id.to_owned(),
                            name: extra_name.to_owned(),
                            entity_ids: Vec::new(),
                            default_light: String::new(),
                            default_climate: String::new(),
                            aliases: extra_aliases.iter().map(|s| (*s).to_owned()).collect(),
                        });
                        idx
                    };
                    let target = &mut builders[idx];
                    if !target.entity_ids.contains(&id) {
                        target.entity_ids.push(id.clone());
                    }
                    if target.default_light.is_empty() && domain == "light" {
                        target.default_light = id.clone();
                    } else if target.default_climate.is_empty() && domain == "climate" {
                        target.default_climate = id.clone();
                    }
                }
            }

            let available = if domain == "scene" {
                state.is_some_and(|s| s.state() != "unavailable")
            } else {
                state.is_some_and(|s| !matches!(s.state(), "unavailable" | "unknown"))
            };
            let attrs = extract_entity_attrs(&domain, &capabilities, state);

            HomeEntity {
                entity_id: id,
                domain,
                room_id,
                friendly_name,
                state: state.map_or("unknown", nexa_ha::EntityState::state).into(),
                available,
                capabilities,
                observed_at_unix_ms: if frame_fresh && state.is_some() {
                    now_ms
                } else {
                    at
                },
                brightness_pct: attrs.brightness_pct,
                color_temp_kelvin: attrs.color_temp_kelvin,
                color_temp_min: attrs.color_temp_min,
                color_temp_max: attrs.color_temp_max,
                temperature: attrs.temperature,
                current_temperature: attrs.current_temperature,
                current_humidity: attrs.current_humidity,
                hvac_action: attrs.hvac_action,
                hvac_modes: attrs.hvac_modes,
                temperature_min: attrs.temperature_min,
                temperature_max: attrs.temperature_max,
                temperature_step: attrs.temperature_step,
                position: attrs.position,
            }
        })
        .collect();

    // Prefer `light.grupo_*` as room default_light if a room only had a leaf default or empty default.
    let entity_index: HashMap<String, usize> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| (e.entity_id.clone(), i))
        .collect();

    let rooms = builders
        .into_iter()
        .map(|room| {
            let mut current_temperature = None;
            let mut current_humidity = None;
            let ordered_ids: Vec<&str> = (!room.default_climate.is_empty())
                .then_some(room.default_climate.as_str())
                .into_iter()
                .chain(room.entity_ids.iter().map(String::as_str))
                .collect();
            for id in ordered_ids {
                if let Some(entity) = entity_index.get(id).map(|&i| &entities[i]) {
                    if current_temperature.is_none() {
                        current_temperature = entity.current_temperature;
                    }
                    if current_humidity.is_none() {
                        current_humidity = entity.current_humidity;
                    }
                }
            }
            HomeRoom {
                room_id: room.room_id,
                name: room.name,
                entity_ids: room.entity_ids,
                default_light: room.default_light,
                default_climate: room.default_climate,
                current_temperature,
                current_humidity,
            }
        })
        .collect();

    let stale = entities.is_empty() || !frame_fresh || observed_count == 0;
    let catalog_observed_at = if !stale { now_ms } else { max_reported_at };
    // Sort entities cleanly inside the response for UI predictability.
    entities.sort_by(|a, b| a.entity_id.cmp(&b.entity_id));

    HomeCatalogSnapshot {
        observed_at_unix_ms: catalog_observed_at,
        stale,
        rooms,
        entities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexa_config::{HaControlPolicyConfig, RoomDefinition, RoomsConfig, XvfTuning};
    use nexa_ha::StateFrame;
    use std::time::Duration;

    fn tuning() -> XvfTuning {
        XvfTuning {
            pp_agc_max_gain: 1.0,
            pp_attns_mode: 0,
            pp_attns_slope: 0.5,
            pp_min_nn: 0.51,
        }
    }

    fn room(room_id: &str, name: &str, default_light: Option<&str>, default_climate: Option<&str>) -> RoomDefinition {
        RoomDefinition {
            room_id: room_id.into(),
            name: name.into(),
            mic_usb_port: None,
            mic_serial: None,
            mic_channel: 0,
            ma1260_zone: 1,
            default_light: default_light.map(str::to_owned),
            default_climate: default_climate.map(str::to_owned),
            xvf_tuning: tuning(),
        }
    }

    #[test]
    fn catalog_reports_fresh_snapshot_attributes_scenes_and_extra_rooms() {
        let rooms_cfg = RoomsConfig {
            rooms: vec![
                room("living", "Living", Some("light.living_techo"), Some("climate.aire_living")),
                room("cocina", "Cocina", Some("light.cocina"), None),
            ],
        };
        let mut entities = HashMap::new();
        entities.insert("light.living_techo".into(), vec!["turn_on".into(), "turn_off".into(), "brightness".into(), "color_temp_kelvin".into()]);
        entities.insert("climate.aire_living".into(), vec!["turn_on".into(), "turn_off".into(), "set_temperature".into(), "set_hvac_mode".into()]);
        entities.insert("light.cocina".into(), vec!["turn_on".into(), "turn_off".into()]);
        entities.insert("light.dormitorio_velador".into(), vec!["turn_on".into(), "turn_off".into()]);
        entities.insert("switch.bano_extractor".into(), vec!["turn_on".into(), "turn_off".into()]);
        entities.insert("light.balcon".into(), vec!["turn_on".into(), "turn_off".into()]);
        entities.insert("scene.living_cine".into(), vec!["activate_scene".into()]);
        entities.insert("light.offline_optional".into(), vec!["turn_on".into()]);
        let controls = HaControlPolicyConfig {
            entities,
            ..Default::default()
        };

        let now = Instant::now();
        let wall_now = 1_800_000_000_000_u64;
        // Old last_updated timestamp from hours ago MUST NOT make fresh snapshot stale.
        let stale_reported = "2026-11-26T00:00:00Z";
        let frame = StateFrame::test_snapshot(
            Duration::from_secs(30),
            now,
            &[
                json!({
                    "entity_id": "light.living_techo",
                    "state": "on",
                    "last_reported": stale_reported,
                    "attributes": {
                        "friendly_name": "Luz Living",
                        "brightness": 128,
                        "color_temp_kelvin": 3000,
                        "min_color_temp_kelvin": 2000,
                        "max_color_temp_kelvin": 6500
                    }
                }),
                json!({
                    "entity_id": "climate.aire_living",
                    "state": "cool",
                    "last_reported": stale_reported,
                    "attributes": {
                        "friendly_name": "Aire Living",
                        "temperature": 23.0,
                        "current_temperature": 25.5,
                        "current_humidity": 48.0,
                        "hvac_action": "cooling",
                        "hvac_modes": ["off", "cool", "heat"],
                        "min_temp": 16.0,
                        "max_temp": 30.0,
                        "target_temp_step": 1.0
                    }
                }),
                json!({
                    "entity_id": "light.cocina",
                    "state": "off",
                    "attributes": { "friendly_name": "Cocina" }
                }),
                json!({
                    "entity_id": "light.dormitorio_velador",
                    "state": "on",
                    "attributes": { "friendly_name": "Velador Cuarto", "brightness_pct": 60 }
                }),
                json!({
                    "entity_id": "switch.bano_extractor",
                    "state": "off",
                    "attributes": { "friendly_name": "Extractor Baño" }
                }),
                json!({
                    "entity_id": "light.balcon",
                    "state": "off",
                    "attributes": { "friendly_name": "Balcón" }
                }),
                json!({
                    "entity_id": "scene.living_cine",
                    "state": "unknown",
                    "attributes": { "friendly_name": "Living Cine" }
                }),
            ],
        );

        let snap = build_catalog_from_frame(&rooms_cfg, &controls, Some(&frame), now, wall_now);
        assert!(!snap.stale);
        assert_eq!(snap.observed_at_unix_ms, wall_now);

        let room_ids: Vec<_> = snap.rooms.iter().map(|r| r.room_id.as_str()).collect();
        assert!(room_ids.contains(&"living"));
        assert!(room_ids.contains(&"cocina"));
        assert!(room_ids.contains(&"dormitorio"));
        assert!(room_ids.contains(&"bano"));
        assert!(room_ids.contains(&"balcon"));

        let living = snap.rooms.iter().find(|r| r.room_id == "living").unwrap();
        assert_eq!(living.current_temperature, Some(25.5));
        assert_eq!(living.current_humidity, Some(48.0));
        assert!(living.entity_ids.contains(&"scene.living_cine".into()));

        let light = snap.entities.iter().find(|e| e.entity_id == "light.living_techo").unwrap();
        assert_eq!(light.brightness_pct, Some(50));
        assert_eq!(light.color_temp_kelvin, Some(3000));
        assert_eq!(light.color_temp_min, Some(2000));
        assert_eq!(light.color_temp_max, Some(6500));
        assert!(light.available);

        let scene = snap.entities.iter().find(|e| e.entity_id == "scene.living_cine").unwrap();
        assert!(scene.available);

        let missing = snap.entities.iter().find(|e| e.entity_id == "light.offline_optional").unwrap();
        assert!(!missing.available);
    }
}
