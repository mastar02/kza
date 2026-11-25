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
        let mut entries = BTreeMap::new();
        let rooms = self
            .config
            .rooms
            .rooms
            .iter()
            .map(|room| {
                let ids = room
                    .default_light
                    .iter()
                    .chain(room.default_climate.iter())
                    .cloned()
                    .collect::<Vec<_>>();
                for id in &ids {
                    entries.insert(id.clone(), room.room_id.clone());
                }
                HomeRoom {
                    room_id: room.room_id.clone(),
                    name: room.name.clone(),
                    entity_ids: ids,
                    default_light: room.default_light.clone().unwrap_or_default(),
                    default_climate: room.default_climate.clone().unwrap_or_default(),
                }
            })
            .collect();
        for id in self.config.ha_controls.entities.keys() {
            entries.entry(id.clone()).or_default();
        }
        let mut stale = entries.is_empty();
        let mut observed = 0;
        let entities = entries
            .into_iter()
            .map(|(id, room_id)| {
                let state = frame
                    .as_ref()
                    .and_then(|frame| frame.entity(&id, instant).ok());
                stale |= state.is_none();
                let at = state
                    .and_then(nexa_ha::EntityState::reported_at)
                    .map_or(0, |time| unsigned(time.timestamp_millis()));
                observed = observed.max(at);
                HomeEntity {
                    domain: id.split_once('.').map_or("", |(domain, _)| domain).into(),
                    capabilities: self
                        .config
                        .ha_controls
                        .entities
                        .get(&id)
                        .cloned()
                        .unwrap_or_default(),
                    entity_id: id.clone(),
                    room_id,
                    friendly_name: state
                        .and_then(|state| state.attributes().get("friendly_name"))
                        .and_then(Value::as_str)
                        .unwrap_or(&id)
                        .into(),
                    state: state.map_or("unknown", nexa_ha::EntityState::state).into(),
                    available: state
                        .is_some_and(|state| !matches!(state.state(), "unavailable" | "unknown")),
                    observed_at_unix_ms: at,
                }
            })
            .collect();
        HomeCatalogSnapshot {
            observed_at_unix_ms: observed,
            stale,
            rooms,
            entities,
        }
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
        let room_id = format!("{prefix}.{}", words(room).replace(' ', "_"));
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
        } else {
            entity.cloned().unwrap_or(room_id)
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
