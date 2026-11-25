use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::state::{EntityState, domain};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Brightness {
    Percent(u8),
    Raw(u8),
}

impl Brightness {
    fn value(self) -> u8 {
        match self {
            Self::Percent(value) | Self::Raw(value) => value,
        }
    }

    fn insert(self, body: &mut Map<String, Value>) {
        let key = match self {
            Self::Percent(_) => "brightness_pct",
            Self::Raw(_) => "brightness",
        };
        body.insert(key.into(), json!(self.value()));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HvacMode {
    Off,
    Cool,
    Heat,
    HeatCool,
    Auto,
    Dry,
    FanOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    content = "parameters",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum Operation {
    TurnOn,
    TurnOff,
    ActivateScene,
    Brightness {
        brightness: Brightness,
    },
    ColorTempKelvin {
        color_temp_kelvin: u32,
        brightness: Option<Brightness>,
    },
    SetTemperature {
        temperature: f64,
    },
    SetHvacMode {
        hvac_mode: HvacMode,
    },
    OpenCover,
    CloseCover,
    StopCover,
    SetCoverPosition {
        position: u8,
    },
}

impl Operation {
    pub fn name(&self) -> &'static str {
        match self {
            Self::TurnOn => "turn_on",
            Self::TurnOff => "turn_off",
            Self::ActivateScene => "activate_scene",
            Self::Brightness { .. } => "brightness",
            Self::ColorTempKelvin { .. } => "color_temp_kelvin",
            Self::SetTemperature { .. } => "set_temperature",
            Self::SetHvacMode { .. } => "set_hvac_mode",
            Self::OpenCover => "open_cover",
            Self::CloseCover => "close_cover",
            Self::StopCover => "stop_cover",
            Self::SetCoverPosition { .. } => "set_cover_position",
        }
    }

    pub(crate) fn same_body(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::SetTemperature { temperature: left },
                Self::SetTemperature { temperature: right },
            ) => left.to_bits() == right.to_bits(),
            _ => self == other,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), OperationError> {
        let valid = match self {
            Self::Brightness {
                brightness: Brightness::Percent(value),
            } => *value <= 100,
            Self::ColorTempKelvin {
                color_temp_kelvin,
                brightness,
            } => {
                *color_temp_kelvin > 0
                    && match brightness {
                        None => true,
                        Some(Brightness::Percent(value)) => (1..=100).contains(value),
                        Some(Brightness::Raw(value)) => *value > 0,
                    }
            }
            Self::SetTemperature { temperature } => temperature.is_finite(),
            Self::SetCoverPosition { position } => *position <= 100,
            _ => true,
        };
        valid.then_some(()).ok_or(OperationError::InvalidParameters)
    }

    /// Check observed capabilities after grants and parameter shape have been verified.
    pub(crate) fn validate_leaf(
        &self,
        state: &EntityState,
        celsius: bool,
    ) -> Result<(), OperationError> {
        let attrs = state.attributes();
        if attrs
            .get("entity_id")
            .is_some_and(|members| !members.is_null())
        {
            return Err(OperationError::UnreviewedGroup);
        }
        let current = state.state();
        let state_supported = current == "unknown"
            || match domain(state.entity_id()) {
                Some("light" | "switch") => matches!(current, "on" | "off"),
                Some("climate") => matches!(
                    current,
                    "off" | "cool" | "heat" | "heat_cool" | "auto" | "dry" | "fan_only"
                ),
                Some("cover") => matches!(current, "open" | "closed" | "opening" | "closing"),
                _ => false,
            };
        if !state_supported {
            return Err(OperationError::Unsupported);
        }
        let features = match attrs.get("supported_features") {
            Some(value) => value.as_u64().ok_or(OperationError::Unsupported)?,
            None => 0,
        };
        let parsed = |name| attrs.get(name).and_then(Value::as_f64);
        let mode = |candidate| {
            attrs
                .get("supported_color_modes")
                .and_then(Value::as_array)
                .is_some_and(|modes| modes.iter().any(|value| value.as_str() == Some(candidate)))
        };
        let dimmable = || {
            [
                "brightness",
                "color_temp",
                "hs",
                "xy",
                "rgb",
                "rgbw",
                "rgbww",
                "white",
            ]
            .into_iter()
            .any(mode)
        };
        let flags = match self {
            Self::OpenCover => 1,
            Self::CloseCover => 2,
            Self::SetCoverPosition { .. } => 4,
            Self::StopCover => 8,
            _ => 0,
        };
        let valid = match self {
            Self::TurnOn | Self::TurnOff => true,
            Self::Brightness { .. } => dimmable(),
            Self::ColorTempKelvin {
                color_temp_kelvin, ..
            } => {
                mode("color_temp")
                    && matches!((parsed("min_color_temp_kelvin"), parsed("max_color_temp_kelvin")),
                        (Some(low), Some(high)) if low > 0.0 && (low..=high).contains(&f64::from(*color_temp_kelvin)))
            }
            Self::SetHvacMode { hvac_mode } => attribute::<Vec<HvacMode>>(attrs, "hvac_modes")
                .is_some_and(|modes| modes.contains(hvac_mode)),
            Self::SetTemperature { temperature } => {
                let verified_unit = celsius
                    && attrs
                        .get("temperature_unit")
                        .is_none_or(|unit| unit == "°C");
                let within_bounds = match (parsed("min_temp"), parsed("max_temp")) {
                    (Some(low), Some(high)) if (low..=high).contains(temperature) => {
                        match attrs.get("target_temp_step") {
                            Some(step) => step.as_f64().is_some_and(|step| {
                                let steps = (temperature - low) / step;
                                step > 0.0 && (steps - steps.round()).abs() <= 1e-7
                            }),
                            None => temperature.fract() == 0.0,
                        }
                    }
                    _ => false,
                };
                features & 1 != 0
                    && verified_unit
                    && within_bounds
                    && attribute::<Vec<HvacMode>>(attrs, "hvac_modes")
                        .is_some_and(|modes| !modes.is_empty())
            }
            Self::OpenCover
            | Self::CloseCover
            | Self::StopCover
            | Self::SetCoverPosition { .. } => {
                features & flags != 0
                    && matches!(
                        attrs.get("device_class").and_then(Value::as_str),
                        Some("blind" | "curtain" | "shade" | "shutter")
                    )
            }
            Self::ActivateScene => false,
        };
        valid.then_some(()).ok_or(OperationError::Unsupported)
    }

    pub(crate) fn service(&self, entity_id: &str) -> (&'static str, Map<String, Value>) {
        let mut body = Map::from_iter([("entity_id".into(), json!(entity_id))]);
        let service = match self {
            Self::Brightness { brightness } if brightness.value() == 0 => "turn_off",
            Self::Brightness { brightness } => {
                brightness.insert(&mut body);
                "turn_on"
            }
            Self::ColorTempKelvin {
                color_temp_kelvin,
                brightness,
            } => {
                body.insert("color_temp_kelvin".into(), json!(color_temp_kelvin));
                if let Some(brightness) = brightness {
                    brightness.insert(&mut body);
                }
                "turn_on"
            }
            Self::SetTemperature { temperature } => {
                body.insert("temperature".into(), json!(temperature));
                self.name()
            }
            Self::SetHvacMode { hvac_mode } => {
                body.insert("hvac_mode".into(), json!(hvac_mode));
                self.name()
            }
            Self::SetCoverPosition { position } => {
                body.insert("position".into(), json!(position));
                self.name()
            }
            other => other.name(),
        };
        (service, body)
    }
}

fn attribute<T: DeserializeOwned>(attrs: &Map<String, Value>, key: &str) -> Option<T> {
    T::deserialize(attrs.get(key)?).ok()
}

#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum OperationError {
    #[error("invalid operation parameters")]
    InvalidParameters,
    #[error("operation exceeds observed Home Assistant capabilities")]
    Unsupported,
    #[error("group effects have not been reviewed")]
    UnreviewedGroup,
}
