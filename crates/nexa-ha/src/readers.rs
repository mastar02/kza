//! Pure projections over checked observations. Callers select the entities they may read.

use std::{
    fmt,
    time::{Duration, SystemTime},
};

use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, Utc};
use serde_json::Value;
use thiserror::Error;
use tokio::time::Instant;

use crate::state::{EntityState, StateFrame, StateReadError, domain};

#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
pub enum ReadError {
    #[error(transparent)]
    Observation(#[from] StateReadError),
    #[error("entity belongs to a different domain")]
    WrongDomain,
    #[error("Home Assistant marks the entity unavailable")]
    Unavailable,
    #[error("Home Assistant reports an unknown value")]
    Unknown,
    #[error("observed attributes are incompatible or invalid")]
    InvalidData,
    #[error("the sensor reading itself has expired")]
    StaleMeasurement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyKind {
    Energy,
    Power,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyReading {
    pub value: f64,
    pub unit: &'static str,
    pub reported_at: DateTime<FixedOffset>,
}

/// Normalize energy to kWh and power to W. A fresh snapshot of an old
/// measurement does not make that measurement current.
pub fn energy(
    frame: &StateFrame,
    entity_id: &str,
    kind: EnergyKind,
    now: Instant,
    wall_time: SystemTime,
    measurement_max_age: Duration,
) -> Result<EnergyReading, ReadError> {
    let entity = checked(frame, entity_id, "sensor", now)?;
    let attrs = entity.attributes();
    let class = attrs.get("device_class").and_then(Value::as_str);
    let unit = attrs.get("unit_of_measurement").and_then(Value::as_str);
    let (unit, scale) = match (kind, class, unit) {
        (EnergyKind::Energy, Some("energy"), Some("Wh")) => ("kWh", 0.001),
        (EnergyKind::Energy, Some("energy"), Some("kWh")) => ("kWh", 1.0),
        (EnergyKind::Energy, Some("energy"), Some("MWh")) => ("kWh", 1000.0),
        (EnergyKind::Power, Some("power"), Some("W")) => ("W", 1.0),
        (EnergyKind::Power, Some("power"), Some("kW")) => ("W", 1000.0),
        _ => return Err(ReadError::InvalidData),
    };
    let value = entity
        .state()
        .parse::<f64>()
        .map_err(|_| ReadError::InvalidData)?
        * scale;
    if !value.is_finite() {
        return Err(ReadError::InvalidData);
    }
    let reported_at = entity.reported_at().ok_or(ReadError::StaleMeasurement)?;
    let wall_time = DateTime::<Utc>::from(wall_time);
    let current = wall_time
        .signed_duration_since(reported_at)
        .to_std()
        .is_ok_and(|age| age < measurement_max_age);
    if !current {
        return Err(ReadError::StaleMeasurement);
    }
    Ok(EnergyReading {
        value,
        unit,
        reported_at,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CalendarTime {
    Date(NaiveDate),
    Local(NaiveDateTime),
    Zoned(DateTime<FixedOffset>),
}

#[derive(Clone, PartialEq, Eq)]
pub struct CalendarEvent {
    pub summary: String,
    pub start: CalendarTime,
    pub end: CalendarTime,
}

impl fmt::Debug for CalendarEvent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CalendarEvent")
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalendarReading {
    pub active: bool,
    pub current_or_next: Option<CalendarEvent>,
}

/// Read the current or next event advertised by a selected HA calendar entity.
/// Local timestamps are preserved without inferring a host timezone.
pub fn calendar(
    frame: &StateFrame,
    entity_id: &str,
    now: Instant,
) -> Result<CalendarReading, ReadError> {
    let entity = checked(frame, entity_id, "calendar", now)?;
    let active = match entity.state() {
        "on" => true,
        "off" => false,
        _ => return Err(ReadError::InvalidData),
    };
    let attr = |name| {
        entity
            .attributes()
            .get(name)
            .filter(|value| !value.is_null())
    };
    let current_or_next = match (attr("message"), attr("start_time"), attr("end_time")) {
        (None, None, None) if !active => None,
        (Some(summary), Some(start), Some(end)) => {
            let start = parse_calendar_time(start).ok_or(ReadError::InvalidData)?;
            let end = parse_calendar_time(end).ok_or(ReadError::InvalidData)?;
            if std::mem::discriminant(&start) != std::mem::discriminant(&end) || start >= end {
                return Err(ReadError::InvalidData);
            }
            Some(CalendarEvent {
                summary: summary.as_str().ok_or(ReadError::InvalidData)?.to_owned(),
                start,
                end,
            })
        }
        _ => return Err(ReadError::InvalidData),
    };
    Ok(CalendarReading {
        active,
        current_or_next,
    })
}

fn parse_calendar_time(value: &Value) -> Option<CalendarTime> {
    let value = value.as_str()?;
    if let Ok(zoned) = DateTime::parse_from_rfc3339(value) {
        return Some(CalendarTime::Zoned(zoned));
    }
    for format in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"] {
        if let Ok(local) = NaiveDateTime::parse_from_str(value, format) {
            return Some(CalendarTime::Local(local));
        }
    }
    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .ok()
        .map(CalendarTime::Date)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaState {
    Playing,
    Paused,
    Idle,
    Off,
    Standby,
    On,
    Buffering,
}

#[derive(Clone, PartialEq)]
pub struct MediaReading {
    pub state: MediaState,
    pub supported_features: u64,
    pub volume: Option<f64>,
    pub muted: Option<bool>,
    pub title: Option<String>,
    pub app: Option<String>,
}

impl fmt::Debug for MediaReading {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MediaReading")
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

pub fn media(frame: &StateFrame, entity_id: &str, now: Instant) -> Result<MediaReading, ReadError> {
    let entity = checked(frame, entity_id, "media_player", now)?;
    let state = match entity.state() {
        "playing" => MediaState::Playing,
        "paused" => MediaState::Paused,
        "idle" => MediaState::Idle,
        "off" => MediaState::Off,
        "standby" => MediaState::Standby,
        "on" => MediaState::On,
        "buffering" => MediaState::Buffering,
        _ => return Err(ReadError::Unknown),
    };
    let attr = |name| entity.attributes().get(name);
    let text = |name| attr(name).and_then(Value::as_str).map(str::to_owned);
    Ok(MediaReading {
        state,
        supported_features: attr("supported_features")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        volume: attr("volume_level")
            .and_then(Value::as_f64)
            .filter(|volume| (0.0..=1.0).contains(volume)),
        muted: attr("is_volume_muted").and_then(Value::as_bool),
        title: text("media_title"),
        app: text("app_name"),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraHealth {
    Online,
    Offline,
    Unknown,
}

/// Camera observations disclose availability without distinguishing streaming,
/// recording or idle activity.
pub fn camera_health(
    frame: &StateFrame,
    entity_id: &str,
    now: Instant,
) -> Result<CameraHealth, ReadError> {
    let state = selected(frame, entity_id, "camera", now)?;
    Ok(match state.state() {
        "idle" | "recording" | "streaming" => CameraHealth::Online,
        "unavailable" => CameraHealth::Offline,
        _ => CameraHealth::Unknown,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticKind {
    Connectivity,
    Power,
    Battery,
    Storage,
    DeviceError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticHealth {
    Normal,
    Issue,
    Unknown,
}

/// Project a selected technical condition without exposing the sensor payload.
pub fn camera_diagnostic(
    frame: &StateFrame,
    entity_id: &str,
    kind: DiagnosticKind,
    now: Instant,
) -> Result<DiagnosticHealth, ReadError> {
    let entity_domain = match domain(entity_id) {
        Some(domain @ ("binary_sensor" | "sensor")) => domain,
        _ => return Err(ReadError::WrongDomain),
    };
    let entity = frame.entity(entity_id, now)?;
    let device_class = entity
        .attributes()
        .get("device_class")
        .filter(|value| !value.is_null())
        .map(|value| value.as_str().ok_or(ReadError::InvalidData))
        .transpose()?;
    let binary_fault = match (kind, device_class) {
        (DiagnosticKind::Connectivity, Some("connectivity"))
        | (DiagnosticKind::Power, Some("power" | "plug")) => Some("off"),
        (DiagnosticKind::Battery, Some("battery"))
        | (
            DiagnosticKind::Power | DiagnosticKind::Storage | DiagnosticKind::DeviceError,
            Some("problem"),
        ) => Some("on"),
        _ => None,
    };
    if entity_domain == "binary_sensor" {
        let fault = binary_fault.ok_or(ReadError::InvalidData)?;
        return Ok(match entity.state() {
            state if state == fault => DiagnosticHealth::Issue,
            "on" | "off" => DiagnosticHealth::Normal,
            _ => DiagnosticHealth::Unknown,
        });
    }
    if !matches!(device_class, None | Some("enum")) && binary_fault.is_none() {
        return Err(ReadError::InvalidData);
    }
    let health = match (kind, entity.state()) {
        (DiagnosticKind::Connectivity, "connected" | "online")
        | (DiagnosticKind::Power, "powered" | "plugged_in")
        | (
            DiagnosticKind::Battery | DiagnosticKind::Storage | DiagnosticKind::DeviceError,
            "normal" | "ok",
        ) => DiagnosticHealth::Normal,
        (DiagnosticKind::Connectivity, "disconnected" | "offline")
        | (DiagnosticKind::Power, "unpowered" | "unplugged" | "power_loss")
        | (DiagnosticKind::Battery, "low" | "low_battery")
        | (DiagnosticKind::Storage, "full" | "storage_full" | "error")
        | (DiagnosticKind::DeviceError, "error" | "fault" | "problem") => DiagnosticHealth::Issue,
        _ if binary_fault.is_some() => DiagnosticHealth::Unknown,
        _ => return Err(ReadError::InvalidData),
    };
    Ok(health)
}

fn selected<'a>(
    frame: &'a StateFrame,
    entity_id: &str,
    expected_domain: &str,
    now: Instant,
) -> Result<&'a EntityState, ReadError> {
    if domain(entity_id) != Some(expected_domain) {
        return Err(ReadError::WrongDomain);
    }
    Ok(frame.entity(entity_id, now)?)
}

fn checked<'a>(
    frame: &'a StateFrame,
    entity_id: &str,
    expected_domain: &str,
    now: Instant,
) -> Result<&'a EntityState, ReadError> {
    let entity = selected(frame, entity_id, expected_domain, now)?;
    match entity.state() {
        "unknown" => Err(ReadError::Unknown),
        "unavailable" => Err(ReadError::Unavailable),
        _ => Ok(entity),
    }
}
