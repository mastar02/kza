//! Browser JSON and request validation at the HTTP/protobuf boundary.

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use chrono::{DateTime, Utc};
use nexa_proto::v1::*;
use serde::de::{Deserialize, Deserializer, Error, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value, json};
use subtle::ConstantTimeEq;

use super::{ApiError, BAD_UPSTREAM, INVALID, Result, StatusCode};
use crate::auth::{AuthError, Lease};

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct Unique;
        impl<'de> Visitor<'de> for Unique {
            type Value = StrictValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value with unique object keys")
            }

            fn visit_map<M: MapAccess<'de>>(
                self,
                mut input: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut result = Map::new();
                while let Some((key, value)) = input.next_entry::<String, StrictValue>()? {
                    if result.insert(key, value.0).is_some() {
                        return Err(M::Error::custom("duplicate key"));
                    }
                }
                Ok(StrictValue(Value::Object(result)))
            }

            fn visit_seq<S: SeqAccess<'de>>(
                self,
                mut input: S,
            ) -> std::result::Result<Self::Value, S::Error> {
                let mut result = Vec::new();
                while let Some(value) = input.next_element::<StrictValue>()? {
                    result.push(value.0);
                }
                Ok(StrictValue(Value::Array(result)))
            }

            fn visit_unit<E: Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(Value::Null))
            }
            fn visit_bool<E: Error>(self, value: bool) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(json!(value)))
            }
            fn visit_i64<E: Error>(self, value: i64) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(json!(value)))
            }
            fn visit_u64<E: Error>(self, value: u64) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(json!(value)))
            }
            fn visit_f64<E: Error>(self, value: f64) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(json!(value)))
            }
            fn visit_str<E: Error>(self, value: &str) -> std::result::Result<Self::Value, E> {
                Ok(StrictValue(json!(value)))
            }
        }
        deserializer.deserialize_any(Unique)
    }
}

pub(super) fn parse_object(bytes: &[u8]) -> Result<Value> {
    let value = serde_json::from_slice::<StrictValue>(bytes)
        .map_err(|_| INVALID)?
        .0;
    if !value.is_object() || contains_client_identity(&value) {
        return Err(INVALID);
    }
    Ok(value)
}

fn contains_client_identity(value: &Value) -> bool {
    match value {
        Value::Object(values) => values.iter().any(|(key, value)| {
            [
                "actor",
                "actor_context",
                "lease_id",
                "lease_epoch",
                "permissions",
            ]
            .contains(&key.as_str())
                || contains_client_identity(value)
        }),
        Value::Array(values) => values.iter().any(contains_client_identity),
        _ => false,
    }
}

pub(super) fn string<'a>(body: &'a Value, name: &str) -> Result<&'a str> {
    optional_string(body, name)?
        .filter(|value| !value.is_empty())
        .ok_or(INVALID)
}

pub(super) fn optional_string<'a>(body: &'a Value, name: &str) -> Result<Option<&'a str>> {
    match body.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value.as_str().map(Some).ok_or(INVALID),
    }
}

pub(super) fn unsigned(body: &Value, name: &str) -> Result<u64> {
    body.get(name).and_then(Value::as_u64).ok_or(INVALID)
}

pub(super) fn boolean(body: &Value, name: &str) -> Result<bool> {
    body.get(name).and_then(Value::as_bool).ok_or(INVALID)
}

pub(super) fn optional_bool(values: &HashMap<String, String>, name: &str) -> Result<Option<bool>> {
    values
        .get(name)
        .map(|value| value.parse::<bool>().map_err(|_| INVALID))
        .transpose()
}

pub(super) fn memory_id(value: &str) -> Result<&str> {
    if value.is_empty()
        || value.len() > 256
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(INVALID);
    }
    Ok(value)
}

pub(super) fn page_cursor<'a>(
    query: &'a HashMap<String, String>,
    keys: &[&str],
) -> Result<&'a str> {
    let mut values = keys.iter().filter_map(|key| query.get(*key));
    let value = values.next().map(String::as_str).unwrap_or_default();
    if values.next().is_some() {
        return Err(INVALID);
    }
    if value.is_empty() {
        Ok(value)
    } else {
        memory_id(value)
    }
}

pub(super) fn voice_ids(body: &Value, field: &str) -> Result<()> {
    let ids = body[field]
        .as_array()
        .filter(|items| (1..=32).contains(&items.len()))
        .ok_or(INVALID)?;
    let mut unique = HashSet::new();
    for value in ids {
        if !unique.insert(memory_id(value.as_str().ok_or(INVALID)?)?) {
            return Err(INVALID);
        }
    }
    Ok(())
}

pub(super) fn allowed_fields(body: &Value, fields: &[&str]) -> Result<()> {
    if body
        .as_object()
        .is_some_and(|map| map.keys().all(|key| fields.contains(&key.as_str())))
    {
        Ok(())
    } else {
        Err(INVALID)
    }
}

pub(super) fn bounded_text(body: &Value, field: &str, max: usize) -> Result<()> {
    if string(body, field)?.chars().count() > max {
        return Err(INVALID);
    }
    Ok(())
}

pub(super) fn exact_fields(body: &Value, fields: &[&str]) -> Result<()> {
    if body.as_object().is_some_and(|map| {
        map.len() == fields.len() && fields.iter().all(|field| map.contains_key(*field))
    }) {
        Ok(())
    } else {
        Err(INVALID)
    }
}

pub(super) fn request_id(value: &str) -> Result<&str> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.:-".contains(&byte))
    {
        return Err(INVALID);
    }
    Ok(value)
}

pub(super) fn home_intent(body: &Value, actor: Option<ActorContext>) -> Result<HomeActionRequest> {
    let entity = string(body, "entity_id")?;
    let (domain, object) = entity.split_once('.').ok_or(INVALID)?;
    if domain.is_empty()
        || object.is_empty()
        || !domain
            .bytes()
            .chain(object.bytes())
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    {
        return Err(INVALID);
    }
    let operation = string(body, "operation")?;
    let fields = body.as_object().ok_or(INVALID)?;
    let mut params = HashMap::new();
    for (key, value) in fields {
        if ["entity_id", "operation", "request_id"].contains(&key.as_str()) {
            continue;
        }
        let valid = match key.as_str() {
            "brightness_pct" | "position" => value.as_u64().is_some_and(|n| n <= 100),
            "brightness" => value.as_u64().is_some_and(|n| n <= 255),
            "color_temp_kelvin" => value.as_u64().is_some_and(|n| n > 0),
            "temperature" => value.as_f64().is_some_and(f64::is_finite),
            "hvac_mode" => value.as_str().is_some_and(|s| request_id(s).is_ok()),
            _ => false,
        };
        if !valid {
            return Err(INVALID);
        }
        params.insert(
            key.clone(),
            value
                .as_str()
                .map_or_else(|| value.to_string(), str::to_owned),
        );
    }
    Ok(HomeActionRequest {
        request_id: request_id(string(body, "request_id")?)?.into(),
        entity_id: entity.into(),
        operation: request_id(operation)?.into(),
        params,
        actor,
    })
}

pub(super) fn projection(body: String) -> Result<Value> {
    serde_json::from_str::<StrictValue>(&body)
        .map(|value| value.0)
        .map_err(|_| BAD_UPSTREAM)
}

fn timestamp(value: u64) -> Option<DateTime<Utc>> {
    if value == 0 {
        return None;
    }
    DateTime::from_timestamp_millis(value.try_into().ok()?)
}

fn datetime(value: u64) -> Option<String> {
    timestamp(value).map(|time| time.to_rfc3339())
}

fn seconds(value: u64) -> Option<f64> {
    (value != 0).then(|| value as f64 / 1000.0)
}

fn nonempty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

pub(super) fn entity(row: HomeEntity) -> Value {
    json!({
        "entity_id":row.entity_id,"domain":row.domain,"name":row.friendly_name,"state":row.state,
        "brightness_pct":null,"room_id":nonempty(row.room_id),"area_id":null,"capabilities":row.capabilities,
        "state_changed_at":null,"observed_at":datetime(row.observed_at_unix_ms),"available":row.available,
        "temperature":null,"color_temp_kelvin":null,"position":null,"current_temperature":null,
        "current_humidity":null,"hvac_action":null,
    })
}

pub(super) fn home_catalog(data: HomeCatalogSnapshot) -> Result<Value> {
    let observed_at = datetime(data.observed_at_unix_ms).ok_or(ApiError(
        StatusCode::SERVICE_UNAVAILABLE,
        "home_catalog_unavailable",
    ))?;
    let rooms: Vec<_> = data.rooms.into_iter().map(|room| json!({
        "room_id":room.room_id,"name":room.name,"entity_ids":room.entity_ids,
        "current_temperature":null,"current_humidity":null,"default_light":nonempty(room.default_light),
        "default_climate":nonempty(room.default_climate)
    })).collect();
    let entities: Vec<_> = data.entities.into_iter().map(entity).collect();
    let unassigned: Vec<_> = entities
        .iter()
        .filter(|row| row["room_id"].is_null())
        .collect();
    Ok(
        json!({"observed_at":observed_at,"stale":data.stale,"rooms":rooms,"entities":entities,
        "unassigned_entities":unassigned,"reason":null,"source":"real","observation":null,
        "registry_observation":null,"selection_observation":null}),
    )
}

pub(super) fn home_receipt(row: HomeActionReceipt) -> Value {
    json!({"request_id":row.request_id,"entity_id":row.entity_id,"status":row.status,
        "reason":nonempty(row.detail),"effects":[],"dispatch":"unknown","effect":"unknown","observation_scope":null})
}

pub(super) fn microphone(row: MicrophoneRoomStatus) -> Value {
    let capture = match MicCaptureState::try_from(row.state) {
        Ok(MicCaptureState::Waiting) => "waiting",
        Ok(MicCaptureState::Receiving) => "receiving",
        Ok(MicCaptureState::Recovering) => "recovering",
        Ok(MicCaptureState::Muted) => "muted",
        Ok(MicCaptureState::Offline) => "offline",
        _ => "unknown",
    };
    let port = nonempty(row.usb_port);
    json!({"mic_id":row.room_id,"room_id":row.room_id,"room_name":row.room_name,"configured_usb_port":port,
        "usb_port":port,"serial":null,"product":null,"alsa_card":null,"portaudio_index":null,
        "identity_observed_at":null,"connection":"unknown","binding":"unknown",
        "capture":{"state":capture,"last_rms":row.last_rms,"frames_captured":row.frames_captured,
            "recoveries":row.recoveries,"last_frame_at":seconds(row.last_frame_unix_ms)},
        "voice":{"state":"unknown"},"direction":{"state":"unknown"},"physical_validation":{"state":"pending"}})
}

pub(super) fn microphones(data: MicrophonesSnapshot) -> Result<Value> {
    let observed_at = seconds(data.observed_at_unix_ms).ok_or(ApiError(
        StatusCode::SERVICE_UNAVAILABLE,
        "microphone_snapshot_unavailable",
    ))?;
    Ok(
        json!({"observed_at":observed_at,"stale":data.stale,"microphones":data.microphones.into_iter().map(microphone).collect::<Vec<_>>(),
        "unconfigured_rooms":[],"schema_version":1,"stale_after_s":null}),
    )
}

pub(super) fn routine(row: RoutineRecord) -> Result<Value> {
    let mut spec = projection(row.spec_json)?;
    let fields = spec.as_object_mut().ok_or(BAD_UPSTREAM)?;
    fields.extend([
        ("id".into(), json!(&row.routine_id)),
        ("routine_id".into(), json!(row.routine_id)),
        ("name".into(), json!(row.name)),
        ("enabled".into(), json!(row.enabled)),
        ("revision".into(), json!(row.revision)),
        ("updated_at".into(), json!(datetime(row.updated_at_unix_ms))),
    ]);
    if row.updated_at_unix_ms != 0 {
        fields.insert("updated_at_unix_ms".into(), json!(row.updated_at_unix_ms));
    }
    if !row.schedule_json.is_empty() {
        fields.insert("schedule".into(), projection(row.schedule_json)?);
    }
    Ok(spec)
}

pub(super) fn task_scope(scope: Option<&str>, lease: &Lease) -> Result<Option<bool>> {
    match scope {
        None => Ok(None),
        Some("shared") => Ok(Some(false)),
        Some("personal") => {
            lease.require(&[Permission::TasksPersonal])?;
            Ok(Some(true))
        }
        _ => Err(INVALID),
    }
}

pub(super) fn scope_matches(owner: &str, caller: &str, scope: Option<bool>) -> bool {
    if owner.is_empty() {
        scope != Some(true)
    } else {
        owner == caller && scope != Some(false)
    }
}

pub(super) fn list_record(row: HouseholdListRecord) -> Value {
    let personal = !row.owner_user_id.is_empty();
    let owner = if personal { "user" } else { "shared" };
    json!({"id":row.list_id,"name":row.name,"owner_type":owner,"owner_id":nonempty(row.owner_user_id)})
}

pub(super) fn list_items(row: &HouseholdListRecord) -> Result<Value> {
    let doc = projection(row.items_json.clone())?;
    let items = doc.as_array().ok_or(BAD_UPSTREAM)?;
    let items = items
        .iter()
        .map(|item| {
            let id = item.get("id").and_then(Value::as_str).ok_or(BAD_UPSTREAM)?;
            let text = item
                .get("text")
                .and_then(Value::as_str)
                .ok_or(BAD_UPSTREAM)?;
            let completed = item
                .get("completed")
                .and_then(Value::as_bool)
                .ok_or(BAD_UPSTREAM)?;
            Ok(json!({"id":id,"text":text,"completed":completed}))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!(items))
}

pub(super) fn reminder(row: ReminderRecord) -> Result<Value> {
    let fields = if row.payload_json.is_empty() {
        json!({})
    } else {
        projection(row.payload_json)?
    };
    let action = fields["ha_actions"].as_array().map(|actions| actions.iter().map(|action| {
        json!({"domain":action["domain"],"service":action["service"],"entity_id":action["entity_id"],"room":action["room"]})
    }).collect::<Vec<_>>());
    Ok(
        json!({"id":row.reminder_id,"text":row.title,"trigger_at":seconds(row.due_at_unix_ms),
        "recurrence":fields["recurrence"],"state":row.status,"scope":if row.owner_user_id.is_empty() { "shared" } else { "private" },
        "ha_actions":action,"delivery_room":fields.get("delivery_room").unwrap_or(&fields["room_id"]),
        "revision":row.revision}),
    )
}

pub(super) fn committed() -> Value {
    json!({"status":"completed","handled":true,"persistence":"committed",
        "dispatch":"not_required","effect":"observed","observation_scope":"software_state","replication":"not_required"})
}

pub(super) fn voice_switch(row: VoiceGlobalSwitchStatus) -> Value {
    json!({"configured":true,"enabled":row.enabled,"effective":row.enabled,
        "updated_at":seconds(row.updated_at_unix_ms),"revision":row.revision})
}

pub(super) fn voice_profile(row: VoiceProfile, person: bool) -> Value {
    let mut profile = json!({"id":row.profile_id,"name":row.profile_id,"revision":row.profile_revision,"sample_count":row.sample_count,
        "attribution_suspended":row.attribution_suspended});
    if person {
        profile["person_link"] =
            json!({"user_id":nonempty(row.linked_person_id),"revision":row.link_revision});
    }
    profile
}

pub(super) fn capture(row: CaptureControlStatus) -> Value {
    let phase = match CapturePhase::try_from(row.phase) {
        Ok(CapturePhase::Running) => "running",
        Ok(CapturePhase::Muting) => "muting",
        Ok(CapturePhase::Muted) => "muted",
        Ok(CapturePhase::Resuming) => "resuming",
        _ => "unknown",
    };
    json!({"phase":phase,"revision":row.revision,"persisted_muted":row.persisted_muted,"ingress_sealed":row.ingress_sealed,
        "supported":row.supported,"pending_operations":row.pending_operations,"live_capture_owners":row.live_capture_owners,
        "carried_capture_owners":row.carried_capture_owners,"total_mute_confirmed":nexa_proto::is_total_mute_confirmed(&row)})
}

pub(super) fn ambient(row: AmbientControlStatus) -> Value {
    json!({"paused":row.paused,"revision":row.revision,"updated_at":seconds(row.updated_at_unix_ms)})
}

pub(super) fn require_evaluator(lease: &Lease) -> Result<()> {
    if lease.actor().role != Role::Evaluator as i32 {
        return Err(ApiError(
            StatusCode::FORBIDDEN,
            "evaluation_scope_unavailable",
        ));
    }
    Ok(())
}

pub(super) fn verified_sample(
    mut doc: Value,
    id: &str,
    mode: &str,
    granted: &[String],
) -> Result<Value> {
    let object = doc.as_object_mut().ok_or(BAD_UPSTREAM)?;
    let digests = object
        .remove("source_digests")
        .ok_or(AuthError::EvaluationUnavailable)?;
    let relevant = digests.as_array().is_some_and(|values| {
        !values.is_empty()
            && values.iter().all(|raw| {
                raw.as_str().is_some_and(|seen| {
                    granted
                        .iter()
                        .any(|digest| bool::from(digest.as_bytes().ct_eq(seen.as_bytes())))
                })
            })
    });
    if !relevant
        || object.get("fingerprint_id").and_then(Value::as_str) != Some(id)
        || object.get("modality").and_then(Value::as_str) != Some(mode)
    {
        return Err(AuthError::EvaluationUnavailable.into());
    }
    object
        .remove("rows")
        .filter(Value::is_array)
        .ok_or(BAD_UPSTREAM)
}

pub(super) fn zone(row: ZoneState) -> Value {
    let channel = row
        .ma1260_zone
        .checked_sub(1)
        .and_then(|n| u8::try_from(n).ok())
        .filter(|n| *n < 26)
        .map_or_else(
            || row.ma1260_zone.to_string(),
            |n| char::from(b'A' + n).to_string(),
        );
    let audio = if row.muted {
        "muted"
    } else if row.speaking {
        "speaking"
    } else {
        "idle"
    };
    json!({"id":row.zone_id,"name":row.room_id,"icon":"◰","user":null,"lastUtterance":null,
        "audioState":audio,"volume":row.volume_pct,"ma1260_zone":channel,"spotify":{"playing":null}})
}

pub(super) fn llm_endpoint(row: LlmEndpointStatus) -> Value {
    json!({"id":row.endpoint_id,"kind":row.kind,"priority":row.priority,"in_cooldown":row.in_cooldown,
        "cooldown_until":seconds(row.cooldown_until_unix_ms),"consecutive_errors":row.consecutive_errors})
}

pub(super) fn alert(row: AlertRecord) -> Value {
    json!({"id":row.alert_id,"ts":timestamp(row.created_at_unix_ms).map(|time| time.format("%H:%M:%S").to_string()),
        "at":datetime(row.created_at_unix_ms),"priority":row.severity,"type":row.kind,"zone":"—",
        "title":row.message,"body":"","acked":row.acknowledged,"revision":row.revision})
}

pub(super) fn alert_zones(row: AlertZonesConfig) -> Value {
    json!({"available":true,"writable":true,"configured_zones":null,"zones":row.zone_ids,
        "using_default":null,"revision":row.revision,"mute_reason":null})
}

pub(super) fn cloud(row: CloudConsentStatus, withdrawal: bool) -> Value {
    if withdrawal {
        json!({"gate_closed":!row.consent_granted,"marker":null,"pending_write":null,
            "failure":null,"message":null,"withdrawn_latched":row.withdrawn_latched})
    } else {
        json!({"gate_withdrawn":row.withdrawn_latched,"gate_source":null,"boot_decision_basis":null,
            "marker":null,"pending_write":null,"pending_write_cause":null,"registration_failure":null,"last_settled":null,
            "consent_granted":row.consent_granted,"updated_at":seconds(row.updated_at_unix_ms)})
    }
}
