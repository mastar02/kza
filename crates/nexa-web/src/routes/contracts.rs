//! Browser JSON and request validation at the HTTP/protobuf boundary.

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use nexa_proto::v1::*;
use serde::de::{Deserialize, Deserializer, Error, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value, json};

use super::{BAD_UPSTREAM, INVALID, Result};

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
