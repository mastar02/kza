use serde_json::{Map, Value};

use crate::db::{Error, Result};

mod captures;
mod consent;
mod groups;
mod observe;
mod personality;
mod profiles;

pub(crate) use groups::voice_ids;
pub(crate) use observe::Observation;
pub(crate) use profiles::global_enabled;

fn query_string<'a>(query: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    match query.get(key) {
        None | Some(Value::Null) => Ok(""),
        Some(value) => value
            .as_str()
            .ok_or(Error::Invalid("invalid capture filter")),
    }
}
