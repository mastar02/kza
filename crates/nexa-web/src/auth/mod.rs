//! SQLite-backed dashboard authority. A lease is re-read at each dispatch boundary.

use std::{
    fmt,
    fs::File,
    os::unix::fs::MetadataExt,
    path::Path,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::http::{HeaderMap, HeaderValue, Uri, header};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use nexa_proto::v1::{ActorContext, Permission, Role};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use rustix::fs::{Mode, OFlags, open};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::sync::watch;
use uuid::Uuid;

mod credentials;
mod rbac;
#[path = "../auth.rs"]
mod schema;
mod session;
mod store;
use rbac::parse_role;
pub use rbac::{permission_name, permissions, role_name};
use schema::SCHEMA;
use session::lookup;
pub(crate) use session::{Lease, Presented, cookie, presented, session_cookie};
pub use store::AuthStore;

pub const COOKIE_NAME: &str = "kza_dashboard_token";
const TTL: f64 = 2_592_000.0;
const CREDENTIAL_LIMIT: u32 = 256;
const SESSION_LIMIT: u32 = 4096;
const ATTEMPT_LIMIT: u32 = 4096;
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("dashboard_session_required")]
    Unauthorized,
    #[error("dashboard_permission_denied")]
    Forbidden,
    #[error("invalid_authority_request")]
    Invalid,
    #[error("authority_record_not_found")]
    NotFound,
    #[error("authority_revision_conflict")]
    Conflict,
    #[error("authority_unavailable")]
    Unavailable,
    #[error("evaluation_source_binding_unavailable")]
    EvaluationUnavailable,
}

impl From<rusqlite::Error> for AuthError {
    fn from(_: rusqlite::Error) -> Self {
        Self::Unavailable
    }
}

pub fn sha256(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn matches(left: &str, right: &str) -> bool {
    bool::from(left.as_bytes().ct_eq(right.as_bytes()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Origin {
    scheme: String,
    host: String,
    port: u16,
}

impl Origin {
    pub fn parse(raw: &str) -> Result<Self, AuthError> {
        let (scheme, authority) = raw.split_once("://").ok_or(AuthError::Forbidden)?;
        let default_port = match scheme {
            "http" => 80,
            "https" => 443,
            _ => return Err(AuthError::Forbidden),
        };
        if authority.is_empty()
            || authority.ends_with(':')
            || authority.contains(['/', '?', '#', '@', '%', '\\', ','])
        {
            return Err(AuthError::Forbidden);
        }
        let uri: Uri = raw.parse().map_err(|_| AuthError::Forbidden)?;
        let authority = uri.authority().ok_or(AuthError::Forbidden)?;
        let host = authority.host();
        let valid_host = if let Some(ip) = host.strip_prefix('[').and_then(|h| h.strip_suffix(']'))
        {
            ip.parse::<std::net::Ipv6Addr>().is_ok()
        } else {
            !host.is_empty()
                && host.split('.').all(|label| {
                    !label.is_empty()
                        && !label.starts_with('-')
                        && !label.ends_with('-')
                        && label
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
                })
        };
        let port = match authority.port() {
            Some(port) => port.as_u16(),
            None if authority.as_str() == host => default_port,
            None => return Err(AuthError::Forbidden),
        };
        if !valid_host || port == 0 {
            return Err(AuthError::Forbidden);
        }
        Ok(Self {
            scheme: scheme.into(),
            host: host.to_ascii_lowercase(),
            port,
        })
    }

    pub fn check(&self, headers: &HeaderMap) -> Result<(), AuthError> {
        let mut origins = headers.get_all(header::ORIGIN).iter();
        let origin = origins
            .next()
            .and_then(|value| value.to_str().ok())
            .ok_or(AuthError::Forbidden)?;
        if origins.next().is_some() || Self::parse(origin)? != *self {
            return Err(AuthError::Forbidden);
        }
        Ok(())
    }
}

fn nonce(prefix: &str) -> String {
    let bytes: [u8; 32] = rand::random();
    format!("{prefix}{}", URL_SAFE_NO_PAD.encode(bytes))
}

pub(crate) fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn clock_floor(db: &Connection) -> Result<f64, AuthError> {
    let floor: Option<String> = db
        .query_row(
            "SELECT value FROM auth_meta WHERE key='clock_floor'",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let floor = floor
        .map(|value| value.parse::<f64>().map_err(|_| AuthError::Unavailable))
        .transpose()?
        .unwrap_or_default();
    Ok(now().max(floor))
}

fn touch_clock(db: &Connection, time: f64) -> Result<(), AuthError> {
    db.execute(
        "INSERT OR REPLACE INTO auth_meta VALUES ('clock_floor',?)",
        [time.to_string()],
    )?;
    Ok(())
}

fn valid_id(value: &str) -> bool {
    !value.is_empty() && value.chars().count() <= 128 && !value.chars().any(char::is_control)
}

fn valid_attempt(value: &str) -> bool {
    Uuid::parse_str(value).is_ok_and(|id| id.to_string() == value)
}
