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

pub const COOKIE_NAME: &str = "kza_dashboard_token";
const TTL: f64 = 2_592_000.0;
const CREDENTIAL_LIMIT: u32 = 256;
const SESSION_LIMIT: u32 = 4096;
const ATTEMPT_LIMIT: u32 = 4096;
const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS auth_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS credentials (
        id TEXT PRIMARY KEY, digest TEXT NOT NULL UNIQUE, subject TEXT NOT NULL,
        role TEXT NOT NULL, user_id TEXT, issued_at REAL NOT NULL, expires_at REAL NOT NULL,
        revision INTEGER NOT NULL, samples TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions (
        digest TEXT PRIMARY KEY, credential TEXT NOT NULL REFERENCES credentials(id) ON DELETE CASCADE,
        legacy_digest TEXT, issued_at REAL NOT NULL, expires_at REAL NOT NULL, revision INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS nexa_sessions_credential ON sessions(credential);
    CREATE TABLE IF NOT EXISTS retired_users (user_id TEXT PRIMARY KEY, retired_at REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS nexa_accounts (
        id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE, permission_level TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS nexa_purge_pending (
        user_id TEXT PRIMARY KEY REFERENCES retired_users(user_id)
    );
    CREATE TABLE IF NOT EXISTS nexa_credential_attempts (
        request_id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, intent TEXT NOT NULL, receipt TEXT NOT NULL
    );";

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

pub fn role_name(role: Role) -> &'static str {
    match role {
        Role::Operator => "operator",
        Role::Home => "home",
        Role::Evaluator => "evaluator",
        Role::Unspecified => "unspecified",
    }
}

fn parse_role(value: &str) -> Option<Role> {
    [Role::Operator, Role::Home, Role::Evaluator]
        .into_iter()
        .find(|role| role_name(*role) == value)
}

pub fn permission_name(permission: Permission) -> &'static str {
    use Permission::*;
    match permission {
        Unspecified => "",
        HomeRead => "home:read",
        HomeControl => "home:control",
        TasksRead => "tasks:read",
        TasksWrite => "tasks:write",
        TasksPersonal => "tasks:personal",
        RoutinesExecute => "routines:execute",
        RoutinesManage => "routines:manage",
        DiagnosticsRead => "diagnostics:read",
        ConversationsRead => "conversations:read",
        MemoryRead => "memory:read",
        MemoryReview => "memory:review",
        MemoryAssign => "memory:assign",
        MemoryOwnRead => "memory:own:read",
        MemoryOwnWrite => "memory:own:write",
        MemoryOwnVoice => "memory:own:voice",
        VoicesRead => "voices:read",
        VoicesReview => "voices:review",
        EvaluationRead => "evaluation:read",
        AmbientControl => "ambient:control",
        AdminManage => "admin:manage",
        AuthManage => "auth:manage",
    }
}

pub fn permissions(role: Role, personal: bool) -> Vec<Permission> {
    use Permission::*;
    let own = [TasksPersonal, MemoryOwnRead, MemoryOwnWrite, MemoryOwnVoice];
    match role {
        Role::Operator => (HomeRead as i32..=AuthManage as i32)
            .filter_map(|value| Permission::try_from(value).ok())
            .filter(|value| !own.contains(value))
            .collect(),
        Role::Home => {
            let mut result = vec![
                HomeRead,
                HomeControl,
                TasksRead,
                TasksWrite,
                RoutinesExecute,
            ];
            if personal {
                result.extend(own);
            }
            result
        }
        Role::Evaluator => vec![EvaluationRead],
        Role::Unspecified => vec![],
    }
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

#[derive(Clone)]
pub(crate) struct Presented {
    digest: String,
    cookie: bool,
    binding: Option<(String, u64)>,
}

impl Presented {
    fn new(raw: &str, cookie: bool) -> Option<Self> {
        if raw.is_empty() || raw.len() > 4096 || (cookie && !raw.starts_with("kzas_")) {
            return None;
        }
        Some(Self {
            digest: sha256(raw),
            cookie,
            binding: None,
        })
    }

    pub(crate) fn bind(mut self, lease: &Lease) -> Self {
        self.binding = Some((lease.lease_id.clone(), lease.revision));
        self
    }
}

pub(crate) fn cookie(headers: &HeaderMap) -> Option<Presented> {
    let mut tokens = headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(';'))
        .filter_map(|pair| pair.trim().split_once('='))
        .filter(|(name, _)| *name == COOKIE_NAME);
    let (_, token) = tokens.next()?;
    if tokens.next().is_some() {
        return None;
    }
    Presented::new(token, true)
}

pub(crate) fn presented(headers: &HeaderMap) -> Option<Presented> {
    let mut values = headers.get_all(header::AUTHORIZATION).iter();
    if let Some(value) = values.next() {
        if values.next().is_some() {
            return None;
        }
        let raw = value.to_str().ok()?;
        let token = raw
            .split_once(' ')
            .filter(|(scheme, _)| scheme.eq_ignore_ascii_case("bearer"))
            .map_or(raw, |(_, value)| value.trim());
        return Presented::new(token, false);
    }
    cookie(headers)
}

pub(crate) fn session_cookie(
    token: &str,
    max_age: u64,
    insecure: bool,
) -> Result<HeaderValue, AuthError> {
    let secure = if insecure { "" } else { "; Secure" };
    let value = format!(
        "{COOKIE_NAME}={token}; Max-Age={max_age}; HttpOnly; SameSite=Strict; Path=/{secure}"
    );
    HeaderValue::from_str(&value).map_err(|_| AuthError::Unavailable)
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

#[derive(Deserialize)]
struct Sample {
    fingerprint_id: String,
    source_digests: Vec<String>,
    modalities: Vec<String>,
    expires_at: f64,
}

pub(crate) struct Lease {
    id: String,
    subject: String,
    role: Role,
    user_id: Option<String>,
    expires_at: f64,
    revision: u64,
    lease_id: String,
    samples: Vec<Sample>,
    session: bool,
}

impl Lease {
    pub(crate) fn actor(&self) -> ActorContext {
        ActorContext {
            subject: self.subject.clone(),
            role: self.role as i32,
            permissions: permissions(self.role, self.user_id.is_some())
                .into_iter()
                .map(|p| p as i32)
                .collect(),
            user_id: self.user_id.clone().unwrap_or_default(),
            lease_id: self.lease_id.clone(),
            lease_epoch: self.revision,
        }
    }

    pub(crate) fn has(&self, permission: Permission) -> bool {
        permissions(self.role, self.user_id.is_some()).contains(&permission)
    }

    pub(crate) fn require(&self, required: &[Permission]) -> Result<(), AuthError> {
        if required.iter().all(|permission| self.has(*permission)) {
            Ok(())
        } else {
            Err(AuthError::Forbidden)
        }
    }

    pub(crate) fn expires_at(&self) -> f64 {
        self.expires_at
    }

    pub(crate) fn whoami(&self) -> Value {
        let role = role_name(self.role);
        let binding = self.user_id.as_ref().map(|user| {
            let material = ["kza-dashboard-account-binding-v1", &self.id, role, user].join("\0");
            sha256(&material)[..32].to_owned()
        });
        let mut names: Vec<_> = permissions(self.role, self.user_id.is_some())
            .into_iter()
            .map(permission_name)
            .collect();
        names.sort_unstable();
        json!({
            "ok": true, "auth_configured": true, "subject": self.subject, "role": role,
            "permissions": names, "expires_at": self.expires_at, "personal_scope": self.user_id.is_some(),
            "account_binding": binding, "routine_actor": format!("\u{001f}routine:{}", sha256(&self.id))
        })
    }

    pub(crate) fn evaluation_grants(&self) -> Value {
        let grants: Vec<_> = self
            .samples
            .iter()
            .filter(|sample| sample.expires_at > now())
            .map(|sample| {
                json!({"fingerprint_id": sample.fingerprint_id, "modalities": sample.modalities,
                                 "expires_at": sample.expires_at.min(self.expires_at)})
            })
            .collect();
        json!({"grants": grants})
    }

    pub(crate) fn evaluation_sources(
        &self,
        id: &str,
        modality: &str,
    ) -> Result<&[String], AuthError> {
        self.samples
            .iter()
            .find(|sample| {
                sample.fingerprint_id == id
                    && sample.expires_at > now()
                    && sample.modalities.iter().any(|value| value == modality)
            })
            .map(|sample| sample.source_digests.as_slice())
            .ok_or(AuthError::Forbidden)
    }
}

struct Row {
    id: String,
    digest: String,
    subject: String,
    role: String,
    user: Option<String>,
    expiry: f64,
    revision: u64,
    samples: String,
}

fn lookup(db: &Connection, proof: &Presented) -> Result<Lease, AuthError> {
    let credential =
        "SELECT c.id,c.digest,c.subject,c.role,c.user_id,c.expires_at,c.revision,c.samples
        FROM credentials c WHERE c.digest=?1 AND c.expires_at>?2
        AND NOT EXISTS(SELECT 1 FROM retired_users r WHERE r.user_id=c.user_id)";
    let session = "SELECT c.id,s.digest,c.subject,c.role,c.user_id,min(c.expires_at,s.expires_at),c.revision,c.samples
        FROM sessions s JOIN credentials c ON c.id=s.credential AND c.revision=s.revision
        WHERE s.digest=?1 AND s.legacy_digest IS NULL AND min(c.expires_at,s.expires_at)>?2
        AND NOT EXISTS(SELECT 1 FROM retired_users r WHERE r.user_id=c.user_id)";
    let time = clock_floor(db)?;
    let read = |sql| {
        db.query_row(sql, params![proof.digest, time], |row| {
            Ok(Row {
                id: row.get(0)?,
                digest: row.get(1)?,
                subject: row.get(2)?,
                role: row.get(3)?,
                user: row.get(4)?,
                expiry: row.get(5)?,
                revision: row.get(6)?,
                samples: row.get(7)?,
            })
        })
        .optional()
    };
    let row = read(session)?;
    let session = row.is_some();
    let row = if row.is_none() && !proof.cookie {
        read(credential)?
    } else {
        row
    };
    // An indexed lookup bounds the candidate set, and the candidate verification is constant-time.
    let verified = row
        .as_ref()
        .is_some_and(|row| matches(&proof.digest, &row.digest));
    if !verified {
        return Err(AuthError::Unauthorized);
    }
    let row = row.ok_or(AuthError::Unauthorized)?;
    let role = parse_role(&row.role).ok_or(AuthError::Unauthorized)?;
    if (row.user.is_some() && role != Role::Home) || row.revision == 0 {
        return Err(AuthError::Unauthorized);
    }
    let samples: Vec<Sample> =
        serde_json::from_str(&row.samples).map_err(|_| AuthError::Unauthorized)?;
    if samples.len() > 64 || (!samples.is_empty() && role != Role::Evaluator) {
        return Err(AuthError::Unauthorized);
    }
    let lease_id = sha256(&format!("nexa-lease-v1:{}", row.digest));
    if proof
        .binding
        .as_ref()
        .is_some_and(|(id, epoch)| id != &lease_id || *epoch != row.revision)
    {
        return Err(AuthError::Unauthorized);
    }
    Ok(Lease {
        id: row.id,
        subject: row.subject,
        role,
        user_id: row.user,
        expires_at: row.expiry,
        revision: row.revision,
        lease_id,
        samples,
        session,
    })
}

struct Inner {
    db: Mutex<Connection>,
    _owner: File,
    changed: watch::Sender<u64>,
}

#[derive(Clone)]
pub struct AuthStore(Arc<Inner>);

impl fmt::Debug for AuthStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("AuthStore").finish_non_exhaustive()
    }
}

fn private_file(path: &Path) -> std::io::Result<File> {
    let file = File::from(open(
        path,
        OFlags::CREATE | OFlags::RDWR | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::RUSR | Mode::WUSR,
    )?);
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.mode() & 0o077 != 0
        || metadata.uid() != rustix::process::geteuid().as_raw()
    {
        return Err(std::io::ErrorKind::PermissionDenied.into());
    }
    Ok(file)
}

impl AuthStore {
    /// Open the migrated dashboard authority under the configured data directory.
    pub fn open_in(data_dir: &Path) -> Result<Self, AuthError> {
        use std::os::unix::fs::DirBuilderExt;
        let directory = data_dir.join("auth");
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&directory)
            .map_err(|_| AuthError::Unavailable)?;
        Self::open(&directory.join("auth.db"))
    }

    /// Open the single authority writer. The first Rust startup invalidates Python-era cookies.
    pub fn open(path: &Path) -> Result<Self, AuthError> {
        let filename = path
            .file_name()
            .ok_or(AuthError::Unavailable)?
            .to_string_lossy();
        let owner = private_file(&path.with_file_name(format!("{filename}.lease")))
            .map_err(|_| AuthError::Unavailable)?;
        owner.try_lock().map_err(|_| AuthError::Unavailable)?;
        private_file(path).map_err(|_| AuthError::Unavailable)?;
        let mut db = Connection::open(path)?;
        db.busy_timeout(std::time::Duration::from_millis(500))?;
        db.pragma_update(None, "journal_mode", "WAL")?;
        db.pragma_update(None, "secure_delete", "ON")?;
        db.pragma_update(None, "foreign_keys", "ON")?;
        let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(SCHEMA)?;
        if tx.execute(
            "INSERT OR IGNORE INTO auth_meta VALUES ('nexa_session_cutover','1')",
            [],
        )? == 1
        {
            tx.execute("DELETE FROM sessions", [])?;
        }
        touch_clock(&tx, clock_floor(&tx)?)?;
        tx.commit()?;
        Ok(Self(Arc::new(Inner {
            db: Mutex::new(db),
            _owner: owner,
            changed: watch::channel(0).0,
        })))
    }

    async fn access<T: Send + 'static>(
        &self,
        work: impl FnOnce(&mut Connection) -> Result<T, AuthError> + Send + 'static,
    ) -> Result<T, AuthError> {
        let inner = self.0.clone();
        tokio::task::spawn_blocking(move || {
            let mut db = inner.db.lock().map_err(|_| AuthError::Unavailable)?;
            work(&mut db)
        })
        .await
        .map_err(|_| AuthError::Unavailable)?
    }

    pub(crate) fn changes(&self) -> watch::Receiver<u64> {
        self.0.changed.subscribe()
    }

    async fn write<T: Send + 'static>(
        &self,
        work: impl FnOnce(&Connection, f64) -> Result<T, AuthError> + Send + 'static,
    ) -> Result<T, AuthError> {
        let inner = self.0.clone();
        self.access(move |db| {
            let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let time = clock_floor(&tx)?;
            let result = work(&tx, time)?;
            touch_clock(&tx, time)?;
            tx.commit()?;
            inner
                .changed
                .send_modify(|counter| *counter = counter.wrapping_add(1));
            Ok(result)
        })
        .await
    }

    pub(crate) async fn validate(
        &self,
        proof: &Presented,
        required: &[Permission],
    ) -> Result<Lease, AuthError> {
        let proof = proof.clone();
        let lease = self.access(move |db| lookup(db, &proof)).await?;
        lease.require(required)?;
        Ok(lease)
    }

    pub async fn bootstrap_operator(&self, token: String) -> Result<(), AuthError> {
        if token.len() < 32 || token.len() > 4096 || token.starts_with("kzas_") {
            return Err(AuthError::Invalid);
        }
        self.seed(
            "bootstrap-operator",
            "dashboard-admin",
            Role::Operator,
            None,
            &token,
        )
        .await
    }

    pub(crate) async fn seed(
        &self,
        id: &str,
        subject: &str,
        role: Role,
        user_id: Option<&str>,
        token: &str,
    ) -> Result<(), AuthError> {
        let (id, subject, user_id, digest) = (
            id.to_owned(),
            subject.to_owned(),
            user_id.map(str::to_owned),
            sha256(token),
        );
        self.write(move |db, time| {
            if let Some(user) = &user_id {
                db.execute(
                    "INSERT OR IGNORE INTO nexa_accounts VALUES (?,?,'adult')",
                    params![user, subject],
                )?;
            }
            let existing: Option<(String, f64)> = db
                .query_row(
                    "SELECT digest,expires_at FROM credentials WHERE id=?1",
                    [&id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if existing.is_some_and(|(old, expiry)| matches(&old, &digest) && expiry > time) {
                return Ok(());
            }
            db.execute("DELETE FROM sessions WHERE credential=?1", [&id])?;
            db.execute(
                "INSERT INTO credentials VALUES (?,?,?,?,?,?,?,1,'[]')
                 ON CONFLICT(id) DO UPDATE SET digest=excluded.digest,issued_at=excluded.issued_at,
                 expires_at=excluded.expires_at,revision=credentials.revision+1",
                params![
                    id,
                    digest,
                    subject,
                    role_name(role),
                    user_id,
                    time,
                    time + TTL
                ],
            )?;
            Ok(())
        })
        .await
    }

    pub(crate) async fn login(
        &self,
        raw: &str,
        previous: Option<Presented>,
    ) -> Result<(String, u64), AuthError> {
        let proof = Presented::new(raw, false).ok_or(AuthError::Unauthorized)?;
        let token = nonce("kzas_");
        let digest = sha256(&token);
        let max_age = self
            .write(move |db, time| {
                let lease = lookup(db, &proof)?;
                if lease.session {
                    return Err(AuthError::Unauthorized);
                }
                db.execute("DELETE FROM sessions WHERE expires_at<=?1", [time])?;
                if let Some(previous) = previous {
                    db.execute("DELETE FROM sessions WHERE digest=?1", [previous.digest])?;
                }
                let count: u32 =
                    db.query_row("SELECT count(*) FROM sessions", [], |row| row.get(0))?;
                if count >= SESSION_LIMIT {
                    return Err(AuthError::Unavailable);
                }
                let expiry = lease.expires_at.min(time + TTL);
                let age = (expiry - time).floor() as u64;
                if age == 0 {
                    return Err(AuthError::Unauthorized);
                }
                db.execute(
                    "INSERT INTO sessions VALUES (?,?,NULL,?,?,?)",
                    params![digest, lease.id, time, expiry, lease.revision],
                )?;
                Ok(age)
            })
            .await?;
        Ok((token, max_age))
    }

    pub(crate) async fn logout(&self, proof: Presented) -> Result<bool, AuthError> {
        self.write(move |db, _| {
            let lease = lookup(db, &proof)?;
            Ok(lease.session
                && db.execute("DELETE FROM sessions WHERE digest=?1", [proof.digest])? == 1)
        })
        .await
    }

    pub(crate) async fn users(&self) -> Result<Value, AuthError> {
        self.access(|db| {
            let mut query =
                db.prepare("SELECT id,name,permission_level FROM nexa_accounts ORDER BY id")?;
            let rows = query
                .query_map([], |row| {
                    Ok(json!({
                        "id": row.get::<_,String>(0)?, "name": row.get::<_,String>(1)?,
                        "permission_level": row.get::<_,String>(2)?, "is_active": true,
                    }))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(json!(rows))
        })
        .await
    }

    pub(crate) async fn write_user(
        &self,
        proof: Presented,
        id: Option<String>,
        body: Value,
    ) -> Result<Value, AuthError> {
        let level = body
            .get("permission_level")
            .and_then(Value::as_str)
            .unwrap_or("adult")
            .to_ascii_lowercase();
        if !["guest", "child", "teen", "adult", "admin"].contains(&level.as_str()) {
            return Err(AuthError::Invalid);
        }
        self.write(move |db, _| {
            lookup(db, &proof)?.require(&[Permission::AdminManage])?;
            if let Some(id) = id {
                if db.execute(
                    "UPDATE nexa_accounts SET permission_level=?1 WHERE id=?2",
                    params![level, id],
                )? != 1
                {
                    return Err(AuthError::NotFound);
                }
                Ok(json!({"ok": true, "permission_level":level.to_ascii_uppercase()}))
            } else {
                let name = body
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|s| valid_id(s) && s.chars().count() <= 64)
                    .ok_or(AuthError::Invalid)?;
                let id = Uuid::new_v4().to_string();
                if db.execute(
                    "INSERT OR IGNORE INTO nexa_accounts VALUES (?,?,?)",
                    params![id, name, level],
                )? != 1
                {
                    return Err(AuthError::Conflict);
                }
                Ok(json!({"ok": true, "user_id":id, "name":name, "message":"Usuario creado"}))
            }
        })
        .await
    }

    pub(crate) async fn retire(
        &self,
        proof: Presented,
        user_id: String,
    ) -> Result<bool, AuthError> {
        if !valid_id(&user_id) {
            return Err(AuthError::Invalid);
        }
        self.write(move |db, time| {
            let lease = lookup(db, &proof)?;
            lease.require(&[Permission::AdminManage, Permission::AuthManage])?;
            let retired = db.query_row("SELECT EXISTS(SELECT 1 FROM retired_users WHERE user_id=?1)", [&user_id],
                |row| row.get::<_,bool>(0))?;
            if !retired {
                let exists = db.query_row(
                    "SELECT EXISTS(SELECT 1 FROM nexa_accounts WHERE id=?1)
                     OR EXISTS(SELECT 1 FROM credentials WHERE user_id=?1)",
                    [&user_id], |row| row.get::<_,bool>(0),
                )?;
                if !exists {
                    return Err(AuthError::NotFound);
                }
                db.execute("INSERT INTO retired_users VALUES (?,?)", params![user_id,time])?;
                db.execute("DELETE FROM sessions WHERE credential IN (SELECT id FROM credentials WHERE user_id=?1)", [&user_id])?;
                db.execute("DELETE FROM credentials WHERE user_id=?1", [&user_id])?;
                db.execute("DELETE FROM nexa_accounts WHERE id=?1", [&user_id])?;
                db.execute("INSERT INTO nexa_purge_pending VALUES (?1)", [&user_id])?;
            }
            let pending = db.query_row("SELECT EXISTS(SELECT 1 FROM nexa_purge_pending WHERE user_id=?1)", [&user_id],
                |row| row.get::<_,bool>(0))?;
            Ok(pending)
        }).await
    }

    pub(crate) async fn finish_purge(&self, user_id: String) -> Result<(), AuthError> {
        self.write(move |db, _| {
            db.execute("DELETE FROM nexa_purge_pending WHERE user_id=?1", [user_id])?;
            Ok(())
        })
        .await
    }

    pub(crate) async fn credential_catalog(
        &self,
        after: String,
        limit: u32,
    ) -> Result<Value, AuthError> {
        self.access(move |db| {
            let observed = clock_floor(db)?;
            let mut query = db.prepare(
                "SELECT id,subject,role,user_id,issued_at,expires_at,revision,samples,
                 NOT EXISTS(SELECT 1 FROM retired_users r WHERE r.user_id=credentials.user_id)
                 FROM credentials WHERE id>?1 ORDER BY id LIMIT ?2",
            )?;
            let mut rows = query.query_map(params![after,limit+1], |row| {
                let expiry: f64 = row.get(5)?;
                let samples: Vec<Sample> = serde_json::from_str(&row.get::<_,String>(7)?)
                    .map_err(|error| rusqlite::Error::FromSqlConversionFailure(7,rusqlite::types::Type::Text,Box::new(error)))?;
                let samples: Vec<_> = samples.into_iter().map(|sample| json!({
                    "fingerprint_id":sample.fingerprint_id,"modalities":sample.modalities,"expires_at":sample.expires_at
                })).collect();
                Ok(json!({
                    "id":row.get::<_,String>(0)?, "subject":row.get::<_,String>(1)?,
                    "role":row.get::<_,String>(2)?, "user_id":row.get::<_,Option<String>>(3)?,
                    "issued_at":row.get::<_,f64>(4)?, "expires_at":expiry, "revision":row.get::<_,u64>(6)?,
                    "active":expiry>observed && row.get::<_,bool>(8)?, "pending":false, "samples":samples,
                }))
            })?.collect::<Result<Vec<_>, _>>()?;
            let next = if rows.len() > limit as usize {
                rows.pop();
                rows.last().map(|row| row["id"].clone())
            } else { None };
            Ok(json!({"rows":rows,"observed_at":observed,"next_cursor":next}))
        }).await
    }

    pub(crate) async fn credential_attempt(
        &self,
        proof: Presented,
        request_id: String,
    ) -> Result<Value, AuthError> {
        if !valid_attempt(&request_id) {
            return Err(AuthError::Invalid);
        }
        self.access(move |db| {
            let actor = lookup(db, &proof)?;
            actor.require(&[Permission::AuthManage])?;
            let receipt = db.query_row("SELECT receipt FROM nexa_credential_attempts WHERE request_id=?1 AND actor_id=?2",
                params![request_id,actor.id], |row| row.get::<_,String>(0)).optional()?.ok_or(AuthError::NotFound)?;
            serde_json::from_str(&receipt).map_err(|_| AuthError::Unavailable)
        }).await
    }

    pub(crate) async fn change_credential(
        &self,
        proof: Presented,
        action: &'static str,
        target: Option<String>,
        body: Value,
    ) -> Result<Value, AuthError> {
        let request_id = body
            .get("request_id")
            .and_then(Value::as_str)
            .filter(|id| valid_attempt(id))
            .ok_or(AuthError::Invalid)?
            .to_owned();
        let intent = sha256(&json!([action, target, body]).to_string());
        self.write(move |db, time| {
            let actor = lookup(db, &proof)?;
            actor.require(&[Permission::AuthManage])?;
            if let Some((owner, prior_intent, receipt)) = db.query_row(
                "SELECT actor_id,intent,receipt FROM nexa_credential_attempts WHERE request_id=?1", [&request_id],
                |row| Ok((row.get::<_,String>(0)?,row.get::<_,String>(1)?,row.get::<_,String>(2)?)),
            ).optional()? {
                if owner != actor.id || !matches(&prior_intent,&intent) {
                    return Err(AuthError::Conflict);
                }
                let receipt: Value = serde_json::from_str(&receipt).map_err(|_| AuthError::Unavailable)?;
                return Ok(json!({"receipt":receipt,"applied":false}));
            }
            let count: u32 = db.query_row("SELECT count(*) FROM nexa_credential_attempts", [], |row| row.get(0))?;
            if count >= ATTEMPT_LIMIT {
                return Err(AuthError::Unavailable);
            }
            let id = target.unwrap_or_else(|| request_id.clone());
            let mut revision = 1;
            let mut expected_revision = None;
            if action != "issue" {
                let expected = body.get("expected_revision").and_then(Value::as_u64).ok_or(AuthError::Invalid)?;
                expected_revision = Some(expected);
                let current: u64 = db.query_row("SELECT revision FROM credentials WHERE id=?1", [&id], |row| row.get(0))
                    .optional()?.ok_or(AuthError::NotFound)?;
                if expected != current || id == actor.id {
                    return Err(AuthError::Conflict);
                }
                revision = if action == "revoke" { current } else {
                    current.checked_add(1).filter(|value| *value <= i64::MAX as u64).ok_or(AuthError::Conflict)?
                };
            }
            let mut token = None;
            if action == "revoke" {
                db.execute("DELETE FROM sessions WHERE credential=?1", [&id])?;
                db.execute("DELETE FROM credentials WHERE id=?1", [&id])?;
            } else {
                let role = body.get("role").and_then(Value::as_str).and_then(parse_role).ok_or(AuthError::Invalid)?;
                let expires = body.get("expires_at").and_then(Value::as_f64).filter(|v| *v > time && *v <= time + TTL)
                    .ok_or(AuthError::Invalid)?;
                let user = match body.get("user_id") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(user)) if role == Role::Home && valid_id(user) => Some(user),
                    _ => return Err(AuthError::Invalid),
                };
                if let Some(user) = user {
                    let exists = db.query_row(
                        "SELECT EXISTS(SELECT 1 FROM nexa_accounts WHERE id=?1)
                         AND NOT EXISTS(SELECT 1 FROM retired_users WHERE user_id=?1)", [user], |row| row.get::<_,bool>(0),
                    )?;
                    if !exists {
                        return Err(AuthError::NotFound);
                    }
                }
                if role == Role::Evaluator || body.get("selections").is_some_and(|value| value.as_array().is_none_or(|rows| !rows.is_empty())) {
                    return Err(AuthError::EvaluationUnavailable);
                }
                if action == "issue" {
                    db.execute("DELETE FROM sessions WHERE credential IN (SELECT id FROM credentials WHERE expires_at<=?1)", [time])?;
                    db.execute("DELETE FROM credentials WHERE expires_at<=?1", [time])?;
                    let count: u32 = db.query_row("SELECT count(*) FROM credentials", [], |row| row.get(0))?;
                    if count >= CREDENTIAL_LIMIT {
                        return Err(AuthError::Unavailable);
                    }
                    let subject = body.get("subject").and_then(Value::as_str).filter(|s| valid_id(s)).ok_or(AuthError::Invalid)?;
                    let secret = nonce("kzac_");
                    db.execute("INSERT INTO credentials VALUES (?,?,?,?,?,?,?,?,'[]')",
                        params![id,sha256(&secret),subject,role_name(role),user,time,expires,revision])?;
                    token = Some(secret);
                } else {
                    db.execute("UPDATE credentials SET role=?1,user_id=?2,issued_at=?3,expires_at=?4,revision=?5,samples='[]' WHERE id=?6",
                        params![role_name(role),user,time,expires,revision,id])?;
                    db.execute("DELETE FROM sessions WHERE credential=?1", [&id])?;
                }
            }
            let receipt = json!({"request_id":request_id,"credential_id":id,"action":action,
                                 "expected_revision":expected_revision,"outcome":"committed",
                                 "reason":"applied","result_revision":revision,"completed_at":time});
            db.execute("INSERT INTO nexa_credential_attempts VALUES (?,?,?,?)",params![request_id,actor.id,intent,receipt.to_string()])?;
            let mut response = json!({"receipt":receipt,"applied":true});
            if let Some(token) = token {
                response["token"] = json!(token);
            }
            Ok(response)
        }).await
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty() && value.chars().count() <= 128 && !value.chars().any(char::is_control)
}

fn valid_attempt(value: &str) -> bool {
    Uuid::parse_str(value).is_ok_and(|id| id.to_string() == value)
}
