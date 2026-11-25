//! Household domain and the SQLite store owned by nexa-brain.
//!
//! Drive deadlines by polling with an injected clock. SQLite operations are
//! synchronous. Async applications can dispatch them through [HouseholdWorker]
//! and perform audio and Home Assistant calls outside SQLite transactions.
//!
//! Start the exclusive database owner with [HouseholdWorker::open_in]. Poll
//! timers for completions, claim notices when a room's output is available,
//! and poll scheduled routines or pass user-initiated occurrences to the
//! routine engine.
//! Execute only effects returned by a new claim and report their observed
//! outcomes. Revalidate authorization in the adapter at execution time.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, Row, Transaction};
use serde::{Deserialize, Serialize};
use thiserror::Error;

macro_rules! sql_enum {
    ($(#[$attr:meta])* pub enum $name:ident { $($variant:ident => $value:literal),+ $(,)? }) => {
        $(#[$attr])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name { $($variant),+ }

        impl rusqlite::ToSql for $name {
            fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
                Ok(match self { $(Self::$variant => $value),+ }.into())
            }
        }

        impl rusqlite::types::FromSql for $name {
            fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
                match value.as_str()? {
                    $($value => Ok(Self::$variant),)+
                    other => Err(rusqlite::types::FromSqlError::Other(
                        Box::new(Error::InvalidState(other.to_owned())),
                    )),
                }
            }
        }
    };
}

pub mod alerts;
pub mod intercom;
pub mod lists;
pub mod reminders;
pub mod routines;
mod store;
pub mod timers;

pub use store::{DATABASE_FILE, HouseholdStore};

/// UTC milliseconds since the Unix epoch.
pub type Timestamp = i64;

pub trait Clock: Send + Sync {
    fn now(&self) -> Timestamp;
    /// Elapsed process time for non-durable voice windows.
    fn monotonic(&self) -> Duration;
}

#[derive(Debug)]
pub struct SystemClock(Instant);

impl Default for SystemClock {
    fn default() -> Self {
        Self(Instant::now())
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => millis(duration),
            Err(error) => -millis(error.duration()),
        }
    }

    fn monotonic(&self) -> Duration {
        self.0.elapsed()
    }
}

impl<C: Clock + ?Sized> Clock for Arc<C> {
    fn now(&self) -> Timestamp {
        (**self).now()
    }

    fn monotonic(&self) -> Duration {
        (**self).monotonic()
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("household schema {found} is newer than supported schema {supported}")]
    NewerSchema { found: i64, supported: i64 },
    #[error("invalid household schema version {0}")]
    InvalidSchema(i64),
    #[error("expected SQLite WAL journal mode, got {0}")]
    JournalMode(String),
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
    #[error("invalid persisted state: {0}")]
    InvalidState(String),
    #[error("{0} does not exist in the requested scope")]
    NotFound(&'static str),
    #[error("{0}")]
    Conflict(&'static str),
    #[error("household worker lock poisoned")]
    PoisonedWorker,
    #[error(transparent)]
    Worker(#[from] tokio::task::JoinError),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Typed command data. The calling service authorizes it before scheduling and
/// admits it through its Home Assistant policy before invoking the service.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HomeAction {
    pub service: String,
    pub entity_id: String,
    pub data: serde_json::Map<String, serde_json::Value>,
}

impl HomeAction {
    fn validate(&self) -> Result<()> {
        required(&self.service, "action service is required")?;
        required(&self.entity_id, "action entity is required")
    }
}

/// Household engines sharing one connection and one clock.
#[derive(Debug)]
pub struct Household<C> {
    store: HouseholdStore,
    clock: C,
    voice_choices: reminders::VoiceChoices,
    reminder_policy: reminders::ReminderPolicy,
    routine_voice_brake: bool,
}

impl<C: Clock> Household<C> {
    pub fn new(store: HouseholdStore, clock: C) -> Self {
        Self {
            store,
            clock,
            voice_choices: Default::default(),
            reminder_policy: Default::default(),
            routine_voice_brake: true,
        }
    }

    fn db(&self) -> &Connection {
        self.store.connection()
    }

    fn transaction<T>(&mut self, write: impl FnOnce(&Transaction<'_>) -> Result<T>) -> Result<T> {
        let tx = self.store.immediate()?;
        let result = write(&tx)?;
        tx.commit()?;
        Ok(result)
    }
}

/// Run a complete synchronous household operation on Tokio's blocking pool.
#[derive(Debug)]
pub struct HouseholdWorker<C>(Arc<Mutex<Household<C>>>);

impl<C> Clone for HouseholdWorker<C> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<C: Clock + 'static> HouseholdWorker<C> {
    pub fn new(household: Household<C>) -> Self {
        Self(Arc::new(Mutex::new(household)))
    }

    /// Open and recover the database at startup of its exclusive owner.
    pub async fn open_in(directory: PathBuf, clock: C) -> Result<Self> {
        tokio::task::spawn_blocking(move || {
            let mut household = Household::new(HouseholdStore::open_in(directory)?, clock);
            household.recover_reminders()?;
            household.recover_routines()?;
            Ok(Self::new(household))
        })
        .await?
    }

    pub async fn call<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut Household<C>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let household = Arc::clone(&self.0);
        tokio::task::spawn_blocking(move || {
            let mut household = household.lock().map_err(|_| Error::PoisonedWorker)?;
            operation(&mut household)
        })
        .await?
    }

    /// Claim the scheduled occurrences due at the given wall-clock instant.
    pub async fn poll_due_scheduled_routines(
        &self,
        now_ms: Timestamp,
    ) -> Result<Vec<routines::RoutineRun>> {
        self.call(move |household| household.poll_due_scheduled_routines(now_ms))
            .await
    }
}

/// Check whether a reminder's home actions are still within the grace window.
#[must_use]
pub fn home_actions_within_grace(delay_seconds: u64, late_action_grace_s: u32) -> bool {
    delay_seconds <= u64::from(late_action_grace_s)
}

fn millis(duration: Duration) -> i64 {
    i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
}

fn required(value: &str, message: &'static str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::InvalidInput(message));
    }
    Ok(())
}

fn name_key(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn query_all<T, P: rusqlite::Params>(
    db: &Connection,
    sql: &str,
    params: P,
    decode: fn(&Row<'_>) -> rusqlite::Result<T>,
) -> Result<Vec<T>> {
    Ok(db
        .prepare(sql)?
        .query_map(params, decode)?
        .collect::<rusqlite::Result<_>>()?)
}

fn json_column<T: serde::de::DeserializeOwned>(
    row: &Row<'_>,
    column: usize,
) -> rusqlite::Result<T> {
    decode_json(&row.get::<_, String>(column)?, column)
}

fn decode_json<T: serde::de::DeserializeOwned>(text: &str, column: usize) -> rusqlite::Result<T> {
    serde_json::from_str(text).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            rusqlite::types::Type::Text,
            Box::new(error),
        )
    })
}
