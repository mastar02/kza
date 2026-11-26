use std::fs::File;
use std::io;
use std::path::Path;

use rusqlite::{Connection, Transaction, TransactionBehavior};
use thiserror::Error;

mod ambient;
mod facts;
mod ownership;
mod schema;
mod util;
mod vec;

pub use util::ensure_private_dir;
pub use vec::{FACT_DIMENSIONS, FACT_MODEL, VOICE_DIMENSIONS, VOICE_MODEL};

pub(crate) use facts::{FactView, conversation_is_live, ensure_conversation_live, person_is_live};
pub(crate) use util::{
    check_revision, id, json_object, json_revision, lock_file, new_id, now_ms, optional_id,
    page_limit, path_str,
};
pub(crate) use vec::{VectorKind, vector};

use schema::set_journal_mode;
use util::private_file;
use vec::register_vec;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(&'static str),
    #[error("{0}")]
    Denied(&'static str),
    #[error("{0}")]
    Unauthenticated(&'static str),
    #[error("{0}")]
    Precondition(&'static str),
    #[error("{0}")]
    Conflict(&'static str),
    #[error("{0}")]
    NotFound(&'static str),
    #[error("database unavailable: {0}")]
    Unavailable(&'static str),
    #[error("invalid database schema or state: {0}")]
    Schema(&'static str),
    #[error(transparent)]
    Sql(#[from] rusqlite::Error),
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Store {
    conn: Connection,
    disk: bool,
    usable: bool,
    _writer_lock: Option<File>,
}

impl Store {
    /// Open the two databases and acquire the daemon's exclusive writer lock.
    pub fn open(data_dir: &Path) -> Result<Self> {
        ensure_private_dir(data_dir)?;
        let writer_lock = lock_file(&data_dir.join("memory.lock"))?;
        let memory = data_dir.join("memory.db");
        let ambient = data_dir.join("ambient.db");
        private_file(&memory)?;
        private_file(&ambient)?;
        register_vec()?;
        Self::initialize(
            Connection::open(memory)?,
            path_str(&ambient)?,
            Some(writer_lock),
        )
    }

    /// Use two in-memory SQLite databases with the production schema and extension.
    pub fn in_memory() -> Result<Self> {
        register_vec()?;
        Self::initialize(Connection::open_in_memory()?, ":memory:", None)
    }

    pub(crate) fn check_alive(&self) -> Result<()> {
        if !self.usable {
            return Err(Error::Unavailable("journal mode recovery is required"));
        }
        Ok(())
    }

    pub(crate) fn read(&self) -> Result<&Connection> {
        self.check_alive()?;
        Ok(&self.conn)
    }

    pub(crate) fn write(&mut self) -> Result<Transaction<'_>> {
        self.check_alive()?;
        Ok(self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?)
    }

    /// Flush previous WAL frames and commit an explicit erasure in rollback-journal mode.
    pub(crate) fn write_both<T>(
        &mut self,
        operation: impl FnOnce(&Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        self.check_alive()?;
        // SQLite's super-journal makes commits across attached files crash-atomic only
        // in rollback-journal mode. The daemon's exclusive writer holds both files
        // throughout the switch, transaction and restoration to WAL.
        let result = (|| {
            if self.disk {
                set_journal_mode(&self.conn, "delete")?;
            }
            let tx = self.write()?;
            let result = operation(&tx)?;
            tx.commit()?;
            Ok(result)
        })();
        if self.disk
            && let Err(error) = set_journal_mode(&self.conn, "wal")
        {
            self.usable = false;
            return Err(error);
        }
        result
    }
}

// The offline migrator reads this vector DDL directly from db.rs.
fn create_memory_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(include_str!("memory.sql"))?;
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE fact_vec USING vec0(
             fact_id TEXT PRIMARY KEY, embedding float[{FACT_DIMENSIONS}] distance_metric=cosine,
             owner_user_id TEXT PARTITION KEY);
         CREATE VIRTUAL TABLE voice_vec USING vec0(
             profile_id TEXT PRIMARY KEY, embedding float[{VOICE_DIMENSIONS}] distance_metric=cosine);"
    ))?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
