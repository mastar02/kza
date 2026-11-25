use std::{path::Path, time::Duration};

use rusqlite::{Connection, Transaction, TransactionBehavior};

use crate::{Error, Result};

pub const DATABASE_FILE: &str = "household.db";

const MIGRATIONS: &[&str] = &[
    include_str!("schema/001_scheduling.sql"),
    include_str!("schema/002_household.sql"),
];

/// Owns a configured SQLite connection. A database can be reopened by path.
#[derive(Debug)]
pub struct HouseholdStore(Connection);

impl HouseholdStore {
    /// Open household.db inside an existing data directory.
    pub fn open_in(directory: impl AsRef<Path>) -> Result<Self> {
        Self::open(directory.as_ref().join(DATABASE_FILE))
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::configure(Connection::open(path)?, "wal")
    }

    pub fn in_memory() -> Result<Self> {
        // SQLite in-memory databases use a memory journal.
        Self::configure(Connection::open_in_memory()?, "memory")
    }

    fn configure(connection: Connection, journal_mode: &str) -> Result<Self> {
        connection.busy_timeout(Duration::from_millis(5_000))?;
        connection.pragma_update(None, "foreign_keys", true)?;
        let mut store = Self(connection);
        store.migrate()?;
        let mode: String =
            store
                .0
                .pragma_update_and_check(None, "journal_mode", journal_mode, |row| row.get(0))?;
        if mode != journal_mode {
            return Err(Error::JournalMode(mode));
        }
        Ok(store)
    }

    pub fn schema_version(&self) -> Result<i64> {
        Ok(self
            .0
            .pragma_query_value(None, "user_version", |row| row.get(0))?)
    }

    /// Apply each pending migration and its version change in one transaction.
    pub fn migrate(&mut self) -> Result<()> {
        loop {
            let tx = self.immediate()?;
            let version: i64 = tx.pragma_query_value(None, "user_version", |row| row.get(0))?;
            let index = usize::try_from(version).map_err(|_| Error::InvalidSchema(version))?;
            if index == MIGRATIONS.len() {
                return Ok(());
            }
            let migration = MIGRATIONS.get(index).ok_or(Error::NewerSchema {
                found: version,
                supported: MIGRATIONS.len() as i64,
            })?;
            tx.execute_batch(migration)?;
            tx.pragma_update(None, "user_version", version + 1)?;
            tx.commit()?;
        }
    }

    pub(crate) fn connection(&self) -> &Connection {
        &self.0
    }

    pub(crate) fn immediate(&mut self) -> Result<Transaction<'_>> {
        Ok(self
            .0
            .transaction_with_behavior(TransactionBehavior::Immediate)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_reach_current_schema_and_can_be_reapplied() -> Result<()> {
        let mut store = HouseholdStore::in_memory()?;
        store.migrate()?;
        assert_eq!(store.schema_version()?, MIGRATIONS.len() as i64);
        let tables = crate::query_all(
            store.connection(),
            "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
            [],
            |row| row.get::<_, String>(0),
        )?;
        assert_eq!(
            tables,
            [
                "alert_preferences",
                "alerts",
                "list_items",
                "lists",
                "reminder_home_actions",
                "reminder_notices",
                "reminders",
                "routine_runs",
                "routines",
                "timers"
            ],
        );
        let timeout: u32 = store
            .connection()
            .pragma_query_value(None, "busy_timeout", |row| row.get(0))?;
        let foreign_keys: bool =
            store
                .connection()
                .pragma_query_value(None, "foreign_keys", |row| row.get(0))?;
        assert_eq!(timeout, 5_000);
        assert!(foreign_keys);
        assert!(
            store
                .connection()
                .execute(
                    "INSERT INTO list_items(list_id, text) VALUES (999, 'leche')",
                    []
                )
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn upgrade_preserves_existing_data_and_rejects_newer_versions() -> Result<()> {
        let connection = Connection::open_in_memory()?;
        connection.execute_batch(MIGRATIONS[0])?;
        connection.pragma_update(None, "user_version", 1)?;
        connection.execute(
            "INSERT INTO timers(name, name_key, room, state, deadline)
                            VALUES ('Mate', 'mate', 'cocina', 'running', 2000)",
            [],
        )?;
        let store = HouseholdStore::configure(connection, "memory")?;
        let name: String = store
            .connection()
            .query_row("SELECT name FROM timers", [], |row| row.get(0))?;
        assert_eq!(name, "Mate");
        assert_eq!(store.schema_version()?, 2);
        store.connection().pragma_update(None, "user_version", 3)?;
        assert!(matches!(
            HouseholdStore::configure(store.0, "memory"),
            Err(Error::NewerSchema {
                found: 3,
                supported: 2
            })
        ));
        Ok(())
    }

    #[test]
    fn failed_migration_rolls_back_ddl_and_keeps_user_version() -> Result<()> {
        let connection = Connection::open_in_memory()?;
        connection.execute_batch(MIGRATIONS[0])?;
        connection.pragma_update(None, "user_version", 1)?;
        connection.execute_batch("CREATE TABLE alerts (existing INTEGER)")?;
        let mut store = HouseholdStore(connection);
        assert!(matches!(store.migrate(), Err(Error::Sqlite(_))));
        assert_eq!(store.schema_version()?, 1);
        let rolled_back: bool = store.connection().query_row(
            "SELECT NOT EXISTS (SELECT 1 FROM sqlite_schema WHERE name IN ('lists', 'list_items'))",
            [],
            |row| row.get(0),
        )?;
        assert!(rolled_back);
        Ok(())
    }

    #[test]
    fn file_database_uses_wal_on_every_reopen()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let directory = tempfile::tempdir()?;
        let store = HouseholdStore::open_in(directory.path())?;
        for connection in [
            &store,
            &HouseholdStore::open(directory.path().join(DATABASE_FILE))?,
        ] {
            let mode: String =
                connection
                    .connection()
                    .pragma_query_value(None, "journal_mode", |row| row.get(0))?;
            assert_eq!(mode, "wal");
        }
        Ok(())
    }
}
