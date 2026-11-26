use std::fs::File;
use std::time::Duration;

use rusqlite::{Connection, TransactionBehavior, params};

use super::ambient::ambient_control;
use super::{Error, Result, Store, VectorKind, create_memory_schema};

pub(super) const SCHEMA_VERSION: u32 = 1;
pub(super) const MAIN: &str = "main";
pub(super) const AMBIENT: &str = "ambient";

impl Store {
    pub(super) fn initialize(
        mut conn: Connection,
        ambient_path: &str,
        lock: Option<File>,
    ) -> Result<Self> {
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute("ATTACH DATABASE ?1 AS ambient", [ambient_path])?;
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA temp_store = MEMORY;
             PRAGMA main.secure_delete = ON;
             PRAGMA ambient.secure_delete = ON;
             PRAGMA main.synchronous = FULL;
             PRAGMA ambient.synchronous = FULL;",
        )?;

        let reference = Connection::open_in_memory()?;
        reference.execute_batch("ATTACH DATABASE ':memory:' AS ambient;")?;
        create_memory_schema(&reference)?;
        reference.execute_batch(include_str!("../ambient.sql"))?;

        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let memory_version = validate_version(&tx, &reference, MAIN)?;
        let ambient_version = validate_version(&tx, &reference, AMBIENT)?;
        if memory_version != ambient_version {
            return Err(Error::Schema("database pair has different revisions"));
        }
        if memory_version == 0 {
            create_memory_schema(&tx)?;
            tx.execute_batch(include_str!("../ambient.sql"))?;
            initialize_metadata(&tx)?;
            for schema in [MAIN, AMBIENT] {
                tx.pragma_update(Some(schema), "user_version", SCHEMA_VERSION)?;
            }
        }
        validate_state(&tx)?;
        tx.commit()?;

        let disk = lock.is_some();
        if disk {
            set_journal_mode(&conn, "wal")?;
        }
        Ok(Self {
            conn,
            disk,
            usable: true,
            _writer_lock: lock,
        })
    }
}

fn initialize_metadata(conn: &Connection) -> Result<()> {
    for kind in [VectorKind::Fact, VectorKind::Voice] {
        let (kind, model, dimensions) = kind.spec();
        conn.execute(
            "INSERT INTO schema_meta VALUES (?1, ?2, ?3)",
            params![model, kind, dimensions],
        )?;
    }
    conn.execute_batch(
        "INSERT INTO deletion_ledger_salt_v1 VALUES (1, randomblob(32));
         INSERT INTO voice_attribution_control VALUES (1, 0, 0, 0);
         INSERT INTO ambient.ambient_runtime_control VALUES (1, 0, 0, 0);",
    )?;
    Ok(())
}

fn validate_version(conn: &Connection, reference: &Connection, schema: &str) -> Result<u32> {
    let version: u32 = conn.pragma_query_value(Some(schema), "user_version", |row| row.get(0))?;
    if version > SCHEMA_VERSION {
        return Err(Error::Schema("unsupported user_version"));
    }
    let sql = format!(
        "SELECT type, name, sql FROM {schema}.sqlite_schema
                       WHERE name NOT GLOB 'sqlite_*' ORDER BY type, name"
    );
    let objects = |db: &Connection| -> Result<Vec<(String, String, Option<String>)>> {
        let mut stmt = db.prepare(&sql)?;
        let rows = stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?;
        Ok(rows.collect::<std::result::Result<_, _>>()?)
    };
    let actual = objects(conn)?;
    if (version == 0 && !actual.is_empty()) || (version != 0 && actual != objects(reference)?) {
        return Err(Error::Schema("incomplete or incompatible schema"));
    }
    Ok(version)
}

fn validate_state(conn: &Connection) -> Result<()> {
    for schema in [MAIN, AMBIENT] {
        let mut integrity =
            conn.pragma_query_value(Some(schema), "quick_check", |row| row.get::<_, String>(0))?;
        if integrity != "ok" {
            return Err(Error::Schema("integrity check failed"));
        }
        conn.pragma_query(Some(schema), "foreign_key_check", |_| {
            integrity.clear();
            Ok(())
        })?;
        if integrity.is_empty() {
            return Err(Error::Schema("foreign key check failed"));
        }
    }
    let mut stmt = conn.prepare(
        "SELECT vector_kind, model_key, dimensions FROM schema_meta ORDER BY vector_kind",
    )?;
    let models = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<std::result::Result<Vec<(String, String, usize)>, _>>()?;
    if !models
        .iter()
        .map(|(kind, model, dimensions)| (kind.as_str(), model.as_str(), *dimensions))
        .eq([VectorKind::Fact.spec(), VectorKind::Voice.spec()])
    {
        return Err(Error::Schema("unrecognized vector model or dimensions"));
    }
    ambient_control(conn)?;
    crate::voice::global_enabled(conn)?;
    crate::erasure::verify_chain(conn)?;
    Ok(())
}

pub(super) fn set_journal_mode(conn: &Connection, target: &str) -> Result<()> {
    for schema in [MAIN, AMBIENT] {
        let actual: String =
            conn.pragma_update_and_check(Some(schema), "journal_mode", target, |row| row.get(0))?;
        if actual != target {
            return Err(Error::Unavailable("could not select SQLite journal mode"));
        }
    }
    Ok(())
}
