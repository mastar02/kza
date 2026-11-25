use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::Path;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use nexa_proto::v1::{AmbientControlStatus, MemoryFact, MemorySearchHit};
use rusqlite::{Connection, OptionalExtension, Row, Transaction, TransactionBehavior, params};
use serde_json::{Map, Value, json};

use crate::erasure::{Kind, ensure_not_erased, is_erased};
use thiserror::Error;

pub const FACT_MODEL: &str = "BAAI/bge-m3";
pub const VOICE_MODEL: &str = "speechbrain/spkrec-ecapa-voxceleb";
pub const FACT_DIMENSIONS: usize = 1024;
pub const VOICE_DIMENSIONS: usize = 192;
const SCHEMA_VERSION: u32 = 1;
const MAIN: &str = "main";
const AMBIENT: &str = "ambient";

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

#[derive(Clone, Copy, Debug)]
pub(crate) enum VectorKind {
    Fact,
    Voice,
}

impl VectorKind {
    fn spec(self) -> (&'static str, &'static str, usize) {
        match self {
            Self::Fact => ("fact", FACT_MODEL, FACT_DIMENSIONS),
            Self::Voice => ("voice", VOICE_MODEL, VOICE_DIMENSIONS),
        }
    }
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

    fn initialize(mut conn: Connection, ambient_path: &str, lock: Option<File>) -> Result<Self> {
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
        reference.execute_batch(include_str!("ambient.sql"))?;

        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let memory_version = validate_version(&tx, &reference, MAIN)?;
        let ambient_version = validate_version(&tx, &reference, AMBIENT)?;
        if memory_version != ambient_version {
            return Err(Error::Schema("database pair has different revisions"));
        }
        if memory_version == 0 {
            create_memory_schema(&tx)?;
            tx.execute_batch(include_str!("ambient.sql"))?;
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

    pub fn ambient_control(&self) -> Result<AmbientControlStatus> {
        ambient_control(self.read()?)
    }

    pub fn set_ambient_pause(
        &mut self,
        paused: bool,
        expected: u64,
    ) -> Result<AmbientControlStatus> {
        let tx = self.write()?;
        let previous = ambient_control(&tx)?;
        if previous.revision != expected {
            return Err(Error::Conflict("ambient revision conflict"));
        }
        if previous.paused != paused {
            tx.execute(
                "UPDATE ambient.ambient_runtime_control
                 SET paused=?1, revision=revision+1, updated_at_unix_ms=?2 WHERE id=1",
                params![paused, now_ms()?],
            )?;
        }
        let current = ambient_control(&tx)?;
        tx.commit()?;
        Ok(current)
    }

    pub(crate) fn store_fact(
        &mut self,
        conversation: &str,
        owner: &str,
        content: &str,
        category: &str,
        embedding: &[f32],
    ) -> Result<MemoryFact> {
        id(conversation)?;
        optional_id(owner)?;
        if content.trim().is_empty() || content.len() > 64 * 1024 || category.len() > 128 {
            return Err(Error::Invalid("invalid fact content or category"));
        }
        let tx = self.write()?;
        let embedding = vector(&tx, VectorKind::Fact, FACT_MODEL, embedding)?;
        ensure_conversation_live(&tx, conversation)?;
        if !owner.is_empty() {
            ensure_not_erased(&tx, Kind::RetiredUser, owner)?;
        }

        let fact_id = new_id(&tx)?;
        let created = now_ms()?;
        // This RPC accepts facts from the trusted, authorized producer. The owner
        // comes only from that control-plane request, never from the acoustic registry.
        tx.execute(
            "INSERT INTO facts VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1)",
            params![fact_id, conversation, owner, content, category, created],
        )?;
        tx.execute(
            "INSERT INTO fact_sources VALUES (?1, ?2)",
            params![fact_id, conversation],
        )?;
        if !owner.is_empty() {
            tx.execute(
                "INSERT INTO fact_owners VALUES (?1, ?2, 'authenticated', 1)",
                params![fact_id, owner],
            )?;
            tx.execute(
                "INSERT INTO fact_owner_revisions VALUES (?1, 1, 'assign', ?2)",
                params![fact_id, created],
            )?;
        }
        tx.execute(
            "INSERT INTO fact_vec VALUES (?1, ?2, ?3)",
            params![fact_id, embedding, owner],
        )?;
        tx.commit()?;
        Ok(MemoryFact {
            fact_id,
            conversation_id: conversation.into(),
            owner_user_id: owner.into(),
            content: content.into(),
            category: category.into(),
            created_at_unix_ms: created,
            confirmed: true,
        })
    }

    pub(crate) fn query_memory(
        &self,
        embedding: &[f32],
        top_k: u32,
        owner: &str,
        include_shared: bool,
    ) -> Result<Vec<MemorySearchHit>> {
        optional_id(owner)?;
        if !(1..=100).contains(&top_k) || (owner.is_empty() && !include_shared) {
            return Err(Error::Invalid("invalid search scope or top_k"));
        }
        let conn = self.read()?;
        let embedding = vector(conn, VectorKind::Fact, FACT_MODEL, embedding)?;
        if !owner.is_empty() {
            ensure_not_erased(conn, Kind::RetiredUser, owner)?;
        }
        let mut stmt = conn.prepare(
            "SELECT f.fact_id, f.conversation_id, f.owner_user_id, f.content, f.category,
                    f.created_at_unix_ms, f.confirmed, v.distance
             FROM (SELECT fact_id, distance FROM fact_vec
                   WHERE embedding MATCH ?1 AND k=?2 AND owner_user_id=?3
                   AND fact_id IN (
                       SELECT candidate.fact_id FROM facts candidate
                       WHERE candidate.confirmed=1 AND candidate.owner_user_id=?3
                       AND NOT EXISTS (
                           SELECT 1 FROM fact_owner_revisions r WHERE r.fact_id=candidate.fact_id
                           AND r.change_kind='unassign'
                           AND r.revision=(SELECT max(revision) FROM fact_owner_revisions
                                           WHERE fact_id=candidate.fact_id)
                       )
                   )) v
             JOIN facts f ON f.fact_id=v.fact_id
             WHERE f.owner_user_id=?3 AND f.confirmed=1",
        )?;
        let mut hits = Vec::new();
        let scopes = [
            (!owner.is_empty()).then_some(owner),
            include_shared.then_some(""),
        ];
        for scope in scopes.into_iter().flatten() {
            let batch = stmt.query_map(params![embedding, top_k, scope], |row| {
                let distance: f32 = row.get(7)?;
                Ok(MemorySearchHit {
                    fact: Some(fact_from_row(row)?),
                    score: 1.0 - distance,
                })
            })?;
            for hit in batch {
                let hit = hit?;
                if let Some(fact) = &hit.fact
                    && fact_is_live(conn, &fact.fact_id)?
                {
                    hits.push(hit);
                }
            }
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(top_k as usize);
        Ok(hits)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum FactView<'a> {
    Facts,
    Proposals,
    Returned,
    Mine(&'a str),
}

impl Store {
    pub(crate) fn fact_page(
        &self,
        view: FactView<'_>,
        cursor: &str,
        limit: usize,
    ) -> Result<Value> {
        optional_id(cursor)?;
        let conn = self.read()?;
        let (predicate, owner, key) = match view {
            FactView::Facts => (
                "f.confirmed=1 AND r.change_kind IS NOT 'unassign'",
                "",
                "facts",
            ),
            FactView::Proposals => (
                "f.confirmed=0 AND r.change_kind IS NOT 'unassign'",
                "",
                "proposals",
            ),
            FactView::Returned => (
                "f.owner_user_id='' AND r.change_kind='unassign'",
                "",
                "facts",
            ),
            FactView::Mine(owner) => {
                ensure_not_erased(conn, Kind::RetiredUser, owner)?;
                (
                    "f.owner_user_id<>'' AND f.owner_user_id=?2 AND o.person_id=?2",
                    owner,
                    "facts",
                )
            }
        };
        let sql = format!(
            "SELECT {FACT_FIELDS}, COALESCE(r.revision, 0)
             FROM facts f LEFT JOIN fact_owners o ON o.fact_id=f.fact_id
             LEFT JOIN fact_owner_revisions r ON r.fact_id=f.fact_id
                 AND r.revision=(SELECT max(revision) FROM fact_owner_revisions WHERE fact_id=f.fact_id)
             WHERE f.fact_id>?1 AND (?2='' OR o.person_id=?2) AND {predicate}
             ORDER BY f.fact_id"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(params![cursor, owner])?;
        let mut page = Vec::with_capacity(limit);
        let mut last_id = None;
        let mut next = None;
        while let Some(row) = rows.next()? {
            let fact = fact_from_row(row)?;
            if !fact_is_live(conn, &fact.fact_id)? {
                continue;
            }
            if page.len() == limit {
                next = last_id;
                break;
            }
            let personal = matches!(view, FactView::Mine(_));
            page.push(fact_json(&fact, row.get(7)?, personal));
            last_id = Some(fact.fact_id);
        }
        Ok(json!({(key): page, "next": next, "limit": limit}))
    }

    pub(crate) fn owner_counts(&self) -> Result<Value> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT f.fact_id, f.owner_user_id, COALESCE(o.person_id, '')
             FROM facts f LEFT JOIN fact_owners o ON o.fact_id=f.fact_id
             WHERE f.owner_user_id<>'' OR o.person_id IS NOT NULL",
        )?;
        let mut rows = stmt.query([])?;
        let mut counts = std::collections::BTreeMap::<String, usize>::new();
        let mut prune_pending = false;
        while let Some(row) = rows.next()? {
            let owner: String = row.get(1)?;
            let assigned: String = row.get(2)?;
            if !person_is_live(conn, &owner)? || !person_is_live(conn, &assigned)? {
                prune_pending = true;
            } else if !owner.is_empty() && fact_is_live(conn, &row.get::<_, String>(0)?)? {
                *counts.entry(owner).or_default() += 1;
            }
        }
        Ok(json!({"counts": counts, "retired_owner_prune_pending": prune_pending}))
    }

    pub(crate) fn confirm_proposal(&mut self, fact_id: &str) -> Result<Value> {
        id(fact_id)?;
        let tx = self.write()?;
        let mut fact = live_fact(&tx, fact_id)?;
        if !fact.confirmed {
            tx.execute("UPDATE facts SET confirmed=1 WHERE fact_id=?1", [fact_id])?;
            fact.confirmed = true;
        }
        let revision = owner_revision(&tx, fact_id)?;
        let payload = json!({"fact": fact_json(&fact, revision, false)});
        tx.commit()?;
        Ok(payload)
    }

    pub(crate) fn assign_owner(
        &mut self,
        fact_id: &str,
        owner: &str,
        expected: Option<u64>,
    ) -> Result<Value> {
        id(fact_id)?;
        id(owner)?;
        let tx = self.write()?;
        ensure_not_erased(&tx, Kind::RetiredUser, owner)?;
        let fact = live_fact(&tx, fact_id)?;
        let previous = owner_revision(&tx, fact_id)?;
        check_revision(expected, previous)?;
        let revision = if fact.owner_user_id == owner {
            previous
        } else {
            change_owner(&tx, fact_id, owner, previous)?
        };
        tx.commit()?;
        Ok(json!({"fact_id": fact_id, "owner_user_id": owner, "owner_revision": revision}))
    }

    pub(crate) fn return_fact(&mut self, fact_id: &str, user: &str) -> Result<Value> {
        id(fact_id)?;
        let tx = self.write()?;
        ensure_not_erased(&tx, Kind::RetiredUser, user)?;
        let owned: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM fact_owners o JOIN facts f USING (fact_id)
             WHERE o.fact_id=?1 AND o.person_id=?2 AND f.owner_user_id=?2)",
            params![fact_id, user],
            |row| row.get(0),
        )?;
        if !owned {
            return Err(Error::NotFound("fact unavailable"));
        }
        live_fact(&tx, fact_id)?;
        let previous = owner_revision(&tx, fact_id)?;
        change_owner(&tx, fact_id, "", previous)?;
        tx.commit()?;
        Ok(json!({"returned": true}))
    }
}

const FACT_FIELDS: &str = "f.fact_id, f.conversation_id, f.owner_user_id, f.content, f.category,
     f.created_at_unix_ms, f.confirmed";

fn fact_json(fact: &MemoryFact, owner_revision: u64, personal: bool) -> Value {
    let mut value = json!({
        "id": fact.fact_id,
        "fact_id": fact.fact_id,
        "owner_user_id": fact.owner_user_id,
        "content": fact.content,
        "category": fact.category,
        "created_at_unix_ms": fact.created_at_unix_ms,
        "confirmed": fact.confirmed,
        "state": if fact.confirmed { "confirmed" } else { "proposed" },
    });
    if !personal {
        value["conversation_id"] = json!(fact.conversation_id);
        value["owner_revision"] = json!(owner_revision);
    }
    value
}

fn live_fact(conn: &Connection, fact_id: &str) -> Result<MemoryFact> {
    let sql = format!("SELECT {FACT_FIELDS} FROM facts f WHERE f.fact_id=?1");
    let fact = conn.query_row(&sql, [fact_id], fact_from_row).optional()?;
    match fact {
        Some(fact) if fact_is_live(conn, fact_id)? => Ok(fact),
        _ => Err(Error::NotFound("fact unavailable")),
    }
}

fn fact_is_live(conn: &Connection, fact_id: &str) -> Result<bool> {
    let mut stmt = conn.prepare_cached(
        "WITH RECURSIVE ancestry(fact_id) AS (
             SELECT ?1 UNION
             SELECT d.depends_on_fact_id FROM fact_dependencies d JOIN ancestry a USING (fact_id)
         ), sources(fact_id, conversation_id) AS (
             SELECT a.fact_id, f.conversation_id FROM ancestry a JOIN facts f USING (fact_id)
             UNION
             SELECT a.fact_id, s.conversation_id FROM ancestry a JOIN fact_sources s USING (fact_id)
         )
         SELECT f.fact_id, s.conversation_id, f.owner_user_id, COALESCE(o.person_id, ''),
                COALESCE(c.owner_user_id, '')
         FROM sources s JOIN facts f USING (fact_id)
         LEFT JOIN fact_owners o USING (fact_id)
         LEFT JOIN ambient.capture_conversations c ON c.conversation_id=s.conversation_id",
    )?;
    let mut rows = stmt.query([fact_id])?;
    while let Some(row) = rows.next()? {
        if is_erased(conn, Kind::Fact, &row.get::<_, String>(0)?)?
            || !conversation_is_live(conn, &row.get::<_, String>(1)?)?
            || !person_is_live(conn, &row.get::<_, String>(2)?)?
            || !person_is_live(conn, &row.get::<_, String>(3)?)?
            || !person_is_live(conn, &row.get::<_, String>(4)?)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn owner_revision(conn: &Connection, fact_id: &str) -> Result<u64> {
    Ok(conn.query_row(
        "SELECT max(COALESCE((SELECT max(revision) FROM fact_owner_revisions WHERE fact_id=?1), 0),
                    COALESCE((SELECT revision FROM fact_owners WHERE fact_id=?1), 0))",
        [fact_id],
        |row| row.get(0),
    )?)
}

fn change_owner(tx: &Transaction<'_>, fact_id: &str, owner: &str, previous: u64) -> Result<u64> {
    // Transferring the source or a derivative independently could leave its content
    // readable in the old scope even after changing ownership.
    let has_dependencies: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM fact_dependencies WHERE fact_id=?1 OR depends_on_fact_id=?1)",
        [fact_id],
        |row| row.get(0),
    )?;
    if has_dependencies {
        return Err(Error::Precondition("fact has linked dependencies"));
    }
    let revision = previous + 1;
    let kind = if owner.is_empty() {
        tx.execute("DELETE FROM fact_owners WHERE fact_id=?1", [fact_id])?;
        "unassign"
    } else {
        tx.execute(
            "INSERT INTO fact_owners VALUES (?1, ?2, 'manual', ?3)
             ON CONFLICT(fact_id) DO UPDATE SET person_id=excluded.person_id,
                 method=excluded.method, revision=excluded.revision",
            params![fact_id, owner, revision],
        )?;
        "assign"
    };
    tx.execute(
        "UPDATE facts SET owner_user_id=?1 WHERE fact_id=?2",
        params![owner, fact_id],
    )?;
    tx.execute(
        "INSERT INTO fact_owner_revisions VALUES (?1, ?2, ?3, ?4)",
        params![fact_id, revision, kind, now_ms()?],
    )?;
    // vec0 uses owner_user_id as a partition key, so move the vector atomically.
    let embedding: Option<Vec<u8>> = tx
        .query_row(
            "SELECT embedding FROM fact_vec WHERE fact_id=?1",
            [fact_id],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(embedding) = embedding {
        tx.execute("DELETE FROM fact_vec WHERE fact_id=?1", [fact_id])?;
        tx.execute(
            "INSERT INTO fact_vec VALUES (?1, ?2, ?3)",
            params![fact_id, embedding, owner],
        )?;
    }
    Ok(revision)
}

pub(crate) fn check_revision(expected: Option<u64>, current: u64) -> Result<()> {
    if expected.is_some_and(|revision| revision != current) {
        return Err(Error::Conflict("revision conflict"));
    }
    Ok(())
}

pub(crate) fn json_object(input: &str) -> Result<Map<String, Value>> {
    if input.is_empty() {
        return Ok(Map::new());
    }
    serde_json::from_str(input).map_err(|_| Error::Invalid("expected JSON object"))
}

pub(crate) fn json_revision(body: &Map<String, Value>, key: &str) -> Result<Option<u64>> {
    body.get(key)
        .map(|value| {
            value
                .as_u64()
                .ok_or(Error::Invalid("invalid expected revision"))
        })
        .transpose()
}

fn ambient_control(conn: &Connection) -> Result<AmbientControlStatus> {
    Ok(conn.query_row(
        "SELECT paused, revision, updated_at_unix_ms FROM ambient.ambient_runtime_control WHERE id=1",
        [],
        |row| Ok(AmbientControlStatus {
            paused: row.get(0)?, revision: row.get(1)?, updated_at_unix_ms: row.get(2)?,
        }),
    )?)
}

fn fact_from_row(row: &Row<'_>) -> rusqlite::Result<MemoryFact> {
    Ok(MemoryFact {
        fact_id: row.get(0)?,
        conversation_id: row.get(1)?,
        owner_user_id: row.get(2)?,
        content: row.get(3)?,
        category: row.get(4)?,
        created_at_unix_ms: row.get(5)?,
        confirmed: row.get(6)?,
    })
}

pub(crate) fn ensure_conversation_live(conn: &Connection, conversation: &str) -> Result<()> {
    if !conversation_is_live(conn, conversation)? {
        return Err(Error::Precondition("conversation erased"));
    }
    Ok(())
}

pub(crate) fn conversation_is_live(conn: &Connection, conversation: &str) -> Result<bool> {
    let tombstoned: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM conversation_tombstones WHERE conversation_id=?1)",
        [conversation],
        |row| row.get(0),
    )?;
    Ok(!tombstoned && !is_erased(conn, Kind::Conversation, conversation)?)
}

pub(crate) fn person_is_live(conn: &Connection, person: &str) -> Result<bool> {
    Ok(person.is_empty() || !is_erased(conn, Kind::RetiredUser, person)?)
}

pub(crate) fn page_limit(limit: u32) -> usize {
    if limit == 0 {
        50
    } else {
        limit.min(100) as usize
    }
}

pub(crate) fn id(value: &str) -> Result<()> {
    if value.trim().is_empty()
        || value.trim() != value
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        return Err(Error::Invalid("invalid identifier"));
    }
    Ok(())
}

pub(crate) fn optional_id(value: &str) -> Result<()> {
    if value.is_empty() { Ok(()) } else { id(value) }
}

pub(crate) fn now_ms() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_millis()).ok())
        .map(|ms| ms as u64)
        .ok_or(Error::Unavailable("system clock out of range"))
}

pub(crate) fn new_id(conn: &Connection) -> Result<String> {
    Ok(conn.query_row("SELECT lower(hex(randomblob(16)))", [], |row| row.get(0))?)
}

pub(crate) fn vector(
    conn: &Connection,
    kind: VectorKind,
    model: &str,
    coordinates: &[f32],
) -> Result<Vec<u8>> {
    let (kind, model_key, expected_dimensions) = kind.spec();
    if model != model_key {
        return Err(Error::Invalid("unregistered vector model"));
    }
    let dimensions: usize = conn.query_row(
        "SELECT dimensions FROM schema_meta WHERE model_key=?1 AND vector_kind=?2",
        params![model, kind],
        |row| row.get(0),
    )?;
    if dimensions != expected_dimensions {
        return Err(Error::Schema("registered vector dimensions changed"));
    }
    if coordinates.len() != dimensions
        || coordinates.iter().any(|coordinate| !coordinate.is_finite())
    {
        return Err(Error::Invalid("invalid vector dimensions or coordinates"));
    }
    // Normalizing in double precision prevents float32 overflow in cosine distance.
    let norm = coordinates
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Err(Error::Invalid("zero vector"));
    }
    Ok(coordinates
        .iter()
        .flat_map(|value| ((f64::from(*value) / norm) as f32).to_ne_bytes())
        .collect())
}

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

fn set_journal_mode(conn: &Connection, target: &str) -> Result<()> {
    for schema in [MAIN, AMBIENT] {
        let actual: String =
            conn.pragma_update_and_check(Some(schema), "journal_mode", target, |row| row.get(0))?;
        if actual != target {
            return Err(Error::Unavailable("could not select SQLite journal mode"));
        }
    }
    Ok(())
}

/// Register the statically linked extension before opening any SQLite connection.
#[allow(unsafe_code)]
fn register_vec() -> Result<()> {
    static STATUS: OnceLock<i32> = OnceLock::new();
    let status = *STATUS.get_or_init(|| {
        // SAFETY: sqlite-vec exposes the SQLite extension entry point with the exact C
        // ABI required by sqlite3_auto_extension. Its linked symbol is process-lived.
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::ffi::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> std::ffi::c_int,
            >(
                sqlite_vec::sqlite3_vec_init as *const ()
            )))
        }
    });
    if status != rusqlite::ffi::SQLITE_OK {
        return Err(Error::Unavailable(
            "sqlite-vec extension registration failed",
        ));
    }
    Ok(())
}

pub(crate) fn path_str(path: &Path) -> Result<&str> {
    path.to_str().ok_or(Error::Invalid("non-UTF-8 path"))
}

pub fn ensure_private_dir(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && metadata.permissions().mode() & 0o077 == 0 => Ok(()),
        Ok(_) => Err(Error::Invalid(
            "database and socket directories must be private directories",
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            DirBuilder::new().recursive(true).mode(0o700).create(path)?;
            ensure_private_dir(path)
        }
        Err(error) => Err(error.into()),
    }
}

fn private_file(path: &Path) -> Result<File> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && metadata.permissions().mode() & 0o077 == 0 => {}
        Ok(_) => {
            return Err(Error::Invalid(
                "database and lock files must be private regular files",
            ));
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)?)
}

pub(crate) fn lock_file(path: &Path) -> Result<File> {
    let file = private_file(path)?;
    file.try_lock()
        .map_err(|_| Error::Unavailable("exclusive memoryd lock unavailable"))?;
    Ok(file)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::voice::Observation;

    fn sample() -> Vec<f32> {
        let mut vector = vec![0.0; VOICE_DIMENSIONS];
        vector[0] = 1.0;
        vector
    }

    fn count(conn: &Connection, table: &str) -> i64 {
        conn.query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
            row.get(0)
        })
        .expect("count rows")
    }

    fn insert_source(store: &mut Store) {
        store
            .configure_voice_attribution(true)
            .expect("enable voice");
        store
            .record_observation(Observation {
                utterance: "source",
                room: "room",
                model: VOICE_MODEL,
                preview: "synthetic",
                captured_at: 123,
                embedding: &sample(),
            })
            .expect("seed capture");
        let mut embedding = vec![0.0; FACT_DIMENSIONS];
        embedding[0] = 1.0;
        store
            .store_fact("source", "", "synthetic fact", "", &embedding)
            .expect("seed fact");
    }

    #[test]
    fn on_disk_migrations_set_pragmas_and_reopen() {
        let root = tempfile::tempdir().expect("tempdir");
        let path = root.path().join("data");
        for _ in 0..2 {
            let store = Store::open(&path).expect("open databases");
            let conn = store.read().expect("read");
            let timeout: u32 = conn
                .pragma_query_value(None, "busy_timeout", |row| row.get(0))
                .expect("timeout");
            assert_eq!(timeout, 5000);
            for schema in [MAIN, AMBIENT] {
                for (pragma, expected) in [
                    ("user_version", 1),
                    ("foreign_keys", 1),
                    ("secure_delete", 1),
                    ("synchronous", 2),
                ] {
                    let actual: u32 = conn
                        .pragma_query_value(Some(schema), pragma, |row| row.get(0))
                        .expect("pragma");
                    assert_eq!(expected, actual, "{schema}.{pragma}");
                }
                let mode: String = conn
                    .pragma_query_value(Some(schema), "journal_mode", |row| row.get(0))
                    .expect("journal mode");
                assert_eq!(mode, "wal");
            }
        }
    }

    #[test]
    fn rejects_future_and_partial_schemas_without_repairing_them() {
        let root = tempfile::tempdir().expect("tempdir");
        let path = root.path().join("data");
        {
            let store = Store::open(&path).expect("initialize");
            store
                .read()
                .expect("read")
                .execute_batch("PRAGMA ambient.user_version=42;")
                .expect("future version");
        }
        assert!(matches!(
            Store::open(&path),
            Err(Error::Schema("unsupported user_version"))
        ));
        {
            let conn = Connection::open(path.join("ambient.db")).expect("open");
            conn.execute_batch("PRAGMA user_version=1; DROP TABLE pending_audio_deletions;")
                .expect("remove table");
        }
        assert!(matches!(
            Store::open(&path),
            Err(Error::Schema("incomplete or incompatible schema"))
        ));
    }

    #[test]
    fn startup_rejects_changed_dimensions_and_corrupted_ledger() {
        let root = tempfile::tempdir().expect("tempdir");
        let path = root.path().join("data");
        {
            let store = Store::open(&path).expect("initialize");
            store
                .read()
                .expect("read")
                .execute(
                    "UPDATE schema_meta SET dimensions=100 WHERE vector_kind='fact'",
                    [],
                )
                .expect("changed dimensions");
        }
        assert!(matches!(
            Store::open(&path),
            Err(Error::Schema("unrecognized vector model or dimensions"))
        ));
        {
            let conn = Connection::open(path.join("memory.db")).expect("open");
            conn.execute(
                "UPDATE schema_meta SET dimensions=?1 WHERE vector_kind='fact'",
                [FACT_DIMENSIONS],
            )
            .expect("restore dimensions");
            drop(conn);
            let mut store = Store::open(&path).expect("open again");
            store.erase_conversation("test").expect("append to ledger");
            store
                .read()
                .expect("read")
                .execute("UPDATE deletion_ledger_v1 SET row_sha256=zeroblob(32)", [])
                .expect("damage ledger");
        }
        assert!(matches!(
            Store::open(&path),
            Err(Error::Schema("deletion ledger chain invalid"))
        ));
    }

    #[test]
    fn erasure_failure_rolls_back_vectors_tombstones_and_both_databases() {
        let root = tempfile::tempdir().expect("tempdir");
        let mut store = Store::open(&root.path().join("data")).expect("open");
        insert_source(&mut store);
        let conn = store.read().expect("read");
        conn.execute_batch(
            "UPDATE ambient.utterances SET audio_path='test-clip';
             INSERT INTO ambient.audio_write_intents VALUES ('test-intent', 'source', 'source', 100);
             CREATE TEMP TRIGGER fail_erase BEFORE DELETE ON main.voice_fingerprints
             BEGIN SELECT RAISE(ABORT, 'injected after both databases have changed'); END;",
        ).expect("inject constraint failure");

        assert!(store.erase_conversation("source").is_err());
        let conn = store.read().expect("read after rollback");
        for table in [
            "facts",
            "fact_vec",
            "voice_fingerprints",
            "voice_vec",
            "ambient.utterances",
            "ambient.audio_write_intents",
            "ambient.capture_conversations",
        ] {
            assert_eq!(count(conn, table), 1, "{table}");
        }
        for table in [
            "conversation_tombstones",
            "deletion_ledger_v1",
            "erasure_ops",
            "ambient.pending_audio_deletions",
        ] {
            assert_eq!(count(conn, table), 0, "{table}");
        }
        conn.execute_batch("DROP TRIGGER fail_erase")
            .expect("allow retry");
        let receipt = store.erase_conversation("source").expect("retry");
        assert_eq!(
            (
                receipt.facts_deleted,
                receipt.vectors_deleted,
                receipt.captures_deleted
            ),
            (1, 2, 1)
        );
        let conn = store.read().expect("read after erasure");
        assert_eq!(count(conn, "ambient.pending_audio_deletions"), 2);
        for schema in [MAIN, AMBIENT] {
            let mode: String = conn
                .pragma_query_value(Some(schema), "journal_mode", |row| row.get(0))
                .expect("WAL restored");
            assert_eq!(mode, "wal");
        }
        for table in ["fact_vec_vector_chunks00", "voice_vec_vector_chunks00"] {
            let remaining: i64 = conn
                .query_row(
                    &format!(
                        "SELECT count(*) FROM {table} WHERE vectors<>zeroblob(length(vectors))"
                    ),
                    [],
                    |row| row.get(0),
                )
                .expect("check residual vector bytes");
            assert_eq!(remaining, 0, "erased bytes remain in {table}");
        }
    }
}
