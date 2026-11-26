use nexa_proto::v1::{MemoryFact, MemorySearchHit};
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde_json::{Value, json};

use crate::erasure::{Kind, ensure_not_erased, is_erased};

use super::{
    Error, FACT_MODEL, Result, Store, VectorKind, id, new_id, now_ms, optional_id, vector,
};

#[derive(Clone, Copy)]
pub(crate) enum FactView<'a> {
    Facts,
    Proposals,
    Returned,
    Mine(&'a str),
}

impl Store {
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
}

const FACT_FIELDS: &str = "f.fact_id, f.conversation_id, f.owner_user_id, f.content, f.category,
     f.created_at_unix_ms, f.confirmed";

pub(super) fn fact_json(fact: &MemoryFact, owner_revision: u64, personal: bool) -> Value {
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

pub(super) fn live_fact(conn: &Connection, fact_id: &str) -> Result<MemoryFact> {
    let sql = format!("SELECT {FACT_FIELDS} FROM facts f WHERE f.fact_id=?1");
    let fact = conn.query_row(&sql, [fact_id], fact_from_row).optional()?;
    match fact {
        Some(fact) if fact_is_live(conn, fact_id)? => Ok(fact),
        _ => Err(Error::NotFound("fact unavailable")),
    }
}

pub(super) fn fact_is_live(conn: &Connection, fact_id: &str) -> Result<bool> {
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
