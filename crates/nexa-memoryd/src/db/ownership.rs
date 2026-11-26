use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde_json::{Value, json};

use crate::erasure::{Kind, ensure_not_erased};

use super::facts::{fact_is_live, fact_json, live_fact};
use super::{Error, Result, Store, check_revision, id, now_ms, person_is_live};

impl Store {
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
