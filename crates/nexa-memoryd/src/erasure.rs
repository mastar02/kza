use nexa_proto::v1::ErasureReceipt;
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use sha2::{Digest, Sha256};

use crate::db::{Error, Result, Store, id, new_id, now_ms};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Kind {
    Fact,
    Conversation,
    RetiredUser,
    Utterance,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Conversation => "conversation",
            Self::RetiredUser => "retired_user",
            Self::Utterance => "utterance",
        }
    }
}

impl Store {
    pub(crate) fn delete_fact(
        &mut self,
        fact_id: &str,
        allowed_owner: Option<&str>,
    ) -> Result<ErasureReceipt> {
        self.erase_fact(fact_id, allowed_owner, false)
    }

    pub(crate) fn reject_proposal(&mut self, fact_id: &str) -> Result<ErasureReceipt> {
        self.erase_fact(fact_id, None, true)
    }

    fn erase_fact(
        &mut self,
        fact_id: &str,
        allowed_owner: Option<&str>,
        proposal_only: bool,
    ) -> Result<ErasureReceipt> {
        id(fact_id)?;
        // Flush previous WAL frames and commit the secure deletion before acknowledging.
        self.write_both(|tx| {
            if let Some(receipt) = previous_receipt(tx, Kind::Fact, fact_id)? {
                // The ledger retains no owner identifier. A personal caller cannot
                // inspect a receipt whose ownership can no longer be verified.
                if allowed_owner.is_some() {
                    return Err(Error::NotFound("fact unavailable"));
                }
                return Ok(receipt);
            }
            let fact: Option<(String, bool)> = tx
                .query_row(
                    "SELECT owner_user_id, confirmed FROM facts WHERE fact_id=?1",
                    [fact_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            let (owner, confirmed) = fact.ok_or(Error::NotFound("fact unavailable"))?;
            if proposal_only && confirmed {
                return Err(Error::Precondition("fact is already confirmed"));
            }
            if allowed_owner.is_some_and(|allowed| allowed != owner) {
                return Err(Error::NotFound("fact unavailable"));
            }

            let now = now_ms()?;
            create_selection(tx)?;
            tx.execute("INSERT INTO erase_facts VALUES (?1)", [fact_id])?;
            expand_fact_dependencies(tx)?;
            let (facts, vectors) = delete_facts(tx, now)?;
            let seq = append(tx, Kind::Fact, fact_id, now)?;
            let receipt = record_receipt(tx, seq, (facts, vectors, 0))?;
            drop_selection(tx)?;
            Ok(receipt)
        })
    }

    pub(crate) fn delete_voice_profile(
        &mut self,
        profile_id: &str,
        expected: Option<u64>,
    ) -> Result<u32> {
        id(profile_id)?;
        self.write_both(|tx| {
            let revision = tx
                .query_row(
                    "SELECT revision FROM voice_profiles WHERE profile_id=?1",
                    [profile_id],
                    |row| row.get(0),
                )
                .optional()?
                .ok_or(Error::NotFound("voice profile unavailable"))?;
            crate::db::check_revision(expected, revision)?;

            let now = now_ms()?;
            create_selection(tx)?;
            // Scrub the original fingerprints as well as the vector, which can
            // hold an exact copy of a fingerprint. Remove other derivatives of
            // those sources so their vectors cannot retain the deleted sample.
            tx.execute(
                "INSERT INTO erase_captures
                 SELECT capture_id FROM voice_profile_sources WHERE profile_id=?1
                 UNION SELECT capture_id FROM voice_fingerprints WHERE profile_id=?1",
                [profile_id],
            )?;
            tx.execute(
                "INSERT INTO erase_profiles
                 SELECT ?1 UNION
                 SELECT profile_id FROM voice_profile_sources WHERE capture_id IN erase_captures",
                [profile_id],
            )?;
            tx.execute(
                "INSERT INTO erase_utterances
                 SELECT utterance_id FROM voice_fingerprints WHERE capture_id IN erase_captures",
                [],
            )?;
            for utterance in selection(tx, "SELECT utterance_id FROM erase_utterances")? {
                append(tx, Kind::Utterance, &utterance, now)?;
            }
            delete_vectors(tx, "voice_vec", "profile_id", "erase_profiles")?;
            let captures = delete_captures(tx, now)?;
            drop_selection(tx)?;
            Ok(captures)
        })
    }

    pub(crate) fn erase_conversation(&mut self, conversation: &str) -> Result<ErasureReceipt> {
        id(conversation)?;
        self.erase(Kind::Conversation, conversation)
    }

    pub(crate) fn purge_retired_user(&mut self, user: &str) -> Result<ErasureReceipt> {
        id(user)?;
        self.erase(Kind::RetiredUser, user)
    }

    fn erase(&mut self, kind: Kind, subject: &str) -> Result<ErasureReceipt> {
        if let Some(receipt) = previous_receipt(self.read()?, kind, subject)? {
            return Ok(receipt);
        }
        self.write_both(|tx| {
            let now = now_ms()?;
            create_selection(tx)?;
            match kind {
                Kind::Conversation => {
                    tx.execute("INSERT INTO erase_conversations VALUES (?1)", [subject])?;
                }
                Kind::RetiredUser => retire_person(tx, subject)?,
                _ => return Err(Error::Schema("invalid erasure scope")),
            }
            tx.execute_batch(
                "INSERT OR IGNORE INTO erase_facts
                 SELECT fact_id FROM facts WHERE conversation_id IN erase_conversations
                 UNION SELECT fact_id FROM fact_sources WHERE conversation_id IN erase_conversations;
                 INSERT OR IGNORE INTO erase_capture_conversations SELECT conversation_id FROM erase_conversations;
                 INSERT OR IGNORE INTO erase_utterances
                 SELECT utterance_id FROM ambient.utterances WHERE conversation_id IN erase_capture_conversations;
                 INSERT OR IGNORE INTO erase_captures
                 SELECT capture_id FROM voice_fingerprints WHERE utterance_id IN erase_utterances
                 UNION SELECT capture_id FROM voice_fingerprint_sources
                       WHERE conversation_id IN erase_capture_conversations;
                 INSERT OR IGNORE INTO erase_profiles
                 SELECT profile_id FROM voice_profile_sources WHERE capture_id IN erase_captures;
                 INSERT OR IGNORE INTO erase_utterances
                 SELECT utterance_id FROM voice_fingerprints WHERE capture_id IN erase_captures;",
            )?;
            tx.execute(
                "INSERT OR IGNORE INTO conversation_tombstones
                 SELECT conversation_id, ?1 FROM erase_conversations", [now],
            )?;

            for conversation in selection(tx, "SELECT conversation_id FROM erase_conversations")? {
                append(tx, Kind::Conversation, &conversation, now)?;
            }
            for utterance in selection(tx, "SELECT utterance_id FROM erase_utterances")? {
                append(tx, Kind::Utterance, &utterance, now)?;
            }
            expand_fact_dependencies(tx)?;
            let (facts, fact_vectors) = delete_facts(tx, now)?;
            let voice_vectors = delete_vectors(tx, "voice_vec", "profile_id", "erase_profiles")?;
            let captures = delete_captures(tx, now)?;
            let seq = append(tx, kind, subject, now)?;
            let receipt = record_receipt(
                tx, seq,
                (facts, checked_sum(fact_vectors, voice_vectors)?, captures),
            )?;
            drop_selection(tx)?;
            Ok(receipt)
        })
    }
}

fn create_selection(tx: &Transaction<'_>) -> Result<()> {
    tx.execute_batch(
        "CREATE TEMP TABLE erase_conversations (conversation_id TEXT PRIMARY KEY) WITHOUT ROWID;
         CREATE TEMP TABLE erase_capture_conversations (conversation_id TEXT PRIMARY KEY) WITHOUT ROWID;
         CREATE TEMP TABLE erase_facts (fact_id TEXT PRIMARY KEY) WITHOUT ROWID;
         CREATE TEMP TABLE erase_utterances (utterance_id TEXT PRIMARY KEY) WITHOUT ROWID;
         CREATE TEMP TABLE erase_captures (capture_id TEXT PRIMARY KEY) WITHOUT ROWID;
         CREATE TEMP TABLE erase_profiles (profile_id TEXT PRIMARY KEY) WITHOUT ROWID;",
    )?;
    Ok(())
}

fn drop_selection(tx: &Transaction<'_>) -> Result<()> {
    tx.execute_batch(
        "DROP TABLE erase_conversations; DROP TABLE erase_capture_conversations; DROP TABLE erase_facts; DROP TABLE erase_utterances;
         DROP TABLE erase_captures; DROP TABLE erase_profiles;",
    )?;
    Ok(())
}

fn selection(conn: &Connection, sql: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |row| row.get(0))?;
    Ok(rows.collect::<std::result::Result<_, _>>()?)
}

fn retire_person(tx: &Transaction<'_>, user: &str) -> Result<()> {
    tx.execute(
        "INSERT OR IGNORE INTO erase_facts
         SELECT fact_id FROM facts WHERE owner_user_id=?1
         UNION SELECT fact_id FROM fact_owners WHERE person_id=?1",
        [user],
    )?;
    tx.execute(
        "INSERT INTO erase_capture_conversations
         SELECT conversation_id FROM ambient.capture_conversations WHERE owner_user_id=?1",
        [user],
    )?;
    // A retirement can erase all sources of an exclusively owned conversation.
    // Mixed conversations retain independent facts while the user's own facts
    // and captures are removed, and the retirement ledger fences their owner ID.
    tx.execute(
        "INSERT INTO erase_conversations
         SELECT candidates.conversation_id FROM (
             SELECT conversation_id FROM facts WHERE fact_id IN erase_facts
             UNION SELECT conversation_id FROM erase_capture_conversations
         ) candidates
         WHERE NOT EXISTS (
             SELECT 1 FROM facts other WHERE other.conversation_id=candidates.conversation_id
             AND other.fact_id NOT IN erase_facts
         )
         AND NOT EXISTS (
             SELECT 1 FROM fact_sources s WHERE s.conversation_id=candidates.conversation_id
             AND s.fact_id NOT IN erase_facts
         )
         AND NOT EXISTS (
             SELECT 1 FROM ambient.capture_conversations c
             WHERE c.conversation_id=candidates.conversation_id
             AND c.owner_user_id IS NOT NULL AND c.owner_user_id<>?1
         )",
        [user],
    )?;
    tx.execute(
        "INSERT OR IGNORE INTO erase_profiles
         SELECT profile_id FROM voice_profile_people WHERE person_id=?1",
        [user],
    )?;
    tx.execute_batch(
        "INSERT OR IGNORE INTO erase_captures
         SELECT capture_id FROM voice_profile_sources WHERE profile_id IN erase_profiles
         UNION SELECT capture_id FROM voice_fingerprints WHERE profile_id IN erase_profiles;",
    )?;
    tx.execute(
        "DELETE FROM voice_verification_attempts WHERE person_id=?1",
        [user],
    )?;
    tx.execute(
        "DELETE FROM voice_attribution_consents WHERE person_id=?1",
        [user],
    )?;
    Ok(())
}

fn expand_fact_dependencies(tx: &Transaction<'_>) -> Result<()> {
    tx.execute_batch(
        "WITH RECURSIVE descendants(fact_id) AS (
             SELECT fact_id FROM erase_facts
             UNION SELECT d.fact_id FROM fact_dependencies d
                   JOIN descendants a ON d.depends_on_fact_id=a.fact_id
         )
         INSERT OR IGNORE INTO erase_facts SELECT fact_id FROM descendants;",
    )?;
    Ok(())
}

fn delete_facts(tx: &Transaction<'_>, now: u64) -> Result<(u32, u32)> {
    for fact_id in selection(tx, "SELECT fact_id FROM erase_facts")? {
        append(tx, Kind::Fact, &fact_id, now)?;
    }
    let vectors = delete_vectors(tx, "fact_vec", "fact_id", "erase_facts")?;
    let facts = tx.execute("DELETE FROM facts WHERE fact_id IN erase_facts", [])?;
    Ok((count(facts)?, vectors))
}

fn delete_vectors(tx: &Transaction<'_>, table: &str, key: &str, selected: &str) -> Result<u32> {
    // The pinned vec0 version clears the vector slot as part of its DELETE.
    let deleted = tx.execute(
        &format!("DELETE FROM {table} WHERE {key} IN {selected}"),
        [],
    )?;
    count(deleted)
}

fn delete_captures(tx: &Transaction<'_>, now: u64) -> Result<u32> {
    tx.execute(
        "INSERT OR IGNORE INTO ambient.pending_audio_deletions
         SELECT audio_path, ?1 FROM ambient.utterances
         WHERE audio_path IS NOT NULL AND utterance_id IN erase_utterances
         UNION SELECT path, ?1 FROM ambient.audio_write_intents
               WHERE conversation_id IN erase_capture_conversations OR utterance_id IN erase_utterances",
        [now],
    )?;
    tx.execute(
        "UPDATE voice_attribution_consents SET granted=0, profile_id=NULL,
         profile_revision=NULL, link_revision=NULL, updated_at_unix_ms=max(updated_at_unix_ms+1, ?1)
         WHERE profile_id IN erase_profiles OR person_id IN (
             SELECT person_id FROM voice_profile_people WHERE profile_id IN erase_profiles
         )",
        [now],
    )?;
    tx.execute(
        "UPDATE voice_groups SET revision=revision+1 WHERE group_id IN (
            SELECT group_id FROM voice_group_members WHERE profile_id IN erase_profiles
         )",
        [],
    )?;
    let captures = tx.query_row(
        "SELECT count(*) FROM (
             SELECT utterance_id FROM ambient.utterances WHERE utterance_id IN erase_utterances
             UNION SELECT utterance_id FROM voice_fingerprints WHERE capture_id IN erase_captures
         )",
        [],
        |row| row.get(0),
    )?;
    tx.execute_batch(
        "DELETE FROM ambient.audio_write_intents
         WHERE conversation_id IN erase_capture_conversations OR utterance_id IN erase_utterances;
         DELETE FROM ambient.utterances WHERE utterance_id IN erase_utterances;
         DELETE FROM ambient.capture_conversations WHERE conversation_id IN erase_capture_conversations;
         DELETE FROM voice_fingerprints WHERE capture_id IN erase_captures;
         DELETE FROM voice_profiles WHERE profile_id IN erase_profiles;",
    )?;
    Ok(captures)
}

fn count(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Unavailable("erasure count exceeds receipt capacity"))
}

fn checked_sum(first: u32, second: u32) -> Result<u32> {
    first
        .checked_add(second)
        .ok_or(Error::Unavailable("erasure count exceeds receipt capacity"))
}

fn salt(conn: &Connection) -> Result<[u8; 32]> {
    Ok(conn.query_row(
        "SELECT salt FROM deletion_ledger_salt_v1 WHERE id=1",
        [],
        |row| row.get(0),
    )?)
}

fn subject_digest(conn: &Connection, kind: Kind, subject: &str) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(salt(conn)?);
    hash.update(kind.as_str().as_bytes());
    hash.update([0]);
    hash.update(subject.as_bytes());
    Ok(hash.finalize().into())
}

fn chain_digest(
    seq: u64,
    kind: &str,
    subject: &[u8; 32],
    at: u64,
    previous: &[u8; 32],
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(previous);
    hash.update(seq.to_be_bytes());
    hash.update(at.to_be_bytes());
    hash.update(subject);
    hash.update(kind.as_bytes());
    hash.finalize().into()
}

fn chain_anchor(conn: &Connection) -> Result<[u8; 32]> {
    Ok(Sha256::digest(salt(conn)?).into())
}

pub(crate) fn verify_chain(conn: &Connection) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT seq, kind, subject_sha256, at, prev_sha256, row_sha256
                                 FROM deletion_ledger_v1 ORDER BY seq",
    )?;
    let mut rows = stmt.query([])?;
    let mut last = chain_anchor(conn)?;
    let mut expected_seq = 1;
    while let Some(row) = rows.next()? {
        let seq: u64 = row.get(0)?;
        let kind: String = row.get(1)?;
        let subject: [u8; 32] = row.get(2)?;
        let at: u64 = row.get(3)?;
        let prev: [u8; 32] = row.get(4)?;
        let hash: [u8; 32] = row.get(5)?;
        if seq != expected_seq
            || prev != last
            || hash != chain_digest(seq, &kind, &subject, at, &prev)
        {
            return Err(Error::Schema("deletion ledger chain invalid"));
        }
        last = hash;
        expected_seq += 1;
    }
    Ok(())
}

pub(crate) fn ensure_not_erased(conn: &Connection, kind: Kind, subject: &str) -> Result<()> {
    if is_erased(conn, kind, subject)? {
        return Err(Error::Precondition("subject erased"));
    }
    Ok(())
}

pub(crate) fn is_erased(conn: &Connection, kind: Kind, subject: &str) -> Result<bool> {
    Ok(entry(conn, kind, &subject_digest(conn, kind, subject)?)?.is_some())
}

fn entry(conn: &Connection, kind: Kind, subject: &[u8; 32]) -> Result<Option<u64>> {
    Ok(conn
        .query_row(
            "SELECT seq FROM deletion_ledger_v1 WHERE kind=?1 AND subject_sha256=?2",
            params![kind.as_str(), subject],
            |row| row.get(0),
        )
        .optional()?)
}

fn append(tx: &Transaction<'_>, kind: Kind, subject: &str, at: u64) -> Result<u64> {
    let subject = subject_digest(tx, kind, subject)?;
    if let Some(seq) = entry(tx, kind, &subject)? {
        return Ok(seq);
    }
    let (seq, last) = tx
        .query_row(
            "SELECT seq, row_sha256 FROM deletion_ledger_v1 ORDER BY seq DESC LIMIT 1",
            [],
            |row| Ok((row.get::<_, u64>(0)?, row.get::<_, [u8; 32]>(1)?)),
        )
        .optional()?
        .unwrap_or((0, chain_anchor(tx)?));
    let seq = seq + 1;
    let hash = chain_digest(seq, kind.as_str(), &subject, at, &last);
    tx.execute(
        "INSERT INTO deletion_ledger_v1 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![seq, kind.as_str(), subject, at, last, hash],
    )?;
    Ok(seq)
}

fn previous_receipt(
    conn: &Connection,
    kind: Kind,
    subject: &str,
) -> Result<Option<ErasureReceipt>> {
    Ok(conn
        .query_row(
            "SELECT e.erasure_id, e.facts_deleted, e.vectors_deleted, e.captures_deleted, d.at
         FROM deletion_ledger_v1 d JOIN erasure_ops e ON e.ledger_seq=d.seq
         WHERE d.kind=?1 AND d.subject_sha256=?2",
            params![kind.as_str(), subject_digest(conn, kind, subject)?],
            |row| {
                Ok(ErasureReceipt {
                    erasure_id: row.get(0)?,
                    tombstone_recorded: true,
                    facts_deleted: row.get(1)?,
                    vectors_deleted: row.get(2)?,
                    captures_deleted: row.get(3)?,
                    completed_at_unix_ms: row.get(4)?,
                })
            },
        )
        .optional()?)
}

fn record_receipt(
    tx: &Transaction<'_>,
    seq: u64,
    (facts, vectors, captures): (u32, u32, u32),
) -> Result<ErasureReceipt> {
    let erasure_id = new_id(tx)?;
    tx.execute(
        "INSERT INTO erasure_ops VALUES (?1, ?2, ?3, ?4, ?5)",
        params![seq, erasure_id, facts, vectors, captures],
    )?;
    // An earlier retirement may have recorded a conversation tombstone already.
    let at = tx.query_row(
        "SELECT at FROM deletion_ledger_v1 WHERE seq=?1",
        [seq],
        |row| row.get(0),
    )?;
    Ok(ErasureReceipt {
        erasure_id,
        tombstone_recorded: true,
        facts_deleted: facts,
        vectors_deleted: vectors,
        captures_deleted: captures,
        completed_at_unix_ms: at,
    })
}
