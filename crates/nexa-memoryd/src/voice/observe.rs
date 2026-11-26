use nexa_proto::v1::VoiceObservationReceipt;
use rusqlite::{Connection, OptionalExtension, Transaction, params};

use crate::db::{
    Error, Result, Store, VectorKind, ensure_conversation_live, id, new_id, now_ms, vector,
};
use crate::erasure::{Kind, ensure_not_erased};

use super::global_enabled;

#[derive(Debug)]
pub(crate) struct Observation<'a> {
    pub utterance: &'a str,
    pub room: &'a str,
    pub model: &'a str,
    pub preview: &'a str,
    pub captured_at: u64,
    pub embedding: &'a [f32],
}

impl Store {
    pub(crate) fn record_observation(
        &mut self,
        observation: Observation<'_>,
    ) -> Result<VoiceObservationReceipt> {
        id(observation.utterance)?;
        id(observation.room)?;
        if observation.preview.len() > 4096 || observation.captured_at > i64::MAX as u64 {
            return Err(Error::Invalid("invalid observation preview or timestamp"));
        }
        // The single writer serializes admission and both commits with pause and erasure.
        let embedding = vector(
            self.read()?,
            VectorKind::Voice,
            observation.model,
            observation.embedding,
        )?;
        if self.ambient_control()?.paused {
            return Err(Error::Precondition("ambient capture paused"));
        }
        let conversation = self.observation_conversation(&observation)?;
        let tx = self.write()?;
        let receipt = store_fingerprint(&tx, &observation, &conversation, &embedding)?;
        tx.commit()?;
        Ok(receipt)
    }

    fn observation_conversation(&mut self, observation: &Observation<'_>) -> Result<String> {
        let utterance = observation.utterance;
        let conn = self.read()?;
        ensure_not_erased(conn, Kind::Utterance, utterance)?;
        let provenance: Option<(String, String, Option<String>)> = conn
            .query_row(
                "SELECT c.conversation_id, u.room_id, c.owner_user_id FROM ambient.utterances u
             JOIN ambient.capture_conversations c ON c.conversation_id=u.conversation_id
             WHERE u.utterance_id=?1",
                [utterance],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if let Some((conversation, room, owner)) = provenance {
            if room != observation.room {
                return Err(Error::Conflict("observation room conflict"));
            }
            if let Some(owner) = owner {
                ensure_not_erased(conn, Kind::RetiredUser, &owner)?;
            }
            ensure_conversation_live(conn, &conversation)?;
            return Ok(conversation);
        }

        // The RPC supplies no conversation ID or account. Use the utterance ID
        // for a standalone conversation, unless it already identifies a conversation.
        ensure_conversation_live(conn, utterance)?;
        if conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM ambient.capture_conversations WHERE conversation_id=?1)",
            [utterance],
            |row| row.get(0),
        )? {
            return Err(Error::Conflict(
                "utterance has no unambiguous capture conversation",
            ));
        }

        // WAL commits are atomic per database. Persist the source first so every
        // committed fingerprint has a source and a retry can reuse the utterance.
        let tx = self.write()?;
        tx.execute(
            "INSERT INTO ambient.capture_conversations VALUES (?1, ?2, NULL, ?3)",
            params![utterance, observation.room, observation.captured_at],
        )?;
        tx.execute(
            "INSERT INTO ambient.utterances VALUES (?1, ?1, ?2, ?3, ?4, NULL)",
            params![
                utterance,
                observation.room,
                observation.preview,
                observation.captured_at
            ],
        )?;
        tx.commit()?;
        Ok(utterance.into())
    }
}

fn store_fingerprint(
    tx: &Transaction<'_>,
    observation: &Observation<'_>,
    conversation: &str,
    embedding: &[u8],
) -> Result<VoiceObservationReceipt> {
    let utterance = observation.utterance;
    if let Some((capture_id, stored_embedding, captured_at)) = tx.query_row(
        "SELECT capture_id, embedding, captured_at_unix_ms FROM voice_fingerprints WHERE utterance_id=?1",
        [utterance], |row| Ok((row.get(0)?, row.get::<_, Vec<u8>>(1)?, row.get::<_, u64>(2)?)),
    ).optional()? {
        if stored_embedding != embedding || captured_at != observation.captured_at {
            return Err(Error::Conflict("observation replay differs"));
        }
        return Ok(VoiceObservationReceipt { capture_id, ..Default::default() });
    }

    let matched = if global_enabled(tx)? {
        nearest_profile(tx, embedding)?
    } else {
        None
    };
    let capture_id = new_id(tx)?;
    let (profile_id, matched_profile_id, similarity_score) = match matched {
        Some((profile, score)) => (profile.clone(), profile, score),
        None => {
            let profile = new_id(tx)?;
            tx.execute(
                "INSERT INTO voice_profiles VALUES (?1, ?2, 1)",
                params![profile, observation.model],
            )?;
            tx.execute(
                "INSERT INTO voice_vec VALUES (?1, ?2)",
                params![profile, embedding],
            )?;
            (profile, String::new(), 0.0)
        }
    };
    tx.execute(
        "INSERT INTO voice_fingerprints VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            capture_id,
            utterance,
            observation.model,
            profile_id,
            embedding,
            observation.captured_at
        ],
    )?;
    tx.execute(
        "INSERT INTO voice_fingerprint_sources VALUES (?1, ?2)",
        params![capture_id, conversation],
    )?;
    if matched_profile_id.is_empty() {
        tx.execute(
            "INSERT INTO voice_profile_sources VALUES (?1, ?2)",
            params![profile_id, capture_id],
        )?;
    } else {
        tx.execute(
            "INSERT INTO voice_attribution_events VALUES (?1, ?2, 'match', ?3)",
            params![new_id(tx)?, profile_id, now_ms()?],
        )?;
    }
    Ok(VoiceObservationReceipt {
        capture_id,
        matched_profile_id,
        similarity_score,
    })
}

fn nearest_profile(conn: &Connection, embedding: &[u8]) -> Result<Option<(String, f32)>> {
    let mut stmt = conn.prepare(
        "SELECT profile_id, distance FROM voice_vec
         WHERE embedding MATCH ?1 AND k=2 AND profile_id IN (
             SELECT p.profile_id FROM voice_profiles p
             JOIN voice_profile_people l ON l.profile_id=p.profile_id
             JOIN voice_attribution_consents c ON c.person_id=l.person_id AND c.profile_id=p.profile_id
             LEFT JOIN voice_attribution_suspensions s ON s.profile_id=p.profile_id
             WHERE c.granted=1 AND c.profile_revision=p.revision AND c.link_revision=l.revision
             AND s.profile_id IS NULL
             AND (SELECT count(*) FROM voice_profile_people WHERE person_id=l.person_id)=1
         ) ORDER BY distance",
    )?;
    let mut rows = stmt.query([embedding])?;
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    let profile = row.get(0)?;
    let distance: f32 = row.get(1)?;
    let second_distance: Option<f32> = rows.next()?.map(|row| row.get(1)).transpose()?;
    // The score is an acoustic hint. A close runner-up makes the hint ambiguous.
    if distance > 0.15 || second_distance.is_some_and(|second| second - distance < 0.05) {
        return Ok(None);
    }
    Ok(Some((profile, 1.0 - distance)))
}
