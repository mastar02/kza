use nexa_proto::v1::{IdentifyVoiceResponse, VoiceObservationReceipt};
use rusqlite::{Connection, OptionalExtension, Transaction, params};

use crate::db::{
    Error, Result, Store, VOICE_DIMENSIONS, VOICE_MODEL, VectorKind, ensure_conversation_live, id,
    new_id, now_ms, vector,
};
use crate::erasure::{Kind, ensure_not_erased};

use super::global_enabled;

const VOICE_MATCH_MAX_DISTANCE: f32 = 0.42;
const UNLINKED_CLUSTER_MAX_DISTANCE: f32 = 0.38;
const MATCH_MARGIN: f32 = 0.05;

#[derive(Debug)]
pub(crate) struct Observation<'a> {
    pub utterance: &'a str,
    pub room: &'a str,
    pub model: &'a str,
    pub preview: &'a str,
    pub captured_at: u64,
    pub embedding: &'a [f32],
    pub conversation_id: &'a str,
}

impl Store {
    pub(crate) fn identify_voice(
        &self,
        embedding: &[f32],
        model: &str,
    ) -> Result<IdentifyVoiceResponse> {
        let conn = self.read()?;
        let model = if model.trim().is_empty() {
            VOICE_MODEL
        } else {
            model
        };
        let embedding = vector(conn, VectorKind::Voice, model, embedding)?;
        if !global_enabled(conn)? {
            return Ok(IdentifyVoiceResponse::default());
        }
        let Some((profile_id, similarity_score)) = nearest_profile(conn, &embedding)? else {
            return Ok(IdentifyVoiceResponse::default());
        };
        let linked_person_id: String = conn
            .query_row(
                "SELECT COALESCE(person_id, '') FROM voice_profile_people WHERE profile_id=?1",
                [&profile_id],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or_default();
        Ok(IdentifyVoiceResponse {
            profile_id,
            linked_person_id,
            similarity_score,
            consented: true,
        })
    }

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

        let conversation = if !observation.conversation_id.is_empty() {
            let conv_id = observation.conversation_id;
            id(conv_id)?;
            ensure_conversation_live(conn, conv_id)?;
            let existing: Option<(String, Option<String>)> = conn
                .query_row(
                    "SELECT room_id, owner_user_id FROM ambient.capture_conversations
                     WHERE conversation_id=?1",
                    [conv_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if let Some((room, owner)) = existing {
                if room != observation.room {
                    return Err(Error::Conflict("observation room conflict"));
                }
                if let Some(owner) = owner {
                    ensure_not_erased(conn, Kind::RetiredUser, &owner)?;
                }
            }
            let tx = self.write()?;
            tx.execute(
                "INSERT OR IGNORE INTO ambient.capture_conversations VALUES (?1, ?2, NULL, ?3)",
                params![conv_id, observation.room, observation.captured_at],
            )?;
            tx.execute(
                "INSERT INTO ambient.utterances VALUES (?1, ?2, ?3, ?4, ?5, NULL)",
                params![
                    utterance,
                    conv_id,
                    observation.room,
                    observation.preview,
                    observation.captured_at
                ],
            )?;
            tx.commit()?;
            return Ok(conv_id.into());
        } else {
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
            utterance
        };

        // WAL commits are atomic per database. Persist the source first so every
        // committed fingerprint has a source and a retry can reuse the utterance.
        let tx = self.write()?;
        tx.execute(
            "INSERT INTO ambient.capture_conversations VALUES (?1, ?2, NULL, ?3)",
            params![conversation, observation.room, observation.captured_at],
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
        Ok(conversation.into())
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
    let (profile_id, matched_profile_id, similarity_score, reused_unlinked) = match matched {
        Some((profile, score)) => (profile.clone(), profile, score, false),
        None => {
            if let Some(candidate) = nearest_unlinked_candidate(tx, embedding, conversation)? {
                (candidate, String::new(), 0.0, true)
            } else {
                let profile = new_id(tx)?;
                tx.execute(
                    "INSERT INTO voice_profiles VALUES (?1, ?2, 1)",
                    params![profile, observation.model],
                )?;
                tx.execute(
                    "INSERT INTO voice_vec VALUES (?1, ?2)",
                    params![profile, embedding],
                )?;
                (profile, String::new(), 0.0, false)
            }
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
    if matched_profile_id.is_empty() || reused_unlinked {
        tx.execute(
            "INSERT INTO voice_profile_sources VALUES (?1, ?2)",
            params![profile_id, capture_id],
        )?;
        if reused_unlinked {
            refresh_unlinked_centroid(tx, &profile_id, observation.model)?;
        }
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

fn refresh_unlinked_centroid(tx: &Transaction<'_>, profile_id: &str, model: &str) -> Result<()> {
    let mut centroid = vec![0.0_f32; VOICE_DIMENSIONS];
    {
        let mut stmt =
            tx.prepare("SELECT embedding FROM voice_fingerprints WHERE profile_id=?1")?;
        let mut rows = stmt.query([profile_id])?;
        let mut count = 0_usize;
        while let Some(row) = rows.next()? {
            let embedding: Vec<u8> = row.get(0)?;
            let (coordinates, rest) = embedding.as_chunks::<4>();
            if !rest.is_empty() || coordinates.len() != centroid.len() {
                return Err(Error::Schema("invalid voice embedding"));
            }
            for (sum, bytes) in centroid.iter_mut().zip(coordinates) {
                *sum += f32::from_ne_bytes(*bytes);
            }
            count += 1;
        }
        if count == 0 {
            return Ok(());
        }
    }
    if let Ok(normalized) = vector(tx, VectorKind::Voice, model, &centroid) {
        tx.execute("DELETE FROM voice_vec WHERE profile_id=?1", [profile_id])?;
        tx.execute(
            "INSERT INTO voice_vec VALUES (?1, ?2)",
            params![profile_id, normalized],
        )?;
    }
    Ok(())
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
    if distance > VOICE_MATCH_MAX_DISTANCE
        || second_distance.is_some_and(|second| second - distance < MATCH_MARGIN)
    {
        return Ok(None);
    }
    Ok(Some((profile, 1.0 - distance)))
}

fn nearest_unlinked_candidate(
    conn: &Connection,
    embedding: &[u8],
    conversation: &str,
) -> Result<Option<String>> {
    let mut stmt = conn.prepare(
        "SELECT profile_id, distance FROM voice_vec
         WHERE embedding MATCH ?1 AND k=2 AND profile_id IN (
             SELECT p.profile_id FROM voice_profiles p
             LEFT JOIN voice_profile_people l ON l.profile_id=p.profile_id
             LEFT JOIN voice_attribution_suspensions s ON s.profile_id=p.profile_id
             WHERE l.profile_id IS NULL AND s.profile_id IS NULL
               AND EXISTS (
                   SELECT 1 FROM voice_profile_sources vps
                   JOIN voice_fingerprint_sources vfs ON vfs.capture_id=vps.capture_id
                   WHERE vps.profile_id=p.profile_id AND vfs.conversation_id=?2
               )
               AND NOT EXISTS (
                   SELECT 1 FROM voice_profile_sources vps
                   JOIN voice_fingerprint_sources vfs ON vfs.capture_id=vps.capture_id
                   WHERE vps.profile_id=p.profile_id AND vfs.conversation_id<>?2
               )
         ) ORDER BY distance",
    )?;
    let mut rows = stmt.query(params![embedding, conversation])?;
    let Some(row) = rows.next()? else {
        return Ok(None);
    };
    let profile: String = row.get(0)?;
    let distance: f32 = row.get(1)?;
    let second_distance: Option<f32> = rows.next()?.map(|row| row.get(1)).transpose()?;
    if distance > UNLINKED_CLUSTER_MAX_DISTANCE
        || second_distance.is_some_and(|second| second - distance < MATCH_MARGIN)
    {
        return Ok(None);
    }
    Ok(Some(profile))
}
