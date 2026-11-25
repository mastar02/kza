use nexa_proto::v1::{VoiceConsentStatus, VoiceObservationReceipt, VoiceProfile};
use rusqlite::{Connection, OptionalExtension, Row, Transaction, params};

use crate::db::{
    Error, Result, Store, VectorKind, ensure_conversation_live, id, new_id, now_ms, optional_id,
    vector,
};
use crate::erasure::{Kind, ensure_not_erased};

const PROFILE_FIELDS: &str =
    "p.profile_id, p.revision, COALESCE(l.person_id, ''), COALESCE(l.revision, 0),
     (SELECT count(*) FROM voice_fingerprints f WHERE f.profile_id=p.profile_id),
     s.profile_id IS NOT NULL";
const PROFILE_JOINS: &str = "FROM voice_profiles p
     LEFT JOIN voice_profile_people l ON l.profile_id=p.profile_id
     LEFT JOIN voice_attribution_suspensions s ON s.profile_id=p.profile_id";

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
        // Validate before switching journal modes. The single writer serializes
        // admission, eligibility and the commit with pause and erasure.
        let embedding = vector(
            self.read()?,
            VectorKind::Voice,
            observation.model,
            observation.embedding,
        )?;
        if self.ambient_control()?.paused {
            return Err(Error::Precondition("ambient capture paused"));
        }
        self.write_both(|tx| record_observation(tx, &observation, &embedding))
    }

    pub(crate) fn list_voice_profiles(&self, include_person: bool) -> Result<Vec<VoiceProfile>> {
        let sql = format!("SELECT {PROFILE_FIELDS} {PROFILE_JOINS} ORDER BY p.profile_id");
        let mut stmt = self.read()?.prepare(&sql)?;
        let profiles = stmt.query_map([], |row| profile_row(row, include_person))?;
        Ok(profiles.collect::<std::result::Result<_, _>>()?)
    }

    pub(crate) fn link_profile(
        &mut self,
        profile_id: &str,
        person_id: &str,
        expected_profile: u64,
        expected_link: u64,
    ) -> Result<VoiceProfile> {
        id(profile_id)?;
        optional_id(person_id)?;
        let tx = self.write()?;
        let previous = load_profile(&tx, profile_id)?;
        if previous.profile_revision != expected_profile || previous.link_revision != expected_link
        {
            return Err(Error::Conflict("voice profile or link revision conflict"));
        }
        if !person_id.is_empty() {
            ensure_not_erased(&tx, Kind::RetiredUser, person_id)?;
        }
        if previous.linked_person_id != person_id {
            let now = now_ms()?;
            revoke_link_consents(&tx, profile_id, person_id, now)?;
            let person = (!person_id.is_empty()).then_some(person_id);
            tx.execute(
                "INSERT INTO voice_profile_people VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(profile_id) DO UPDATE SET person_id=excluded.person_id,
                 revision=excluded.revision, updated_at_unix_ms=excluded.updated_at_unix_ms",
                params![profile_id, person, previous.link_revision + 1, now],
            )?;
        }
        let result = load_profile(&tx, profile_id)?;
        tx.commit()?;
        Ok(result)
    }

    pub(crate) fn set_consent(&mut self, user: &str, granted: bool) -> Result<VoiceConsentStatus> {
        id(user)?;
        let tx = self.write()?;
        ensure_not_erased(&tx, Kind::RetiredUser, user)?;
        let global = global_enabled(&tx)?;
        let now = now_ms()?;
        let (profile_id, profile_rev, link_rev) = if granted {
            if !global {
                return Err(Error::Precondition("global voice attribution disabled"));
            }
            let binding = tx
                .query_row(
                    "SELECT p.profile_id, p.revision, l.revision FROM voice_profile_people l
                 JOIN voice_profiles p ON p.profile_id=l.profile_id
                 LEFT JOIN voice_attribution_suspensions s ON s.profile_id=p.profile_id
                 WHERE l.person_id=?1 AND s.profile_id IS NULL
                 AND (SELECT count(*) FROM voice_profile_people WHERE person_id=?1)=1",
                    [user],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .optional()?
                .ok_or(Error::Precondition("no eligible voice profile link"))?;
            let (profile_id, profile_rev, link_rev): (String, u64, u64) = binding;
            (Some(profile_id), Some(profile_rev), Some(link_rev))
        } else {
            (None, None, None)
        };
        tx.execute(
            "INSERT INTO voice_attribution_consents VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(person_id) DO UPDATE SET profile_id=excluded.profile_id,
             profile_revision=excluded.profile_revision, link_revision=excluded.link_revision,
             granted=excluded.granted, updated_at_unix_ms=excluded.updated_at_unix_ms",
            params![user, profile_id, profile_rev, link_rev, granted, now],
        )?;
        tx.commit()?;
        Ok(VoiceConsentStatus {
            user_id: user.into(),
            granted,
            global_switch_enabled: global,
            updated_at_unix_ms: now,
        })
    }
}

fn record_observation(
    tx: &Transaction<'_>,
    observation: &Observation<'_>,
    embedding: &[u8],
) -> Result<VoiceObservationReceipt> {
    let utterance = observation.utterance;
    ensure_not_erased(tx, Kind::Utterance, utterance)?;
    let provenance: Option<(String, String, Option<String>)> = tx
        .query_row(
            "SELECT c.conversation_id, u.room_id, c.owner_user_id FROM ambient.utterances u
         JOIN ambient.capture_conversations c ON c.conversation_id=u.conversation_id
         WHERE u.utterance_id=?1",
            [utterance],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    let conversation =
        if let Some((conversation, room, owner)) = provenance {
            if room != observation.room {
                return Err(Error::Conflict("observation room conflict"));
            }
            if let Some(owner) = owner {
                ensure_not_erased(tx, Kind::RetiredUser, &owner)?;
            }
            conversation
        } else {
            // This RPC supplies no conversation ID or account. Preserve a source via
            // a standalone capture conversation, keyed by the supplied utterance ID.
            // A pre-existing conversation under that ID would be ambiguous.
            ensure_conversation_live(tx, utterance)?;
            if tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM ambient.capture_conversations WHERE conversation_id=?1)",
            [utterance], |row| row.get(0),
        )? {
            return Err(Error::Conflict("utterance has no unambiguous capture conversation"));
        }
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
            utterance.into()
        };
    ensure_conversation_live(tx, &conversation)?;
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

fn load_profile(conn: &Connection, profile_id: &str) -> Result<VoiceProfile> {
    let sql = format!("SELECT {PROFILE_FIELDS} {PROFILE_JOINS} WHERE p.profile_id=?1");
    conn.query_row(&sql, [profile_id], |row| profile_row(row, true))
        .optional()?
        .ok_or(Error::NotFound("voice profile unavailable"))
}

fn profile_row(row: &Row<'_>, include_person: bool) -> rusqlite::Result<VoiceProfile> {
    Ok(VoiceProfile {
        profile_id: row.get(0)?,
        profile_revision: row.get(1)?,
        linked_person_id: if include_person {
            row.get(2)?
        } else {
            String::new()
        },
        link_revision: if include_person { row.get(3)? } else { 0 },
        sample_count: row.get(4)?,
        attribution_suspended: row.get(5)?,
    })
}

pub(crate) fn global_enabled(conn: &Connection) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT enabled FROM voice_attribution_control WHERE id=1",
        [],
        |row| row.get(0),
    )?)
}

fn revoke_link_consents(
    tx: &Transaction<'_>,
    profile_id: &str,
    person_id: &str,
    now: u64,
) -> Result<()> {
    tx.execute(
        "UPDATE voice_attribution_consents SET granted=0, profile_id=NULL,
         profile_revision=NULL, link_revision=NULL, updated_at_unix_ms=?1 WHERE profile_id=?2 OR person_id=?3",
        params![now, profile_id, person_id],
    )?;
    Ok(())
}
