use nexa_proto::v1::VoicePersonality;
use rusqlite::{Connection, OptionalExtension, params};

use crate::db::{Error, Result, Store, ensure_conversation_live, id, json_object, now_ms};

const MAX_SUMMARY_BYTES: usize = 16 * 1024;

impl Store {
    pub(crate) fn upsert_voice_personality(
        &mut self,
        profile_id: &str,
        summary_json: &str,
        source_conversation_id: &str,
    ) -> Result<VoicePersonality> {
        id(profile_id)?;
        id(source_conversation_id)?;
        if summary_json.len() > MAX_SUMMARY_BYTES {
            return Err(Error::Invalid("voice personality summary too large"));
        }
        json_object(summary_json)?;
        let tx = self.write()?;
        ensure_conversation_live(&tx, source_conversation_id)?;
        require_consented_profile(&tx, profile_id)?;
        let now = now_ms()?;
        tx.execute(
            "INSERT INTO voice_personalities VALUES (?1, ?2, ?3, 0)
             ON CONFLICT(profile_id) DO UPDATE SET
                 summary_json=excluded.summary_json,
                 updated_at_unix_ms=excluded.updated_at_unix_ms",
            params![profile_id, summary_json, now],
        )?;
        tx.execute(
            "INSERT OR IGNORE INTO voice_personality_sources VALUES (?1, ?2)",
            params![profile_id, source_conversation_id],
        )?;
        tx.execute(
            "UPDATE voice_personalities
             SET source_count=(SELECT count(*) FROM voice_personality_sources WHERE profile_id=?1)
             WHERE profile_id=?1",
            [profile_id],
        )?;
        let record = load_personality(&tx, profile_id)?;
        tx.commit()?;
        Ok(record)
    }

    pub(crate) fn get_voice_personality(&self, profile_id: &str) -> Result<VoicePersonality> {
        id(profile_id)?;
        let conn = self.read()?;
        if require_consented_profile(conn, profile_id).is_err() {
            return Err(Error::NotFound("voice personality unavailable"));
        }
        load_personality(conn, profile_id)
    }
}

fn require_consented_profile(conn: &Connection, profile_id: &str) -> Result<()> {
    let consented: bool = conn.query_row(
        "SELECT EXISTS(
             SELECT 1 FROM voice_profiles p
             JOIN voice_profile_people l ON l.profile_id=p.profile_id
             JOIN voice_attribution_consents c ON c.person_id=l.person_id AND c.profile_id=p.profile_id
             LEFT JOIN voice_attribution_suspensions s ON s.profile_id=p.profile_id
             WHERE p.profile_id=?1
               AND c.granted=1
               AND c.profile_revision=p.revision
               AND c.link_revision=l.revision
               AND s.profile_id IS NULL
               AND (SELECT count(*) FROM voice_profile_people WHERE person_id=l.person_id)=1
         )",
        [profile_id],
        |row| row.get(0),
    )?;
    if !consented {
        return Err(Error::Precondition(
            "voice profile lacks active attribution consent",
        ));
    }
    Ok(())
}

fn load_personality(conn: &Connection, profile_id: &str) -> Result<VoicePersonality> {
    conn.query_row(
        "SELECT profile_id, summary_json, updated_at_unix_ms, source_count
         FROM voice_personalities WHERE profile_id=?1",
        [profile_id],
        |row| {
            Ok(VoicePersonality {
                profile_id: row.get(0)?,
                summary_json: row.get(1)?,
                updated_unix_ms: row.get(2)?,
                source_count: row.get(3)?,
            })
        },
    )
    .optional()?
    .ok_or(Error::NotFound("voice personality unavailable"))
}
