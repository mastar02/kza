use std::collections::BTreeSet;

use nexa_proto::v1::{
    ListVoiceProfilesResponse, VoiceConsentStatus, VoiceGlobalSwitchStatus,
    VoiceObservationReceipt, VoiceProfile,
};
use rusqlite::{Connection, OptionalExtension, Row, Transaction, params};
use serde_json::{Map, Value, json};

use crate::db::{
    Error, Result, Store, VectorKind, check_revision, conversation_is_live,
    ensure_conversation_live, id, json_revision, new_id, now_ms, optional_id, page_limit,
    person_is_live, vector,
};
use crate::erasure::{Kind, ensure_not_erased, is_erased};

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

    pub(crate) fn list_voice_profiles(
        &self,
        include_person: bool,
        cursor: &str,
    ) -> Result<ListVoiceProfilesResponse> {
        optional_id(cursor)?;
        let limit = page_limit(0);
        let conn = self.read()?;
        let sql = format!(
            "SELECT {PROFILE_FIELDS} {PROFILE_JOINS} WHERE p.profile_id>?1 ORDER BY p.profile_id"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query([cursor])?;
        let mut profiles: Vec<VoiceProfile> = Vec::with_capacity(limit);
        let mut next_cursor = String::new();
        while let Some(row) = rows.next()? {
            if !person_is_live(conn, &row.get::<_, String>(2)?)? {
                continue;
            }
            if profiles.len() == limit {
                next_cursor = profiles
                    .last()
                    .map(|p| p.profile_id.clone())
                    .unwrap_or_default();
                break;
            }
            profiles.push(profile_row(row, include_person)?);
        }
        Ok(ListVoiceProfilesResponse {
            profiles,
            next_cursor,
        })
    }

    /// Seed the switch from the deployment default before its first revision.
    pub fn configure_voice_attribution(&mut self, enabled: bool) -> Result<()> {
        let tx = self.write()?;
        let previous = global_switch(&tx)?;
        if previous.revision == 0 && enabled != previous.enabled {
            update_global_switch(&tx, enabled)?;
        }
        Ok(tx.commit()?)
    }

    pub(crate) fn voice_global_switch(&self) -> Result<VoiceGlobalSwitchStatus> {
        global_switch(self.read()?)
    }

    pub(crate) fn set_voice_global_switch(
        &mut self,
        enabled: bool,
        expected: u64,
    ) -> Result<VoiceGlobalSwitchStatus> {
        let tx = self.write()?;
        check_revision(Some(expected), global_switch(&tx)?.revision)?;
        update_global_switch(&tx, enabled)?;
        let current = global_switch(&tx)?;
        tx.commit()?;
        Ok(current)
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
            for user in [&previous.linked_person_id, person_id] {
                if !user.is_empty() {
                    write_consent(&tx, user, None, now)?;
                }
            }
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
        let updated = self.update_person_consent(user, granted, &Map::new())?;
        Ok(VoiceConsentStatus {
            user_id: user.into(),
            granted: updated.granted,
            global_switch_enabled: updated.global,
            updated_at_unix_ms: updated.revision,
        })
    }

    pub(crate) fn change_voice_consent(
        &mut self,
        user: &str,
        granted: bool,
        payload: &Map<String, Value>,
    ) -> Result<Value> {
        Ok(self
            .update_person_consent(user, granted, payload)?
            .json(user))
    }

    fn update_person_consent(
        &mut self,
        user: &str,
        granted: bool,
        payload: &Map<String, Value>,
    ) -> Result<PersonVoice> {
        id(user)?;
        let tx = self.write()?;
        let previous = person_voice(&tx, user)?;
        check_revision(
            json_revision(payload, "expected_revision")?,
            previous.revision,
        )?;
        let binding = previous.binding();
        check_revision(
            json_revision(payload, "expected_profile_revision")?,
            binding.map_or(0, |profile| profile.profile_revision),
        )?;
        check_revision(
            json_revision(payload, "expected_link_revision")?,
            binding.map_or(0, |profile| profile.link_revision),
        )?;
        let binding = if granted {
            if !previous.global || previous.locked {
                return Err(Error::Precondition("voice consent unavailable"));
            }
            Some(
                binding
                    .filter(|profile| !profile.attribution_suspended)
                    .ok_or(Error::Precondition("no eligible voice profile link"))?,
            )
        } else {
            None
        };
        write_consent(&tx, user, binding, now_ms()?)?;
        let current = person_voice(&tx, user)?;
        tx.commit()?;
        Ok(current)
    }

    pub(crate) fn voice_mine(&self, user: &str) -> Result<Value> {
        Ok(person_voice(self.read()?, user)?.json(user))
    }

    pub(crate) fn voice_consents(&self, cursor: &str, limit: usize) -> Result<Value> {
        optional_id(cursor)?;
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT person_id FROM (
               SELECT person_id FROM voice_attribution_consents UNION
               SELECT person_id FROM voice_profile_people WHERE person_id IS NOT NULL UNION
               SELECT person_id FROM voice_verification_attempts
             ) WHERE person_id>?1 ORDER BY person_id",
        )?;
        let mut rows = stmt.query([cursor])?;
        let mut consents = Vec::with_capacity(limit);
        let mut last_id = None;
        let mut next = None;
        while let Some(row) = rows.next()? {
            let user: String = row.get(0)?;
            if !person_is_live(conn, &user)? {
                continue;
            }
            if consents.len() == limit {
                next = last_id;
                break;
            }
            consents.push(person_voice(conn, &user)?.json(&user));
            last_id = Some(user);
        }
        Ok(json!({"consents": consents, "next": next, "limit": limit}))
    }

    pub(crate) fn reopen_voice_consent(
        &mut self,
        user: &str,
        expected: Option<u64>,
    ) -> Result<Value> {
        id(user)?;
        let tx = self.write()?;
        let previous = person_voice(&tx, user)?;
        check_revision(expected, previous.revision)?;
        if !previous.locked {
            return Err(Error::Precondition("voice verification is not locked"));
        }
        tx.execute(
            "DELETE FROM voice_verification_attempts WHERE person_id=?1 AND outcome='failed'",
            [user],
        )?;
        // Reopening only clears the person's lock; profile suspensions have a separate authority.
        write_consent(&tx, user, None, now_ms()?)?;
        let current = person_voice(&tx, user)?;
        tx.commit()?;
        Ok(current.json(user))
    }

    pub(crate) fn record_voice_verification(
        &mut self,
        user: &str,
        outcome: &str,
        expected: Option<u64>,
    ) -> Result<Value> {
        id(user)?;
        if !matches!(outcome, "passed" | "failed") {
            return Err(Error::Invalid("invalid verification outcome"));
        }
        let tx = self.write()?;
        let previous = person_voice(&tx, user)?;
        check_revision(expected, previous.revision)?;
        if !previous.global || previous.locked {
            return Err(Error::Precondition("voice verification unavailable"));
        }
        let now = now_ms()?;
        let at: u64 = tx.query_row(
            "SELECT max(?2, COALESCE(max(attempted_at_unix_ms) + 1, 0))
             FROM voice_verification_attempts WHERE person_id=?1",
            params![user, now],
            |row| row.get(0),
        )?;
        tx.execute(
            "INSERT INTO voice_verification_attempts VALUES (?1, ?2, ?3)",
            params![user, at, outcome],
        )?;
        if verification_locked(&tx, user)? {
            write_consent(&tx, user, None, now)?;
        }
        let current = person_voice(&tx, user)?;
        tx.commit()?;
        Ok(
            json!({"recorded": true, "revision": current.revision, "verification_locked": current.locked}),
        )
    }

    pub(crate) fn resume_attribution(
        &mut self,
        profile: &str,
        expected: Option<u64>,
    ) -> Result<Value> {
        id(profile)?;
        let tx = self.write()?;
        let previous = load_profile(&tx, profile)?;
        check_revision(expected, previous.profile_revision)?;
        if previous.attribution_suspended {
            let now = now_ms()?;
            revoke_link_consents(&tx, profile, &previous.linked_person_id, now)?;
            tx.execute(
                "DELETE FROM voice_attribution_suspensions WHERE profile_id=?1",
                [profile],
            )?;
            tx.execute(
                "UPDATE voice_profiles SET revision=revision+1 WHERE profile_id=?1",
                [profile],
            )?;
            tx.execute(
                "INSERT INTO voice_attribution_events VALUES (?1, ?2, 'resume', ?3)",
                params![new_id(&tx)?, profile, now],
            )?;
        }
        let current = load_profile(&tx, profile)?;
        tx.commit()?;
        Ok(json!({"resumed": true, "revision": current.profile_revision}))
    }
}

const CAPTURE_FIELDS: &str =
    "f.capture_id, f.utterance_id, f.encoder_model_key, f.profile_id, f.captured_at_unix_ms,
     u.room_id, u.conversation_id, COALESCE(c.owner_user_id, ''), COALESCE(p.person_id, ''),
     COALESCE(r.decision, 'pending')";
const CAPTURE_JOINS: &str = "FROM voice_fingerprints f
     JOIN ambient.utterances u ON u.utterance_id=f.utterance_id
     JOIN ambient.capture_conversations c ON c.conversation_id=u.conversation_id
     LEFT JOIN voice_profile_people p ON p.profile_id=f.profile_id
     LEFT JOIN voice_capture_reviews r ON r.capture_id=f.capture_id";

struct Capture {
    fingerprint_id: String,
    utterance_id: String,
    model: String,
    profile_id: Option<String>,
    captured_at: u64,
    room: String,
    conversation: String,
    owner: String,
    linked_person: String,
    review: String,
}

impl Capture {
    fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            fingerprint_id: row.get(0)?,
            utterance_id: row.get(1)?,
            model: row.get(2)?,
            profile_id: row.get(3)?,
            captured_at: row.get(4)?,
            room: row.get(5)?,
            conversation: row.get(6)?,
            owner: row.get(7)?,
            linked_person: row.get(8)?,
            review: row.get(9)?,
        })
    }

    fn is_live(&self, conn: &Connection) -> Result<bool> {
        if !conversation_is_live(conn, &self.conversation)?
            || is_erased(conn, Kind::Utterance, &self.utterance_id)?
            || !person_is_live(conn, &self.owner)?
            || !person_is_live(conn, &self.linked_person)?
        {
            return Ok(false);
        }
        let mut stmt = conn.prepare_cached(
            "SELECT s.conversation_id, COALESCE(c.owner_user_id, '')
             FROM voice_fingerprint_sources s
             LEFT JOIN ambient.capture_conversations c USING (conversation_id)
             WHERE s.capture_id=?1",
        )?;
        let mut rows = stmt.query([&self.fingerprint_id])?;
        while let Some(row) = rows.next()? {
            if !conversation_is_live(conn, &row.get::<_, String>(0)?)?
                || !person_is_live(conn, &row.get::<_, String>(1)?)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn review_json(&self) -> Value {
        json!({
            "fingerprint_id": self.fingerprint_id,
            "capture_id": self.fingerprint_id,
            "utterance_id": self.utterance_id,
            "encoder_model_key": self.model,
            "profile_id": self.profile_id,
            "captured_at_unix_ms": self.captured_at,
            "room_id": self.room,
            "review_status": match self.review.as_str() {
                "accept" => "accepted",
                "reject" => "rejected",
                _ => "pending",
            },
        })
    }
}

fn live_capture(conn: &Connection, fingerprint: &str) -> Result<Capture> {
    id(fingerprint)?;
    let sql = format!("SELECT {CAPTURE_FIELDS} {CAPTURE_JOINS} WHERE f.capture_id=?1");
    let capture = conn
        .query_row(&sql, [fingerprint], Capture::from_row)
        .optional()?;
    match capture {
        Some(capture) if capture.is_live(conn)? => Ok(capture),
        _ => Err(Error::NotFound("voice capture unavailable")),
    }
}

impl Store {
    pub(crate) fn voice_capture_review(
        &self,
        fingerprint: &str,
        include_transcript: bool,
    ) -> Result<Value> {
        let conn = self.read()?;
        let capture = live_capture(conn, fingerprint)?;
        let mut review = capture.review_json();
        if include_transcript {
            let preview: String = conn.query_row(
                "SELECT transcript_preview FROM ambient.utterances WHERE utterance_id=?1",
                [&capture.utterance_id],
                |row| row.get(0),
            )?;
            review["transcript_preview"] = json!(preview);
        }
        Ok(review)
    }

    pub(crate) fn evaluation_sample(&self, fingerprint: &str) -> Result<Value> {
        let sample = live_capture(self.read()?, fingerprint)?;
        // The RPC carries no modality or source grant. Expose sample metadata only.
        Ok(json!({
            "fingerprint_id": sample.fingerprint_id,
            "encoder_model_key": sample.model,
            "captured_at_unix_ms": sample.captured_at,
        }))
    }

    pub(crate) fn query_voice_captures(&self, query: &Map<String, Value>) -> Result<Value> {
        for key in query.keys() {
            if !matches!(
                key.as_str(),
                "room" | "room_id" | "cursor" | "limit" | "review_status"
            ) {
                return Err(Error::Invalid("unsupported capture filter"));
            }
        }
        if query.contains_key("room") && query.contains_key("room_id") {
            return Err(Error::Invalid("ambiguous room filter"));
        }
        let room = match query.get("room").or_else(|| query.get("room_id")) {
            None | Some(Value::Null) => "",
            Some(Value::Object(room))
                if room.len() == 2 && room.get("kind").and_then(Value::as_str) == Some("room") =>
            {
                room.get("id")
                    .and_then(Value::as_str)
                    .ok_or(Error::Invalid("invalid room filter"))?
            }
            Some(value) => value
                .as_str()
                .ok_or(Error::Invalid("invalid room filter"))?,
        };
        optional_id(room)?;
        let cursor = query_string(query, "cursor")?;
        optional_id(cursor)?;
        let review = match query_string(query, "review_status")? {
            "" => "",
            "pending" => "pending",
            "accept" | "accepted" => "accept",
            "reject" | "rejected" => "reject",
            _ => return Err(Error::Invalid("invalid capture review status")),
        };
        let limit = match query.get("limit") {
            None | Some(Value::Null) => 0,
            Some(Value::String(value)) => value
                .parse::<u32>()
                .map_err(|_| Error::Invalid("invalid capture limit"))?,
            Some(value) => value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or(Error::Invalid("invalid capture limit"))?,
        };
        let limit = page_limit(limit);
        let conn = self.read()?;
        let sql = format!(
            "SELECT {CAPTURE_FIELDS} {CAPTURE_JOINS}
             WHERE f.capture_id>?1 AND (?2='' OR u.room_id=?2)
             AND (?3='' OR COALESCE(r.decision, 'pending')=?3)
             ORDER BY f.capture_id"
        );
        let mut stmt = conn.prepare(&sql)?;
        let mut rows = stmt.query(params![cursor, room, review])?;
        let mut captures = Vec::with_capacity(limit);
        let mut last_id = None;
        let mut next = None;
        while let Some(row) = rows.next()? {
            let capture = Capture::from_row(row)?;
            if !capture.is_live(conn)? {
                continue;
            }
            if captures.len() == limit {
                next = last_id;
                break;
            }
            captures.push(capture.review_json());
            last_id = Some(capture.fingerprint_id);
        }
        Ok(json!({"captures": captures, "next": next, "limit": limit}))
    }

    pub(crate) fn confirm_voice_profile(
        &mut self,
        target: &str,
        fingerprints: &BTreeSet<&str>,
        expected: Option<u64>,
    ) -> Result<Value> {
        optional_id(target)?;
        let tx = self.write()?;
        let mut centroid = vec![0.0; crate::db::VOICE_DIMENSIONS];
        for fingerprint in fingerprints {
            let capture = live_capture(&tx, fingerprint)?;
            if capture.review == "reject" {
                return Err(Error::Precondition("voice capture rejected"));
            }
            let embedding: Vec<u8> = tx.query_row(
                "SELECT embedding FROM voice_fingerprints WHERE capture_id=?1",
                [fingerprint],
                |row| row.get(0),
            )?;
            let (coordinates, rest) = embedding.as_chunks::<4>();
            if !rest.is_empty() || coordinates.len() != centroid.len() {
                return Err(Error::Schema("invalid voice embedding"));
            }
            for (sum, bytes) in centroid.iter_mut().zip(coordinates) {
                *sum += f32::from_ne_bytes(*bytes);
            }
        }
        let embedding = vector(&tx, VectorKind::Voice, crate::db::VOICE_MODEL, &centroid)?;
        let now = now_ms()?;
        let profile_id = if target.is_empty() {
            check_revision(expected, 0)?;
            // An exact, previously accepted selection is idempotent on retry.
            if let Some(profile) = confirmed_profile_for(&tx, fingerprints)? {
                return Ok(
                    json!({"profile_id": profile.profile_id, "revision": profile.profile_revision}),
                );
            }
            let profile_id = new_id(&tx)?;
            tx.execute(
                "INSERT INTO voice_profiles VALUES (?1, ?2, 1)",
                params![profile_id, crate::db::VOICE_MODEL],
            )?;
            profile_id
        } else {
            let previous = load_profile(&tx, target)?;
            check_revision(expected, previous.profile_revision)?;
            revoke_link_consents(&tx, target, &previous.linked_person_id, now)?;
            tx.execute(
                "UPDATE voice_profiles SET revision=revision+1 WHERE profile_id=?1",
                [target],
            )?;
            tx.execute("DELETE FROM voice_vec WHERE profile_id=?1", [target])?;
            tx.execute(
                "DELETE FROM voice_profile_sources WHERE profile_id=?1",
                [target],
            )?;
            target.into()
        };
        tx.execute(
            "INSERT INTO voice_vec VALUES (?1, ?2)",
            params![profile_id, embedding],
        )?;
        for fingerprint in fingerprints {
            tx.execute(
                "INSERT INTO voice_capture_reviews VALUES (?1, 'accept', ?2)
                 ON CONFLICT(capture_id) DO UPDATE SET decision='accept', reviewed_at_unix_ms=excluded.reviewed_at_unix_ms",
                params![fingerprint, now],
            )?;
            tx.execute(
                "INSERT INTO voice_profile_sources VALUES (?1, ?2)",
                params![profile_id, fingerprint],
            )?;
            tx.execute(
                "UPDATE voice_fingerprints SET profile_id=?1 WHERE capture_id=?2",
                params![profile_id, fingerprint],
            )?;
        }
        let profile = load_profile(&tx, &profile_id)?;
        tx.commit()?;
        Ok(json!({"profile_id": profile.profile_id, "revision": profile.profile_revision}))
    }
}

impl Store {
    pub(crate) fn correct_voice_group(
        &mut self,
        target: &str,
        payload: &Map<String, Value>,
    ) -> Result<Value> {
        optional_id(target)?;
        let action = query_string(payload, "action")?;
        if !matches!(action, "" | "merge" | "split" | "unknown")
            || (!target.is_empty() && matches!(action, "split" | "unknown"))
        {
            return Err(Error::Invalid("invalid voice group correction"));
        }
        let tx = self.write()?;
        let mut profiles = BTreeSet::new();
        if let Some(ids) = payload.get("profile_ids") {
            for profile in voice_ids(ids)? {
                load_profile(&tx, profile)?;
                profiles.insert(profile.to_owned());
            }
        } else {
            let ids = payload
                .get("fingerprint_ids")
                .ok_or(Error::Invalid("missing voice group members"))?;
            for capture in voice_ids(ids)? {
                let profile = live_capture(&tx, capture)?
                    .profile_id
                    .ok_or(Error::Precondition("voice capture has no profile"))?;
                profiles.insert(profile);
            }
        }
        let mut previous_groups = BTreeSet::new();
        for profile in &profiles {
            let mut stmt =
                tx.prepare_cached("SELECT group_id FROM voice_group_members WHERE profile_id=?1")?;
            for group in stmt.query_map([profile], |row| row.get::<_, String>(0))? {
                previous_groups.insert(group?);
            }
        }
        if !target.is_empty() && group_revision(&tx, target)? != 0 {
            previous_groups.insert(target.into());
        }
        let revisions = payload
            .get("expected_revisions")
            .map(|value| {
                value
                    .as_object()
                    .ok_or(Error::Invalid("invalid group revisions"))
            })
            .transpose()?;
        if let Some(revisions) = revisions {
            if revisions.len() != previous_groups.len() {
                return Err(Error::Conflict("voice group revision conflict"));
            }
            for group in &previous_groups {
                let expected = json_revision(revisions, group)?
                    .ok_or(Error::Conflict("voice group revision conflict"))?;
                check_revision(Some(expected), group_revision(&tx, group)?)?;
            }
        }
        if let Some(expected) = json_revision(payload, "expected_revision")? {
            check_revision(Some(expected), group_revision(&tx, target)?)?;
        }

        let new_group = if matches!(action, "" | "merge") {
            Some(if target.is_empty() {
                new_id(&tx)?
            } else {
                target.into()
            })
        } else {
            None
        };
        let mut updated = std::collections::BTreeMap::new();
        if let Some(group) = &new_group {
            tx.execute("INSERT OR IGNORE INTO voice_groups VALUES (?1, 1)", [group])?;
            // A target plus no action replaces its membership. A merge
            // extends it and takes the selected members from other groups.
            if action.is_empty() && !target.is_empty() {
                tx.execute("DELETE FROM voice_group_members WHERE group_id=?1", [group])?;
            }
        }
        for profile in &profiles {
            tx.execute(
                "DELETE FROM voice_group_members WHERE profile_id=?1",
                [profile],
            )?;
            if let Some(group) = &new_group {
                tx.execute(
                    "INSERT INTO voice_group_members VALUES (?1, ?2)",
                    params![group, profile],
                )?;
            } else if action == "split" {
                let group = new_id(&tx)?;
                tx.execute("INSERT INTO voice_groups VALUES (?1, 1)", [&group])?;
                tx.execute(
                    "INSERT INTO voice_group_members VALUES (?1, ?2)",
                    params![group, profile],
                )?;
                updated.insert(group, 1);
            }
        }
        for group in previous_groups {
            tx.execute(
                "UPDATE voice_groups SET revision=revision+1 WHERE group_id=?1",
                [&group],
            )?;
            updated.insert(group.clone(), group_revision(&tx, &group)?);
        }
        if let Some(group) = &new_group {
            updated.insert(group.clone(), group_revision(&tx, group)?);
        }
        tx.commit()?;
        Ok(json!({"updated": profiles.len(), "group_id": new_group, "revisions": updated}))
    }
}

fn group_revision(conn: &Connection, group: &str) -> Result<u64> {
    Ok(conn
        .query_row(
            "SELECT revision FROM voice_groups WHERE group_id=?1",
            [group],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(0))
}

pub(crate) fn voice_ids(value: &Value) -> Result<BTreeSet<&str>> {
    let values = value
        .as_array()
        .filter(|items| (1..=32).contains(&items.len()))
        .ok_or(Error::Invalid("invalid voice selection"))?;
    let mut ids = BTreeSet::new();
    for value in values {
        let item = value
            .as_str()
            .ok_or(Error::Invalid("invalid voice selection"))?;
        id(item)?;
        if !ids.insert(item) {
            return Err(Error::Invalid("duplicate voice selection"));
        }
    }
    Ok(ids)
}

fn confirmed_profile_for(
    conn: &Connection,
    fingerprints: &BTreeSet<&str>,
) -> Result<Option<VoiceProfile>> {
    let Some(first) = fingerprints.first() else {
        return Ok(None);
    };
    let Some(profile_id) = live_capture(conn, first)?.profile_id else {
        return Ok(None);
    };
    let mut stmt = conn.prepare(
        "SELECT s.capture_id FROM voice_profile_sources s
         JOIN voice_fingerprints f USING (capture_id)
         JOIN voice_capture_reviews r USING (capture_id)
         WHERE s.profile_id=?1 AND f.profile_id=?1 AND r.decision='accept' ORDER BY s.capture_id",
    )?;
    let ids = stmt
        .query_map([&profile_id], |row| row.get::<_, String>(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let sources: usize = conn.query_row(
        "SELECT count(*) FROM voice_profile_sources WHERE profile_id=?1",
        [&profile_id],
        |row| row.get(0),
    )?;
    if fingerprints.len() == sources
        && ids
            .iter()
            .map(String::as_str)
            .eq(fingerprints.iter().copied())
    {
        Ok(Some(load_profile(conn, &profile_id)?))
    } else {
        Ok(None)
    }
}

fn query_string<'a>(query: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    match query.get(key) {
        None | Some(Value::Null) => Ok(""),
        Some(value) => value
            .as_str()
            .ok_or(Error::Invalid("invalid capture filter")),
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
    let profile = conn
        .query_row(&sql, [profile_id], |row| profile_row(row, true))
        .optional()?;
    match profile {
        Some(profile) if person_is_live(conn, &profile.linked_person_id)? => Ok(profile),
        _ => Err(Error::NotFound("voice profile unavailable")),
    }
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
    Ok(global_switch(conn)?.enabled)
}

fn global_switch(conn: &Connection) -> Result<VoiceGlobalSwitchStatus> {
    Ok(conn.query_row(
        "SELECT enabled, revision, updated_at_unix_ms FROM voice_attribution_control WHERE id=1",
        [],
        |row| {
            Ok(VoiceGlobalSwitchStatus {
                enabled: row.get(0)?,
                revision: row.get(1)?,
                updated_at_unix_ms: row.get(2)?,
            })
        },
    )?)
}

fn update_global_switch(tx: &Transaction<'_>, enabled: bool) -> Result<()> {
    let now = now_ms()?;
    tx.execute(
        "UPDATE voice_attribution_control SET enabled=?1, revision=revision+1,
         updated_at_unix_ms=?2 WHERE id=1",
        params![enabled, now],
    )?;
    if !enabled {
        tx.execute(
            "UPDATE voice_attribution_consents SET granted=0, profile_id=NULL,
             profile_revision=NULL, link_revision=NULL,
             updated_at_unix_ms=max(updated_at_unix_ms+1, ?1) WHERE granted=1",
            [now],
        )?;
    }
    Ok(())
}

struct PersonVoice {
    profiles: Vec<VoiceProfile>,
    global: bool,
    granted: bool,
    locked: bool,
    revision: u64,
}

impl PersonVoice {
    fn binding(&self) -> Option<&VoiceProfile> {
        match self.profiles.as_slice() {
            [profile] => Some(profile),
            _ => None,
        }
    }

    fn json(&self, user: &str) -> Value {
        let profile = self.binding();
        json!({
            "user_id": user,
            "global_switch_enabled": self.global,
            "granted": self.granted,
            "revision": self.revision,
            "linked": !self.profiles.is_empty(),
            "multiple_links": self.profiles.len() > 1,
            "profile_id": profile.map(|p| &p.profile_id),
            "profile_revision": profile.map(|p| p.profile_revision),
            "link_revision": profile.map(|p| p.link_revision),
            "attribution_suspended": self.profiles.iter().any(|p| p.attribution_suspended),
            "verification_locked": self.locked,
        })
    }
}

fn person_voice(conn: &Connection, user: &str) -> Result<PersonVoice> {
    ensure_not_erased(conn, Kind::RetiredUser, user)?;
    let mut stmt = conn.prepare(&format!(
        "SELECT {PROFILE_FIELDS} {PROFILE_JOINS} WHERE l.person_id=?1"
    ))?;
    let profiles = stmt
        .query_map([user], |row| profile_row(row, true))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    // A profile is unique for this person only when there is a single link.
    let global = global_enabled(conn)?;
    let locked = verification_locked(conn, user)?;
    let revision = conn
        .query_row(
            "SELECT updated_at_unix_ms FROM voice_attribution_consents WHERE person_id=?1",
            [user],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(0);
    let granted = if let [profile] = profiles.as_slice() {
        global
            && !locked
            && !profile.attribution_suspended
            && conn.query_row(
                "SELECT EXISTS(
                SELECT 1 FROM voice_attribution_consents c
                JOIN voice_profiles p ON p.profile_id=c.profile_id
                JOIN voice_profile_people l ON l.profile_id=p.profile_id AND l.person_id=c.person_id
                WHERE c.person_id=?1 AND c.profile_id=?2 AND c.granted=1
                AND c.profile_revision=p.revision AND c.link_revision=l.revision
             )",
                params![user, profile.profile_id],
                |row| row.get::<_, bool>(0),
            )?
    } else {
        false
    };
    Ok(PersonVoice {
        profiles,
        global,
        granted,
        locked,
        revision,
    })
}

fn verification_locked(conn: &Connection, user: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS (
            SELECT 1 FROM (
                SELECT attempted_at_unix_ms,
                       lag(attempted_at_unix_ms, 2) OVER (ORDER BY attempted_at_unix_ms) AS first
                FROM voice_verification_attempts WHERE person_id=?1 AND outcome='failed'
            ) WHERE attempted_at_unix_ms - first <= 86400000
         )",
        [user],
        |row| row.get(0),
    )?)
}

fn write_consent(
    tx: &Transaction<'_>,
    user: &str,
    binding: Option<&VoiceProfile>,
    now: u64,
) -> Result<()> {
    // The existing timestamp is the opaque consent CAS token. It increments even
    // when two updates reach SQLite in the same millisecond or the clock moves back.
    tx.execute(
        "INSERT INTO voice_attribution_consents VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(person_id) DO UPDATE SET profile_id=excluded.profile_id,
         profile_revision=excluded.profile_revision, link_revision=excluded.link_revision,
         granted=excluded.granted, updated_at_unix_ms=max(updated_at_unix_ms+1, excluded.updated_at_unix_ms)",
        params![
            user,
            binding.map(|profile| &profile.profile_id),
            binding.map(|profile| profile.profile_revision),
            binding.map(|profile| profile.link_revision),
            binding.is_some(),
            now,
        ],
    )?;
    Ok(())
}

fn revoke_link_consents(
    tx: &Transaction<'_>,
    profile_id: &str,
    person_id: &str,
    now: u64,
) -> Result<()> {
    tx.execute(
        "UPDATE voice_attribution_consents SET granted=0, profile_id=NULL,
         profile_revision=NULL, link_revision=NULL, updated_at_unix_ms=max(updated_at_unix_ms+1, ?1)
         WHERE profile_id=?2 AND person_id<>?3",
        params![now, profile_id, person_id],
    )?;
    if !person_id.is_empty() {
        write_consent(tx, person_id, None, now)?;
    }
    Ok(())
}
