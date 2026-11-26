use nexa_proto::v1::{VoiceConsentStatus, VoiceProfile};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde_json::{Map, Value, json};

use crate::db::{
    Error, Result, Store, check_revision, id, json_revision, new_id, now_ms, optional_id,
    person_is_live,
};
use crate::erasure::{Kind, ensure_not_erased};

use super::profiles::{PROFILE_FIELDS, PROFILE_JOINS, global_enabled, load_profile, profile_row};

impl Store {
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

pub(super) fn write_consent(
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

pub(super) fn revoke_link_consents(
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
