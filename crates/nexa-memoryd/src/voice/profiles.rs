use std::collections::BTreeSet;

use nexa_proto::v1::{ListVoiceProfilesResponse, VoiceGlobalSwitchStatus, VoiceProfile};
use rusqlite::{Connection, OptionalExtension, Row, Transaction, params};
use serde_json::{Value, json};

use crate::db::{
    Error, Result, Store, VOICE_DIMENSIONS, VOICE_MODEL, VectorKind, check_revision, id, new_id,
    now_ms, optional_id, page_limit, person_is_live, vector,
};
use crate::erasure::{Kind, ensure_not_erased};

use super::captures::live_capture;
use super::consent::{revoke_link_consents, write_consent};

pub(super) const PROFILE_FIELDS: &str =
    "p.profile_id, p.revision, COALESCE(l.person_id, ''), COALESCE(l.revision, 0),
     (SELECT count(*) FROM voice_fingerprints f WHERE f.profile_id=p.profile_id),
     s.profile_id IS NOT NULL";
pub(super) const PROFILE_JOINS: &str = "FROM voice_profiles p
     LEFT JOIN voice_profile_people l ON l.profile_id=p.profile_id
     LEFT JOIN voice_attribution_suspensions s ON s.profile_id=p.profile_id";

impl Store {
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

    pub(crate) fn confirm_voice_profile(
        &mut self,
        target: &str,
        fingerprints: &BTreeSet<&str>,
        expected: Option<u64>,
    ) -> Result<Value> {
        optional_id(target)?;
        let tx = self.write()?;
        let mut centroid = vec![0.0; VOICE_DIMENSIONS];
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
        let embedding = vector(&tx, VectorKind::Voice, VOICE_MODEL, &centroid)?;
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
                params![profile_id, VOICE_MODEL],
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

pub(super) fn load_profile(conn: &Connection, profile_id: &str) -> Result<VoiceProfile> {
    let sql = format!("SELECT {PROFILE_FIELDS} {PROFILE_JOINS} WHERE p.profile_id=?1");
    let profile = conn
        .query_row(&sql, [profile_id], |row| profile_row(row, true))
        .optional()?;
    match profile {
        Some(profile) if person_is_live(conn, &profile.linked_person_id)? => Ok(profile),
        _ => Err(Error::NotFound("voice profile unavailable")),
    }
}

pub(super) fn profile_row(row: &Row<'_>, include_person: bool) -> rusqlite::Result<VoiceProfile> {
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
