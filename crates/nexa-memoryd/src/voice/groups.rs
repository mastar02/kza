use std::collections::BTreeSet;

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Map, Value, json};

use crate::db::{Error, Result, Store, check_revision, id, json_revision, new_id, optional_id};

use super::captures::live_capture;
use super::profiles::load_profile;
use super::query_string;

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
