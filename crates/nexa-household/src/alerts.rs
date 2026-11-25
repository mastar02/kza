//! Alert episodes, per-episode acknowledgements, and voice zone preferences.

use std::collections::HashSet;
use std::time::Duration;

use rusqlite::{OptionalExtension, Row, params};

use crate::intercom::Intercom;
use crate::{Clock, Error, Household, Result, Timestamp, millis, query_all, required};

pub const ALERT_COOLDOWN: Duration = Duration::from_secs(300);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlertId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeLifetime {
    Cooldown(Duration),
    UntilResolved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlertObservation<'a> {
    /// Stable identity of the condition (including its source).
    pub key: &'a str,
    pub priority: AlertPriority,
    pub message: &'a str,
    pub lifetime: EpisodeLifetime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alert {
    pub id: AlertId,
    pub key: String,
    pub priority: AlertPriority,
    pub message: String,
    pub opened_at: Timestamp,
    pub last_seen_at: Timestamp,
    pub expires_at: Option<Timestamp>,
    pub acknowledged_at: Option<Timestamp>,
    pub resolved_at: Option<Timestamp>,
    pub revision: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertUpdate {
    Opened(Alert),
    Escalated(Alert),
    Deduplicated(Alert),
}

impl AlertUpdate {
    pub fn should_announce(&self) -> bool {
        match self {
            Self::Opened(alert) | Self::Escalated(alert) => alert.priority >= AlertPriority::High,
            Self::Deduplicated(_) => false,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AlertPreferences {
    /// None uses the caller's configured default. An empty list mutes alerts.
    pub zones: Option<Vec<String>>,
    pub revision: i64,
}

impl AlertPreferences {
    /// Resolve current zone IDs, omitting zones retired since the last update.
    pub fn selected_zones<'a>(
        &self,
        intercom: &'a Intercom,
        fallback: Option<&str>,
    ) -> Vec<&'a str> {
        match &self.zones {
            None => fallback
                .and_then(|id| intercom.zone(id))
                .map(|zone| zone.id.as_str())
                .into_iter()
                .collect(),
            Some(zones) => zones
                .iter()
                .filter_map(|id| intercom.zone(id).map(|zone| zone.id.as_str()))
                .collect(),
        }
    }
}

const COLUMNS: &str = "id, condition_key, priority, message, opened_at, last_seen_at,
                       expires_at, acknowledged_at, resolved_at, revision";

fn read(row: &Row<'_>) -> rusqlite::Result<Alert> {
    let priority = match row.get::<_, i64>(2)? {
        0 => AlertPriority::Low,
        1 => AlertPriority::Medium,
        2 => AlertPriority::High,
        3 => AlertPriority::Critical,
        other => {
            return Err(rusqlite::Error::FromSqlConversionFailure(
                2,
                rusqlite::types::Type::Integer,
                Box::new(Error::InvalidState(format!("alert priority {other}"))),
            ));
        }
    };
    Ok(Alert {
        id: AlertId(row.get(0)?),
        key: row.get(1)?,
        priority,
        message: row.get(3)?,
        opened_at: row.get(4)?,
        last_seen_at: row.get(5)?,
        expires_at: row.get(6)?,
        acknowledged_at: row.get(7)?,
        resolved_at: row.get(8)?,
        revision: row.get(9)?,
    })
}

fn read_preferences(row: &Row<'_>) -> rusqlite::Result<AlertPreferences> {
    Ok(AlertPreferences {
        zones: crate::json_column(row, 0)?,
        revision: row.get(1)?,
    })
}

impl<C: Clock> Household<C> {
    /// Preserve the current episode until its cooldown or explicit resolution.
    /// Escalation keeps its ID and reopens its acknowledgement.
    pub fn observe_alert(&mut self, observation: AlertObservation<'_>) -> Result<AlertUpdate> {
        required(observation.key, "alert condition key is required")?;
        required(observation.message, "alert message is required")?;
        let now = self.clock.now();
        let expires_at = match observation.lifetime {
            EpisodeLifetime::Cooldown(duration) if millis(duration) > 0 => {
                Some(now.saturating_add(millis(duration)))
            }
            EpisodeLifetime::Cooldown(_) => {
                return Err(Error::InvalidInput("alert cooldown must be positive"));
            }
            EpisodeLifetime::UntilResolved => None,
        };
        self.transaction(|db| {
            db.execute(
                "UPDATE alerts SET resolved_at = expires_at WHERE condition_key = ?1
                 AND resolved_at IS NULL AND expires_at <= ?2",
                params![observation.key, now],
            )?;
            let existing = db.query_row(
                &format!("SELECT {COLUMNS} FROM alerts WHERE condition_key = ?1 AND resolved_at IS NULL"),
                [observation.key], read,
            ).optional()?;
            if let Some(existing) = existing {
                if observation.priority > existing.priority {
                    let alert = db.query_row(
                        &format!("UPDATE alerts SET last_seen_at = max(last_seen_at, ?2), priority = ?3,
                                  message = ?4, acknowledged_at = NULL, revision = revision + 1
                                  WHERE id = ?1 RETURNING {COLUMNS}"),
                        params![existing.id.0, now, observation.priority as i64, observation.message],
                        read,
                    )?;
                    return Ok(AlertUpdate::Escalated(alert));
                }
                let alert = db.query_row(
                    &format!("UPDATE alerts SET last_seen_at = max(last_seen_at, ?2)
                              WHERE id = ?1 RETURNING {COLUMNS}"),
                    params![existing.id.0, now], read,
                )?;
                return Ok(AlertUpdate::Deduplicated(alert));
            }
            let alert = db.query_row(
                &format!("INSERT INTO alerts(condition_key, priority, message, opened_at, last_seen_at, expires_at)
                          VALUES (?1, ?2, ?3, ?4, ?4, ?5) RETURNING {COLUMNS}"),
                params![observation.key, observation.priority as i64, observation.message, now, expires_at], read,
            )?;
            Ok(AlertUpdate::Opened(alert))
        })
    }

    pub fn alert(&self, id: AlertId) -> Result<Option<Alert>> {
        Ok(self
            .db()
            .query_row(
                &format!("SELECT {COLUMNS} FROM alerts WHERE id = ?1"),
                [id.0],
                read,
            )
            .optional()?)
    }

    pub fn active_alerts(&self) -> Result<Vec<Alert>> {
        query_all(
            self.db(),
            &format!(
                "SELECT {COLUMNS} FROM alerts WHERE acknowledged_at IS NULL AND resolved_at IS NULL
                      AND (expires_at IS NULL OR expires_at > ?1) ORDER BY priority DESC, id"
            ),
            [self.clock.now()],
            read,
        )
    }

    pub fn acknowledge_alert(&mut self, id: AlertId, revision: i64) -> Result<bool> {
        Ok(self.db().execute(
            "UPDATE alerts SET acknowledged_at = ?3 WHERE id = ?1 AND revision = ?2
             AND acknowledged_at IS NULL AND resolved_at IS NULL AND (expires_at IS NULL OR expires_at > ?3)",
            params![id.0, revision, self.clock.now()],
        )? == 1)
    }

    pub fn resolve_alert(&mut self, id: AlertId) -> Result<bool> {
        Ok(self.db().execute(
            "UPDATE alerts SET resolved_at = min(coalesce(expires_at, ?2), ?2)
             WHERE id = ?1 AND resolved_at IS NULL",
            params![id.0, self.clock.now()],
        )? == 1)
    }

    pub fn alert_preferences(&self) -> Result<AlertPreferences> {
        Ok(self
            .db()
            .query_row(
                "SELECT zones, revision FROM alert_preferences WHERE singleton = 1",
                [],
                read_preferences,
            )
            .optional()?
            .unwrap_or_default())
    }

    pub fn set_alert_zones(
        &mut self,
        expected_revision: i64,
        zones: Option<&[&str]>,
        intercom: &Intercom,
    ) -> Result<Option<AlertPreferences>> {
        let mut seen = HashSet::new();
        let zones = zones
            .map(|zones| {
                zones
                    .iter()
                    .copied()
                    .filter(|id| seen.insert(*id))
                    .map(|id| {
                        intercom
                            .zone(id)
                            .ok_or(Error::NotFound("alert zone"))
                            .map(|zone| &zone.id)
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let serialized = serde_json::to_string(&zones)?;
        self.transaction(|db| {
            let revision: i64 = db.query_row(
                "SELECT revision FROM alert_preferences WHERE singleton = 1", [], |row| row.get(0),
            ).optional()?.unwrap_or_default();
            if revision != expected_revision {
                return Ok(None);
            }
            Ok(Some(db.query_row(
                "INSERT INTO alert_preferences(singleton, zones, revision) VALUES (1, ?1, 1)
                 ON CONFLICT(singleton) DO UPDATE SET zones = excluded.zones, revision = revision + 1
                 RETURNING zones, revision",
                [serialized],
                read_preferences,
            )?))
        })
    }
}
