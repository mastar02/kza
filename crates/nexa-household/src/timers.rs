//! Named timers and their ring-stop window.

use std::time::Duration;

use rusqlite::{OptionalExtension, Row, params};

use crate::{Clock, Error, Household, Result, Timestamp, millis, name_key, query_all, required};

pub const RING_STOP_WINDOW: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimerId(pub i64);

sql_enum! {
    pub enum TimerState {
        Running => "running",
        Paused => "paused",
        Completed => "completed",
        Cancelled => "cancelled",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timer {
    pub id: TimerId,
    pub name: String,
    pub room: String,
    pub state: TimerState,
    pub deadline: Option<Timestamp>,
    pub paused_remaining: Option<Duration>,
    pub ring_until: Option<Timestamp>,
}

impl Timer {
    pub fn remaining(&self, now: Timestamp) -> Duration {
        if let Some(deadline) = self.deadline {
            return Duration::from_millis(deadline.saturating_sub(now).max(0) as u64);
        }
        self.paused_remaining.unwrap_or_default()
    }
}

const COLUMNS: &str = "id, name, room, state, deadline, remaining_ms, ring_until";

fn read(row: &Row<'_>) -> rusqlite::Result<Timer> {
    Ok(Timer {
        id: TimerId(row.get(0)?),
        name: row.get(1)?,
        room: row.get(2)?,
        state: row.get(3)?,
        deadline: row.get(4)?,
        paused_remaining: row.get::<_, Option<u64>>(5)?.map(Duration::from_millis),
        ring_until: row.get(6)?,
    })
}

impl<C: Clock> Household<C> {
    pub fn create_timer(&mut self, name: &str, room: &str, duration: Duration) -> Result<Timer> {
        required(name, "timer name is required")?;
        required(room, "timer room is required")?;
        let duration = millis(duration);
        if duration == 0 {
            return Err(Error::InvalidInput(
                "timer duration must be at least one millisecond",
            ));
        }
        let deadline = self.clock.now().saturating_add(duration);
        self.transaction(|db| {
            Ok(db.query_row(
                &format!(
                    "INSERT INTO timers(name, name_key, room, state, deadline)
                      VALUES (?1, ?2, ?3, 'running', ?4) RETURNING {COLUMNS}"
                ),
                params![name.trim(), name_key(name), room, deadline],
                read,
            )?)
        })
    }

    pub fn timer(&self, id: TimerId) -> Result<Option<Timer>> {
        Ok(self
            .db()
            .query_row(
                &format!("SELECT {COLUMNS} FROM timers WHERE id = ?1"),
                [id.0],
                read,
            )
            .optional()?)
    }

    pub fn active_timers(&self, room: &str) -> Result<Vec<Timer>> {
        query_all(
            self.db(),
            &format!(
                "SELECT {COLUMNS} FROM timers WHERE room = ?1
                      AND state IN ('running', 'paused') ORDER BY id"
            ),
            [room],
            read,
        )
    }

    pub fn find_active_timer(&self, name: &str, room: &str) -> Result<Option<Timer>> {
        Ok(self
            .db()
            .query_row(
                &format!(
                    "SELECT {COLUMNS} FROM timers WHERE room = ?1 AND name_key = ?2
                      AND state IN ('running', 'paused')"
                ),
                params![room, name_key(name)],
                read,
            )
            .optional()?)
    }

    pub fn pause_timer(&mut self, id: TimerId) -> Result<bool> {
        let now = self.clock.now();
        Ok(self.db().execute(
            "UPDATE timers SET state = 'paused', remaining_ms = min(deadline - ?2, ?3), deadline = NULL
             WHERE id = ?1 AND state = 'running' AND deadline > ?2",
            params![id.0, now, i64::MAX],
        )? == 1)
    }

    pub fn resume_timer(&mut self, id: TimerId) -> Result<bool> {
        let now = self.clock.now();
        Ok(self.db().execute(
            "UPDATE timers SET state = 'running', deadline = min(?2 + remaining_ms, ?3), remaining_ms = NULL
             WHERE id = ?1 AND state = 'paused'",
            params![id.0, now, i64::MAX],
        )? == 1)
    }

    pub fn cancel_timer(&mut self, id: TimerId) -> Result<bool> {
        Ok(self.db().execute(
            "UPDATE timers SET state = 'cancelled', deadline = NULL, remaining_ms = NULL
             WHERE id = ?1 AND (state = 'paused' OR (state = 'running' AND deadline > ?2))",
            params![id.0, self.clock.now()],
        )? == 1)
    }

    /// Complete due timers once and open their stop windows when observed.
    pub fn complete_due_timers(&mut self) -> Result<Vec<Timer>> {
        let now = self.clock.now();
        self.transaction(|db| {
            let mut completed = query_all(
                db,
                &format!(
                    "UPDATE timers SET state = 'completed', deadline = NULL, ring_until = ?2
                          WHERE state = 'running' AND deadline <= ?1 RETURNING {COLUMNS}"
                ),
                params![now, now.saturating_add(millis(RING_STOP_WINDOW))],
                read,
            )?;
            completed.sort_unstable_by_key(|timer| timer.id.0);
            Ok(completed)
        })
    }

    pub fn ringing_timers(&self, room: &str) -> Result<Vec<Timer>> {
        query_all(
            self.db(),
            &format!(
                "SELECT {COLUMNS} FROM timers WHERE room = ?1 AND ring_until > ?2 ORDER BY id"
            ),
            params![room, self.clock.now()],
            read,
        )
    }

    /// Admit a stop request for the specified ring in its originating room.
    pub fn stop_timer_ring(&mut self, id: TimerId, room: &str) -> Result<bool> {
        Ok(self.db().execute(
            "UPDATE timers SET ring_until = NULL WHERE id = ?1 AND room = ?2 AND ring_until > ?3",
            params![id.0, room, self.clock.now()],
        )? == 1)
    }
}
