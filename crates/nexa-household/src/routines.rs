//! Routine definitions, occurrence claims, and durable step execution.

use std::time::Duration;

use chrono::{DateTime, Datelike, NaiveTime, Offset, TimeZone};
use chrono_tz::Tz;
use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};

use crate::{
    Clock, Error, HomeAction, Household, HouseholdStore, Result, Timestamp, decode_json,
    json_column, millis, name_key, query_all, required,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutineId(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunId(pub i64);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RoutineStep {
    Home { action: HomeAction },
    Announce { room: String, message: String },
    Delay { duration: Duration },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutineSpec {
    pub name: String,
    pub enabled: bool,
    pub steps: Vec<RoutineStep>,
}

impl RoutineSpec {
    pub fn with_schedule(self, schedule: RoutineSchedule) -> ScheduledRoutineSpec {
        ScheduledRoutineSpec {
            spec: self,
            schedule: Some(schedule),
        }
    }

    fn validated_steps(&self) -> Result<String> {
        required(&self.name, "routine name is required")?;
        if self.steps.is_empty() {
            return Err(Error::InvalidInput("routine requires at least one step"));
        }
        for step in &self.steps {
            match step {
                RoutineStep::Home { action } => action.validate()?,
                RoutineStep::Announce { message, room } => {
                    required(message, "announcement message is required")?;
                    required(room, "announcement room is required")?;
                }
                RoutineStep::Delay { .. } => (),
            }
        }
        Ok(serde_json::to_string(&self.steps)?)
    }
}

/// A routine definition with an optional schedule. Existing `RoutineSpec`
/// literals remain valid for callers that only manage the name and steps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduledRoutineSpec {
    #[serde(flatten)]
    pub spec: RoutineSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<RoutineSchedule>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoutineSchedule {
    Once {
        at_ms: Timestamp,
    },
    Daily {
        hour: u32,
        minute: u32,
        /// An IANA time zone name.
        timezone: String,
        /// ISO weekdays, Monday = 1. An empty list means every day.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        days: Vec<u32>,
    },
}

impl RoutineSchedule {
    fn canonical_json(&self) -> Result<String> {
        let mut schedule = self.clone();
        if let Self::Daily { days, .. } = &mut schedule {
            days.sort_unstable();
            days.dedup();
        }
        Ok(serde_json::to_string(&schedule)?)
    }

    /// Find the first occurrence at or after the given instant. Skip a local
    /// time in a DST gap and use the first instant in a repeated local hour.
    fn next_at_or_after(&self, now: Timestamp) -> Result<Option<Timestamp>> {
        let (hour, minute, timezone, days) = match self {
            Self::Once { at_ms } => return Ok((*at_ms >= now).then_some(*at_ms)),
            Self::Daily {
                hour,
                minute,
                timezone,
                days,
            } => (hour, minute, timezone, days),
        };
        let time = NaiveTime::from_hms_opt(*hour, *minute, 0)
            .ok_or(Error::InvalidInput("invalid routine schedule time"))?;
        let zone: Tz = timezone
            .parse()
            .map_err(|_| Error::InvalidInput("invalid routine schedule timezone"))?;
        if days.iter().any(|day| !(1..=7).contains(day)) {
            return Err(Error::InvalidInput("invalid routine schedule weekday"));
        }
        let local = DateTime::from_timestamp_millis(now)
            .ok_or(Error::InvalidInput("routine clock is out of range"))?
            .with_timezone(&zone);
        let mut date = local
            .naive_utc()
            .checked_add_offset(local.offset().fix())
            .ok_or(Error::InvalidInput("routine local date is out of range"))?
            .date();
        loop {
            if (days.is_empty() || days.contains(&date.weekday().number_from_monday()))
                && let Some(at) = zone.from_local_datetime(&date.and_time(time)).earliest()
                && at.timestamp_millis() >= now
            {
                return Ok(Some(at.timestamp_millis()));
            }
            let Some(next) = date.succ_opt() else {
                return Ok(None);
            };
            date = next;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Routine {
    pub id: RoutineId,
    pub revision: i64,
    pub spec: RoutineSpec,
    pub updated_at_ms: Timestamp,
    pub schedule: Option<RoutineSchedule>,
    pub next_due_at_ms: Option<Timestamp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occurrence<'a> {
    Scheduled(Timestamp),
    /// A stable turn ID, common to all replays of the same voice request.
    Voice(&'a str),
    Manual(&'a str),
}

sql_enum! {
    pub enum RunSource {
        Scheduled => "scheduled",
        Voice => "voice",
        Manual => "manual",
    }
}

sql_enum! {
    pub enum RunState {
        Running => "running",
        Succeeded => "succeeded",
        Failed => "failed",
        Stopped => "stopped",
        Skipped => "skipped",
        Interrupted => "interrupted",
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutineRun {
    pub id: RunId,
    pub routine: RoutineId,
    pub source: RunSource,
    pub occurrence_key: String,
    pub state: RunState,
    pub started_at: Timestamp,
    pub finished_at: Option<Timestamp>,
    pub next_step: usize,
    pub steps: Vec<RoutineStep>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunClaim {
    Created(RoutineRun),
    Existing(RoutineRun),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutineEffect {
    Home(HomeAction),
    Announce { room: String, message: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepClaim {
    pub run: RunId,
    pub position: usize,
    pub effect: RoutineEffect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    Succeeded,
    Failed,
    Uncertain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceBrake {
    Stop,
    Disable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrakeReceipt {
    pub routine: RoutineId,
    pub stopped_run: Option<RunId>,
    pub disabled: bool,
}

const ROUTINE_COLUMNS: &str =
    "id, revision, name, enabled, steps, updated_at_ms, schedule_json, next_due_at_ms";
const RUN_COLUMNS: &str =
    "id, routine_id, source, occurrence_key, status, started_at, finished_at, next_step, steps";

fn read_routine(row: &Row<'_>) -> rusqlite::Result<Routine> {
    Ok(Routine {
        id: RoutineId(row.get(0)?),
        revision: row.get(1)?,
        spec: RoutineSpec {
            name: row.get(2)?,
            enabled: row.get(3)?,
            steps: json_column(row, 4)?,
        },
        updated_at_ms: row.get(5)?,
        schedule: row
            .get::<_, Option<String>>(6)?
            .map(|json| decode_json(&json, 6))
            .transpose()?,
        next_due_at_ms: row.get(7)?,
    })
}

fn read_run(row: &Row<'_>) -> rusqlite::Result<RoutineRun> {
    Ok(RoutineRun {
        id: RunId(row.get(0)?),
        routine: RoutineId(row.get(1)?),
        source: row.get(2)?,
        occurrence_key: row.get(3)?,
        state: row.get(4)?,
        started_at: row.get(5)?,
        finished_at: row.get(6)?,
        next_step: row.get(7)?,
        steps: json_column(row, 8)?,
    })
}

fn stop_running(db: &Connection, id: RoutineId, now: Timestamp) -> Result<Option<RunId>> {
    Ok(db.query_row(
        "UPDATE routine_runs SET status = 'stopped', finished_at = ?2, step_claimed = 0, step_due_at = NULL
         WHERE routine_id = ?1 AND status = 'running' RETURNING id",
        params![id.0, now],
        |row| row.get::<_, i64>(0).map(RunId),
    ).optional()?)
}

fn next_unclaimed_due(
    db: &Connection,
    id: RoutineId,
    schedule: &RoutineSchedule,
    now: Timestamp,
) -> Result<Option<Timestamp>> {
    let mut next = schedule.next_at_or_after(now)?;
    while let Some(at) = next {
        let claimed: bool = db.query_row(
            "SELECT EXISTS (SELECT 1 FROM routine_runs WHERE routine_id = ?1 AND occurrence_key = ?2)",
            params![id.0, format!("scheduled:{at}")],
            |row| row.get(0),
        )?;
        if !claimed {
            break;
        }
        next = next_due_after(schedule, at)?;
    }
    Ok(next)
}

fn next_due_after(schedule: &RoutineSchedule, at: Timestamp) -> Result<Option<Timestamp>> {
    match at.checked_add(1) {
        Some(next) => schedule.next_at_or_after(next),
        None => Ok(None),
    }
}

fn advance_schedule(db: &Connection, id: RoutineId, at: Timestamp, now: Timestamp) -> Result<()> {
    let schedule: Option<RoutineSchedule> = db
        .query_row(
            "SELECT schedule_json FROM routines WHERE id = ?1 AND next_due_at_ms = ?2",
            params![id.0, at],
            |row| json_column(row, 0),
        )
        .optional()?;
    if let Some(schedule) = schedule {
        let next = next_due_after(&schedule, now)?;
        db.execute(
            "UPDATE routines SET next_due_at_ms = ?2 WHERE id = ?1",
            params![id.0, next],
        )?;
    }
    Ok(())
}

fn claim_occurrence(
    db: &Connection,
    id: RoutineId,
    occurrence: Occurrence<'_>,
    expected_revision: Option<i64>,
    now: Timestamp,
) -> Result<RunClaim> {
    let (source, key) = match occurrence {
        Occurrence::Scheduled(at) if at <= now => (RunSource::Scheduled, format!("scheduled:{at}")),
        Occurrence::Scheduled(_) => {
            return Err(Error::InvalidInput("scheduled occurrence is in the future"));
        }
        Occurrence::Voice(turn) => {
            required(turn, "voice occurrence requires a turn ID")?;
            (RunSource::Voice, format!("voice:{turn}"))
        }
        Occurrence::Manual(request) => {
            required(request, "manual occurrence requires a request ID")?;
            (RunSource::Manual, format!("manual:{request}"))
        }
    };
    if let Occurrence::Scheduled(at) = occurrence {
        advance_schedule(db, id, at, now)?;
    }
    let existing = db
        .query_row(
            &format!(
                "SELECT {RUN_COLUMNS} FROM routine_runs
                  WHERE occurrence_key = ?2 AND (routine_id = ?1 OR source = 'voice')"
            ),
            params![id.0, key],
            read_run,
        )
        .optional()?;
    if let Some(existing) = existing {
        if existing.routine != id {
            return Err(Error::Conflict(
                "voice turn already selected a different routine",
            ));
        }
        return Ok(RunClaim::Existing(existing));
    }
    let (enabled, revision, steps): (bool, i64, String) = db
        .query_row(
            "SELECT enabled, revision, steps FROM routines WHERE id = ?1 AND deleted_at IS NULL",
            [id.0],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?
        .ok_or(Error::NotFound("routine"))?;
    if !enabled {
        return Err(Error::Conflict("routine is disabled"));
    }
    if expected_revision.is_some_and(|expected| expected != revision) {
        return Err(Error::Conflict("routine revision changed"));
    }
    let running: bool = db.query_row(
        "SELECT EXISTS (SELECT 1 FROM routine_runs WHERE routine_id = ?1 AND status = 'running')",
        [id.0],
        |row| row.get(0),
    )?;
    let (status, finished_at) = match (running, source) {
        (false, _) => (RunState::Running, None),
        (true, RunSource::Scheduled) => (RunState::Skipped, Some(now)),
        (true, _) => return Err(Error::Conflict("routine is already running")),
    };
    let run = db.query_row(
        &format!("INSERT INTO routine_runs(routine_id, occurrence_key, source, started_at, finished_at, status, steps)
                  VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) RETURNING {RUN_COLUMNS}"),
        params![id.0, key, source, now, finished_at, status, steps], read_run,
    )?;
    Ok(RunClaim::Created(run))
}

impl HouseholdStore {
    /// Claim one outstanding occurrence per enabled routine and advance its
    /// schedule in the same transaction. Missed daily slots are coalesced,
    /// so the next due time is strictly after the poll time.
    ///
    /// Returns newly written receipts. Only running receipts admit execution.
    pub fn poll_due_scheduled_routines(&mut self, now_ms: Timestamp) -> Result<Vec<RoutineRun>> {
        let tx = self.immediate()?;
        let due = query_all(
            &tx,
            "SELECT id, next_due_at_ms FROM routines
             WHERE enabled = 1 AND deleted_at IS NULL AND next_due_at_ms <= ?1
             ORDER BY next_due_at_ms, id",
            [now_ms],
            |row| Ok((RoutineId(row.get(0)?), row.get(1)?)),
        )?;
        let mut runs = Vec::with_capacity(due.len());
        for (id, at) in due {
            if let RunClaim::Created(run) =
                claim_occurrence(&tx, id, Occurrence::Scheduled(at), None, now_ms)?
            {
                runs.push(run);
            }
        }
        tx.commit()?;
        Ok(runs)
    }
}

impl<C: Clock> Household<C> {
    pub fn create_routine(&mut self, spec: &RoutineSpec) -> Result<Routine> {
        self.create_routine_definition(spec, None)
    }

    pub fn create_scheduled_routine(
        &mut self,
        definition: &ScheduledRoutineSpec,
    ) -> Result<Routine> {
        self.create_routine_definition(&definition.spec, definition.schedule.as_ref())
    }

    fn create_routine_definition(
        &mut self,
        spec: &RoutineSpec,
        schedule: Option<&RoutineSchedule>,
    ) -> Result<Routine> {
        let steps = spec.validated_steps()?;
        let now = self.clock.now();
        let mut next = schedule
            .map(|schedule| schedule.next_at_or_after(now))
            .transpose()?
            .flatten();
        if !spec.enabled {
            next = None;
        }
        let schedule_json = schedule.map(RoutineSchedule::canonical_json).transpose()?;
        self.transaction(|db| {
            Ok(db.query_row(
                &format!(
                    "INSERT INTO routines(name, name_key, enabled, revision, steps, updated_at_ms, schedule_json, next_due_at_ms)
                      VALUES (?1, ?2, ?3, 1, ?4, ?5, ?6, ?7) RETURNING {ROUTINE_COLUMNS}"
                ),
                params![spec.name.trim(), name_key(&spec.name), spec.enabled, steps, now, schedule_json, next],
                read_routine,
            )?)
        })
    }

    pub fn routine(&self, id: RoutineId) -> Result<Option<Routine>> {
        Ok(self
            .db()
            .query_row(
                &format!(
                    "SELECT {ROUTINE_COLUMNS} FROM routines WHERE id = ?1 AND deleted_at IS NULL"
                ),
                [id.0],
                read_routine,
            )
            .optional()?)
    }

    pub fn routines(&self) -> Result<Vec<Routine>> {
        query_all(
            self.db(),
            &format!("SELECT {ROUTINE_COLUMNS} FROM routines WHERE deleted_at IS NULL ORDER BY id"),
            [],
            read_routine,
        )
    }

    pub fn update_routine(
        &mut self,
        id: RoutineId,
        revision: i64,
        spec: &RoutineSpec,
    ) -> Result<Option<Routine>> {
        self.update_routine_definition(id, revision, |routine| (spec.clone(), routine.schedule))
    }

    /// Replace the definition and its schedule. A missing schedule clears it.
    pub fn update_scheduled_routine(
        &mut self,
        id: RoutineId,
        revision: i64,
        definition: &ScheduledRoutineSpec,
    ) -> Result<Option<Routine>> {
        self.update_routine_definition(id, revision, |_| {
            (definition.spec.clone(), definition.schedule.clone())
        })
    }

    pub fn set_routine_enabled(
        &mut self,
        id: RoutineId,
        revision: i64,
        enabled: bool,
    ) -> Result<Option<Routine>> {
        self.update_routine_definition(id, revision, |mut routine| {
            routine.spec.enabled = enabled;
            (routine.spec, routine.schedule)
        })
    }

    fn update_routine_definition(
        &mut self,
        id: RoutineId,
        revision: i64,
        edit: impl FnOnce(Routine) -> (RoutineSpec, Option<RoutineSchedule>),
    ) -> Result<Option<Routine>> {
        let now = self.clock.now();
        self.transaction(|db| {
            let current = db.query_row(
                &format!("SELECT {ROUTINE_COLUMNS} FROM routines
                          WHERE id = ?1 AND revision = ?2 AND deleted_at IS NULL"),
                params![id.0, revision],
                read_routine,
            ).optional()?;
            let Some(current) = current else {
                return Ok(None);
            };
            let (spec, schedule) = edit(current);
            let steps = spec.validated_steps()?;
            let next = match &schedule {
                Some(schedule) if spec.enabled => next_unclaimed_due(db, id, schedule, now)?,
                Some(schedule) => {
                    schedule.next_at_or_after(now)?;
                    None
                }
                None => None,
            };
            let schedule_json = schedule.as_ref().map(RoutineSchedule::canonical_json).transpose()?;
            let routine = db.query_row(
                &format!("UPDATE routines SET name = ?2, name_key = ?3, enabled = ?4, steps = ?5, revision = revision + 1,
                              updated_at_ms = ?6, schedule_json = ?7, next_due_at_ms = ?8
                          WHERE id = ?1 RETURNING {ROUTINE_COLUMNS}"),
                params![id.0, spec.name.trim(), name_key(&spec.name), spec.enabled, steps, now, schedule_json, next], read_routine,
            )?;
            if !spec.enabled {
                stop_running(db, id, now)?;
            }
            Ok(Some(routine))
        })
    }

    /// Remove the definition from use while retaining historical run receipts.
    pub fn delete_routine(&mut self, id: RoutineId, revision: i64) -> Result<bool> {
        let now = self.clock.now();
        self.transaction(|db| {
            let changed = db.execute(
                "UPDATE routines SET deleted_at = ?3, enabled = 0, revision = revision + 1,
                     updated_at_ms = ?3, next_due_at_ms = NULL
                 WHERE id = ?1 AND revision = ?2 AND deleted_at IS NULL",
                params![id.0, revision, now],
            )? == 1;
            if changed {
                stop_running(db, id, now)?;
            }
            Ok(changed)
        })
    }

    pub fn routine_run(&self, id: RunId) -> Result<Option<RoutineRun>> {
        Ok(self
            .db()
            .query_row(
                &format!("SELECT {RUN_COLUMNS} FROM routine_runs WHERE id = ?1"),
                [id.0],
                read_run,
            )
            .optional()?)
    }

    pub fn running_routines(&self) -> Result<Vec<RoutineRun>> {
        query_all(
            self.db(),
            &format!("SELECT {RUN_COLUMNS} FROM routine_runs WHERE status = 'running' ORDER BY id"),
            [],
            read_run,
        )
    }

    pub fn poll_due_scheduled_routines(&mut self, now_ms: Timestamp) -> Result<Vec<RoutineRun>> {
        self.store.poll_due_scheduled_routines(now_ms)
    }

    /// Only a newly created, running claim admits execution for an occurrence.
    /// The transaction binds a snapshot of the current definition to the run.
    pub fn claim_routine(
        &mut self,
        id: RoutineId,
        occurrence: Occurrence<'_>,
        expected_revision: Option<i64>,
    ) -> Result<RunClaim> {
        let now = self.clock.now();
        self.transaction(|db| claim_occurrence(db, id, occurrence, expected_revision, now))
    }

    /// Claim the next effect, or arm a durable delay. Repeated polls while a
    /// delay or claimed step is in progress have no external effect.
    pub fn claim_routine_step(&mut self, id: RunId) -> Result<Option<StepClaim>> {
        let now = self.clock.now();
        self.transaction(|db| {
            let progress: Option<(Vec<RoutineStep>, usize, Option<Timestamp>)> = db.query_row(
                "SELECT steps, next_step, step_due_at FROM routine_runs
                 WHERE id = ?1 AND status = 'running' AND step_claimed = 0",
                [id.0], |row| Ok((json_column(row, 0)?, row.get(1)?, row.get(2)?)),
            ).optional()?;
            let Some((steps, position, mut due_at)) = progress else {
                return Ok(None);
            };
            for (position, step) in steps.into_iter().enumerate().skip(position) {
                let effect = match step {
                    RoutineStep::Home { action } => RoutineEffect::Home(action),
                    RoutineStep::Announce { room, message } => RoutineEffect::Announce { room, message },
                    RoutineStep::Delay { duration } => {
                        let deadline = due_at.take().unwrap_or_else(|| now.saturating_add(millis(duration)));
                        if deadline > now {
                            db.execute(
                                "UPDATE routine_runs SET next_step = ?2, step_due_at = ?3 WHERE id = ?1",
                                params![id.0, position, deadline],
                            )?;
                            return Ok(None);
                        }
                        continue;
                    }
                };
                db.execute(
                    "UPDATE routine_runs SET next_step = ?2, step_due_at = NULL, step_claimed = 1 WHERE id = ?1",
                    params![id.0, position],
                )?;
                return Ok(Some(StepClaim { run: id, position, effect }));
            }
            db.execute(
                "UPDATE routine_runs SET status = 'succeeded', next_step = json_array_length(steps),
                     step_due_at = NULL, finished_at = ?2 WHERE id = ?1",
                params![id.0, now],
            )?;
            Ok(None)
        })
    }

    pub fn finish_routine_step(&mut self, step: &StepClaim, outcome: StepOutcome) -> Result<bool> {
        let status = match outcome {
            StepOutcome::Succeeded => RunState::Succeeded,
            StepOutcome::Failed => RunState::Failed,
            StepOutcome::Uncertain => RunState::Interrupted,
        };
        Ok(self.db().execute(
            "UPDATE routine_runs SET step_claimed = 0,
                 status = CASE WHEN ?3 = 'succeeded' AND next_step + 1 < json_array_length(steps)
                               THEN 'running' ELSE ?3 END,
                 finished_at = CASE WHEN ?3 = 'succeeded' AND next_step + 1 < json_array_length(steps)
                                    THEN NULL ELSE ?4 END,
                 next_step = next_step + (?3 = 'succeeded')
             WHERE id = ?1 AND next_step = ?2 AND status = 'running' AND step_claimed = 1",
            params![step.run.0, step.position, status, self.clock.now()],
        )? == 1)
    }

    pub fn set_routine_voice_brake(&mut self, enabled: bool) {
        self.routine_voice_brake = enabled;
    }

    /// Stop admitting steps for the named routine. The caller can use the run
    /// ID to interrupt any external effect the adapter can still cancel.
    pub fn brake_routine_by_voice(
        &mut self,
        name: &str,
        brake: VoiceBrake,
    ) -> Result<Option<BrakeReceipt>> {
        if !self.routine_voice_brake {
            return Err(Error::Conflict("routine voice brake is disabled"));
        }
        required(name, "routine brake requires a name")?;
        let now = self.clock.now();
        self.transaction(|db| {
            let selected: Option<(i64, bool)> = db
                .query_row(
                    "SELECT id, enabled FROM routines WHERE name_key = ?1 AND deleted_at IS NULL",
                    [name_key(name)],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            let Some((id, enabled)) = selected else {
                return Ok(None);
            };
            let disabled = brake == VoiceBrake::Disable;
            if disabled && enabled {
                db.execute(
                    "UPDATE routines SET enabled = 0, revision = revision + 1,
                         updated_at_ms = ?2, next_due_at_ms = NULL WHERE id = ?1",
                    params![id, now],
                )?;
            }
            let routine = RoutineId(id);
            let stopped_run = stop_running(db, routine, now)?;
            Ok(Some(BrakeReceipt {
                routine,
                stopped_run,
                disabled,
            }))
        })
    }

    /// Recover from an owner crash after all previous workers have exited.
    /// Runs between steps remain resumable. In-flight effects become interrupted.
    pub fn recover_routines(&mut self) -> Result<usize> {
        Ok(self.db().execute(
            "UPDATE routine_runs SET status = 'interrupted', finished_at = ?1, step_claimed = 0
             WHERE status = 'running' AND step_claimed = 1",
            [self.clock.now()],
        )?)
    }
}
