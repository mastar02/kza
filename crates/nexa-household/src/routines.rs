//! Routine definitions, occurrence claims, and durable step execution.

use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, Row, params};
use serde::{Deserialize, Serialize};

use crate::{
    Clock, Error, HomeAction, Household, Result, Timestamp, json_column, millis, name_key,
    query_all, required,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutineSpec {
    pub name: String,
    pub enabled: bool,
    pub steps: Vec<RoutineStep>,
}

impl RoutineSpec {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Routine {
    pub id: RoutineId,
    pub revision: i64,
    pub spec: RoutineSpec,
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

const ROUTINE_COLUMNS: &str = "id, revision, name, enabled, steps";
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

impl<C: Clock> Household<C> {
    pub fn create_routine(&mut self, spec: &RoutineSpec) -> Result<Routine> {
        let steps = spec.validated_steps()?;
        self.transaction(|db| {
            Ok(db.query_row(
                &format!(
                    "INSERT INTO routines(name, name_key, enabled, revision, steps)
                      VALUES (?1, ?2, ?3, 1, ?4) RETURNING {ROUTINE_COLUMNS}"
                ),
                params![spec.name.trim(), name_key(&spec.name), spec.enabled, steps],
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
        let steps = spec.validated_steps()?;
        let now = self.clock.now();
        self.transaction(|db| {
            let routine = db.query_row(
                &format!("UPDATE routines SET name = ?3, name_key = ?4, enabled = ?5, steps = ?6, revision = revision + 1
                          WHERE id = ?1 AND revision = ?2 AND deleted_at IS NULL RETURNING {ROUTINE_COLUMNS}"),
                params![id.0, revision, spec.name.trim(), name_key(&spec.name), spec.enabled, steps], read_routine,
            ).optional()?;
            if routine.is_some() && !spec.enabled {
                stop_running(db, id, now)?;
            }
            Ok(routine)
        })
    }

    /// Remove the definition from use while retaining historical run receipts.
    pub fn delete_routine(&mut self, id: RoutineId, revision: i64) -> Result<bool> {
        let now = self.clock.now();
        self.transaction(|db| {
            let changed = db.execute(
                "UPDATE routines SET deleted_at = ?3, enabled = 0, revision = revision + 1
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

    /// Only a newly created, running claim admits execution for an occurrence.
    /// The transaction binds a snapshot of the current definition to the run.
    pub fn claim_routine(
        &mut self,
        id: RoutineId,
        occurrence: Occurrence<'_>,
        expected_revision: Option<i64>,
    ) -> Result<RunClaim> {
        let now = self.clock.now();
        let (source, key) = match occurrence {
            Occurrence::Scheduled(at) if at <= now => {
                (RunSource::Scheduled, format!("scheduled:{at}"))
            }
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
        self.transaction(|db| {
            let existing = db.query_row(
                &format!("SELECT {RUN_COLUMNS} FROM routine_runs
                          WHERE occurrence_key = ?2 AND (routine_id = ?1 OR source = 'voice')"),
                params![id.0, key], read_run,
            ).optional()?;
            if let Some(existing) = existing {
                if existing.routine != id {
                    return Err(Error::Conflict("voice turn already selected a different routine"));
                }
                return Ok(RunClaim::Existing(existing));
            }
            let (enabled, revision, steps): (bool, i64, String) = db.query_row(
                "SELECT enabled, revision, steps FROM routines WHERE id = ?1 AND deleted_at IS NULL",
                [id.0], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            ).optional()?.ok_or(Error::NotFound("routine"))?;
            if !enabled {
                return Err(Error::Conflict("routine is disabled"));
            }
            if expected_revision.is_some_and(|expected| expected != revision) {
                return Err(Error::Conflict("routine revision changed"));
            }
            let running: bool = db.query_row(
                "SELECT EXISTS (SELECT 1 FROM routine_runs WHERE routine_id = ?1 AND status = 'running')",
                [id.0], |row| row.get(0),
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
        })
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
                    "UPDATE routines SET enabled = 0, revision = revision + 1 WHERE id = ?1",
                    [id],
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
