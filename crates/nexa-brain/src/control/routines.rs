use super::*;

impl Control {
    fn routine_spec(&self, value: &str) -> Result<ScheduledRoutineSpec, Status> {
        let payload: RoutinePayload = parse_json(value)?;
        let spec = RoutineSpec {
            name: payload.name,
            enabled: payload.enabled,
            steps: payload.steps,
        };
        if spec.name.trim().is_empty()
            || spec.name.len() > 128
            || !(1..=32).contains(&spec.steps.len())
        {
            return Err(Status::invalid_argument(
                "routine requires a name and 1 to 32 steps",
            ));
        }
        for step in &spec.steps {
            match step {
                RoutineStep::Home { action } => self.0.allow_home(&scheduled_action(action)?)?,
                RoutineStep::Announce { room, message } => {
                    if !self.0.zone(room) || message.trim().is_empty() || message.len() > 1024 {
                        return Err(Status::invalid_argument("invalid routine announcement"));
                    }
                }
                RoutineStep::Delay { duration } if *duration > Duration::from_secs(86400) => {
                    return Err(Status::invalid_argument("routine delay exceeds one day"));
                }
                RoutineStep::Delay { .. } => {}
            }
        }
        if let Some(RoutineSchedule::Daily {
            hour,
            minute,
            timezone,
            days,
        }) = &payload.schedule
            && (chrono::NaiveTime::from_hms_opt(*hour, *minute, 0).is_none()
                || timezone.parse::<chrono_tz::Tz>().is_err()
                || days.iter().any(|day| !(1..=7).contains(day)))
        {
            return Err(Status::invalid_argument("invalid daily routine schedule"));
        }
        Ok(ScheduledRoutineSpec {
            spec,
            schedule: payload.schedule,
        })
    }
}
pub(super) async fn list_routines(
    control: &Control,
    request: Request<v1::ListRoutinesRequest>,
) -> Reply<v1::RoutinesSnapshot> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::HomeRead])?;
    let routines = control
        .0
        .household
        .call(|db| db.routines())
        .await
        .map_err(household_status)?;
    ok(v1::RoutinesSnapshot {
        routines: routines
            .into_iter()
            .filter(|routine| !request.enabled_only.unwrap_or(false) || routine.spec.enabled)
            .map(routine_record)
            .collect::<Result<_, _>>()?,
    })
}

pub(super) async fn get_routine(
    control: &Control,
    request: Request<v1::GetRoutineRequest>,
) -> Reply<v1::RoutineRecord> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::HomeRead])?;
    let id = RoutineId(id(&request.routine_id)?);
    let routine = control
        .0
        .household
        .call(move |db| db.routine(id))
        .await
        .map_err(household_status)?
        .ok_or_else(|| Status::not_found("routine not found"))?;
    ok(routine_record(routine)?)
}

pub(super) async fn preview_routine(
    control: &Control,
    request: Request<v1::PreviewRoutineRequest>,
) -> Reply<v1::RoutinePreviewResponse> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::RoutinesManage])?;
    match control.routine_spec(&request.spec_json) {
        Ok(spec) => ok(v1::RoutinePreviewResponse {
            valid: true,
            preview_json: routine_json(&spec.spec, spec.schedule.as_ref()),
            errors: Vec::new(),
        }),
        Err(error) => ok(v1::RoutinePreviewResponse {
            valid: false,
            preview_json: String::new(),
            errors: vec![error.message().into()],
        }),
    }
}

pub(super) async fn upsert_routine(
    control: &Control,
    request: Request<v1::UpsertRoutineRequest>,
) -> Reply<v1::RoutineRecord> {
    let request = request.into_inner();
    let actor = authorize(request.actor.as_ref(), &[P::RoutinesManage])?;
    let spec = control.routine_spec(&request.spec_json)?;
    if spec
        .spec
        .steps
        .iter()
        .any(|step| matches!(step, RoutineStep::Home { .. }))
    {
        authorize(Some(actor), &[P::HomeControl])?;
    }
    let expected = revision(request.expected_revision)?;
    let id = if request.routine_id.is_empty() {
        None
    } else {
        Some(RoutineId(id(&request.routine_id)?))
    };
    let routine = control
        .0
        .household
        .call(move |db| match id {
            None if expected == 0 => db.create_scheduled_routine(&spec),
            None => Err(nexa_household::Error::Conflict(
                "new routine cannot have a revision",
            )),
            Some(id) => {
                let conflict = || nexa_household::Error::Conflict("routine revision changed");
                let current = db
                    .routine(id)?
                    .filter(|current| current.revision == expected)
                    .ok_or_else(conflict)?;
                let updated = if current.spec.enabled != spec.spec.enabled
                    && current.spec.name == spec.spec.name
                    && current.spec.steps == spec.spec.steps
                    && current.schedule == spec.schedule
                {
                    db.set_routine_enabled(id, expected, spec.spec.enabled)?
                } else {
                    db.update_scheduled_routine(id, expected, &spec)?
                };
                updated.ok_or_else(conflict)
            }
        })
        .await
        .map_err(household_status)?;
    ok(routine_record(routine)?)
}

pub(super) async fn delete_routine(
    control: &Control,
    request: Request<v1::DeleteRoutineRequest>,
) -> Reply<v1::Empty> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::RoutinesManage])?;
    let id = RoutineId(id(&request.routine_id)?);
    let expected = revision(request.expected_revision)?;
    let changed = control
        .0
        .household
        .call(move |db| db.delete_routine(id, expected))
        .await
        .map_err(household_status)?;
    if !changed {
        return Err(Status::aborted("routine revision changed"));
    }
    ok(v1::Empty {})
}

pub(super) async fn execute_routine(
    control: &Control,
    request: Request<v1::ExecuteRoutineRequest>,
) -> Reply<v1::RoutineExecutionReceipt> {
    let request = request.into_inner();
    let actor = authorize(request.actor.as_ref(), &[P::HomeControl])?;
    if !(has(actor, P::RoutinesManage) || has(actor, P::RoutinesExecute)) {
        return Err(Status::permission_denied(
            "routine execution permission required",
        ));
    }
    if !valid_identifier(&request.request_id) {
        return Err(Status::invalid_argument("invalid routine request ID"));
    }
    let id = RoutineId(id(&request.routine_id)?);
    let result = control
        .0
        .household
        .call(move |db| db.claim_routine(id, Occurrence::Manual(&request.request_id), None))
        .await
        .map_err(household_status)?;
    let run = match result {
        RunClaim::Created(run) | RunClaim::Existing(run) => run,
    };
    let status = run_status(run.state);
    ok(v1::RoutineExecutionReceipt {
        run_id: run.id.0.to_string(),
        routine_id: run.routine.0.to_string(),
        status: status.into(),
        accepted_at_unix_ms: unsigned(run.started_at),
    })
}

pub(super) async fn get_routine_run(
    control: &Control,
    request: Request<v1::GetRoutineRunRequest>,
) -> Reply<v1::RoutineRunRecord> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::HomeRead])?;
    let routine = RoutineId(id(&request.routine_id)?);
    let run_id = nexa_household::routines::RunId(id(&request.run_id)?);
    let run = control
        .0
        .household
        .call(move |db| {
            if db.routine(routine)?.is_none() {
                return Ok(None);
            }
            Ok(db.routine_run(run_id)?.filter(|run| run.routine == routine))
        })
        .await
        .map_err(household_status)?
        .ok_or_else(|| Status::not_found("routine run not found"))?;
    ok(run_record(run))
}

pub(super) async fn list_routine_runs(
    control: &Control,
    request: Request<v1::ListRoutineRunsRequest>,
) -> Reply<v1::RoutineRunsSnapshot> {
    let request = request.into_inner();
    authorize(request.actor.as_ref(), &[P::HomeRead])?;
    let routine = RoutineId(id(&request.routine_id)?);
    let runs = control
        .0
        .household
        .call(move |db| {
            if db.routine(routine)?.is_none() {
                return Err(nexa_household::Error::NotFound("routine"));
            }
            db.list_routine_runs(routine, limit(request.limit))
        })
        .await
        .map_err(household_status)?;
    ok(v1::RoutineRunsSnapshot {
        routine_id: routine.0.to_string(),
        runs: runs.into_iter().map(run_record).collect(),
    })
}

fn run_status(state: RunState) -> &'static str {
    match state {
        RunState::Running => "running",
        RunState::Succeeded => "succeeded",
        RunState::Failed => "failed",
        RunState::Stopped => "stopped",
        RunState::Skipped => "skipped",
        RunState::Interrupted => "interrupted",
    }
}

fn run_record(run: nexa_household::routines::RoutineRun) -> v1::RoutineRunRecord {
    use nexa_household::routines::RunSource;
    v1::RoutineRunRecord {
        run_id: run.id.0.to_string(),
        routine_id: run.routine.0.to_string(),
        status: run_status(run.state).into(),
        source: match run.source {
            RunSource::Voice => "voice",
            RunSource::Manual => "manual",
            RunSource::Scheduled => "scheduled",
        }
        .into(),
        started_at_unix_ms: unsigned(run.started_at),
        finished_at_unix_ms: run.finished_at.map(unsigned),
    }
}
