use super::*;

#[derive(Clone, Copy)]
pub(super) enum Action {
    Routines,
    Routine,
    RoutinePreview,
    RoutineUpsert,
    RoutineDelete,
    RoutineExecute,
    RoutineEnabled(bool),
    RoutineRun,
    RoutineRuns,
}

pub(super) async fn execute(
    op: Action,
    state: &AppState,
    fields: &Fields,
    lease: &Lease,
    proof: &Presented,
    policy: Policy,
) -> Result<Response> {
    use Action::*;
    let brain = &state.clients.brain;
    let id = fields.id();
    let body = &fields.body;
    let actor = || Some(lease.actor());
    let mut status = StatusCode::OK;
    let value = match op {
        Routines => {
            let enabled_only = optional_bool(&fields.query, "enabled_only")?;
            let snapshot = brain
                .list_routines(ListRoutinesRequest {
                    enabled_only,
                    actor: actor(),
                })
                .await?;
            let rows = snapshot
                .routines
                .into_iter()
                .map(routine)
                .collect::<Result<Vec<_>>>()?;
            json!(rows)
        }
        Routine => routine(
            brain
                .get_routine(GetRoutineRequest {
                    routine_id: id.into(),
                    actor: actor(),
                })
                .await?,
        )?,
        RoutinePreview => {
            let response = brain
                .preview_routine(PreviewRoutineRequest {
                    spec_json: body.to_string(),
                    actor: actor(),
                })
                .await?;
            if response.preview_json.is_empty() {
                json!({"valid":response.valid,"errors":response.errors})
            } else {
                projection(response.preview_json)?
            }
        }
        RoutineUpsert | RoutineEnabled(_) => {
            let revision = if id.is_empty() { 0 } else { fields.revision()? };
            let mut spec = body.clone();
            if let RoutineEnabled(enabled) = op {
                let old = brain
                    .get_routine(GetRoutineRequest {
                        routine_id: id.into(),
                        actor: actor(),
                    })
                    .await?;
                if old.revision != revision {
                    return Err(ApiError(StatusCode::CONFLICT, "routine_revision_conflict"));
                }
                spec = projection(old.spec_json)?;
                spec.as_object_mut()
                    .ok_or(BAD_UPSTREAM)?
                    .insert("enabled".into(), json!(enabled));
                state.auth.validate(proof, policy.permissions).await?;
            } else if let Some(object) = spec.as_object_mut() {
                object.remove("expected_revision");
            }
            routine(
                brain
                    .upsert_routine(UpsertRoutineRequest {
                        routine_id: id.into(),
                        expected_revision: revision,
                        spec_json: spec.to_string(),
                        actor: actor(),
                    })
                    .await?,
            )?
        }
        RoutineDelete => {
            brain
                .delete_routine(DeleteRoutineRequest {
                    routine_id: id.into(),
                    expected_revision: fields.revision()?,
                    actor: actor(),
                })
                .await?;
            json!({"success":true})
        }
        RoutineExecute => {
            let receipt = brain
                .execute_routine(ExecuteRoutineRequest {
                    routine_id: id.into(),
                    request_id: request_id(string(body, "request_id")?)?.into(),
                    actor: actor(),
                })
                .await?;
            status = StatusCode::ACCEPTED;
            json!({"run_id":receipt.run_id,"routine_id":receipt.routine_id,"status":receipt.status,
                "accepted_at_unix_ms":receipt.accepted_at_unix_ms})
        }

        RoutineRun => {
            let run = brain
                .get_routine_run(GetRoutineRunRequest {
                    routine_id: id.into(),
                    run_id: fields.path.get("run_id").ok_or(INVALID)?.clone(),
                    actor: actor(),
                })
                .await?;
            json!({"run": run_json(run)})
        }
        RoutineRuns => {
            let snapshot = brain
                .list_routine_runs(ListRoutineRunsRequest {
                    routine_id: id.into(),
                    limit: fields.limit(50, 128)?,
                    actor: actor(),
                })
                .await?;
            json!({"routine_id": snapshot.routine_id, "runs": snapshot.runs.into_iter().map(run_json).collect::<Vec<_>>()})
        }
    };
    Ok((status, Json(value)).into_response())
}

fn run_json(run: RoutineRunRecord) -> Value {
    json!({"run_id":run.run_id,"routine_id":run.routine_id,"status":run.status,"source":run.source,
           "started_at_unix_ms":run.started_at_unix_ms,"finished_at_unix_ms":run.finished_at_unix_ms})
}
