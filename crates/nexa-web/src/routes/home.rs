use super::*;

#[derive(Clone, Copy)]
pub(super) enum Action {
    Model,
    Plan,
    Catalog,
    Microphones,
    Execute,
    HomeReceipt,
}

pub(super) async fn execute(
    op: Action,
    state: &AppState,
    fields: &Fields,
    lease: &Lease,
) -> Result<Response> {
    use Action::*;
    let brain = &state.clients.brain;
    let audio = &state.clients.audio;
    let body = &fields.body;
    let actor = || Some(lease.actor());
    let status = StatusCode::OK;
    let value = match op {
        Model => {
            if !fields.query.is_empty() {
                return Err(NOT_FOUND);
            }
            return private_asset(state, &["casa.glb"], "model/gltf-binary").await;
        }
        Plan => {
            if !fields.query.is_empty() {
                return Err(NOT_FOUND);
            }
            match private_asset(state, &["house-plan.js"], "text/javascript; charset=utf-8").await {
                Err(ApiError(StatusCode::NOT_FOUND, _)) => {
                    return private_asset(state, &["house-plan.json"], "application/json").await;
                }
                result => return result,
            }
        }
        Catalog => home_catalog(
            brain
                .get_home_catalog(GetHomeCatalogRequest { actor: actor() })
                .await?,
        )?,
        Microphones => microphones(audio.get_microphones(Empty {}).await?)?,
        Execute => home_receipt(
            brain
                .execute_home_action(home_intent(body, actor())?)
                .await?,
        ),
        HomeReceipt => home_receipt(
            brain
                .get_home_action_receipt(GetHomeActionReceiptRequest {
                    request_id: request_id(fields.query("request_id"))?.into(),
                    actor: actor(),
                })
                .await?,
        ),
    };
    Ok((status, Json(value)).into_response())
}
