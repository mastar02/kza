use super::*;

pub(super) fn valid_identifier(value: &str) -> bool {
    (1..=128).contains(&value.len())
        && !value.chars().any(char::is_control)
        && !value.trim().is_empty()
}

pub(super) fn authorize<'a>(
    actor: Option<&'a v1::ActorContext>,
    permissions: &[P],
) -> Result<&'a v1::ActorContext, Status> {
    let actor = actor
        .filter(|actor| valid_identifier(&actor.subject))
        .ok_or_else(|| Status::unauthenticated("ActorContext required"))?;
    let role = Role::try_from(actor.role)
        .ok()
        .filter(|role| *role != Role::Unspecified)
        .ok_or_else(|| Status::unauthenticated("recognized actor role required"))?;
    if !actor.user_id.is_empty() && (role != Role::Home || !valid_identifier(&actor.user_id)) {
        return Err(Status::permission_denied("invalid personal actor scope"));
    }
    for &raw in &actor.permissions {
        let permission =
            P::try_from(raw).map_err(|_| Status::permission_denied("unknown actor permission"))?;
        let personal = matches!(
            permission,
            P::TasksPersonal | P::MemoryOwnRead | P::MemoryOwnWrite | P::MemoryOwnVoice
        );
        let allowed = match role {
            Role::Operator => permission != P::Unspecified && !personal,
            Role::Home => {
                matches!(
                    permission,
                    P::HomeRead
                        | P::HomeControl
                        | P::TasksRead
                        | P::TasksWrite
                        | P::RoutinesExecute
                ) || personal && !actor.user_id.is_empty()
            }
            Role::Evaluator => permission == P::EvaluationRead,
            Role::Unspecified => false,
        };
        if !allowed {
            return Err(Status::permission_denied(
                "permission is outside the actor role",
            ));
        }
    }
    if !permissions.iter().all(|permission| has(actor, *permission)) {
        return Err(Status::permission_denied("required permission missing"));
    }
    Ok(actor)
}

pub(super) fn has(actor: &v1::ActorContext, permission: P) -> bool {
    actor.permissions.contains(&(permission as i32))
}

pub(super) fn authorize_metadata(
    request: &Request<v1::Empty>,
    permissions: &[P],
) -> Result<(), Status> {
    let encoded = request
        .metadata()
        .get_bin(ACTOR_METADATA)
        .ok_or_else(|| Status::unauthenticated("ActorContext metadata required"))?;
    let bytes = encoded
        .to_bytes()
        .map_err(|_| Status::unauthenticated("invalid ActorContext metadata"))?;
    let actor = v1::ActorContext::decode(bytes)
        .map_err(|_| Status::unauthenticated("invalid ActorContext metadata"))?;
    authorize(Some(&actor), permissions)?;
    Ok(())
}

pub(super) fn personal(actor: &v1::ActorContext) -> Result<String, Status> {
    authorize(Some(actor), &[P::TasksPersonal])?;
    if actor.user_id.is_empty() {
        return Err(Status::permission_denied("person-bound actor required"));
    }
    Ok(actor.user_id.clone())
}
