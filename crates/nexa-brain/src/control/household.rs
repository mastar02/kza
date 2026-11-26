use super::*;

pub(super) async fn list_household_lists(
    control: &Control,
    request: Request<v1::ActorRequest>,
) -> Reply<v1::HouseholdListsSnapshot> {
    let actor = authorize(request.get_ref().actor.as_ref(), &[P::TasksRead])?;
    let owner = if has(actor, P::TasksPersonal) {
        Some(personal(actor)?)
    } else {
        None
    };
    let lists = control
        .0
        .household
        .call(move |db| {
            let mut lists = db.lists(None)?;
            if let Some(owner) = &owner {
                lists.extend(db.lists(Some(owner))?);
            }
            lists
                .into_iter()
                .map(|list| {
                    db.list_snapshot(list.id, list.owner.as_deref())?
                        .ok_or(nexa_household::Error::NotFound("list"))
                        .map(list_record)
                })
                .collect()
        })
        .await
        .map_err(household_status)?;
    ok(v1::HouseholdListsSnapshot { lists })
}

pub(super) async fn mutate_household_list(
    control: &Control,
    request: Request<v1::HouseholdListMutationRequest>,
) -> Reply<v1::HouseholdListRecord> {
    let request = request.into_inner();
    let actor = authorize(request.actor.as_ref(), &[P::TasksWrite])?;
    let body: ListPayload = parse_json(if request.payload_json.is_empty() {
        "{}"
    } else {
        &request.payload_json
    })?;
    let owner = if body.personal {
        Some(personal(actor)?)
    } else {
        None
    };
    let expected = revision(request.expected_revision)?;
    let operation = request.operation;
    let list_id = if operation == "create_list" {
        None
    } else {
        Some(ListId(id(&request.list_id)?))
    };
    ok(control
        .0
        .household
        .call(move |db| {
            use nexa_household::Error;
            let owner = owner.as_deref();
            let Some(id) = list_id else {
                if expected != 0 {
                    return Err(Error::Conflict("new list cannot have a revision"));
                }
                let list = db.create_list(&body.name, owner, None)?;
                return Ok(list_record(ListSnapshot {
                    list,
                    items: Vec::new(),
                }));
            };
            let snapshot = db
                .list_snapshot(id, owner)?
                .ok_or(Error::NotFound("list"))?;
            if snapshot.list.revision != expected {
                return Err(Error::Conflict("list revision changed"));
            }
            match operation.as_str() {
                "rename_list" => {
                    db.rename_list(id, owner, &body.name)?;
                }
                "delete_list" => {
                    db.delete_list(id, owner)?;
                    return Ok(list_record(ListSnapshot {
                        list: snapshot.list,
                        items: Vec::new(),
                    }));
                }
                "add_item" => {
                    db.add_list_item(id, owner, &body.text)?;
                }
                "toggle_item" | "delete_item" => {
                    let item_id = body
                        .item_id
                        .parse::<i64>()
                        .ok()
                        .map(ItemId)
                        .ok_or(Error::InvalidInput("invalid item ID"))?;
                    let item = snapshot
                        .items
                        .iter()
                        .find(|item| item.id == item_id)
                        .ok_or(Error::NotFound("item"))?;
                    if operation == "toggle_item" {
                        db.update_list_item(
                            id,
                            owner,
                            item.id,
                            &item.text,
                            body.completed.unwrap_or(!item.completed),
                        )?;
                    } else {
                        db.delete_list_item(id, owner, item.id)?;
                    }
                }
                _ => return Err(Error::InvalidInput("unknown list operation")),
            }
            db.list_snapshot(id, owner)?
                .map(list_record)
                .ok_or(Error::NotFound("list"))
        })
        .await
        .map_err(household_status)?)
}

pub(super) async fn sync_household_lists(
    control: &Control,
    request: Request<v1::ActorRequest>,
) -> Reply<v1::ListSyncResponse> {
    authorize(request.into_inner().actor.as_ref(), &[P::TasksWrite])?;
    let lists = control
        .0
        .household
        .call(|db| db.lists(None))
        .await
        .map_err(household_status)?;
    let configured = lists.iter().any(|list| list.ha_entity.is_some());
    ok(v1::ListSyncResponse {
        payload_json: json!({
            "lists": lists.len(),
            "synced": 0,
            "ha_todo_configured": configured,
            "status": if configured { "adapter_unavailable" } else { "not_configured" },
        })
        .to_string(),
    })
}
