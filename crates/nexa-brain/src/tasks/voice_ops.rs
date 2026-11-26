use super::*;

type State = Household<Arc<dyn crate::BrainClock>>;

pub(super) fn timer(
    db: &mut State,
    room: &str,
    name: &str,
    op: Op,
    now: i64,
) -> nexa_household::Result<String> {
    let parts = name.split_whitespace().collect::<Vec<_>>();
    if op == Op::Create {
        let Some((index, duration)) = parts.windows(2).enumerate().find_map(|(index, pair)| {
            duration_at(pair[0], pair[1]).map(|duration| (index, duration))
        }) else {
            return Ok("¿Por cuánto tiempo?".into());
        };
        let label = parts.get(index + 2..).unwrap_or_default().join(" ");
        let label = label.strip_prefix("para ").unwrap_or(&label);
        db.create_timer(
            if label.is_empty() {
                "temporizador"
            } else {
                label
            },
            room,
            duration,
        )?;
        return Ok(format!(
            "Temporizador de {} segundos iniciado.",
            duration.as_secs()
        ));
    }
    let name = name.trim_start_matches("de ");
    if op == Op::Stop && name.is_empty() {
        let timers = db.ringing_timers(room)?;
        if !timers.is_empty() {
            for timer in timers {
                db.stop_timer_ring(timer.id, room)?;
            }
            return Ok("Detuve el timbre.".into());
        }
        return Ok("No hay un timbre sonando.".into());
    }
    let selected = if name.is_empty() {
        match db.active_timers(room)?.as_slice() {
            [only] => Some(only.clone()),
            [] => None,
            _ => return Ok("¿Qué temporizador?".into()),
        }
    } else {
        db.find_active_timer(name, room)?
    };
    let Some(timer) = selected else {
        return Ok("No encontré ese temporizador.".into());
    };
    let response = match op {
        Op::Query => {
            return Ok(format!(
                "Quedan {} segundos.",
                timer.remaining(now).as_secs()
            ));
        }
        Op::Cancel | Op::Stop if db.cancel_timer(timer.id)? => "Cancelé el temporizador.",
        Op::Pause if db.pause_timer(timer.id)? => "Pausé el temporizador.",
        Op::Resume if db.resume_timer(timer.id)? => "Reanudé el temporizador.",
        _ => "El temporizador no cambió.",
    };
    Ok(response.into())
}

pub(super) fn reminder(
    db: &mut State,
    room: &str,
    name: &str,
    op: Op,
    reason: Option<&str>,
    now: chrono::DateTime<chrono_tz::Tz>,
) -> nexa_household::Result<String> {
    if reason == Some("other_person") {
        return Ok("Podés crear un recordatorio compartido desde aquí.".into());
    }
    let response = match op {
        Op::Query => {
            let records = db.pending_reminders(None)?;
            if records.is_empty() {
                return Ok("No hay recordatorios compartidos pendientes.".into());
            }
            return Ok(format!(
                "Recordatorios compartidos: {}.",
                records
                    .iter()
                    .take(3)
                    .map(|record| record.message.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        Op::Cancel | Op::Stop => {
            let context = VoiceContext {
                speaker: "unattributed".into(),
                room: room.into(),
            };
            match db.begin_voice_cancel(&context, name)? {
                VoiceCancel::NoMatch => "No encontré ese recordatorio compartido.",
                VoiceCancel::Cancelled(_) => "Cancelé el recordatorio compartido.",
                VoiceCancel::Choose { .. } => {
                    "Hay más de una opción. Decime el nombre del recordatorio."
                }
            }
        }
        Op::Create => {
            let tokens = name.split_whitespace().collect::<Vec<_>>();
            let relative = tokens.windows(3).enumerate().find_map(|(index, part)| {
                if part[0] == "en" {
                    duration_at(part[1], part[2]).map(|duration| (index, duration))
                } else {
                    None
                }
            });
            let (title, due) = if let Some((index, delay)) = relative {
                let title = tokens[..index]
                    .iter()
                    .chain(&tokens[index + 3..])
                    .copied()
                    .collect::<Vec<_>>()
                    .join(" ");
                let due = now
                    .timestamp_millis()
                    .saturating_add(i64::try_from(delay.as_millis()).unwrap_or(i64::MAX));
                (title, due)
            } else {
                return Ok("¿En cuánto tiempo te lo recuerdo?".into());
            };
            if title.trim().is_empty() {
                return Ok("¿Qué querés que recuerde?".into());
            }
            db.create_reminder(&title, None, room, due, &[])?;
            "Agendé el recordatorio compartido."
        }
        _ => CLARIFY,
    };
    Ok(response.into())
}

pub(super) fn list_voice(
    db: &mut State,
    name: &str,
    op: Op,
    default_name: &str,
) -> nexa_household::Result<String> {
    if op == Op::Create {
        let name = name
            .trim_start_matches("llamada ")
            .trim_start_matches("de ");
        if name.is_empty() {
            return Ok("¿Cómo se llama la lista?".into());
        }
        db.create_list(name, None, None)?;
        return Ok("Creé la lista compartida.".into());
    }
    let (item, list_name) = [" en la lista", " de la lista", " a la lista"]
        .iter()
        .find_map(|marker| name.split_once(marker))
        .unwrap_or((
            name,
            if matches!(op, Op::AddItem | Op::RemoveItem) {
                ""
            } else {
                name
            },
        ));
    let list_name = list_name.trim().trim_start_matches("de ");
    let wanted = words(if list_name.is_empty() {
        default_name
    } else {
        list_name
    });
    let found = db
        .lists(None)?
        .into_iter()
        .find(|list| words(&list.name) == wanted);
    let list = match (found, op) {
        (Some(list), _) => list,
        (None, Op::AddItem) if wanted == words(default_name) && !item.trim().is_empty() => {
            db.create_list(default_name, None, None)?
        }
        (None, _) => return Ok("No encontré la lista compartida.".into()),
    };
    let response = match op {
        Op::AddItem if !item.trim().is_empty() => {
            db.add_list_item(list.id, None, item)?;
            "Lo agregué a la lista."
        }
        Op::RemoveItem => {
            let snapshot = db
                .list_snapshot(list.id, None)?
                .ok_or(Error::NotFound("list"))?;
            let mut matching = snapshot
                .items
                .iter()
                .filter(|value| words(&value.text) == words(item));
            match (matching.next(), matching.next()) {
                (Some(value), None) => {
                    db.delete_list_item(list.id, None, value.id)?;
                    "Lo quité de la lista."
                }
                _ => "¿Qué artículo querés quitar?",
            }
        }
        Op::Query => {
            let snapshot = db
                .list_snapshot(list.id, None)?
                .ok_or(Error::NotFound("list"))?;
            let pending = snapshot
                .items
                .iter()
                .filter(|item| !item.completed)
                .take(12)
                .map(|item| item.text.as_str())
                .collect::<Vec<_>>();
            return Ok(if pending.is_empty() {
                "La lista está vacía.".into()
            } else {
                pending.join(", ")
            });
        }
        _ => CLARIFY,
    };
    Ok(response.into())
}

pub(super) fn routine(
    db: &mut State,
    name: &str,
    op: Op,
    turn_id: &str,
) -> nexa_household::Result<String> {
    let routines = db.routines()?;
    let name = name.trim_start_matches("llamada ");
    let mut matches = routines
        .iter()
        .filter(|item| words(&item.spec.name) == name);
    let Some(routine) = matches.next() else {
        return Ok(if op == Op::Propose {
            "Podés configurar una rutina en el panel.".into()
        } else {
            "No encontré esa rutina.".into()
        });
    };
    if matches.next().is_some() {
        return Ok("¿A qué rutina te referís?".into());
    }
    let response = match op {
        Op::Query if routine.spec.enabled => "La rutina está habilitada.",
        Op::Query => "La rutina está deshabilitada.",
        Op::Execute => match db.claim_routine(
            routine.id,
            Occurrence::Voice(turn_id),
            Some(routine.revision),
        )? {
            RunClaim::Created(_) => "Inicié la rutina.",
            RunClaim::Existing(_) => "Ya recibí esa orden.",
        },
        Op::Stop | Op::Disable => {
            let brake = if op == Op::Stop {
                VoiceBrake::Stop
            } else {
                VoiceBrake::Disable
            };
            if db
                .brake_routine_by_voice(&routine.spec.name, brake)?
                .is_some()
            {
                "Apliqué el freno de la rutina."
            } else {
                "La rutina no cambió."
            }
        }
        _ => CLARIFY,
    };
    Ok(response.into())
}
