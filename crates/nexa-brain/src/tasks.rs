//! Voice-only household scopes and periodic delivery of durable work.

#![allow(
    clippy::result_large_err,
    reason = "Tonic status carries gRPC metadata and codes across this boundary"
)]

use std::{sync::Arc, time::Duration};

use futures_util::stream::{self, StreamExt};
use nexa_household::{
    Error, Household,
    alerts::{AlertObservation, AlertUpdate},
    intercom::{Intercom, Playback, Zone},
    reminders::{ActionOutcome, NoticeOutcome, VoiceCancel, VoiceContext},
    routines::{Occurrence, RoutineEffect, RoutineRun, RunClaim, StepOutcome, VoiceBrake},
};
use nexa_nlu::{
    grammar::canonical_room,
    household::{HouseholdFamily as Family, HouseholdIntent, HouseholdOperation as Op},
    normalize::words,
};
use nexa_proto::v1::LiveEventKind;
use serde_json::json;
use tonic::Status;

use crate::{Brain, home::scheduled_action, pipeline::Turn};

pub(crate) const SECURITY_REFUSAL: &str = "No controlo cerraduras ni alarmas de seguridad por voz.";
const CLARIFY: &str = "Me falta un dato. ¿Podés repetirlo?";

pub(crate) fn household_status(error: Error) -> Status {
    match error {
        Error::InvalidInput(detail) => Status::invalid_argument(detail),
        Error::Conflict(detail) => Status::aborted(detail),
        Error::NotFound(_) => Status::not_found("household record not found"),
        _ => Status::internal("household store operation failed"),
    }
}

fn duration_at(amount: &str, unit: &str) -> Option<Duration> {
    let amount = amount.parse::<u64>().ok().or(match amount {
        "un" | "uno" | "una" => Some(1),
        "dos" => Some(2),
        "tres" => Some(3),
        "cuatro" => Some(4),
        "cinco" => Some(5),
        "seis" => Some(6),
        "siete" => Some(7),
        "ocho" => Some(8),
        "nueve" => Some(9),
        "diez" => Some(10),
        "quince" => Some(15),
        "veinte" => Some(20),
        "treinta" => Some(30),
        _ => None,
    })?;
    let multiplier = match unit {
        "segundo" | "segundos" => 1,
        "minuto" | "minutos" | "min" => 60,
        "hora" | "horas" => 3600,
        "dia" | "dias" => 86400,
        _ => return None,
    };
    let seconds = amount.checked_mul(multiplier)?;
    (seconds > 0 && seconds <= 365 * 86400).then(|| Duration::from_secs(seconds))
}

impl Brain {
    pub(crate) fn intercom(&self) -> Result<Intercom, Status> {
        Intercom::new(
            self.config
                .rooms
                .rooms
                .iter()
                .map(|room| Zone {
                    id: self.output_zone(&room.room_id),
                    output: room.ma1260_zone.to_string(),
                })
                .collect(),
        )
        .map_err(household_status)
    }

    pub(crate) async fn voice_household(
        &self,
        intent: &HouseholdIntent,
        input: &str,
        turn: &Turn,
        conversation: &str,
    ) -> Result<String, Status> {
        use Family::*;
        if intent.operation == Op::Clarify {
            return Ok(CLARIFY.into());
        }
        if intent.family == Intercom {
            return self.voice_intercom(intent, input, turn, conversation).await;
        }
        let room = turn.zone.clone();
        let name = words(&intent.complement);
        let operation = intent.operation;
        let family = intent.family;
        let reason = intent.reason;
        let list = self.config.core.household.default_list_name.clone();
        let turn_id = format!("{room}:{conversation}");
        let now = self.clock.now();
        let local_now = self.local_now();
        if family == Notification {
            let mut quiet = self
                .quiet
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            return Ok(match operation {
                Op::Disable => {
                    quiet.remove(&room);
                    "Desactivé No molestar para esta habitación.".into()
                }
                Op::Execute => {
                    quiet.insert(room);
                    "Activé No molestar para esta habitación.".into()
                }
                _ => CLARIFY.into(),
            });
        }
        self.household
            .call(move |db| match family {
                Timer => timer(db, &room, &name, operation, now),
                Reminder => reminder(db, &room, &name, operation, reason, local_now),
                List => list_voice(db, &name, operation, &list),
                Routine => routine(db, &name, operation, &turn_id),
                DeferredAction => Ok("Programá esa acción desde el panel.".into()),
                _ => Ok("Esa consulta no está disponible.".into()),
            })
            .await
            .map_err(household_status)
    }

    async fn voice_intercom(
        &self,
        intent: &HouseholdIntent,
        input: &str,
        turn: &Turn,
        conversation: &str,
    ) -> Result<String, Status> {
        let Some((start, end)) = intent.literal_span else {
            return Ok(CLARIFY.into());
        };
        let message = input
            .get(start..end)
            .unwrap_or_default()
            .trim_matches(|c: char| c.is_whitespace() || ['"', '\'', '«', '»'].contains(&c));
        if message.is_empty() || message.chars().count() > 320 {
            return Ok(CLARIFY.into());
        }
        let prefix = words(input.get(..start).unwrap_or_default());
        let directed = prefix
            .split_whitespace()
            .nth(1)
            .is_some_and(|word| matches!(word, "en" | "a" | "al" | "por" | "para"));
        let all = [
            "toda la casa",
            "todo el hogar",
            "todos lados",
            "todas las habitaciones",
        ]
        .iter()
        .any(|phrase| prefix.contains(phrase));
        let room = self.config.rooms.rooms.iter().find(|room| {
            prefix.split_whitespace().any(|word| {
                word == room.room_id || canonical_room(word) == Some(room.room_id.as_str())
            }) || prefix.contains(&words(&room.name))
        });
        let intercom = self.intercom()?;
        let target =
            self.output_zone(room.map_or(turn.zone.as_str(), |room| room.room_id.as_str()));
        let ids = [target.as_str()];
        let targets = if all {
            nexa_household::intercom::Targets::All
        } else {
            if directed && room.is_none() {
                return Ok("¿En qué habitación?".into());
            }
            nexa_household::intercom::Targets::Zones(&ids)
        };
        let mut heard = false;
        let mut uncertain = false;
        let mut channels = std::collections::HashSet::new();
        for zone in intercom.route(targets).map_err(household_status)? {
            if !channels.insert(&zone.output) {
                continue;
            }
            let Some(room) = self.room_for_zone(&zone.id) else {
                continue;
            };
            let playback = if room == turn.zone {
                match self.say(turn, conversation, message).await {
                    Ok(receipt) if receipt.completed => Playback::Confirmed,
                    _ => Playback::Uncertain,
                }
            } else {
                self.announce(room, message).await
            };
            heard |= playback == Playback::Confirmed;
            uncertain |= playback != Playback::Confirmed;
        }
        Ok(match (heard, uncertain) {
            (true, false) => "Aviso enviado.",
            (true, true) => "El aviso se reprodujo en parte de las zonas.",
            _ => "No pude confirmar el aviso.",
        }
        .into())
    }

    /// Feed an alert observation from a local detector and announce new high-priority episodes.
    pub async fn observe_alert(
        &self,
        observation: AlertObservation<'_>,
    ) -> Result<AlertUpdate, Status> {
        let key = observation.key.to_owned();
        let message = observation.message.to_owned();
        let lifetime = observation.lifetime;
        let priority = observation.priority;
        let (update, preferences) = self
            .household
            .call(move |db| {
                let update = db.observe_alert(AlertObservation {
                    key: &key,
                    priority,
                    message: &message,
                    lifetime,
                })?;
                Ok((update, db.alert_preferences()?))
            })
            .await
            .map_err(household_status)?;
        let alert = match &update {
            AlertUpdate::Opened(alert) | AlertUpdate::Escalated(alert) => alert,
            AlertUpdate::Deduplicated(_) => return Ok(update),
        };
        self.event(
            LiveEventKind::Alert,
            "",
            "",
            json!({ "alert_id": alert.id.0, "revision": alert.revision }),
        );
        if update.should_announce() {
            let intercom = self.intercom()?;
            let zones = preferences.selected_zones(
                &intercom,
                Some(&self.config.core.household.default_output_zone),
            );
            let mut channels = std::collections::HashSet::new();
            for zone in zones {
                let Some(output) = intercom.zone(zone).map(|zone| &zone.output) else {
                    continue;
                };
                let Some(room) = self.room_for_zone(zone) else {
                    continue;
                };
                let muted = self
                    .quiet
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .contains(room);
                if !muted && channels.insert(output) {
                    self.announce(room, &alert.message).await;
                }
            }
        }
        Ok(update)
    }

    pub(crate) async fn run_household(self: Arc<Self>) {
        let mut tick = tokio::time::interval(Duration::from_millis(250));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = self.shutdown.cancelled() => return,
                _ = tick.tick() => {
                    if self.poll_household().await.is_err() {
                        tracing::warn!("household polling failed");
                    }
                }
            }
        }
    }

    /// Advance due timers, shared notices, and routine runs using the injected clock.
    pub async fn poll_household(&self) -> Result<(), Status> {
        let clock = Arc::clone(&self.clock);
        let (timers, runs, delivered) = self
            .household
            .call(move |db| {
                // The running snapshot already includes newly claimed active occurrences.
                db.poll_due_scheduled_routines(clock.now())?;
                Ok((
                    db.complete_due_timers()?,
                    db.running_routines()?,
                    db.reminders_with_delivered_actions()?,
                ))
            })
            .await
            .map_err(household_status)?;
        let timers = stream::iter(timers).for_each_concurrent(8, |timer| async move {
            self.event(
                LiveEventKind::Alert,
                &timer.room,
                "",
                json!({ "event": "timer_due", "timer_id": timer.id.0 }),
            );
            self.announce(
                &timer.room,
                &format!("Terminó el temporizador {}.", timer.name),
            )
            .await;
        });
        let rooms = stream::iter(self.config.rooms.rooms.iter())
            .map(|room| self.deliver_notice(&room.room_id))
            .buffer_unordered(8)
            .collect::<Vec<_>>();
        let runs = stream::iter(runs)
            .map(|run| self.advance_routine(run))
            .buffer_unordered(8)
            .collect::<Vec<_>>();
        let actions = stream::iter(delivered)
            .map(|id| self.advance_reminder_action(id))
            .buffer_unordered(8)
            .collect::<Vec<_>>();
        let (_, rooms, runs, actions) = tokio::join!(timers, rooms, runs, actions);
        rooms.into_iter().chain(runs).chain(actions).collect()
    }

    async fn deliver_notice(&self, room: &str) -> Result<(), Status> {
        if self.shutdown.is_cancelled() || self.output_busy(room) {
            return Ok(());
        }
        let zone = room.to_owned();
        let claim = self
            .household
            .call(move |db| db.claim_reminder_notice(&zone))
            .await
            .map_err(household_status)?;
        let Some(claim) = claim else { return Ok(()) };
        let result = self.announce(room, &claim.reminder.message).await;
        let outcome = match result {
            Playback::Confirmed => NoticeOutcome::Delivered,
            Playback::NotStarted => NoticeOutcome::NotStarted,
            Playback::Uncertain => NoticeOutcome::Uncertain,
        };
        let id = claim.reminder.id;
        self.household
            .call(move |db| db.finish_reminder_notice(&claim, outcome))
            .await
            .map_err(household_status)?;
        self.event(
            LiveEventKind::Alert,
            room,
            "",
            json!({ "event": "reminder_notice", "reminder_id": id.0 }),
        );
        Ok(())
    }

    async fn advance_reminder_action(
        &self,
        id: nexa_household::reminders::ReminderId,
    ) -> Result<(), Status> {
        if self.shutdown.is_cancelled() {
            return Ok(());
        }
        let action = self
            .household
            .call(move |db| db.claim_reminder_action(id))
            .await
            .map_err(household_status)?;
        let Some(action) = action else { return Ok(()) };
        let outcome = match self
            .home_effect(
                format!("reminder-{}-{}", id.0, action.position),
                &action.command,
            )
            .await
        {
            StepOutcome::Succeeded => ActionOutcome::Succeeded,
            StepOutcome::Failed => ActionOutcome::Failed,
            StepOutcome::Uncertain => ActionOutcome::Uncertain,
        };
        self.household
            .call(move |db| db.finish_reminder_action(&action, outcome))
            .await
            .map_err(household_status)?;
        Ok(())
    }

    async fn advance_routine(&self, run: RoutineRun) -> Result<(), Status> {
        if self.shutdown.is_cancelled() {
            return Ok(());
        }
        if let Some(nexa_household::routines::RoutineStep::Announce { room, .. }) =
            run.steps.get(run.next_step)
            && self.output_busy(room)
        {
            return Ok(());
        }
        let Some(step) = self
            .household
            .call(move |db| db.claim_routine_step(run.id))
            .await
            .map_err(household_status)?
        else {
            return Ok(());
        };
        let outcome = match &step.effect {
            RoutineEffect::Announce { room, message } => match self.announce(room, message).await {
                Playback::Confirmed => StepOutcome::Succeeded,
                Playback::NotStarted => StepOutcome::Failed,
                Playback::Uncertain => StepOutcome::Uncertain,
            },
            RoutineEffect::Home(action) => {
                self.home_effect(format!("routine-{}-{}", step.run.0, step.position), action)
                    .await
            }
        };
        self.household
            .call(move |db| db.finish_routine_step(&step, outcome))
            .await
            .map_err(household_status)?;
        Ok(())
    }

    async fn home_effect(
        &self,
        request_id: String,
        action: &nexa_household::HomeAction,
    ) -> StepOutcome {
        let Ok(action) = scheduled_action(action) else {
            return StepOutcome::Failed;
        };
        match self
            .execute_home(request_id, action)
            .await
            .as_ref()
            .map(|receipt| receipt.status.as_str())
        {
            Ok("accepted") => StepOutcome::Succeeded,
            Ok("uncertain" | "partial") => StepOutcome::Uncertain,
            _ => StepOutcome::Failed,
        }
    }
}

type State = Household<Arc<dyn crate::BrainClock>>;

fn timer(
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

fn reminder(
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

fn list_voice(
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

fn routine(db: &mut State, name: &str, op: Op, turn_id: &str) -> nexa_household::Result<String> {
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
