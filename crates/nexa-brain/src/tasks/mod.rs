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

mod intercom;
mod poll;
mod voice_ops;
use voice_ops::{list_voice, reminder, routine, timer};

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
}
