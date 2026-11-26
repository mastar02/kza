//! Timer, alarm, reminder and list operations.

use crate::normalize::fold;

use super::HouseholdFamily::{List, Reminder, Routine, Timer};
use super::HouseholdOperation::{
    AddItem, Cancel, Clarify, Create, Disable, Pause, Query, RemoveItem, Resume, Stop,
};
use super::lexicon::{
    ADD, ADJECTIVES, ALARM_NOUNS, ALARM_VERBS, CANCEL, CLEAR, CREATE, DETERMINERS, DISABLE,
    HAVE_HEADS, MUSIC_WORDS, PAUSE, QUERY_HEADS, QUERY_VERBS, READ_HEADS, REMOVE, RESUME,
    SECURITY_ALARM_VERBS, SOME, STOP, STOP_RINGING, TIMER_CREATE,
};
use super::utterance::{Utterance, Verb};
use super::{HouseholdFamily, HouseholdIntent, HouseholdOperation};

pub(super) fn object(utt: &Utterance<'_>, i: usize, alarms: bool) -> Option<HouseholdFamily> {
    match utt.at(i) {
        "timer" | "timers" | "temporizador" | "temporizadores" => Some(Timer),
        "recordatorio" | "recordatorios" => Some(Reminder),
        "rutina" | "rutinas" => Some(Routine),
        "lista" | "listas" if !(utt.at(i + 1) == "de" && MUSIC_WORDS.contains(&utt.at(i + 2))) => {
            Some(List)
        }
        word if alarms && ALARM_NOUNS.contains(&word) => Some(Timer),
        _ => None,
    }
}

pub(super) fn governed(
    utt: &Utterance<'_>,
    mut i: usize,
    adjective: bool,
    alarms: bool,
    automation: bool,
) -> Option<(HouseholdFamily, usize)> {
    if utt.at(i) == "me" {
        i += 1;
    }
    for _ in 0..2 {
        if DETERMINERS.contains(&utt.at(i)) {
            i += 1;
        }
    }
    if adjective && ADJECTIVES.contains(&utt.at(i)) {
        i += 1;
    }
    if automation && matches!(utt.at(i), "automatizacion" | "automatizaciones") {
        return Some((Routine, i));
    }
    object(utt, i, alarms).map(|family| (family, i))
}

fn clock_alarm(tail: &str) -> bool {
    let text = fold(tail);
    if let Some(m) = CLOCK_ALARM_HOUR.find(&text) {
        let next = text[m.end()..]
            .split_whitespace()
            .next()
            .unwrap_or_default();
        if !matches!(
            next,
            "segundo" | "segundos" | "minuto" | "minutos" | "min" | "mins"
        ) {
            return true;
        }
    }
    CLOCK_ALARM_DAY.is_match(&text)
}

pub(super) fn control_object(
    utt: &Utterance<'_>,
    verb: Verb<'_>,
) -> Option<(HouseholdFamily, HouseholdOperation, usize)> {
    let (family, i) = governed(utt, verb.end, false, true, verb.has(CANCEL))?;
    if matches!(utt.at(i), "alarma" | "alarmas") && SECURITY_ALARM_VERBS.contains(&verb.form) {
        return None;
    }
    let operation = [
        (CANCEL, Cancel),
        (CLEAR, Cancel),
        (STOP | STOP_RINGING, Stop),
        (PAUSE, Pause),
        (RESUME, Resume),
        (DISABLE, Disable),
    ]
    .into_iter()
    .find_map(|(mask, op)| (verb.has(mask) && (mask != CLEAR || family == List)).then_some(op))?;
    Some((family, operation, i))
}

fn task_intent(
    utt: &Utterance<'_>,
    family: HouseholdFamily,
    op: HouseholdOperation,
    i: usize,
    verb: &str,
) -> HouseholdIntent {
    let rest = utt.rest(i + 1);
    if ALARM_NOUNS.contains(&utt.at(i))
        && clock_alarm(&rest)
        && matches!(op, Cancel | Stop | Disable | Query)
    {
        return HouseholdIntent::new(
            Reminder,
            if op == Query { Query } else { Cancel },
            "R1",
            rest,
            verb,
        )
        .because("clock_alarm");
    }
    let complement = if family == Routine && matches!(op, Stop | Disable) {
        utt.literal(i)
    } else {
        rest
    };
    HouseholdIntent::new(family, op, "R1", complement, verb)
}

pub(super) fn task_control(utt: &Utterance<'_>, verb: Option<Verb<'_>>) -> Option<HouseholdIntent> {
    if let Some(verb) = verb
        && let Some((family, op, i)) = control_object(utt, verb)
    {
        return Some(task_intent(utt, family, op, i, verb.form));
    }
    let head = utt.at(0);
    let (family, i) = if HAVE_HEADS.contains(&head) || READ_HEADS.contains(&head) {
        let skip = usize::from(HAVE_HEADS.contains(&head) && SOME.contains(&utt.at(1)));
        governed(utt, 1 + skip, false, true, false)?
    } else {
        let (i, family) =
            (0..utt.len()).find_map(|i| object(utt, i, true).map(|family| (i, family)))?;
        let remaining = head == "cuanto"
            && family == Timer
            && utt
                .folded
                .iter()
                .take(5)
                .any(|w| matches!(w.as_str(), "falta" | "queda"));
        let query = QUERY_HEADS.contains(&head) && utt.has(QUERY_VERBS);
        let alarm = head == "a"
            && utt.at(1) == "que"
            && utt.at(2) == "hora"
            && ALARM_NOUNS.contains(&utt.at(i));
        if !(remaining || query || alarm) {
            return None;
        }
        (family, i)
    };
    Some(task_intent(utt, family, Query, i, head))
}

pub(super) fn task_update(utt: &Utterance<'_>, verb: Option<Verb<'_>>) -> Option<HouseholdIntent> {
    let Some(verb) = verb else {
        let tail = fold(&utt.rest(1));
        return (matches!(utt.at(0), "timer" | "temporizador")
            && (matches!(utt.at(1), "de" | "para" | "por")
                || DURATION.find(&tail).is_some_and(|m| m.start() == 0)))
        .then(|| HouseholdIntent::new(Timer, Create, "R2", utt.rest(1), ""));
    };
    let set = matches!(verb.form, "pone" | "pon" | "ponga" | "poner");
    if verb.has(CREATE | TIMER_CREATE)
        && let Some((family, i)) = governed(utt, verb.end, true, false, false)
        && (family == Timer || (family == List && verb.has(CREATE) && !set))
    {
        return Some(HouseholdIntent::new(
            family,
            Create,
            "R2",
            utt.rest(i + 1),
            verb.form,
        ));
    }
    if !(0..utt.len()).any(|i| object(utt, i, false) == Some(List)) {
        return None;
    }
    let list_target = set
        && !utt.has(MUSIC_WORDS)
        && (0..utt.len()).any(|i| {
            matches!(utt.at(i), "a" | "en")
                && matches!(utt.at(i + 1), "la" | "mi" | "nuestra")
                && object(utt, i + 2, false) == Some(List)
        });
    let op = if verb.has(ADD) || list_target {
        AddItem
    } else if verb.has(REMOVE) {
        RemoveItem
    } else {
        return None;
    };
    Some(HouseholdIntent::new(
        List,
        op,
        "R2",
        utt.rest(verb.end),
        verb.form,
    ))
}

pub(super) fn alarm(utt: &Utterance<'_>, verb: Option<Verb<'_>>) -> Option<HouseholdIntent> {
    let (i, form) = if matches!(utt.at(0), "alarma" | "despertador") {
        (0, "")
    } else {
        let verb = verb.filter(|v| ALARM_VERBS.contains(&v.form))?;
        let i = verb.end + usize::from(utt.at(verb.end) == "me");
        let i = i + usize::from(DETERMINERS.contains(&utt.at(i)));
        if !matches!(utt.at(i), "alarma" | "despertador") {
            return None;
        }
        (i, verb.form)
    };
    let tail = utt.rest(i + 1);
    let text = fold(&tail);
    let article_unit = ARTICLE_CLOCK_UNIT.is_match(&text);
    let clock = CLOCK.is_match(&text);
    let (family, op, reason) = if article_unit {
        (Reminder, Create, "clock_time")
    } else if DURATION.is_match(&text) {
        (Timer, Create, "duration")
    } else if clock {
        (Reminder, Create, "clock_time")
    } else {
        (Timer, Clarify, "no_time")
    };
    Some(HouseholdIntent::new(family, op, "R3a", tail, form).because(reason))
}

regex_patterns! {
    CLOCK_ALARM_HOUR => r"\b(?:de|a|para) las? (?:[0-9]{1,2}(?:[:.][0-9]{2})?|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieciseis|diecisiete|dieciocho|diecinueve|veinte|veintiun[ao]|veintidos|veintitres)\b";
    CLOCK_ALARM_DAY => r"\b(?:del|al|de la|a la|para el|para la) (?:mediodia|medianoche)\b|\b(?:de|para) (?:hoy|pasado manana|manana)\b|\b(?:del|de este|para el) (?:proximo )?(?:lunes|martes|miercoles|jueves|viernes|sabado|domingo)\b";
    DURATION => r"\b(?:[0-9]+(?:[.,][0-9]+)?|un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieci(?:seis|siete|ocho|nueve)|veinte|veinti(?:un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|media|medio|cuartos?)(?: y \w+)? (?:de )?(?:segundos?|minutos?|mins?|horas?)\b|\b(?:hora y (?:media|cuarto)|minuto y medio)\b";
    ARTICLE_CLOCK_UNIT => r"\b(?:a|para|de) las? (?:[0-9]{1,2}(?:[:.][0-9]{2})?|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieciseis|diecisiete|dieciocho|diecinueve|veinte|veintiun[ao]|veintidos|veintitres) (?:h|hs|hrs|horas?|y \w+ (?:minutos?|mins?))\b";
    CLOCK => r"\b(?:a|para|cuando sean?) (?:las? )?(?:[0-9]{1,2}(?:[:.][0-9]{2})?|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|mediodia|medianoche)\b|\bal mediodia\b";
}
