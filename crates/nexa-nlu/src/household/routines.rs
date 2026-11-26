//! Routine proposals and execution, and notification settings.

use super::HouseholdFamily::{Notification, Routine};
use super::HouseholdIntent;
use super::HouseholdOperation::{Disable, Execute, Propose};
use super::lexicon::{
    ADJECTIVES, AUTOMATE, DESIRE_HEADS, DETERMINERS, ENCLITICS, INDEFINITE, ROUTINE_CREATE_FORMS,
    ROUTINE_EXEC, ROUTINE_NEW_ONLY_FORMS, ROUTINE_NOUNS, ROUTINE_PROPOSE, known_command_word,
};
use super::tasks::governed;
use super::utterance::{Utterance, Verb};

fn routine_creation(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    let text = utt.text();
    let m = ROUTINE_REQUEST.captures(&text)?;
    let mut i = m[0].split_whitespace().count();
    let request = m.name("request").map(|m| m.as_str());
    if m.name("que").is_some() && !request.is_some_and(|head| DESIRE_HEADS.contains(&head)) {
        return None;
    }
    let word = utt.at(i);
    let form = std::iter::once(word)
        .chain(
            ENCLITICS
                .iter()
                .filter_map(|suffix| word.strip_suffix(suffix)),
        )
        .find(|form| ROUTINE_CREATE_FORMS.contains(form) || ROUTINE_NEW_ONLY_FORMS.contains(form));
    let slot = form.is_none()
        && (request.is_some() || m.name("clitic").is_some())
        && !known_command_word(word)
        && !DETERMINERS.contains(&word)
        && !ADJECTIVES.contains(&word)
        && !ROUTINE_NOUNS.contains(&word);
    i += usize::from(form.is_some() || slot);
    let mut indefinite = false;
    for _ in 0..2 {
        if DETERMINERS.contains(&utt.at(i)) {
            indefinite |= INDEFINITE.contains(&utt.at(i));
            i += 1;
        }
    }
    let mut new = ADJECTIVES.contains(&utt.at(i));
    i += usize::from(new);
    if !ROUTINE_NOUNS.contains(&utt.at(i)) {
        return None;
    }
    if ADJECTIVES.contains(&utt.at(i + 1)) {
        new |= form.is_none() && !slot;
        i += 1;
    }
    if !(form.is_some_and(|f| ROUTINE_CREATE_FORMS.contains(&f)) || indefinite || new) {
        return None;
    }
    Some(
        HouseholdIntent::new(
            Routine,
            Propose,
            "R6",
            utt.literal(i).to_lowercase(),
            form.unwrap_or_default(),
        )
        .because("new_routine"),
    )
}

pub(super) fn routine_or_notification(
    utt: &Utterance<'_>,
    verb: Option<Verb<'_>>,
) -> Option<HouseholdIntent> {
    if let Some(intent) = routine_creation(utt) {
        return Some(intent);
    }
    if let Some(verb) = verb {
        if verb.has(AUTOMATE) {
            return Some(
                HouseholdIntent::new(
                    Routine,
                    Propose,
                    "R6",
                    utt.literal(verb.end - 1).to_lowercase(),
                    verb.form,
                )
                .because("new_routine"),
            );
        }
        if let Some((Routine, noun)) = governed(utt, verb.end, true, false, false) {
            if verb.has(ROUTINE_EXEC) {
                return Some(HouseholdIntent::new(
                    Routine,
                    Execute,
                    "R6",
                    utt.literal(noun),
                    verb.form,
                ));
            }
            if verb.has(ROUTINE_PROPOSE) {
                return Some(HouseholdIntent::new(
                    Routine,
                    Propose,
                    "R6",
                    utt.rest(noun + 1),
                    verb.form,
                ));
            }
        }
    }
    let text = utt.text();
    let m = NOTIFICATION.captures(&text)?;
    Some(HouseholdIntent::new(
        Notification,
        if m.name("disable").is_some() {
            Disable
        } else {
            Execute
        },
        "R6",
        utt.heard(),
        verb.map_or("", |v| v.form),
    ))
}

regex_patterns! {
    ROUTINE_REQUEST => r"^(?:(?:dale|bueno|bue|ahora|entonces|yo|y|e|eh|ehh|em|emm|mm|mmm|este) )*(?:(?P<request>te pido que|me ayudas a|me ayudarias a|quiero|quisiera|queria|querria|necesito|necesitaria|preciso|me gustaria|nos gustaria|me encantaria|vamos a|voy a|podemos|podriamos|hay que|habria que|se puede|me podes|me podrias|me puedes|nos podes|nos podrias|nos puedes|ayudame a|a ver) )?(?:(?P<que>que) )?(?:(?P<clitic>me|nos) )?";
    NOTIFICATION => r"^(?:(?:(?P<disable>desactiva|desactive|quita|quite|saca|saque)|activa|active|pone|pon) )?(?:el )?(?:modo )?no molestar$";
}
