//! Precedence for single claims and literal messages.

use crate::grammar::{self, GrammarBlock};

use super::HouseholdFamily::{DeferredAction, Intercom, Reminder};
use super::HouseholdIntent;
use super::HouseholdOperation::{Clarify, Create, Execute};
use super::lexicon::{CONJUNCTIONS, CONNECTORS, DOMOTICS, INTERCOM};
use super::queries::integration;
use super::routines::routine_or_notification;
use super::tasks::{alarm, task_control, task_update};
use super::utterance::{Utterance, Verb, verb_at};

pub(super) fn message(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    reminder(utt).or_else(|| intercom(utt))
}

pub(super) fn single(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    message(utt).or_else(|| single_after_messages(utt))
}

pub(super) fn single_after_messages(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    let verb = verb_at(utt, 0, false);
    task_control(utt, verb)
        .or_else(|| task_update(utt, verb))
        .or_else(|| alarm(utt, verb))
        .or_else(|| deferred(utt, verb))
        .or_else(|| routine_or_notification(utt, verb))
        .or_else(|| integration(utt))
}

fn reminder(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    let text = utt.text();
    let m = REMINDER_HEAD.captures(&text)?;
    let used = m[0].split_whitespace().count();
    if matches!(utt.at(0), "acordate" | "acuerdate") && utt.at(used) == "que" {
        return None;
    }
    if m.name("other").is_some() {
        return Some(
            HouseholdIntent::new(Reminder, Create, "R3", utt.rest(used), utt.at(0))
                .because("other_person"),
        );
    }
    let mut rest = used;
    while matches!(utt.at(rest), "que" | "de") {
        rest += 1;
    }
    let intent = HouseholdIntent::new(Reminder, Create, "R3", utt.rest(rest), utt.at(0));
    if (used..utt.len())
        .any(|i| CONJUNCTIONS.contains(&utt.at(i)) && clause_head(utt, connector_end(utt, i)))
    {
        return Some(HouseholdIntent {
            operation: Clarify,
            ..intent.because("second_clause")
        });
    }
    Some(intent)
}

pub(super) fn intercom(utt: &Utterance<'_>) -> Option<HouseholdIntent> {
    let verb = verb_at(utt, 0, false)?;
    if !verb.has(INTERCOM) || verb.infinitive {
        return None;
    }
    let head = utt.at(0);
    let next = utt.at(verb.end);
    let directed = matches!(next, "en" | "a" | "al" | "por" | "para");
    let announce = head == verb.form
        && (head.starts_with("anunci")
            || head.starts_with("comunic")
            || (matches!(head, "avisa" | "avise") && next == "que"));
    if !announce && (!directed || head.ends_with("me") || head.ends_with("nos")) {
        return None;
    }
    let mut intent = HouseholdIntent::new(Intercom, Execute, "R6", utt.literal(0), verb.form);
    if let Some((start, end)) = utt.literal_after(0) {
        let tail = &utt.source[start..end];
        let clean = tail
            .trim_start_matches(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | '¿' | '¡'));
        let lead = tail.len() - clean.len();
        let message = if clean.starts_with(['"', '\'', '«']) {
            lead
        } else if directed {
            INTERCOM_MARKER
                .find(clean)
                .map_or(tail.len(), |m| lead + m.end())
        } else if let Some(m) = LEADING_INTERCOM_MARKER.find(clean) {
            lead + m.end()
        } else {
            lead
        };
        intent.literal_span = Some((start + message, end));
    }
    Some(intent)
}

fn deferred(utt: &Utterance<'_>, verb: Option<Verb<'_>>) -> Option<HouseholdIntent> {
    let verb = verb.filter(|v| v.has(DOMOTICS))?;
    (grammar::non_immediate(&utt.grammar_text()) == Some(GrammarBlock::Deferred))
        .then(|| HouseholdIntent::new(DeferredAction, Create, "R4", utt.heard(), verb.form))
}

pub(super) fn clause_head(utt: &Utterance<'_>, i: usize) -> bool {
    !utt.is_quoted(i)
        && (verb_at(utt, i, true).is_some()
            || (i < utt.len() && REMINDER_HEAD.is_match(&utt.folded[i..].join(" "))))
}

pub(super) fn connector_end(utt: &Utterance<'_>, index: usize) -> usize {
    let mut next = index + 1;
    while CONNECTORS.contains(&utt.at(next)) {
        next += 1;
    }
    next
}

regex_patterns! {
    REMINDER_HEAD => r"^(?:no (?:me|nos) dejes olvidar|hace(?:me|nos) acordar|(?:me|nos) (?:podes|podrias|puedes) (?:recordar|avisar)|(?:me|nos) (?:recordas|avisas)|recorda(?:me|nos|melo)|recuerda(?:me|nos|melo)|acorda(?:me|nos)|acuerda(?:me|nos)|avisa(?:me|nos)|acordate de|acuerdate de|recordar(?:me|nos)|avisar(?:me|nos)|(?P<other>recorda(?:le|les)|recuerda(?:le|les)|avisa(?:le|les)|recordar(?:le|les)|avisar(?:le|les)|(?:le|les) (?:podes|podrias|puedes) (?:recordar|avisar)))\b";
    INTERCOM_MARKER => r"(?i)\bque\b|:";
    LEADING_INTERCOM_MARKER => r"(?i)^(?:que\b|:)";
}
