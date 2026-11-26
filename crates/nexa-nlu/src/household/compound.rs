//! Compound clauses, shared targets and all-or-nothing admission.

use crate::grammar::{self, GrammarBlock, GrammarMatch, Operation as DeviceOperation};
use crate::normalize::fold;

use super::HouseholdFamily::{Compound, DeferredAction, List, Reminder};
use super::HouseholdOperation::{Clarify, Create, Execute};
use super::lexicon::{
    CANCEL, CLEAR, CONJUNCTIONS, CONTENT_MARKERS, DETERMINERS, DISABLE, DOMOTICS, INTERCOM, PAUSE,
    RESUME, STOP, STOP_RINGING, verb_classes,
};
use super::rules::{clause_head, connector_end, intercom, single};
use super::tasks::{control_object, governed, object};
use super::utterance::{Utterance, Verb, verb_at};
use super::{HouseholdIntent, HouseholdIntentSettings, HouseholdPart};

fn split_clauses<'a>(utt: &Utterance<'a>) -> Vec<Utterance<'a>> {
    let mut segments = Vec::new();
    let mut start = 0;
    let mut content = false;
    for i in 1..utt.len() {
        if i <= start {
            continue;
        }
        content |= CONTENT_MARKERS.contains(&utt.at(i - 1));
        if utt.is_quoted(i) || !CONJUNCTIONS.contains(&utt.at(i)) {
            continue;
        }
        if verb_classes(utt.at(start)) & INTERCOM != 0
            && intercom(&utt.slice(start, utt.len())).is_some()
        {
            break;
        }
        if content {
            return Vec::new();
        }
        let next = connector_end(utt, i);
        if clause_head(utt, next) {
            segments.push(utt.slice(start, i));
            start = next;
            content = false;
        }
    }
    if !segments.is_empty() {
        segments.push(utt.slice(start, utt.len()));
    }
    segments
}

fn split_objects<'a>(utt: &Utterance<'a>, verb: Verb<'_>) -> Vec<Utterance<'a>> {
    let Some((family, _, noun)) = control_object(utt, verb) else {
        return Vec::new();
    };
    if family == Reminder && matches!(utt.at(noun + 1), "de" | "para" | "que") {
        return Vec::new();
    }
    let mut cuts = (noun + 1..utt.len())
        .filter_map(|i| {
            if utt.is_quoted(i) || utt.at(i) != "y" {
                return None;
            }
            let next = i + 1 + usize::from(utt.at(i + 1) == "tambien");
            governed(utt, next, false, false, false).map(|_| (i, next))
        })
        .peekable();
    let Some((first, _)) = cuts.peek().copied() else {
        return Vec::new();
    };
    let mut segments = vec![utt.slice(0, first)];
    let head = utt.slice(0, verb.end);
    while let Some((_, start)) = cuts.next() {
        let end = cuts.peek().map_or(utt.len(), |&(end, _)| end);
        segments.push(head.clone().join(&utt.slice(start, end)));
    }
    segments
}

fn room_at(utt: &Utterance<'_>, index: usize) -> bool {
    grammar::ROOM_PREPS.iter().any(|prep| {
        let mut i = index;
        prep.split_whitespace().all(|word| {
            let ok = utt.at(i) == word;
            i += 1;
            ok
        }) && grammar::canonical_room(utt.at(i)).is_some()
    })
}

fn split_targets<'a>(utt: &Utterance<'a>, verb: Verb<'_>) -> Vec<Utterance<'a>> {
    let domotics = verb.has(DOMOTICS);
    if !domotics && !verb.has(CANCEL | CLEAR | STOP | STOP_RINGING | PAUSE | RESUME | DISABLE) {
        return Vec::new();
    }
    let own_object = governed(utt, verb.end, false, false, false).is_some();
    let lists = (0..utt.len()).any(|i| object(utt, i, false) == Some(List));
    let mut segments = Vec::new();
    let mut carried = utt.slice(0, 0);
    let mut start = 0;
    let mut content = false;
    for i in verb.end..utt.len() {
        content |= CONTENT_MARKERS.contains(&utt.at(i - 1));
        if utt.is_quoted(i) || utt.at(i) != "y" || i <= start {
            continue;
        }
        if content {
            return Vec::new();
        }
        let next = i + 1 + usize::from(utt.at(i + 1) == "tambien");
        let target = next + usize::from(DETERMINERS.contains(&utt.at(next)));
        let repeat_room = domotics && !lists && room_at(utt, next);
        let repeat_device =
            domotics && !lists && target > next && grammar::entity_at(utt.at(target)).is_some();
        let repeat_object = !own_object && governed(utt, next, false, false, false).is_some();
        if !(repeat_room || repeat_device || repeat_object) {
            continue;
        }
        let previous = carried.join(&utt.slice(start, i));
        carried = if repeat_room {
            let end = (verb.end..previous.len())
                .find(|&i| room_at(&previous, i))
                .unwrap_or(previous.len());
            previous.slice(0, end)
        } else {
            utt.slice(0, verb.end)
        };
        segments.push(previous);
        start = next;
    }
    if !segments.is_empty() {
        segments.push(carried.join(&utt.slice(start, utt.len())));
    }
    segments
}

fn conflicts(parts: &[HouseholdPart]) -> bool {
    parts
        .iter()
        .filter_map(|part| part.command.as_ref())
        .enumerate()
        .any(|(i, left)| {
            parts
                .iter()
                .filter_map(|part| part.command.as_ref())
                .skip(i + 1)
                .any(|right| {
                    left.domain == right.domain
                        && left.room == right.room
                        && matches!(
                            (left.operation, right.operation),
                            (DeviceOperation::On, DeviceOperation::Off)
                                | (DeviceOperation::Off, DeviceOperation::On)
                                | (DeviceOperation::OpenCover, DeviceOperation::CloseCover)
                                | (DeviceOperation::CloseCover, DeviceOperation::OpenCover)
                                | (DeviceOperation::MediaPlay, DeviceOperation::MediaPause)
                                | (DeviceOperation::MediaPause, DeviceOperation::MediaPlay)
                        )
                })
        })
}

pub(super) fn compound(
    utt: &Utterance<'_>,
    settings: &HouseholdIntentSettings,
    source_room: Option<&str>,
) -> Option<HouseholdIntent> {
    let verb = verb_at(utt, 0, false)?;
    let mut segments = split_clauses(utt);
    if segments.is_empty() {
        segments = split_objects(utt, verb);
    }
    if segments.is_empty() {
        segments = split_targets(utt, verb);
    }
    if segments.len() < 2 {
        return None;
    }
    let clarify = |reason| {
        HouseholdIntent::new(Compound, Clarify, "R5", utt.heard(), verb.form).because(reason)
    };
    let intents: Vec<_> = segments.iter().map(single).collect();
    if intents
        .iter()
        .flatten()
        .any(HouseholdIntent::controls_routine)
    {
        return Some(clarify("routine_control_part"));
    }
    if segments.len() > settings.compound_max_parts {
        return Some(clarify("too_many_parts"));
    }
    if !settings.compound_execute {
        return Some(clarify("compound_disabled"));
    }
    let mut parts = Vec::with_capacity(segments.len());
    for (segment, intent) in segments.iter().zip(intents) {
        let text = segment.grammar_text();
        let security_refusal = intent.is_none() && grammar::security_order(&text);
        if intent.as_ref().is_some_and(|i| {
            i.family == DeferredAction || (i.family == Reminder && i.operation == Create)
        }) || (intent.is_none()
            && ((!security_refusal
                && grammar::non_immediate(&text) == Some(GrammarBlock::Deferred))
                || (security_refusal && grammar::has_deferral_time(&fold(&text)))))
        {
            return Some(clarify("deferred_part"));
        }
        let command = if intent.is_none() && !security_refusal {
            match grammar::parse_core(&text, source_room) {
                GrammarMatch::Command(command) => Some(command),
                _ => return Some(clarify("unparsed_part")),
            }
        } else {
            None
        };
        if intent.as_ref().is_some_and(|i| i.operation == Clarify) {
            return Some(clarify("unparsed_part"));
        }
        parts.push(HouseholdPart {
            segment: segment.heard(),
            intent,
            command,
            security_refusal,
        });
    }
    if conflicts(&parts) {
        return Some(clarify("conflicting_parts"));
    }
    Some(HouseholdIntent {
        parts,
        ..HouseholdIntent::new(Compound, Execute, "R5", utt.heard(), verb.form)
    })
}
