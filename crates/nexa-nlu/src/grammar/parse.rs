//! Parse device commands and dispatch domain-specific settings.

use crate::normalize::{fold, prefix};

use super::guards::{non_immediate, security_order};
use super::lexicon::{Verb, action, entity_at};
use super::light::light;
use super::rooms::{has_multiple_targets, resolve_room};
use super::slots::{bounded_percent, number};
use super::{
    CommandParts, Domain, GrammarBlock, GrammarMatch, HomeCommand, HvacMode, Operation, Slots,
};

pub(crate) fn parse_core(core: &str, source_room: Option<&str>) -> GrammarMatch {
    if security_order(core) {
        return GrammarMatch::Blocked(GrammarBlock::Security);
    }
    if let Some(reason) = non_immediate(core) {
        return GrammarMatch::Blocked(reason);
    }
    let folded = fold(
        core.trim_start_matches(['¡', '¿', '?'])
            .trim_end_matches(['!', '.', ',', '?', '¿']),
    );
    let polite = ["podes ", "puedes ", "podrias ", "podria "]
        .iter()
        .find_map(|lead| folded.strip_prefix(lead));
    let direct_question = core.starts_with(['¿', '?']) || core.ends_with(['?', '¿']);
    if (direct_question && polite.is_none()) || folded.contains([';', '?', '¿']) {
        return GrammarMatch::Unclaimed;
    }
    let core = polite.unwrap_or(&folded);
    let core = core.strip_suffix(" ahora").unwrap_or(core);
    let (head, rest) = core.split_once(' ').unwrap_or((core, ""));
    let verb = action(head);
    let rest = trim_article(if verb.is_some() { rest } else { core });
    let (noun, tail) = rest.split_once(' ').unwrap_or((rest, ""));
    let (domain, tail) = match entity_at(noun) {
        Some(Domain::Climate) if noun == "aire" && prefix(tail, "acondicionado") => (
            Domain::Climate,
            tail.strip_prefix("acondicionado")
                .unwrap_or(tail)
                .trim_start(),
        ),
        Some(Domain::Light) if noun == "brillo" => (
            Domain::Light,
            tail.strip_prefix("de la luz ")
                .or_else(|| tail.strip_prefix("de las luces "))
                .unwrap_or(tail),
        ),
        Some(domain) => (domain, tail),
        None if matches!(head, "ponela" | "ponelo") && core.split_whitespace().count() <= 6 => {
            (Domain::Light, rest)
        }
        None if matches!(verb, Some(Verb::On | Verb::Off))
            && super::canonical_room(noun).is_some() =>
        {
            (Domain::Light, rest)
        }
        None => return GrammarMatch::Unclaimed,
    };
    let parsed = if domain == Domain::Light {
        light(verb, noun, tail)
    } else {
        extended(domain, verb, tail)
    };
    let (operation, target, slots) = match parsed {
        Ok(Some(value)) => value,
        Ok(None) => return GrammarMatch::Unclaimed,
        Err(reason) => return GrammarMatch::Blocked(reason),
    };
    let fixture = domain == Domain::Light
        && !matches!(noun, "luz" | "luces" | "brillo")
        && entity_at(noun).is_some();
    let room = if domain == Domain::Scene {
        None
    } else {
        match resolve_room(target, source_room, domain, noun) {
            Ok(room) => room,
            Err(reason) => return GrammarMatch::Blocked(reason),
        }
    };
    let target = if fixture {
        [noun, target].join(" ").trim_end().to_owned()
    } else {
        target.to_owned()
    };
    GrammarMatch::Command(HomeCommand {
        domain,
        operation,
        target,
        room,
        slots,
    })
}

fn trim_article(text: &str) -> &str {
    ["el ", "la ", "los ", "las "]
        .iter()
        .find_map(|p| text.strip_prefix(p))
        .unwrap_or(text)
}

fn extended(domain: Domain, verb: Option<Verb>, tail: &str) -> CommandParts<'_> {
    let Some(verb) = verb else {
        return Ok(None);
    };
    let mut slots = Slots::default();
    let (operation, target) = match (domain, verb) {
        (Domain::Scene, Verb::On | Verb::Set) if !tail.is_empty() => {
            (Operation::ActivateScene, tail)
        }
        (Domain::Climate, Verb::Set) => {
            if let Some(m) = TEMPERATURE_SETTING.captures(tail)
                && let Some(value) = number(m[2].trim())
            {
                if !(10.0..=35.0).contains(&value) {
                    return Err(GrammarBlock::InvalidTemperature);
                }
                slots.temperature = Some(value);
                (
                    Operation::SetTemperature,
                    m.get(1).map_or("", |m| m.as_str()),
                )
            } else if let Some(m) = HVAC_MODE.captures(tail) {
                slots.hvac_mode = Some(match &m[2] {
                    "apagado" | "apagada" | "off" => HvacMode::Off,
                    "frio" | "fria" | "refrigeracion" => HvacMode::Cool,
                    "calor" | "calefaccion" => HvacMode::Heat,
                    "seco" | "deshumidificacion" | "deshumidificar" => HvacMode::Dry,
                    "ventilacion" | "solo ventilador" => HvacMode::FanOnly,
                    _ => HvacMode::Auto,
                });
                (Operation::SetHvacMode, m.get(1).map_or("", |m| m.as_str()))
            } else {
                return Err(GrammarBlock::Incomplete);
            }
        }
        (Domain::Climate, Verb::Off) => {
            slots.hvac_mode = Some(HvacMode::Off);
            (Operation::SetHvacMode, tail)
        }
        (Domain::Climate | Domain::Fan, Verb::On) => (Operation::On, tail),
        (Domain::Fan, Verb::Off) => (Operation::Off, tail),
        (Domain::Cover, Verb::Set | Verb::Raise | Verb::Lower | Verb::Open | Verb::Close) => {
            if let Some(m) = COVER_POSITION.captures(tail) {
                slots.position = Some(
                    number(m[2].trim())
                        .and_then(bounded_percent)
                        .ok_or(GrammarBlock::Incomplete)?,
                );
                (
                    Operation::SetCoverPosition,
                    m.get(1).map_or("", |m| m.as_str()),
                )
            } else if MALFORMED_COVER_POSITION.is_match(tail) {
                return Err(GrammarBlock::Incomplete);
            } else {
                let operation = match verb {
                    Verb::Raise | Verb::Open => Operation::OpenCover,
                    Verb::Lower | Verb::Close => Operation::CloseCover,
                    _ => return Err(GrammarBlock::Incomplete),
                };
                (operation, tail)
            }
        }
        (Domain::Cover, Verb::Stop) => (Operation::StopCover, tail),
        (Domain::MediaPlayer, Verb::Set | Verb::Play) => (Operation::MediaPlay, tail),
        (Domain::MediaPlayer, Verb::Pause | Verb::Stop) => (Operation::MediaPause, tail),
        (Domain::MediaPlayer, Verb::Next) => (Operation::MediaNext, tail),
        (Domain::Scene | Domain::Climate | Domain::Cover, _) => {
            return Err(GrammarBlock::Incomplete);
        }
        _ => return Ok(None),
    };
    if domain != Domain::Scene && has_multiple_targets(target) {
        return Err(GrammarBlock::Incomplete);
    }
    Ok(Some((operation, target, slots)))
}

regex_patterns! {
    TEMPERATURE_SETTING => r"^(?:(.*?) )?(?:a|en) (.+?)(?: *(?:grados?(?: celsius| centigrados)?|° *c|celsius))?$";
    HVAC_MODE => r"^(?:(.*?) )?(?:en )?(?:modo )?(apagado|apagada|off|frio|fria|refrigeracion|calor|calefaccion|seco|deshumidificacion|deshumidificar|ventilacion|solo ventilador|automatico|automatica|auto)$";
    COVER_POSITION => r"^(?:(.*?) )?(?:al?|hasta el) (.+?) *(?:%|por ciento|porciento)$";
    MALFORMED_COVER_POSITION => r"\b(?:al?|hasta el) [0-9]|%|\bporciento\b|\bpor ciento\b";
}
