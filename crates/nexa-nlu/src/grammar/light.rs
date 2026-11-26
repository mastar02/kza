//! Light commands, brightness, color and color temperature.

use crate::normalize::{fold, has_phrase, words};

use super::lexicon::Verb;
use super::rooms::has_multiple_targets;
use super::slots::{bounded_percent, cardinal, number};
use super::{CommandParts, GrammarBlock, Operation, SlotExtractor, Slots};

const BRIGHTNESS_LEVELS: &[(&str, u8)] = &[
    ("maximo", 100),
    ("maxima", 100),
    ("full", 100),
    ("todo", 100),
    ("fuerte", 90),
    ("potente", 90),
    ("medio", 50),
    ("media", 50),
    ("mitad", 50),
    ("suave", 30),
    ("bajo", 30),
    ("baja", 30),
    ("tenue", 15),
    ("minimo", 10),
    ("minima", 10),
];

pub(super) fn brightness(text: &str) -> Result<Option<u8>, GrammarBlock> {
    let patterns = [&*EXPLICIT_PERCENT, &*BARE_PERCENT];
    let mut result = None;
    for pattern in patterns {
        for capture in pattern.captures_iter(text) {
            let value = capture[1]
                .parse::<u8>()
                .ok()
                .filter(|n| *n <= 100)
                .filter(|_| capture[1].bytes().all(|byte| byte.is_ascii_digit()))
                .ok_or(GrammarBlock::InvalidBrightness)?;
            result.get_or_insert(value);
        }
    }
    if result.is_some() {
        return Ok(result);
    }
    let spoken = SPOKEN_PERCENT
        .captures_iter(text)
        .find_map(|m| cardinal(&m[1]));
    if spoken.is_some() {
        return Ok(spoken);
    }
    let text = words(text);
    Ok(BRIGHTNESS_LEVELS
        .iter()
        .find_map(|&(w, n)| has_phrase(&text, w).then_some(n)))
}

const COLORS: &[(&str, [u8; 3])] = &[
    ("rojo", [255, 0, 0]),
    ("roja", [255, 0, 0]),
    ("verde", [0, 255, 0]),
    ("azul", [0, 0, 255]),
    ("amarillo", [255, 255, 0]),
    ("amarilla", [255, 255, 0]),
    ("naranja", [255, 128, 0]),
    ("naranjo", [255, 128, 0]),
    ("rosa", [255, 105, 180]),
    ("rosado", [255, 105, 180]),
    ("rosada", [255, 105, 180]),
    ("violeta", [148, 0, 211]),
    ("morado", [148, 0, 211]),
    ("morada", [148, 0, 211]),
    ("purpura", [148, 0, 211]),
    ("celeste", [135, 206, 235]),
    ("turquesa", [64, 224, 208]),
    ("cian", [0, 255, 255]),
    ("blanco", [255, 255, 255]),
    ("blanca", [255, 255, 255]),
    ("dorado", [255, 215, 0]),
    ("dorada", [255, 215, 0]),
    ("plateado", [192, 192, 192]),
    ("plateada", [192, 192, 192]),
    ("negro", [0, 0, 0]),
    ("negra", [0, 0, 0]),
    ("magenta", [255, 0, 255]),
    ("fucsia", [255, 0, 255]),
];

const TEMPS: &[(&str, u16)] = &[
    ("calida", 2700),
    ("calido", 2700),
    ("tibia", 2700),
    ("tibio", 2700),
    ("amarilla", 2700),
    ("amarillenta", 2700),
    ("neutra", 4000),
    ("neutro", 4000),
    ("natural", 4000),
    ("fria", 6500),
    ("frio", 6500),
    ("azulada", 6500),
    ("fluorescente", 5500),
];

impl SlotExtractor {
    /// Extract the supported light slots; malformed numerical values block the request.
    pub fn extract(&self, text: &str) -> Result<Slots, GrammarBlock> {
        let text = fold(text);
        let numeric = NUMERIC_KELVIN.captures(&text);
        let plain = words(&text);
        let color_temp_kelvin = if let Some(m) = numeric {
            Some(
                m[1].parse::<u16>()
                    .ok()
                    .filter(|n| (1500..=9000).contains(n))
                    .ok_or(GrammarBlock::Incomplete)?,
            )
        } else {
            TEMPS
                .iter()
                .find_map(|&(w, n)| has_phrase(&plain, w).then_some(n))
        };
        Ok(Slots {
            brightness_pct: brightness(&text)?,
            color_temp_kelvin,
            rgb_color: COLORS
                .iter()
                .find_map(|&(w, rgb)| has_phrase(&plain, w).then_some(rgb)),
            ..Slots::default()
        })
    }
}

pub(super) fn light<'a>(verb: Option<Verb>, noun: &str, tail: &'a str) -> CommandParts<'a> {
    let tail = strip_light_reason(tail);
    let (target, settings) = split_light_settings(tail);
    let target = target.trim_end_matches([',', ' ']);
    let target = if matches!(target, "encendida" | "encendidas" | "apagada" | "apagadas") {
        ""
    } else {
        target
    };
    let mut slots = Slots::default();
    if let Some(settings) = settings {
        slots = parse_light_settings(settings)?;
    }
    let set = slots != Slots::default();
    let operation = match verb {
        Some(Verb::On) => Operation::On,
        Some(Verb::Off) if !set => Operation::Off,
        Some(Verb::Raise) if !set => Operation::On,
        Some(Verb::Set | Verb::Raise | Verb::Lower) | None if set => Operation::SetLight,
        Some(Verb::Set | Verb::Lower) => return Err(GrammarBlock::Incomplete),
        _ => return Ok(None),
    };
    if verb.is_none()
        && (!matches!(noun, "luz" | "luces" | "brillo")
            || slots.brightness_pct.is_none()
            || slots.color_temp_kelvin.is_some()
            || slots.rgb_color.is_some())
    {
        return Ok(None);
    }
    if has_multiple_targets(target) {
        return Err(GrammarBlock::Incomplete);
    }
    Ok(Some((operation, target, slots)))
}

fn strip_light_reason(tail: &str) -> &str {
    LIGHT_REASON.find(tail).map_or(tail, |m| &tail[..m.start()])
}

fn split_light_settings(text: &str) -> (&str, Option<&str>) {
    for token in TOKEN.find_iter(text) {
        let start = token.start();
        let word = token.as_str();
        let suffix = &text[start..];
        let values = suffix.trim_start_matches(' ');
        let word = word.trim_end_matches('%');
        if LIGHT_ADJUSTMENT.is_match(values)
            || ["a ", "al ", "el ", "del "]
                .iter()
                .any(|p| values.strip_prefix(p).and_then(number).is_some())
            || word.chars().next().is_some_and(|c| c.is_ascii_digit())
            || TEMPS.iter().any(|&(name, _)| name == word)
            || COLORS.iter().any(|&(name, _)| name == word)
            || BRIGHTNESS_LEVELS.iter().any(|&(name, _)| name == word)
        {
            let target = text[..start].trim_end();
            let target = [
                "y al", "y a", "y en", "y con", "hasta el", "al", "a", "en", "con", "y",
            ]
            .iter()
            .find_map(|p| {
                target
                    .strip_suffix(p)
                    .filter(|left| left.is_empty() || left.ends_with(' '))
            })
            .unwrap_or(target)
            .trim_end();
            return (target, Some(values));
        }
    }
    (text, None)
}

fn parse_light_settings(text: &str) -> Result<Slots, GrammarBlock> {
    let text = ["hasta el ", "al ", "a ", "en ", "con ", "el ", "del "]
        .iter()
        .find_map(|p| text.strip_prefix(p))
        .unwrap_or(text);
    let (setting, brightness_tail) = LIGHT_BRIGHTNESS_SETTING
        .captures(text)
        .and_then(|m| Some((m.get(1)?.as_str(), Some(m.get(2)?.as_str()))))
        .unwrap_or((text, None));
    let slots = SlotExtractor.extract(setting)?;
    let percent = |value: &str| {
        let value = PERCENT_SUFFIX.replace(value, "");
        number(&value).and_then(bounded_percent)
    };
    if let Some(value) = percent(setting)
        && brightness_tail.is_none()
    {
        return Ok(Slots {
            brightness_pct: Some(value),
            ..Slots::default()
        });
    }
    let word = setting.split_whitespace().next().unwrap_or_default();
    let single = word == setting
        && (slots.brightness_pct.is_some()
            || slots.color_temp_kelvin.is_some()
            || slots.rgb_color.is_some());
    let kelvin = KELVIN_SETTING.is_match(setting);
    if !single && !kelvin {
        return Err(GrammarBlock::Incomplete);
    }
    if let Some(raw) = brightness_tail {
        let value = percent(raw)
            .filter(|level| *level > 0)
            .ok_or(GrammarBlock::Incomplete)?;
        return Ok(Slots {
            brightness_pct: Some(value),
            ..slots
        });
    }
    Ok(slots)
}

regex_patterns! {
    EXPLICIT_PERCENT => r"(?:^|[^\w.,+\-−])([+\-−]?(?:[0-9]+(?:[.,][0-9]+)?|[.,][0-9]+)) *(?:%|por *ciento\b)(?:$|[^\w]|\b)";
    BARE_PERCENT => r"\bal ([+\-−]?(?:[0-9]+(?:[.,][0-9]+)?|[.,][0-9]+))($|[^\w.,])";
    SPOKEN_PERCENT => r"\b(\w+(?: y \w+)?) (?:por ciento|porciento)\b";
    NUMERIC_KELVIN => r"\b([0-9]+) *(?:k|kelvin)\b";
    LIGHT_REASON => r", (?:hace (?:frio|calor)|llueve|esta lloviendo)\b";
    TOKEN => r"\S+";
    LIGHT_ADJUSTMENT => r"^(?:al?|con|hasta el) (?:[+\-−]?[0-9]|[.,][0-9]|[a-z]+ (?:y [a-z]+ )?(?:por *ciento|porciento)\b)";
    LIGHT_BRIGHTNESS_SETTING => r"^(.*?)(?: y)? (?:al|a|con) (.+)$";
    PERCENT_SUFFIX => r" *(?:%|por *ciento|porciento)$";
    KELVIN_SETTING => r"^[0-9]+ *(?:k|kelvin)$";
}
