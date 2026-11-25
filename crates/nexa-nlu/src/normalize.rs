//! Text views for matching Spanish commands and preserving spoken content.

use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

pub const WAKE_WORDS: &[&str] = &[
    "nexa", "kaza", "alexa", "nexia", "neza", "nexta", "nesa", "next", "necks", "casa",
];

/// Lowercase NFC text with canonical whitespace. Voseo accents remain significant.
pub fn normalize_utterance(raw: &str) -> String {
    collapse(&raw.to_lowercase().nfc().collect::<String>())
}

/// Lowercase via NFD, removing combining marks (including the tilde in ñ).
pub fn fold(text: &str) -> String {
    text.to_lowercase()
        .nfd()
        .filter(|c| !is_combining_mark(*c))
        .collect()
}

/// Fold accents and turn punctuation into word boundaries.
pub fn words(text: &str) -> String {
    let plain: String = fold(text)
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                ' '
            }
        })
        .collect();
    collapse(&plain)
}

pub(crate) fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(crate) fn has_phrase(text: &str, phrase: &str) -> bool {
    text.match_indices(phrase).any(|(start, _)| {
        let end = start + phrase.len();
        (start == 0 || text.as_bytes()[start - 1] == b' ')
            && (end == text.len() || text.as_bytes()[end] == b' ')
    })
}

pub(crate) fn prefix(text: &str, phrase: &str) -> bool {
    text == phrase
        || text
            .strip_prefix(phrase)
            .is_some_and(|tail| tail.starts_with(' '))
}

pub(crate) fn strip_head(text: &str, wakes: &[String]) -> String {
    let mut text = normalize_utterance(text);
    for _ in 0..6 {
        let core = text.trim_start_matches(['¿', '¡', ' ']);
        let next = wakes
            .iter()
            .map(String::as_str)
            .chain(WAKE_WORDS.iter().copied())
            .chain([
                "por favor",
                "porfa",
                "che",
                "hola",
                "ey",
                "oye",
                "bueno",
                "bue",
                "dale",
            ])
            .filter(|word| !word.is_empty())
            .find_map(|word| {
                let tail = core.get(word.len()..)?;
                (core
                    .get(..word.len())
                    .is_some_and(|head| fold(head) == word)
                    && (tail.is_empty() || tail.starts_with([' ', ',', '.', ':', ';', '-'])))
                .then(|| tail.trim_start_matches([' ', ',', '.', ':', ';', '-']))
            });
        match next {
            Some(tail) => text = tail.to_owned(),
            None => break,
        }
    }
    if let Some(tail) = text
        .strip_suffix("por favor")
        .or_else(|| text.strip_suffix("porfa"))
        && tail.ends_with([' ', ','])
    {
        text.truncate(tail.trim_end_matches([' ', ',']).len());
    }
    text
}
