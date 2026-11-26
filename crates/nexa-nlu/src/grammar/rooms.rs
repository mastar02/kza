//! Explicit room aliases and target resolution.

use crate::normalize::{fold, has_phrase};

use super::{Domain, GrammarBlock, canonical_room};

pub(super) const ROOMS: &[(&str, &str)] = &[
    ("escritorio", "escritorio"),
    ("oficina", "escritorio"),
    ("estudio", "escritorio"),
    ("living", "living"),
    ("sala", "living"),
    ("salon", "living"),
    ("libby", "living"),
    ("livin", "living"),
    ("libin", "living"),
    ("hall", "hall"),
    ("pasillo", "hall"),
    ("entrada", "hall"),
    ("escalera", "hall"),
    ("cocina", "cocina"),
    ("kitchen", "cocina"),
    ("bano", "bano"),
    ("banio", "bano"),
    ("bathroom", "bano"),
    ("cuarto", "cuarto"),
    ("dormitorio", "cuarto"),
    ("habitacion", "cuarto"),
    ("pieza", "cuarto"),
    ("balcon", "balcon"),
];
pub(crate) const ROOM_PREPS: &[&str] = &[
    "de la", "de el", "en el", "en la", "a la", "del", "al", "de", "en",
];

fn unqualified_target(target: &str) -> &str {
    ROOM_PREPS
        .iter()
        .find_map(|prep| target.strip_prefix(prep)?.strip_prefix(' '))
        .unwrap_or(target)
}

/// An explicit unresolved location is a clarification, never a microphone fallback.
pub(super) fn resolve_room(
    target: &str,
    source_room: Option<&str>,
    domain: Domain,
    noun: &str,
) -> Result<Option<String>, GrammarBlock> {
    let explicit = unqualified_target(target);
    if let Some(room) = canonical_room(explicit) {
        return Ok(Some(room.into()));
    }
    if matches!(
        explicit,
        "hogar" | "toda la casa" | "todo el hogar" | "la casa entera"
    ) {
        return Ok(None);
    }
    if explicit != target && (domain != Domain::Light || matches!(noun, "luz" | "luces" | "brillo"))
    {
        return Err(GrammarBlock::UnknownRoom);
    }
    if target.is_empty() {
        return Ok(source_room.map(|room| canonical_room(&fold(room)).unwrap_or(room).to_owned()));
    }
    // Named fixtures can be resolved by the caller, with explicit scope when present.
    for (index, _) in target.match_indices(' ') {
        let tail = &target[index + 1..];
        let explicit = unqualified_target(tail);
        if let Some(room) = canonical_room(explicit)
            && explicit != tail
        {
            return Ok(Some(room.into()));
        }
    }
    if ["del", "de la", "de el", "en", "a la"]
        .iter()
        .any(|p| has_phrase(target, p))
    {
        return Err(GrammarBlock::UnknownRoom);
    }
    let known_fixture = matches!(
        (noun, target),
        ("luz" | "lampara", "principal")
            | ("luces" | "lamparas", "principales")
            | ("lampara" | "lamparas", "de pie")
    );
    if known_fixture || domain == Domain::MediaPlayer {
        return Ok(source_room.map(str::to_owned));
    }
    Err(GrammarBlock::UnknownRoom)
}

pub(super) fn has_multiple_targets(target: &str) -> bool {
    target
        .split_whitespace()
        .any(|word| matches!(word, "y" | "tambien" | "o"))
}
