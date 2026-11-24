"""Inline wake-word clause extractor for continuous conversation streams.

When a user says ``"Nexa <action>"`` in the middle of an ongoing human
conversation (e.g. ``"che mirá, listo, Nexa prendé la luz del living, y
después te cuento"``), forwarding the entire 15-30s transcript to
``CommandAcceptanceGate`` and ``LLMCommandRouter`` causes two failure modes:

1. Earlier filler/noise words (``"listo"``, ``"perfecto"``, ``"mañana"``)
   trigger hard rejections in ``CommandGate`` even though the command itself
   is clean.
2. Entities or rooms mentioned earlier/later in the conversation pollute
   entity resolution and inflate the 7B router prompt budget.

``extract_inline_command`` isolates the bounded command clause starting at the
wake mention while flagging ``embedded_in_conversation=True`` so downstream
routing can execute the action without opening an intrusive follow-up window.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from src.ambient.textual_wake import (
    _DEFAULT_VARIANTS,
    _FUZZY_DENYLIST,
    _within_edit_distance,
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Discourse markers right after a comma that typically resume human chat
# rather than continuing a home-automation command.
_COMMA_RESUME_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("bueno",),
    ("che",),
    ("entonces",),
    ("pero",),
    ("igual",),
    ("viste",),
    ("mira",),
    ("nada",),
    ("como", "te", "decia"),
    ("te", "decia"),
    ("te", "seguia"),
    ("y", "despues", "te"),
    ("despues", "te"),
    ("en", "fin"),
)

# Multi-word phrases that clearly resume human chat even without a comma.
_INLINE_RESUME_PHRASES: tuple[tuple[str, ...], ...] = (
    ("bueno", "te", "seguia"),
    ("como", "te", "decia"),
    ("te", "seguia", "contando"),
    ("te", "decia", "que"),
    ("y", "despues", "te", "cuento"),
    ("en", "fin"),
)

_STRONG_STOP_CHARS = frozenset(".?!;\n—")


def _norm_token(raw: str) -> str:
    norm = unicodedata.normalize("NFD", raw.lower())
    return "".join(c for c in norm if unicodedata.category(c) != "Mn")


@dataclass(frozen=True)
class InlineCommandMatch:
    """Extracted command clause from a continuous-speech transcript."""

    command_text: str
    matched_variant: str
    prefix_text: str
    suffix_text: str
    embedded_in_conversation: bool


def _match_wake_at(
    norm_tokens: list[str],
    idx: int,
    single_variants: list[str],
    bigram_variants: list[tuple[str, ...]],
    max_edit_distance: int,
) -> tuple[int, int, str] | None:
    """Return ``(start_idx, end_idx_exclusive, matched_variant)`` if wake starts at ``idx``."""
    tok = norm_tokens[idx]
    if tok in single_variants:
        return (idx, idx + 1, tok)
    if bigram_variants and idx + 1 < len(norm_tokens):
        pair = (tok, norm_tokens[idx + 1])
        if pair in bigram_variants:
            return (idx, idx + 2, f"{pair[0]} {pair[1]}")
    if max_edit_distance > 0 and tok not in _FUZZY_DENYLIST:
        for base in single_variants:
            if _within_edit_distance(tok, base, max_edit_distance):
                return (idx, idx + 1, tok)
    return None


def _find_wake_token_span(
    norm_tokens: list[str],
    variants: tuple[str, ...],
    max_edit_distance: int,
    *,
    start_from: int = 0,
) -> tuple[int, int, str] | None:
    """Return ``(start_idx, end_idx_exclusive, matched_variant)`` for the next wake."""
    norm_variants = tuple(v.strip().lower() for v in variants if v and v.strip())
    single_variants = [v for v in norm_variants if " " not in v]
    bigram_variants = [tuple(v.split()) for v in norm_variants if " " in v and len(v.split()) == 2]

    for i in range(start_from, len(norm_tokens)):
        tok = norm_tokens[i]
        if tok in single_variants:
            return (i, i + 1, tok)
        if bigram_variants and i + 1 < len(norm_tokens):
            pair = (tok, norm_tokens[i + 1])
            if pair in bigram_variants:
                return (i, i + 2, f"{pair[0]} {pair[1]}")

    if max_edit_distance > 0:
        for i in range(start_from, len(norm_tokens)):
            tok = norm_tokens[i]
            if tok in _FUZZY_DENYLIST:
                continue
            for base in single_variants:
                if _within_edit_distance(tok, base, max_edit_distance):
                    return (i, i + 1, tok)
    return None


def _matches_prefix(tokens: list[str], idx: int, pattern: tuple[str, ...]) -> bool:
    if idx + len(pattern) > len(tokens):
        return False
    return tuple(tokens[idx : idx + len(pattern)]) == pattern


def extract_all_inline_commands(
    text: str,
    *,
    variants: tuple[str, ...] = _DEFAULT_VARIANTS,
    max_edit_distance: int = 1,
    max_command_words: int = 14,
    min_words_before_stop: int = 2,
) -> list[InlineCommandMatch]:
    """Extract every bounded ``Nexa <action>`` clause in order from a transcript.

    Handles back-to-back commands inside one continuous segment (e.g.
    ``"Hola. Nexa, prendé la luz del escritorio. Nexa, ¿qué temperatura hace? ..."``)
    as well as adjacent commands without punctuation between them.
    """
    if not text or not text.strip():
        return []
    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return []
    norm_tokens = [_norm_token(m.group(0)) for m in matches]
    norm_variants = tuple(v.strip().lower() for v in variants if v and v.strip())
    single_variants = [v for v in norm_variants if " " not in v]
    bigram_variants = [tuple(v.split()) for v in norm_variants if " " in v and len(v.split()) == 2]

    results: list[InlineCommandMatch] = []
    cursor = 0
    while cursor < len(norm_tokens):
        span = None
        for i in range(cursor, len(norm_tokens)):
            span = _match_wake_at(
                norm_tokens, i, single_variants, bigram_variants, max_edit_distance
            )
            if span is not None:
                break
        if span is None:
            break
        wake_start_idx, wake_end_idx, matched_variant = span

        char_start = matches[wake_start_idx].start()
        max_end_idx = min(len(matches), wake_end_idx + max(1, max_command_words))
        end_token_idx = max_end_idx

        for idx in range(wake_end_idx, max_end_idx):
            words_after_wake = idx - wake_end_idx
            # Always stop before a subsequent wake mention (even without punctuation).
            if words_after_wake >= 1 and _match_wake_at(
                norm_tokens, idx, single_variants, bigram_variants, max_edit_distance
            ) is not None:
                end_token_idx = idx
                break
            if words_after_wake >= min_words_before_stop:
                between = text[matches[idx - 1].end() : matches[idx].start()]
                if any(ch in _STRONG_STOP_CHARS for ch in between):
                    end_token_idx = idx
                    break
                if "," in between and any(
                    _matches_prefix(norm_tokens, idx, pat) for pat in _COMMA_RESUME_PREFIXES
                ):
                    end_token_idx = idx
                    break
                if any(_matches_prefix(norm_tokens, idx, pat) for pat in _INLINE_RESUME_PHRASES):
                    end_token_idx = idx
                    break

        if end_token_idx <= wake_start_idx:
            end_token_idx = wake_end_idx
        char_end = matches[end_token_idx - 1].end()
        if char_end < len(text) and text[char_end] in "?!":
            char_end += 1

        command_text = text[char_start:char_end].strip()
        prefix_text = text[:char_start].strip(" ,;:-—\t\n")
        suffix_text = text[char_end:].strip(" ,;:-—\t\n")
        embedded = bool(prefix_text) or bool(suffix_text)
        results.append(
            InlineCommandMatch(
                command_text=command_text,
                matched_variant=matched_variant,
                prefix_text=prefix_text,
                suffix_text=suffix_text,
                embedded_in_conversation=embedded,
            )
        )
        cursor = max(end_token_idx, wake_end_idx)
    return results


def extract_inline_command(
    text: str,
    *,
    variants: tuple[str, ...] = _DEFAULT_VARIANTS,
    max_edit_distance: int = 1,
    max_command_words: int = 14,
    min_words_before_stop: int = 2,
) -> InlineCommandMatch | None:
    """Extract the first bounded ``Nexa <action>`` clause from a conversation transcript."""
    found = extract_all_inline_commands(
        text,
        variants=variants,
        max_edit_distance=max_edit_distance,
        max_command_words=max_command_words,
        min_words_before_stop=min_words_before_stop,
    )
    return found[0] if found else None
