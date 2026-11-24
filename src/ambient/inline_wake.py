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


def _find_wake_token_span(
    norm_tokens: list[str],
    variants: tuple[str, ...],
    max_edit_distance: int,
) -> tuple[int, int, str] | None:
    """Return ``(start_idx, end_idx_exclusive, matched_variant)`` for the first wake."""
    norm_variants = tuple(v.strip().lower() for v in variants if v and v.strip())
    single_variants = [v for v in norm_variants if " " not in v]
    bigram_variants = [tuple(v.split()) for v in norm_variants if " " in v and len(v.split()) == 2]

    for i, tok in enumerate(norm_tokens):
        if tok in single_variants:
            return (i, i + 1, tok)
        if bigram_variants and i + 1 < len(norm_tokens):
            pair = (tok, norm_tokens[i + 1])
            if pair in bigram_variants:
                return (i, i + 2, f"{pair[0]} {pair[1]}")

    if max_edit_distance > 0:
        for i, tok in enumerate(norm_tokens):
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


def extract_inline_command(
    text: str,
    *,
    variants: tuple[str, ...] = _DEFAULT_VARIANTS,
    max_edit_distance: int = 1,
    max_command_words: int = 14,
    min_words_before_stop: int = 2,
) -> InlineCommandMatch | None:
    """Extract a bounded ``Nexa <action>`` clause from a conversation transcript.

    Args:
        text: Raw transcript (possibly a multi-sentence conversation chunk).
        variants: Known wake-word variants (single tokens or 2-word bigrams).
        max_edit_distance: Maximum Levenshtein distance for single-word fuzzy match.
        max_command_words: Hard cap of words after the wake mention.
        min_words_before_stop: Minimum words after the wake before sentence-stop
            punctuation or discourse markers cut the clause (avoids cutting
            ``"Nexa. Prendé la luz"`` right after the wake).

    Returns:
        ``InlineCommandMatch`` if a wake mention was found, else ``None``.
    """
    if not text or not text.strip():
        return None
    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return None
    norm_tokens = [_norm_token(m.group(0)) for m in matches]

    span = _find_wake_token_span(norm_tokens, variants, max_edit_distance)
    if span is None:
        return None
    wake_start_idx, wake_end_idx, matched_variant = span

    char_start = matches[wake_start_idx].start()
    max_end_idx = min(len(matches), wake_end_idx + max(1, max_command_words))
    end_token_idx = max_end_idx

    for idx in range(wake_end_idx, max_end_idx):
        words_after_wake = idx - wake_end_idx
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
    # Keep a single trailing ? or ! if it directly terminates the command token.
    if char_end < len(text) and text[char_end] in "?!":
        char_end += 1

    command_text = text[char_start:char_end].strip()
    prefix_text = text[:char_start].strip(" ,;:-—\t\n")
    suffix_text = text[char_end:].strip(" ,;:-—\t\n")
    embedded = bool(prefix_text) or bool(suffix_text)
    return InlineCommandMatch(
        command_text=command_text,
        matched_variant=matched_variant,
        prefix_text=prefix_text,
        suffix_text=suffix_text,
        embedded_in_conversation=embedded,
    )
