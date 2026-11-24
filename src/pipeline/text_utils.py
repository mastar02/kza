"""Lightweight text normalization & truncation heuristics for voice capture.

Extracted from legacy ``src/wakeword/whisper_wake.py`` so active capture
modules (``MultiRoomAudioLoop``, ``WakePipelineMixin``) have zero runtime
dependency on inactive Whisper-based wake engines.
"""
from __future__ import annotations

import re
import unicodedata

_TRUNCATING_TAILS = frozenset({
    # preposiciones
    "de", "del", "a", "al", "en", "con", "por", "para", "sin", "sobre",
    "tras", "hacia", "desde",
    # artículos solos al final
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # demostrativos (raramente cierran oración)
    "este", "esta", "esto", "ese", "esa", "eso", "aquel", "aquella",
    # conjunciones
    "y", "o", "u", "pero", "ni", "que",
    # adverbios/intensificadores que esperan un complemento
    "como", "muy", "tan", "mas", "mucho", "poco",
})


def _normalize_spanish_text(text: str) -> str:
    norm = unicodedata.normalize("NFD", text.lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    norm = re.sub(r"[^\w\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def _text_likely_truncated(text: str) -> bool:
    """Heuristic: return True when a wake transcript looks cut off mid-clause."""
    if not text:
        return False
    raw = text.strip()
    if raw.endswith(("...", "…")):
        return True
    stripped = raw.rstrip(",.;:!?¡¿\"' ")
    if not stripped:
        return False
    parts = _normalize_spanish_text(stripped).split()
    if not parts:
        return False
    return parts[-1] in _TRUNCATING_TAILS
