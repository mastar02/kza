"""Command Acceptance Gate — decide whether a post-wake capture is a real command.

Consolidates noise/echo heuristics (previously in request_router._is_noise_text)
and incorporates STT confidence. Reject = silent discard upstream.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.stt.whisper_fast import STTResult

logger = logging.getLogger(__name__)


# Frases que NO son comandos (TV/YouTube + eco típico del TTS). Match por
# substring sobre texto normalizado (lowercase, sin acentos).
_NOISE_PHRASES = (
    "suscribe", "suscrib", "campanita", "gracias por ver",
    "dale like", "dale lie", "dale mega like",
    "canal de youtube", "activa la",
    "luz encendida", "luz apagada", "luces encendidas", "luces apagadas",
    "hecho", "perfecto", "listo",
)
_FILLER_WORDS = {"gracias", "si", "no", "ok", "bueno", "dale"}


def _normalize(text: str) -> str:
    """Normalise to lowercase without accents or punctuation, collapsed spaces."""
    norm = unicodedata.normalize("NFD", text.lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    norm = re.sub(r"[^\w\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


@dataclass(frozen=True)
class AcceptanceDecision:
    """Gate result: accept/reject flag, reason code, and telemetry signals."""
    accept: bool
    reason: str
    signals: dict = field(default_factory=dict)


class CommandAcceptanceGate:
    """Gate de aceptación de comandos. Hard rules (enforce) + confidence (shadow)."""

    def __init__(
        self,
        wake_words: Iterable[str] = (),
        enforce_confidence: bool = False,
        max_no_speech_prob: float = 0.60,
        min_avg_logprob: float = -1.20,
    ):
        """Configure the gate.

        Args:
            wake_words: Normalised wake words; if non-empty, transcriptions
                not containing any of them are rejected as noise.
            enforce_confidence: When True, STT confidence thresholds reject
                low-quality captures (added in a later task). Shadow-only today.
            max_no_speech_prob: Whisper no_speech_prob ceiling (0-1).
            min_avg_logprob: Whisper avg_logprob floor (negative).
        """
        self._wake_words = tuple(
            w.lower().strip() for w in wake_words if w and w.strip()
        )
        self._enforce_confidence = enforce_confidence
        self._max_no_speech_prob = max_no_speech_prob
        self._min_avg_logprob = min_avg_logprob

    def evaluate(self, text: str, stt_confidence: "STTResult | None" = None) -> AcceptanceDecision:
        """Evaluate a capture. Fail-open: on internal error, accept."""
        try:
            return self._evaluate(text, stt_confidence)
        except Exception as e:  # fail-open: nunca tumbar el control de voz
            logger.error(f"CommandAcceptanceGate error (fail-open accept): {e}")
            return AcceptanceDecision(True, "gate_error", {"error": str(e)})

    def _hard_reason(self, text: str) -> str | None:
        """Return the first matching hard rule name, or None. Migrated from _is_noise_text."""
        if not text:
            return "empty"
        norm = _normalize(text)
        if not norm:
            return "empty_after_norm"
        for phrase in _NOISE_PHRASES:
            if phrase in norm:
                return f"noise_phrase:{phrase!r}"
        if norm in _FILLER_WORDS:
            return f"filler_word:{norm!r}"
        words = norm.split()
        if len(words) >= 4 and len(set(words)) == 1:
            return f"word_repetition:{words[0]!r}"
        if self._wake_words and not any(w in norm for w in self._wake_words):
            return f"missing_wake:{self._wake_words[0]!r}"
        return None

    def _evaluate(self, text: str, stt_confidence) -> AcceptanceDecision:
        # stt_confidence is intentionally unused until the confidence-rules task.
        hard = self._hard_reason(text)
        if hard is not None:
            return AcceptanceDecision(False, hard, {})
        return AcceptanceDecision(True, "ok", {})
