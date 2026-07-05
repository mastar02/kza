"""Command Acceptance Gate — decide whether a post-wake capture is a real command.

Consolidates noise/echo heuristics (previously in request_router._is_noise_text)
and incorporates STT confidence. Reject = silent discard upstream.
"""
from __future__ import annotations

import difflib
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
    "canal de youtube",
    # "activa la" REMOVIDO 2026-06-02: substring over-broad — false-rejectaba
    # comandos válidos "activá la rutina/alarma/escena" (0 hits en 78k líneas de
    # logs, así que no protegía de nada real). Ver command_detection_rootcause.
    "luz encendida", "luz apagada", "luces encendidas", "luces apagadas",
    "hecho", "perfecto", "listo",
    # BoH-es (spec 2026-07-05): alucinación de Whisper sobre audio
    # ininteligible; 25 ocurrencias en 48h de prod, 0 comandos reales.
    "aplausos",
)
_FILLER_WORDS = {"gracias", "si", "no", "ok", "bueno", "dale"}

# prompt_echo: fracción mínima de la transcripción que debe aparecer como
# bloque contiguo dentro de una oración del initial_prompt para considerarla
# eco (Whisper regurgitando el prompt). Ratio sobre len(transcripción), NO
# ratio simétrico de SequenceMatcher: el eco es un FRAGMENTO corto de una
# oración larga y el ratio simétrico quedaría ~0.3.
# 0.9 (no 0.8): ecos reales miden 0.963-1.0; la enumeración multi-room sin verbo llega a 0.879 (false positive con 0.8; margen 0.021). Con verbo baja a ~0.75. El único texto >=0.9 restante (lista de cuartos verbatim sin verbo, ~0.95) jamás ejecutaría nada igual (router lo descarta).
_PROMPT_ECHO_RATIO = 0.9
_PROMPT_ECHO_MIN_WORDS = 4


def _normalize(text: str) -> str:
    """Normalise to lowercase without accents or punctuation, collapsed spaces."""
    norm = unicodedata.normalize("NFD", text.lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    norm = re.sub(r"[^\w\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def _prompt_sentences(prompt: str | None) -> tuple[str, ...]:
    """Oraciones normalizadas del initial_prompt con ≥4 palabras.

    Se usan como referencia para detectar eco del prompt en la transcripción
    (Whisper a veces devuelve texto del propio initial_prompt).
    """
    if not prompt:
        return ()
    sentences = []
    for raw in prompt.split("."):
        norm = _normalize(raw)
        if len(norm.split()) >= _PROMPT_ECHO_MIN_WORDS:
            sentences.append(norm)
    return tuple(sentences)


@dataclass(frozen=True)
class AcceptanceDecision:
    """Gate result: accept/reject flag, reason code, and telemetry signals."""
    accept: bool
    reason: str
    signals: dict = field(default_factory=dict)


class CommandAcceptanceGate:
    """Command acceptance gate. Hard rules (enforce) + confidence (shadow)."""

    def __init__(
        self,
        wake_words: Iterable[str] = (),
        enforce_confidence: bool = False,
        max_no_speech_prob: float = 0.60,
        min_avg_logprob: float = -1.20,
        enforce_compression_ratio: bool = False,
        max_compression_ratio: float = 2.2,
        initial_prompt: str | None = None,
    ):
        """Configure the gate.

        Args:
            wake_words: Normalised wake words; if non-empty, transcriptions
                not containing any of them are rejected as noise.
            enforce_confidence: When True, low STT-confidence captures are
                rejected; when False, they are accepted but flagged in signals
                (shadow mode).
            max_no_speech_prob: Whisper no_speech_prob ceiling (0-1).
            min_avg_logprob: Whisper avg_logprob floor (negative).
            enforce_compression_ratio: Separate enforce flag for the
                compression-ratio rule. Unlike no_speech_prob/avg_logprob
                (dead signals on turbo — inverted distribution, 14k events),
                this one DOES separate hallucinations (openai/whisper #2378),
                so it gets its own flag instead of riding enforce_confidence.
            max_compression_ratio: Per-segment max compression_ratio ceiling.
                Repetitive hallucinated text compresses high; short real
                commands stay well below 2.2 (zlib overhead).
            initial_prompt: Texto del stt.initial_prompt configurado. Si se
                provee, transcripciones que son eco del prompt (Whisper lo
                regurgita sobre audio ininteligible) se rechazan con reason
                'prompt_echo'. None = regla inactiva.
        """
        self._wake_words = tuple(
            w.lower().strip() for w in wake_words if w and w.strip()
        )
        self._enforce_confidence = enforce_confidence
        self._max_no_speech_prob = max_no_speech_prob
        self._min_avg_logprob = min_avg_logprob
        self._enforce_compression_ratio = enforce_compression_ratio
        self._max_compression_ratio = max_compression_ratio
        self._prompt_sentences = _prompt_sentences(initial_prompt)

    def evaluate(self, text: str, stt_confidence: "STTResult | None" = None) -> AcceptanceDecision:
        """Evaluate a capture. Fail-open: on internal error, accept."""
        try:
            return self._evaluate(text, stt_confidence)
        except Exception as e:  # fail-open: never drop voice control on an internal bug
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
        # autojunk=False: el default purga chars frecuentes en oraciones >=200 chars
        # y pierde ecos reales (size 0). Con autojunk=False capturamos ecos verbatim.
        if self._prompt_sentences and len(words) >= _PROMPT_ECHO_MIN_WORDS:
            for sentence in self._prompt_sentences:
                m = difflib.SequenceMatcher(
                    None, norm, sentence, autojunk=False
                ).find_longest_match(0, len(norm), 0, len(sentence))
                # autojunk=False: el default purga chars frecuentes en oraciones >=200 chars
                # y pierde ecos reales (size 0). Con autojunk=False capturamos ecos verbatim.
                if m.size / len(norm) >= _PROMPT_ECHO_RATIO:
                    return "prompt_echo"
        if self._wake_words and not any(w in norm for w in self._wake_words):
            return f"missing_wake:{self._wake_words[0]!r}"
        return None

    def _confidence_reason(self, stt_confidence: "STTResult | None") -> tuple[str | None, dict]:
        """Evaluate confidence rules. Returns (reason|None, signals)."""
        if stt_confidence is None:
            return None, {"no_speech_prob": None, "avg_logprob": None}
        nsp = stt_confidence.no_speech_prob
        alp = stt_confidence.avg_logprob
        signals = {"no_speech_prob": nsp, "avg_logprob": alp}
        bad = []
        if nsp is not None and nsp > self._max_no_speech_prob:
            bad.append(f"no_speech>{self._max_no_speech_prob}")
        if alp is not None and alp < self._min_avg_logprob:
            bad.append(f"avg_logprob<{self._min_avg_logprob}")
        reason = f"low_confidence:{','.join(bad)}" if bad else None
        return reason, signals

    def _compression_reason(self, stt_confidence: "STTResult | None") -> tuple[str | None, float | None]:
        """Evaluate the compression-ratio rule. Returns (reason|None, ratio|None).

        getattr defensivo: motores que no exponen compression_ratio (Moonshine,
        mocks viejos) cuentan como None = sin penalizar.
        """
        cr = getattr(stt_confidence, "compression_ratio", None)
        if cr is not None and cr > self._max_compression_ratio:
            return f"high_compression:{cr:.2f}>{self._max_compression_ratio}", cr
        return None, cr

    def _evaluate(self, text: str, stt_confidence: "STTResult | None") -> AcceptanceDecision:
        hard = self._hard_reason(text)
        conf_reason, signals = self._confidence_reason(stt_confidence)
        comp_reason, comp_ratio = self._compression_reason(stt_confidence)
        signals["compression_ratio"] = comp_ratio

        if hard is not None:
            decision = AcceptanceDecision(False, hard, signals)
        elif comp_reason is not None and self._enforce_compression_ratio:
            decision = AcceptanceDecision(False, comp_reason, signals)
        elif conf_reason is not None and self._enforce_confidence:
            decision = AcceptanceDecision(False, conf_reason, signals)
        else:
            # shadow: accept but record what we WOULD reject
            would = ",".join(r for r in (comp_reason, conf_reason) if r)
            if would:
                decision = AcceptanceDecision(
                    True, "ok", {**signals, "would_reject": would}
                )
            else:
                decision = AcceptanceDecision(True, "ok", signals)

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"[CommandGate] accept={decision.accept} reason={decision.reason} "
                f"no_speech={signals.get('no_speech_prob')} "
                f"avg_logprob={signals.get('avg_logprob')} "
                f"compression={signals.get('compression_ratio')} "
                f"would_reject={decision.signals.get('would_reject')} "
                f"text={text[:60]!r}"
            )
        return decision
