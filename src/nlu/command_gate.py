"""Command Acceptance Gate — decide whether a post-wake capture is a real command.

Consolidates noise/echo heuristics (previously in request_router._is_noise_text)
and incorporates STT confidence. Reject = silent discard upstream.
"""
from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.nlu.household_intents import HouseholdIntentClassifier
    from src.routines.voice_routines import VoiceRoutineService
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
    # Alucinación recurrente del turbo sobre garble (familia "Esto es un
    # asistente de voz…"). 2026-07-23 15:06 una variante CON verbo ("Esto es
    # un asistente, apagá, subí.") pasó el gate y EJECUTÓ light.turn_off —
    # acción fantasma real. Ningún comando legítimo contiene esta frase.
    "esto es un asistente",
)
_FILLER_WORDS = {"gracias", "si", "no", "ok", "bueno", "dale"}
_STATE_ECHO_PHRASES = {"luz encendida", "luz apagada", "luces encendidas", "luces apagadas"}

# prompt_echo: fracción mínima de la transcripción que debe aparecer como
# bloque contiguo dentro de una oración del initial_prompt para considerarla
# eco (Whisper regurgitando el prompt). Ratio sobre len(transcripción), NO
# ratio simétrico de SequenceMatcher: el eco es un FRAGMENTO corto de una
# oración larga y el ratio simétrico quedaría ~0.3.
# 0.9 (no 0.8): ecos reales miden 0.963-1.0; enumeración multi-room sin
# verbo llega a 0.879 (false positive con 0.8; margen 0.021). Con verbo
# baja a ~0.75. El único texto >=0.9 restante (lista cuartos verbatim sin
# verbo, ~0.95) jamás ejecutaría nada igual (router lo descarta).
_PROMPT_ECHO_RATIO = 0.9
_PROMPT_ECHO_MIN_WORDS = 4


def _normalize(text: str) -> str:
    """Normalise to lowercase without accents or punctuation, collapsed spaces."""
    norm = unicodedata.normalize("NFD", text.lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    norm = re.sub(r"[^\w\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def normalize_wake_words(words: Iterable[str]) -> tuple[str, ...]:
    """Lowercase, stripped, non-empty wake words (the gate's own form)."""
    return tuple(w.lower().strip() for w in words if w and w.strip())


def contains_wake_word(text: str, wake_words: Iterable[str]) -> bool:
    """The missing_wake rule: any wake word inside the normalised transcript.

    Shared by the gate (whisper engine) and the shadow wake verification of
    acoustic wakes, so both apply the exact same rule and name variants.

    Args:
        text: Raw transcript.
        wake_words: Wake words in the gate's normalised form.

    Returns:
        True when one of the words occurs in the normalised text.
    """
    norm = _normalize(text or "")
    return any(w in norm for w in wake_words)


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
        household_intents: HouseholdIntentClassifier | None = None,
        voice_routines: VoiceRoutineService | None = None,
        ring_stop_open: Callable[[], bool] | None = None,
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
            household_intents: Household intent classifier shared with the
                router (P3-F2). Claimed reminders, routine creation and exact
                DND requests can reach their household handler through the
                non-immediate rule. The other hard rules still apply. None
                builds one with the default settings.
            voice_routines: Read-only voice routines (T033-U1-RV1), the
                instance the router and the dispatcher answer from. A routine
                question it detects ("¿está activa la rutina riego?") is a
                read, not an immediate action, so the non-immediate rule does
                not drop it; the dispatcher and the legacy router answer it
                from the store before any domotics producer, and every
                producer keeps its own non-immediate check. None: no exemption.
            ring_stop_open: Cheap synchronous check, no I/O (T030-U2b): True
                while a timer rings or its ring is in the stop follow-up
                (``TimerVoiceCommands.ring_stop_open``). Only then a bare
                "listo" (with the shared fillers, BFR-1) is not noise; the
                accepted decision carries the signal ``ring_stop``. On every
                other turn "listo" stays a noise phrase. None: no exemption.
        """
        self._wake_words = normalize_wake_words(wake_words)
        self._enforce_confidence = enforce_confidence
        self._max_no_speech_prob = max_no_speech_prob
        self._min_avg_logprob = min_avg_logprob
        self._enforce_compression_ratio = enforce_compression_ratio
        self._max_compression_ratio = max_compression_ratio
        self._prompt_sentences = _prompt_sentences(initial_prompt)
        if household_intents is None:
            from src.nlu.household_intents import HouseholdIntentClassifier
            household_intents = HouseholdIntentClassifier()
        self._household_intents = household_intents
        self._voice_routines = voice_routines
        self._ring_stop_open = ring_stop_open

    def evaluate(self, text: str, stt_confidence: "STTResult | None" = None) -> AcceptanceDecision:
        """Evaluate a capture. Fail-open: on internal error, accept."""
        try:
            return self._evaluate(text, stt_confidence)
        except Exception as e:  # fail-open: never drop voice control on an internal bug
            logger.error(f"CommandAcceptanceGate error (fail-open accept): {e}")
            return AcceptanceDecision(True, "gate_error", {"error": str(e)})

    def _claimed_non_immediate(self, text: str) -> bool:
        """Allow claimed requests whose action words are content or a mode name.

        In "recordame apagar la luz a las 11" and "armame una rutina que
        apague las luces a las 11" the action words are the reminder's or the
        routine's content: the household layer (router S5) claims the turn and
        no domotics producer sees it, so the non-immediate rule must not drop
        it before an honest reply (P3-F2, user answer P3F2-R1-Q4). Other claimed
        families keep the rule. An exact "no molestar" voice request (T030-U4)
        reaches the household refusal so the user can be directed to the web.
        """
        from src.nlu.household_intents import HouseholdFamily

        claimed = self._household_intents.classify(text)
        if claimed is None:
            return False
        return claimed.family in {HouseholdFamily.REMINDER, HouseholdFamily.NOTIFICATION} or (
            claimed.family is HouseholdFamily.ROUTINE and claimed.reason == "new_routine")

    def _routine_read(self, text: str) -> bool:
        """Whether the turn is a routine question the read always answers (T033-U1-RV1/RV2)."""
        return self._voice_routines is not None and self._voice_routines.claims(text)

    def _ring_stop(self, text: str) -> bool:
        """A bare "listo" while a ring is open (T030-U2b); a failing check keeps it noise."""
        if self._ring_stop_open is None or not text:
            return False
        from src.nlu.household_intents import is_bare_ring_stop

        if not is_bare_ring_stop(text, self._household_intents.settings.wake_words):
            return False
        try:
            return self._ring_stop_open() is True
        except Exception as error:
            logger.warning("[CommandGate] ring check failed, 'listo' stays noise: %s", type(error).__name__)
            return False

    def _without_intercom_message(self, text: str) -> str:
        """Mask literal intercom bodies for phrase filtering, including in compounds."""
        claimed = self._household_intents.classify(text)
        if claimed is None:
            return text
        intents = (part.intent for part in claimed.parts) if claimed.parts else (claimed,)
        spans = [intent.literal_span for intent in intents if intent is not None and intent.literal_span]
        for start, end in sorted(spans, reverse=True):
            text = text[:start] + " [mensaje] " + text[end:]
        return text

    def _hard_reason(self, text: str, ring_stop: bool = False) -> str | None:
        """Return the first matching hard rule name, or None. Migrated from _is_noise_text.

        ``ring_stop`` lifts only the "listo" noise phrase (T030-U2b); every other rule still applies.
        """
        if not text:
            return "empty"
        from src.nlu.command_grammar import non_immediate_action_reason

        action_block = non_immediate_action_reason(text)
        if action_block and not self._claimed_non_immediate(text) and not self._routine_read(text):
            return action_block
        norm = _normalize(text)
        if not norm:
            return "empty_after_norm"
        noise_text = norm
        if any(phrase in norm for phrase in _NOISE_PHRASES):
            noise_text = _normalize(self._without_intercom_message(text))
        for phrase in _NOISE_PHRASES:
            if phrase in noise_text:
                if ring_stop and phrase == "listo":
                    continue
                if phrase in _STATE_ECHO_PHRASES:
                    from src.nlu.command_grammar import parse_command

                    command = parse_command(text)
                    # A state echo alone still fails. A separate imperative
                    # such as "apagá la luz encendida" retains its direction;
                    # the participle itself cannot supply a binary verb.
                    if command.ready_to_dispatch() and command.intent in {"turn_on", "turn_off"}:
                        continue
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
                if m.size / len(norm) >= _PROMPT_ECHO_RATIO:
                    return "prompt_echo"
        if self._wake_words and not contains_wake_word(text, self._wake_words):
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
        ring_stop = self._ring_stop(text)
        hard = self._hard_reason(text, ring_stop)
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
            if ring_stop:
                decision = AcceptanceDecision(True, decision.reason, {**decision.signals, "ring_stop": True})

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
