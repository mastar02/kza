"""
Request Router Module
Routes processed voice commands through the appropriate execution path.

Owns all command routing logic:
- Orchestrated (multi-user) vs legacy (single-user) paths
- Feedback detection, suggestion handling, shared household control policy
- Cache, prompt building, latency logging, event logging
"""

import asyncio
import inspect
from contextvars import ContextVar
import logging
import re
import threading
import time
import unicodedata
from dataclasses import asdict

import numpy as np

from src.home_assistant.home_actions import HomeActionRequest, HomeActionResult, HomeActionService
from src.home_assistant.operations import compile_effect, finite_number
from src.home_assistant.voice_actions import VoiceHomeActionExecutor
from src.home_assistant.voice_resolver import HomeVoiceResolver
from src.nlu.home_command import parse_home_command
from src.nlu.command_gate import AcceptanceDecision, CommandAcceptanceGate, contains_wake_word
from src.nlu.command_grammar import (
    WAKE_TERMS, PartialCommand, has_unresolved_light_location, non_immediate_action_reason, parse_command,
    parse_default_light_target, parse_partial_command, security_order,
)
from src.nlu.household_intents import (
    RING_STOP_INTENT, HouseholdFamily, HouseholdIntent, HouseholdIntentClassifier, HouseholdIntentSettings,
    HouseholdOperation,
)
from src.nlu.sensitive_actions import is_sensitive
from src.orchestrator import PathType
from src.orchestrator.compound import CompoundPartContext, CompoundRunner
from src.orchestrator.dispatcher import (
    DispatchResult, compound_dispatch_result, compound_part_target, has_domotics_action_verb, notification_dnd_refusal,
)
from src.orchestrator.priority_queue import Priority
from src.pipeline.earcon_gate import should_play_earcon
from src.pipeline.post_wake_endpoint import CaptureTrigger
from src.pipeline.wake_verification import WakeVerifier
from src.routines.voice_routines import VoiceRoutineService, legacy_routine_outcome
from src.analytics.asr_quality import alog_asr_outcome
from src.memory.transient import TransientCache
from src.memory.voice_phrase import VERIFICATION_REPLY
from src.memory.prompt_context import SourcesInvalidated, run_private
from src.llm.generation_stream import GuardedGeneration
from src.core.household_clock import HouseholdClock
from src.core.owned_work import OwnedBlockingWork
from src.core.sqlite_work import AuthorityGuard
from src.core.task_outcome import (
    CURRENT_REQUEST_ID, DeliveryOutcome, DispatchState, EffectState, PersistenceState, ResponseRequirement,
    TaskOutcome, TaskStatus, TurnOutcome, project_turn_outcome,
)
from src.core.voice_delivery import GenerationState, ResponseResult
from src.core.observation import Observation
from src.nlu.llm_router import LLMCommandRouter
from src.world.household_time import WEEKDAYS, parse_clock_query

logger = logging.getLogger(__name__)

# (wake_score, wake_vad) of the acoustic-wake CommandEvent being processed by
# this task; None for any other input (V2 shadow wake verification).
_WAKE_CONTEXT: ContextVar = ContextVar("kza_wake_context", default=None)
# Endpoint code of the CommandEvent being processed (post-wake rule); None
# for other inputs. A capture dispatched at the cap must name the wake word.
_ENDPOINT_CODE: ContextVar = ContextVar("kza_endpoint_code", default=None)
_MAX_UTTERANCE_CODE = "close_max_utterance"
# Which STT produced the transcript of the command being processed
# (ProcessedCommand.stt_source); carried into the result and ASR outcomes.
_STT_SOURCE: ContextVar = ContextVar("kza_stt_source", default=None)


# Mapeo intent → dominio HA requerido. Usado por el guard previo al NLU
# pre-filter (Fix 1B): si el LLMRouter clasificó un intent que requiere un
# dominio específico (climate, media_player, cover) y HA no tiene ninguna
# entidad de ese dominio, rechazar con mensaje amistoso en vez de hacer
# fallback a una luz random vía vector similarity.
_INTENT_REQUIRED_DOMAINS: dict[str, str] = {
    "set_temperature": "climate",
    "set_volume": "media_player",
    "media_play": "media_player",
    "media_pause": "media_player",
    "media_stop": "media_player",
    "media_next": "media_player",
    "media_previous": "media_player",
}

# Mensajes de respuesta cuando el dominio no existe en HA. En español
# rioplatense neutro, brevemente — no queremos sermonear al usuario.
_UNSUPPORTED_INTENT_MESSAGES: dict[str, str] = {
    "set_temperature": "Todavía no tengo climatización configurada.",
    "set_volume": "Todavía no tengo control de audio configurado.",
    "media_play": "No tengo reproductor configurado.",
    "media_pause": "No tengo reproductor configurado.",
    "media_stop": "No tengo reproductor configurado.",
    "media_next": "No tengo reproductor configurado.",
    "media_previous": "No tengo reproductor configurado.",
}

# OWW a 0.40–0.45 solapa con TV. Nexa real en prod mide 0.54–0.93. El bonus
# de wake sobre 'prende la luz' (conf 0.70) no aplica debajo de este piso.
_GRAMMAR_BONUS_MIN_WAKE_SCORE = 0.50


_OWN_WAKE_TRIGGERS = frozenset({CaptureTrigger.WAKE, CaptureTrigger.INLINE})


def _capture_without_own_wake(capture_trigger: str | None) -> bool:
    """Whether the producer says this capture was not opened by its own wake (T037-U5).

    MultiRoomAudioLoop tags every capture. ``wake`` was opened by an acoustic
    wake, and ``inline`` is that wake's own utterance ('nexa apagá la luz'
    said in one go, when the wake detector hands over the command audio), so
    its wake score belongs to this turn (T037-U5-R2). A follow-up or barge-in
    capture (or an ``unknown`` one, fail closed) carries the score of the
    last real wake, so that score says nothing about this turn. None means
    the producer does not say (raw audio or the textual wake channel):
    those keep their own signals as before.

    Args:
        capture_trigger: ``CommandEvent.capture_trigger``, or None.

    Returns:
        True when the capture has no wake of its own.
    """
    return capture_trigger is not None and capture_trigger not in _OWN_WAKE_TRIGGERS


_LIGHT_WORDS = frozenset({
    "light", "luz", "luces", "lampara", "lamparas",
    # The dispatcher's other light nouns (T037-U5-FV2).
    "foco", "focos", "luminaria", "luminarias", "veladora", "veladoras",
})


def _dispatch_targets_light(text: str, classification) -> bool:
    """Whether a turn about to reach the dispatcher acts on a light (T037-U5-R1).

    The unresolved-light-location guard holds whichever producer accepted the
    turn: the grammar fast path (entity_hint ``light``), the 7B (its
    entity_hint is a domain such as ``light`` or free text such as ``luz del
    quincho``), or neither (no LLMCommandRouter configured, the textual wake
    channel). The text's own grammar parse also counts when it names a light
    and either finds an action on it, or the accepted classification carries
    an intent, or the text has a verb the dispatcher routes to FAST_DOMOTICS
    ('bajá la luz del quincho', 'cambiá la luz del quincho': the grammar has
    no intent for those verbs, T037-U5-FV1). A question that only mentions a
    light, with no action verb, does not.

    Args:
        text: Accepted transcript of the turn.
        classification: The accepted grammar or 7B classification, or None.

    Returns:
        True when the turn acts on a light.
    """
    hint = getattr(classification, "entity_hint", None)
    accepted_intent = getattr(classification, "intent", None) is not None
    if accepted_intent and isinstance(hint, str):
        folded = unicodedata.normalize("NFKD", hint.lower()).encode("ascii", "ignore").decode()
        if _LIGHT_WORDS.intersection(re.findall(r"[a-z]+", folded)):
            return True
    parsed = parse_partial_command(text)
    if parsed.entity != "light":
        return False
    return parsed.intent is not None or accepted_intent or has_domotics_action_verb(text)


def _grammar_fastpath_classification(
    text: str,
    confidence_threshold: float = 0.75,
    wake_confirmed: bool = False,
):
    """CommandClassification si la gramática determinística produce un comando
    dispatchable de alta confianza (domótica o media); None si no.

    Para target='music' igual se devuelve is_command=True: el comando es válido
    y debe pasar el gate; el dispatcher lo rutea a su music path por
    _classify_request. El intent/target quedan en la clasificación para audit.

    Args:
        text: Transcribed text from the STT pipeline (post-wake-word).
        confidence_threshold: Minimum grammar confidence to trust the parse.
            Defaults to 0.75 (same as ``RequestRouter.confidence_threshold``).

    Returns:
        A ``CommandClassification`` with ``is_command=True`` when the grammar
        produces a full-quality parse above the confidence threshold, or
        ``None`` when it does not — the caller then falls through to the
        LLMCommandRouter as before.
    """
    from src.nlu.command_grammar import parse_command
    from src.nlu.llm_router import CommandClassification

    pc = parse_command(text)
    confidence = pc.confidence
    # Con wake confirmado ACÚSTICAMENTE (openwakeword/porcupine), el STT a
    # veces no transcribe "Nexa" y el parse pierde el bonus de wake (+0.15):
    # 'Prende la luz.' quedaba en 0.7 < 0.75 y caía al LLMRouter, que lo
    # rechazaba como noise (caso real 2026-06-04 18:24). El bonus corresponde
    # igual — la confirmación llegó por otro canal, no por el texto.
    # SOLO para textos cortos (2026-06-05): el wake espurio con TV de fondo
    # invalida la premisa "wake ⇒ usuario" — charla larga que embebe un
    # comando ('…le dije que apague la luz cuando se vaya') no debe ganar
    # el bonus (amplificó la acción fantasma del 06-04 19:49). El rescate
    # es para comandos cortos sin "Nexa" transcripta.
    _MAX_BONUS_WORDS = 6
    if (
        wake_confirmed
        and pc.quality == "full"
        and not pc.has_wake
        and len(text.split()) <= _MAX_BONUS_WORDS
    ):
        confidence += 0.15
    if pc.quality != "full" or confidence < confidence_threshold:
        return None

    return CommandClassification(
        is_command=True,
        confidence=confidence,
        intent=pc.intent,
        entity_hint=pc.domain,
        slots=dict(pc.slots),
        raw_response=f"<grammar:{pc.target}>",
        elapsed_ms=0.0,
    )


class RequestRouter:
    """
    Routes voice commands through the correct execution path.

    Handles orchestrated (multi-user) and legacy (single-user) processing,
    feedback detection, suggestion handling, household control policy, caching,
    prompt building, and latency/event logging.

    One public method: process_command(audio) -> dict
    """

    # V2 shadow wake verification; class default keeps partially built
    # routers (tests that bypass __init__) on the previous behaviour.
    _wake_verifier: WakeVerifier | None = None

    # Frases de feedback
    FEEDBACK_GOOD = [
        "buena respuesta", "eso estuvo bien", "correcto", "perfecto",
        "excelente", "muy bien", "eso es", "exacto", "gracias",
        "good response", "that's right", "perfect"
    ]
    FEEDBACK_BAD = [
        "mala respuesta", "eso estuvo mal", "incorrecto", "no era eso",
        "mal", "no", "equivocado", "bad response", "wrong", "that's wrong"
    ]
    FEEDBACK_CORRECTION_PREFIX = [
        "debiste decir", "deberias haber dicho", "la respuesta correcta es",
        "mejor di", "di mejor", "you should have said", "the correct answer is"
    ]

    # Frases para sugerencias de automatizacion
    SUGGESTION_ACCEPT = [
        "si", "acepto", "si quiero", "hazlo", "adelante", "ok", "esta bien",
        "yes", "accept", "do it", "go ahead"
    ]
    SUGGESTION_REJECT = [
        "no", "no quiero", "rechazar", "cancelar", "no gracias",
        "reject", "cancel", "no thanks"
    ]
    SUGGESTION_SNOOZE = [
        "despues", "mas tarde", "luego", "recordar despues",
        "later", "remind later", "snooze"
    ]

    def __init__(
        self,
        command_processor,
        response_handler,
        audio_manager,
        room_context_manager=None,
        orchestrator=None,
        orchestrator_enabled: bool = True,
        chroma_sync=None,
        ha_client=None,
        llm_reasoner=None,
        fast_router=None,
        memory_manager=None,
        user_manager=None,
        enrollment=None,
        conversation_collector=None,
        command_learner=None,
        event_logger=None,
        suggestion_engine=None,
        latency_monitor=None,
        features=None,
        voice_routine_handler=None,
        routine_manager=None,
        vector_search_threshold: float = 0.65,
        latency_target_ms: int = 300,
        suggestion_interval: int = 50,
        cache_max_size: int = 100,
        action_tracker=None,
        confidence_threshold: float = 0.75,
        metrics_emitter=None,
        wake_words: tuple[str, ...] | list[str] | None = None,
        llm_command_router=None,
        wake_acoustically_confirmed: bool = False,
        hooks=None,  # plan #3 OpenClaw — HookRegistry instance or None
        command_gate: CommandAcceptanceGate | None = None,
        llm_min_command_confidence: float = 0.6,
        earcon_cfg: dict | None = None,
        asr_event_logger=None,
        home_actions: HomeActionService | None = None,
        home_voice_resolver: HomeVoiceResolver | None = None,
        before_handler_warn_ms: float = 5.0,
        unavailable_precheck_enabled: bool = True,
        privacy=None,
        voice_attributor=None,
        wake_verifier: WakeVerifier | None = None,
        household_clock: HouseholdClock | None = None,
        household_intents: HouseholdIntentClassifier | None = None,
        voice_routines: VoiceRoutineService | None = None,
    ):
        """
        Initialize RequestRouter with injected dependencies.

        Args:
            command_processor: CommandProcessor for STT + speaker ID + emotion.
            response_handler: ResponseHandler for TTS output.
            audio_manager: AudioManager for zone detection.
            room_context_manager: RoomContextManager for room resolution (optional).
            orchestrator: MultiUserOrchestrator (optional).
            orchestrator_enabled: Whether to use orchestrated path.
            chroma_sync: ChromaSync for vector search.
            ha_client: HomeAssistantClient for service calls.
            llm_reasoner: LLMReasoner for deep reasoning.
            fast_router: FastRouter for classify-and-respond.
            memory_manager: MemoryManager for context/preferences.
            user_manager: UserManager for profile and enrollment operations.
            enrollment: VoiceEnrollment for new user registration.
            conversation_collector: For feedback tracking.
            command_learner: For learning new commands.
            event_logger: For domotics event logging.
            suggestion_engine: For automation suggestions.
            latency_monitor: For latency tracking.
            features: FeatureManager for timer/intercom/notification commands.
            voice_routine_handler: VoiceRoutineHandler for routine commands.
            routine_manager: RoutineManager (legacy fallback).
            vector_search_threshold: Minimum similarity for vector search.
            latency_target_ms: Target latency in milliseconds.
            suggestion_interval: Commands between suggestion prompts.
            cache_max_size: Maximum cache entries.
            confidence_threshold: Below this confidence, sensitive commands
                ask for explicit confirmation; reversible ones execute with a
                warning log. Range [0.0, 1.0]. Default 0.75.
            hooks: Optional HookRegistry instance (plan #3 OpenClaw). When set,
                before_ha_action / before_tts_speak hooks fire on each invocation
                and after-events emit at pipeline checkpoints. Backward-compat:
                None disables hook calls; mandatory home control policy remains active.
            wake_verifier: Second-stage transcript wake check for acoustic
                wakes (V2 2026-11-17); None disables it.
            household_clock: Household clock built once in main (P3-F1) for
                the legacy prompt's current time; None builds a default one,
                never the process's local time.
            household_intents: Household intent classifier (P3-F2), the same
                instance the dispatcher uses; None builds one with the default
                settings and this router's wake words.
            voice_routines: Saved routine reads and controls shared with the
                orchestrator's dispatcher. Main binds the scheduler after
                construction. None keeps the legacy handlers.
        """
        # Core pipeline components
        self.command_processor = command_processor
        self.response_handler = response_handler
        self.audio_manager = audio_manager
        self.room_context_manager = room_context_manager
        self.household_clock = household_clock if household_clock is not None else HouseholdClock()

        # Orchestrator
        self._orchestrator = orchestrator
        self.orchestrator_enabled = orchestrator_enabled

        # Service dependencies
        self.chroma = chroma_sync
        self.ha = ha_client
        self.home_actions = home_actions if home_actions is not None else HomeActionService(ha_client)
        self.home_voice_resolver = home_voice_resolver if home_voice_resolver is not None else HomeVoiceResolver(self.home_actions)
        self.llm = llm_reasoner
        self.router = fast_router
        self.memory = memory_manager
        self._privacy = privacy
        self.user_manager = user_manager
        self.enrollment = enrollment
        self.conversation_collector = conversation_collector
        self.command_learner = command_learner
        self.event_logger = event_logger
        self.suggestion_engine = suggestion_engine
        self.latency_monitor = latency_monitor
        self.features = features
        self.voice_routine_handler = voice_routine_handler
        self.routines = routine_manager
        self.voice_routines = voice_routines
        self._routine_capture_allowed: AuthorityGuard | None = None

        # Configuration
        self.vector_search_threshold = vector_search_threshold
        self.latency_target_ms = latency_target_ms
        self.suggestion_interval = suggestion_interval
        self.confidence_threshold = confidence_threshold
        self.metrics_emitter = metrics_emitter

        # Wake words (lowercase, sin acentos) usadas por el noise filter
        # para descartar capturas que no contienen la palabra de despertar.
        # Default coincide con config base (rooms.wake_word.words).
        _default_wakes = ("nexa", "kaza")
        if wake_words:
            self._wake_words = tuple(
                w.lower().strip() for w in wake_words if w and w.strip()
            )
        else:
            self._wake_words = _default_wakes

        # P3-F2: claims household turns before any domotics producer.
        self.household_intents = household_intents if household_intents is not None else HouseholdIntentClassifier(
            HouseholdIntentSettings(wake_words=self._wake_words))
        # T037-U4: the same wake terms the dispatcher strips before recognizing
        # a clock question, so both agree on which turns are clock turns.
        self._clock_wake_words = tuple(sorted(WAKE_TERMS | set(self.household_intents.settings.wake_words)))

        # Command acceptance gate (consolidates _is_noise_text + STT confidence).
        self.command_gate = command_gate or CommandAcceptanceGate(
            wake_words=self._wake_words, household_intents=self.household_intents, voice_routines=voice_routines,
        )

        # LLM-based command validator. Si está configurado, intercepta el
        # texto post-wake antes del orchestrator y rechaza alucinaciones
        # de TV / replays / frases noise. Ver src/nlu/llm_router.py.
        self.llm_command_router = llm_command_router
        # True cuando el wake engine confirma acústicamente (openwakeword/
        # porcupine): el grammar fastpath otorga el bonus de wake aunque el
        # STT no haya transcripto "Nexa" (2026-06-04).
        self.wake_acoustically_confirmed = wake_acoustically_confirmed
        # Confianza mínima auto-reportada por el LLMCommandRouter para aceptar
        # un comando. El 7B marca is_command=True sobre charla ambiente con
        # confianza baja → acción fantasma. Solo aplica al path LLM, NO al
        # grammar fast-path (gramática = alta confianza, ya filtrada).
        # (2026-06-02 command_detection_rootcause.) Calibrable vía config.
        self.llm_min_command_confidence = llm_min_command_confidence

        # Earcon "te oí, no te entendí" (2026-06-15)
        self._earcon_cfg = earcon_cfg or {"enabled": False}
        self._asr_event_logger = asr_event_logger
        # U3b-4b-2: present only when memory.voice_proposals.enabled is true
        # (src/users/voice_attributor.py build_voice_attribution); None keeps
        # every path, reply and timing exactly as without the voice stage.
        self._voice_attributor = voice_attributor

        # Plan #3 OpenClaw — plugin hooks registry (or None)
        self._hooks = hooks
        self._voice_actions = VoiceHomeActionExecutor(
            ha_client, self.home_actions, response_handler=response_handler, hooks=hooks,
            before_handler_warn_ms=before_handler_warn_ms,
            unavailable_precheck_enabled=unavailable_precheck_enabled,
        )
        # T037-U6: the same action boundary for a compound part, without its
        # own failure earcon or hook reason (the compound's one reply reports it).
        self._compound_voice_actions = VoiceHomeActionExecutor(
            ha_client, self.home_actions, response_handler=None, hooks=hooks,
            before_handler_warn_ms=before_handler_warn_ms,
            unavailable_precheck_enabled=unavailable_precheck_enabled,
        )
        # T037-U6: compound turns run here part by part (each part meets this
        # router's own gates); the runner remembers parts by request id.
        self._compound = CompoundRunner()
        # T037-U6: accepted turns per microphone room. A compound stops its
        # remaining parts when another turn (a barge-in) is accepted in its room.
        self._room_turns: dict[str | None, int] = {}

        # State
        from src.memory.memory_manager import MemoryManager
        if privacy is not None and isinstance(self.memory, MemoryManager) and self.memory.long_term.ledger is not privacy.ledger:
            raise ValueError("privacy_authority_mismatch")
        self._source_policy = privacy.ledger if privacy is not None else (self.memory.long_term.ledger if isinstance(self.memory, MemoryManager) else None)
        self._query_cache = TransientCache(cache_max_size, source_policy=self._source_policy)
        self._cache_max_size = cache_max_size
        self._private_maintenance_task = None
        self._query_work = OwnedBlockingWork()
        self._suggestion_lock = threading.RLock()
        self._suggestion_flows: set[asyncio.Task] = set()
        self._suggestions_closed = False
        self._suggestion_closing = None
        self._closed = False
        self._pending_suggestion = None
        self._last_response = None
        self._command_count = 0
        self._response_zone = ContextVar("response_zone", default=None)
        self._memory_source = ContextVar("memory_source", default=None)
        self._response_sources = ContextVar("response_sources", default=())
        self._turn_outcome = ContextVar("turn_outcome", default=None)
        self._wake_verifier = wake_verifier

        # Last-action tracker para comandos ambiguos (Q6: contextual + pregunta fallback)
        if action_tracker is None:
            from src.orchestrator.action_context import LastActionTracker
            action_tracker = LastActionTracker(ttl_seconds=60.0)
        self.action_tracker = action_tracker

    async def _memory_call(self, name: str, *args, **kwargs):
        operation = getattr(self.memory, "a"+name, None)
        if inspect.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        return await asyncio.to_thread(getattr(self.memory, name), *args, **kwargs)

    async def _speak_response(self, text: str, **kwargs) -> ResponseResult:
        """Serialize synthesis/playback in a worker while capture keeps running."""
        kwargs.setdefault("zone_id", self._response_zone.get())
        response = await self.response_handler.aspeak(text, **kwargs)
        self._record_delivery(response)
        return response

    def _record_delivery(self, response) -> None:
        turn = self._turn_outcome.get()
        if turn is not None:
            turn.record_delivery(DeliveryOutcome.from_response(response)
                                 if isinstance(response, ResponseResult) else DeliveryOutcome())

    def _record_home_action(self, action_result) -> None:
        turn = self._turn_outcome.get()
        if turn is not None:
            turn.task = action_result.receipt.task_outcome
            turn.response_requirement = (ResponseRequirement.SILENT if action_result.receipt.status == "confirmed"
                                         else ResponseRequirement.REQUIRED)
            if action_result.feedback_attempted:
                turn.record_delivery(action_result.feedback or DeliveryOutcome())

    def _complete_turn(self, result: dict) -> None:
        turn = self._turn_outcome.get()
        if turn is None:
            return
        supplied = result.pop("_task_outcome", None)
        if isinstance(supplied, TaskOutcome):
            turn.task = supplied
        if turn.task is None:
            intent = result.get("intent") or ""
            if result.get("needs_confirmation") or intent == "clarification_requested":
                turn.task = TaskOutcome(TaskStatus.NEEDS_INPUT)
            elif intent == "cancelled":
                turn.task = TaskOutcome(TaskStatus.CANCELLED)
            elif intent.startswith(("gate_rejected", "llm_rejected", "noise")) or not intent:
                turn.task = TaskOutcome(TaskStatus.FAILED, handled=False)
            elif intent in {"conversation", "simple_query"}:
                generated = turn.response is not None and turn.response.generation is GenerationState.COMPLETE
                turn.task = TaskOutcome(TaskStatus.COMPLETED if generated else TaskStatus.FAILED)
            else:
                turn.task = TaskOutcome(TaskStatus.UNKNOWN)
        result.update(turn.to_dict())
        if turn.response is not None:
            if turn.response.delivery is not None:
                result["voice_delivery"] = turn.response.delivery.value
            if turn.response.timeout_stage is not None:
                result["timeout_stage"] = turn.response.timeout_stage.value

    async def _try_memory_proposal(self, text: str, result: dict) -> bool:
        """Create a review proposal from an explicit request, never confirm it."""
        import re
        from src.memory.memory_manager import MemoryManager
        if not isinstance(self.memory, MemoryManager):
            return False
        value = text.strip()
        for word in self._wake_words or ("nexa",):
            value = re.sub(r"^"+re.escape(word)+r"\b[, :]*", "", value, count=1, flags=re.IGNORECASE).strip()
        match = re.fullmatch(r"(?:record[aá]|acordate|acu[eé]rdate|guard[aá] en (?:la )?memoria)\s+que\s+(.{1,2000})", value, re.IGNORECASE)
        if match is None:
            return False
        source = self._memory_source.get()
        if source is None:
            result.update(intent="memory_unavailable", success=False,
                          response="No pude preparar ese recuerdo para revisión.",
                          _task_outcome=TaskOutcome(TaskStatus.FAILED, persistence=PersistenceState.NOT_WRITTEN,
                                                   effect=EffectState.NOT_OBSERVED))
        else:
            try:
                proposal = await self.memory.long_term.acall(self.memory.long_term.propose, match[1].strip(), (source,))
                result.update(intent="memory_proposal", success=True, memory_proposal=proposal,
                              memory_review_required=True, response="Dejé la propuesta de recuerdo para que la revises en el panel.",
                              _task_outcome=TaskOutcome(TaskStatus.NEEDS_INPUT, persistence=PersistenceState.COMMITTED,
                                                       effect=EffectState.OBSERVED, observation_scope="software_state"))
            except Exception:
                logger.exception("[Memory] Proposal could not be retained")
                result.update(intent="memory_unavailable", success=False,
                              response="No pude preparar ese recuerdo para revisión.",
                              _task_outcome=TaskOutcome(TaskStatus.UNCERTAIN, persistence=PersistenceState.UNCERTAIN,
                                                       effect=EffectState.UNCERTAIN))
        delivered = await self._speak_response(result["response"])
        result["voice_delivery"] = delivered.delivery.value
        if self._voice_attributor is not None and result.get("success") is True and source is not None:
            # U3b-4b-2: after the reply, whose text and timing never depend on it.
            self._voice_attributor.schedule(result["memory_proposal"]["id"], source)
        return True

    async def _try_voice_verification(self, text: str, result: dict, *, room_id, textual_wake) -> bool:
        """Route a verification phrase, or a near miss of one, before the command gate and the ASR log (U3b-4b-2).

        Nothing changes while no window is open. Only a multi-room
        room-microphone capture can match, and only the open window of its own
        room whose phrase equals the normalised transcript; the window is
        consumed. An utterance close to any open phrase (review L2) takes the
        same turn without an evaluation. The transcript is redacted from the
        result (the turn event), nothing is recorded or logged, the fixed reply
        is spoken for every outcome, and the evaluation runs after it.
        """
        attributor = self._voice_attributor
        if attributor is None:
            return False
        screened = attributor.screen(text, room_id, textual_wake, self._active_sources(), self._wake_words)
        if screened is None:
            return False
        window, source = screened
        result.update(text="", intent="voice_verification", success=True, response=VERIFICATION_REPLY,
                      _task_outcome=TaskOutcome(TaskStatus.QUEUED))
        await self.response_handler.cancel(zone_id=f"zone_{room_id}" if room_id else None)
        delivered = await self._speak_response(VERIFICATION_REPLY)
        result["voice_delivery"] = delivered.delivery.value
        if window is not None:
            attributor.verify(window, source)
        return True

    async def _try_routine_read(self, text: str, result: dict, *, pipeline_start: float) -> bool:
        """Legacy path: answer a routine question from the scheduler store (T033-U1).

        Read-only and never from the legacy Chroma index; detection reads only
        the words of the turn. It runs before the legacy stages (feedback,
        learning, the legacy routine handlers, the vector search), so a
        question such as "¿está activa la rutina riego?" never reaches a
        producer that could act on 'activa' (T033-U1-RV1).

        Returns:
            True when the turn was a routine question and has been answered;
            False for any other turn, and for a consult the read declined
            (no saved routine by that name and no state, run or schedule cue:
            "¿qué rutina de skincare me recomendás?", T033-U1-RV2), which
            continues as it did before this service existed.
        """
        query = self.voice_routines.detect(text) if self.voice_routines is not None else None
        if query is None:
            return False
        started = time.perf_counter()
        read = await self.voice_routines.answer(query)
        if read.declined:
            logger.info("[Routines] voice consult declined: no saved routine named and no routine cue")
            return False
        result["timings"]["routine_check"] = (time.perf_counter() - started) * 1000
        result["intent"] = "routine"
        result["response"] = read.response
        result["success"] = read.outcome.task_completed
        result["_task_outcome"] = read.outcome
        await self._speak_response(result["response"])
        result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
        return True

    async def _try_home_command(self, text: str, result: dict, *, room_id: str | None,
                                pipeline_start: float, extra_gate: bool) -> bool:
        """Own complete extended controls in both modes, with no guessing fallback.

        The model can veto a textual/weak wake capture. It cannot supply slots
        or choose a target. Selection and current capabilities remain mandatory
        in the shared service after resolution and any hook rewrites.
        """
        command = parse_home_command(text, wake_words=self._wake_words)
        if command is None:
            return False
        result["path"] = PathType.FAST_DOMOTICS.value
        result["intent"] = "domotics"
        zone_id = f"zone_{room_id}" if room_id else None
        if extra_gate:
            classification = None
            started = time.perf_counter()
            if self.llm_command_router is not None:
                try:
                    classification = await self._classify_command(text, room_id=room_id)
                except Exception:
                    logger.exception("[HomeVoice] Command gate unavailable")
            result["timings"]["llm_router"] = (time.perf_counter() - started) * 1000
            confidence = getattr(classification, "confidence", None)
            if (
                getattr(classification, "is_command", None) is not True
                or getattr(classification, "rejection_reason", None) == "unavailable"
                or not finite_number(confidence)
                or not self.llm_min_command_confidence <= confidence <= 1
            ):
                result["intent"] = "llm_rejected:home_command"
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return True
        target, reason = await self._resolve_home_command(command, result, room_id=room_id)
        if reason is not None or target is None or target.entity_id is None:
            question = self._home_command_question(command, reason)
            result.update(intent="clarification_requested", needs_confirmation=True,
                          confirmation_question=question, response=question, reason=reason)
            await self._speak_response(question, zone_id=zone_id)
        else:
            request = HomeActionRequest(target.entity_id, command.operation, "voice-parse", **command.parameters)
            effect = compile_effect(request)
            action = {"domain": effect.domain, "service": effect.service, "data": effect.data,
                      "entity_id": target.entity_id, "zone_id": zone_id}
            result["action"] = action
            started = time.perf_counter()
            action_result = await self._voice_actions.execute_with_feedback(action, request_id=CURRENT_REQUEST_ID.get())
            self._record_home_action(action_result)
            receipt = action_result.receipt
            result["timings"]["home_assistant"] = (time.perf_counter() - started) * 1000
            result["receipt"] = asdict(receipt)
            result["success"] = receipt.status == "confirmed"
        result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
        return True

    async def _resolve_home_command(self, command, result: dict, *, room_id: str | None):
        """Resolve an extended home command's target through the shared selection.

        Returns:
            ``(target, reason)``: the resolver's target (or None) and the
            reason it cannot act, or None when it resolved.
        """
        reason = command.reason
        target = None
        if reason is None:
            try:
                target = await self.home_voice_resolver.resolve(command, room_id=room_id)
                reason = target.reason
                if target.observation is not None:
                    result["capability_observation"] = target.observation.to_dict(self.home_voice_resolver.capabilities.now())
            except Exception:
                logger.exception("[HomeVoice] Target resolution unavailable")
                reason = "selection_unavailable"
        return target, reason

    @staticmethod
    def _home_command_question(command, reason: str | None) -> str:
        """What KZA asks when an extended home command cannot act (missing value, target or capability)."""
        if command.reason is not None:
            return {
                "climate": "¿A qué temperatura en grados Celsius o en qué modo querés poner el aire?",
                "cover": "¿Querés abrir, cerrar, detener o poner la persiana en un porcentaje concreto?",
                "light": "¿Qué temperatura de luz en kelvin querés usar?",
                "scene": "¿Qué escena querés activar?",
            }[command.domain]
        if reason == "capability_unavailable":
            return "No tengo una observación actual suficiente para realizar esa operación."
        if reason == "operation_unavailable":
            return "Esa operación no está disponible para el dispositivo seleccionado."
        if command.domain == "scene":
            return "¿Qué escena querés activar? Necesito su nombre exacto entre las escenas seleccionadas."
        return "¿En qué habitación? Necesito un dispositivo seleccionado para esa habitación."

    def bind_routine_capture_gate(self, capture_allowed: AuthorityGuard) -> None:
        """Attach the total-mute gate built after the audio loop."""
        self._routine_capture_allowed = capture_allowed

    async def process_command(self, audio_or_event, *, request_id: str | None = None) -> dict:
        """Compose operation and actual speech receipts for one captured turn."""
        from uuid import uuid4
        turn = TurnOutcome(request_id=request_id or "voice:" + uuid4().hex)
        token = self._turn_outcome.set(turn)
        identity_token = CURRENT_REQUEST_ID.set(turn.request_id)
        try:
            result = await self._process_received_command(audio_or_event)
            self._complete_turn(result)
            result["request_id"] = turn.request_id
            self._log_latency(result)
            return result
        finally:
            CURRENT_REQUEST_ID.reset(identity_token)
            self._turn_outcome.reset(token)

    async def _process_received_command(self, audio_or_event) -> dict:
        """
        Process a complete audio command.

        Routes through orchestrated (multi-user) or legacy (single-user) path
        depending on configuration. Accepts CommandEvent or raw np.ndarray.

        Args:
            audio_or_event: CommandEvent with room metadata, or raw np.ndarray.

        Returns:
            Dict with keys: text, intent, action, response, success,
            latency_ms, user, timings, and optionally path, room.
        """
        from src.pipeline.command_event import CommandEvent, TranscriptSource
        self._memory_source.set(None)
        self._response_sources.set(())
        if self._closed:
            return {"text": "", "intent": "cancelled", "success": False, "response": "", "timings": {}, "latency_ms": 0}

        pretranscribed_text: str | None = None
        audio_provenance = transcript_provenance = transcript_span = transcript_truncated = None
        used_wake_text = False
        ambient_transcript = False
        early_dispatch = False
        ambient_strict = False
        if isinstance(audio_or_event, CommandEvent):
            audio = audio_or_event.audio
            audio_provenance = audio_or_event.audio_provenance
            room_id = audio_or_event.room_id
            early_dispatch = audio_or_event.early_dispatch
            ambient_strict = getattr(audio_or_event, "ambient_strict", False)
            if audio_or_event.capture_complete is False or (
                early_dispatch and audio_or_event.capture_complete is not True
            ):
                return {
                    "text": "", "intent": "gate_rejected:incomplete_capture",
                    "action": None, "response": "", "success": False,
                    "latency_ms": 0, "timings": {}, "user": None, "path": None,
                }
            transcript = audio_or_event.transcript
            if transcript is not None and transcript.covers(audio):
                pretranscribed_text = transcript.text
                transcript_provenance = transcript.transcript_provenance
                transcript_span = transcript.transcript_span
                transcript_truncated = transcript.transcript_truncated
                used_wake_text = transcript.source is TranscriptSource.WAKE
                ambient_transcript = transcript.source is TranscriptSource.AMBIENT
            elif transcript is not None or audio_or_event.wake_text or audio_or_event.partial_command:
                logger.debug("[Capture] Transcript does not cover final audio; running final STT")
        else:
            audio = audio_or_event
            room_id = None

        audio_duration_ms = (
            len(audio) / 16000.0 * 1000.0 if audio is not None and len(audio) else None
        )
        self._last_request_meta = {
            "used_wake_text": used_wake_text,
            "early_dispatch": early_dispatch,
            "audio_duration_ms": audio_duration_ms,
        }
        turn = self._turn_outcome.get()
        if turn is not None:
            turn.capture_meta = self._last_request_meta

        wake_score = getattr(audio_or_event, "wake_score", 1.0)
        textual_wake = ambient_transcript or bool(getattr(audio_or_event, "textual_wake", False))
        # The single-device loop carries its wake score for routine admission,
        # but its command frames do not include the wake word for verification.
        _WAKE_CONTEXT.set(
            (wake_score, getattr(audio_or_event, "wake_vad", None))
            if room_id and not textual_wake and getattr(audio_or_event, "capture_trigger", None) == "wake" else None
        )
        _ENDPOINT_CODE.set(getattr(audio_or_event, "endpoint_code", None))
        _STT_SOURCE.set(None)
        wake_rms = (
            float(np.sqrt(np.mean(np.square(audio))))
            if audio is not None and len(audio) else 0.0
        )

        from src.memory.memory_manager import MemoryManager
        from src.memory.fact_ledger import MemoryOrigin
        from uuid import uuid4
        self._memory_source.set(None)
        self._response_sources.set(())
        try:
            captured_at = getattr(audio_or_event, "captured_at", None)
            if captured_at is None:
                captured_at = getattr(audio_or_event, "timestamp", time.time())
            capture_id = getattr(audio_or_event, "capture_id", str(uuid4()))
            conversation_id = getattr(audio_or_event, "conversation_id", None)
            expires_at = getattr(audio_or_event, "source_expires_at", None)
            origin = MemoryOrigin.AMBIENT if textual_wake else MemoryOrigin.VOICE
            if self._privacy is not None:
                source = self._privacy.source_for_capture(capture_id, captured_at, room_id,
                    conversation_id=conversation_id, expires_at=expires_at, origin=origin)
            elif isinstance(self.memory, MemoryManager):
                source = self.memory.source_for_capture(capture_id, captured_at, room_id,
                    conversation_id=conversation_id, expires_at=expires_at, origin=origin)
            else:
                from src.memory.fact_ledger import MemorySource
                source = MemorySource(capture_id, conversation_id or str(uuid4()), captured_at,
                    expires_at if expires_at is not None else min(captured_at, time.time())+48*3600, origin)
            self._memory_source.set(source)
        except (ValueError, TypeError, RuntimeError, PermissionError):
            logger.warning("[Privacy] Capture provenance unavailable")

        if self._source_policy is not None and not self._current_source_available():
            return {"intent": "cancelled", "success": False, "text": "", "response": "", "timings": {}, "latency_ms": 0}

        from src.memory.producer_context import producer_context
        from src.memory.prompt_context import SourcesInvalidated
        review_options = ({'audio_provenance': audio_provenance,
            'transcript_provenance': transcript_provenance, 'transcript_span': transcript_span,
            'transcript_truncated': transcript_truncated}
            if any(value is not None for value in (audio_provenance, transcript_provenance,
                transcript_span, transcript_truncated)) else {})
        try:
            with producer_context(room_id, self._active_sources()):
                if self.orchestrator_enabled and self._orchestrator:
                    result = await self._process_command_orchestrated(
                        audio, room_id=room_id, pretranscribed_text=pretranscribed_text,
                        ambient_strict=ambient_strict,
                        wake_score=wake_score, wake_rms=wake_rms,
                        textual_wake=textual_wake,
                        capture_trigger=getattr(audio_or_event, "capture_trigger", None),
                        **review_options,
                    )
                else:
                    result = await self._process_command_legacy(
                        audio, room_id=room_id, pretranscribed_text=pretranscribed_text,
                        wake_score=wake_score, wake_rms=wake_rms,
                        textual_wake=textual_wake, ambient_strict=ambient_strict,
                        **review_options,
                    )
        except SourcesInvalidated:
            return {"intent": "cancelled", "success": False, "text": "", "response": "", "timings": {}, "latency_ms": 0}
        self._complete_turn(result)
        classification = result.get("llm_classification")
        if self.llm_command_router is not None and result.get("success") and (result.get("intent") == "domotics" or classification is not None):
            try:
                source = self._memory_source.get()
                options = {"sources": (source,) if source is not None else ()} if isinstance(self.llm_command_router, LLMCommandRouter) else {}
                action = result.get("action") or {}
                self.llm_command_router.record_command(result.get("text", ""),
                    getattr(classification, "intent", None) or action.get("service") or "unknown", **options)
            except Exception as error:
                logger.debug("Command history write skipped: %s", type(error).__name__)
        if self.memory is not None and result.get("success") and result.get("intent") in {"conversation", "domotics"}:
            try:
                source_options = {}
                if isinstance(self.memory, MemoryManager):
                    source = self._memory_source.get()
                    if source is None:
                        return result
                    source_options["sources"] = self._active_sources()
                    result["capture_id"] = source.capture_id
                    result["conversation_id"] = source.conversation_id
                await self._memory_call("record_interaction", user_input=result.get("text", ""),
                    assistant_response=result.get("response", ""), intent=result["intent"], **source_options)
            except Exception:
                logger.exception("[Memory] Completed interaction could not be recorded")
        if self._closed or not self._current_source_available():
            result["text"] = result["response"] = ""
            result.pop("llm_classification", None)
            result.pop("memory_proposal", None)
            result.pop("confirmation_question", None)
        return result

    @staticmethod
    def _note_stt_source(cmd, result: dict) -> None:
        """Record which STT produced the transcript (result and ASR outcomes)."""
        source = getattr(cmd, "stt_source", None)
        source = source if isinstance(source, str) else None
        _STT_SOURCE.set(source)
        if source is not None:
            result["stt_source"] = source

    @staticmethod
    def _asr_signals(signals: dict | None) -> dict:
        """Gate signals plus the transcript's STT source, for ASR outcome events."""
        merged = dict(signals or {})
        source = _STT_SOURCE.get()
        if source is not None:
            merged["stt_source"] = source
        return merged

    def _legacy_household_drop(self, text: str, gate_decision: AcceptanceDecision) -> AcceptanceDecision:
        """Legacy path: the non-immediate rule without the household exemption.

        The command gate lets a claimed reminder ("recordame apagar la luz a
        las once") or routine creation ("creá una rutina para apagar las luces
        a las once") through its non-immediate rule because the orchestrated
        household layer claims the turn and answers it. The legacy path has no
        household layer (its refusal is a later unit, P3F2-R1-Q3), so the
        reminder's or routine's content would reach the extended home command,
        the grammar and the vector search as an immediate action (review
        Q4-RV-1). An accepted turn that carries a non-immediate reason is
        dropped with that reason, as before the exemption; with the real gate
        only an exempted (claimed) turn gets here. A bare "listo" the gate
        let through because a ring is open (signal ``ring_stop``, T030-U2b)
        is dropped as the noise phrase it is on every other turn: the legacy
        timer commands do not stop a ring. Never fails the turn on an
        internal error.

        Args:
            text: Non-empty transcript of the capture.
            gate_decision: The decision after the gate and the wake checks.

        Returns:
            The decision to apply.
        """
        if not gate_decision.accept:
            return gate_decision
        household = self.household_intents.classify(text)
        if household is not None and household.controls_routine:
            return AcceptanceDecision(False, "legacy_routine_control", gate_decision.signals)
        if household is not None:
            if household.family is HouseholdFamily.INTERCOM:
                return AcceptanceDecision(False, "legacy_household_voice", gate_decision.signals)
            if household.family is HouseholdFamily.NOTIFICATION:
                return gate_decision
        if (gate_decision.signals or {}).get("ring_stop") is True:
            # T030-U2b: the legacy timer commands do not stop a ring, so a bare "listo" stays noise here.
            return AcceptanceDecision(False, "noise_phrase:'listo'", gate_decision.signals)
        try:
            reason = non_immediate_action_reason(text)
            if reason is None:
                return gate_decision
            # T033-U1-RV1: the legacy path answers a routine question itself
            # (_try_routine_read, before every producer), so the gate's
            # exemption for it stands.
            if self.voice_routines is not None and self.voice_routines.claims(text):
                return gate_decision
        except Exception as e:  # fail-open like the gate: never drop voice control on an internal bug
            logger.error(f"[LegacyHouseholdDrop] error (keeping the gate decision): {e}")
            return gate_decision
        return AcceptanceDecision(False, reason, gate_decision.signals)

    def _verify_wake(self, text: str, gate_decision: AcceptanceDecision,
                     room_id: str | None) -> AcceptanceDecision:
        """Capture checks after the gate: max-utterance wake word and wake verification.

        A capture dispatched at the post-wake cap (only when
        ``dispatch_at_max_utterance`` is set) is rejected as
        ``max_utterance_without_wake`` unless the transcript names a wake word,
        so long background speech after a false wake goes no further. Then the
        second-stage wake check of an acoustic wake (shadow by default): shadow
        mode only records ``would_reject``; enforce mode turns a missing wake
        word into the gate's ``missing_wake`` rejection. Never fails the
        command on an internal error.

        Args:
            text: Non-empty transcript of the capture.
            gate_decision: The CommandGate decision.
            room_id: Capturing room.

        Returns:
            The decision to apply.
        """
        if (_ENDPOINT_CODE.get() == _MAX_UTTERANCE_CODE and gate_decision.accept
                and not contains_wake_word(text, self._wake_words or ("nexa",))):
            logger.info("[endpoint] code=max_utterance_without_wake room=%s", room_id)
            return AcceptanceDecision(False, "max_utterance_without_wake", gate_decision.signals)
        context = _WAKE_CONTEXT.get()
        if self._wake_verifier is None or context is None:
            return gate_decision
        try:
            wake_score, wake_vad = context
            check = self._wake_verifier.evaluate(text, room_id=room_id, wake_score=wake_score,
                                                 wake_vad=wake_vad, gate_accept=gate_decision.accept)
            self._wake_verifier.schedule_flush()
        except Exception as error:
            logger.warning("[wake-verify] check skipped: %s", type(error).__name__)
            return gate_decision
        if check is not None and check.reject and gate_decision.accept:
            return AcceptanceDecision(False, check.reason, gate_decision.signals)
        return gate_decision

    async def _process_command_orchestrated(self, audio: np.ndarray, room_id: str = None,
                                              pretranscribed_text: str | None = None,
                                              ambient_strict: bool = False,
                                              wake_score: float = 1.0,
                                              wake_rms: float = 0.0,
                                              textual_wake: bool = False,
                                              capture_trigger: str | None = None,
                                              audio_provenance=None, transcript_provenance=None,
                                              transcript_span=None, transcript_truncated=None) -> dict:
        """Process command with multi-user orchestrator.

        ``capture_trigger`` is what opened this capture (``CommandEvent``);
        None for producers that do not say.
        """
        self._response_zone.set(f"zone_{room_id}" if room_id else None)
        result = {
            "text": "",
            "intent": None,
            "action": None,
            "response": "",
            "success": False,
            "latency_ms": 0,
            "timings": {},
            "user": None,
            "path": None
        }

        pipeline_start = time.perf_counter()

        # 1. Process command (STT + Speaker ID + Emotion in parallel).
        #    Si pretranscribed_text viene del early-dispatch, saltamos el STT.
        review_options = ({'audio_provenance': audio_provenance,
            'transcript_provenance': transcript_provenance, 'transcript_span': transcript_span,
            'transcript_truncated': transcript_truncated}
            if any(value is not None for value in (audio_provenance, transcript_provenance,
                transcript_span, transcript_truncated)) else {})
        cmd = await self.command_processor.process_command(
            audio, use_parallel=True, pretranscribed_text=pretranscribed_text, **review_options,
        )
        text = cmd.text
        result["text"] = text
        result["timings"].update(cmd.timings)
        self._note_stt_source(cmd, result)

        if not text.strip():
            # Distinguir "el veto híbrido se comió habla real" de "el STT no
            # oyó nada": el primero merece earcon aunque el RMS sea bajo
            # (incidente 2026-07-25). Ver earcon_gate._REAL_SPEECH_PREFIXES.
            _reason = "stt_veto" if getattr(cmd, "stt_vetoed", False) else "empty"
            result["intent"] = f"gate_rejected:{_reason}"
            result["response"] = ""
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            await self._earcon_and_log(
                _reason, text or "", {}, room_id, wake_score, wake_rms,
                outcome="gate_rejected",
            )
            return result

        # T032-U1 (review T032-NUM-FV-2): a turn that does not answer the room's
        # numbered reminder choice drops it here, before the router ends turns
        # of its own (verification, gate, memory proposal, home clarification,
        # 7B reject), so a number said later never answers it.
        self._drop_reminder_choice_unless_answered(text, room_id)

        # 0. U3b-4b-2: a verification phrase never reaches the gate, its log or the ASR event.
        if await self._try_voice_verification(text, result, room_id=room_id, textual_wake=textual_wake):
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 1a. Command acceptance gate: corta antes del orchestrator si el texto
        #     es claramente ruido (TV, eco del TTS, repetición) o confianza baja.
        #     Evita que esas muestras vayan al slow path LLM (caro + bloquea queue).
        gate_decision = self._verify_wake(text, self.command_gate.evaluate(
            text, stt_confidence=getattr(cmd, "stt_confidence", None)
        ), room_id)
        if not gate_decision.accept:
            logger.info(f"Gate reject ({gate_decision.reason}): {text!r}")
            result["intent"] = f"gate_rejected:{gate_decision.reason}"
            result["success"] = False
            result["response"] = ""
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            await self._earcon_and_log(
                gate_decision.reason, text, gate_decision.signals or {},
                room_id, wake_score, wake_rms, outcome="gate_rejected",
            )
            return result

        household = self.household_intents.classify(text)
        routine_control = household is not None and household.controls_routine
        routine_guard = None
        if routine_control:
            capture_allowed = self._routine_capture_allowed
            sources = self._active_sources()
            privacy = self._privacy
            source = self._memory_source.get()

            def routine_guard() -> bool:
                return (not self._closed and capture_allowed is not None and capture_allowed()
                        and privacy is not None and source is not None
                        and privacy.scopes.buffer_boundary(room_id, origin=source.origin).allows(source.captured_at)
                        and privacy.sources_available(sources))

            # The command gate can accept as a fallback after an internal error.
            # Routine effects require a successful evaluation.
            if (gate_decision.reason != "ok" or textual_wake or ambient_strict or not self.wake_acoustically_confirmed
                    or capture_trigger not in _OWN_WAKE_TRIGGERS
                    or not finite_number(wake_score) or not _GRAMMAR_BONUS_MIN_WAKE_SCORE <= wake_score <= 1
                    or (gate_decision.signals or {}).get("would_reject") or not routine_guard()):
                result["intent"] = "gate_rejected:routine_capture"
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        await alog_asr_outcome(
            self._asr_event_logger, room_id, "accepted", "ok", text,
            self._asr_signals(gate_decision.signals), wake_score, wake_rms,
            sources=(self._memory_source.get(),) if self._memory_source.get() is not None else (),
        )
        # T037-U6: an accepted turn (a barge-in included) stops the remaining
        # parts of a compound still running in this room.
        self._room_turns[room_id] = self._room_turns.get(room_id, 0) + 1
        await self.response_handler.cancel(zone_id=f"zone_{room_id}" if room_id else None)

        if not routine_control and await self._try_memory_proposal(text, result):
            result["latency_ms"] = (time.perf_counter()-pipeline_start)*1000
            return result

        # P3-F2 S5: the household layer runs before the extended home
        # command, the grammar and the 7B, so a claimed turn never reaches a
        # domotics producer (design F26: never move it after the grammar).
        # T032-U1: the answer to a numbered reminder choice ("el primero")
        # goes back to it on the same household path, never to a producer.
        # T037-U5 (user answer T037-U4-Q1): a follow-up or barge-in capture
        # (or an unknown one) has no wake of its own, so it counts as a TV signal: household
        # shortcuts and the extended home command need the 7B veto, and the
        # grammar fast path below is skipped. While :8101 is down, or with no
        # LLMCommandRouter configured (T037-U5-R1), such a turn fails silently
        # and the user repeats it with the wake word.
        borrowed_wake = _capture_without_own_wake(capture_trigger)
        tv_signal = (textual_wake or ambient_strict or wake_score < _GRAMMAR_BONUS_MIN_WAKE_SCORE
                     or borrowed_wake)
        if household is None and (gate_decision.signals or {}).get("ring_stop") is True:
            # T030-U2b: the gate let a bare "listo" through because a ring is open; it stops that ring
            # on the household path (same wake guards and 7B veto as "pará el timer").
            household = RING_STOP_INTENT
        if household is not None or self._answers_reminder_choice(text, room_id):
            return await self._process_household_turn(
                text, household, result, cmd=cmd, audio=audio, room_id=room_id, pipeline_start=pipeline_start,
                tv_signal=tv_signal, capture_trigger=capture_trigger, routine_guard=routine_guard,
            )

        if await self._try_home_command(
            text, result, room_id=room_id, pipeline_start=pipeline_start, extra_gate=tv_signal,
        ):
            return result

        # T037-U4 (user answer T037-U3-Q1): a whole-phrase time/date question
        # on a clear wake skips the 7B veto, like a short household request, so
        # the household clock answers it even while :8101 is down. A weak wake
        # (TV signal or a transcript-based wake engine) or a capture not opened
        # by its own wake (T037-U4-R1) keeps the 7B gate below.
        clock_turn = self._is_clear_wake_clock_question(text, tv_signal=tv_signal, capture_trigger=capture_trigger)
        if clock_turn:
            logger.info("[Clock] clear wake, skipping the 7B veto: %r", text)
        # T033-U1b (user answer 2026-11-19, T033-U1 routines): a routine question
        # the read always answers skips the 7B veto under the same clear-wake rule.
        routine_turn = not clock_turn and self._is_clear_wake_routine_question(
            text, tv_signal=tv_signal, capture_trigger=capture_trigger)
        if routine_turn:
            logger.info("[Routines] clear wake, skipping the 7B veto: %r", text)

        # 1a-grammar. Grammar fast-path: la gramática determinística manda en
        # domótica.  Si extrae intent+entity con alta confianza, construimos
        # una fast_classification y nos saltamos el LLMCommandRouter (poco
        # fiable en el modelo actual: rechaza comandos válidos como 'noise').
        fast_classification = None
        # Wake textual: el utterance YA contiene "nexa" (así matcheó). Grammar
        # lo marca full y ejecutaba luces fantasma; el 7B es el gate. El canal
        # textual no está en el presupuesto <300ms.
        if textual_wake or borrowed_wake:
            # T037-U5: a capture without its own wake (follow-up, barge-in,
            # unknown) never bypasses the 7B either, whatever wake score it carries.
            grammar_cls = None
            logger.info(
                "[GRAMMAR_FASTPATH] skip %s — forcing LLM router: %r",
                "textual_wake" if textual_wake else f"capture_trigger={capture_trigger}", text,
            )
        else:
            grammar_cls = _grammar_fastpath_classification(
                text, self.confidence_threshold,
                # STRICT: TV dispara OWW. Score en banda TV (0.40–0.45): mismo
                # problema — no otorgar bonus a 'prende la luz' genérico.
                wake_confirmed=(
                    self.wake_acoustically_confirmed
                    and not ambient_strict
                    and wake_score >= _GRAMMAR_BONUS_MIN_WAKE_SCORE
                ),
            )
        if grammar_cls is not None:
            logger.info(
                f"[GRAMMAR_FASTPATH] intent={grammar_cls.intent} "
                f"entity={grammar_cls.entity_hint} "
                f"conf={grammar_cls.confidence:.2f} "
                f"— bypassing LLM router: {text!r}"
            )
            fast_classification = grammar_cls

        # 1a-bis. LLM-based command validator (Opción 2 de la sesión wake-fixes).
        # Reemplaza eventualmente el regex+Chroma; por ahora corre en paralelo
        # como gate adicional contra alucinaciones de TV/replays.
        # Si el fast path ya validó (fast_classification != None), saltamos
        # esta llamada cara para no doble-pagar la latencia.
        if fast_classification is not None:
            classification = fast_classification
            result["llm_classification"] = classification
        elif self.llm_command_router is not None and not (clock_turn or routine_turn):
            t_llm = time.perf_counter()
            classification = await self._classify_command(text, room_id=room_id)
            llm_ms = (time.perf_counter() - t_llm) * 1000
            result["timings"]["llm_router"] = llm_ms
            logger.info(
                f"[LLMRouter {llm_ms:.0f}ms] is_command={classification.is_command} "
                f"intent={classification.intent} reason={classification.rejection_reason} "
                f"text={text!r}"
            )
            # Plan #3 OpenClaw — emit after_event(intent) for audit/observability.
            # C3 fix: derive user_id explicitly here (before the early-return for
            # non-commands, and before the formal `user_id = ...` assignment in
            # the user-info block below). locals().get("user_id") was always None.
            # I7 fix: wrap in try/except so a payload bug never takes down the pipeline.
            if self._hooks is not None:
                _emit_user_id = cmd.user.user_id if cmd.user else None
                try:
                    import time as _time
                    from src.hooks import IntentPayload, execute_after_event
                    execute_after_event(
                        self._hooks, "intent",
                        IntentPayload(
                            timestamp=_time.time(),
                            text=text,
                            intent=classification.intent or "unknown",
                            entities=tuple(),  # IntentPayload.entities: tuple[str, ...]
                            user_id=_emit_user_id,
                        ),
                    )
                except Exception as e:
                    logger.error(
                        f"[hooks] failed to emit intent after-event: {e}",
                        exc_info=True,
                    )
            # ¿El comando quedó respaldado por la gramática determinística?
            # Si es así, NO le aplicamos el confidence gate del LLM (la
            # gramática ya es señal de alta confianza).
            grammar_backed = False
            # T037-U5: without its own wake the turn has no signal but the 7B;
            # while :8101 cannot decide it ends silently (no grammar dispatch,
            # no confirmation question) and the user repeats it with the wake.
            if borrowed_wake and classification.rejection_reason == "unavailable":
                result["intent"] = "llm_rejected:unavailable"
                result["success"] = False
                result["response"] = ""
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result
            # Fallback: el LLM no pudo decidir (timeout/error → 'unavailable').
            # NO descartar si la gramática tenía señal parcial.
            if classification.rejection_reason == "unavailable":
                pc = parse_partial_command(text)
                if pc.intent is not None and pc.entity is not None:
                    # señal completa → dispatch best-effort
                    classification.is_command = True
                    classification.intent = pc.intent
                    classification.entity_hint = pc.entity
                    classification.slots = dict(pc.slots)
                    grammar_backed = True
                elif pc.intent is not None or pc.entity is not None:
                    # señal parcial → pedir confirmación por voz
                    question = self._build_confirmation_question(pc)
                    result["needs_confirmation"] = True
                    result["confirmation_question"] = question
                    result["response"] = question
                    result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                    return result
                # sin señal → cae al manejo normal de no-comando abajo
            if not classification.is_command:
                result["intent"] = f"llm_rejected:{classification.rejection_reason or 'unknown'}"
                result["success"] = False
                result["response"] = ""
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result
            # Confidence gate (2026-06-02): el 7B marca is_command=True sobre
            # charla ambiente coherente (TV/conversación) con confianza baja →
            # acción fantasma. Si la confianza es baja Y la gramática NO respaldó
            # el comando, rechazar en vez de dispatchar. (command_detection_rootcause)
            if not grammar_backed and classification.confidence < self.llm_min_command_confidence:
                logger.info(
                    f"[LLMRouter] rechazado por confianza baja "
                    f"({classification.confidence:.2f} < {self.llm_min_command_confidence}): {text!r}"
                )
                _reason_lc = f"low_confidence:{classification.confidence:.2f}"
                result["intent"] = _reason_lc
                result["success"] = False
                result["response"] = ""
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                await self._earcon_and_log(
                    _reason_lc, text, {}, room_id, wake_score, wake_rms,
                    outcome="low_confidence",
                )
                return result
            # Verbo-evidencia (2026-06-06): el 7B ADIVINA turn_on/turn_off con
            # confianza alta sobre texto con el verbo garbleado por el STT
            # far-field ('apagá'→'pero a la luz'→turn_on → acción INVERTIDA en
            # vivo). La auto-confianza no protege de esto (0.95 a 'haba la
            # luz'). Si el intent binario no tiene un verbo de esa acción en
            # el texto → rechazo honesto (el usuario repite) en vez de acción
            # equivocada. grammar_backed implica verbo presente por construcción.
            from src.nlu.command_grammar import llm_intent_evidenced
            if not grammar_backed and not llm_intent_evidenced(
                classification.intent, text
            ):
                logger.info(
                    f"[LLMRouter] intent {classification.intent} sin verbo que "
                    f"lo evidencie en el texto — rechazado: {text!r}"
                )
                _reason_ui = f"unverified_intent:{classification.intent}"
                result["intent"] = _reason_ui
                result["success"] = False
                # Solo sonido (2026-06-15): reemplaza el reprompt de voz por el
                # earcon. La TV no llega a este punto (pasó el CommandGate + 7B);
                # casi seguro es un humano con STT garbleado → earcon apropiado.
                result["response"] = ""
                await self._earcon_and_log(
                    _reason_ui, text, {}, room_id, wake_score, wake_rms,
                    outcome="low_confidence",
                )
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result
            # Resultado válido — guardar para que el caller registre en history
            # tras dispatch exitoso (line ~480, post-orchestrator).
            result["llm_classification"] = classification
        elif borrowed_wake:
            # T037-U5-R1: with no LLMCommandRouter configured nothing can veto
            # a capture without its own wake, so it fails closed (as a weakly
            # woken household claim does) and the user repeats it with the wake.
            logger.info("[LLMRouter] not configured — capture_trigger=%s fails closed: %r", capture_trigger, text)
            result["intent"] = "llm_rejected:no_router"
            result["success"] = False
            result["response"] = ""
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 1b. Confidence gate — low-confidence sensitive commands pide
        #     confirmación antes de dispatchar al orchestrator.
        self._check_confidence_gate(text, result, speak=False)
        if result.get("pending_confirmation"):
            await self._speak_response(result["response"])
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # Speaker metadata does not bind this microphone turn to an account.
        user = None
        emotion = cmd.emotion
        user_id = user.user_id if user else None
        user_name = user.name if user else None

        # 3. Resolve room context
        room_context = None
        if self.room_context_manager and room_id:
            room_context = self.room_context_manager.resolve_room(
                mic_zone_id=room_id,
                user_id=user_id,
            )

        # 4. Detect source zone (room context overrides audio-based detection)
        if room_context:
            zone_id = f"zone_{room_context.room_id}"
            self.response_handler.set_active_zone(zone_id)
        else:
            zone_id = self.audio_manager.detect_source_zone(audio)
            if zone_id:
                self.response_handler.set_active_zone(zone_id)

        if user:
            result["user"] = {
                "name": user_name,
                "permission_level": user.permission_level.name
            }

        logger.info(
            f"[Orchestrator] User={user_name or 'unknown'}, "
            f"Zone={zone_id}, Text={text[:50]}, Emotion={emotion.emotion if emotion else 'none'}"
        )

        # 5. Process with orchestrator
        self._response_zone.set(zone_id)
        on_response = self._queued_reply(zone_id, room_context)

        # Propagar intent + slots del LLMRouter al orchestrator. Migración del
        # patrón ya existente en el path legacy (líneas 962-968 abajo). Sin
        # esto, el dispatcher hacía vector search puro y descartaba el intent
        # clasificado (bug 2026-05-03 — el intent set_brightness con slot 50%
        # se perdía y la acción terminaba como turn_on).
        _llm_classification = result.get("llm_classification")
        _service_filter = None
        _query_slots = None
        if _llm_classification is not None:
            _intent = _llm_classification.intent
            # Solo propagar intents que el chroma_sync conoce como `service`
            # de HA. Intents de alto nivel (set_brightness, set_temperature)
            # no son services HA — el chroma_sync los rechazaría.
            if _intent in ("turn_on", "turn_off"):
                _service_filter = _intent
            _query_slots = dict(_llm_classification.slots or {}) or None

        target_entity_id = None
        # Target selection is deterministic after either the acoustic grammar
        # or the textual channel's LLM gate has accepted the command.
        default_target = parse_default_light_target(text)
        # T037-U5-R1: a light location the rooms cannot resolve asks, whatever
        # accepted the turn (grammar, 7B, or no model at all); a vector search
        # never guesses the light.
        if (
            _dispatch_targets_light(text, result.get("llm_classification"))
            and has_unresolved_light_location(text)
        ):
            result["intent"] = "unresolved_target"
            result["response"] = "¿En qué ambiente querés controlar la luz?"
            await self._speak_response(result["response"], zone_id=zone_id)
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result
        if default_target is not None and self.room_context_manager is not None:
            target_room = default_target.room or (room_context.room_id if room_context else None)
            config = self.room_context_manager.get_room_config(target_room) if target_room else None
            if config is None or not config.default_light:
                result["intent"] = "unresolved_target"
                result["response"] = "No tengo una luz configurada para ese ambiente."
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                await self._speak_response(result["response"], zone_id=zone_id)
                return result
            target_entity_id = config.default_light
            if default_target.slots.keys() == {"brightness_pct"}:
                _service_filter = "turn_on"
                _query_slots = dict(default_target.slots)

        dispatch_result = await self._orchestrator.process(
            user_id=user_id,
            text=text,
            audio=audio if not user_id else None,
            zone_id=zone_id,
            # The capturing mic's room is also the turn's origin room (P3F1-R3
            # option A); the dispatcher normalizes it, never the resolved room.
            context_room_id=room_id,
            on_response=on_response,
            service_filter=_service_filter,
            query_slots=_query_slots,
            **({"target_entity_id": target_entity_id} if target_entity_id else {}),
            memory_sources=(self._memory_source.get(),) if self._memory_source.get() is not None else (),
        )
        return await self._finish_orchestrated_turn(
            dispatch_result, result, text=text, zone_id=zone_id, room_context=room_context, emotion=emotion,
            pipeline_start=pipeline_start,
        )

    def _queued_reply(self, zone_id: str | None, room_context):
        """Callback that speaks a queued slow-path reply in the turn's zone."""
        async def on_response(dispatch_result):
            if dispatch_result.was_queued and dispatch_result.queue_position and not self.response_handler.is_responding:
                await self._speak_response(
                    dispatch_result.response,
                    zone_id=zone_id,
                    room_context=room_context,
                )
        return on_response

    def _is_clear_wake_clock_question(self, text: str, *, tv_signal: bool, capture_trigger: str | None) -> bool:
        """Whether a turn is a clock question that skips the 7B veto (T037-U4).

        Only a whole-phrase time or date question (the dispatcher's own
        recognizer, with the same wake terms) on a clear wake qualifies. A
        weak wake as P3-F2 defines it (textual wake, ambient_strict, a wake
        score below the grammar bonus, or a transcript-based wake engine)
        keeps the 7B gate. The wake is decided per capture (T037-U4-R1): only
        a capture opened by its own acoustic wake counts. A follow-up or
        barge-in capture carries the score of the last real wake, so it keeps
        the gate, and so does a producer that does not say what opened the
        capture (None: the single-room loop, raw audio).

        Args:
            text: Accepted transcript of the turn.
            tv_signal: Textual wake, ambient_strict, a low wake score or a
                capture without its own wake (T037-U5).
            capture_trigger: What opened this capture, or None if unknown.

        Returns:
            True when the turn goes to the dispatcher with no 7B call.
        """
        if not self._has_clear_own_wake(tv_signal=tv_signal, capture_trigger=capture_trigger):
            return False
        return parse_clock_query(text, self._clock_wake_words) is not None

    def _is_clear_wake_routine_question(self, text: str, *, tv_signal: bool, capture_trigger: str | None) -> bool:
        """Whether a turn is a routine question that skips the 7B veto (T033-U1b).

        User answer of 2026-11-19 (T033-U1 routines): a routine question with
        a clear own wake skips the 7B, like the clock questions of T037-U4,
        and with the same wake rule (``_has_clear_own_wake``); a follow-up or
        barge-in capture keeps the veto (T037-U5). Only a question the read
        always answers qualifies (``VoiceRoutineService.claims``: a list, or
        a consult about a saved routine's state, runs or schedule). An uncued
        consult can be a general question that only mentions "rutina"
        (T033-U1-RV2) and goes to the conversation path when no saved routine
        has that name, so it keeps the 7B gate. Words only, no I/O.

        Args:
            text: Accepted transcript of the turn.
            tv_signal: Textual wake, ambient_strict, a low wake score or a
                capture without its own wake (T037-U5).
            capture_trigger: What opened this capture, or None if unknown.

        Returns:
            True when the turn goes to the dispatcher with no 7B call.
        """
        if self.voice_routines is None:
            return False
        if not self._has_clear_own_wake(tv_signal=tv_signal, capture_trigger=capture_trigger):
            return False
        return self.voice_routines.claims(text)

    def _is_clear_wake_household_routine_query(self, text: str, household: HouseholdIntent | None, *,
                                               tv_signal: bool, capture_trigger: str | None) -> bool:
        """Whether a household-claimed turn is a routine question that skips the veto (T033-U1b-RV1).

        The user's answer (routine questions with a clear own wake skip the
        7B, like the clock questions) does not depend on the length rule
        ``unvetoed_max_words``, as the clock's bypass does not. Only a turn
        the household layer claims as a routine QUERY qualifies, with
        ``_is_clear_wake_routine_question`` (clear own wake and
        ``VoiceRoutineService.claims``). A compound claim (R5) that also
        names a list keeps the S6 veto, since one of its parts may act.

        Args:
            text: Accepted transcript of the turn.
            household: The household layer's claim, or None for the answer
                to a numbered reminder choice.
            tv_signal: Textual wake, ambient_strict, a low wake score or a
                capture without its own wake (T037-U5).
            capture_trigger: What opened this capture, or None if unknown.

        Returns:
            True when the claimed turn is dispatched with no 7B call.
        """
        if household is None or household.family is not HouseholdFamily.ROUTINE:
            return False
        if household.operation is not HouseholdOperation.QUERY:
            return False
        return self._is_clear_wake_routine_question(text, tv_signal=tv_signal, capture_trigger=capture_trigger)

    def _has_clear_own_wake(self, *, tv_signal: bool, capture_trigger: str | None) -> bool:
        """Whether this capture was opened by a clear acoustic wake of its own (T037-U4, T037-U4-R1).

        Not a weak wake as P3-F2 defines it (TV signal or a transcript-based
        wake engine), and the producer tags the capture ``wake``. A follow-up
        or barge-in capture carries the last real wake's score (T037-U5), and
        ``inline`` or an untagged capture (None) keep the gate too, as they
        did for the clock.
        """
        return not tv_signal and self.wake_acoustically_confirmed and capture_trigger == CaptureTrigger.WAKE

    async def _household_veto_passes(self, text: str, result: dict, *, room_id: str | None, tv_signal: bool) -> bool:
        """Run the 7B as a veto on a weakly woken claimed turn (P3-F2 S6).

        Same acceptance test as the extended home command: is_command, not
        'unavailable', and a finite confidence within the configured range.
        The model can only stop the turn; it never chooses a family, a part,
        a target or a service. Without a configured LLMCommandRouter a turn
        with a TV signal (textual wake, ambient_strict, low wake score, or a
        capture without its own wake, T037-U5) fails closed; otherwise it proceeds as an unclaimed turn would without a
        model gate.
        """
        if self.llm_command_router is None:
            return not tv_signal
        classification = None
        started = time.perf_counter()
        try:
            classification = await self._classify_command(text, room_id=room_id)
        except Exception:
            logger.exception("[Household] Veto unavailable")
        result["timings"]["llm_router"] = (time.perf_counter() - started) * 1000
        confidence = getattr(classification, "confidence", None)
        return (
            getattr(classification, "is_command", None) is True
            and getattr(classification, "rejection_reason", None) != "unavailable"
            and finite_number(confidence)
            and self.llm_min_command_confidence <= confidence <= 1
        )

    def _answers_reminder_choice(self, text: str, room_id: str | None) -> bool:
        """The turn answers a numbered reminder choice open in its room (T032-U1)."""
        awaits = getattr(self._orchestrator, "awaits_reminder_choice", None)
        return callable(awaits) and awaits(text, room_id) is True

    def _drop_reminder_choice_unless_answered(self, text: str, room_id: str | None) -> None:
        """Drop the room's open reminder choice unless this turn answers it (review T032-NUM-FV-2)."""
        if self._answers_reminder_choice(text, room_id):
            return
        drop = getattr(self._orchestrator, "drop_reminder_choice", None)
        if callable(drop) and drop(room_id) is True:
            logger.info("[Household] reminder choice dropped by another turn room=%s", room_id)

    async def _process_household_turn(self, text: str, household: HouseholdIntent | None, result: dict, *, cmd,
                                      audio, room_id: str | None, pipeline_start: float, tv_signal: bool,
                                      capture_trigger: str | None = None,
                                      routine_guard: AuthorityGuard | None = None) -> dict:
        """Dispatch a claimed household turn (P3-F2 S6).

        A weak wake (TV signal, a capture without its own wake such as a
        follow-up or barge-in (T037-U5), a transcript-based wake engine, or a
        turn longer than ``household_intents.unvetoed_max_words``) needs the
        7B veto; a rejection ends the turn with no effect. The turn never
        carries a service_filter, slots or a target from any producer, never
        uses audio_manager.detect_source_zone, and its origin room is only the
        capturing microphone's room (the dispatcher derives it from
        ``context_room_id``, P3F1-R3 option A). ``household`` is None for
        the answer to a numbered reminder choice (T032-U1), which takes the
        same veto. A routine question with a clear own wake skips it whatever
        its length (T033-U1b-RV1, ``_is_clear_wake_household_routine_query``).
        """
        summary = household.summary() if household is not None else "reminder_choice"
        result["household"] = summary
        logger.info("[Household] claimed %s", summary)
        weak_wake = (
            tv_signal
            or not self.wake_acoustically_confirmed
            or len(text.split()) > self.household_intents.settings.unvetoed_max_words
        )
        if weak_wake and self._is_clear_wake_household_routine_query(
                text, household, tv_signal=tv_signal, capture_trigger=capture_trigger):
            logger.info("[Routines] clear wake, skipping the household 7B veto: %r", text)
            weak_wake = False
        if weak_wake and not await self._household_veto_passes(text, result, room_id=room_id, tv_signal=tv_signal):
            logger.info("[Household] vetoed %s", summary)
            result["intent"] = "llm_rejected:household"
            result["success"] = False
            result["response"] = ""
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # The reply zone follows the room context as for any turn; the audio
        # source-zone detector is never consulted for a household object.
        room_context = None
        if self.room_context_manager and room_id:
            room_context = self.room_context_manager.resolve_room(mic_zone_id=room_id, user_id=None)
        zone_id = f"zone_{room_context.room_id}" if room_context else None
        if zone_id:
            self.response_handler.set_active_zone(zone_id)
            self._response_zone.set(zone_id)
        if (household is not None and household.family is HouseholdFamily.COMPOUND
                and household.operation is HouseholdOperation.EXECUTE
                and not household.controls_routine
                and self.household_intents.settings.compound_execute):
            return await self._process_compound_turn(
                household, result, text=text, cmd=cmd, room_id=room_id, room_context=room_context,
                zone_id=zone_id, pipeline_start=pipeline_start)
        dispatch_result = await self._orchestrator.process(
            user_id=None,
            text=text,
            audio=audio,
            zone_id=zone_id,
            context_room_id=room_id,
            on_response=self._queued_reply(zone_id, room_context),
            service_filter=None,
            query_slots=None,
            memory_sources=(self._memory_source.get(),) if self._memory_source.get() is not None else (),
            **({"routine_guard": routine_guard} if routine_guard is not None else {}),
        )
        return await self._finish_orchestrated_turn(
            dispatch_result, result, text=text, zone_id=zone_id, room_context=room_context, emotion=cmd.emotion,
            pipeline_start=pipeline_start,
        )

    async def _process_compound_turn(self, household: HouseholdIntent, result: dict, *, text: str, cmd,
                                     room_id: str | None, room_context, zone_id: str | None,
                                     pipeline_start: float) -> dict:
        """Run an admissible compound now, part by part (T037-U6, user answer Q21).

        The turn already passed the household veto once (P3-F2 S6). Each part
        runs in spoken order through the gates and handler it would meet
        alone (:meth:`_run_compound_part`), with its own request id
        (``<turn>:part<n>``) and receipt. A failed, refused or uncertain part
        never stops independent later parts; another turn accepted in this
        room (a barge-in) stops the remaining ones. One reply reports every
        part, and the turn's TaskOutcome aggregates them (design
        ``compound_contract``).
        """
        started = self._room_turns.get(room_id, 0)

        async def run_part(part: CompoundPartContext):
            return await self._run_compound_part(part, room_id=room_id, room_context=room_context,
                                                 zone_id=zone_id)

        run = await self._compound.run(household.parts, run_part,
                                       interrupted=lambda: self._room_turns.get(room_id, 0) != started,
                                       part_target=compound_part_target)
        return await self._finish_orchestrated_turn(
            compound_dispatch_result(run), result, text=text, zone_id=zone_id, room_context=room_context,
            emotion=cmd.emotion, pipeline_start=pipeline_start)

    async def _run_compound_part(self, part: CompoundPartContext, *, room_id: str | None, room_context,
                                 zone_id: str | None):
        """One compound part through the gates it would meet alone, without the 7B (T037-U6).

        A claimed part goes to its family path. An unclaimed part meets, in
        order: the security-device refusal, the extended home command
        (shared selection and capabilities, then HomeActionService), the
        confidence gate for sensitive actions, the unresolved-light-location
        question (T037-U5), the room's default light, and the dispatcher's
        domotics or music path with the part's own grammar service and
        slots. The 7B never chooses a part, a target or a service.
        """
        segment = part.part.segment
        memory = (self._memory_source.get(),) if self._memory_source.get() is not None else ()

        async def dispatch(**evidence):
            return await self._orchestrator.process(
                user_id=None, text=segment, audio=None, zone_id=zone_id, context_room_id=room_id,
                memory_sources=memory, compound_part=part, **evidence)

        if part.part.intent is not None or security_order(segment) is not None:
            return await dispatch(service_filter=None, query_slots=None)
        command = parse_home_command(segment, wake_words=self._wake_words)
        if command is not None:
            return await self._home_command_part(command, part, room_id=room_id, zone_id=zone_id)
        gate: dict = {}
        self._check_confidence_gate(segment, gate, speak=False)
        if gate.get("pending_confirmation"):
            return self._part_question(gate["response"], "pending_confirmation")
        parsed = parse_command(segment)
        service_filter = parsed.intent if parsed.intent in ("turn_on", "turn_off") else None
        query_slots = dict(parsed.slots) or None
        if _dispatch_targets_light(segment, None) and has_unresolved_light_location(segment):
            return self._part_question("¿En qué ambiente querés controlar la luz?", "unresolved_target")
        target_entity_id = None
        default_target = parse_default_light_target(segment)
        if default_target is not None and self.room_context_manager is not None:
            target_room = default_target.room or (room_context.room_id if room_context else None)
            config = self.room_context_manager.get_room_config(target_room) if target_room else None
            if config is None or not config.default_light:
                return self._part_question("No tengo una luz configurada para ese ambiente.", "unresolved_target")
            target_entity_id = config.default_light
            if default_target.slots.keys() == {"brightness_pct"}:
                service_filter = "turn_on"
                query_slots = dict(default_target.slots)
        return await dispatch(service_filter=service_filter, query_slots=query_slots,
                              **({"target_entity_id": target_entity_id} if target_entity_id else {}))

    async def _home_command_part(self, command, part: CompoundPartContext, *, room_id: str | None,
                                 zone_id: str | None):
        """An extended home command part: shared selection, then HomeActionService with the part's id."""
        target, reason = await self._resolve_home_command(command, {}, room_id=room_id)
        if reason is not None or target is None or target.entity_id is None:
            return self._part_question(self._home_command_question(command, reason), "clarification_requested")
        request = HomeActionRequest(target.entity_id, command.operation, "voice-parse", **command.parameters)
        effect = compile_effect(request)
        action = {"domain": effect.domain, "service": effect.service, "data": effect.data,
                  "entity_id": target.entity_id, "zone_id": zone_id}
        skipped = part.check(target.entity_id, effect.service, effect.data)
        if skipped is not None:
            logger.info("[Compound] part %d not sent: %s", part.index, skipped.value)
            return DispatchResult(PathType.FAST_DOMOTICS, Priority.HIGH, False, "", intent="domotics",
                                  action=action, outcome=part.skip(skipped))
        action_result = await self._compound_voice_actions.execute_with_feedback(action, request_id=part.request_id)
        receipt = action_result.receipt
        confirmed = receipt.status == "confirmed"
        return DispatchResult(
            PathType.FAST_DOMOTICS, Priority.HIGH, confirmed, "", intent="domotics", action=action,
            receipt=receipt, outcome=receipt.task_outcome,
            response_requirement=ResponseRequirement.SILENT if confirmed else ResponseRequirement.REQUIRED)

    @staticmethod
    def _part_question(question: str, intent: str):
        """A compound part that asks instead of acting: NEEDS_INPUT, nothing sent."""
        return DispatchResult(PathType.FAST_DOMOTICS, Priority.HIGH, False, question, intent=intent,
                              outcome=TaskOutcome(TaskStatus.NEEDS_INPUT, dispatch=DispatchState.NOT_SENT,
                                                  effect=EffectState.NOT_OBSERVED))

    async def _finish_orchestrated_turn(self, dispatch_result, result: dict, *, text: str, zone_id: str | None,
                                        room_context, emotion, pipeline_start: float) -> dict:
        """Record the dispatch outcome, speak the reply and log the turn."""
        if isinstance(getattr(dispatch_result, "memory_sources", None), tuple):
            self._response_sources.set(dispatch_result.memory_sources)
        if dispatch_result.intent in {"conversation", "cancelled"} and not self._current_source_available():
            result.update(intent="cancelled", success=False, response="")
            result["latency_ms"] = (time.perf_counter()-pipeline_start)*1000
            return result

        # 6. Build result
        result["intent"] = dispatch_result.intent
        result["response"] = dispatch_result.response
        result["success"] = dispatch_result.success
        result["action"] = dispatch_result.action
        result["path"] = dispatch_result.path.value if dispatch_result.path else None
        result["timings"].update(dispatch_result.timings)
        observation = getattr(dispatch_result, "observation", None)
        if isinstance(observation, Observation):
            result["source_observation"] = observation.to_dict(self.home_voice_resolver.capabilities.now())
        turn = self._turn_outcome.get()
        if turn is not None:
            if isinstance(dispatch_result.outcome, TaskOutcome):
                turn.task = dispatch_result.outcome
            if isinstance(dispatch_result.delivery_outcome, DeliveryOutcome):
                turn.record_delivery(dispatch_result.delivery_outcome)
            if isinstance(dispatch_result.response_requirement, ResponseRequirement):
                turn.response_requirement = dispatch_result.response_requirement
        if isinstance(dispatch_result.receipt, HomeActionResult):
            result["receipt"] = asdict(dispatch_result.receipt)
        intercom_receipt = getattr(dispatch_result, "intercom_receipt", None)
        if intercom_receipt is not None:
            result["intercom_receipt"] = intercom_receipt
        compound_parts = getattr(dispatch_result, "compound_parts", None)
        if isinstance(compound_parts, tuple):
            result["compound_parts"] = list(compound_parts)
        for key in ("timeout_stage", "voice_delivery"):
            value = getattr(dispatch_result, key, None)
            if isinstance(value, str):
                result[key] = value

        # 6b. Attach room context to result
        if room_context:
            result["room"] = {
                "id": room_context.room_id,
                "name": room_context.room_name,
                "confidence": room_context.confidence,
                "source": room_context.source.value,
            }

        # 7. Speak response
        #
        # Domótica EXITOSA = silenciosa (2026-07-30). La regla ya vivía a
        # medias en el dispatcher, que dispara el HA call fire-and-forget
        # "porque el usuario valida visualmente ... no quiere TTS ack" y baja
        # `home_assistant` a 0ms en el camino crítico — pero devolvía igual un
        # `response` (la description del vector search) que se sintetizaba acá.
        #
        # Medido sobre "prendé la luz del living": 308ms de Kokoro, y como
        # `speak()` es SÍNCRONO bloqueaba el event loop justo cuando corría la
        # task del HA call. La luz prendía a los 521ms en vez de ~210ms: se
        # pagaba media latencia por locutar un ack que nadie pidió. Encima la
        # description es texto variable, así que ni pegaba en el response_cache.
        #
        # Los FALLOS sí hablan: un fallo mudo es justo lo que hizo invisible el
        # bug del primer comando post-idle (2026-07-25).
        _domotics_ok = (
            dispatch_result.intent == "domotics" and dispatch_result.success
        )
        if dispatch_result.speech_handled is not True and not _domotics_ok:
            emotion_adj = emotion.response_adjustment if emotion else None
            operation = self._speak_response(
                result["response"], zone_id=zone_id, emotion_adjustment=emotion_adj, room_context=room_context,
            )
            try:
                delivery = await run_private(operation, self._current_source_available) if dispatch_result.intent == "conversation" else await operation
            except SourcesInvalidated:
                result.update(intent="cancelled", success=False, response="")
                return result
            result["voice_delivery"] = delivery.delivery.value
            if dispatch_result.intent == "conversation" and not delivery.success:
                result["success"] = False

        # 8. Logging
        result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000

        if self.event_logger and result["success"] and result["intent"] == "domotics":
            await self._log_domotics_event(
                entity_id=result["action"]["entity_id"] if result["action"] else None,
                action=result["action"]["service"] if result["action"] else None,
                trigger_phrase=text
            )

        return result

    async def _process_command_legacy(self, audio: np.ndarray, room_id: str = None,
                                        pretranscribed_text: str | None = None,
                                        wake_score: float = 1.0,
                                        wake_rms: float = 0.0,
                                        textual_wake: bool = False,
                                        ambient_strict: bool = False,
                                        audio_provenance=None, transcript_provenance=None,
                                        transcript_span=None, transcript_truncated=None) -> dict:
        """Legacy processing (single-user) for backwards compatibility."""
        self._response_zone.set(f"zone_{room_id}" if room_id else None)
        result = {
            "text": "",
            "intent": None,
            "action": None,
            "response": "",
            "success": False,
            "latency_ms": 0,
            "timings": {},
            "user": None
        }

        pipeline_start = time.perf_counter()

        # 1. Process command (salteando STT si viene de early dispatch)
        review_options = ({'audio_provenance': audio_provenance,
            'transcript_provenance': transcript_provenance, 'transcript_span': transcript_span,
            'transcript_truncated': transcript_truncated}
            if any(value is not None for value in (audio_provenance, transcript_provenance,
                transcript_span, transcript_truncated)) else {})
        cmd = await self.command_processor.process_command(
            audio, use_parallel=True, pretranscribed_text=pretranscribed_text, **review_options,
        )
        text = cmd.text
        result["text"] = text
        result["timings"].update(cmd.timings)
        self._note_stt_source(cmd, result)

        if not text.strip():
            # Igual que en el path orchestrated: el veto híbrido no es lo mismo
            # que una captura muda (incidente 2026-07-25).
            _reason = "stt_veto" if getattr(cmd, "stt_vetoed", False) else "empty"
            result["intent"] = f"gate_rejected:{_reason}"
            result["response"] = ""
            await self._earcon_and_log(
                _reason, text or "", {}, room_id, wake_score, wake_rms,
                outcome="gate_rejected",
            )
            return result

        # 0. U3b-4b-2: a verification phrase never reaches the gate, its log or the ASR event.
        if await self._try_voice_verification(text, result, room_id=room_id, textual_wake=textual_wake):
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 1a. Command acceptance gate — mismo corto-circuito que el path orchestrated
        gate_decision = self._verify_wake(text, self.command_gate.evaluate(
            text, stt_confidence=getattr(cmd, "stt_confidence", None)
        ), room_id)
        gate_decision = self._legacy_household_drop(text, gate_decision)
        if not gate_decision.accept:
            logger.info(f"Gate reject ({gate_decision.reason}): {text!r}")
            result["intent"] = f"gate_rejected:{gate_decision.reason}"
            result["success"] = False
            result["response"] = ""
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            await self._earcon_and_log(
                gate_decision.reason, text, gate_decision.signals or {},
                room_id, wake_score, wake_rms, outcome="gate_rejected",
            )
            return result

        await alog_asr_outcome(
            self._asr_event_logger, room_id, "accepted", "ok", text,
            self._asr_signals(gate_decision.signals), wake_score, wake_rms,
            sources=(self._memory_source.get(),) if self._memory_source.get() is not None else (),
        )
        # T037-U6: an accepted turn (a barge-in included) stops the remaining
        # parts of a compound still running in this room.
        self._room_turns[room_id] = self._room_turns.get(room_id, 0) + 1
        await self.response_handler.cancel(zone_id=f"zone_{room_id}" if room_id else None)

        household = self.household_intents.classify(text)
        if household is not None and household.family is HouseholdFamily.NOTIFICATION:
            return await self._finish_orchestrated_turn(
                notification_dnd_refusal(), result, text=text, zone_id=self._response_zone.get(),
                room_context=None, emotion=cmd.emotion, pipeline_start=pipeline_start,
            )

        if await self._try_memory_proposal(text, result):
            result["latency_ms"] = (time.perf_counter()-pipeline_start)*1000
            return result

        if await self._try_home_command(
            text, result, room_id=room_id, pipeline_start=pipeline_start,
            extra_gate=textual_wake or ambient_strict or wake_score < _GRAMMAR_BONUS_MIN_WAKE_SCORE,
        ):
            return result

        user = None
        emotion = cmd.emotion
        user_id = user.user_id if user else None

        if user:
            result["user"] = {
                "name": user.name,
                "permission_level": user.permission_level.name
            }

        # Resolve room context
        room_context = None
        if self.room_context_manager and room_id:
            room_context = self.room_context_manager.resolve_room(
                mic_zone_id=room_id,
                user_id=user_id,
            )
        if room_context:
            result["room"] = {
                "id": room_context.room_id,
                "name": room_context.room_name,
                "confidence": room_context.confidence,
                "source": room_context.source.value,
            }

        if room_context is not None:
            self._response_zone.set(f"zone_{room_context.room_id}")
        logger.info(f"[STT {cmd.timings.get('stt', 0):.0f}ms] {text}")

        # Plan #3 OpenClaw — emit after_event(stt) for audit/observability.
        # C3 fix: user_id and room_id are both already in scope here (user_id
        # is defined at the top of legacy path; room_id is a method param).
        # I7 fix: wrap in try/except so a payload bug never takes down the pipeline.
        if self._hooks is not None:
            try:
                import time as _time
                from src.hooks import SttPayload, execute_after_event
                execute_after_event(
                    self._hooks, "stt",
                    SttPayload(
                        timestamp=_time.time(),
                        text=text,
                        latency_ms=float(cmd.timings.get("stt", 0.0)),
                        user_id=user_id,
                        zone_id=room_id,
                        success=True,
                    ),
                )
            except Exception as e:
                logger.error(
                    f"[hooks] failed to emit stt after-event: {e}",
                    exc_info=True,
                )

        # T033-U1-RV1: a routine question is read from the scheduler store before
        # the confidence gate, feedback, learning and the legacy routine handlers.
        if await self._try_routine_read(text, result, pipeline_start=pipeline_start):
            return result

        # 1b. Confidence gate — si confidence baja y combo sensible, pide
        #     confirmación en vez de ejecutar. Comandos sin intent/entity o
        #     reversibles con confidence baja siguen de largo (con log).
        self._check_confidence_gate(text, result, speak=False)
        if result.get("pending_confirmation"):
            await self._speak_response(result["response"])
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 2. Check feedback
        feedback_result = await self._check_feedback(text)
        if feedback_result["is_feedback"]:
            result["intent"] = "feedback"
            result["response"] = feedback_result["response"]
            result["success"] = feedback_result.get("success", True)
            result["_task_outcome"] = TaskOutcome(
                TaskStatus.COMPLETED if result["success"] else TaskStatus.FAILED,
                persistence=PersistenceState.COMMITTED if result["success"] else PersistenceState.NOT_WRITTEN,
                effect=EffectState.OBSERVED if result["success"] else EffectState.NOT_OBSERVED,
                observation_scope="software_state" if result["success"] else None)
            await self._speak_response(result["response"])
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 3. Check pending suggestion
        if self._pending_suggestion:
            suggestion_result = await self._handle_suggestion_response(text)
            if suggestion_result["handled"]:
                result["intent"] = "suggestion_response"
                result["response"] = suggestion_result["response"]
                result["success"] = suggestion_result.get("success", True)
                await self._speak_response(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # 4. Check command learner
        if self.command_learner:
            learning_result = self.command_learner.handle(text, self.ha, self.chroma)
            if learning_result["handled"]:
                result["intent"] = "learning"
                result["response"] = learning_result["response"]
                result["success"] = learning_result.get("success", True)
                await self._speak_response(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # 5. Check enrollment
        if self.enrollment and (self.enrollment.is_active or self._is_enrollment_command(text)):
            enrollment_result = self.enrollment.handle(
                text=text,
                audio=audio,
                current_user=user
            )
            if enrollment_result["handled"]:
                result["intent"] = "enrollment"
                result["response"] = enrollment_result["response"]
                result["success"] = enrollment_result["state"].name == "COMPLETED"
                result["_task_outcome"] = (TaskOutcome(TaskStatus.COMPLETED,
                    persistence=PersistenceState.COMMITTED, effect=EffectState.OBSERVED,
                    observation_scope="software_state") if result["success"] else TaskOutcome(TaskStatus.NEEDS_INPUT))
                await self._speak_response(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # 6. Check sync command (exact phrases; ADULT 'sync_commands' permission)
        if self._is_sync_command(text):
            result["intent"] = "sync"
            if not self._may_sync_commands(user):
                logger.info("[Sync] command sync refused: no identified user with sync_commands")
                result["response"] = "Solo un adulto identificado puede sincronizar los comandos."
                result["_task_outcome"] = TaskOutcome(TaskStatus.FAILED, persistence=PersistenceState.NOT_WRITTEN)
            else:
                await self._speak_response("Sincronizando comandos...")
                await self._run_command_sync(result)
            await self._speak_response(result["response"])
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 6b. Check timer command (via FeatureManager)
        if self.features and self.features.timer_manager:
            timer_result = self.features.timer_manager.handle_voice_command(
                text,
                user_id=user.user_id if user else None,
                zone_id=self.audio_manager.detect_source_zone(audio) or "default"
            )
            if timer_result["handled"]:
                result["intent"] = "timer"
                result["_task_outcome"] = timer_result.get("outcome")
                result["response"] = timer_result["response"]
                result["success"] = True
                await self._speak_response(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # 7. Check routine (new system with VoiceRoutineHandler)
        t_routine = time.perf_counter()

        # Try the new voice handler first (more complete)
        if self.voice_routine_handler:
            user_id = user.user_id if user else None
            routine_result = await self.voice_routine_handler.handle(text, user_id)

            if routine_result["handled"]:
                result["timings"]["routine_check"] = (time.perf_counter() - t_routine) * 1000
                result["intent"] = "routine"
                result["response"] = routine_result["response"]
                result["success"] = routine_result["success"]
                result["_task_outcome"] = legacy_routine_outcome(bool(routine_result["success"]))
                await self._speak_response(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # Fallback to original RoutineManager (compatibility)
        routine_result = await self.routines.handle(text)
        result["timings"]["routine_check"] = (time.perf_counter() - t_routine) * 1000

        if routine_result["handled"]:
            result["intent"] = "routine"
            result["response"] = routine_result["response"]
            result["success"] = routine_result["success"]
            result["_task_outcome"] = legacy_routine_outcome(bool(routine_result["success"]))
            await self._speak_response(result["response"])
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 7b. Guard de intents que requieren dominio HA específico ausente.
        # Sin esto, el probe ambiguo de luces más abajo puede mapear (por
        # similitud de embedding) un `set_temperature` a `light.balcon` o un
        # `set_volume` a una luz random — visto en logs 2026-05-02 06:57:02:
        # 'Nexa bajá la temperatura del aire' → light.turn_off@light.balcon.
        _llm_classification = result.get("llm_classification")
        if _llm_classification is not None and _llm_classification.intent in _INTENT_REQUIRED_DOMAINS:
            required_domain = _INTENT_REQUIRED_DOMAINS[_llm_classification.intent]
            capabilities = await self.home_voice_resolver.capabilities.snapshot()
            present = capabilities.domain_present(required_domain, self.home_voice_resolver.capabilities.now())
            result["capability_observation"] = capabilities.observation.to_dict(self.home_voice_resolver.capabilities.now())
            if present is not True:
                msg = ("No puedo comprobar ahora la disponibilidad de ese dispositivo." if present is None else _UNSUPPORTED_INTENT_MESSAGES.get(
                    _llm_classification.intent,
                    "Eso no lo manejo todavía.",
                ))
                result["_task_outcome"] = TaskOutcome(TaskStatus.UNCERTAIN if present is None else TaskStatus.UNSUPPORTED,
                    dispatch=DispatchState.NOT_SENT, effect=EffectState.NOT_OBSERVED)
                logger.info(
                    f"[INTENT_REJECT] intent={_llm_classification.intent} "
                    f"required_domain={required_domain} no presente en HA — {msg!r}"
                )
                result["intent"] = _llm_classification.intent
                result["response"] = msg
                result["success"] = False
                await self._speak_response(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # 8. NLU pre-filter (intent léxico + slots) + Vector DB search (with cache)
        from src.nlu import classify_intent, extract_slots

        intent = classify_intent(text)            # turn_on | turn_off | None
        query_slots = extract_slots(text)         # {brightness_pct, rgb_color, ...}
        ambiguous_entity = None                   # Para el caso Q6 (sin verbo, con entity clara)

        # Caso ambiguo (Q6 C+B): la query no tiene verbo reconocible.
        # Estrategia: 1) Buscar por entity sin filtro de service, 2) mirar LastActionTracker,
        # 3) toggle si hay contexto reciente, si no → TTS "¿prendo o apago?".
        if intent is None:
            probe = await self._query_work.run(self.chroma.search_command,
                text, threshold=0.55, service_filter=None, query_slots={},
            )
            if probe and probe["similarity"] >= 0.60:
                last = self.action_tracker.get_recent(probe["entity_id"])
                if last is not None:
                    intent = self.action_tracker.toggle_service(last.service)
                    logger.info(
                        f"Query ambigua sobre {probe['entity_id']} — toggle implícito "
                        f"(last={last.service}, ahora={intent})"
                    )
                else:
                    ambiguous_entity = probe

        if ambiguous_entity is not None:
            # Preguntar al usuario (B fallback)
            area = ambiguous_entity.get("value_label") or ambiguous_entity["entity_id"].split(".")[-1]
            friendly = area.replace("_", " ")
            result["intent"] = "clarification_requested"
            result["response"] = f"¿Prendo o apago la luz del {friendly}?"
            result["success"] = False
            await self._speak_response(result["response"])
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        if intent:
            logger.debug(f"NLU: intent={intent} slots={query_slots}")

        t_search = time.perf_counter()
        cache_key = f"{intent or 'any'}::{text.lower().strip()}"
        command = self._query_cache.get(cache_key)

        if not command:
            command = await self._query_work.run(self.chroma.search_command,
                text,
                self.vector_search_threshold,
                service_filter=intent,
                query_slots=query_slots,
            )
            if command:
                self._add_to_cache(cache_key, command)

        result["timings"]["vector_search"] = (time.perf_counter() - t_search) * 1000

        if self._closed or not self._current_source_available():
            result.update(intent="cancelled", success=False, response="")
            return result

        if command:
            result["intent"] = "domotics"
            result["action"] = command
            logger.info(f"[Vector] {command['description']} (sim={command['similarity']:.2f})")

            # Household controls do not derive authority from a voice profile.
            t_ha = time.perf_counter()
            action_result = await self._voice_actions.execute_with_feedback({
                **command, "zone_id": f"zone_{room_id}" if room_id else None,
                "user_id": user_id, "user_name": user.name if user else None,
            }, request_id=CURRENT_REQUEST_ID.get())
            self._record_home_action(action_result)
            receipt = action_result.receipt
            result["receipt"] = asdict(receipt)
            success = receipt.status == "confirmed"
            result["timings"]["home_assistant"] = (time.perf_counter() - t_ha) * 1000

            result["success"] = success
            result["response"] = command["description"] if success else "No pude hacerlo"

            # Registrar en LastActionTracker para toggle implícito en la próxima query ambigua
            if success:
                self.action_tracker.record(
                    command["entity_id"], command["service"], command.get("data"),
                )

            # Speak response with emotion adjustments
            emotion_adj = emotion.response_adjustment if emotion else None
            if not success and not action_result.feedback_attempted:
                await self._speak_response(result["response"], emotion_adjustment=emotion_adj,
                                           room_context=room_context)

            # Log event
            if self.event_logger and success:
                await self._log_domotics_event(
                    entity_id=command["entity_id"],
                    action=command["service"],
                    trigger_phrase=text
                )

            # Suggestions
            if self.suggestion_engine:
                self._command_count += 1
                if self._command_count >= self.suggestion_interval:
                    self._command_count = 0
                    await self._maybe_present_suggestion()

            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 9. Conversation with LLM/Router
        result["intent"] = "conversation"

        use_deep = True
        router_response = ""

        if self.router:
            try:
                t_router = time.perf_counter()
                # Classify and respond in ONE inference (saves ~20ms)
                recent_context = await self._get_conversation_context()
                def infer_router():
                    if self._closed or not self._current_source_available():
                        raise SourcesInvalidated("prompt_sources_invalidated")
                    response = self.router.classify_and_respond(text, context=recent_context, max_tokens=256)
                    if self._closed or not self._current_source_available():
                        response = None
                        raise SourcesInvalidated("prompt_sources_invalidated")
                    return response
                use_deep, router_response = await run_private(self._query_work.run(infer_router), self._current_source_available)
                result["timings"]["router"] = (time.perf_counter() - t_router) * 1000
            except Exception as e:
                logger.warning(f"Router failed: {e}")

        if self._closed or not self._current_source_available():
            result.update(intent="cancelled", success=False, response="")
            return result

        if not use_deep and router_response:
            # Router already generated the response, no second inference needed
            result["response"] = router_response
            result["success"] = True
            emotion_adj = emotion.response_adjustment if emotion else None
            try:
                delivery = await run_private(self._speak_response(
                    result["response"], emotion_adjustment=emotion_adj, room_context=room_context,
                ), self._current_source_available)
            except SourcesInvalidated:
                result.update(intent="cancelled", success=False, response="")
                return result
            result["success"] = delivery.success
        else:
            prompt = await self._build_prompt(text)
            emotion_adj = emotion.response_adjustment if emotion else None
            try:
                delivery = await run_private(self.response_handler.arespond(
                    prompt,
                    zone_id=f"zone_{room_context.room_id}" if room_context else (f"zone_{room_id}" if room_id else None),
                    generation=GuardedGeneration(self.response_handler._generation, self._current_source_available),
                    max_tokens=512,
                    temperature=0.7,
                    emotion_adjustment=emotion_adj
                ), self._current_source_available)
            except SourcesInvalidated:
                result.update(intent="cancelled", success=False, response="")
                return result
            result["timings"]["llm_stream"] = (time.perf_counter() - pipeline_start) * 1000
            self._record_delivery(delivery)
            result["response"] = delivery.text.strip()
            result["success"] = delivery.success
            result["voice_delivery"] = delivery.delivery.value
            if delivery.timeout_stage is not None:
                result["timeout_stage"] = delivery.timeout_stage.value
                result["intent"] = "timeout"

        result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
        return result

    async def _earcon_and_log(
        self, reason: str, text: str, signals: dict,
        room_id, wake_score: float, wake_rms: float,
        outcome: str = "gate_rejected",
    ) -> None:
        """Suena el earcon si el reject es 'humano plausible' + loguea el outcome."""
        if should_play_earcon(reason, wake_score, wake_rms, self._earcon_cfg):
            _rc = None
            if self.room_context_manager and room_id:
                _rc = self.room_context_manager.resolve_room(
                    mic_zone_id=room_id, user_id=None,
                )
            await self.response_handler.aplay_earcon(
                zone_id=f"zone_{room_id}" if room_id and _rc is None else None, room_context=_rc,
            )
            await alog_asr_outcome(
                self._asr_event_logger, room_id, "earcon_fired",
                reason, text, self._asr_signals(signals), wake_score, wake_rms,
                sources=(self._memory_source.get(),) if self._memory_source.get() is not None else (),
            )
        await alog_asr_outcome(
            self._asr_event_logger, room_id, outcome,
            reason, text, self._asr_signals(signals), wake_score, wake_rms,
            sources=(self._memory_source.get(),) if self._memory_source.get() is not None else (),
        )

    def _build_confirmation_question(self, pc: PartialCommand) -> str:
        """
        Construye la pregunta en español pidiendo confirmación de un comando
        sensible con baja confidence.

        Args:
            pc: PartialCommand con al menos intent y entity.

        Returns:
            Pregunta natural tipo "¿Querés apagar el aire del cuarto? Decí sí o no."
        """
        action_verb = {
            "turn_off": "apagar",
            "turn_on": "prender",
            "set_cover_position": "mover",
        }.get(pc.intent or "", "hacer eso con")

        entity_name = {
            "light": "la luz",
            "climate": "el aire",
            "cover": "las persianas",
            "fan": "el ventilador",
            "media_player": "la música",
        }.get(pc.entity or "", "el dispositivo")

        if pc.room:
            room_name = {
                "escritorio": "del escritorio",
                "living": "del living",
                "cocina": "de la cocina",
                "bano": "del baño",
                "hall": "del hall",
                "cuarto": "del cuarto",
            }.get(pc.room, "")
            if room_name:
                return f"¿Querés {action_verb} {entity_name} {room_name}? Decí sí o no."
        return f"¿Querés {action_verb} {entity_name}? Decí sí o no."

    def _check_confidence_gate(self, text: str, result: dict, speak: bool = True) -> PartialCommand | None:
        """
        Chequea la confidence del comando y decide si hay que pedir confirmación.

        Si confidence < threshold y el combo (intent, entity) es sensible,
        habla la pregunta y marca `result["pending_confirmation"] = True`.
        Si es reversible, sólo logea la incertidumbre y deja seguir.

        Args:
            text: Transcripción de la query del usuario.
            result: Dict de resultado del pipeline (mutado in-place si hay
                confirmación pendiente).

        Returns:
            PartialCommand parseado, o None si el texto no produjo nada útil.
            Si `result["pending_confirmation"]` == True, el caller debe
            retornar inmediatamente sin dispatchar.
        """
        pc = parse_partial_command(text)
        # Si el parser no sacó intent+entity, no aplica gate — que siga el
        # pipeline normal (vector search, LLM, etc.) decidiendo.
        if pc.intent is None or pc.entity is None:
            return pc

        if pc.confidence >= self.confidence_threshold:
            return pc

        if is_sensitive(pc.intent, pc.entity):
            question = self._build_confirmation_question(pc)
            logger.info(
                f"Low confidence ({pc.confidence:.2f}) + sensitive "
                f"(intent={pc.intent}, entity={pc.entity}) — pidiendo confirmación"
            )
            # TODO(S4-followup): integrar con FollowUpMode para escuchar sí/no
            # por 5s. Por ahora sólo hablamos la pregunta y NO dispatchamos el
            # comando — el usuario puede repetirlo con más contexto.
            if speak:
                self.response_handler.speak(question)
            result["intent"] = "pending_confirmation"
            result["response"] = question
            result["success"] = False
            result["pending_confirmation"] = True
            result["pending_pc"] = pc
            return pc

        # Reversible (ej: light turn_on): ejecutar igual pero dejar traza.
        logger.warning(
            f"Low confidence ({pc.confidence:.2f}) on reversible command "
            f"(intent={pc.intent}, entity={pc.entity}) — ejecutando igual"
        )
        return pc

    async def _check_feedback(self, text: str) -> dict:
        """Mark a scoped record and acknowledge only an actual saved receipt."""
        from src.training.conversation_collector import ConversationCollector
        collector = self.conversation_collector
        managed = isinstance(collector, ConversationCollector)
        if collector is None or not managed and not self._last_response:
            return {"is_feedback": False, "response": ""}
        quality = correction = None
        text_lower = text.lower().strip()
        if any(phrase in text_lower for phrase in self.FEEDBACK_GOOD):
            quality = "good"
        elif any(phrase in text_lower for phrase in self.FEEDBACK_BAD):
            quality = "bad"
        else:
            for prefix in self.FEEDBACK_CORRECTION_PREFIX:
                if text_lower.startswith(prefix):
                    correction = text[len(prefix):].strip()
                    if correction:
                        quality = "corrected"
                    break
        if quality is None:
            return {"is_feedback": False, "response": ""}
        try:
            if managed:
                source = self._memory_source.get()
                saved = await collector.mark_last_response(quality, correction,
                    sources=(source,) if source is not None else ())
            else:
                saved = collector.mark_last_response(quality, correction) if correction is not None else collector.mark_last_response(quality)
            if not saved:
                return {"is_feedback": True, "success": False,
                        "response": "No encuentro una respuesta vigente para marcar en esta conversación."}
            response = {"good": "Gracias, marqué esa respuesta como buena.",
                        "bad": "Entendido, marqué esa respuesta para revisar y mejorar.",
                        "corrected": f"Entendido, guardé la corrección: {correction}"}[quality]
            return {"is_feedback": True, "success": True, "response": response}
        finally:
            text = correction = None

    # Whole-utterance matches only (accents and punctuation ignored): a phrase
    # that merely contains "actualiza" or "sync" never rebuilds the index.
    SYNC_PHRASES = frozenset({
        "sincroniza los comandos", "sincronizar los comandos",
        "actualiza los comandos", "actualizar los comandos",
    })

    def _is_sync_command(self, text: str) -> bool:
        """Detect the explicit command-index sync request."""
        decomposed = unicodedata.normalize("NFKD", (text or "").lower())
        plain = "".join(c for c in decomposed if not unicodedata.combining(c))
        words = "".join(c if c.isalnum() else " " for c in plain).split()
        if words[:1] == ["nexa"]:
            words = words[1:]
        return " ".join(words) in self.SYNC_PHRASES

    def _may_sync_commands(self, user) -> bool:
        """The sync_commands action permission (ADULT) for an identified user."""
        if user is None or self.user_manager is None:
            return False
        return self.user_manager.check_action_permission(user, "sync_commands").allowed

    async def _run_command_sync(self, result: dict) -> None:
        """Run the guarded index rebuild and record what it left persisted."""
        try:
            # U3b-0d2: HA reads and LLM calls await on the loop; BGE-M3 and the
            # Chroma staging writes run in ChromaSync's owned worker thread.
            count = await self.chroma.sync_commands(self.ha, self.llm)
        except Exception as error:
            code = getattr(error, "code", type(error).__name__)
            logger.warning("[Sync] command sync failed: %s", code)
            # A sync switches the index only when complete: on failure the
            # previous index is kept, unless the switch itself was uncertain.
            kept = getattr(error, "index_kept", False)
            persistence = PersistenceState.NOT_WRITTEN if kept else PersistenceState.UNKNOWN
            if code == "chroma_sync_already_running":
                result["response"] = "Ya estoy sincronizando los comandos."
            elif code in ("chroma_sync_aborted", "chroma_sync_closed"):
                result["response"] = "Cancelé la sincronización; los comandos anteriores siguen activos."
                result["_task_outcome"] = TaskOutcome(TaskStatus.CANCELLED, persistence=persistence)
                return
            else:
                result["response"] = ("No pude sincronizar los comandos; los anteriores siguen activos."
                                      if kept else "No pude sincronizar los comandos")
            result["_task_outcome"] = TaskOutcome(TaskStatus.FAILED, persistence=persistence)
        else:
            result["response"] = f"Listo, actualicé {count} comandos"
            result["success"] = True
            result["_task_outcome"] = TaskOutcome(TaskStatus.COMPLETED, persistence=PersistenceState.COMMITTED,
                effect=EffectState.OBSERVED, observation_scope="software_state")

    def _is_enrollment_command(self, text: str) -> bool:
        """Detect voice enrollment command."""
        enrollment_triggers = [
            "agregar persona", "agregar usuario", "nueva persona",
            "nuevo usuario", "registrar persona", "registrar usuario",
            "anadir persona", "add user", "add person"
        ]
        return any(trigger in text.lower() for trigger in enrollment_triggers)

    async def _call_suggestion(self, function, *args, **kwargs):
        """Keep serialized analysis/file I/O in an owned native scope."""
        from src.analytics.event_logger import EventLogger

        def call():
            with self._suggestion_lock:
                return function(*args, **kwargs)

        events = getattr(self.suggestion_engine, "event_logger", None)
        if isinstance(events, EventLogger):
            return await events.acall(call)
        return await self._query_work.run(call)

    async def _run_suggestion_flow(self, operation, cancelled_result):
        """Own the native-read to voice handoff without cancelling the parent loop."""
        if self._suggestions_closed or len(self._suggestion_flows) >= 8:
            operation.close()
            return cancelled_result
        task = asyncio.create_task(operation)
        self._suggestion_flows.add(task)

        def finished(done):
            self._suggestion_flows.discard(done)
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # A cleanup driver cancels only this child scope. User/parent
            # cancellation still propagates after the child's actual cleanup.
            parent_cancelled = bool(asyncio.current_task().cancelling())
            if not task.done() and not task.cancelling():
                task.cancel()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    parent_cancelled |= bool(asyncio.current_task().cancelling())
                except Exception:
                    # Ownership remains with the response/native backends;
                    # cleanup failure cannot replace the parent's cancellation.
                    break
            if parent_cancelled:
                raise
            return cancelled_result
        finally:
            task, operation, cancelled_result = None, None, None

    def snapshot_suggestion_flows(self) -> tuple[asyncio.Task, ...]:
        """Return actual admitted async scopes for an on-loop erasure barrier."""
        return tuple(self._suggestion_flows)

    def begin_close_suggestions(self) -> asyncio.Task:
        """Close only proposal admission and retain cancelled flow cleanup."""
        self._suggestions_closed = True
        if self._suggestion_closing is None:
            tasks = self.snapshot_suggestion_flows()
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            async def finish():
                await asyncio.gather(*tasks, return_exceptions=True)
                self._pending_suggestion = None
            self._suggestion_closing = asyncio.create_task(finish())
        return self._suggestion_closing

    async def _handle_suggestion_response(self, text: str) -> dict:
        return await self._run_suggestion_flow(self._handle_suggestion_response_inner(text), {"handled": False})

    async def _handle_suggestion_response_inner(self, text: str) -> dict:
        """Handle user response to a pending automation suggestion."""
        pending = self._pending_suggestion
        if pending is None or not self.suggestion_engine:
            return {"handled": False}

        text_lower = text.lower().strip()
        suggestion_id = pending.id

        for phrase in self.SUGGESTION_ACCEPT:
            if phrase in text_lower or text_lower == phrase:
                result = await self._record_suggestion_response(suggestion_id, accept=True)
                self._pending_suggestion = None
                return {
                    "handled": True,
                    "success": result.get("success", True),
                    "response": result.get("message") or result.get("error") or "Registré tu respuesta.",
                }

        for phrase in self.SUGGESTION_REJECT:
            if phrase in text_lower or text_lower == phrase:
                result = await self._record_suggestion_response(suggestion_id, accept=False)
                self._pending_suggestion = None
                return {
                    "handled": True,
                    "success": result.get("success", True),
                    "response": result.get("message") or result.get("error") or "Registré tu respuesta.",
                }

        for phrase in self.SUGGESTION_SNOOZE:
            if phrase in text_lower:
                result = await self._record_suggestion_response(suggestion_id, accept=False, snooze_hours=24)
                self._pending_suggestion = None
                return {
                    "handled": True,
                    "success": result.get("success", True),
                    "response": result.get("message") or result.get("error") or "Registré tu respuesta.",
                }

        self._pending_suggestion = None
        return {"handled": False}

    async def _record_suggestion_response(self, suggestion_id: str, **kwargs) -> dict:
        try:
            return await self._call_suggestion(self.suggestion_engine.respond_to_suggestion, suggestion_id, **kwargs)
        except Exception as error:
            logger.warning("Suggestion response deferred: %s", type(error).__name__)
            return {"success": False, "error": "No pude registrar tu respuesta a la sugerencia."}

    async def _maybe_present_suggestion(self):
        """Present an automation suggestion if one is available."""
        return await self._run_suggestion_flow(self._present_suggestion_inner(), None)

    async def _present_suggestion_inner(self):
        if not self.suggestion_engine:
            return

        try:
            suggestion = await self._call_suggestion(self.suggestion_engine.get_suggestion_to_present)
        except Exception as error:
            logger.warning("Suggestion presentation deferred: %s", type(error).__name__)
            self._pending_suggestion = None
            return
        if suggestion:
            self._pending_suggestion = suggestion
            message = suggestion.message + " Responde si, no, o despues."
            from src.analytics.suggestion_engine import SuggestionEngine
            if isinstance(self.suggestion_engine, SuggestionEngine):
                try:
                    await run_private(self._speak_response(message),
                        lambda: not self._closed and self.suggestion_engine.available_now(suggestion))
                except SourcesInvalidated:
                    self._pending_suggestion = None
            else:
                await self._speak_response(message)

    @property
    def _pending_suggestion(self):
        from src.analytics.suggestion_engine import SuggestionEngine
        value = getattr(self, "_pending_proposal", None)
        if value is not None and isinstance(self.suggestion_engine, SuggestionEngine) and not self.suggestion_engine.available_now(value):
            self._pending_proposal = None
            return None
        return value

    @_pending_suggestion.setter
    def _pending_suggestion(self, value):
        self._pending_proposal = value

    async def _log_domotics_event(self, entity_id: str, action: str, trigger_phrase: str = None):
        """Log a domotics event."""
        if not self.event_logger:
            return

        from src.analytics.event_logger import EventType
        user = None
        arguments = dict(entity_id=entity_id, action=action, event_type=EventType.COMMAND,
            user_id=user.user_id if user else None, user_name=user.name if user else None,
            trigger_phrase=trigger_phrase)
        try:
            from src.analytics.event_logger import EventLogger
            if isinstance(self.event_logger, EventLogger):
                source = self._memory_source.get()
                await self.event_logger.alog(**arguments, sources=(source,) if source is not None else ())
            else:
                await asyncio.to_thread(self.event_logger.log, **arguments)
        except Exception as error:
            logger.debug("Domotics metric write deferred: %s", type(error).__name__)

    def _log_latency(self, result: dict):
        """Log latency metrics."""
        from contextvars import Context
        from src.obs.metrics_emitter import diagnostic_timings, project_diagnostic
        timings = diagnostic_timings(result.get('timings'))
        safe = project_diagnostic({'event_type': 'request', 'total_ms': result.get('latency_ms'),
            'intent': result.get('intent'), 'path': result.get('path')})
        total = safe.get('total_ms', 0.)
        target = self.latency_target_ms
        status = "OK" if total <= target else "SLOW"
        timings_str = " + ".join(
            f"{k}={v:.0f}" for k, v in timings.items()
        )
        Context().run(logger.info, "[%s] Total: %.0fms (target: %sms) [%s]", status, total, target, timings_str)

        if self.latency_monitor:
            self.latency_monitor.record(
                timings=timings,
                user=None,
                intent=safe.get('intent')
            )

        if self.metrics_emitter:
            turn = self._turn_outcome.get()
            meta = turn.capture_meta if turn is not None else (getattr(self, "_last_request_meta", {}) or {})
            timings["total"] = total
            try:
                self.metrics_emitter.emit_request(
                    user_id='', zone_id=None, text='',
                    intent=safe.get('intent'),
                    path=safe.get('path'),
                    success=bool(result.get("success")),
                    timings=timings,
                    audio_duration_ms=meta.get("audio_duration_ms"),
                    entity_id=None, service=None,
                    used_wake_text=bool(meta.get("used_wake_text")),
                    early_dispatch=bool(meta.get("early_dispatch")),
                    sources=self._active_sources(),
                    outcome=project_turn_outcome(result),
                )
            except Exception:
                Context().run(logger.warning, "Diagnostic request metric deferred")

    async def _get_conversation_context(self) -> str:
        """Microphone provenance does not authorize reading personal conversation history."""
        return ""

    async def _build_prompt(self, user_text: str) -> str:
        """Reason over the current microphone turn without personal or unscoped cached data."""
        # P3-F1: household wall time, not the process's local time.
        # T037-U3-R3: fixed Spanish weekday, never the process locale's %A.
        now = self.household_clock.now_local()
        time_context = f"{now:%H:%M} del {WEEKDAYS[now.weekday()]}"
        # Explicit household state queries use the current typed capability path.
        # A raw HA cache and global memory are not authority for a generic prompt.
        return f"""Eres un asistente de hogar inteligente. Responde de forma concisa y natural en español.
Hora actual: {time_context}
Puedes ayudar con:
- Consultas sobre el hogar y dispositivos cuando hay datos actuales disponibles
- Preguntas generales
- Conversacion casual

Se breve y util.

Usuario: {user_text}

Asistente:"""

    def _add_to_cache(self, key: str, value: dict):
        """Retain a bounded query result only while its actual capture survives."""
        return self._query_cache.put(key, value, self._active_sources())

    async def _classify_command(self, text: str, *, room_id: str | None = None):
        options = {}
        if isinstance(self.llm_command_router, LLMCommandRouter):
            source = self._memory_source.get()
            options["sources"] = (source,) if source is not None else ()
        return await self.llm_command_router.classify(text, room_id=room_id, **options)

    def _active_sources(self):
        """Use derived lineage once selected; original capture remains the fallback."""
        sources = self._response_sources.get()
        source = self._memory_source.get()
        return sources or ((source,) if source is not None else ())

    def _current_source_available(self) -> bool:
        from src.memory.fact_ledger import sources_authorized
        sources = self._active_sources()
        return not self._closed and (
            self._privacy.sources_available(sources) if self._privacy is not None
            else sources_authorized(sources, self._query_cache.clock(), self._source_policy))

    def prune_private(self) -> None:
        """Drop idle query/history copies; every read enforces the same policy."""
        from src.pipeline.command_processor import CommandProcessor
        if isinstance(self.command_processor, CommandProcessor):
            self.command_processor.prune_private()
        self._query_cache.prune()
        self._pending_suggestion  # Resolve expiry/revocation on idle proposals too.
        if isinstance(self.llm_command_router, LLMCommandRouter):
            self.llm_command_router.prune_private()
        if self._orchestrator is not None:
            queue = getattr(self._orchestrator, "queue", None)
            if queue is not None and hasattr(queue, "prune_private"):
                queue.prune_private()

    def start(self) -> None:
        """Own one cheap in-memory maintenance task for idle text copies."""
        if self._closed:
            raise RuntimeError("request_router_closed")
        if self._private_maintenance_task is None:
            async def maintain():
                while True:
                    self.prune_private()
                    await asyncio.sleep(1)
            self._private_maintenance_task = asyncio.create_task(maintain())

    async def aclose(self) -> None:
        """Stop maintenance and discard private cached text before classifier close."""
        self._closed = True
        suggestions = self.begin_close_suggestions()
        self._pending_suggestion = None
        self._query_cache.close()
        if self._private_maintenance_task is not None:
            self._private_maintenance_task.cancel()
            await asyncio.gather(self._private_maintenance_task, return_exceptions=True)
        try:
            if isinstance(self.llm_command_router, LLMCommandRouter):
                await self.llm_command_router.aclose()
        finally:
            await asyncio.shield(suggestions)
            await self._query_work.aclose()
            await self.home_voice_resolver.aclose()

    def resource_status(self) -> dict:
        """Expose retained query/classifier work and private cache admission."""
        return {"closed": self._closed, "queries": self._query_work.status(),
                "classifier": self.llm_command_router.resource_status() if isinstance(self.llm_command_router, LLMCommandRouter) else None}
