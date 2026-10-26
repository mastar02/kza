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
import time
from dataclasses import asdict

import numpy as np

from src.home_assistant.home_actions import HomeActionRequest, HomeActionService
from src.home_assistant.operations import compile_effect, finite_number
from src.home_assistant.voice_actions import VoiceHomeActionExecutor
from src.home_assistant.voice_resolver import HomeVoiceResolver
from src.nlu.home_command import parse_home_command
from src.nlu.command_gate import CommandAcceptanceGate
from src.nlu.command_grammar import PartialCommand, has_unresolved_light_location, parse_default_light_target, parse_partial_command
from src.nlu.sensitive_actions import is_sensitive
from src.orchestrator import PathType
from src.pipeline.earcon_gate import should_play_earcon
from src.analytics.asr_quality import alog_asr_outcome

logger = logging.getLogger(__name__)


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
        """
        # Core pipeline components
        self.command_processor = command_processor
        self.response_handler = response_handler
        self.audio_manager = audio_manager
        self.room_context_manager = room_context_manager

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

        # Command acceptance gate (consolidates _is_noise_text + STT confidence).
        self.command_gate = command_gate or CommandAcceptanceGate(
            wake_words=self._wake_words
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

        # Plan #3 OpenClaw — plugin hooks registry (or None)
        self._hooks = hooks
        self._voice_actions = VoiceHomeActionExecutor(
            ha_client, self.home_actions, response_handler=response_handler, hooks=hooks,
            before_handler_warn_ms=before_handler_warn_ms,
            unavailable_precheck_enabled=unavailable_precheck_enabled,
        )

        # State
        self._query_cache = {}
        self._cache_max_size = cache_max_size
        self._pending_suggestion = None
        self._last_response = None
        self._command_count = 0
        self._response_zone = ContextVar("response_zone", default=None)
        self._memory_source = ContextVar("memory_source", default=None)

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

    async def _speak_response(self, text: str, **kwargs) -> None:
        """Serialize synthesis/playback in a worker while capture keeps running."""
        kwargs.setdefault("zone_id", self._response_zone.get())
        return await self.response_handler.aspeak(text, **kwargs)

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
                          response="No pude preparar ese recuerdo para revisión.")
        else:
            try:
                proposal = await self.memory.long_term.acall(self.memory.long_term.propose, match[1].strip(), (source,))
                result.update(intent="memory_proposal", success=True, memory_proposal=proposal,
                              memory_review_required=True, response="Dejé la propuesta de recuerdo para que la revises en el panel.")
            except Exception:
                logger.exception("[Memory] Proposal could not be retained")
                result.update(intent="memory_unavailable", success=False,
                              response="No pude preparar ese recuerdo para revisión.")
        delivered = await self._speak_response(result["response"])
        result["voice_delivery"] = delivered.delivery.value
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
                    classification = await self.llm_command_router.classify(text, room_id=room_id)
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
        reason = command.reason
        target = None
        if reason is None:
            try:
                target = await self.home_voice_resolver.resolve(command, room_id=room_id)
                reason = target.reason
            except Exception:
                logger.exception("[HomeVoice] Target resolution unavailable")
                reason = "selection_unavailable"
        if reason is not None or target is None or target.entity_id is None:
            if command.reason is not None:
                question = {
                    "climate": "¿A qué temperatura en grados Celsius o en qué modo querés poner el aire?",
                    "cover": "¿Querés abrir, cerrar, detener o poner la persiana en un porcentaje concreto?",
                    "light": "¿Qué temperatura de luz en kelvin querés usar?",
                    "scene": "¿Qué escena querés activar?",
                }[command.domain]
            else:
                question = "¿Qué escena querés activar? Necesito su nombre exacto entre las escenas seleccionadas." if command.domain == "scene" else "¿En qué habitación? Necesito un dispositivo seleccionado para esa habitación."
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
            receipt = await self._voice_actions.execute(action)
            result["timings"]["home_assistant"] = (time.perf_counter() - started) * 1000
            result["receipt"] = asdict(receipt)
            result["success"] = receipt.status == "confirmed"
        result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
        return True

    async def process_command(self, audio_or_event) -> dict:
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

        pretranscribed_text: str | None = None
        used_wake_text = False
        early_dispatch = False
        ambient_strict = False
        if isinstance(audio_or_event, CommandEvent):
            audio = audio_or_event.audio
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
                used_wake_text = transcript.source is TranscriptSource.WAKE
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

        wake_score = getattr(audio_or_event, "wake_score", 1.0)
        textual_wake = bool(getattr(audio_or_event, "textual_wake", False))
        wake_rms = (
            float(np.sqrt(np.mean(np.square(audio))))
            if audio is not None and len(audio) else 0.0
        )

        from src.memory.memory_manager import MemoryManager
        from src.memory.fact_ledger import MemoryOrigin
        from uuid import uuid4
        self._memory_source.set(None)
        try:
            captured_at = getattr(audio_or_event, "captured_at", None)
            if captured_at is None:
                captured_at = getattr(audio_or_event, "timestamp", time.time())
            capture_id = getattr(audio_or_event, "capture_id", str(uuid4()))
            conversation_id = getattr(audio_or_event, "conversation_id", None)
            expires_at = getattr(audio_or_event, "source_expires_at", None)
            origin = MemoryOrigin.AMBIENT if textual_wake else MemoryOrigin.VOICE
            if isinstance(self.memory, MemoryManager):
                source = self.memory.source_for_capture(capture_id, captured_at, room_id,
                    conversation_id=conversation_id, expires_at=expires_at, origin=origin)
            else:
                from src.memory.fact_ledger import MemorySource
                source = MemorySource(capture_id, conversation_id or str(uuid4()), captured_at,
                    expires_at if expires_at is not None else min(captured_at, time.time())+48*3600, origin)
            self._memory_source.set(source)
        except (ValueError, TypeError):
            logger.warning("[Memory] Capture provenance unavailable")

        if self.orchestrator_enabled and self._orchestrator:
            result = await self._process_command_orchestrated(
                audio, room_id=room_id, pretranscribed_text=pretranscribed_text,
                ambient_strict=ambient_strict,
                wake_score=wake_score, wake_rms=wake_rms,
                textual_wake=textual_wake,
            )
        else:
            result = await self._process_command_legacy(
                audio, room_id=room_id, pretranscribed_text=pretranscribed_text,
                wake_score=wake_score, wake_rms=wake_rms,
                textual_wake=textual_wake, ambient_strict=ambient_strict,
            )
        if self.memory is not None and result.get("success") and result.get("intent") in {"conversation", "domotics"}:
            try:
                source_options = {}
                if isinstance(self.memory, MemoryManager):
                    source = self._memory_source.get()
                    if source is None:
                        return result
                    source_options["sources"] = (source,)
                    result["capture_id"] = source.capture_id
                    result["conversation_id"] = source.conversation_id
                await self._memory_call("record_interaction", user_input=result.get("text", ""),
                    assistant_response=result.get("response", ""), intent=result["intent"], **source_options)
            except Exception:
                logger.exception("[Memory] Completed interaction could not be recorded")
        return result

    async def _process_command_orchestrated(self, audio: np.ndarray, room_id: str = None,
                                              pretranscribed_text: str | None = None,
                                              ambient_strict: bool = False,
                                              wake_score: float = 1.0,
                                              wake_rms: float = 0.0,
                                              textual_wake: bool = False) -> dict:
        """Process command with multi-user orchestrator."""
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
        cmd = await self.command_processor.process_command(
            audio, use_parallel=True, pretranscribed_text=pretranscribed_text,
        )
        text = cmd.text
        result["text"] = text
        result["timings"].update(cmd.timings)

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

        # 1a. Command acceptance gate: corta antes del orchestrator si el texto
        #     es claramente ruido (TV, eco del TTS, repetición) o confianza baja.
        #     Evita que esas muestras vayan al slow path LLM (caro + bloquea queue).
        gate_decision = self.command_gate.evaluate(
            text, stt_confidence=getattr(cmd, "stt_confidence", None)
        )
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
            gate_decision.signals or {}, wake_score, wake_rms,
            sources=(self._memory_source.get(),) if self._memory_source.get() is not None else (),
        )
        await self.response_handler.cancel(zone_id=f"zone_{room_id}" if room_id else None)

        if await self._try_memory_proposal(text, result):
            result["latency_ms"] = (time.perf_counter()-pipeline_start)*1000
            return result

        if await self._try_home_command(
            text, result, room_id=room_id, pipeline_start=pipeline_start,
            extra_gate=textual_wake or ambient_strict or wake_score < _GRAMMAR_BONUS_MIN_WAKE_SCORE,
        ):
            return result

        # 1a-grammar. Grammar fast-path: la gramática determinística manda en
        # domótica.  Si extrae intent+entity con alta confianza, construimos
        # una fast_classification y nos saltamos el LLMCommandRouter (poco
        # fiable en el modelo actual: rechaza comandos válidos como 'noise').
        fast_classification = None
        # Wake textual: el utterance YA contiene "nexa" (así matcheó). Grammar
        # lo marca full y ejecutaba luces fantasma; el 7B es el gate. El canal
        # textual no está en el presupuesto <300ms.
        if textual_wake:
            grammar_cls = None
            logger.info(
                "[GRAMMAR_FASTPATH] skip textual_wake — forcing LLM router: %r",
                text,
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
        elif self.llm_command_router is not None:
            t_llm = time.perf_counter()
            classification = await self.llm_command_router.classify(text, room_id=room_id)
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

        # 1b. Confidence gate — low-confidence sensitive commands pide
        #     confirmación antes de dispatchar al orchestrator.
        self._check_confidence_gate(text, result, speak=False)
        if result.get("pending_confirmation"):
            await self._speak_response(result["response"])
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 2. Get user info
        user = cmd.user
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
        async def on_response(dispatch_result):
            if dispatch_result.was_queued and dispatch_result.queue_position and not self.response_handler.is_responding:
                await self._speak_response(
                    dispatch_result.response,
                    zone_id=zone_id,
                    room_context=room_context,
                )

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
        if (
            fast_classification is not None
            and fast_classification.entity_hint == "light"
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
            on_response=on_response,
            service_filter=_service_filter,
            query_slots=_query_slots,
            **({"target_entity_id": target_entity_id} if target_entity_id else {}),
            memory_sources=(self._memory_source.get(),) if self._memory_source.get() is not None else (),
        )

        # 6. Build result
        result["intent"] = dispatch_result.intent
        result["response"] = dispatch_result.response
        result["success"] = dispatch_result.success
        result["action"] = dispatch_result.action
        result["path"] = dispatch_result.path.value if dispatch_result.path else None
        result["timings"].update(dispatch_result.timings)
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
            delivery = await self._speak_response(
                result["response"],
                zone_id=zone_id,
                emotion_adjustment=emotion_adj,
                room_context=room_context,
            )
            result["voice_delivery"] = delivery.delivery.value
            if dispatch_result.intent == "conversation" and not delivery.success:
                result["success"] = False

        # 8. Logging
        result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
        self._log_latency(result)

        if self.event_logger and result["success"] and result["intent"] == "domotics":
            await self._log_domotics_event(
                entity_id=result["action"]["entity_id"] if result["action"] else None,
                action=result["action"]["service"] if result["action"] else None,
                trigger_phrase=text
            )

        # Registrar el comando aceptado en el historial del LLMCommandRouter
        # para que próximas clasificaciones puedan detectar replays.
        if (
            self.llm_command_router is not None
            and result["success"]
            and result.get("llm_classification") is not None
        ):
            try:
                self.llm_command_router.record_command(
                    text=text,
                    intent=result["llm_classification"].intent or "unknown",
                )
            except Exception as e:
                logger.debug(f"LLMCommandRouter.record_command falló: {e}")

        return result

    async def _process_command_legacy(self, audio: np.ndarray, room_id: str = None,
                                        pretranscribed_text: str | None = None,
                                        wake_score: float = 1.0,
                                        wake_rms: float = 0.0,
                                        textual_wake: bool = False,
                                        ambient_strict: bool = False) -> dict:
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
        cmd = await self.command_processor.process_command(
            audio, use_parallel=True, pretranscribed_text=pretranscribed_text,
        )
        text = cmd.text
        result["text"] = text
        result["timings"].update(cmd.timings)

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

        # 1a. Command acceptance gate — mismo corto-circuito que el path orchestrated
        gate_decision = self.command_gate.evaluate(
            text, stt_confidence=getattr(cmd, "stt_confidence", None)
        )
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
            gate_decision.signals or {}, wake_score, wake_rms,
            sources=(self._memory_source.get(),) if self._memory_source.get() is not None else (),
        )
        await self.response_handler.cancel(zone_id=f"zone_{room_id}" if room_id else None)

        if await self._try_memory_proposal(text, result):
            result["latency_ms"] = (time.perf_counter()-pipeline_start)*1000
            return result

        if await self._try_home_command(
            text, result, room_id=room_id, pipeline_start=pipeline_start,
            extra_gate=textual_wake or ambient_strict or wake_score < _GRAMMAR_BONUS_MIN_WAKE_SCORE,
        ):
            return result

        user = cmd.user
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

        # 1b. Confidence gate — si confidence baja y combo sensible, pide
        #     confirmación en vez de ejecutar. Comandos sin intent/entity o
        #     reversibles con confidence baja siguen de largo (con log).
        self._check_confidence_gate(text, result, speak=False)
        if result.get("pending_confirmation"):
            await self._speak_response(result["response"])
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 2. Check feedback
        feedback_result = self._check_feedback(text)
        if feedback_result["is_feedback"]:
            result["intent"] = "feedback"
            result["response"] = feedback_result["response"]
            result["success"] = True
            await self._speak_response(result["response"])
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 3. Check pending suggestion
        if self._pending_suggestion:
            suggestion_result = self._handle_suggestion_response(text)
            if suggestion_result["handled"]:
                result["intent"] = "suggestion_response"
                result["response"] = suggestion_result["response"]
                result["success"] = True
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
                await self._speak_response(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # 6. Check sync command
        if self._is_sync_command(text):
            result["intent"] = "sync"
            await self._speak_response("Sincronizando comandos...")
            count = self.chroma.sync_commands(self.ha, self.llm)
            result["response"] = f"Listo, actualicé {count} comandos"
            result["success"] = True
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
                result["response"] = timer_result["response"]
                result["success"] = True
                await self._speak_response(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # 6c. Check intercom/announcement command (via FeatureManager)
        if self.features and self.features.intercom:
            intercom_result = self.features.intercom.handle_voice_command(
                text,
                user_id=user.user_id if user else None,
                source_zone=self.audio_manager.detect_source_zone(audio)
            )
            if intercom_result["handled"]:
                result["intent"] = "intercom"
                result["response"] = intercom_result["response"]
                result["success"] = True
                await self._speak_response(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # 6d. Check notification commands (via FeatureManager)
        if self.features and self.features.notifications:
            notif_result = self.features.notifications.handle_voice_command(
                text,
                user_id=user.user_id if user else None
            )
            if notif_result["handled"]:
                result["intent"] = "notification_config"
                result["response"] = notif_result["response"]
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
            if not self.ha.has_domain(required_domain):
                msg = _UNSUPPORTED_INTENT_MESSAGES.get(
                    _llm_classification.intent,
                    "Eso no lo manejo todavía.",
                )
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
            probe = await asyncio.to_thread(self.chroma.search_command,
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
            command = await asyncio.to_thread(self.chroma.search_command,
                text,
                self.vector_search_threshold,
                service_filter=intent,
                query_slots=query_slots,
            )
            if command:
                self._add_to_cache(cache_key, command)

        result["timings"]["vector_search"] = (time.perf_counter() - t_search) * 1000

        if command:
            result["intent"] = "domotics"
            result["action"] = command
            logger.info(f"[Vector] {command['description']} (sim={command['similarity']:.2f})")

            # Household controls do not derive authority from a voice profile.
            t_ha = time.perf_counter()
            receipt = await self._voice_actions.execute({
                **command, "zone_id": f"zone_{room_id}" if room_id else None,
                "user_id": user_id, "user_name": user.name if user else None,
            })
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
            await self._speak_response(
                result["response"],
                emotion_adjustment=emotion_adj,
                room_context=room_context,
            )

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
            self._log_latency(result)
            return result

        # 9. Conversation with LLM/Router
        result["intent"] = "conversation"

        use_deep = True
        router_response = ""

        if self.router:
            try:
                t_router = time.perf_counter()
                # Classify and respond in ONE inference (saves ~20ms)
                use_deep, router_response = await asyncio.to_thread(self.router.classify_and_respond,
                    text,
                    context=self._get_conversation_context(),
                    max_tokens=256
                )
                result["timings"]["router"] = (time.perf_counter() - t_router) * 1000
            except Exception as e:
                logger.warning(f"Router failed: {e}")

        if not use_deep and router_response:
            # Router already generated the response, no second inference needed
            result["response"] = router_response
            result["success"] = True
            emotion_adj = emotion.response_adjustment if emotion else None
            delivery = await self._speak_response(
                result["response"],
                emotion_adjustment=emotion_adj,
                room_context=room_context,
            )
            result["success"] = delivery.success
        else:
            prompt = await self._build_prompt(text)
            emotion_adj = emotion.response_adjustment if emotion else None
            delivery = await self.response_handler.arespond(
                prompt,
                zone_id=f"zone_{room_context.room_id}" if room_context else (f"zone_{room_id}" if room_id else None),
                max_tokens=512,
                temperature=0.7,
                emotion_adjustment=emotion_adj
            )
            result["timings"]["llm_stream"] = (time.perf_counter() - pipeline_start) * 1000
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
                reason, text, signals or {}, wake_score, wake_rms,
                sources=(self._memory_source.get(),) if self._memory_source.get() is not None else (),
            )
        await alog_asr_outcome(
            self._asr_event_logger, room_id, outcome,
            reason, text, signals or {}, wake_score, wake_rms,
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

    def _check_feedback(self, text: str) -> dict:
        """Detect if text is feedback about the previous response."""
        if not self.conversation_collector or not self._last_response:
            return {"is_feedback": False, "response": ""}

        text_lower = text.lower().strip()

        for phrase in self.FEEDBACK_GOOD:
            if phrase in text_lower:
                self.conversation_collector.mark_last_response("good")
                return {
                    "is_feedback": True,
                    "response": "Gracias, lo recordare para mejorar."
                }

        for phrase in self.FEEDBACK_BAD:
            if phrase in text_lower:
                self.conversation_collector.mark_last_response("bad")
                return {
                    "is_feedback": True,
                    "response": "Entendido, tratare de mejorar."
                }

        for prefix in self.FEEDBACK_CORRECTION_PREFIX:
            if text_lower.startswith(prefix):
                correction = text[len(prefix):].strip()
                if correction:
                    self.conversation_collector.mark_last_response("corrected", correction)
                    return {
                        "is_feedback": True,
                        "response": f"Entendido, deberia haber dicho: {correction}"
                    }

        return {"is_feedback": False, "response": ""}

    def _is_sync_command(self, text: str) -> bool:
        """Detect sync/update command."""
        sync_phrases = [
            "sincroniza", "sincronizar", "actualiza", "actualizar",
            "refresca", "refrescar", "sync", "update",
            "aprende los comandos", "actualiza los comandos"
        ]
        return any(phrase in text.lower() for phrase in sync_phrases)

    def _is_enrollment_command(self, text: str) -> bool:
        """Detect voice enrollment command."""
        enrollment_triggers = [
            "agregar persona", "agregar usuario", "nueva persona",
            "nuevo usuario", "registrar persona", "registrar usuario",
            "anadir persona", "add user", "add person"
        ]
        return any(trigger in text.lower() for trigger in enrollment_triggers)

    def _handle_suggestion_response(self, text: str) -> dict:
        """Handle user response to a pending automation suggestion."""
        if not self._pending_suggestion or not self.suggestion_engine:
            return {"handled": False}

        text_lower = text.lower().strip()
        suggestion_id = self._pending_suggestion.id

        for phrase in self.SUGGESTION_ACCEPT:
            if phrase in text_lower or text_lower == phrase:
                result = self.suggestion_engine.respond_to_suggestion(suggestion_id, accept=True)
                self._pending_suggestion = None
                return {
                    "handled": True,
                    "response": "Perfecto, automatizacion creada. " + result.get("message", "")
                }

        for phrase in self.SUGGESTION_REJECT:
            if phrase in text_lower or text_lower == phrase:
                self.suggestion_engine.respond_to_suggestion(suggestion_id, accept=False)
                self._pending_suggestion = None
                return {
                    "handled": True,
                    "response": "Entendido, no creare esa automatizacion."
                }

        for phrase in self.SUGGESTION_SNOOZE:
            if phrase in text_lower:
                self.suggestion_engine.respond_to_suggestion(
                    suggestion_id, accept=False, snooze_hours=24
                )
                self._pending_suggestion = None
                return {
                    "handled": True,
                    "response": "Ok, te lo recordare manana."
                }

        self._pending_suggestion = None
        return {"handled": False}

    async def _maybe_present_suggestion(self):
        """Present an automation suggestion if one is available."""
        if not self.suggestion_engine:
            return

        suggestion = self.suggestion_engine.get_suggestion_to_present()
        if suggestion:
            self._pending_suggestion = suggestion
            message = suggestion.message + " Responde si, no, o despues."
            await self._speak_response(message)

    async def _log_domotics_event(self, entity_id: str, action: str, trigger_phrase: str = None):
        """Log a domotics event."""
        if not self.event_logger:
            return

        from src.analytics.event_logger import EventType
        user = self.command_processor.get_current_user()
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
        total = result["latency_ms"]
        target = self.latency_target_ms
        status = "OK" if total <= target else "SLOW"
        timings_str = " + ".join(
            f"{k}={v:.0f}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in result["timings"].items()
        )
        logger.info(f"[{status}] Total: {total:.0f}ms (target: {target}ms) [{timings_str}]")

        if self.latency_monitor:
            user_name = result.get("user", {}).get("name") if result.get("user") else None
            self.latency_monitor.record(
                timings={
                    key: value for key, value in result["timings"].items()
                    if isinstance(value, (int, float))
                },
                user=user_name,
                intent=result.get("intent")
            )

        if self.metrics_emitter:
            meta = getattr(self, "_last_request_meta", {}) or {}
            action = result.get("action") or {}
            timings = dict(result.get("timings") or {})
            timings["total"] = total
            try:
                self.metrics_emitter.emit_request(
                    user_id=(result.get("user") or {}).get("name", "unknown"),
                    zone_id=(result.get("room") or {}).get("id") or None,
                    text=result.get("text", ""),
                    intent=result.get("intent"),
                    path=result.get("path"),
                    success=bool(result.get("success")),
                    timings=timings,
                    audio_duration_ms=meta.get("audio_duration_ms"),
                    entity_id=action.get("entity_id"),
                    service=action.get("service"),
                    used_wake_text=bool(meta.get("used_wake_text")),
                    early_dispatch=bool(meta.get("early_dispatch")),
                )
            except Exception as e:
                logger.warning(f"MetricsEmitter emit_request failed: {e}")

    def _get_conversation_context(self) -> str:
        """Get brief context for the router (optimized for speed)."""
        context_parts = []

        # Memory context (if available)
        if self.memory:
            try:
                recent = self.memory.get_recent_context(limit=3)
                if recent:
                    context_parts.append(f"Reciente: {recent}")
            except Exception as e:
                logger.debug(f"Failed to get memory context: {e}")

        return " | ".join(context_parts) if context_parts else ""

    async def _build_prompt(self, user_text: str) -> str:
        """Build prompt with context for LLM reasoning."""
        from datetime import datetime

        now = datetime.now()
        time_context = now.strftime("%H:%M del %A")

        memory_context = ""
        if self.memory:
            try:
                memory_context = await self._memory_call("format_context_for_prompt", user_text)
                if memory_context:
                    memory_context = f"\n{memory_context}\n"
            except Exception as e:
                logger.debug(f"Error getting memory context: {e}")

        home_context = ""
        try:
            important_domains = ["climate", "light", "binary_sensor"]
            entities = await self.ha.get_domotics_entities()

            relevant_states = []
            for entity in entities[:10]:
                domain = entity["entity_id"].split(".")[0]
                if domain in important_domains:
                    name = entity.get("attributes", {}).get("friendly_name", entity["entity_id"])
                    state = entity.get("state", "unknown")
                    relevant_states.append(f"- {name}: {state}")

            if relevant_states:
                home_context = "\nEstado del hogar:\n" + "\n".join(relevant_states[:5])
        except Exception as e:
            logger.debug(f"Failed to get HA entity states for prompt: {e}")

        system_prompt = f"""Eres un asistente de hogar inteligente. Responde de forma concisa y natural en español.
Hora actual: {time_context}{home_context}{memory_context}
Puedes ayudar con:
- Consultas sobre el hogar y dispositivos
- Preguntas generales
- Conversacion casual

Usa el contexto de memoria para dar respuestas personalizadas. Se breve y util."""

        return f"""{system_prompt}

Usuario: {user_text}

Asistente:"""

    def _add_to_cache(self, key: str, value: dict):
        """Add to cache with size limit."""
        if len(self._query_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        self._query_cache[key] = value
