"""
Request Router Module
Routes processed voice commands through the appropriate execution path.

Owns all command routing logic:
- Orchestrated (multi-user) vs legacy (single-user) paths
- Feedback detection, suggestion handling, permission checks
- Cache, prompt building, latency logging, event logging
"""

import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from src.nlu.command_grammar import PartialCommand, parse_partial_command
from src.nlu.sensitive_actions import is_sensitive
from src.orchestrator import PathType

logger = logging.getLogger(__name__)


# Frases que NO son comandos reales aunque llegaron al router — evitan que
# vayan al slow path del LLM (caro) cuando son ruido de TV, eco del TTS o
# filler conversacional. Match por substring sobre el texto normalizado
# (lowercase, sin acentos). Comparado con `_TV_STOP_PHRASES` del wake
# detector, acá incluimos eco del TTS típico del sistema.
_NOISE_PHRASES = (
    # TV/streaming (puede llegar al router si un wake previo abrió el flow)
    "suscribe", "suscrib", "campanita", "gracias por ver",
    "dale like", "dale lie", "dale mega like",
    "canal de youtube", "activa la",
    # Eco típico del TTS de respuestas/confirmaciones (audio del propio
    # sistema capturado por el mic cuando el barge-in/echo suppressor
    # falla puntualmente).
    "luz encendida", "luz apagada", "luces encendidas", "luces apagadas",
    "hecho", "perfecto", "listo",
)


def _is_noise_text(text: str, wake_words: tuple[str, ...] = ()) -> str | None:
    """Chequea si `text` parece ruido/eco, no un comando real.

    Args:
        text: Texto a evaluar (transcripción del comando post-wake).
        wake_words: Wake words configuradas (lowercase, sin acentos). Si
            ninguna aparece en el texto normalizado, se descarta como
            ruido — el buffer post-wake siempre debería incluir la palabra
            de despertar; si no está, la captura es probablemente TV/ruido
            de un wake disparado en un chunk anterior. Vacío = check off.

    Returns:
        Nombre de la regla que matcheó (para logging), o None si es un
        comando potencialmente válido.
    """
    if not text:
        return "empty"
    norm = unicodedata.normalize("NFD", text.lower())
    norm = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    norm = re.sub(r"[^\w\s]", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    if not norm:
        return "empty_after_norm"
    # Frases conocidas de noise / TV / eco
    for phrase in _NOISE_PHRASES:
        if phrase in norm:
            return f"noise_phrase:{phrase!r}"
    # Sólo "gracias" (sin verbo/entity) — eco del TTS agradeciendo.
    if norm in {"gracias", "si", "no", "ok", "bueno", "dale"}:
        return f"filler_word:{norm!r}"
    # Repetición extrema: una sola palabra repetida >= 4 veces
    words = norm.split()
    if len(words) >= 4 and len(set(words)) == 1:
        return f"word_repetition:{words[0]!r}"
    # Wake-word ausente — el comando se capturó pero el texto no contiene
    # ninguna de las wake words. Análisis de logs 24-25 abr (24 FPs únicos
    # de TV) mostró 0% de comandos reales sin wake en transcripción.
    if wake_words:
        if not any(w in norm for w in wake_words):
            return f"missing_wake:{wake_words[0]!r}"
    return None


def _texts_diverge(a: str, b: str, min_ratio: float = 0.95) -> bool:
    """True if two transcriptions differ enough to suspect hallucination.

    Uses accent-stripped lowercase SequenceMatcher ratio. Below min_ratio,
    the texts are considered divergent.

    Default 0.95 calibrated so single-word hallucinations (e.g. 'Nexa' → 'Para'
    in a ~6-word utterance, ratio ≈ 0.917) cross the threshold. Exact matches
    and near-duplicates caused by e.g. a one-letter insertion (ratio ≥ 0.97)
    stay silent.

    Returns False if either input is empty (nothing meaningful to compare).
    """
    if not a or not b:
        return False

    def _norm(t: str) -> str:
        t = unicodedata.normalize("NFD", t.lower())
        return "".join(c for c in t if unicodedata.category(c) != "Mn").strip()

    return SequenceMatcher(None, _norm(a), _norm(b)).ratio() < min_ratio


@dataclass
class PermissionResult:
    """Result of a permission check when no UserManager is available."""
    allowed: bool = True
    message: str = ""


class RequestRouter:
    """
    Routes voice commands through the correct execution path.

    Handles orchestrated (multi-user) and legacy (single-user) processing,
    feedback detection, suggestion handling, permission checks, caching,
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
        regex_extractor=None,
        llm_gate=None,
        hooks=None,  # plan #3 OpenClaw — HookRegistry instance or None
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
            user_manager: UserManager for permission checks.
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
                None → no hook calls, behavior identical to baseline.
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

        # LLM-based command validator. Si está configurado, intercepta el
        # texto post-wake antes del orchestrator y rechaza alucinaciones
        # de TV / replays / frases noise. Ver src/nlu/llm_router.py.
        self.llm_command_router = llm_command_router

        # Fast path determinístico: regex_extractor + llm_gate (binario).
        # Si el regex matchea limpio un patrón de domótica conocido, el LLM
        # gate valida con ~70-100ms y bypass del LLMCommandRouter completo
        # (que toma ~500-900ms). Si gate=false o regex no matchea, se cae
        # al flujo del LLMCommandRouter como antes. Ver src/nlu/regex/ y
        # src/nlu/llm_gate.py.
        self.regex_extractor = regex_extractor
        self.llm_gate = llm_gate

        # Plan #3 OpenClaw — plugin hooks registry (or None)
        self._hooks = hooks

        # State
        self._query_cache = {}
        self._cache_max_size = cache_max_size
        self._pending_suggestion = None
        self._last_response = None
        self._command_count = 0

        # Last-action tracker para comandos ambiguos (Q6: contextual + pregunta fallback)
        if action_tracker is None:
            from src.orchestrator.action_context import LastActionTracker
            action_tracker = LastActionTracker(ttl_seconds=60.0)
        self.action_tracker = action_tracker

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
        from src.pipeline.command_event import CommandEvent

        pretranscribed_text: str | None = None
        used_wake_text = False
        early_dispatch = False
        if isinstance(audio_or_event, CommandEvent):
            audio = audio_or_event.audio
            room_id = audio_or_event.room_id
            early_dispatch = audio_or_event.early_dispatch
            wake_text = audio_or_event.wake_text
            partial_text = (
                audio_or_event.partial_command.raw_text
                if audio_or_event.partial_command is not None else None
            )
            # Preferencia: wake_text > partial_command.raw_text.
            # Motivo: el wake detector transcribe con initial_prompt sesgado a la
            # keyword ("nexa"); el partial del streaming worker a veces alucina
            # la primera palabra ("Nexa" -> "Para"). Si ambos difieren mucho,
            # log para diagnóstico pero igual elegimos wake_text.
            if wake_text:
                pretranscribed_text = wake_text
                used_wake_text = True
                if partial_text and _texts_diverge(wake_text, partial_text):
                    logger.warning(
                        f"Wake/partial text mismatch — using wake: "
                        f"wake={wake_text!r} partial={partial_text!r}"
                    )
                else:
                    logger.info(
                        f"Using wake-detector text as pretranscribed: {wake_text!r}"
                    )
            elif partial_text is not None:
                pretranscribed_text = partial_text
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

        if self.orchestrator_enabled and self._orchestrator:
            return await self._process_command_orchestrated(
                audio, room_id=room_id, pretranscribed_text=pretranscribed_text,
            )
        else:
            return await self._process_command_legacy(
                audio, room_id=room_id, pretranscribed_text=pretranscribed_text,
            )

    async def _process_command_orchestrated(self, audio: np.ndarray, room_id: str = None,
                                              pretranscribed_text: str | None = None) -> dict:
        """Process command with multi-user orchestrator."""
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
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 1a. Noise filter: corta antes del orchestrator si el texto es
        #     claramente ruido (TV, eco del TTS, repetición). Evita que
        #     esas muestras vayan al slow path LLM (caro + bloquea queue).
        noise_reason = _is_noise_text(text, wake_words=self._wake_words)
        if noise_reason:
            logger.info(f"Noise discard ({noise_reason}): {text!r}")
            result["intent"] = "noise_discarded"
            result["success"] = False
            result["response"] = ""
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 1a-fast. Fast path determinístico: regex extractor + LLM gate binario.
        # Para single-intent que matchea limpio (apagá luz X, prendé Y, etc.),
        # bypaseamos el LLMCommandRouter completo (~500-900ms) y validamos solo
        # con un gate thin (~70-100ms). Si regex no matchea o gate=false, se
        # cae al flujo del LLMCommandRouter como antes.
        # Multi-intent (len>1) por ahora también cae al flujo normal — feature
        # futuro: dispatch paralelo iterando segments.
        fast_classification = None
        if self.regex_extractor is not None and self.llm_gate is not None:
            t_regex = time.perf_counter()
            rmatches = self.regex_extractor.extract(text)
            regex_ms = (time.perf_counter() - t_regex) * 1000
            result["timings"]["regex"] = regex_ms

            if len(rmatches) == 1:
                rmatch = rmatches[0]
                t_gate = time.perf_counter()
                gate_result = await self.llm_gate.validate(
                    text=text,
                    intent=rmatch.intent,
                    entity_hint=rmatch.entity_canonical,
                )
                gate_ms = (time.perf_counter() - t_gate) * 1000
                result["timings"]["llm_gate"] = gate_ms

                if gate_result.valid:
                    logger.info(
                        f"[FAST_PATH regex={regex_ms:.0f}ms gate={gate_ms:.0f}ms] "
                        f"intent={rmatch.intent} entity={rmatch.entity_canonical} "
                        f"text={text!r}"
                    )
                    # Construir CommandClassification para que el resto del flujo
                    # funcione idéntico (history recording, etc.). is_command=True
                    # asegura que NO se rechace abajo. Marcamos llm_classification
                    # para que el bloque del LLMCommandRouter se saltee.
                    from src.nlu.llm_router import CommandClassification
                    fast_classification = CommandClassification(
                        is_command=True,
                        confidence=0.95,
                        intent=rmatch.intent,
                        entity_hint=rmatch.entity_canonical,
                        slots=dict(rmatch.slots),
                        raw_response="<regex+gate>",
                        elapsed_ms=regex_ms + gate_ms,
                    )
                else:
                    logger.info(
                        f"[FAST_PATH_REJECT regex={regex_ms:.0f}ms gate={gate_ms:.0f}ms] "
                        f"text={text!r} — falling through to LLM router"
                    )
            elif len(rmatches) > 1:
                logger.debug(
                    f"[FAST_PATH_MULTI regex={regex_ms:.0f}ms n={len(rmatches)}] "
                    f"multi-intent no soportado en fast path; fall-through"
                )

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
            if not classification.is_command:
                result["intent"] = f"llm_rejected:{classification.rejection_reason or 'unknown'}"
                result["success"] = False
                result["response"] = ""
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result
            # Resultado válido — guardar para que el caller registre en history
            # tras dispatch exitoso (line ~480, post-orchestrator).
            result["llm_classification"] = classification

        # 1b. Confidence gate — low-confidence sensitive commands pide
        #     confirmación antes de dispatchar al orchestrator.
        self._check_confidence_gate(text, result)
        if result.get("pending_confirmation"):
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
        def on_response(dispatch_result):
            if dispatch_result.was_queued and dispatch_result.queue_position:
                self.response_handler.speak(
                    dispatch_result.response,
                    zone_id=zone_id,
                    room_context=room_context,
                )

        dispatch_result = await self._orchestrator.process(
            user_id=user_id,
            text=text,
            audio=audio if not user_id else None,
            zone_id=zone_id,
            on_response=on_response
        )

        # 6. Build result
        result["intent"] = dispatch_result.intent
        result["response"] = dispatch_result.response
        result["success"] = dispatch_result.success
        result["action"] = dispatch_result.action
        result["path"] = dispatch_result.path.value if dispatch_result.path else None
        result["timings"].update(dispatch_result.timings)

        # 6b. Attach room context to result
        if room_context:
            result["room"] = {
                "id": room_context.room_id,
                "name": room_context.room_name,
                "confidence": room_context.confidence,
                "source": room_context.source.value,
            }

        # 7. Speak response
        if dispatch_result.path not in [PathType.SLOW_LLM]:
            emotion_adj = emotion.response_adjustment if emotion else None
            self.response_handler.speak(
                result["response"],
                zone_id=zone_id,
                emotion_adjustment=emotion_adj,
                room_context=room_context,
            )

        # 8. Logging
        result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
        self._log_latency(result)

        if self.event_logger and result["success"] and result["intent"] == "domotics":
            self._log_domotics_event(
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
                                        pretranscribed_text: str | None = None) -> dict:
        """Legacy processing (single-user) for backwards compatibility."""
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
            return result

        # 1a. Noise filter — mismo corto-circuito que el path orchestrated
        noise_reason = _is_noise_text(text, wake_words=self._wake_words)
        if noise_reason:
            logger.info(f"Noise discard ({noise_reason}): {text!r}")
            result["intent"] = "noise_discarded"
            result["success"] = False
            result["response"] = ""
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
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
        self._check_confidence_gate(text, result)
        if result.get("pending_confirmation"):
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 2. Check feedback
        feedback_result = self._check_feedback(text)
        if feedback_result["is_feedback"]:
            result["intent"] = "feedback"
            result["response"] = feedback_result["response"]
            result["success"] = True
            self.response_handler.speak(result["response"])
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 3. Check pending suggestion
        if self._pending_suggestion:
            suggestion_result = self._handle_suggestion_response(text)
            if suggestion_result["handled"]:
                result["intent"] = "suggestion_response"
                result["response"] = suggestion_result["response"]
                result["success"] = True
                self.response_handler.speak(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # 4. Check command learner
        if self.command_learner:
            learning_result = self.command_learner.handle(text, self.ha, self.chroma)
            if learning_result["handled"]:
                result["intent"] = "learning"
                result["response"] = learning_result["response"]
                result["success"] = learning_result.get("success", True)
                self.response_handler.speak(result["response"])
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
                self.response_handler.speak(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # 6. Check sync command
        if self._is_sync_command(text):
            result["intent"] = "sync"
            self.response_handler.speak("Sincronizando comandos...")
            count = self.chroma.sync_commands(self.ha, self.llm)
            result["response"] = f"Listo, actualicé {count} comandos"
            result["success"] = True
            self.response_handler.speak(result["response"])
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
                self.response_handler.speak(result["response"])
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
                self.response_handler.speak(result["response"])
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
                self.response_handler.speak(result["response"])
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
                self.response_handler.speak(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

        # Fallback to original RoutineManager (compatibility)
        routine_result = await self.routines.handle(text)
        result["timings"]["routine_check"] = (time.perf_counter() - t_routine) * 1000

        if routine_result["handled"]:
            result["intent"] = "routine"
            result["response"] = routine_result["response"]
            result["success"] = routine_result["success"]
            self.response_handler.speak(result["response"])
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
            probe = self.chroma.search_command(
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
            self.response_handler.speak(result["response"])
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        if intent:
            logger.debug(f"NLU: intent={intent} slots={query_slots}")

        t_search = time.perf_counter()
        cache_key = f"{intent or 'any'}::{text.lower().strip()}"
        command = self._query_cache.get(cache_key)

        if not command:
            command = self.chroma.search_command(
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

            # Check permissions
            user_for_check = user
            if not user_for_check:
                from src.users.user_manager import User, PermissionLevel
                user_for_check = User(
                    user_id="unknown",
                    name="Desconocido",
                    permission_level=PermissionLevel.GUEST
                )

            check = self._check_permission(user_for_check, command["entity_id"], command["service"])
            if not check.allowed:
                result["success"] = False
                result["response"] = check.message
                self.response_handler.speak(result["response"])
                result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
                return result

            # Execute in HA (always use call_service_ws which auto-falls back)
            t_ha = time.perf_counter()
            success = await self.ha.call_service_ws(
                command["domain"],
                command["service"],
                command["entity_id"],
                command.get("data")
            )
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
            self.response_handler.speak(
                result["response"],
                emotion_adjustment=emotion_adj,
                room_context=room_context,
            )

            # Record in memory
            if self.memory:
                self.memory.record_interaction(
                    user_input=text,
                    assistant_response=result["response"],
                    intent="domotics",
                    entities_used=[command["entity_id"]]
                )

            # Log event
            if self.event_logger and success:
                self._log_domotics_event(
                    entity_id=command["entity_id"],
                    action=command["service"],
                    trigger_phrase=text
                )

            # Suggestions
            if self.suggestion_engine:
                self._command_count += 1
                if self._command_count >= self.suggestion_interval:
                    self._command_count = 0
                    self._maybe_present_suggestion()

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
                use_deep, router_response = self.router.classify_and_respond(
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
            self.response_handler.speak(
                result["response"],
                emotion_adjustment=emotion_adj,
                room_context=room_context,
            )
        else:
            prompt = await self._build_prompt(text)
            emotion_adj = emotion.response_adjustment if emotion else None
            response = self.response_handler.speak_with_llm_stream(
                prompt,
                max_tokens=512,
                temperature=0.7,
                use_filler=True,
                emotion_adjustment=emotion_adj
            )
            result["timings"]["llm_stream"] = (time.perf_counter() - pipeline_start) * 1000
            result["response"] = response.strip()
            result["success"] = True

        if self.memory:
            self.memory.record_interaction(
                user_input=text,
                assistant_response=result["response"],
                intent="conversation"
            )

        result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
        return result

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

    def _check_confidence_gate(self, text: str, result: dict) -> PartialCommand | None:
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

    def _check_permission(self, user, entity_id: str, service: str) -> object:
        """Check user permissions for an entity action."""
        if not self.user_manager:
            return PermissionResult()

        check = self.user_manager.check_entity_permission(user, entity_id, service)
        if not check.allowed:
            check.message = self.user_manager.format_permission_denied_message(user, check)
        return check

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

    def _maybe_present_suggestion(self):
        """Present an automation suggestion if one is available."""
        if not self.suggestion_engine:
            return

        suggestion = self.suggestion_engine.get_suggestion_to_present()
        if suggestion:
            self._pending_suggestion = suggestion
            message = suggestion.message + " Responde si, no, o despues."
            self.response_handler.speak(message)

    def _log_domotics_event(self, entity_id: str, action: str, trigger_phrase: str = None):
        """Log a domotics event."""
        if not self.event_logger:
            return

        from src.analytics.event_logger import EventType
        user = self.command_processor.get_current_user()
        self.event_logger.log(
            entity_id=entity_id,
            action=action,
            event_type=EventType.COMMAND,
            user_id=user.user_id if user else None,
            user_name=user.name if user else None,
            trigger_phrase=trigger_phrase
        )

    def _log_latency(self, result: dict):
        """Log latency metrics."""
        total = result["latency_ms"]
        target = self.latency_target_ms
        status = "OK" if total <= target else "SLOW"
        timings_str = " + ".join([f"{k}={v:.0f}" for k, v in result["timings"].items()])
        logger.info(f"[{status}] Total: {total:.0f}ms (target: {target}ms) [{timings_str}]")

        if self.latency_monitor:
            user_name = result.get("user", {}).get("name") if result.get("user") else None
            self.latency_monitor.record(
                timings=result["timings"],
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
                memory_context = self.memory.format_context_for_prompt(user_text)
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
