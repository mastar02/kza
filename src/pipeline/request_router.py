"""
Request Router Module
Routes processed voice commands through the appropriate execution path.

Owns all command routing logic:
- Orchestrated (multi-user) vs legacy (single-user) paths
- Feedback detection, suggestion handling, permission checks
- Cache, prompt building, latency logging, event logging
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.orchestrator import PathType

logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize RequestRouter with injected dependencies.

        Args:
            command_processor: CommandProcessor for STT + speaker ID + emotion.
            response_handler: ResponseHandler for TTS output.
            audio_manager: AudioManager for zone detection.
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
        """
        # Core pipeline components
        self.command_processor = command_processor
        self.response_handler = response_handler
        self.audio_manager = audio_manager

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

        # State
        self._query_cache = {}
        self._cache_max_size = cache_max_size
        self._pending_suggestion = None
        self._last_response = None
        self._command_count = 0

    async def process_command(self, audio: np.ndarray) -> dict:
        """
        Process a complete audio command.

        Routes through orchestrated (multi-user) or legacy (single-user) path
        depending on configuration.

        Args:
            audio: Raw audio data as numpy array.

        Returns:
            Dict with keys: text, intent, action, response, success,
            latency_ms, user, timings, and optionally path.
        """
        if self.orchestrator_enabled and self._orchestrator:
            return await self._process_command_orchestrated(audio)
        else:
            return await self._process_command_legacy(audio)

    async def _process_command_orchestrated(self, audio: np.ndarray) -> dict:
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

        # 1. Process command (STT + Speaker ID + Emotion in parallel)
        cmd_result = await self.command_processor.process_command(audio, use_parallel=True)
        text = cmd_result["text"]
        result["text"] = text
        result["timings"].update(cmd_result["timings"])

        if not text.strip():
            result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
            return result

        # 2. Detect source zone
        zone_id = self.audio_manager.detect_source_zone(audio)
        if zone_id:
            self.response_handler.set_active_zone(zone_id)

        # 3. Get user info
        user = cmd_result.get("user")
        emotion = cmd_result.get("emotion")
        user_id = user.user_id if user else None
        user_name = user.name if user else None

        if user:
            result["user"] = {
                "name": user_name,
                "permission_level": user.permission_level.name
            }

        logger.info(
            f"[Orchestrator] User={user_name or 'unknown'}, "
            f"Zone={zone_id}, Text={text[:50]}, Emotion={emotion.emotion if emotion else 'none'}"
        )

        # 4. Process with orchestrator
        def on_response(dispatch_result):
            if dispatch_result.was_queued and dispatch_result.queue_position:
                self.response_handler.speak(
                    dispatch_result.response,
                    zone_id=zone_id
                )

        dispatch_result = await self._orchestrator.process(
            user_id=user_id,
            text=text,
            audio=audio if not user_id else None,
            zone_id=zone_id,
            on_response=on_response
        )

        # 5. Build result
        result["intent"] = dispatch_result.intent
        result["response"] = dispatch_result.response
        result["success"] = dispatch_result.success
        result["action"] = dispatch_result.action
        result["path"] = dispatch_result.path.value if dispatch_result.path else None
        result["timings"].update(dispatch_result.timings)

        # 6. Speak response
        if dispatch_result.path not in [PathType.SLOW_LLM]:
            emotion_adj = emotion.response_adjustment if emotion else None
            self.response_handler.speak(
                result["response"],
                zone_id=zone_id,
                emotion_adjustment=emotion_adj
            )

        # 7. Logging
        result["latency_ms"] = (time.perf_counter() - pipeline_start) * 1000
        self._log_latency(result)

        if self.event_logger and result["success"] and result["intent"] == "domotics":
            self._log_domotics_event(
                entity_id=result["action"]["entity_id"] if result["action"] else None,
                action=result["action"]["service"] if result["action"] else None,
                trigger_phrase=text
            )

        return result

    async def _process_command_legacy(self, audio: np.ndarray) -> dict:
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

        # 1. Process command
        cmd_result = await self.command_processor.process_command(audio, use_parallel=True)
        text = cmd_result["text"]
        result["text"] = text
        result["timings"].update(cmd_result["timings"])

        if not text.strip():
            return result

        user = cmd_result.get("user")
        emotion = cmd_result.get("emotion")

        if user:
            result["user"] = {
                "name": user.name,
                "permission_level": user.permission_level.name
            }

        logger.info(f"[STT {cmd_result['timings'].get('stt', 0):.0f}ms] {text}")

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

        # 8. Vector DB search (with cache)
        t_search = time.perf_counter()
        cache_key = text.lower().strip()
        command = self._query_cache.get(cache_key)

        if not command:
            command = self.chroma.search_command(text, self.vector_search_threshold)
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

            # Speak response with emotion adjustments
            emotion_adj = emotion.response_adjustment if emotion else None
            self.response_handler.speak(result["response"], emotion_adjustment=emotion_adj)

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
            self.response_handler.speak(result["response"], emotion_adjustment=emotion_adj)
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

        user = self.command_processor.get_current_user()
        self.event_logger.log(
            entity_id=entity_id,
            action=action,
            event_type="command",
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

    def _get_conversation_context(self) -> str:
        """Get brief context for the router (optimized for speed)."""
        context_parts = []

        # Memory context (if available)
        if self.memory:
            try:
                recent = self.memory.get_recent_context(limit=3)
                if recent:
                    context_parts.append(f"Reciente: {recent}")
            except Exception:
                pass

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
        except Exception:
            pass

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
