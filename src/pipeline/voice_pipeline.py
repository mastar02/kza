"""
Voice Pipeline Module
Pipeline principal de procesamiento de voz - coordinador de alto nivel.
"""

import asyncio
import logging
from typing import Optional

import numpy as np

from src.pipeline.audio_manager import AudioManager
from src.pipeline.audio_loop import AudioLoop
from src.pipeline.command_processor import CommandProcessor
from src.pipeline.response_handler import ResponseHandler
from src.orchestrator import MultiUserOrchestrator
from src.routines import VoiceRoutineHandler, RoutineScheduler, RoutineExecutor

# Nuevos módulos de funcionalidades diferenciadores
from src.conversation import FollowUpMode, ConversationState
from src.ambient import AudioEventDetector, AudioEventType
from src.proactive import MorningBriefing, BriefingConfig
from src.learning import EntityLearner, PatternLearner

# Audio processing
from src.audio import EchoSuppressor, EchoSuppressionConfig, SpeakerState

logger = logging.getLogger(__name__)


class VoicePipeline:
    """
    Pipeline completo de voz para domótica inteligente.

    Coordina tres componentes principales:
    1. AudioManager: Captura de audio y detección de wake word
    2. CommandProcessor: STT, identificación de usuario, clasificación
    3. ResponseHandler: TTS, streaming, enrutamiento de audio

    Además:
    - Soporte para orquestación multi-usuario
    - Gestión de feedback y correcciones
    - Sugerencias de automatización
    - Cache de búsquedas
    """

    def __init__(
        self,
        stt,
        tts,
        chroma_sync,
        routine_manager,
        ha_client,
        llm_reasoner,
        fast_router=None,
        memory_manager=None,
        speaker_identifier=None,
        user_manager=None,
        emotion_detector=None,
        voice_enrollment=None,
        presence_detector=None,  # Detección de presencia BLE
        routine_scheduler=None,  # Scheduler de rutinas (nuevo sistema)
        routine_executor=None,   # Executor de rutinas
        latency_monitor=None,
        conversation_collector=None,
        command_learner=None,
        event_logger=None,
        suggestion_engine=None,
        zone_manager=None,
        features=None,             # FeatureManager (timers, intercom, alerts, etc.)
        audio_loop=None,           # AudioLoop (extracted audio capture loop)
        request_router=None,       # RequestRouter (extracted command routing)
        # ===== NUEVAS FUNCIONALIDADES DIFERENCIADORES =====
        weather_provider=None,     # Para briefings
        calendar_provider=None,    # Para briefings
        traffic_provider=None,     # Para briefings
        task_provider=None,        # Para briefings
        # ===== CONFIGURACIÓN =====
        wake_word_model: str = "hey_jarvis",
        wake_word_threshold: float = 0.5,
        sample_rate: int = 16000,
        command_duration: float = 2.0,
        vector_search_threshold: float = 0.65,
        latency_target_ms: int = 300,
        suggestion_interval: int = 50,
        streaming_enabled: bool = True,
        streaming_buffer_ms: int = 150,
        streaming_prebuffer_ms: int = 30,
        llm_buffer_preset: str = "balanced",
        llm_use_filler: bool = True,
        llm_filler_phrases: list = None,
        orchestrator_enabled: bool = True,
        orchestrator_max_history: int = 10,
        orchestrator_context_timeout: float = 300,
        orchestrator_auto_cancel: bool = True,
        orchestrator_queue_size: int = 100,
        # ===== CONFIG NUEVAS FUNCIONALIDADES =====
        follow_up_window: float = 8.0,      # Segundos sin wake word
        ambient_detection: bool = True,      # Detectar eventos ambientales
        proactive_briefings: bool = True,    # Briefings matutinos
        pattern_learning: bool = True,       # Aprender patrones
        data_dir: str = "data"               # Directorio para datos aprendidos
    ):
        """
        Inicializar VoicePipeline con inyección de dependencias.

        Agrupa configuración en tres dominios:
        - Audio: captura, wake word, VAD
        - Command: STT, speaker ID, intent
        - Response: TTS, streaming, routing
        """
        # Servicios principales
        self.chroma = chroma_sync
        self.routines = routine_manager
        self.ha = ha_client
        self.llm = llm_reasoner
        self.router = fast_router
        self.memory = memory_manager

        # Sistemas complementarios
        self.latency_monitor = latency_monitor
        self.conversation_collector = conversation_collector
        self.command_learner = command_learner
        self.event_logger = event_logger
        self.suggestion_engine = suggestion_engine
        self.enrollment = voice_enrollment
        self.presence = presence_detector  # Detección de presencia BLE

        # Sistema de rutinas mejorado
        self.routine_scheduler = routine_scheduler
        self.routine_executor = routine_executor
        self.voice_routine_handler = None

        if routine_scheduler:
            self.voice_routine_handler = VoiceRoutineHandler(
                routine_scheduler=routine_scheduler,
                routine_manager=routine_manager,
                llm_reasoner=llm_reasoner
            )
            logger.info("VoiceRoutineHandler inicializado")

        # Configuración común
        self.sample_rate = sample_rate
        self.vector_search_threshold = vector_search_threshold
        self.latency_target_ms = latency_target_ms

        # Estados
        self._running = False
        self._ws_connected = False

        # Componentes principales
        self.audio_manager = AudioManager(
            zone_manager=zone_manager,
            wake_word_model=wake_word_model,
            wake_word_threshold=wake_word_threshold,
            sample_rate=sample_rate,
            command_duration=command_duration
        )

        self.command_processor = CommandProcessor(
            stt=stt,
            speaker_identifier=speaker_identifier,
            user_manager=user_manager,
            emotion_detector=emotion_detector,
            sample_rate=sample_rate
        )

        self.response_handler = ResponseHandler(
            tts=tts,
            zone_manager=zone_manager,
            llm=llm_reasoner,
            streaming_enabled=streaming_enabled,
            streaming_buffer_ms=streaming_buffer_ms,
            streaming_prebuffer_ms=streaming_prebuffer_ms,
            llm_buffer_preset=llm_buffer_preset,
            llm_use_filler=llm_use_filler,
            llm_filler_phrases=llm_filler_phrases
        )

        # Echo Suppressor: evita que el TTS se capture como comando
        self.echo_suppressor = EchoSuppressor(
            sample_rate=sample_rate,
            config=EchoSuppressionConfig(
                ducking_enabled=True,
                post_speech_buffer_ms=400,     # 400ms de cooldown después del TTS
                echo_detection_enabled=True,
                echo_correlation_threshold=0.5,
                vad_enabled=True
            )
        )
        logger.info("Echo suppressor inicializado")

        # Parchear response_handler.speak para notificar al echo suppressor
        self._original_speak = self.response_handler.speak
        self.response_handler.speak = self._speak_with_echo_notify

        # Multi-user orchestrator
        self.orchestrator_enabled = orchestrator_enabled
        self._orchestrator: Optional[MultiUserOrchestrator] = None

        if orchestrator_enabled:
            self._orchestrator = MultiUserOrchestrator(
                chroma_sync=chroma_sync,
                ha_client=ha_client,
                routine_manager=routine_manager,
                router=fast_router,
                llm=llm_reasoner,
                tts=tts,
                speaker_identifier=speaker_identifier,
                user_manager=user_manager,
                max_context_history=orchestrator_max_history,
                context_timeout=orchestrator_context_timeout,
                auto_cancel_previous=orchestrator_auto_cancel
            )
            logger.info("Multi-user orchestrator inicializado")

        # ===== NUEVAS FUNCIONALIDADES DIFERENCIADORES =====

        # 1. Follow-Up Mode: Conversación sin wake word
        self.follow_up = FollowUpMode(
            follow_up_window=follow_up_window,
            question_window=15.0,  # Más tiempo si KZA preguntó
            whisper_detection=True
        )
        self.follow_up.on_conversation_end(self._on_conversation_end)
        logger.info("Follow-up mode inicializado")

        # 2. Detección Ambiental
        self.ambient_detector = None
        if ambient_detection:
            self.ambient_detector = AudioEventDetector(
                sample_rate=sample_rate,
                chunk_duration_ms=1000
            )
            self.ambient_detector.on_event(self._on_ambient_event)
            self.ambient_detector.on_security_event(self._on_security_event)
            logger.info("Detección ambiental inicializada")

        # 3. Entity Learner: Aprendizaje dinámico
        self.entity_learner = EntityLearner(
            ha_client=ha_client,
            user_manager=user_manager,
            data_dir=f"{data_dir}/learned"
        )
        logger.info("Entity learner inicializado")

        # 4. Pattern Learner: Automatizaciones aprendidas
        self.pattern_learner = None
        if pattern_learning:
            self.pattern_learner = PatternLearner(
                data_dir=f"{data_dir}/patterns"
            )
            self.pattern_learner.on_suggestion_ready(self._on_routine_suggestion)
            logger.info("Pattern learner inicializado")

        # 5. Briefings Proactivos
        self.briefing = None
        if proactive_briefings:
            self.briefing = MorningBriefing(
                weather_provider=weather_provider,
                calendar_provider=calendar_provider,
                traffic_provider=traffic_provider,
                task_provider=task_provider,
                ha_client=ha_client,
                llm_client=llm_reasoner
            )
            self.briefing.on_briefing_ready(self._on_briefing_ready)
            logger.info("Briefings proactivos inicializados")

        # 6. Feature Manager (timers, intercom, notifications, alerts, HA integration)
        self.features = features
        if features:
            logger.info("FeatureManager injected")

        # 7. AudioLoop (extracted audio capture loop)
        self.audio_loop = audio_loop

        # 8. RequestRouter (extracted command routing)
        self.request_router = request_router

    async def process_command(self, audio: np.ndarray) -> dict:
        """
        Process a complete audio command.

        Delegates to RequestRouter if configured, otherwise returns an error.

        Args:
            audio: Raw audio data as numpy array.

        Returns:
            Dict with text, intent, action, response, success, latency_ms, user.
        """
        if self.request_router:
            return await self.request_router.process_command(audio)
        return {"text": "", "success": False, "error": "No request router configured"}

    async def _connect_websocket(self):
        """Conectar WebSocket a Home Assistant."""
        try:
            self._ws_connected = await self.ha.connect_websocket()
            if self._ws_connected:
                logger.info("WebSocket conectado")
            else:
                logger.warning("WebSocket no disponible - usando REST API")
        except Exception as e:
            logger.warning(f"Error conectando WebSocket: {e}")
            self._ws_connected = False

    async def run(self):
        """Loop principal de escucha."""
        logger.info("Inicializando pipeline...")

        # Cargar modelos
        self.command_processor.load_models()
        self.audio_manager.load_wake_word()
        self.chroma.initialize()

        if self.memory:
            self.memory.initialize()

        if self.user_manager:
            users = self.user_manager.get_all_users()
            if not users:
                self.user_manager.create_default_admin()

        await self._connect_websocket()

        if self.orchestrator_enabled and self._orchestrator:
            await self._orchestrator.start()

        # Iniciar scheduler de rutinas
        if self.routine_scheduler:
            await self.routine_scheduler.start()
            logger.info(f"RoutineScheduler iniciado con {len(self.routine_scheduler.get_all_routines())} rutinas")

        stats = self.chroma.get_stats()
        if stats["commands_phrases"] == 0:
            logger.info("Base vectorial vacía, sincronizando...")
            self.chroma.sync_commands(self.ha, self.llm)
        else:
            logger.info(f"Base vectorial: {stats['commands_phrases']} frases")

        # ===== INICIALIZAR NUEVAS FUNCIONALIDADES =====

        # Sincronizar entidades de Home Assistant
        await self.sync_entities_from_ha()

        # Ambient detector initialization is handled by AudioLoop.start()

        # Iniciar análisis de patrones en background
        if self.pattern_learner:
            asyncio.create_task(self.pattern_learner.run_analysis_loop(interval_hours=24))
            logger.info("🧠 Aprendizaje de patrones activo")

        # Iniciar FeatureManager (timers, intercom, notifications, alerts, HA)
        if self.features:
            await self.features.start()
            logger.info("FeatureManager started")

        logger.info(f"Sistema listo. Target latency: {self.latency_target_ms}ms")
        logger.info(f"  Pattern learning: {'ON' if self.pattern_learner else 'OFF'}")
        logger.info(f"  Features: {'ON' if self.features else 'OFF'}")

        self._running = True

        # ===== AUDIO LOOP =====
        if self.audio_loop:
            # Register command callback
            self.audio_loop.on_command(self.process_command)

            # Register post-command callback for learning and follow-up
            self.audio_loop.on_post_command(self._post_command_handler)

            # Start audio subsystems (wake word, ambient)
            await self.audio_loop.start()

            # Run the audio loop (blocks until stop)
            await self.audio_loop.run()
        else:
            logger.warning("No AudioLoop configured — pipeline running without audio capture")

    async def _post_command_handler(self, result: dict, audio_data: np.ndarray):
        """
        Post-command processing after AudioLoop captures and processes a command.

        Handles pattern learning, entity alias learning, follow-up notifications,
        and question detection for conversation window extension.

        Args:
            result: Command processing result dict.
            audio_data: Raw audio data of the captured command.
        """
        # Learn from the action
        if result.get("success") and result.get("action"):
            action = result["action"]
            user = result.get("user", {})

            # Record for pattern learning
            self.record_user_action(
                action_type=f"{action.get('domain', 'unknown')}_{action.get('service', 'unknown')}",
                entity_id=action.get("entity_id", ""),
                user_id=user.get("user_id"),
                data=action.get("data")
            )

            # Learn alias if natural language was used
            if result.get("text"):
                self.learn_entity_alias(
                    text=result["text"],
                    entity_id=action.get("entity_id", ""),
                    user_id=user.get("user_id")
                )

        # Notify follow-up mode
        self.notify_user_spoke(
            result.get("text", ""),
            result.get("user", {}).get("user_id")
        )

        # If KZA responded with a question, extend follow-up window
        asked_question = "?" in result.get("response", "")
        self.notify_kza_responded(asked_question=asked_question)

    async def stop(self):
        """Detener pipeline."""
        self._running = False

        if self._orchestrator:
            await self._orchestrator.stop()

        if self.routine_scheduler:
            await self.routine_scheduler.stop()

        # Stop audio loop (handles ambient detector)
        if self.audio_loop:
            await self.audio_loop.stop()

        if self.follow_up.is_active:
            self.follow_up.end_conversation("shutdown")

        # Detener FeatureManager
        if self.features:
            await self.features.stop()

        if self._ws_connected:
            await self.ha.close()
            self._ws_connected = False

    async def test_from_text(self, text: str) -> dict:
        """Testear directamente con texto."""
        result = {"text": text, "intent": None, "action": None, "response": "", "success": False}

        command = self.chroma.search_command(text, self.vector_search_threshold)
        if command:
            result["intent"] = "domotics"
            result["action"] = command
            result["success"] = True
            result["response"] = command["description"]
        else:
            result["intent"] = "no_match"

        return result

    # Métodos de control del orquestador
    def get_orchestrator_stats(self) -> dict:
        """Obtener estadísticas del orquestador."""
        if not self._orchestrator:
            return {"error": "Orchestrator not enabled"}
        return self._orchestrator.get_stats()

    def get_active_contexts(self) -> list:
        """Obtener usuarios con contexto activo."""
        if not self._orchestrator:
            return []
        return list(self._orchestrator._context_manager._contexts.keys())

    def clear_user_context(self, user_id: str) -> bool:
        """Limpiar contexto de usuario."""
        if not self._orchestrator:
            return False
        return self._orchestrator._context_manager.clear(user_id)

    def get_queue_size(self) -> int:
        """Obtener tamaño de cola."""
        if not self._orchestrator:
            return 0
        return self._orchestrator._queue.qsize()

    def cancel_user_requests(self, user_id: str) -> int:
        """Cancelar peticiones de usuario."""
        if not self._orchestrator:
            return 0
        return self._orchestrator._cancel_manager.cancel_user(user_id)

    def get_automation_suggestions(self, min_confidence: float = 0.7) -> list:
        """Obtener sugerencias de automatización."""
        if not self.suggestion_engine:
            return []
        return self.suggestion_engine.generate_suggestions(
            min_confidence=min_confidence,
            max_suggestions=10
        )

    def get_event_stats(self) -> dict:
        """Obtener estadísticas de eventos."""
        if not self.event_logger:
            return {"error": "Event logger not configured"}
        return self.event_logger.get_stats()

    def analyze_patterns(self, days: int = 30) -> list:
        """Analizar patrones de uso."""
        if not self.suggestion_engine:
            return []
        patterns = self.suggestion_engine.pattern_analyzer.analyze_all(days)
        return [p.to_dict() for p in patterns]

    @property
    def is_orchestrator_running(self) -> bool:
        """Verificar si orquestador está corriendo."""
        return self._orchestrator is not None and self._orchestrator._running

    # =========================================================================
    # Métodos de Presencia (BLE)
    # =========================================================================

    def who_is_home(self) -> list[str]:
        """Obtener lista de usuarios en casa."""
        if not self.presence:
            return []
        return self.presence.who_is_home()

    def is_anyone_home(self) -> bool:
        """¿Hay alguien en casa?"""
        if not self.presence:
            return True  # Asumir que sí si no hay detector
        return self.presence.is_anyone_home()

    def get_presence_summary(self) -> dict:
        """Obtener resumen completo de presencia."""
        if not self.presence:
            return {"error": "Presence detector not configured"}
        return self.presence.get_summary()

    def get_zone_occupancy(self, zone_id: str) -> dict:
        """Obtener ocupación de una zona."""
        if not self.presence:
            return {}
        occupancy = self.presence.get_zone_occupancy(zone_id)
        if occupancy:
            return {
                "zone": occupancy.zone_name,
                "people": occupancy.estimated_people,
                "users": occupancy.known_users,
                "occupied": occupancy.is_occupied
            }
        return {}

    def is_user_home(self, user_id: str) -> bool:
        """¿Está un usuario específico en casa?"""
        if not self.presence:
            return True  # Asumir que sí si no hay detector
        return self.presence.is_user_home(user_id)

    def get_user_location(self, user_id: str) -> str:
        """Obtener zona actual de un usuario."""
        if not self.presence:
            return "unknown"
        return self.presence.get_user_zone(user_id) or "unknown"

    # =========================================================================
    # Métodos de Rutinas
    # =========================================================================

    def get_all_routines(self) -> list:
        """Obtener todas las rutinas registradas."""
        if not self.routine_scheduler:
            return []
        return self.routine_scheduler.get_all_routines()

    def get_routine_stats(self) -> dict:
        """Obtener estadísticas de rutinas."""
        if not self.routine_scheduler:
            return {"error": "Routine scheduler not configured"}

        routines = self.routine_scheduler.get_all_routines()
        return {
            "total": len(routines),
            "enabled": sum(1 for r in routines if r.enabled),
            "disabled": sum(1 for r in routines if not r.enabled),
            "by_voice": sum(1 for r in routines if r.created_by == "voice"),
            "by_dashboard": sum(1 for r in routines if r.created_by == "dashboard"),
            "total_executions": sum(r.execution_count for r in routines)
        }

    async def execute_routine_by_name(self, name: str, context: dict = None) -> dict:
        """Ejecutar una rutina por nombre."""
        if not self.routine_scheduler:
            return {"success": False, "error": "Routine scheduler not configured"}

        result = await self.routine_scheduler.execute_by_name(name, context)
        return {"success": result is not None, "result": result}

    def save_routines(self, filepath: str = None):
        """Guardar rutinas a archivo."""
        if not self.routine_scheduler:
            return

        if not filepath:
            filepath = "config/routines.json"

        self.routine_scheduler.save_to_file(filepath)

    def load_routines(self, filepath: str = None):
        """Cargar rutinas desde archivo."""
        if not self.routine_scheduler:
            return

        if not filepath:
            filepath = "config/routines.json"

        self.routine_scheduler.load_from_file(filepath)

    # =========================================================================
    # NUEVAS FUNCIONALIDADES DIFERENCIADORES
    # =========================================================================

    # ----- Follow-Up Mode (Conversación sin wake word) -----

    def _on_conversation_end(self, context, reason: str):
        """Callback cuando termina una conversación"""
        if context:
            logger.info(
                f"Conversación terminada ({reason}): "
                f"{context.turn_count} turnos, usuario={context.user_id}"
            )

    def should_process_without_wake_word(self, has_wake_word: bool = False) -> bool:
        """
        ¿Debería procesar este audio sin wake word?
        Permite conversación natural tipo follow-up.
        """
        return self.follow_up.should_accept_speech(has_wake_word)

    def start_conversation(self, user_id: str = None):
        """Iniciar modo conversación (permite follow-ups sin wake word)"""
        self.follow_up.start_conversation(user_id)

    def end_conversation(self):
        """Terminar modo conversación"""
        self.follow_up.end_conversation("manual")

    # ----- Echo Suppression (evitar captura de TTS) -----

    def _speak_with_echo_notify(self, text: str, zone_id: str = None, **kwargs):
        """
        Wrapper de speak que notifica al echo suppressor.
        Reemplaza response_handler.speak para interceptar todas las llamadas.
        """
        # Notificar inicio de TTS
        self.echo_suppressor.notify_tts_start()

        try:
            # Llamar al speak original
            return self._original_speak(text, zone_id=zone_id, **kwargs)
        finally:
            # Notificar fin de TTS (siempre, incluso si hay error)
            self.echo_suppressor.notify_tts_end()

    def speak_with_echo_suppression(self, text: str, zone_id: str = None, **kwargs):
        """
        Hablar con supresión de eco (método explícito).
        Alias del wrapper automático.
        """
        return self.response_handler.speak(text, zone_id=zone_id, **kwargs)

    def get_echo_suppressor_stats(self) -> dict:
        """Obtener estadísticas del supresor de eco"""
        return self.echo_suppressor.get_stats()

    def set_echo_suppression_enabled(self, enabled: bool):
        """Habilitar/deshabilitar supresión de eco"""
        self.echo_suppressor.config.ducking_enabled = enabled
        self.echo_suppressor.config.echo_detection_enabled = enabled
        logger.info(f"Echo suppression {'habilitado' if enabled else 'deshabilitado'}")

    def notify_user_spoke(self, text: str, user_id: str = None):
        """Notificar que el usuario habló (para mantener conversación activa)"""
        self.follow_up.on_user_speech(text, user_id)

    def notify_kza_responded(self, asked_question: bool = False):
        """Notificar que KZA respondió"""
        self.follow_up.on_kza_response(asked_question=asked_question)
        self.follow_up.on_kza_finished_speaking()

    # ----- Detección Ambiental -----

    def _on_ambient_event(self, event):
        """Callback para eventos ambientales detectados"""
        logger.info(f"🔔 Evento ambiental: {event.event_type.value} ({event.confidence:.0%})")

        # Mapear eventos a respuestas
        responses = {
            AudioEventType.DOORBELL: "Alguien está tocando el timbre. ¿Quieres que muestre la cámara?",
            AudioEventType.BABY_CRYING: "El bebé está llorando.",
            AudioEventType.DOG_BARKING: "El perro está ladrando.",
            AudioEventType.APPLIANCE_BEEP: "Parece que un electrodoméstico terminó.",
            AudioEventType.PHONE_RINGING: "Está sonando un teléfono.",
        }

        if event.event_type in responses:
            self.response_handler.speak(responses[event.event_type])

    def _on_security_event(self, event):
        """Callback para eventos de seguridad (prioridad alta)"""
        logger.warning(f"🚨 ALERTA DE SEGURIDAD: {event.event_type.value}")

        alerts = {
            AudioEventType.SMOKE_ALARM: "¡Alerta! Detecté una alarma de humo. ¿Estás bien?",
            AudioEventType.GLASS_BREAKING: "¡Alerta! Detecté sonido de vidrio rompiéndose.",
            AudioEventType.SCREAM: "¡Alerta! Detecté un grito. ¿Necesitas ayuda?",
        }

        if event.event_type in alerts:
            # Interrumpir cualquier cosa y alertar
            self.response_handler.speak(alerts[event.event_type], priority=True)

            # TODO: Enviar notificación push, activar cámaras, etc.

    def configure_ambient_event(self, event_type: str, enabled: bool = None, sensitivity: float = None):
        """Configurar detección de un tipo de evento"""
        if not self.ambient_detector:
            return

        from src.ambient import AudioEventType
        try:
            et = AudioEventType(event_type)
            if enabled is not None:
                if enabled:
                    self.ambient_detector.enable_event(et)
                else:
                    self.ambient_detector.disable_event(et)
            if sensitivity is not None:
                self.ambient_detector.set_sensitivity(et, sensitivity)
        except ValueError:
            logger.warning(f"Tipo de evento desconocido: {event_type}")

    # ----- Entity Learning (Aprendizaje Dinámico) -----

    async def sync_entities_from_ha(self):
        """Sincronizar entidades desde Home Assistant"""
        await self.entity_learner.sync_from_home_assistant()
        logger.info("Entidades sincronizadas desde Home Assistant")

    def resolve_entity(self, text: str, user_id: str = None, domain: str = None) -> str:
        """
        Resolver texto a entity_id usando aprendizaje.
        Ej: "luz del cuarto de mi hijo" -> "light.kids_room"
        """
        return self.entity_learner.resolve_entity(text, user_id, domain)

    def learn_entity_alias(self, text: str, entity_id: str, user_id: str = None):
        """Aprender un nuevo alias para una entidad"""
        self.entity_learner.learn_from_utterance(
            text=text,
            resolved_entity=entity_id,
            user_id=user_id
        )

    def register_user(self, user_id: str, name: str, voice_profile_id: str = None):
        """Registrar nuevo usuario en el sistema de aprendizaje"""
        self.entity_learner.register_user(user_id, name, voice_profile_id)

    def get_user_name(self, user_id: str) -> str:
        """Obtener nombre de usuario"""
        return self.entity_learner.get_user_name(user_id) or user_id

    def get_learned_entities(self, domain: str = None) -> list:
        """Obtener entidades aprendidas"""
        return self.entity_learner.get_all_entities(domain)

    # ----- Pattern Learning (Automatizaciones Aprendidas) -----

    def record_user_action(self, action_type: str, entity_id: str, user_id: str = None, data: dict = None):
        """
        Registrar una acción del usuario para aprendizaje de patrones.
        Llamar después de cada comando exitoso.
        """
        if self.pattern_learner:
            self.pattern_learner.record_action(
                action_type=action_type,
                entity_id=entity_id,
                user_id=user_id,
                data=data,
                trigger="voice"
            )

    async def analyze_user_patterns(self) -> list:
        """Analizar patrones de uso y detectar automatizaciones potenciales"""
        if not self.pattern_learner:
            return []
        return await self.pattern_learner.analyze_patterns()

    def _on_routine_suggestion(self, suggestion):
        """Callback cuando el sistema detecta un patrón y sugiere una rutina"""
        text = self.pattern_learner.get_suggestion_text(suggestion)
        logger.info(f"💡 Sugerencia de rutina: {text}")

        # Guardar para preguntar al usuario
        self._pending_pattern_suggestion = suggestion

        # Preguntar al usuario
        self.response_handler.speak(text)

    def accept_routine_suggestion(self, suggestion_id: str) -> dict:
        """Aceptar sugerencia de rutina aprendida"""
        if not self.pattern_learner:
            return None

        routine_data = self.pattern_learner.accept_suggestion(suggestion_id)
        if routine_data and self.routine_scheduler:
            # Crear la rutina
            from src.routines import ScheduledRoutine
            import uuid

            routine = ScheduledRoutine(
                routine_id=f"learned_{uuid.uuid4().hex[:8]}",
                **routine_data
            )
            self.routine_scheduler.register_routine(routine)
            return {"success": True, "routine": routine_data}

        return {"success": False}

    def dismiss_routine_suggestion(self, suggestion_id: str):
        """Rechazar sugerencia de rutina"""
        if self.pattern_learner:
            self.pattern_learner.dismiss_suggestion(suggestion_id)

    def get_pending_suggestions(self) -> list:
        """Obtener sugerencias de rutinas pendientes"""
        if not self.pattern_learner:
            return []
        return self.pattern_learner.get_pending_suggestions()

    # ----- Briefings Proactivos -----

    def _on_briefing_ready(self, user_id: str, text: str, data):
        """Callback cuando un briefing está listo"""
        logger.info(f"📋 Briefing listo para {user_id}")
        # El briefing se entrega cuando el usuario lo solicite o por presencia

    async def deliver_morning_briefing(self, user_id: str, user_name: str = None) -> str:
        """Entregar briefing matutino a un usuario"""
        if not self.briefing:
            return "Briefings no configurados"

        name = user_name or self.get_user_name(user_id)
        text = await self.briefing.deliver_briefing(user_id, name)
        self.response_handler.speak(text)
        return text

    async def should_deliver_briefing(self, user_id: str) -> bool:
        """¿Debería entregar briefing a este usuario ahora?"""
        if not self.briefing:
            return False
        return await self.briefing.should_deliver_briefing(user_id, trigger="presence")

    def configure_user_briefing(self, user_id: str, config: dict):
        """Configurar briefing para un usuario"""
        if not self.briefing:
            return

        from src.proactive import BriefingConfig
        briefing_config = BriefingConfig(**config)
        self.briefing.configure_user(user_id, briefing_config)

    # ----- Estado General de Nuevas Funcionalidades -----

    def get_new_features_status(self) -> dict:
        """Obtener estado de las nuevas funcionalidades"""
        status = {
            "follow_up_mode": {
                "active": self.follow_up.is_active,
                "state": self.follow_up.state.value,
                "needs_wake_word": self.follow_up.needs_wake_word
            },
            "ambient_detection": {
                "enabled": self.ambient_detector is not None,
                "status": self.ambient_detector.get_status() if self.ambient_detector else None
            },
            "entity_learning": {
                "status": self.entity_learner.get_status()
            },
            "pattern_learning": {
                "enabled": self.pattern_learner is not None,
                "status": self.pattern_learner.get_status() if self.pattern_learner else None
            },
            "briefings": {
                "enabled": self.briefing is not None,
                "status": self.briefing.get_status() if self.briefing else None
            },
            "echo_suppression": {
                "enabled": self.echo_suppressor.config.ducking_enabled,
                "state": self.echo_suppressor.state.value,
                "stats": self.echo_suppressor.get_stats()
            }
        }
        # Merge FeatureManager status (timers, intercom, notifications, alerts)
        if self.features:
            status.update(self.features.get_status())
        return status

    # Timer, intercom, notification, and alert methods have been moved to
    # FeatureManager (src/pipeline/feature_manager.py).
    # Access them via self.features.<method> or pipeline.features.<method>.
