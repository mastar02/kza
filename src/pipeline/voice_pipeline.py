"""
Voice Pipeline Module
Thin orchestrator that wires together the 5 extracted components.

Components:
1. AudioLoop — audio capture, wake word, echo suppression, ambient detection
2. CommandProcessor — STT, speaker ID, emotion detection
3. RequestRouter — command routing (orchestrated + legacy paths)
4. ResponseHandler — TTS, streaming, zone routing
5. FeatureManager — timers, intercom, notifications, alerts, HA integration
"""

import asyncio
import logging

import numpy as np

from src.pipeline.audio_loop import AudioLoop
from src.pipeline.command_event import CommandEvent
from src.pipeline.command_processor import CommandProcessor
from src.pipeline.request_router import RequestRouter
from src.pipeline.response_handler import ResponseHandler
from src.pipeline.feature_manager import FeatureManager
from src.dashboard.live_event_bus import LiveEvent, LiveEventBus, LiveEventType

logger = logging.getLogger(__name__)
_PUBLISH_FAILURE_LOGGED = False


class VoicePipeline:
    """
    Pipeline completo de voz para domotica inteligente.

    Thin orchestrator that wires five extracted components and manages
    their lifecycle (start / run / stop).  All domain logic lives in
    the components themselves; VoicePipeline only coordinates them.
    """

    def __init__(
        self,
        audio_loop: AudioLoop,
        command_processor: CommandProcessor,
        request_router: RequestRouter,
        response_handler: ResponseHandler,
        feature_manager: FeatureManager,
        chroma_sync: object | None = None,
        memory_manager: object | None = None,
        orchestrator: object | None = None,
        event_bus: LiveEventBus | None = None,
    ):
        """
        Initialize VoicePipeline with pre-built components.

        Args:
            audio_loop: Audio capture and wake word detection loop.
            command_processor: STT, speaker ID, emotion detection.
            request_router: Command routing (orchestrated + legacy).
            response_handler: TTS, streaming, zone routing.
            feature_manager: Timers, intercom, notifications, alerts.
            chroma_sync: Optional ChromaDB sync for vector search.
            memory_manager: Optional conversation memory.
            orchestrator: Optional MultiUserOrchestrator.
        """
        # Core components
        self.audio_loop = audio_loop
        self.command_processor = command_processor
        self.request_router = request_router
        self.response_handler = response_handler
        self.features = feature_manager

        # Optional services
        self.chroma = chroma_sync
        self.memory = memory_manager
        self._orchestrator = orchestrator
        self._event_bus = event_bus

        # State
        self._running = False

    async def run(self) -> None:
        """Main listening loop — initialize services and delegate to audio_loop."""
        logger.info("Initializing pipeline...")

        # Load models
        self.command_processor.load_models()

        # Initialize vector DB
        if self.chroma:
            self.chroma.initialize()
            stats = self.chroma.get_stats()
            logger.info(f"Vector DB: {stats.get('commands_phrases', 0)} phrases")

        # Initialize memory
        if self.memory:
            self.memory.initialize()

        # Start orchestrator
        if self._orchestrator:
            try:
                await self._orchestrator.start()
            except Exception as e:
                logger.warning(f"Orchestrator start failed (non-fatal): {e}")

        # Start features (timers, intercom, notifications, alerts, HA)
        if self.features:
            try:
                await self.features.start()
                logger.info("FeatureManager started")
            except Exception as e:
                logger.warning(f"FeatureManager start failed (non-fatal): {e}")

        self._running = True
        logger.info("System ready")

        # Audio loop — blocks until stop
        if self.audio_loop:
            self.audio_loop.on_command(self.process_command)
            await self.audio_loop.start()
            await self.audio_loop.run()
        else:
            logger.warning("No AudioLoop configured — pipeline running without audio capture")

    async def stop(self) -> None:
        """Stop all components."""
        self._running = False

        for name, coro in [
            ("orchestrator", self._orchestrator.stop() if self._orchestrator else None),
            ("audio_loop", self.audio_loop.stop() if self.audio_loop else None),
            ("features", self.features.stop() if self.features else None),
        ]:
            if coro:
                try:
                    await coro
                except Exception as e:
                    logger.warning(f"Error stopping {name}: {e}")

        logger.info("Pipeline stopped")

    async def process_command(self, audio_or_event) -> dict:
        """
        Process a complete audio command.

        Delegates to RequestRouter if configured, otherwise returns an error.

        Args:
            audio_or_event: CommandEvent with room metadata, or raw np.ndarray.

        Returns:
            Dict with text, intent, action, response, success, latency_ms, user.
        """
        if self.request_router:
            result = await self.request_router.process_command(audio_or_event)
            await self._publish_turn_event(audio_or_event, result)
            return result
        return {"text": "", "success": False, "error": "No request router configured"}

    async def _publish_turn_event(self, audio_or_event, result: dict) -> None:
        """Publica un LiveEvent type=turn al bus si está configurado.

        Best-effort: errores se loguean a WARN una sola vez por proceso (para no
        spamear si el bus está roto) y NO propagan, así no se rompe el pipeline
        de voz si el dashboard cae. `result` se accede defensivamente porque
        request_router puede no completar todos los campos en paths de error.
        """
        if not self._event_bus:
            return
        global _PUBLISH_FAILURE_LOGGED
        try:
            zone = (
                getattr(audio_or_event, "room_id", None)
                if not isinstance(audio_or_event, np.ndarray)
                else None
            )
            if zone is None and not isinstance(audio_or_event, np.ndarray):
                zone = getattr(audio_or_event, "zone", None)
            payload = {
                "id": result.get("turn_id"),
                "user": result.get("user"),
                "zone": zone,
                "stt": result.get("text"),
                "intent": result.get("intent"),
                "tts": result.get("response"),
                "latency_ms": result.get("latency_ms"),
                "success": result.get("success", False),
                "path": result.get("path", "fast"),
            }
            await self._event_bus.publish(LiveEvent(type=LiveEventType.TURN, payload=payload))
        except (RuntimeError, AttributeError, TypeError, asyncio.QueueFull) as e:
            if not _PUBLISH_FAILURE_LOGGED:
                _PUBLISH_FAILURE_LOGGED = True
                logger.warning(
                    f"event_bus publish failed (suppressing future events): "
                    f"{type(e).__name__}: {e}"
                )

    async def test_from_text(self, text: str) -> dict:
        """
        Test pipeline directly with text (bypasses audio capture and STT).

        Args:
            text: Text to process as if transcribed by STT.

        Returns:
            Dict with text, intent, action, response, success.
        """
        result = {"text": text, "intent": None, "action": None, "response": "", "success": False}

        if not self.chroma:
            result["intent"] = "no_match"
            return result

        command = self.chroma.search_command(text)
        if command:
            result["intent"] = "domotics"
            result["action"] = command
            result["success"] = True
            result["response"] = command.get("description", "")
        else:
            result["intent"] = "no_match"

        return result
