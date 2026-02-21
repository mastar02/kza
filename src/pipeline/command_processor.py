"""
Command Processor Module
Procesa comandos: STT, identificación de speaker, clasificación de intent.
"""

import asyncio
import logging
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CommandProcessor:
    """
    Procesa comandos de audio: transcripción, speaker ID, intent classification.

    Responsabilidades:
    - STT (Speech-to-Text)
    - Identificación de speaker (user identification)
    - Clasificación de intent
    - Procesamiento paralelo de STT y speaker ID
    - Integración con EmotionDetector
    """

    def __init__(
        self,
        stt,
        speaker_identifier=None,
        user_manager=None,
        emotion_detector=None,
        sample_rate: int = 16000,
    ):
        """
        Inicializar CommandProcessor.

        Args:
            stt: STT service (ej: GoogleSTT, WhisperSTT)
            speaker_identifier: Speaker identification service
            user_manager: User management service
            emotion_detector: EmotionDetector para análisis de emociones
            sample_rate: Sample rate de audio (Hz)
        """
        self.stt = stt
        self.speaker_id = speaker_identifier
        self.user_manager = user_manager
        self.emotion_detector = emotion_detector
        self.sample_rate = sample_rate

        self._current_user = None
        self._current_emotion = None

        # Cache de embeddings para speaker ID (evita fetch repetido)
        self._embeddings_cache: dict[str, np.ndarray] = {}
        self._embeddings_cache_version: int = 0
        self._embeddings_cache_time: float = 0
        self._embeddings_cache_ttl: float = 60.0  # Refresh cada 60s máximo

    def load_models(self):
        """Cargar modelos necesarios."""
        logger.info("CommandProcessor: Cargando modelos...")

        # Cargar STT si es necesario
        if hasattr(self.stt, 'load'):
            self.stt.load()

        # Cargar speaker ID si es disponible
        if self.speaker_id and hasattr(self.speaker_id, 'load'):
            self.speaker_id.load()
            logger.info("Speaker identification loaded")

        # Cargar emotion detector si es disponible
        if self.emotion_detector and hasattr(self.emotion_detector, 'load'):
            self.emotion_detector.load()
            logger.info("Emotion detector loaded")

    async def process_command(
        self,
        audio: np.ndarray,
        use_parallel: bool = True
    ) -> dict:
        """
        Procesar comando completo (STT + Speaker ID + Emotion).

        Args:
            audio: Audio del comando
            use_parallel: Usar procesamiento paralelo para STT + Speaker ID

        Returns:
            {
                "text": str,
                "user": User object o None,
                "emotion": EmotionResult o None,
                "timings": dict,
                "success": bool
            }
        """
        result = {
            "text": "",
            "user": None,
            "emotion": None,
            "timings": {},
            "success": False
        }

        pipeline_start = time.perf_counter()

        # Procesamiento paralelo: STT + Speaker ID + Emotion
        if use_parallel and (self.speaker_id or self.emotion_detector):
            text, stt_ms, speaker_result, emotion_result = await self._process_parallel(audio)
            result["timings"]["stt"] = stt_ms
            if speaker_result:
                result["user"] = speaker_result
                result["timings"]["speaker_id"] = speaker_result.get("timing_ms", 0)
            if emotion_result:
                result["emotion"] = emotion_result
                result["timings"]["emotion"] = emotion_result.processing_time_ms
        else:
            # Procesamiento secuencial (fallback)
            t_stt = time.perf_counter()
            text, stt_ms = self.stt.transcribe(audio, self.sample_rate)
            result["timings"]["stt"] = stt_ms

            speaker_result = None
            emotion_result = None

        result["text"] = text
        result["success"] = bool(text.strip())

        if result["success"]:
            self._current_user = speaker_result
            self._current_emotion = emotion_result

        result["timings"]["total"] = (time.perf_counter() - pipeline_start) * 1000

        logger.info(
            f"[CommandProcessor] Text='{text[:50]}' | "
            f"User={speaker_result.name if speaker_result else 'unknown'} | "
            f"Emotion={emotion_result.emotion if emotion_result else 'none'}"
        )

        return result

    async def _process_parallel(self, audio: np.ndarray) -> Tuple[str, float, Optional[dict], Optional[object]]:
        """
        Procesar STT, Speaker ID y Emotion en paralelo REAL con asyncio.gather().

        Returns:
            Tuple[text, stt_ms, speaker_result, emotion_result]
        """
        loop = asyncio.get_running_loop()
        t_parallel = time.perf_counter()

        # Crear todas las tasks
        stt_task = loop.run_in_executor(None, self.stt.transcribe, audio, self.sample_rate)

        speaker_task = (
            loop.run_in_executor(None, self._identify_speaker, audio)
            if self.speaker_id and self.user_manager
            else asyncio.sleep(0)  # Placeholder que completa inmediatamente
        )

        emotion_task = (
            loop.run_in_executor(None, self.emotion_detector.detect, audio)
            if self.emotion_detector
            else asyncio.sleep(0)  # Placeholder que completa inmediatamente
        )

        # Ejecutar TODAS en paralelo con gather (no secuencial)
        results = await asyncio.gather(
            stt_task, speaker_task, emotion_task,
            return_exceptions=True
        )

        parallel_ms = (time.perf_counter() - t_parallel) * 1000

        # Procesar resultados
        stt_result = results[0] if not isinstance(results[0], Exception) else ("", 0)
        text, stt_ms = stt_result if isinstance(stt_result, tuple) else ("", 0)

        speaker_result = (
            results[1] if self.speaker_id and self.user_manager
            and not isinstance(results[1], Exception) and results[1] is not None
            else None
        )

        emotion_result = (
            results[2] if self.emotion_detector
            and not isinstance(results[2], Exception) and results[2] is not None
            else None
        )

        logger.debug(f"[Parallel GATHER {parallel_ms:.0f}ms] STT + Speaker ID + Emotion completed")

        return text, stt_ms, speaker_result, emotion_result

    def _get_cached_embeddings(self) -> dict[str, np.ndarray]:
        """
        Obtener embeddings con cache (evita fetch repetido cada comando).

        El cache se invalida si:
        - Han pasado más de _embeddings_cache_ttl segundos
        - El user_manager tiene una versión más nueva
        """
        current_time = time.time()

        # Verificar si el cache está válido
        cache_expired = (current_time - self._embeddings_cache_time) > self._embeddings_cache_ttl

        # Verificar versión del user_manager (si soporta versionado)
        manager_version = getattr(self.user_manager, '_version', 0)
        version_changed = manager_version != self._embeddings_cache_version

        if cache_expired or version_changed or not self._embeddings_cache:
            # Refresh cache
            self._embeddings_cache = self.user_manager.get_all_embeddings()
            self._embeddings_cache_time = current_time
            self._embeddings_cache_version = manager_version
            logger.debug(f"Embeddings cache refreshed: {len(self._embeddings_cache)} users")

        return self._embeddings_cache

    def invalidate_embeddings_cache(self):
        """Forzar invalidación del cache de embeddings (llamar al agregar/modificar usuarios)."""
        self._embeddings_cache = {}
        self._embeddings_cache_time = 0

    def _identify_speaker(self, audio: np.ndarray) -> Optional[dict]:
        """
        Identificar el usuario que está hablando.

        Returns:
            User object si se identifica, None si es desconocido
        """
        if self.speaker_id is None or self.user_manager is None:
            return None

        # Obtener embeddings con cache (más rápido que fetch cada vez)
        registered_embeddings = self._get_cached_embeddings()

        if not registered_embeddings:
            logger.debug("No hay usuarios con voz registrada")
            return None

        # Identificar speaker
        t_start = time.perf_counter()
        match = self.speaker_id.identify(audio, registered_embeddings)
        timing_ms = (time.perf_counter() - t_start) * 1000

        if match.is_known and match.user_id:
            user = self.user_manager.get_user(match.user_id)
            if user:
                self.user_manager.update_last_seen(match.user_id)
                logger.debug(f"Speaker identificado: {user.name} (conf={match.confidence:.2f})")
                return {
                    "user": user,
                    "confidence": match.confidence,
                    "timing_ms": timing_ms
                }

        logger.debug(f"Speaker desconocido (best conf={match.confidence:.2f})")
        return None

    def get_current_user(self):
        """Obtener usuario actual identificado."""
        return self._current_user

    def get_current_emotion(self):
        """Obtener emoción detectada en último comando."""
        return self._current_emotion

    def detect_emotion(self, audio: np.ndarray):
        """
        Detectar emoción en audio.

        Returns:
            EmotionResult o None
        """
        if self.emotion_detector is None:
            return None

        return self.emotion_detector.detect(audio)
