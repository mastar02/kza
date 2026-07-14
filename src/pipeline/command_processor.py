"""
Command Processor Module
Procesa comandos: STT, identificación de speaker, clasificación de intent.
"""

import asyncio
import logging
import time

import numpy as np

from dataclasses import dataclass, field

from src.stt.whisper_fast import STTResult


@dataclass
class ProcessedCommand:
    """Typed result from CommandProcessor.process_command()."""
    text: str
    user: object | None = None  # User from user_manager or None
    emotion: object | None = None  # EmotionResult or None
    speaker_confidence: float = 0.0
    stt_confidence: STTResult | None = None  # confianza del STT (None = desconocida)
    timings: dict = field(default_factory=dict)
    success: bool = False


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
        shadow_stt=None,
    ):
        """
        Inicializar CommandProcessor.

        Args:
            stt: STT service (ej: GoogleSTT, WhisperSTT)
            speaker_identifier: Speaker identification service
            user_manager: User management service
            emotion_detector: EmotionDetector para análisis de emociones
            sample_rate: Sample rate de audio (Hz)
            shadow_stt: STT secundario opcional para A/B en vivo. Si se provee,
                se transcribe el MISMO audio en paralelo (fire-and-forget, NO
                bloquea la respuesta ni cambia el texto usado) y se loguea
                ``[STT-shadow]`` con ambas transcripciones y latencias. None =
                off (default). Se usa para comparar Parakeet vs Whisper sobre
                tráfico real antes de decidir un flip del STT primario.
        """
        self.stt = stt
        self.speaker_id = speaker_identifier
        self.user_manager = user_manager
        self.emotion_detector = emotion_detector
        self.sample_rate = sample_rate
        self.shadow_stt = shadow_stt
        # Referencias fuertes a las tasks de shadow en vuelo: sin esto el GC
        # de asyncio puede cancelar una task fire-and-forget a mitad de camino.
        self._shadow_tasks: set = set()

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

        # Cargar STT shadow si está configurado (A/B en vivo)
        if self.shadow_stt is not None and hasattr(self.shadow_stt, 'load'):
            self.shadow_stt.load()
            logger.info("Shadow STT loaded (A/B en vivo)")

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
        use_parallel: bool = True,
        pretranscribed_text: str | None = None,
    ) -> ProcessedCommand:
        """
        Procesar comando completo (STT + Speaker ID + Emotion).

        Args:
            audio: Audio del comando
            use_parallel: Usar procesamiento paralelo para STT + Speaker ID
            pretranscribed_text: Si viene, saltea el STT y usa este texto.
                Lo setea el early-dispatch worker (ya transcribió en el loop
                de streaming del wake). Speaker ID + emotion se hacen igual
                sobre el audio.

        Returns:
            ProcessedCommand with text, user, emotion, timings, success
        """
        result = ProcessedCommand(text="")
        pipeline_start = time.perf_counter()

        if pretranscribed_text is not None:
            # Shortcut: texto ya venía. Corremos sólo speaker_id + emotion.
            result.timings["stt"] = 0.0
            result.timings["stt_skipped"] = 1
            if self.speaker_id and self.user_manager:
                user, confidence, spk_ms = self._identify_speaker(audio)
                if user is not None:
                    result.user = user
                    result.speaker_confidence = confidence
                result.timings["speaker_id"] = spk_ms
            if self.emotion_detector:
                try:
                    emotion = self.emotion_detector.detect(audio, self.sample_rate)
                    if emotion:
                        result.emotion = emotion
                        result.timings["emotion"] = emotion.processing_time_ms
                except Exception as e:
                    logger.debug(f"Emotion detection skipped: {e}")
            text = pretranscribed_text
        elif use_parallel and (self.speaker_id or self.emotion_detector):
            text, stt_ms, speaker_result, emotion_result, stt_res = await self._process_parallel(audio)
            result.timings["stt"] = stt_ms
            result.stt_confidence = stt_res
            if speaker_result:
                user, confidence, spk_ms = speaker_result
                if user is not None:
                    result.user = user
                    result.speaker_confidence = confidence
                result.timings["speaker_id"] = spk_ms
            if emotion_result:
                result.emotion = emotion_result
                result.timings["emotion"] = emotion_result.processing_time_ms
        else:
            stt_res = self.stt.transcribe_with_confidence(audio, self.sample_rate)
            text = stt_res.text
            result.stt_confidence = stt_res
            result.timings["stt"] = stt_res.elapsed_ms

            # Sequential speaker ID (fallback)
            if self.speaker_id and self.user_manager:
                user, confidence, spk_ms = self._identify_speaker(audio)
                if user is not None:
                    result.user = user
                    result.speaker_confidence = confidence
                result.timings["speaker_id"] = spk_ms

        result.text = text
        result.success = bool(text.strip())

        if result.success:
            self._current_user = result.user
            self._current_emotion = result.emotion

        result.timings["total"] = (time.perf_counter() - pipeline_start) * 1000

        logger.info(
            f"[CommandProcessor] Text='{text[:50]}' | "
            f"User={result.user.name if result.user else 'unknown'} | "
            f"Emotion={result.emotion.emotion if result.emotion else 'none'}"
        )

        # A/B en vivo: transcribir el mismo audio con el STT shadow SIN bloquear
        # la respuesta (fire-and-forget). Solo loguea la comparación.
        if self.shadow_stt is not None:
            self._dispatch_shadow(audio, text, result.timings.get("stt", 0.0))

        return result

    def _dispatch_shadow(self, audio: np.ndarray, primary_text: str, primary_ms: float) -> None:
        """Lanzar la transcripción shadow en background (no bloquea el return)."""
        try:
            task = asyncio.create_task(
                self._run_shadow(audio, primary_text, primary_ms)
            )
            self._shadow_tasks.add(task)
            task.add_done_callback(self._shadow_tasks.discard)
        except RuntimeError:
            # Sin event loop corriendo (p. ej. tests sync): shadow es opcional.
            pass

    async def _run_shadow(self, audio: np.ndarray, primary_text: str, primary_ms: float) -> None:
        """Transcribir con el STT shadow y loguear la comparación A/B."""
        try:
            loop = asyncio.get_running_loop()
            r = await loop.run_in_executor(
                None, self.shadow_stt.transcribe_with_confidence, audio, self.sample_rate
            )
            logger.info(
                f"[STT-shadow] primary={primary_text[:60]!r} primary_ms={primary_ms:.0f} "
                f"shadow={r.text[:60]!r} shadow_ms={r.elapsed_ms:.0f}"
            )
        except Exception as e:  # el shadow jamás debe afectar el command path
            logger.debug(f"[STT-shadow] error (ignorado): {e}")

    async def _process_parallel(self, audio: np.ndarray) -> tuple[str, float, tuple | None, object | None, STTResult | None]:
        """
        Procesar STT, Speaker ID y Emotion en paralelo REAL con asyncio.gather().

        Returns:
            Tuple[text, stt_ms, speaker_result, emotion_result, stt_res]
        """
        loop = asyncio.get_running_loop()
        t_parallel = time.perf_counter()

        # Crear todas las tasks
        stt_task = loop.run_in_executor(
            None, self.stt.transcribe_with_confidence, audio, self.sample_rate
        )

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
        stt_res = results[0] if not isinstance(results[0], Exception) else None
        if stt_res is not None:
            text, stt_ms = stt_res.text, stt_res.elapsed_ms
        else:
            text, stt_ms = "", 0

        if self.speaker_id and self.user_manager and not isinstance(results[1], Exception):
            speaker_result = results[1]  # tuple: (User|None, confidence, timing_ms)
        else:
            speaker_result = None

        emotion_result = (
            results[2] if self.emotion_detector
            and not isinstance(results[2], Exception) and results[2] is not None
            else None
        )

        logger.debug(f"[Parallel GATHER {parallel_ms:.0f}ms] STT + Speaker ID + Emotion completed")

        return text, stt_ms, speaker_result, emotion_result, stt_res

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

    def _identify_speaker(self, audio: np.ndarray) -> tuple:
        """
        Identificar el usuario que está hablando.

        Returns:
            Tuple of (User|None, confidence, timing_ms)
        """
        if self.speaker_id is None or self.user_manager is None:
            return None, 0.0, 0.0

        # Obtener embeddings con cache (más rápido que fetch cada vez)
        registered_embeddings = self._get_cached_embeddings()

        if not registered_embeddings:
            logger.debug("No hay usuarios con voz registrada")
            return None, 0.0, 0.0

        # Identificar speaker
        t_start = time.perf_counter()
        match = self.speaker_id.identify(audio, registered_embeddings)
        timing_ms = (time.perf_counter() - t_start) * 1000

        if match.is_known and match.user_id:
            user = self.user_manager.get_user(match.user_id)
            if user:
                self.user_manager.update_last_seen(match.user_id)
                logger.debug(f"Speaker identificado: {user.name} (conf={match.confidence:.2f})")
                return user, match.confidence, timing_ms

        logger.debug(f"Speaker desconocido (best conf={match.confidence:.2f})")
        return None, 0.0, timing_ms

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
