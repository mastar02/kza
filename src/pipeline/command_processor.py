"""
Command Processor Module
Procesa comandos: STT, identificación de speaker, clasificación de intent.
"""

import asyncio
import logging
import time

import numpy as np

from dataclasses import dataclass, field


@dataclass
class ProcessedCommand:
    """Typed result from CommandProcessor.process_command()."""
    text: str
    user: object | None = None  # User from user_manager or None
    emotion: object | None = None  # EmotionResult or None
    speaker_confidence: float = 0.0
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
        self._deferred_speaker_tasks: set[asyncio.Task] = set()

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
        use_parallel: bool = True,
        pretranscribed_text: str | None = None,
        await_speaker_id: bool = True,
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
            await_speaker_id: Si False (fast path domótica), NO espera el
                speaker ID — transcribe, retorna con user=None, y corre el
                speaker en background populando _current_user al terminar.
                La acción HA no necesita identidad. voice_auth lo espera aparte.

        Returns:
            ProcessedCommand with text, user, emotion, timings, success
        """
        result = ProcessedCommand(text="")
        pipeline_start = time.perf_counter()

        if pretranscribed_text is not None:
            # Shortcut: texto ya venía. Corremos sólo speaker_id + emotion.
            speaker_deferred = False
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
            speaker_deferred = (
                not await_speaker_id
                and self.speaker_id is not None
                and self.user_manager is not None
            )
            text, stt_ms, speaker_result, emotion_result = await self._process_parallel(
                audio, defer_speaker=speaker_deferred
            )
            result.timings["stt"] = stt_ms
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
            speaker_deferred = False
            text, stt_ms = self.stt.transcribe(audio, self.sample_rate)
            result.timings["stt"] = stt_ms

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
            if not speaker_deferred:
                self._current_user = result.user
            self._current_emotion = result.emotion

        result.timings["total"] = (time.perf_counter() - pipeline_start) * 1000

        logger.info(
            f"[CommandProcessor] Text='{text[:50]}' | "
            f"User={result.user.name if result.user else 'unknown'} | "
            f"Emotion={result.emotion.emotion if result.emotion else 'none'}"
        )

        return result

    async def _process_parallel(
        self, audio: np.ndarray, defer_speaker: bool = False
    ) -> tuple[str, float, tuple | None, object | None]:
        """
        Procesar STT, Speaker ID y Emotion en paralelo REAL con asyncio.gather().

        Args:
            audio: Audio del comando
            defer_speaker: Si True, el speaker ID se lanza en background
                (deferred) y no bloquea el retorno. Ver _spawn_deferred_speaker_id.
                El llamador ya habrá computado este flag.

        Returns:
            Tuple[text, stt_ms, speaker_result, emotion_result]
        """
        loop = asyncio.get_running_loop()
        t_parallel = time.perf_counter()

        stt_task = loop.run_in_executor(None, self.stt.transcribe, audio, self.sample_rate)

        speaker_task = (
            loop.run_in_executor(None, self._identify_speaker, audio)
            if self.speaker_id and self.user_manager and not defer_speaker
            else asyncio.sleep(0)
        )

        emotion_task = (
            loop.run_in_executor(None, self.emotion_detector.detect, audio)
            if self.emotion_detector
            else asyncio.sleep(0)
        )

        results = await asyncio.gather(stt_task, speaker_task, emotion_task, return_exceptions=True)
        parallel_ms = (time.perf_counter() - t_parallel) * 1000

        stt_result = results[0] if not isinstance(results[0], Exception) else ("", 0)
        text, stt_ms = stt_result if isinstance(stt_result, tuple) else ("", 0)

        if defer_speaker:
            self._spawn_deferred_speaker_id(audio)
            speaker_result = None
        elif self.speaker_id and self.user_manager and not isinstance(results[1], Exception):
            speaker_result = results[1]  # tuple: (User|None, confidence, timing_ms)
        else:
            speaker_result = None

        emotion_result = (
            results[2] if self.emotion_detector
            and not isinstance(results[2], Exception) and results[2] is not None
            else None
        )

        logger.debug(
            f"[Parallel GATHER {parallel_ms:.0f}ms] STT + "
            f"{'Speaker(deferred) + ' if defer_speaker else 'Speaker ID + '}Emotion"
        )

        return text, stt_ms, speaker_result, emotion_result

    def _spawn_deferred_speaker_id(self, audio: np.ndarray) -> None:
        """Resolver speaker ID en background (fast path domótica).

        No bloquea el retorno de process_command. Cuando termina, popula
        self._current_user para el siguiente turno. Excepciones se loguean.
        """
        loop = asyncio.get_running_loop()

        async def _runner():
            try:
                result = await loop.run_in_executor(None, self._identify_speaker, audio)
                user, confidence, _ = result
                if user is not None:
                    self._current_user = user
            except Exception as e:
                logger.debug(f"Deferred speaker ID skipped: {e}")

        task = loop.create_task(_runner())
        self._deferred_speaker_tasks.add(task)
        task.add_done_callback(self._deferred_speaker_tasks.discard)

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
