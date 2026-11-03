"""
Speaker Identifier Module
Identifica quién está hablando usando embeddings de voz.

Usa modelos de speaker verification/identification para crear
"huellas de voz" únicas para cada usuario registrado.
"""

import asyncio
from contextlib import contextmanager
from contextvars import Context
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeakerMatch:
    """Resultado de identificación de speaker"""
    user_id: str | None
    confidence: float
    embedding: np.ndarray
    is_known: bool


class _SpeakerLease:
    def __init__(self, owner):
        self.owner = owner
        self.active = True


class _SpeakerLifecycle:
    """Retain this adapter's real operations without owning its callers' data."""

    def _speaker_initialize(self) -> None:
        self._speaker_condition = threading.Condition()
        self._speaker_local = threading.local()
        self._speaker_active = 0
        self._speaker_sealed = False
        self._speaker_close_started = False
        self._speaker_close_owner = None
        self._speaker_close_done = threading.Event()
        self._speaker_failure = None
        self._speaker_closing = None

    def _speaker_current(self):
        lease = getattr(self._speaker_local, "lease", None)
        return lease if lease is not None and lease.owner is self and lease.active else None

    @contextmanager
    def _speaker_admit(self):
        lease = self._speaker_current()
        owned = lease is None
        if owned:
            with self._speaker_condition:
                if self._speaker_sealed:
                    raise RuntimeError("speaker_closed")
                lease = _SpeakerLease(self)
                self._speaker_active += 1
        previous = getattr(self._speaker_local, "lease", None)
        self._speaker_local.lease = lease
        try:
            yield
        finally:
            self._speaker_local.lease = previous
            if owned:
                with self._speaker_condition:
                    lease.active = False
                    self._speaker_active -= 1
                    self._speaker_condition.notify_all()

    def _speaker_require_close_caller(self) -> None:
        if self._speaker_current() is not None:
            raise RuntimeError("speaker_close_from_worker")
        with self._speaker_condition:
            if (self._speaker_close_owner is threading.current_thread()
                    and not self._speaker_close_done.is_set()):
                raise RuntimeError("speaker_close_from_owner")

    def request_close(self) -> None:
        """Seal admission using memory only, without waiting for model locks."""
        with self._speaker_condition:
            self._speaker_sealed = True
            self._speaker_condition.notify_all()

    def wait_closed(self) -> None:
        """Drain actual operations and drop only this adapter's Python references."""
        self._speaker_require_close_caller()
        self.request_close()
        with self._speaker_condition:
            own_close = not self._speaker_close_started
            if own_close:
                self._speaker_close_started = True
                self._speaker_close_owner = threading.current_thread()
        if own_close:
            failure = None
            try:
                with self._speaker_condition:
                    while self._speaker_active:
                        self._speaker_condition.wait()
                self._speaker_release_resources()
            except BaseException:
                failure = "speaker_close_failed"
            with self._speaker_condition:
                self._speaker_failure = failure
                self._speaker_close_owner = None
                self._speaker_close_done.set()
        else:
            self._speaker_close_done.wait()
        if self._speaker_failure is not None:
            raise RuntimeError(self._speaker_failure)

    def close(self) -> None:
        """Synchronous retirement; async callers use begin_close or aclose."""
        self.wait_closed()

    def begin_close(self) -> asyncio.Task:
        """Return the sole retained owner; never cancel this owner directly."""
        self._speaker_require_close_caller()
        loop = asyncio.get_running_loop()
        self.request_close()
        with self._speaker_condition:
            if self._speaker_closing is None:
                async def finish():
                    # Publish the task outside the condition even with eager tasks.
                    await asyncio.sleep(0)
                    await asyncio.to_thread(self.wait_closed)

                self._speaker_closing = loop.create_task(finish(), context=Context())
                self._speaker_closing.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception(),
                    context=Context(),
                )
            return self._speaker_closing

    async def aclose(self) -> None:
        """Keep the real retirement owner through repeated observer cancellation."""
        closing = self.begin_close()
        cancelled = False
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                cancelled = True
        closing.result()
        if cancelled:
            raise asyncio.CancelledError


class SpeakerIdentifier(_SpeakerLifecycle):
    """
    Identifica speakers usando embeddings de voz.

    Arquitectura:
    1. Audio → Modelo de embeddings → Vector 192/256-dim
    2. Comparar vector con embeddings registrados
    3. Retornar usuario con mayor similitud (si supera umbral)

    Modelos soportados:
    - speechbrain/spkrec-ecapa-voxceleb (recomendado)
    - pyannote/embedding (alternativa)
    - resemblyzer (más ligero)
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        similarity_threshold: float = 0.75,
        sample_rate: int = 16000
    ):
        self.model_name = model_name
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.sample_rate = sample_rate

        self._model = None
        self._model_type = None
        self._embeddings_cache: dict[str, np.ndarray] = {}
        self._speaker_initialize()
        self._model_lock = threading.RLock()

    def load(self):
        """Cargar modelo de speaker embeddings"""
        with self._speaker_admit():
            with self._model_lock:
                if self._model is not None:
                    return
                logger.info(f"Cargando speaker model: {self.model_name}")
                start = time.time()

                if "speechbrain" in self.model_name:
                    self._load_speechbrain()
                elif "resemblyzer" in self.model_name.lower():
                    self._load_resemblyzer()
                else:
                    self._load_speechbrain()  # Default

                elapsed = time.time() - start
                logger.info(f"Speaker model cargado en {elapsed:.1f}s")

    def _load_speechbrain(self):
        """Cargar modelo SpeechBrain ECAPA-TDNN"""
        try:
            # Workaround: torchaudio 2.10 removed list_audio_backends()
            # which speechbrain 1.0.x calls on import
            import torchaudio
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["ffmpeg"]

            from speechbrain.inference.speaker import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=f"./models/speaker/{self.model_name.replace('/', '_')}",
                run_opts={"device": self.device}
            )
            self._model_type = "speechbrain"

        except ImportError:
            logger.warning("SpeechBrain no instalado, usando resemblyzer")
            self._load_resemblyzer()

    def _load_resemblyzer(self):
        """Cargar Resemblyzer (más ligero)"""
        try:
            from resemblyzer import VoiceEncoder

            self._model = VoiceEncoder(device=self.device)
            self._model_type = "resemblyzer"

        except ImportError:
            raise ImportError(
                "Necesitas instalar speechbrain o resemblyzer:\n"
                "  pip install speechbrain\n"
                "  # o\n"
                "  pip install resemblyzer"
            )

    def get_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extraer embedding de voz de un audio.

        Args:
            audio: Audio como numpy array (float32, mono)

        Returns:
            Vector de embedding (192 o 256 dimensiones)
        """
        with self._speaker_admit():
            with self._model_lock:
                if self._model is None:
                    self.load()

                # Asegurar formato correcto
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)

                # Normalizar si es necesario
                if np.abs(audio).max() > 1.0:
                    audio = audio / np.abs(audio).max()

                start = time.perf_counter()

                if self._model_type == "speechbrain":
                    import torch
                    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                    embedding = self._model.encode_batch(audio_tensor)
                    embedding = embedding.squeeze().cpu().numpy()

                elif self._model_type == "resemblyzer":
                    embedding = self._model.embed_utterance(audio)

                else:
                    raise ValueError(f"Unknown model type: {self._model_type}")

                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.debug(f"Speaker embedding extracted in {elapsed_ms:.0f}ms")

                return embedding

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calcular similitud coseno entre dos embeddings.

        Returns:
            Similitud entre 0 y 1
        """
        # Normalizar
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Similitud coseno
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Convertir de [-1, 1] a [0, 1]
        return float((similarity + 1) / 2)

    def identify(
        self,
        audio: np.ndarray,
        registered_embeddings: dict[str, np.ndarray]
    ) -> SpeakerMatch:
        """
        Identificar speaker comparando con usuarios registrados.

        Args:
            audio: Audio del speaker a identificar
            registered_embeddings: {user_id: embedding} de usuarios registrados

        Returns:
            SpeakerMatch con el usuario identificado o None si desconocido
        """
        # Extraer embedding del audio actual
        with self._speaker_admit():
            current_embedding = self.get_embedding(audio)

            if not registered_embeddings:
                return SpeakerMatch(
                    user_id=None,
                    confidence=0.0,
                    embedding=current_embedding,
                    is_known=False
                )

            # Comparar con todos los registrados
            best_match = None
            best_similarity = 0.0

            for user_id, registered_embedding in registered_embeddings.items():
                similarity = self.compute_similarity(current_embedding, registered_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = user_id

            # Verificar si supera el umbral
            is_known = bool(best_similarity >= self.similarity_threshold)

            return SpeakerMatch(
                user_id=best_match if is_known else None,
                confidence=best_similarity,
                embedding=current_embedding,
                is_known=is_known
            )

    def verify(
        self,
        audio: np.ndarray,
        claimed_user_embedding: np.ndarray
    ) -> tuple[bool, float]:
        """
        Verificar si el audio corresponde a un usuario específico.

        Args:
            audio: Audio a verificar
            claimed_user_embedding: Embedding del usuario que dice ser

        Returns:
            (es_mismo_usuario, confianza)
        """
        with self._speaker_admit():
            current_embedding = self.get_embedding(audio)
            similarity = self.compute_similarity(current_embedding, claimed_user_embedding)

            is_match = bool(similarity >= self.similarity_threshold)

            return is_match, similarity

    def create_enrollment_embedding(
        self,
        audio_samples: list[np.ndarray]
    ) -> np.ndarray:
        """
        Crear embedding promedio de múltiples muestras de voz.
        Usar durante el enrollment para mayor precisión.

        Args:
            audio_samples: Lista de audios del mismo speaker

        Returns:
            Embedding promediado
        """
        with self._speaker_admit():
            if not audio_samples:
                raise ValueError("Se necesita al menos una muestra de audio")

            embeddings = [self.get_embedding(audio) for audio in audio_samples]

            # Promediar embeddings
            mean_embedding = np.mean(embeddings, axis=0)

            # Normalizar
            norm = np.linalg.norm(mean_embedding)
            if norm > 0:
                mean_embedding = mean_embedding / norm

            return mean_embedding

    def _speaker_release_resources(self) -> None:
        # No caller/peer model, output embedding or global CUDA cache is owned here.
        with self._model_lock:
            self._model = None
            self._model_type = None
            self._embeddings_cache.clear()
