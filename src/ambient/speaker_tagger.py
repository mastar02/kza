"""SpeakerTagger — etiqueta de hablante por utterance (ECAPA en cuda:0).

Instancia de SpeakerIdentifier SEPARADA de la del command path (aislamiento).
embeddings_loader es un callable (DI) que devuelve {user_id: embedding} — en
producción lee los enrolados; en tests, un dict fijo. Best-effort: cualquier
error → ("unknown", 0.0), nunca propaga.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

import numpy as np

from src.core.owned_work import OwnedBlockingWork
from src.memory.producer_context import PRODUCER_CONTEXT, model_admission
from src.memory.prompt_context import SourcesInvalidated

logger = logging.getLogger(__name__)


class SpeakerTagger:
    """(speaker, confidence) para el audio mono de una utterance."""

    def __init__(
        self,
        identifier,
        embeddings_loader: Callable[[], dict[str, np.ndarray]],
        min_audio_s: float = 0.8,
        sample_rate: int = 16000,
        *, source_policy=None, voice_registry=None,
    ):
        self._identifier = identifier
        self._embeddings_loader = embeddings_loader
        self.min_audio_s = min_audio_s
        self.sample_rate = sample_rate
        self._work = OwnedBlockingWork()
        self._source_policy = source_policy
        self._inference_lock = threading.Lock()
        self._voice_registry = voice_registry
        self.review_managed = voice_registry is not None
        if voice_registry is not None:
            self._inference_lock = voice_registry.register_encoder(identifier)

    async def tag(self, audio_mono: np.ndarray, *, transcript: str = '', room_id: str | None = None) -> tuple[str, float]:
        """Etiquetar el hablante de una utterance.

        Best-effort: audio corto, sin enrolados, o cualquier error del
        identifier → ("unknown", 0.0). Nunca propaga excepciones.
        """
        try:
            if self._work.status()["closed"]:
                raise RuntimeError("speaker_tagger_closed")
            if len(audio_mono) < self.min_audio_s * self.sample_rate:
                return ("unknown", 0.0)
            if self._voice_registry is not None:
                self._require_live()
                self._voice_registry.observe(self._identifier,audio_mono,PRODUCER_CONTEXT.get().sources,
                    sample_rate=self.sample_rate,transcript=transcript,room_id=room_id)
                return ('unknown',0.)
            return await self._work.run(self._identify, audio_mono)
        finally:
            audio_mono = None

    def _identify(self, audio_mono):
        """Load profiles and identify inside the retained native scope."""
        with model_admission(self._inference_lock, self._require_live):
            return self._identify_locked(audio_mono)

    def _identify_locked(self, audio_mono):
        self._require_live()
        try:
            registered = self._embeddings_loader() or {}
        except Exception as e:
            logger.debug("Speaker profile loading deferred: %s", type(e).__name__)
            return ("unknown", 0.0)
        if not registered:
            return ("unknown", 0.0)  # sin enrolados: no gastar GPU
        self._require_live()
        try:
            match = self._identifier.identify(audio_mono, registered)
        except Exception as e:
            logger.debug("Speaker identification deferred: %s", type(e).__name__)
            return ("unknown", 0.0)
        if match.is_known and match.user_id:
            return (match.user_id, float(match.confidence))
        return ("unknown", float(match.confidence))

    def _require_live(self):
        if self._work.status()["closed"] or self._source_policy is not None and not self._source_policy.sources_available(PRODUCER_CONTEXT.get().sources):
            raise SourcesInvalidated("ambient_speaker_sources_invalidated")

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Expose actual profile reads and speaker inference without embeddings."""
        return tuple(self._work._tasks)

    def begin_close(self) -> asyncio.Task:
        """Reject new tagging and retain profile/inference completion."""
        return self._work.begin_close()
