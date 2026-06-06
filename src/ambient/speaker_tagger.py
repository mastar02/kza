"""SpeakerTagger — etiqueta de hablante por utterance (ECAPA en cuda:0).

Instancia de SpeakerIdentifier SEPARADA de la del command path (aislamiento).
embeddings_loader es un callable (DI) que devuelve {user_id: embedding} — en
producción lee los enrolados; en tests, un dict fijo. Best-effort: cualquier
error → ("unknown", 0.0), nunca propaga.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerTagger:
    """(speaker, confidence) para el audio mono de una utterance."""

    def __init__(
        self,
        identifier,
        embeddings_loader: Callable[[], dict[str, np.ndarray]],
        min_audio_s: float = 0.8,
        sample_rate: int = 16000,
    ):
        self._identifier = identifier
        self._embeddings_loader = embeddings_loader
        self.min_audio_s = min_audio_s
        self.sample_rate = sample_rate

    async def tag(self, audio_mono: np.ndarray) -> tuple[str, float]:
        """Etiquetar el hablante de una utterance.

        Best-effort: audio corto, sin enrolados, o cualquier error del
        identifier → ("unknown", 0.0). Nunca propaga excepciones.
        """
        if len(audio_mono) < self.min_audio_s * self.sample_rate:
            return ("unknown", 0.0)
        try:
            registered = self._embeddings_loader() or {}
        except Exception as e:
            logger.debug(f"SpeakerTagger embeddings_loader error: {e}")
            return ("unknown", 0.0)
        if not registered:
            return ("unknown", 0.0)  # sin enrolados: no gastar GPU
        try:
            match = await asyncio.to_thread(
                self._identifier.identify, audio_mono, registered
            )
        except Exception as e:
            logger.debug(f"SpeakerTagger identify error (best-effort): {e}")
            return ("unknown", 0.0)
        if match.is_known and match.user_id:
            return (match.user_id, float(match.confidence))
        return ("unknown", float(match.confidence))
