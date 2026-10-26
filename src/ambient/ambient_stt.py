"""AmbientSTT — STT del ambient path (instancia separada en cuda:0).

NO comparte el FastWhisperSTT del command path (cuda:1): aislamiento por
contrato del spec — el ambient path jamás compite con la latencia <300ms.
El builder lo construye con beam_size=1 (velocidad), initial_prompt=None
(el prompt domótico causaba copias literales en audio ambiguo) y
vad_filter=False (el VAD ya lo hizo el segmenter sobre el mic crudo).
"""
from __future__ import annotations

import asyncio
import threading

import numpy as np

from src.stt.whisper_fast import STTResult
from src.core.owned_work import OwnedBlockingWork
from src.memory.producer_context import PRODUCER_CONTEXT, model_admission
from src.memory.prompt_context import SourcesInvalidated


class AmbientSTT:
    """Transcribe la columna ASR de un RawSegment, sin bloquear el loop."""

    def __init__(self, stt, asr_col: int = 1, *, source_policy=None):
        """
        Args:
            stt: instancia FastWhisperSTT dedicada (cuda:0). DI por constructor.
            asr_col: columna del beam ASR en el audio multicanal (fw 6ch: 1).
        """
        self._stt = stt
        self.asr_col = asr_col
        self._work = OwnedBlockingWork()
        self._source_policy = source_policy
        self._inference_lock = threading.Lock()

    def asr_mono(self, audio: np.ndarray) -> np.ndarray:
        """Extraer la columna ASR mono de un audio potencialmente multicanal.

        No-op passthrough si `audio` ya es 1D. Método público para que
        consumidores externos (p. ej. el hook de wake textual en
        `transcriber.py`) obtengan la misma vista mono que usa internamente
        el STT, en vez de pasar el segmento crudo multicanal río abajo
        (rompe SpeakerID/ECAPA, que espera `(time,)`).

        Args:
            audio: Audio del segmento, 1D `(frames,)` o 2D `(frames, channels)`.

        Returns:
            Vista/copia 1D `(frames,)` — la columna `asr_col` si hay multicanal,
            o el mismo array si ya era mono.
        """
        return self._asr_mono(audio)

    def _asr_mono(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio
        col = self.asr_col if audio.shape[1] > self.asr_col else 0
        return np.ascontiguousarray(audio[:, col])

    async def transcribe(self, audio: np.ndarray) -> STTResult:
        """Transcripción con confianza, en thread (CTranslate2 es sync)."""
        try:
            return await self._work.run(self._invoke, audio)
        finally:
            audio = None

    def _invoke(self, audio):
        mono = None
        try:
            with model_admission(self._inference_lock, self._require_live):
                mono = self._asr_mono(audio)
                return self._stt.transcribe_with_confidence(mono)
        finally:
            audio = mono = None

    def _require_live(self):
        if self._work.status()["closed"] or self._source_policy is not None and not self._source_policy.sources_available(PRODUCER_CONTEXT.get().sources):
            raise SourcesInvalidated("ambient_asr_sources_invalidated")

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Expose actual ASR calls, including an engine's pinned native delegate."""
        return tuple(self._work._tasks)

    def begin_close(self) -> asyncio.Task:
        """Reject new ASR and retain its actual native exit."""
        return self._work.begin_close()
