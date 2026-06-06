"""AmbientSTT — STT del ambient path (instancia separada en cuda:0).

NO comparte el FastWhisperSTT del command path (cuda:1): aislamiento por
contrato del spec — el ambient path jamás compite con la latencia <300ms.
El builder lo construye con beam_size=1 (velocidad), initial_prompt=None
(el prompt domótico causaba copias literales en audio ambiguo) y
vad_filter=False (el VAD ya lo hizo el segmenter sobre el mic crudo).
"""
from __future__ import annotations

import asyncio

import numpy as np

from src.stt.whisper_fast import STTResult


class AmbientSTT:
    """Transcribe la columna ASR de un RawSegment, sin bloquear el loop."""

    def __init__(self, stt, asr_col: int = 1):
        """
        Args:
            stt: instancia FastWhisperSTT dedicada (cuda:0). DI por constructor.
            asr_col: columna del beam ASR en el audio multicanal (fw 6ch: 1).
        """
        self._stt = stt
        self.asr_col = asr_col

    def _asr_mono(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 1:
            return audio
        col = self.asr_col if audio.shape[1] > self.asr_col else 0
        return np.ascontiguousarray(audio[:, col])

    async def transcribe(self, audio: np.ndarray) -> STTResult:
        """Transcripción con confianza, en thread (CTranslate2 es sync)."""
        mono = self._asr_mono(audio)
        return await asyncio.to_thread(self._stt.transcribe_with_confidence, mono)
