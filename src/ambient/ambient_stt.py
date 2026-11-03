"""AmbientSTT — STT del ambient path (instancia separada en cuda:0).

NO comparte el FastWhisperSTT del command path (cuda:1): aislamiento por
contrato del spec — el ambient path jamás compite con la latencia <300ms.
El builder lo construye con beam_size=1 (velocidad), initial_prompt=None
(el prompt domótico causaba copias literales en audio ambiguo) y
vad_filter=False (el VAD ya lo hizo el segmenter sobre el mic crudo).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import threading
from uuid import uuid4

import numpy as np

from src.stt.whisper_fast import STTResult
from src.core.capture_opening import CaptureOpening
from src.core.owned_work import OwnedBlockingWork
from src.evaluation.provenance import CaptureLane, CaptureProvenance, FrameInterval
from src.memory.producer_context import PRODUCER_CONTEXT, model_admission
from src.memory.prompt_context import SourcesInvalidated


def _validate_stt_scalars(text, elapsed_ms, no_speech_prob, avg_logprob, compression_ratio):
    if (type(text) is not str or type(elapsed_ms) not in (int, float)
            or not math.isfinite(elapsed_ms) or elapsed_ms < 0
            or any(value is not None and (type(value) not in (int, float) or not math.isfinite(value))
                   for value in (no_speech_prob, avg_logprob, compression_ratio))):
        raise ValueError('invalid_ambient_stt_result')


def _stt_scalars(result):
    """Copy only closed immutable values, never an engine result's attributes bag."""
    values = None
    invalid = False
    try:
        if type(result) is not STTResult:
            raise ValueError('invalid_ambient_stt_result')
        values = (result.text, result.elapsed_ms, result.no_speech_prob,
                  result.avg_logprob, result.compression_ratio)
        _validate_stt_scalars(*values)
    except (ValueError, TypeError, AttributeError, OverflowError):
        invalid = True
    finally:
        result = None
    if invalid:
        values = None
        raise ValueError('invalid_ambient_stt_result')
    return values


@dataclass(frozen=True, slots=True, repr=False)
class AmbientSTTResult:
    """Five immutable engine scalars and selection; no engine object or extras."""

    text: str
    elapsed_ms: float
    no_speech_prob: float | None
    avg_logprob: float | None
    compression_ratio: float | None
    selected_channel: int
    source_channels: int
    frame_count: int
    provenance: CaptureProvenance | None = None
    span: FrameInterval | None = None

    def __post_init__(self):
        _validate_stt_scalars(self.text, self.elapsed_ms, self.no_speech_prob,
                              self.avg_logprob, self.compression_ratio)
        if (type(self.selected_channel) is not int
                or type(self.source_channels) is not int or not 1 <= self.source_channels <= 64
                or not 0 <= self.selected_channel < self.source_channels
                or type(self.frame_count) is not int or self.frame_count < 1):
            raise ValueError('invalid_ambient_stt_selection')
        if self.provenance is None:
            if self.span is not None:
                raise ValueError('invalid_ambient_stt_selection')
        else:
            if type(self.provenance) is not CaptureProvenance or type(self.span) is not FrameInterval:
                raise ValueError('invalid_ambient_stt_selection')
            CaptureProvenance.__post_init__(self.provenance)
            FrameInterval.__post_init__(self.span)
            if (self.provenance.source_channels != self.source_channels
                    or self.provenance.selected_channel != self.selected_channel
                    or self.provenance.sample_rate_hz != 16000
                    or self.provenance.frame_count != self.frame_count
                    or self.span != FrameInterval(0, self.frame_count)):
                raise ValueError('invalid_ambient_stt_selection')

    def mono_from(self, audio: np.ndarray) -> np.ndarray:
        """Use the actual native selection, even if configuration has changed."""
        self.__post_init__()
        if (type(audio) is not np.ndarray or audio.ndim not in (1, 2)
                or audio.shape[0] != self.frame_count
                or (1 if audio.ndim == 1 else audio.shape[1]) != self.source_channels
                or type(self.selected_channel) is not int
                or not 0 <= self.selected_channel < self.source_channels):
            raise ValueError('ambient_stt_selection_changed')
        return audio if audio.ndim == 1 else np.ascontiguousarray(audio[:, self.selected_channel])


class AmbientSTT:
    """Transcribe la columna ASR de un RawSegment, sin bloquear el loop."""

    capture_metadata = True

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

        No-op passthrough si `audio` ya es 1D. Este selector conserva la API
        anterior y usa la configuración actual. Para reutilizar el PCM de una
        transcripción terminada, usar AmbientSTTResult.mono_from: allí el canal
        ya está fijado y no vuelve a consultar asr_col.

        Args:
            audio: Audio del segmento, 1D `(frames,)` o 2D `(frames, channels)`.

        Returns:
            Vista/copia 1D `(frames,)` — la columna `asr_col` si hay multicanal,
            o el mismo array si ya era mono.
        """
        return self._asr_mono(audio)

    def _asr_mono(self, audio: np.ndarray) -> np.ndarray:
        return self._select_asr(audio)[0]

    def _select_asr(self, audio: np.ndarray) -> tuple[np.ndarray, int, int]:
        if type(audio) is not np.ndarray or audio.ndim not in (1, 2) or not audio.shape[0]:
            raise ValueError('invalid_ambient_stt_pcm')
        if audio.ndim == 1:
            return audio, 0, 1
        channels, requested = audio.shape[1], self.asr_col
        if not channels or type(requested) is not int or requested < 0:
            raise ValueError('invalid_ambient_stt_channel')
        col = requested if requested < channels else 0
        return np.ascontiguousarray(audio[:, col]), col, channels

    async def transcribe(self, audio: np.ndarray, *, opening: CaptureOpening | None = None,
                         segment_generation: int | None = None) -> AmbientSTTResult:
        """Transcripción con confianza, en thread (CTranslate2 es sync)."""
        try:
            return await self._work.run(self._invoke, audio, opening, segment_generation)
        finally:
            audio = opening = segment_generation = None

    def _invoke(self, audio, opening=None, segment_generation=None):
        mono = result = scalars = provenance = span = None
        try:
            with model_admission(self._inference_lock, self._require_live):
                mono, selected, channels = self._select_asr(audio)
                if opening is not None:
                    if (type(opening) is not CaptureOpening or audio.dtype != np.float32
                            or opening.sample_rate_hz != 16000 or opening.channels != channels):
                        raise ValueError('ambient_stt_opening_mismatch')
                    if segment_generation is not None:
                        provenance = opening.project(lane=CaptureLane.AMBIENT,
                            segment_generation=segment_generation, selected_channel=selected,
                            frame_count=mono.shape[0], input_ref=uuid4().hex)
                        span = FrameInterval(0, mono.shape[0])
                result = self._stt.transcribe_with_confidence(mono)
                self._require_live()
                scalars = _stt_scalars(result)
                result = None
                return AmbientSTTResult(*scalars, selected, channels, mono.shape[0], provenance, span)
        finally:
            audio = mono = opening = segment_generation = result = scalars = provenance = span = None

    def _require_live(self):
        if self._work.status()["closed"] or self._source_policy is not None and not self._source_policy.sources_available(PRODUCER_CONTEXT.get().sources):
            raise SourcesInvalidated("ambient_asr_sources_invalidated")

    def snapshot_operations(self) -> tuple[asyncio.Task, ...]:
        """Expose actual ASR calls, including an engine's pinned native delegate."""
        return tuple(self._work._tasks)

    def begin_close(self) -> asyncio.Task:
        """Reject new ASR and retain its actual native exit."""
        return self._work.begin_close()
