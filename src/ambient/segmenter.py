"""UtteranceSegmenter — Silero VAD chunked sobre mic crudo → RawSegment.

El XVF3800 post-DSP rompe a Silero (prob~0, visto 2026-06-04 en whisper_fast);
por eso el VAD del ambient path corre sobre la COLUMNA CRUDA (vad_col, default
2 = mic 0 del firmware 6ch). Con firmware 2ch la columna no existe y se degrada
a col 0 (peor señal para VAD, pero funcional).
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import replace
from typing import Callable

import numpy as np

from src.ambient.types import RawSegment
from src.core.capture_opening import CaptureOpening, coherent_opening

logger = logging.getLogger(__name__)
_SILERO_LOAD_LOCK = threading.Lock()


def make_silero_predictor() -> Callable[[np.ndarray], float]:
    """Create a stateful Silero predictor with serialized inference and reset.

    Callers processing separate audio streams must own separate predictors.
    The lock covers the entire chunk, including all native inference windows.
    """
    import torch  # torch PRIMERO (regla orden imports CUDA del proyecto)

    # Native room factories can start concurrently. Torch Hub mutates one
    # shared repository cache during first load; preserve serialized loading.
    with _SILERO_LOAD_LOCK:
        model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True,
        )
        model.eval()
    model_lock = threading.Lock()

    def predict(chunk_mono: np.ndarray) -> float:
        if chunk_mono.dtype != np.float32:
            chunk_mono = chunk_mono.astype(np.float32)
        with model_lock:
            probs = []
            for i in range(0, len(chunk_mono) - 511, 512):
                win = torch.from_numpy(chunk_mono[i : i + 512])
                with torch.no_grad():
                    probs.append(float(model(win, 16000).item()))
            if not probs:
                logger.warning(f"Silero predict: chunk de {len(chunk_mono)} samples (<512) — tratado como silencio")
                return 0.0
            return max(probs)

    def reset() -> None:
        with model_lock:
            model.reset_states()

    predict.reset = reset
    return predict


class UtteranceSegmenter:
    """Máquina de estados: silencio → voz → cola de silencio → RawSegment."""

    def __init__(
        self,
        vad_predict: Callable[[np.ndarray], float],
        sample_rate: int = 16000,
        vad_col: int = 2,
        speech_threshold: float = 0.5,
        close_silence_ms: int = 700,
        preroll_ms: int = 500,
        max_segment_s: float = 30.0,
        min_speech_ms: int = 300,
    ):
        self._vad_predict = vad_predict
        self.sample_rate = sample_rate
        self.vad_col = vad_col
        self.speech_threshold = speech_threshold
        self.close_silence_ms = close_silence_ms
        self.max_segment_s = max_segment_s
        self.min_speech_ms = min_speech_ms

        chunk_ms = 80.0  # CHUNK_SIZE=1280 @ 16kHz — mismo del pipeline
        self._preroll: deque[tuple[float, np.ndarray, bool, CaptureOpening | None]] = deque(
            maxlen=max(0, int(round(preroll_ms / chunk_ms)))
        )
        self._in_speech = False
        self._buf: list[tuple[float, np.ndarray, bool, CaptureOpening | None]] = []
        self._probs: list[float] = []  # prob Silero por chunk del buffer
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._col_warned = False
        self._state_since: float | None = None
        self._reset_pending = False
        # This known cohort is only a recurrent-state conflict fence. Unknown
        # PCM downgrades the emitted descriptor while retained, not forever.
        self._vad_opening: CaptureOpening | None = None
        self._segment_generation = 0

    @property
    def needs_reset(self) -> bool:
        """Expose unresolved recurrent-state cleanup without projecting input."""
        return self._reset_pending

    @property
    def oldest_timestamp(self) -> float | None:
        """Timestamp of the oldest input that may still affect VAD state."""
        return self._state_since

    def reset(self) -> None:
        """Discard speech, preroll and recurrent VAD state on its native owner."""
        self._preroll.clear()
        self._buf.clear()
        self._probs.clear()
        self._in_speech = False
        self._silence_ms = self._speech_ms = 0.0
        self._vad_opening = None
        reset_fn = getattr(self._vad_predict, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:
                self._reset_pending = True
                raise
        self._state_since = None
        self._reset_pending = False

    def _vad_column(self, chunk: np.ndarray) -> np.ndarray:
        col = self.vad_col
        if chunk.ndim == 1:
            return chunk
        if chunk.shape[1] <= col:
            if not self._col_warned:
                self._col_warned = True
                logger.warning(
                    f"vad_col={col} no existe (device de {chunk.shape[1]}ch) "
                    f"— fallback a col 0 (sin mic crudo, VAD degradado)"
                )
            col = 0
        return chunk[:, col]

    def _opening_snapshot(self, opening: CaptureOpening | None, chunk: np.ndarray) -> CaptureOpening | None:
        """Copy descriptor values only; PCM ownership remains unchanged."""
        if opening is None:
            return None
        if type(opening) is not CaptureOpening:
            raise ValueError('invalid_ambient_capture_opening')
        coherent_opening((opening,))
        channels = 1 if chunk.ndim == 1 else chunk.shape[1]
        if opening.sample_rate_hz != self.sample_rate or opening.channels != channels:
            raise ValueError('ambient_opening_pcm_mismatch')
        # CaptureOpening construction deep-copies its frozen DSP observation.
        return replace(opening)

    def feed(
        self, ts: float, chunk: np.ndarray, tts_active: bool = False,
        opening: CaptureOpening | None = None,
    ) -> list[RawSegment]:
        """Procesar un chunk; devolver los segmentos cerrados (0 o 1)."""
        if self._reset_pending:
            raise RuntimeError('ambient_segmenter_reset_pending')
        opening = self._opening_snapshot(opening, chunk)
        retained = (*self._buf, *self._preroll)
        _, conflict = coherent_opening((*[item[3] for item in retained], opening))
        if not conflict and opening is not None and self._vad_opening is not None:
            _, conflict = coherent_opening((self._vad_opening, opening))
        if conflict:
            # Preserve this valid packet, but never classify or concatenate it
            # with PCM/recurrent state from another known capture opening.
            self.reset()
        if opening is not None and self._vad_opening is None:
            self._vad_opening = opening
        chunk_ms = (chunk.shape[0] / self.sample_rate) * 1000
        if self._state_since is None:
            self._state_since = ts
        prob = self._vad_predict(self._vad_column(chunk))
        is_speech = prob >= self.speech_threshold
        closed: list[RawSegment] = []

        if not self._in_speech:
            if is_speech:
                self._in_speech = True
                self._buf = list(self._preroll)
                self._buf.append((ts, chunk, tts_active, opening))
                # vad_prob promedia los chunks evaluados in_speech; el
                # pre-roll (silencio previo) queda fuera a propósito.
                self._probs = [prob]
                self._preroll.clear()
                self._silence_ms = 0.0
                self._speech_ms = chunk_ms
            else:
                if self._preroll.maxlen:
                    self._preroll.append((ts, chunk, tts_active, opening))
            return closed

        # in_speech
        self._buf.append((ts, chunk, tts_active, opening))
        self._probs.append(prob)
        if is_speech:
            self._silence_ms = 0.0
            self._speech_ms += chunk_ms
        else:
            self._silence_ms += chunk_ms

        dur_s = sum(c.shape[0] for _, c, _, _ in self._buf) / self.sample_rate
        if self._silence_ms >= self.close_silence_ms or dur_s >= self.max_segment_s:
            seg = self._close(complete=self._silence_ms >= self.close_silence_ms)
            if seg is not None:
                closed.append(seg)
        return closed

    def _close(self, *, complete: bool = True) -> RawSegment | None:
        buf, self._buf = self._buf, []
        probs, self._probs = self._probs, []
        self._in_speech = False
        self._silence_ms = 0.0
        speech_ms, self._speech_ms = self._speech_ms, 0.0
        # Silero stateful: reset de la GRU en el límite de utterance (doc
        # oficial). Opcional vía getattr — los predictores fake no lo exponen.
        reset_fn = getattr(self._vad_predict, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
                self._state_since = None
                self._vad_opening = None
            except Exception as e:
                self._reset_pending = True
                logger.debug("Silero state reset deferred: %s", type(e).__name__)
        else:
            self._state_since = None
            self._vad_opening = None
        if not buf or speech_ms < self.min_speech_ms:
            return None  # blip: menos voz que min_speech_ms → descartar
        t0 = buf[0][0]
        last_ts, last_chunk, _, _ = buf[-1]
        t1 = last_ts + last_chunk.shape[0] / self.sample_rate
        opening, conflict = coherent_opening(item[3] for item in buf)
        if conflict:
            raise RuntimeError('ambient_segment_opening_conflict')
        audio = np.concatenate([c for _, c, _, _ in buf], axis=0)
        during_tts = any(t for _, _, t, _ in buf)
        vad_prob = float(sum(probs) / len(probs)) if probs else 0.0
        self._segment_generation += 1
        return RawSegment(
            t0=t0, t1=t1, audio=audio, during_tts=during_tts, vad_prob=vad_prob,
            complete=complete,
            input_opening=opening, segment_generation=self._segment_generation,
        )
