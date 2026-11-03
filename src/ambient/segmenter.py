"""UtteranceSegmenter — Silero VAD chunked sobre mic crudo → RawSegment.

El XVF3800 post-DSP rompe a Silero (prob~0, visto 2026-06-04 en whisper_fast);
por eso el VAD del ambient path corre sobre la COLUMNA CRUDA (vad_col, default
2 = mic 0 del firmware 6ch). Con firmware 2ch la columna no existe y se degrada
a col 0 (peor señal para VAD, pero funcional).
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import Context
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


class _SileroLease:
    def __init__(self, owner):
        self.owner = owner
        self.active = True

class _SileroLifecycle:
    """Retain this adapter's real operations without owning its callers' data."""

    def _silero_initialize(self) -> None:
        self._silero_condition = threading.Condition()
        self._silero_local = threading.local()
        self._silero_active = 0
        self._silero_sealed = False
        self._silero_close_started = False
        self._silero_close_owner = None
        self._silero_close_done = threading.Event()
        self._silero_failure = None
        self._silero_closing = None

    def _silero_current(self):
        lease = getattr(self._silero_local, "lease", None)
        return lease if lease is not None and lease.owner is self and lease.active else None

    @contextmanager
    def _silero_admit(self):
        lease = self._silero_current()
        owned = lease is None
        if owned:
            with self._silero_condition:
                if self._silero_sealed:
                    raise RuntimeError("silero_closed")
                lease = _SileroLease(self)
                self._silero_active += 1
        previous = getattr(self._silero_local, "lease", None)
        self._silero_local.lease = lease
        try:
            yield
        finally:
            self._silero_local.lease = previous
            if owned:
                with self._silero_condition:
                    lease.active = False
                    self._silero_active -= 1
                    self._silero_condition.notify_all()

    def _silero_require_close_caller(self) -> None:
        if self._silero_current() is not None:
            raise RuntimeError("silero_close_from_worker")
        with self._silero_condition:
            if (self._silero_close_owner is threading.current_thread()
                    and not self._silero_close_done.is_set()):
                raise RuntimeError("silero_close_from_owner")

    def request_close(self) -> None:
        """Seal admission using memory only, without waiting for model locks."""
        with self._silero_condition:
            self._silero_sealed = True
            self._silero_condition.notify_all()

    def wait_closed(self) -> None:
        """Drain actual operations and drop only this adapter's Python references."""
        self._silero_require_close_caller()
        self.request_close()
        with self._silero_condition:
            own_close = not self._silero_close_started
            if own_close:
                self._silero_close_started = True
                self._silero_close_owner = threading.current_thread()
        if own_close:
            failure = None
            try:
                with self._silero_condition:
                    while self._silero_active:
                        self._silero_condition.wait()
                self._silero_release_resources()
            except BaseException:
                failure = "silero_close_failed"
            with self._silero_condition:
                self._silero_failure = failure
                self._silero_close_owner = None
                self._silero_close_done.set()
        else:
            self._silero_close_done.wait()
        if self._silero_failure is not None:
            raise RuntimeError(self._silero_failure)

    def close(self) -> None:
        """Synchronous retirement; async callers use begin_close or aclose."""
        self.wait_closed()

    def begin_close(self) -> asyncio.Task:
        """Return the sole retained owner; never cancel this owner directly."""
        self._silero_require_close_caller()
        loop = asyncio.get_running_loop()
        self.request_close()
        with self._silero_condition:
            if self._silero_closing is None:
                async def finish():
                    # Publish the task outside the condition even with eager tasks.
                    await asyncio.sleep(0)
                    await asyncio.to_thread(self.wait_closed)

                self._silero_closing = loop.create_task(finish(), context=Context())
                self._silero_closing.add_done_callback(
                    lambda done: None if done.cancelled() else done.exception(),
                    context=Context(),
                )
            return self._silero_closing

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


class _SileroPredictor(_SileroLifecycle):
    """Own one recurrent model; independent streams need separate predictors."""

    def __init__(self, torch_module):
        self._silero_initialize()
        self._model_lock = threading.RLock()
        self._torch = torch_module
        self._model = None
        self._utils = None
        self._ready = False

    def _load(self) -> None:
        with self._silero_admit():
            with self._model_lock:
                if self._ready:
                    return
                # Only Hub loading/eval shares this global cache exclusion.
                # Terminal retirement never acquires this global lock.
                with _SILERO_LOAD_LOCK:
                    self._model, self._utils = self._torch.hub.load(
                        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True,
                    )
                    self._model.eval()
                self._ready = True
                # The existing predictor never used the returned utility bundle.
                self._utils = None

    def __call__(self, chunk_mono: np.ndarray) -> float:
        with self._silero_admit():
            with self._model_lock:
                win = probs = None
                try:
                    if not self._ready:
                        raise RuntimeError("silero_not_loaded")
                    if chunk_mono.dtype != np.float32:
                        chunk_mono = chunk_mono.astype(np.float32)
                    probs = []
                    for i in range(0, len(chunk_mono) - 511, 512):
                        win = self._torch.from_numpy(chunk_mono[i : i + 512])
                        with self._torch.no_grad():
                            probs.append(float(self._model(win, 16000).item()))
                    if not probs:
                        logger.warning(f"Silero predict: chunk de {len(chunk_mono)} samples (<512) — tratado como silencio")
                        return 0.0
                    return max(probs)
                finally:
                    chunk_mono = win = probs = None

    def reset(self) -> None:
        with self._silero_admit():
            with self._model_lock:
                if not self._ready:
                    raise RuntimeError("silero_not_loaded")
                self._model.reset_states()

    def _silero_release_resources(self) -> None:
        with self._model_lock:
            self._model = None
            self._utils = None
            self._torch = None
            self._ready = False


def make_silero_predictor() -> Callable[[np.ndarray], float]:
    """Create a separately owned recurrent predictor, loaded before returning.

    The callable/reset contract and whole-chunk exclusion remain reusable until
    terminal close. Retiring it releases only this adapter's Python references.
    """
    import torch  # torch PRIMERO (regla orden imports CUDA del proyecto)

    predictor = _SileroPredictor(torch)
    failure = None
    exit_code = None
    try:
        try:
            predictor._load()
        except KeyboardInterrupt:
            failure = "interrupt"
        except SystemExit as error:
            failure = "exit"
            code = error.code
            exit_code = code if code is None or type(code) in (int, bool) else 1
            code = None
        except asyncio.CancelledError:
            failure = "cancel"
        except GeneratorExit:
            failure = "generator"
        except BaseException:
            failure = "failed"
        if failure is None:
            return predictor
        # Native admission/global/model locks have exited before cleanup.
        predictor.close()
    finally:
        predictor = torch = None
    # Raise fresh values after the SDK handler, without native cause/context.
    if failure == "interrupt":
        raise KeyboardInterrupt
    if failure == "exit":
        raise SystemExit(exit_code)
    if failure == "cancel":
        raise asyncio.CancelledError
    if failure == "generator":
        raise GeneratorExit
    raise RuntimeError("silero_initialization_failed")


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
        *, owns_vad_predictor: bool = False,
    ):
        self._vad_predict = vad_predict
        self._owns_vad_predictor = owns_vad_predictor
        self._closed = False
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

    def close(self) -> None:
        """Retire an owned predictor on the serialized segmenter worker."""
        self._closed = True
        predictor = self._vad_predict
        if self._owns_vad_predictor and issubclass(type(predictor), _SileroPredictor):
            predictor.close()
        self._vad_predict = None
        self.reset()

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
        if self._closed:
            raise RuntimeError('ambient_segmenter_closed')
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
