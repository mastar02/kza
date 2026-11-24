"""Concurrent multi-worker STT pool for load-balancing across multiple rooms.

Each worker wraps an independent ``STTEngine`` (typically a ``FastWhisperSTT``
with its own pinned CUDA thread) so multiple rooms can transcribe in parallel
without sharing CTranslate2/CUDA thread state or blocking behind a single lock.

Capacity is strictly bounded to ``len(workers)``: callers acquire an idle
worker in round-robin order and wait on a condition variable when all workers
are busy, preventing GPU oversubscription on 8 GB cards.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from contextvars import Context
import logging
import threading
import time

import numpy as np

from src.stt.hybrid import STTEngine
from src.stt.whisper_fast import STTResult

logger = logging.getLogger(__name__)


class STTWorkerPool:
    """Bounded concurrent pool of independent ``STTEngine`` workers."""

    def __init__(
        self,
        workers: Sequence[STTEngine],
        *,
        owns_workers: bool = True,
    ):
        worker_tuple = tuple(workers)
        if not worker_tuple:
            raise ValueError("stt_worker_pool_requires_workers")
        self._workers: tuple[STTEngine, ...] = worker_tuple
        self._owns_workers = bool(owns_workers)
        n = len(self._workers)
        self._in_flight: list[int] = [0] * n
        self._requests: list[int] = [0] * n
        self._errors: list[int] = [0] * n
        self._total_ms: list[float] = [0.0] * n
        self._rr_cursor: int = 0
        self._closed: bool = False
        self._cond = threading.Condition()

    @property
    def max_concurrency(self) -> int:
        """Maximum concurrent transcriptions supported by this pool."""
        return len(self._workers)

    @property
    def workers(self) -> tuple[STTEngine, ...]:
        """Immutable tuple of underlying STT workers."""
        return self._workers

    def load(self) -> None:
        """Warm/load each underlying worker sequentially (safe CUDA init)."""
        for worker in self._workers:
            load_fn = getattr(worker, "load", None)
            if callable(load_fn):
                load_fn()

    def _acquire_worker_index(self) -> int:
        n = len(self._workers)
        with self._cond:
            while not self._closed:
                for offset in range(n):
                    idx = (self._rr_cursor + offset) % n
                    if self._in_flight[idx] == 0:
                        self._in_flight[idx] = 1
                        self._rr_cursor = (idx + 1) % n
                        return idx
                self._cond.wait()
            raise RuntimeError("stt_pool_closed")

    def _release_worker_index(self, idx: int, elapsed_ms: float, *, failed: bool) -> None:
        with self._cond:
            self._in_flight[idx] = max(0, self._in_flight[idx] - 1)
            self._requests[idx] += 1
            self._total_ms[idx] += elapsed_ms
            if failed:
                self._errors[idx] += 1
            self._cond.notify_all()

    def transcribe_with_confidence(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> STTResult:
        """Dispatch transcription to an idle worker with bounded concurrency."""
        idx = self._acquire_worker_index()
        worker = self._workers[idx]
        t0 = time.perf_counter()
        failed = False
        try:
            return worker.transcribe_with_confidence(audio, sample_rate)
        except Exception:
            failed = True
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._release_worker_index(idx, elapsed_ms, failed=failed)

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> tuple[str, float]:
        """Compatibility tuple API ``(text, elapsed_ms)``."""
        result = self.transcribe_with_confidence(audio, sample_rate)
        return result.text, result.elapsed_ms

    def health_snapshot(self) -> dict:
        """Return content-free pool telemetry (no audio, no transcripts)."""
        with self._cond:
            per_worker = []
            for idx, worker in enumerate(self._workers):
                reqs = self._requests[idx]
                avg_ms = round(self._total_ms[idx] / reqs, 1) if reqs else 0.0
                per_worker.append(
                    {
                        "index": idx,
                        "device": getattr(worker, "device", None),
                        "compute_type": getattr(worker, "compute_type", None),
                        "model": getattr(worker, "model_name", None),
                        "in_flight": self._in_flight[idx],
                        "requests": reqs,
                        "errors": self._errors[idx],
                        "avg_ms": avg_ms,
                    }
                )
            return {
                "workers": len(self._workers),
                "max_concurrency": len(self._workers),
                "closed": self._closed,
                "in_flight_total": sum(self._in_flight),
                "per_worker": per_worker,
            }

    def request_close(self) -> None:
        """Seal the pool immediately and wake any blocked callers."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if not self._owns_workers:
            return
        for worker in self._workers:
            fn = getattr(worker, "request_close", None)
            if callable(fn):
                fn()

    def begin_close(self) -> asyncio.Task:
        """Async retirement hook compatible with ``AmbientSTT`` and runtime closure."""
        self.request_close()

        async def _finish() -> None:
            if not self._owns_workers:
                return
            pending = []
            for worker in self._workers:
                begin = getattr(worker, "begin_close", None)
                if callable(begin):
                    pending.append(begin())
                else:
                    close = getattr(worker, "close", None)
                    if callable(close):
                        pending.append(asyncio.to_thread(close))
            if pending:
                await asyncio.gather(*pending)

        return asyncio.get_running_loop().create_task(_finish(), context=Context())

    def close(self) -> None:
        """Seal, drain in-flight calls, and close owned workers synchronously."""
        self.request_close()
        with self._cond:
            while any(self._in_flight):
                self._cond.wait()
        if not self._owns_workers:
            return
        errors: list[Exception] = []
        for worker in self._workers:
            close_fn = getattr(worker, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise errors[0]
