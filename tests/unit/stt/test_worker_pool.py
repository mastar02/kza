"""Unit tests for STTWorkerPool, multi-worker create_stt, and parallel AmbientSTT."""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ambient.ambient_stt import AmbientSTT
from src.stt.hybrid import HybridVetoSTT, STTEngine
from src.stt.whisper_fast import FastWhisperSTT, STTResult, create_stt
from src.stt.worker_pool import STTWorkerPool


class _TimedFakeSTT:
    def __init__(self, label: str, delay_s: float = 0.08, *, device: str = "cuda:0",
                 compute_type: str = "int8_float16", fail: bool = False):
        self.label = label
        self.delay_s = delay_s
        self.device = device
        self.compute_type = compute_type
        self.model_name = "large-v3"
        self.fail = fail
        self.calls = 0
        self.loaded = False
        self.closed = False
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.closed = True

    def transcribe_with_confidence(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        with self._lock:
            self.calls += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay_s)
            if self.fail:
                raise RuntimeError("worker_boom")
            return STTResult(text=f"ok:{self.label}", elapsed_ms=self.delay_s * 1000.0)
        finally:
            with self._lock:
                self.in_flight -= 1


def _audio() -> np.ndarray:
    return np.zeros(1600, dtype=np.float32)


def test_worker_pool_requires_non_empty_workers():
    with pytest.raises(ValueError, match="stt_worker_pool_requires_workers"):
        STTWorkerPool([])


def test_worker_pool_runs_two_rooms_in_parallel_and_bounds_concurrency():
    w0 = _TimedFakeSTT("w0", delay_s=0.12)
    w1 = _TimedFakeSTT("w1", delay_s=0.12)
    pool = STTWorkerPool([w0, w1])
    assert isinstance(pool, STTEngine)
    assert pool.max_concurrency == 2

    # 2 concurrent requests run simultaneously on w0 and w1 (~0.12s, not ~0.24s)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(pool.transcribe_with_confidence, _audio()) for _ in range(2)]
        results = [f.result() for f in futs]
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.22
    assert {r.text for r in results} == {"ok:w0", "ok:w1"}
    assert w0.max_in_flight == 1
    assert w1.max_in_flight == 1

    # 3 concurrent requests never oversubscribe either worker (max_in_flight stays 1)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(pool.transcribe_with_confidence, _audio()) for _ in range(3)]
        _ = [f.result() for f in futs]
    assert w0.max_in_flight == 1
    assert w1.max_in_flight == 1
    assert w0.calls + w1.calls == 5

    snap = pool.health_snapshot()
    assert snap["workers"] == 2
    assert snap["in_flight_total"] == 0
    assert sum(w["requests"] for w in snap["per_worker"]) == 5


def test_worker_pool_records_errors_and_seals_on_close():
    w0 = _TimedFakeSTT("w0", delay_s=0.01, fail=True)
    w1 = _TimedFakeSTT("w1", delay_s=0.01, fail=False)
    pool = STTWorkerPool([w0, w1], owns_workers=True)
    pool.load()
    assert w0.loaded and w1.loaded

    with pytest.raises(RuntimeError, match="worker_boom"):
        pool.transcribe_with_confidence(_audio())
    res = pool.transcribe_with_confidence(_audio())
    assert res.text == "ok:w1"

    snap = pool.health_snapshot()
    assert snap["per_worker"][0]["errors"] == 1
    assert snap["per_worker"][1]["errors"] == 0

    pool.close()
    assert w0.closed and w1.closed
    with pytest.raises(RuntimeError, match="stt_pool_closed"):
        pool.transcribe_with_confidence(_audio())


def test_ambient_stt_unlocks_parallel_rooms_with_worker_pool():
    w0 = _TimedFakeSTT("w0", delay_s=0.12)
    w1 = _TimedFakeSTT("w1", delay_s=0.12)
    pool = STTWorkerPool([w0, w1])
    ambient = AmbientSTT(stt=pool, asr_col=0)
    assert ambient.max_concurrency == 2

    async def run_both():
        t0 = time.perf_counter()
        a, b = await asyncio.gather(
            ambient.transcribe(np.zeros((1600, 2), dtype=np.float32)),
            ambient.transcribe(np.zeros((1600, 2), dtype=np.float32)),
        )
        return a, b, time.perf_counter() - t0

    a, b, elapsed = asyncio.run(run_both())
    assert elapsed < 0.22
    assert {a.text, b.text} == {"ok:w0", "ok:w1"}


def test_create_stt_builds_homogeneous_asymmetric_and_hybrid_pools(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "onnx_asr",
        types.SimpleNamespace(load_model=lambda name, **kw: MagicMock()),
    )

    pool = create_stt({
        "engine": "whisper",
        "model": "large-v3",
        "device": "cuda:0",
        "compute_type": "int8_float16",
        "workers": 2,
    })
    assert isinstance(pool, STTWorkerPool)
    assert pool.max_concurrency == 2
    assert all(isinstance(w, FastWhisperSTT) and w.compute_type == "int8_float16" for w in pool.workers)

    asym = create_stt({
        "engine": "whisper",
        "model": "large-v3",
        "device": "cuda:0",
        "worker_configs": [
            {"compute_type": "float16"},
            {"compute_type": "int8_float16"},
        ],
    })
    assert isinstance(asym, STTWorkerPool)
    assert [w.compute_type for w in asym.workers] == ["float16", "int8_float16"]

    hybrid = create_stt({
        "engine": "hybrid",
        "model": "large-v3",
        "device": "cuda:0",
        "compute_type": "int8_float16",
        "workers": 2,
    })
    assert isinstance(hybrid, HybridVetoSTT)
    assert hybrid.max_concurrency == 2
    assert isinstance(hybrid._primary, STTWorkerPool)
