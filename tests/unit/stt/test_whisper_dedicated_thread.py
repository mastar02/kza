"""Tests: thread dedicado por modelo en FastWhisperSTT (fix CUDA invalid argument 2026-06-07).

Root cause: dos WhisperModel CT2 en GPUs distintas (command path cuda:1 +
ambient path cuda:0) ejecutados desde el pool default compartido de
asyncio.to_thread / run_in_executor(None). CUDA y CTranslate2 mantienen
estado per-thread; el reuso de threads alternando devices produjo
`RuntimeError: CUDA failed with error invalid argument` intermitente
(12×/1.5h en prod, solo con ambos modelos activos — 2026-06-06).

Fix: cada FastWhisperSTT ejecuta load() y _transcribe_impl SIEMPRE en su
propio thread (ThreadPoolExecutor(max_workers=1)). Un modelo = un thread:
sin alternancia de device por thread y sin generate concurrente per-model.
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.stt.whisper_fast import FastWhisperSTT


@pytest.fixture
def thread_recorder(monkeypatch):
    """Stub de faster_whisper que registra el thread de cada llamada."""
    record = {"ctor_threads": [], "transcribe_threads": []}

    def fake_transcribe(*args, **kwargs):
        record["transcribe_threads"].append(threading.get_ident())
        seg = MagicMock()
        seg.text = "hola"
        seg.no_speech_prob = 0.1
        seg.avg_logprob = -0.3
        seg.compression_ratio = 1.0
        return iter([seg]), MagicMock()

    def fake_ctor(*args, **kwargs):
        record["ctor_threads"].append(threading.get_ident())
        inst = MagicMock()
        inst.transcribe.side_effect = fake_transcribe
        return inst

    fake_module = MagicMock()
    fake_module.WhisperModel = MagicMock(side_effect=fake_ctor)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    return record


def _stt(device: str = "cuda:1") -> FastWhisperSTT:
    return FastWhisperSTT(
        model="./models/whisper-v3-turbo",
        device=device,
        compute_type="float16",
        language="es",
    )


AUDIO = np.zeros(16000, dtype=np.float32)


def test_transcribe_runs_in_dedicated_thread(thread_recorder):
    # Dos llamadas → mismo thread del modelo, distinto del caller.
    stt = _stt()
    stt.transcribe_with_confidence(AUDIO)
    stt.transcribe_with_confidence(AUDIO)
    threads = thread_recorder["transcribe_threads"]
    assert len(threads) == 2
    assert threads[0] == threads[1], "cada modelo debe usar SIEMPRE su thread"
    assert threads[0] != threading.get_ident(), (
        "el generate no debe correr en el thread del caller"
    )


def test_load_runs_in_same_thread_as_transcribe(thread_recorder):
    # El modelo debe NACER en el mismo thread donde luego ejecuta.
    stt = _stt()
    stt.load()
    stt.transcribe_with_confidence(AUDIO)
    assert thread_recorder["ctor_threads"] == [
        thread_recorder["transcribe_threads"][0]
    ]


def test_two_instances_use_distinct_threads(thread_recorder):
    # command path (cuda:1) y ambient (cuda:0): threads dedicados DISTINTOS,
    # así un device jamás ejecuta en el thread del otro.
    a, b = _stt("cuda:1"), _stt("cuda:0")
    a.transcribe_with_confidence(AUDIO)
    b.transcribe_with_confidence(AUDIO)
    t_a, t_b = thread_recorder["transcribe_threads"]
    assert t_a != t_b


def test_concurrent_callers_serialize_on_model_thread(thread_recorder):
    # Reproduce el patrón de prod: N threads del pool default llamando al
    # mismo modelo → todas las ejecuciones aterrizan en el thread dedicado.
    stt = _stt()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(stt.transcribe_with_confidence, AUDIO) for _ in range(8)
        ]
        results = [f.result(timeout=10) for f in futures]
    assert all(r.text == "hola" for r in results)
    assert len(set(thread_recorder["transcribe_threads"])) == 1


def test_load_stays_idempotent_via_pinned_thread(thread_recorder):
    # La idempotencia del fix OOM 2026-05-29 debe sobrevivir al pinning.
    stt = _stt()
    stt.load()
    stt.load()
    stt.transcribe_with_confidence(AUDIO)
    assert len(thread_recorder["ctor_threads"]) == 1
