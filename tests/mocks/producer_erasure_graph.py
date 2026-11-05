"""Public application assembly with nominal speech owners and CPU-only leaves."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np

from src.main import build_application
from src.memory.fact_ledger import MemoryReviewer
from src.memory.producer_context import PRODUCER_CONTEXT, producer_context
from src.memory.producer_scopes import current_producer_scope
from src.stt.parakeet_stt import ParakeetSTT
from src.tts.piper_tts import KokoroTTS
from src.wakeword.detector import WakeWordDetector
from tests.integration.test_application_graph import leaves as application_leaves
from tests.mocks.application_graph import offline_config, offline_home
from tests.mocks.collector_erasure import configuration as collector_configuration
from tests.mocks.memory_sources import memory_source


ACTOR = MemoryReviewer("synthetic-producer-reviewer")
CHILD_ENV = "KZA_PRODUCER_UNKNOWN_CHILD"
CHILD_PREFIX = "KZA_PRODUCER_UNKNOWN_RECEIPT "


async def until(predicate, *, timeout=4):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.001)


class NativeGate:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.returned, self.completed = threading.Event(), threading.Event()
        self.thread = self.sources = self.scope = None

    def call(self, function, *args, **kwargs):
        self.thread = threading.current_thread()
        self.sources = PRODUCER_CONTEXT.get().sources
        scope = current_producer_scope()
        self.scope = None if scope is None else scope.snapshot()
        self.entered.set()
        try:
            assert self.release.wait(8), "fixture native gate was not released"
            result = function(*args, **kwargs)
            self.completed.set()
            return result
        finally:
            self.returned.set()


class CPURecognizer:
    def __init__(self, subject, unrelated):
        self.subject, self.unrelated = subject, unrelated
        self.calls = []

    def recognize(self, audio, *, language):
        assert audio.dtype == np.float32 and audio.shape == (16000,)
        value = float(audio[0])
        assert value in (.125, .25) and np.all(audio == value)
        label, gate = ("S", self.subject) if value == .125 else ("U", self.unrelated)
        self.calls.append(label)
        def result():
            return f"Synthetic {label} command"
        return result() if gate is None else gate.call(result)


def nominal_factories(monkeypatch, root, *, subject=None, unrelated=None):
    base = application_leaves.__wrapped__(monkeypatch, root)
    stt = ParakeetSTT()
    leaf = CPURecognizer(subject, unrelated)
    stt._model = leaf
    # No model initialization or waveform synthesis occurs during construction.
    tts = KokoroTTS(device="cpu")
    factories = replace(base, stt=lambda _: stt, tts=lambda _: tts,
        wake_detector=WakeWordDetector)
    return factories, SimpleNamespace(stt=stt, tts=tts, leaf=leaf)


def configuration(root, url, *, ambient=False, orchestrator=True, rooms=False):
    config = offline_config(root, url, rooms=rooms)
    data = collector_configuration(root)
    config["training"].update(data["training"])
    config["rooms"].update(data["rooms"])
    if rooms:
        config["rooms"]["synthetic-room"] = {"name": "Synthetic room",
            "mic_device_index": 0, "capture_channel": 0}
    config["ambient"].update(enabled=ambient, db_path=str(root / "ambient.db"),
        keep_audio={"enabled": False, "dir": str(root / "audio")})
    config["analytics"].update(events_db=str(root / "events.db"),
        suggestions_path=str(root / "suggestions.json"))
    config["observability"]["enabled"] = False
    config["orchestrator"].update(enabled=orchestrator,
        context={"persistence": {"enabled": False, "base_path": str(root / "contexts")}})
    return config


@asynccontextmanager
async def application(root, monkeypatch, *, gates=(), ambient=False, orchestrator=True, rooms=False):
    root = Path(root)
    factories, models = nominal_factories(monkeypatch, root,
        subject=gates[0] if gates else None, unrelated=gates[1] if len(gates) > 1 else None)
    async with offline_home() as (url, observations):
        stack = AsyncExitStack()
        try:
            graph = await build_application(configuration(root, url,
                ambient=ambient, orchestrator=orchestrator, rooms=rooms), shutdown=stack, factories=factories)
            assert graph.erasure_services.privacy is graph.privacy
            yield graph, models, observations
        finally:
            # Includes setup failures before yield, before any close can wait.
            for gate in gates:
                gate.release.set()
            await stack.aclose()


def source(graph):
    return memory_source(now=graph.privacy.ledger.clock())


def launch(graph, origin, value):
    with producer_context("synthetic-room", (origin,)):
        return asyncio.create_task(graph.pipeline.command_processor.process_command(
            np.full(16000, value, np.float32), use_parallel=False))


async def producer_step(graph, operation, *, version=1):
    def read():
        with graph.privacy.ledger._transaction() as db:
            row = db.execute("""SELECT * FROM erasure_steps WHERE operation_id=?
                AND participant='producers' AND version=?""", (operation["id"], version)).fetchone()
            return None if row is None else dict(row)
    return await graph.privacy.acall(read)


async def observed_step(graph, operation, predicate, *, version=1):
    async with asyncio.timeout(4):
        while True:
            await graph.erasure_services.reconcile()
            step = await producer_step(graph, operation, version=version)
            if step is not None and predicate(step):
                return step
            await asyncio.sleep(.01)


class CPUInputBackend:
    """SDK leaves; the real OwnedInputStream decides retirement."""
    def __init__(self, *, fail_close=False):
        self.fail_close = fail_close
        self.stream = None

    def InputStream(self, **options):
        backend = self

        class Stream:
            def __init__(self):
                self.pointer = object()
                self.starts = self.stops = self.closes = 0

            def start(self):
                self.starts += 1

            def stop(self, *, ignore_errors=True):
                assert ignore_errors is False
                self.stops += 1

            def close(self, *, ignore_errors=True):
                assert ignore_errors is False
                self.closes += 1
                self.pointer = None
                if backend.fail_close:
                    raise RuntimeError("synthetic native close did not confirm")

            def emit(self, audio):
                options["callback"](audio[:, None], len(audio), None, None)

        self.stream = Stream()
        return self.stream


class HeldAudio(np.ndarray):
    def __new__(cls, gate):
        audio = np.full(1280, .125, np.float32).view(cls)
        audio.gate = gate
        return audio

    def __array_finalize__(self, parent):
        if parent is not None:
            self.gate = getattr(parent, "gate", None)

    def copy(self, *args, **kwargs):
        return self.gate.call(np.asarray(self).copy, *args, **kwargs)


def unknown_child_command():
    """Run one contained pytest child, so conftest precedes product imports."""
    root = Path(__file__).resolve().parents[2]
    node = "tests/integration/test_producer_erasure_graph.py::test_native_close_unknown_cannot_be_acknowledged"
    env = {key: value for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "TMPDIR", "VIRTUAL_ENV", "SYSTEMROOT"}}
    env.update({CHILD_ENV: "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "WANDB_DISABLED": "true", "PY_COLORS": "0",
        "PYTHONNOUSERSITE": "1", "KZA_DASHBOARD_TOKEN": "synthetic-producer-reader"})
    process = subprocess.Popen([sys.executable, "-m", "pytest", "-q", "-s", node],
        cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    try:
        output, _ = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
        raise AssertionError("unknown-resource child exceeded containment budget\n" + output)
    assert process.wait() == 0, output
    documents = [json.loads(line[len(CHILD_PREFIX):]) for line in output.splitlines()
        if line.startswith(CHILD_PREFIX)]
    assert len(documents) == 1, output
    return documents[0]


def construction_sdk_guards(monkeypatch):
    """Trip SDK leaves if construction accidentally starts loading weights."""
    import torch
    loads = []

    def forbidden(*args, **kwargs):
        loads.append(True)
        raise AssertionError("construction-only ambient fixture attempted SDK model loading")

    onnx = ModuleType("onnx_asr")
    onnx.load_model = forbidden
    monkeypatch.setitem(sys.modules, "onnx_asr", onnx)
    wake = ModuleType("openwakeword")
    wake.Model = forbidden
    monkeypatch.setitem(sys.modules, "openwakeword", wake)
    speechbrain = ModuleType("speechbrain")
    speechbrain.__path__ = []
    inference = ModuleType("speechbrain.inference")
    inference.__path__ = []
    speaker = ModuleType("speechbrain.inference.speaker")
    speaker.EncoderClassifier = SimpleNamespace(from_hparams=forbidden)
    for name, module in (("speechbrain", speechbrain), ("speechbrain.inference", inference),
            ("speechbrain.inference.speaker", speaker)):
        monkeypatch.setitem(sys.modules, name, module)
    audio = ModuleType("torchaudio")
    def backends():
        return []
    audio.list_audio_backends = backends
    monkeypatch.setitem(sys.modules, "torchaudio", audio)
    monkeypatch.setattr(torch.hub, "load", forbidden)
    return loads


@asynccontextmanager
async def real_ambient_application(root, monkeypatch):
    """Build the actual ambient adapters; leave their model/VAD work unstarted."""
    root = Path(root)
    factories, models = nominal_factories(monkeypatch, root)
    sdk_loads = construction_sdk_guards(monkeypatch)
    async with offline_home() as (url, observations):
        config = configuration(root, url, ambient=True, rooms=True)
        config["ambient"].update(shadow_mode=True, quality={"enabled": False},
            stt={"engine": "parakeet", "device": "cpu", "parakeet_model": "synthetic-ambient"})
        stack = AsyncExitStack()
        try:
            graph = await build_application(config, shutdown=stack, factories=factories)
            assert graph.ambient_path is not None and not sdk_loads
            yield graph, models, observations, sdk_loads
        finally:
            await stack.aclose()
