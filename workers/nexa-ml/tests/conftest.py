"""Torch-free fakes of the *models* (not the slots) plus a real grpc.aio server on a UDS.

The fakes mimic the duck-typed surfaces the real slot code calls:
``WhisperModel.transcribe`` (+ ``model.unload_model``), ``KPipeline`` Results with
``.audio``, the ECAPA adapter ``embed`` and ``SentenceTransformer.encode``. All slot,
pool, streaming, drain and reload logic under test is the production code.
"""

import pathlib
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import grpc
import numpy as np
import pytest

from nexa_ml.config import MlConfig
from nexa_ml.pb import ml_pb2_grpc
from nexa_ml.runtime import MlRuntime, ModelLoaders
from nexa_ml.server import start_server

GATE_TIMEOUT_S = 5.0


@dataclass
class FakeSegment:
    text: str
    compression_ratio: float = 1.2
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.05


class _FakeCt2:
    def __init__(self) -> None:
        self.unloaded = 0

    def unload_model(self) -> None:
        self.unloaded += 1


class FakeWhisperModel:
    """Stands in for ``faster_whisper.WhisperModel``."""

    def __init__(self, label: str, hook: Callable[[], None] | None) -> None:
        self.label = label
        self.model = _FakeCt2()
        self.calls: list[dict[str, Any]] = []
        self._hook = hook
        self.thread_names: list[str] = []

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, object]:
        self.calls.append({"samples": int(audio.size), **kwargs})
        self.thread_names.append(threading.current_thread().name)
        if self._hook is not None:
            self._hook()
        return iter(
            [FakeSegment(f"{self.label} hola"), FakeSegment("mundo", compression_ratio=1.8)]
        ), object()


class FakeWhisperFactory:
    """Counts loads, can fail on demand and hands out :class:`FakeWhisperModel`."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.loads = 0
        self.fail = False
        self.hook: Callable[[], None] | None = None
        self.models: list[FakeWhisperModel] = []

    def __call__(self) -> FakeWhisperModel:
        self.loads += 1
        if self.fail:
            raise RuntimeError("simulated load failure")
        model = FakeWhisperModel(self.label, lambda: self.hook() if self.hook else None)
        self.models.append(model)
        return model


@dataclass
class FakeResult:
    audio: Any


class FakeKokoroEngine:
    """Stands in for the Kokoro adapter: ``pipeline(lang)`` -> KPipeline-like callable.

    ``gates[i]`` (if present) must be set before Result ``i + 1`` is produced, which
    lets tests prove chunk ``i`` reached the client before synthesis continued.
    """

    def __init__(self, results: int, gates: list[threading.Event]) -> None:
        self.results = results
        self.gates = gates
        self.pulled = 0
        self.closed = threading.Event()
        self.langs: list[str] = []
        self.requests: list[tuple[str, float]] = []

    def pipeline(self, lang_code: str) -> Callable[..., Any]:
        self.langs.append(lang_code)

        def run(text: str, *, voice: str, speed: float) -> Any:
            self.requests.append((voice, speed))
            try:
                for i in range(self.results):
                    if i > 0 and i - 1 < len(self.gates):
                        if not self.gates[i - 1].wait(GATE_TIMEOUT_S):
                            raise TimeoutError("gate never opened")
                    self.pulled += 1
                    yield FakeResult(audio=np.full(240, float(i + 1), dtype=np.float32))
            finally:
                self.closed.set()

        return run


class FakeEcapa:
    def __init__(self, dim: int = 192) -> None:
        self.dim = dim
        self.peaks: list[float] = []

    def embed(self, audio: np.ndarray) -> np.ndarray:
        self.peaks.append(float(np.abs(audio).max()))
        return np.linspace(-1.0, 1.0, self.dim, dtype=np.float32)


class FakeSentenceTransformer:
    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.thread_names: list[str] = []

    def encode(self, texts: list[str], convert_to_numpy: bool = True) -> np.ndarray:
        self.thread_names.append(threading.current_thread().name)
        return np.ones((len(texts), self.dim), dtype=np.float32) / np.sqrt(self.dim)


@dataclass
class Fakes:
    command: FakeWhisperFactory = field(default_factory=lambda: FakeWhisperFactory("cmd"))
    ambient: list[FakeWhisperFactory] = field(default_factory=list)
    tts_results: int = 3
    tts_gates: list[threading.Event] = field(default_factory=list)
    tts_engines: list[FakeKokoroEngine] = field(default_factory=list)
    ecapa: FakeEcapa = field(default_factory=FakeEcapa)
    embedder: FakeSentenceTransformer = field(default_factory=FakeSentenceTransformer)

    def loaders(self, workers: int) -> ModelLoaders:
        self.ambient = [FakeWhisperFactory(f"amb{i}") for i in range(workers)]

        def tts() -> FakeKokoroEngine:
            engine = FakeKokoroEngine(self.tts_results, self.tts_gates)
            self.tts_engines.append(engine)
            return engine

        return ModelLoaders(
            command_stt=self.command,
            ambient_stt=lambda i: self.ambient[i],
            tts=tts,
            speaker=lambda: self.ecapa,
            embed=lambda: self.embedder,
        )


@dataclass
class Harness:
    runtime: MlRuntime
    stub: ml_pb2_grpc.MlServiceStub
    fakes: Fakes
    socket_path: pathlib.Path


@pytest.fixture
def short_dir(tmp_path: pathlib.Path) -> Any:
    """Directory short enough for ``sun_path`` (macOS: 104 bytes)."""
    if len(str(tmp_path)) < 70:
        yield tmp_path
        return
    path = pathlib.Path(tempfile.mkdtemp(prefix="nxml-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


@pytest.fixture
def config() -> MlConfig:
    return MlConfig(ambient_stt_workers=2, max_pending=4)


@pytest.fixture
async def harness(short_dir: pathlib.Path, fakes: Fakes, config: MlConfig) -> AsyncIterator[Harness]:
    runtime = MlRuntime(config, fakes.loaders(config.ambient_stt_workers))
    await runtime.load_all()
    socket_path = short_dir / "ml.sock"
    server = await start_server(runtime, socket_path)
    channel = grpc.aio.insecure_channel(f"unix:{socket_path}")
    try:
        yield Harness(runtime, ml_pb2_grpc.MlServiceStub(channel), fakes, socket_path)
    finally:
        await channel.close()
        await server.stop(0)
        runtime.close()


def pcm(seconds: float = 1.0, rate: int = 16_000, amplitude: float = 0.1) -> bytes:
    """Synthetic float32 LE sine (never real audio)."""
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype("<f4").tobytes()
