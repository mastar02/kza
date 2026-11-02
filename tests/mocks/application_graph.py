"""Deterministic model/transport leaves for the actual application factory."""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
import threading

from aiohttp import web
import numpy as np

from src.stt.whisper_fast import STTResult


class OfflineEncoder:
    """Replace only the embedding model; keep the real Chroma owner and SDK."""

    def encode(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class OfflineWake:
    """Keep configured input wiring visible without loading a wake model."""

    def __init__(self, **kwargs):
        self.loads = 0

    def load(self) -> None:
        self.loads += 1

    def reset(self) -> None:
        pass


@dataclass
class OfflineSTT:
    """Supply a known transcription without downloading or loading a model."""

    text: str = "pon un timer de 5 minutos"
    calls: int = 0
    loads: int = 0
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event | None = None

    def load(self) -> None:
        self.loads += 1

    def transcribe_with_confidence(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        self.calls += 1
        return STTResult(self.text, elapsed_ms=1, avg_logprob=-0.1, no_speech_prob=0.01)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(2), "Synthetic STT was not released"
        return self.transcribe_with_confidence(audio, sample_rate).text


@dataclass
class OfflineTTS:
    """Render known PCM; the real response owner still delivers it."""

    sample_rate: int = 24000
    loads: int = 0
    texts: list[str] = field(default_factory=list)

    def load(self) -> None:
        self.loads += 1

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.texts.append(text)
        return np.full(240, 0.125, dtype=np.float32), self.sample_rate


def offline_config(root: Path, ha_url: str, *, rooms: bool = False) -> dict:
    """Isolate all private paths and explicitly disable optional cloud/models."""
    return {
        "home_assistant": {"url": ha_url, "token": "synthetic-token", "control_selection_path": "",
                           "state_prefetch": {"enabled": False}},
        "embeddings": {"model": "synthetic", "device": "cpu"},
        "vectordb": {"path": str(root / "index")},
        "reasoner": {"mode": "http", "cloud": {"consent": False}},
        "router": {"enabled": False},
        "speaker_id": {"enabled": False},
        "emotion": {"enabled": False},
        "monitoring": {"enabled": False},
        "analytics": {"enabled": False},
        "memory": {"enabled": False, "privacy_db_path": str(root / "privacy.db"),
                   "chroma_path": str(root / "memory-index")},
        "lists": {"db_path": str(root / "lists.db")},
        "reminders": {"db_path": str(root / "reminders.db")},
        "routines": {"db_path": str(root / "routines.db")},
        "observability": {"metrics_path": str(root / "metrics.jsonl")},
        "rooms": {"enabled": rooms},
        "zones": {"enabled": False},
        "ambient": {"enabled": False},
        "dashboard": {"enabled": True, "home": {"enabled": True}, "observability_use_mocks": False},
        "orchestrator": {"enabled": True},
        "alerts": {"enabled": False},
        "warmup": {"enabled": False},
        "tts": {"streaming": {"enabled": False}},
        "training": {"nightly": {"enabled": False}},
    }


@asynccontextmanager
async def offline_home():
    """Serve a local read fixture and count every attempted house effect."""
    observations = {"reads": 0, "service_calls": 0}

    async def api(request):
        observations["reads"] += 1
        return web.json_response({"message": "API running"})

    async def states(request):
        observations["reads"] += 1
        return web.json_response([])

    async def service(request):
        observations["service_calls"] += 1
        return web.json_response([])

    app = web.Application()
    app.add_routes([web.get("/api/", api), web.get("/api/states", states),
                    web.post("/api/services/{domain}/{service}", service)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", observations
    finally:
        await runner.cleanup()
