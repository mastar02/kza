"""Actual ambient owners and SQLite, with deterministic native model endpoints."""

from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
import threading
import time
from uuid import uuid4
from weakref import ref

import numpy as np

from src.ambient.ambient_stt import AmbientSTT
from src.ambient.doa import DoAResult
from src.ambient.segmenter import UtteranceSegmenter
from src.ambient.source_classifier import SourceClassifier, SourceClassifierConfig
from src.ambient.speaker_tagger import SpeakerTagger
from src.ambient.store import AmbientStore
from src.ambient.tap import MultiChannelTap
from src.ambient.transcriber import AmbientPath, AmbientTranscriber
from src.ambient.types import RawSegment
from src.memory.privacy_services import PrivacyServices
from src.memory.producer_context import PRODUCER_CONTEXT
from src.stt.whisper_fast import STTResult
from src.users.speaker_identifier import SpeakerMatch


@dataclass
class Clock:
    now: float = field(default_factory=time.time)

    def __call__(self):
        return self.now


class Gate:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.calls = 0
        self.threads = []

    def block(self):
        self.release.clear()

    def step(self):
        self.calls += 1
        self.threads.append(threading.get_ident())
        self.entered.set()
        assert self.release.wait(4)


class Engine:
    def __init__(self, gate):
        self.gate = gate
        self.results = []

    def transcribe_with_confidence(self, audio):
        self.active_room = PRODUCER_CONTEXT.get().room_id
        self.gate.step()
        value = STTResult("Synthetic private transcript", 1.0, 0.01, -0.1, 1.0)
        self.results.append(ref(value))
        return value


class Identifier:
    def __init__(self, gate):
        self.gate = gate
        self.embeddings = []

    def identify(self, audio, profiles):
        self.active_room = PRODUCER_CONTEXT.get().room_id
        self.gate.step()
        embedding = np.ones(192, dtype=np.float32)
        self.embeddings.append(ref(embedding))
        return SpeakerMatch("synthetic-speaker", 0.9, embedding, True)


@dataclass
class AmbientGraph:
    privacy: PrivacyServices
    store: AmbientStore
    transcriber: AmbientTranscriber
    path: AmbientPath
    clock: Clock
    gates: dict[str, Gate]
    engine: Engine
    identifier: Identifier

    @property
    def tap(self):
        return self.path.tap

    def segment(self):
        return RawSegment(
            self.clock.now,
            self.clock.now + 1,
            np.full((16000, 6), 0.2, dtype=np.float32),
            vad_prob=0.9,
        )


def chunk(value=0.2):
    return np.full((1280, 6), value, dtype=np.float32)


@asynccontextmanager
async def ambient_graph(tmp_path, *, stage=None, rooms=("synthetic-a",), clock=None):
    gates = {
        name: Gate()
        for name in (
            "factory",
            "vad",
            "stt",
            "profiles",
            "speaker",
            "quality",
            "doa",
            "reset",
        )
    }
    if stage is not None:
        gates[stage].block()
    clock = clock or Clock()
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(
            str(tmp_path / "privacy.db"), clock=clock, close_timeout_s=0.03
        )
        await privacy.acall(privacy.initialize)
        stack.push_async_callback(privacy.aclose)
        store = AmbientStore(
            str(tmp_path / "ambient.db"), source_policy=privacy.ledger, clock=clock
        )
        await store.init()
        stack.push_async_callback(store.close)
        engine, identifier = Engine(gates["stt"]), Identifier(gates["speaker"])
        stt = AmbientSTT(engine, source_policy=privacy.ledger)

        def profiles():
            gates["profiles"].step()
            return {"synthetic-speaker": np.ones(192, dtype=np.float32)}

        tagger = SpeakerTagger(identifier, profiles, source_policy=privacy.ledger)

        class Direction:
            def estimate(self, audio):
                gates["doa"].step()
                return DoAResult(0.5, 0.9)

        def quality(text, probability):
            gates["quality"].step()
            return "es", 0.99, True

        def factory():
            gates["factory"].step()

            def vad(audio):
                gates["vad"].step()
                return float(np.max(np.abs(audio)) > 0.05)

            vad.reset = gates["reset"].step
            return UtteranceSegmenter(
                vad, close_silence_ms=80, preroll_ms=80, min_speech_ms=80
            )

        tap = MultiChannelTap(clock=clock)
        transcriber = AmbientTranscriber(
            tap,
            factory,
            stt,
            tagger,
            Direction(),
            SourceClassifier(SourceClassifierConfig()),
            store,
            list(rooms),
            poll_interval_s=0.005,
            quality_fn=quality,
            privacy=privacy,
            close_timeout_s=0.03,
        )
        path = AmbientPath(tap, transcriber, store, None)
        privacy.add_borrower(path.begin_close)
        stack.push_async_callback(path.stop)
        try:
            yield AmbientGraph(
                privacy, store, transcriber, path, clock, gates, engine, identifier
            )
        finally:
            for gate in gates.values():
                gate.release.set()


def capture_source(graph, room="synthetic-a"):
    from src.memory.fact_ledger import MemoryOrigin

    return graph.privacy.source_for_capture(
        str(uuid4()), graph.clock.now, room, origin=MemoryOrigin.AMBIENT
    )
