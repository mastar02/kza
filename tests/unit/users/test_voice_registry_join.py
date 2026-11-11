"""U3b-4b-2: the capture join in VoiceRegistry (completion map, bounded, off by default)."""

import asyncio
import threading
import time

import numpy as np
import pytest

from src.users.voice_registry import COMPLETION_CAPACITY, encoder_model_key
from src.users.voice_ledger import fingerprint_identity
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_registry import AUDIO, Encoder, graph, idle

pytestmark = pytest.mark.asyncio


class Held(Encoder):
    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def get_embedding(self, audio):
        self.release.wait(5)
        return super().get_embedding(audio)


async def test_off_by_default_no_completion_is_created(tmp_path):
    async with graph(tmp_path) as (privacy, owner, voices, clock):
        encoder = Encoder()
        voices.register_encoder(encoder)
        source = memory_source(now=clock.now)
        assert voices.observe(encoder, AUDIO, (source,))
        await idle(voices)
        assert voices._completions is None
        wait = await voices.await_fingerprint(source.capture_id, 1.0)
        assert (wait.fingerprint_id, wait.reason) == (None, "no_fingerprint")


async def test_the_join_returns_the_stored_fingerprint_once(tmp_path):
    async with graph(tmp_path) as (privacy, owner, voices, clock):
        voices.enable_completions()
        encoder = Encoder()
        voices.register_encoder(encoder)
        source = memory_source(now=clock.now)
        assert voices.observe(encoder, AUDIO, (source,))
        wait = await voices.await_fingerprint(source.capture_id, 2.0)
        assert wait.reason is None and wait.fingerprint_id == fingerprint_identity(encoder_model_key(encoder), (source,))
        again = await voices.await_fingerprint(source.capture_id, 2.0)
        assert (again.fingerprint_id, again.reason) == (None, "no_fingerprint"), "single consumer"
        started = time.perf_counter()
        assert (await voices.await_fingerprint("never-observed", 5.0)).reason == "no_fingerprint"
        assert time.perf_counter() - started < 0.1, "no entry answers at once"


async def test_a_late_embedding_times_out_and_still_resolves_later(tmp_path):
    async with graph(tmp_path) as (privacy, owner, voices, clock):
        voices.enable_completions()
        encoder = Held()
        voices.register_encoder(encoder)
        source = memory_source(now=clock.now)
        assert voices.observe(encoder, AUDIO, (source,))
        started = time.perf_counter()
        wait = await voices.await_fingerprint(source.capture_id, 0.05)
        assert wait.reason == "late_embedding" and wait.fingerprint_id is None and time.perf_counter() - started < 0.5
        encoder.release.set()
        assert type(await asyncio.wait_for(asyncio.shield(wait.completion), 5)) is str


async def test_a_failed_capture_is_no_fingerprint(tmp_path):
    async with graph(tmp_path) as (privacy, owner, voices, clock):
        voices.enable_completions()

        class Broken(Encoder):
            def get_embedding(self, audio):
                return np.full(192, np.nan, dtype=np.float32)

        encoder = Broken()
        voices.register_encoder(encoder)
        source = memory_source(now=clock.now)
        assert voices.observe(encoder, AUDIO, (source,))
        assert (await voices.await_fingerprint(source.capture_id, 2.0)).reason == "no_fingerprint"


async def test_the_map_is_bounded_and_evicts_the_oldest_resolved_entry(tmp_path):
    async with graph(tmp_path) as (privacy, owner, voices, clock):
        voices.enable_completions()
        encoder = Encoder()
        voices.register_encoder(encoder)
        sources = [memory_source(now=clock.now) for _ in range(COMPLETION_CAPACITY + 3)]
        for source in sources:
            assert voices.observe(encoder, AUDIO, (source,))
            await idle(voices)
        assert len(voices._completions) == COMPLETION_CAPACITY == 6
        assert list(voices._completions) == [source.capture_id for source in sources[-6:]]
        assert (await voices.await_fingerprint(sources[0].capture_id, 1.0)).reason == "no_fingerprint"
        assert (await voices.await_fingerprint(sources[-1].capture_id, 1.0)).reason is None
