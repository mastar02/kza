"""A retained native result transport must never retain delivered ambient PCM."""

import asyncio
import threading
from weakref import ref

import pytest

from src.memory.fact_ledger import MemoryOrigin
from tests.integration.test_ambient_reset_retirement import owned_room, stamped
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_idle_ingress_retirement import invalidate
from tests.mocks.ambient_graph import ambient_graph, chunk

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('mode', ['revoke', 'expire', 'rotate'])
async def test_retained_native_outcome_does_not_keep_ready_audio_after_epoch_retirement(tmp_path, monkeypatch, mode):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        native = room._work._worker
        carriers = []
        async def retained(*args, **kwargs):
            result = await native(*args, **kwargs)
            carriers.append(result)
            return result
        monkeypatch.setattr(room._work, '_worker', retained)
        await room.feed(stamped(g, [(g.clock.now, chunk(0.), False)]))
        source = room._source
        await room.feed(stamped(g, [(g.clock.now+240, chunk(0.), False),
            (g.clock.now+240.08, chunk(), False), (g.clock.now+240.16, chunk(0.), False)]))
        assert len(room.ready) == 1 and room.ready[0].audio.size > 0
        reference = ref(room.ready[0].audio)
        await invalidate(g, source, 'synthetic-a', MemoryOrigin.AMBIENT, mode)
        g.transcriber.prune_private()
        assert not room.ready and carriers
        assert reference() is None
        assert all(carrier.value is None for carrier in carriers)


async def test_cancelled_receiver_retires_result_before_a_held_transport_completes(tmp_path, monkeypatch):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        await room.feed(stamped(g, [(g.clock.now, chunk(0.), False)]))
        native = room._work._worker
        entered, release = asyncio.Event(), asyncio.Event()
        carriers, references = [], []
        async def held(*args, **kwargs):
            result = await native(*args, **kwargs)
            if isinstance(result.value, list) and result.value:
                carriers.append(result)
                references.append(ref(result.value[0].audio))
                entered.set()
                await release.wait()
            return result
        monkeypatch.setattr(room._work, '_worker', held)
        feeding = asyncio.create_task(room.feed(stamped(g, [(g.clock.now+240, chunk(0.), False),
            (g.clock.now+240.08, chunk(), False), (g.clock.now+240.16, chunk(0.), False)])))
        try:
            await until(entered.is_set)
            feeding.cancel()
            await asyncio.gather(feeding, return_exceptions=True)
            await until(lambda: not room._batches)
            assert room.snapshot_operations() and not room.ready
            assert references[0]() is None
            assert carriers[0].value is None
        finally:
            release.set()


async def test_abandoned_receiver_retires_a_result_created_later_by_the_actual_native_worker(tmp_path, monkeypatch):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        await room.feed(stamped(g, [(g.clock.now, chunk(0.), False)]))
        feed, native = room._feed, room._work._worker
        entered, release = threading.Event(), threading.Event()
        carriers, references = [], []
        def held_feed(items):
            value = None
            try:
                entered.set()
                assert release.wait(4)
                value = feed(items)
                if value:
                    references.append(ref(value[0].audio))
                return value
            finally:
                value = items = None
        async def retained(*args, **kwargs):
            result = await native(*args, **kwargs)
            carriers.append(result)
            return result
        monkeypatch.setattr(room, '_feed', held_feed)
        monkeypatch.setattr(room._work, '_worker', retained)
        feeding = asyncio.create_task(room.feed(stamped(g, [(g.clock.now+240, chunk(0.), False),
            (g.clock.now+240.08, chunk(), False), (g.clock.now+240.16, chunk(0.), False)])))
        try:
            await until(entered.is_set)
            feeding.cancel()
            await asyncio.gather(feeding, return_exceptions=True)
            await until(lambda: not room._batches)
            assert room.snapshot_operations()
        finally:
            release.set()
        await until(lambda: not room.snapshot_operations())
        assert references and references[0]() is None and not room.ready
        assert carriers and all(carrier.value is None for carrier in carriers)
