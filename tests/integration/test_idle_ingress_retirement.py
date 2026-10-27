"""Idle privacy cleanup retires admission metadata independently of frame dates."""

import asyncio
from collections import deque
import threading
from uuid import uuid4
from weakref import ref

import numpy as np
import pytest

from src.memory.fact_ledger import MemoryOrigin
from tests.integration.test_capture_producer_privacy import ACTOR, graph as multi_graph, until
from tests.integration.test_input_callback_retirement import HeldInput, graph, invoke
from tests.integration.test_shared_input_retirement import ambient
from tests.integration.test_ambient_reset_retirement import owned_room, stamped
from tests.mocks.ambient_graph import ambient_graph, chunk

pytestmark = pytest.mark.asyncio


async def invalidate(g, source, room_id, origin, mode):
    if mode == 'revoke':
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
    elif mode == 'expire':
        g.clock.now += 48*3600
    else:
        g.clock.now += 121
        current = g.privacy.source_for_capture(str(uuid4()), g.clock.now, room_id, origin=origin)
        assert current.conversation_id != source.conversation_id


@pytest.mark.parametrize('transport', ['single', 'multi'])
@pytest.mark.parametrize('mode', ['revoke', 'expire', 'rotate'])
async def test_idle_command_prune_removes_retired_admission_with_future_frame_date(tmp_path, monkeypatch, transport, mode):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        future = g.clock.now+48*3600+1
        monkeypatch.setattr(g.loop, '_clock' if transport == 'single' else '_microphone_wall_clock', lambda: future+.08)
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        reference = ref(room._queue[0].audio)
        await invalidate(g, source, room.room_id, MemoryOrigin.VOICE, mode)
        await room.prune_private()
        assert room.resource_status()['queued'] == 0
        assert reference() is None
        assert not room.conversation_busy(source.conversation_id)


@pytest.mark.parametrize('mode', ['revoke', 'expire', 'rotate'])
async def test_idle_ambient_prune_removes_retired_admission_without_starting_workers(tmp_path, monkeypatch, mode):
    async with graph(tmp_path, monkeypatch, 'multi') as (g, _):
        source = g.privacy.source_for_capture(str(uuid4()), g.clock.now, 'synthetic-a', origin=MemoryOrigin.AMBIENT)
        async with ambient(g, tmp_path) as path:
            audio = np.ones((1280, 1), dtype=np.float32)
            reference = ref(audio)
            path.tap.push('synthetic-a', audio, ts=g.clock.now+48*3600+1)
            audio = None
            assert path.tap.buffered_count() == 1
            await invalidate(g, source, 'synthetic-a', MemoryOrigin.AMBIENT, mode)
            path.transcriber.prune_private()
            assert path.tap.buffered_count() == 0
            assert reference() is None
            assert not path.transcriber.conversation_busy(source.conversation_id)


@pytest.mark.parametrize('mode', ['revoke', 'expire', 'rotate'])
async def test_idle_ready_segment_retires_its_causal_epoch_even_with_a_future_start(tmp_path, mode):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        await room.feed(stamped(g, [(g.clock.now, chunk(0.), False)]))
        source = room._source
        await room.feed(stamped(g, [
            (g.clock.now+240, chunk(0.), False),
            (g.clock.now+240.08, chunk(), False),
            (g.clock.now+240.16, chunk(0.), False),
        ]))
        assert len(room.ready) == 1
        assert room.ready[0].input_source == source
        reference = ref(room.ready[0].audio)
        await invalidate(g, source, 'synthetic-a', MemoryOrigin.AMBIENT, mode)
        g.transcriber.prune_private()
        assert not room.ready
        assert reference() is None
        assert not g.transcriber.conversation_busy(source.conversation_id)


@pytest.mark.parametrize('stage', ['ready', 'native-return'])
async def test_retired_causal_epoch_cannot_reach_stt_after_native_reset(tmp_path, monkeypatch, stage):
    async with ambient_graph(tmp_path) as g:
        room = owned_room(g)
        await room.feed(stamped(g, [(g.clock.now, chunk(0.), False)]))
        source = room._source
        entered, release = asyncio.Event(), asyncio.Event()
        references = []
        original = room._call

        async def held_call(function, *args, **kwargs):
            result = None
            try:
                result = await original(function, *args, **kwargs)
                if function == room._feed and result:
                    references.append(ref(result[0].audio))
                    entered.set()
                    await release.wait()
                return result
            finally:
                result = None

        if stage == 'native-return':
            monkeypatch.setattr(room, '_call', held_call)
        items = [
            (g.clock.now+240, chunk(0.), False),
            (g.clock.now+240.08, chunk(), False),
            (g.clock.now+240.16, chunk(0.), False),
        ]
        consuming = segment = None
        if stage == 'ready':
            await room.feed(stamped(g, items))
            segment = room.ready.pop()
            references.append(ref(segment.audio))
        else:
            for timestamp, audio, tts_active in items:
                g.tap.push('synthetic-a', audio, ts=timestamp, tts_active=tts_active)
            consuming = asyncio.create_task(g.transcriber._consume_batch('synthetic-a', room))
        items = None
        try:
            if stage == 'native-return':
                await asyncio.wait_for(entered.wait(), 3)
                assert g.transcriber.conversation_busy(source.conversation_id)
            await invalidate(g, source, 'synthetic-a', MemoryOrigin.AMBIENT, 'rotate')
            # The original conversation is still live. A date-only or source-only
            # guard would admit this raw audio despite its retired epoch.
            assert g.privacy.sources_available((source,))
            if segment is not None:
                await g.transcriber._handle_segment('synthetic-a', segment)
                segment = None
        finally:
            release.set()
            if consuming is not None:
                await consuming
        assert g.gates['stt'].calls == 0
        assert not room.ready
        assert references and all(value() is None for value in references)


async def test_idle_multiroom_prune_keeps_the_other_rooms_valid_queue(tmp_path, monkeypatch):
    async with multi_graph(tmp_path, controllers=False, rooms=('synthetic-a', 'synthetic-b')) as g:
        for name in ('synthetic-a', 'synthetic-b'):
            g.feed(name)
            await g.loop._capture_rooms[name].step()
        source = g.source('synthetic-a')
        future = g.clock.now+1
        monkeypatch.setattr(g.loop, '_microphone_wall_clock', lambda: future)
        g.feed('synthetic-a')
        g.feed('synthetic-b')
        other = g.loop._capture_rooms['synthetic-b']
        reference = ref(other._queue[0].audio)
        await invalidate(g, source, 'synthetic-a', MemoryOrigin.VOICE, 'revoke')
        await g.loop.prune_private()
        assert g.loop._capture_rooms['synthetic-a'].resource_status()['queued'] == 0
        assert other.resource_status()['queued'] == 1 and reference() is not None
        assert g.wakes['synthetic-b'].history


async def next_scope_at_same_clock(g, room_id, origin):
    first = g.privacy.source_for_capture(str(uuid4()), g.clock.now, room_id, origin=origin)
    await invalidate(g, first, room_id, origin, 'revoke')
    return g.privacy.source_for_capture(str(uuid4()), g.clock.now+.1, room_id, origin=origin)


@pytest.mark.parametrize('transport', ['single', 'multi'])
@pytest.mark.parametrize('stage', ['queued', 'copy'])
async def test_command_prune_detaches_on_epoch_change_with_identical_date_fence(tmp_path, monkeypatch, transport, stage):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        source = await next_scope_at_same_clock(g, room.room_id, MemoryOrigin.VOICE)
        await room.prune_private()
        boundary = room.input_boundary
        monkeypatch.setattr(g.loop, '_clock' if transport == 'single' else '_microphone_wall_clock', lambda: g.clock.now+1)
        audio = HeldInput('copy' if stage == 'copy' else None)
        pushing = asyncio.create_task(asyncio.to_thread(invoke, g, transport, audio))
        try:
            if stage == 'copy':
                await until(audio.entered.is_set)
            else:
                await pushing
                assert room.resource_status()['queued'] == 1
            await invalidate(g, source, room.room_id, MemoryOrigin.VOICE, 'revoke')
            assert g.privacy.scopes.buffer_boundary(room.room_id, origin=MemoryOrigin.VOICE) == boundary
            await room.prune_private()
            assert room.resource_status()['queued'] == 0
            if stage == 'copy':
                assert room.conversation_busy(source.conversation_id)
        finally:
            audio.release.set()
            await pushing
        assert room.resource_status()['queued'] == 0
        assert not room.conversation_busy(source.conversation_id)


@pytest.mark.parametrize('stage', ['queued', 'append'])
async def test_ambient_prune_detaches_on_epoch_change_with_identical_date_fence(tmp_path, monkeypatch, stage):
    async with graph(tmp_path, monkeypatch, 'multi') as (g, _):
        async with ambient(g, tmp_path) as path:
            source = await next_scope_at_same_clock(g, 'synthetic-a', MemoryOrigin.AMBIENT)
            path.transcriber.prune_private()
            boundary = path.tap.input_boundary('synthetic-a')
            entered, release = threading.Event(), threading.Event()

            class DelayedDeque(deque):
                def append(self, item):
                    entered.set()
                    assert release.wait(3)
                    super().append(item)

            if stage == 'append':
                path.tap._queues['synthetic-a'] = DelayedDeque(maxlen=1250)
            audio = np.ones((1280, 1), dtype=np.float32)
            reference = ref(audio)
            pushing = asyncio.create_task(asyncio.to_thread(path.tap.push, 'synthetic-a', audio, g.clock.now+1))
            audio = None
            try:
                if stage == 'append':
                    await until(entered.is_set)
                else:
                    await pushing
                await invalidate(g, source, 'synthetic-a', MemoryOrigin.AMBIENT, 'revoke')
                assert g.privacy.scopes.buffer_boundary('synthetic-a') == boundary
                path.transcriber.prune_private()
                assert path.tap.buffered_count() == 0
                if stage == 'append':
                    assert reference() is not None
                    assert path.transcriber.conversation_busy(source.conversation_id)
            finally:
                release.set()
                await pushing
            assert reference() is None
            assert path.tap.buffered_count() == 0
            assert not path.transcriber.conversation_busy(source.conversation_id)
