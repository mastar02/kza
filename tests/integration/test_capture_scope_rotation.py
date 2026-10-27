"""Rotated capture scopes never relabel delayed raw input as a new exchange."""

from contextlib import AsyncExitStack
from uuid import uuid4
from weakref import ref

import pytest

from src.memory.fact_ledger import MemoryOrigin, MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from tests.integration.test_ambient_producer_privacy import until
from tests.integration.test_capture_producer_privacy import graph as command_graph
from tests.mocks.ambient_graph import Clock, ambient_graph, chunk

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')


async def initialize(stack, path):
    clock = Clock()
    privacy = PrivacyServices(str(path/'privacy.db'), clock=clock)
    await privacy.acall(privacy.initialize)
    stack.push_async_callback(privacy.aclose)
    clock.now += 1
    return privacy, clock


def capture(privacy, clock, origin, *, room='synthetic-a', timestamp=None, **kwargs):
    return privacy.source_for_capture(str(uuid4()), clock.now if timestamp is None else timestamp,
                                      room, origin=origin, **kwargs)


def rotate(privacy, clock, origin, mode):
    if mode == 'idle':
        clock.now += 121
    else:
        for _ in range(30):
            clock.now += 120
            capture(privacy, clock, origin)
        clock.now += 120
    return capture(privacy, clock, origin)


@pytest.mark.parametrize('origin', list(MemoryOrigin))
@pytest.mark.parametrize('mode', ['idle', 'hour'])
@pytest.mark.parametrize('revoke', [False, True], ids=['rotated', 'old-revoked'])
async def test_previous_scope_input_is_retired_after_rotation(tmp_path, origin, mode, revoke):
    async with AsyncExitStack() as stack:
        privacy, clock = await initialize(stack, tmp_path)
        first = capture(privacy, clock, origin)
        other = capture(privacy, clock, origin, room='synthetic-b')
        current = rotate(privacy, clock, origin, mode)
        assert current.conversation_id != first.conversation_id
        if revoke:
            await privacy.acall(privacy.ledger.delete_conversation, first.conversation_id, ACTOR)
        boundary = privacy.scopes.buffer_boundary('synthetic-a', origin=origin)
        assert not boundary.allows(first.captured_at)
        assert boundary.allows(current.captured_at)
        with pytest.raises(PermissionError, match='capture_precedes_current_scope'):
            capture(privacy, clock, origin, timestamp=first.captured_at)
        assert privacy.sources_available((current,))
        assert privacy.sources_available((other,))
        assert privacy.scopes.buffer_boundary('synthetic-b', origin=origin).allows(other.captured_at)
        assert capture(privacy, clock, origin).conversation_id == current.conversation_id


@pytest.mark.parametrize('revoke', [False, True], ids=['rotated', 'old-revoked'])
async def test_real_ambient_tap_drops_previous_exchange_without_touching_another_room(tmp_path, revoke):
    async with ambient_graph(tmp_path, rooms=('synthetic-a', 'synthetic-b')) as g:
        first = capture(g.privacy, g.clock, MemoryOrigin.AMBIENT)
        for room in ('synthetic-a', 'synthetic-b'):
            g.tap.register_room(room)
        old_segment = g.segment()
        audio = chunk()
        reference = ref(audio)
        g.tap.push('synthetic-a', audio, ts=g.clock.now)
        g.tap.push('synthetic-b', chunk(), ts=g.clock.now)
        del audio
        current = rotate(g.privacy, g.clock, MemoryOrigin.AMBIENT, 'idle')
        if revoke:
            await g.privacy.acall(g.privacy.ledger.delete_conversation, first.conversation_id, ACTOR)
        g.transcriber.prune_private()
        assert reference() is None
        assert not g.tap.drain('synthetic-a') and len(g.tap.drain('synthetic-b')) == 1
        await g.transcriber._handle_segment('synthetic-a', old_segment)
        assert g.gates['stt'].calls == 0
        await g.transcriber._handle_segment('synthetic-a', g.segment())
        rows = await g.store.undistilled_live(require_provenance=True)
        assert len(rows) == 1 and rows[0]['conversation_id'] == current.conversation_id


@pytest.mark.parametrize('revoke', [False, True], ids=['rotated', 'old-revoked'])
async def test_real_command_queue_drops_old_packets_before_wake_and_keeps_other_room(tmp_path, revoke):
    async with command_graph(tmp_path, rooms=('synthetic-a', 'synthetic-b'), controllers=False) as g:
        first = capture(g.privacy, g.clock, MemoryOrigin.VOICE)
        capture(g.privacy, g.clock, MemoryOrigin.VOICE, room='synthetic-b')
        g.clock.now += .1
        g.feed('synthetic-a')
        g.feed('synthetic-b')
        current = rotate(g.privacy, g.clock, MemoryOrigin.VOICE, 'idle')
        if revoke:
            await g.privacy.acall(g.privacy.ledger.delete_conversation, first.conversation_id, ACTOR)
        room = g.loop._capture_rooms['synthetic-a']
        assert await room.step() is None
        assert not g.wakes['synthetic-a'].history and not room._queue
        assert len(g.loop._capture_rooms['synthetic-b']._queue) == 1
        await g.loop._capture_rooms['synthetic-b'].step()
        assert g.wakes['synthetic-b'].history
        g.clock.now += .1
        g.feed('synthetic-a')
        await room.step()
        assert g.wakes['synthetic-a'].history
        assert room.source.conversation_id == current.conversation_id


async def test_real_vad_retains_native_input_until_exit_but_cannot_publish_rotated_audio(tmp_path):
    async with ambient_graph(tmp_path, stage='vad') as g:
        first = capture(g.privacy, g.clock, MemoryOrigin.AMBIENT)
        await g.path.start()
        audio = chunk()
        reference = ref(audio)
        g.tap.push('synthetic-a', audio, ts=g.clock.now)
        del audio
        await until(g.gates['vad'].entered.is_set)
        current = rotate(g.privacy, g.clock, MemoryOrigin.AMBIENT, 'idle')
        await g.privacy.acall(g.privacy.ledger.delete_conversation, first.conversation_id, ACTOR)
        g.transcriber.prune_private()
        state = g.transcriber._room_states['synthetic-a']
        assert state.snapshot_operations() and reference() is not None
        assert g.privacy.resource_status()['writer_lease']
        g.gates['vad'].release.set()
        await until(lambda: not state.snapshot_operations() and not state.status()['state_retained'])
        await until(lambda: reference() is None)
        assert state.ready == [] and g.gates['stt'].calls == 0
        assert await g.store.undistilled_live(require_provenance=True) == []
        assert g.privacy.sources_available((current,))


@pytest.mark.parametrize('origin', list(MemoryOrigin))
async def test_delayed_input_within_current_scope_keeps_identity_and_high_watermark(tmp_path, origin):
    async with AsyncExitStack() as stack:
        privacy, clock = await initialize(stack, tmp_path)
        first = capture(privacy, clock, origin)
        clock.now += 60
        second = capture(privacy, clock, origin)
        delayed = capture(privacy, clock, origin, timestamp=first.captured_at+30)
        assert delayed.conversation_id == second.conversation_id == first.conversation_id
        clock.now += 120
        assert capture(privacy, clock, origin).conversation_id == first.conversation_id
        clock.now += 121
        assert capture(privacy, clock, origin).conversation_id != first.conversation_id


@pytest.mark.parametrize('origin', list(MemoryOrigin))
async def test_explicit_lineage_stays_original_without_moving_current_scope_backwards(tmp_path, origin):
    async with AsyncExitStack() as stack:
        privacy, clock = await initialize(stack, tmp_path)
        first = capture(privacy, clock, origin)
        current = rotate(privacy, clock, origin, 'idle')
        explicit = capture(privacy, clock, origin, timestamp=first.captured_at,
                           conversation_id=first.conversation_id, expires_at=first.expires_at)
        assert explicit.conversation_id == first.conversation_id
        assert privacy.sources_available((explicit,))
        await privacy.acall(privacy.ledger.delete_conversation, first.conversation_id, ACTOR)
        assert not privacy.sources_available((explicit,))
        assert capture(privacy, clock, origin).conversation_id == current.conversation_id
