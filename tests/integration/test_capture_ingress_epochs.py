"""Admitted input cannot acquire a new conversation after a delayed clock."""

import asyncio
from collections import deque
from contextlib import AsyncExitStack
import threading
from uuid import uuid4

import numpy as np
import pytest

from src.ambient.segmenter import UtteranceSegmenter
from src.ambient.room_producer import RoomProducer
from src.memory.fact_ledger import MemoryOrigin
from src.memory.privacy_services import PrivacyServices
from tests.integration.test_capture_producer_privacy import ACTOR, until
from tests.integration.test_input_callback_retirement import HeldInput, graph, invoke
from tests.integration.test_shared_input_retirement import ambient

pytestmark = pytest.mark.asyncio


async def test_sourced_ambient_actor_requires_an_admission_validator():
    with pytest.raises(ValueError, match='ambient_admission_validator_required'):
        RoomProducer('synthetic-a', lambda: None, source_fn=lambda room, timestamp: None)


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_voice_input_admitted_before_revocation_cannot_use_a_later_clock(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        wake = g.wake if transport == 'single' else g.wakes['synthetic-a']
        audio = HeldInput('clock')

        def delayed_clock():
            audio.wait('clock')
            return g.clock.now

        monkeypatch.setattr(g.loop, '_clock' if transport == 'single' else '_microphone_wall_clock', delayed_clock)
        pushing = asyncio.create_task(asyncio.to_thread(invoke, g, transport, audio))
        try:
            await until(audio.entered.is_set)
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            g.clock.now += 1
        finally:
            audio.release.set()
            await pushing
        await room.step()
        assert not wake.history
        assert not room.resource_status()['has_state']
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        assert wake.history
        assert room.source.conversation_id != source.conversation_id


async def test_ambient_old_callback_cannot_relabel_audio_before_fence_publication(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, 'multi') as (g, voice):
        source = g.privacy.source_for_capture(str(uuid4()), g.clock.now, 'synthetic-a', origin=MemoryOrigin.AMBIENT)
        async with ambient(g, tmp_path) as path:
            calls = []

            def factory():
                def vad(audio):
                    calls.append('inference')
                    return 0.
                vad.reset = lambda: None
                return UtteranceSegmenter(vad, preroll_ms=80)

            monkeypatch.setattr(path.transcriber, '_segmenter_factory', factory)
            # Delay the control publisher; native source admission still queries
            # the actual authority and its current capture scope.
            monkeypatch.setattr(path.transcriber, 'prune_private', lambda: None)
            await path.transcriber.start()
            state = path.transcriber._room_states['synthetic-a']
            g.clock.now += .2
            audio = HeldInput('clock')

            def delayed_clock():
                audio.wait('clock')
                return g.clock.now

            monkeypatch.setattr(g.loop, '_microphone_wall_clock', delayed_clock)
            pushing = asyncio.create_task(asyncio.to_thread(invoke, g, 'multi', audio))
            try:
                await until(audio.entered.is_set)
                await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
                g.clock.now += 1
            finally:
                audio.release.set()
                await pushing
            await path.transcriber._consume_batch('synthetic-a', state)
            assert not calls
            assert not state.status()['state_retained']

            assert voice.resource_status()['queued'] == 1
            g.feed()
            await path.transcriber._consume_batch('synthetic-a', state)
            assert calls
            assert state._source.conversation_id != source.conversation_id


@pytest.mark.parametrize('stage', ['clock', 'reservation'])
async def test_direct_tap_push_keeps_admission_before_its_pool_or_clock_wait(tmp_path, monkeypatch, stage):
    async with graph(tmp_path, monkeypatch, 'multi') as (g, _):
        source = g.privacy.source_for_capture(str(uuid4()), g.clock.now, 'synthetic-a', origin=MemoryOrigin.AMBIENT)
        async with ambient(g, tmp_path) as path:
            calls = []

            def factory():
                def vad(audio):
                    calls.append('inference')
                    return 0.
                vad.reset = lambda: None
                return UtteranceSegmenter(vad, preroll_ms=80)

            monkeypatch.setattr(path.transcriber, '_segmenter_factory', factory)
            monkeypatch.setattr(path.transcriber, 'prune_private', lambda: None)
            await path.transcriber.start()
            state = path.transcriber._room_states['synthetic-a']
            audio = HeldInput(stage)

            def delayed_clock():
                audio.wait('clock')
                return g.clock.now

            if stage == 'clock':
                monkeypatch.setattr(path.tap, '_clock', delayed_clock)
            else:
                class HeldPool(deque):
                    def popleft(self):
                        ticket = super().popleft()
                        audio.wait('reservation')
                        return ticket
                path.tap._pushes._free = HeldPool(path.tap._pushes._free)
            pushing = asyncio.create_task(asyncio.to_thread(path.tap.push, 'synthetic-a', audio))
            try:
                await until(audio.entered.is_set)
                await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
                g.clock.now += 1
                # The source reference can advance without changing the tap's
                # numeric boundary or transferring ownership of this old call.
                g.privacy.scopes.buffer_boundary('synthetic-a')
            finally:
                audio.release.set()
                await pushing
            await path.transcriber._consume_batch('synthetic-a', state)
            assert not calls
            assert not state.status()['state_retained']


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_callback_stamps_epochs_without_authority_queries_or_native_work(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room), AsyncExitStack() as stack:
        path = await stack.enter_async_context(ambient(g, tmp_path)) if transport == 'multi' else None
        queries = []

        def forbidden(*args, **kwargs):
            queries.append(True)
            raise AssertionError('authority queried on input callback')

        with monkeypatch.context() as guard:
            for method in ('ingress', 'buffer_boundary', 'epoch_available', 'source_for_capture'):
                guard.setattr(g.privacy.scopes, method, forbidden)
            guard.setattr(g.privacy.ledger, 'conversation_revoked', forbidden)
            invoke(g, transport, np.ones(1280, dtype=np.float32))
        assert not queries
        assert room.resource_status()['queued'] == 1
        if path is not None:
            assert path.tap.buffered_count() == 1


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_unstamped_callback_is_bound_when_another_input_creates_the_first_source(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        audio = HeldInput('clock')
        control_thread = threading.get_ident()

        def delayed_clock():
            if threading.get_ident() != control_thread:
                audio.wait('clock')
            return g.clock.now

        monkeypatch.setattr(g.loop, '_clock' if transport == 'single' else '_microphone_wall_clock', delayed_clock)
        pushing = asyncio.create_task(asyncio.to_thread(invoke, g, transport, audio))
        try:
            await until(audio.entered.is_set)
            invoke(g, transport, np.ones(1280, dtype=np.float32))
            await room.step()
            source = room.source
            assert source is not None
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            g.clock.now += 1
        finally:
            audio.release.set()
            await pushing
        await room.step()
        wake = g.wake if transport == 'single' else g.wakes['synthetic-a']
        assert not wake.history
        assert not room.resource_status()['has_state']


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_queued_future_dated_input_cannot_bypass_its_revoked_admission(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        monkeypatch.setattr(g.loop, '_clock' if transport == 'single' else '_microphone_wall_clock', lambda: g.clock.now+1)
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        assert room.resource_status()['queued'] == 1
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        await room.step()
        wake = g.wake if transport == 'single' else g.wakes['synthetic-a']
        assert not wake.history
        assert not room.resource_status()['has_state']


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_callback_from_a_retired_scope_does_not_join_the_new_conversation(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        audio = HeldInput('clock')

        def delayed_clock():
            audio.wait('clock')
            return g.clock.now

        monkeypatch.setattr(g.loop, '_clock' if transport == 'single' else '_microphone_wall_clock', delayed_clock)
        pushing = asyncio.create_task(asyncio.to_thread(invoke, g, transport, audio))
        try:
            await until(audio.entered.is_set)
            g.clock.now += 121
            fresh = g.privacy.source_for_capture(str(uuid4()), g.clock.now-.08, room.room_id, origin=MemoryOrigin.VOICE)
            assert fresh.conversation_id != source.conversation_id
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        finally:
            audio.release.set()
            await pushing
        await room.step()
        wake = g.wake if transport == 'single' else g.wakes['synthetic-a']
        assert not wake.history
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        assert room.source.conversation_id == fresh.conversation_id
        assert wake.history


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_unknown_input_admission_expires_even_if_delayed_clock_returns_a_fresh_date(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        audio = HeldInput('clock')

        def delayed_clock():
            audio.wait('clock')
            return g.clock.now

        monkeypatch.setattr(g.loop, '_clock' if transport == 'single' else '_microphone_wall_clock', delayed_clock)
        pushing = asyncio.create_task(asyncio.to_thread(invoke, g, transport, audio))
        try:
            await until(audio.entered.is_set)
            g.clock.now += 48*3600
        finally:
            audio.release.set()
            await pushing
        await room.step()
        wake = g.wake if transport == 'single' else g.wakes['synthetic-a']
        assert not wake.history and room.source is None
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        assert wake.history


@pytest.mark.parametrize('mismatch', ['authority', 'room', 'origin', 'missing'])
async def test_capture_rejects_admission_from_another_namespace_before_native_work(tmp_path, monkeypatch, mismatch):
    async with graph(tmp_path, monkeypatch, 'multi') as (g, room):
        other = PrivacyServices(str(tmp_path/'other.db'), clock=g.clock)
        await other.acall(other.initialize)
        try:
            scopes = other.scopes if mismatch == 'authority' else g.privacy.scopes
            epoch = scopes.ingress('synthetic-b' if mismatch == 'room' else room.room_id,
                origin=MemoryOrigin.AMBIENT if mismatch == 'origin' else MemoryOrigin.VOICE).current
            room.push(g.clock.now, np.ones(1280, dtype=np.float32), 1280,
                epoch=None if mismatch == 'missing' else epoch)
            await room.step()
            assert not g.wakes['synthetic-a'].history
            assert room.source is None
        finally:
            await other.begin_close()


async def test_ambient_native_waiter_keeps_its_admission_after_tap_drain(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, 'multi') as (g, voice):
        source = g.privacy.source_for_capture(str(uuid4()), g.clock.now, 'synthetic-a', origin=MemoryOrigin.AMBIENT)
        async with ambient(g, tmp_path) as path:
            calls = []

            def factory():
                def vad(audio):
                    calls.append('inference')
                    return 0.
                vad.reset = lambda: None
                return UtteranceSegmenter(vad, preroll_ms=80)

            monkeypatch.setattr(path.transcriber, '_segmenter_factory', factory)
            monkeypatch.setattr(path.transcriber, 'prune_private', lambda: None)
            drained = []
            drain = path.tap.drain_owned

            def observe_drain(room_id):
                drained.append(True)
                return drain(room_id)

            monkeypatch.setattr(path.tap, 'drain_owned', observe_drain)
            await path.transcriber.start()
            # Let the real idle worker finish its first empty batch.
            await until(lambda: bool(drained) and not path.transcriber._batches)
            state = path.transcriber._room_states['synthetic-a']
            await state._lock.acquire()
            consuming = None
            try:
                path.tap.push('synthetic-a', np.ones((1280, 1), dtype=np.float32), ts=g.clock.now+1)
                consuming = asyncio.create_task(path.transcriber._consume_batch('synthetic-a', state))
                await until(lambda: bool(state._batches))
                assert path.tap.buffered_count() == 0
                await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
                g.clock.now += .2
            finally:
                state._lock.release()
                if consuming is not None:
                    await consuming
            assert not calls
            assert not state.status()['state_retained']


@pytest.mark.parametrize('transport', ['single', 'multi'])
@pytest.mark.parametrize('outcome', ['commit', 'rollback'])
async def test_revocation_retires_old_admission_before_commit_and_admits_fresh_input(tmp_path, monkeypatch, transport, outcome):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        audio = HeldInput('clock')
        commit_entered, commit_release = threading.Event(), threading.Event()
        connection = g.privacy.ledger._db

        class HeldCommit:
            def __getattr__(self, name):
                return getattr(connection, name)

            def commit(self):
                commit_entered.set()
                assert commit_release.wait(3)
                if outcome == 'rollback':
                    raise OSError('synthetic_commit_failure')
                connection.commit()

        def delayed_clock():
            audio.wait('clock')
            return g.clock.now

        monkeypatch.setattr(g.loop, '_clock' if transport == 'single' else '_microphone_wall_clock', delayed_clock)
        g.privacy.ledger._db = HeldCommit()
        pushing = asyncio.create_task(asyncio.to_thread(invoke, g, transport, audio))
        deleting = None
        try:
            await until(audio.entered.is_set)
            deleting = asyncio.create_task(g.privacy.acall(
                g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR))
            await until(commit_entered.is_set)
            assert g.privacy.ledger.conversation_revoked(source.conversation_id)
            g.clock.now += 1
            audio.release.set()
            await pushing
            invoke(g, transport, np.full(1280, .25, dtype=np.float32))
            await room.step()
            wake = g.wake if transport == 'single' else g.wakes['synthetic-a']
            assert len(wake.history) == 1 and np.allclose(wake.history[0], .25)
            assert room.source.conversation_id != source.conversation_id
        finally:
            audio.release.set()
            commit_release.set()
            await pushing
            try:
                if deleting is not None:
                    if outcome == 'rollback':
                        with pytest.raises(OSError, match='synthetic_commit_failure'):
                            await deleting
                    else:
                        await deleting
            finally:
                g.privacy.ledger._db = connection
        assert g.privacy.sources_available((source,)) == (outcome == 'rollback')
