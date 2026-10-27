"""The real microphone tee owns ambient copies independently of voice work."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from uuid import uuid4
from weakref import ref

import numpy as np
import pytest

from src.ambient.store import AmbientStore
from src.ambient.tap import MultiChannelTap
from src.ambient.transcriber import AmbientPath, AmbientTranscriber
from src.memory.fact_ledger import MemoryOrigin
from tests.integration.test_capture_producer_privacy import ACTOR, graph, until
from tests.integration.test_input_callback_retirement import HeldInput, invoke

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def ambient(g, tmp_path):
    tap = MultiChannelTap(clock=g.clock)
    tap.register_room('synthetic-a')
    store = AmbientStore(str(tmp_path / 'ambient.db'), source_policy=g.privacy.ledger)
    await store.init()
    transcriber = AmbientTranscriber(tap, None, None, None, None, None,
        store, ['synthetic-a'], privacy=g.privacy)
    path = AmbientPath(tap, transcriber, store, None)
    g.loop.attach_ambient(tap, transcriber)
    try:
        yield path
    finally:
        await path.stop()


async def test_ambient_unknown_copy_is_pending_even_with_known_unrelated_voice_source(tmp_path):
    async with graph(tmp_path, controllers=False) as g:
        g.feed()
        await g.loop._capture_rooms['synthetic-a'].step()
        source = g.privacy.source_for_capture(str(uuid4()), g.clock.now, 'synthetic-a', origin=MemoryOrigin.AMBIENT)
        assert source.conversation_id != g.source().conversation_id
        async with ambient(g, tmp_path) as path:
            audio = HeldInput('copy')
            pushing = asyncio.create_task(asyncio.to_thread(invoke, g, 'multi', audio))
            try:
                await until(audio.entered.is_set)
                assert path.transcriber.conversation_busy(source.conversation_id)
            finally:
                audio.release.set()
                await pushing


@pytest.mark.parametrize('stage', ['reserved', 'any', 'copy'])
async def test_ambient_close_retains_store_until_actual_input_branch_retires(tmp_path, stage):
    async with graph(tmp_path, controllers=False) as g:
        async with ambient(g, tmp_path) as path:
            room = g.loop._capture_rooms['synthetic-a']
            audio = HeldInput(stage)
            if stage == 'reserved':
                class HeldPool(deque):
                    def popleft(self):
                        ticket = super().popleft()
                        audio.entered.set()
                        assert audio.release.wait(3)
                        return ticket
                room.callbacks._free = HeldPool(room.callbacks._free)
            pushing = asyncio.create_task(asyncio.to_thread(invoke, g, 'multi', audio))
            closing = None
            try:
                await until(audio.entered.is_set)
                closing = path.begin_close()
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(closing), .03)
                assert path.store._db is not None
            finally:
                audio.release.set()
                await pushing
                if closing is not None:
                    await closing
            assert path.store._db is None
            assert path.tap.buffered_count() == 0
            assert room.resource_status()['queued'] == 1


async def test_stopping_ambient_does_not_wait_for_the_following_voice_copy(tmp_path, monkeypatch):
    async with graph(tmp_path, controllers=False) as g:
        async with ambient(g, tmp_path) as path:
            room = g.loop._capture_rooms['synthetic-a']
            audio = HeldInput('copy')
            push = room.push

            def held_voice(timestamp, chunk, frames, **kwargs):
                audio.wait('copy')
                push(timestamp, chunk, frames, **kwargs)

            # Hold only the voice branch, after the ambient copy has been handed off.
            monkeypatch.setattr(room, 'push', held_voice)
            pushing = asyncio.create_task(asyncio.to_thread(invoke, g, 'multi', np.ones(1280, dtype=np.float32)))
            try:
                await until(audio.entered.is_set)
                await asyncio.wait_for(path.stop(), .1)
                assert room.callbacks.status()['active'] == 1
                assert not g.loop._closed
            finally:
                audio.release.set()
                await pushing
            assert room.resource_status()['queued'] == 1


@pytest.mark.parametrize('stage', ['clock', 'any', 'copy'])
async def test_ambient_revocation_rejects_its_old_branch_and_preserves_voice(tmp_path, monkeypatch, stage):
    async with graph(tmp_path, controllers=False) as g:
        g.feed()
        room = g.loop._capture_rooms['synthetic-a']
        await room.step()
        voice_source = g.source()
        source = g.privacy.source_for_capture(str(uuid4()), g.clock.now, 'synthetic-a', origin=MemoryOrigin.AMBIENT)
        async with ambient(g, tmp_path) as path:
            path.transcriber.prune_private()
            g.clock.now += .2
            audio = HeldInput(stage)
            if stage == 'clock':
                def held_clock():
                    audio.wait('clock')
                    return g.clock.now
                monkeypatch.setattr(g.loop, '_microphone_wall_clock', held_clock)
            pushing = asyncio.create_task(asyncio.to_thread(invoke, g, 'multi', audio))
            try:
                await until(audio.entered.is_set)
                await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
                g.clock.now += 1
                path.transcriber.prune_private()
                assert path.transcriber.conversation_busy(source.conversation_id)
            finally:
                audio.release.set()
                await pushing
            assert path.tap.buffered_count() == 0
            assert not path.transcriber.conversation_busy(source.conversation_id)
            assert room.resource_status()['queued'] == 1
            await room.step()
            assert g.source().conversation_id == voice_source.conversation_id


async def test_replacing_tap_does_not_redirect_an_already_admitted_copy(tmp_path):
    async with graph(tmp_path, controllers=False) as g:
        async with ambient(g, tmp_path) as path:
            audio = HeldInput('copy')
            replacement = MultiChannelTap(clock=g.clock)
            replacement.register_room('synthetic-a')
            pushing = asyncio.create_task(asyncio.to_thread(invoke, g, 'multi', audio))
            try:
                await until(audio.entered.is_set)
                g.loop.attach_ambient(replacement)
                closing = path.begin_close()
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(closing), .03)
            finally:
                audio.release.set()
                await pushing
                await path.begin_close()
            assert replacement.buffered_count() == path.tap.buffered_count() == 0
            g.feed()
            assert replacement.buffered_count() == 1
            await replacement.begin_close()


async def test_tap_close_owns_direct_push_arguments_through_a_blocked_clock():
    audio = HeldInput('clock')
    reference = ref(audio)

    def clock():
        audio.wait('clock')
        return 1.

    tap = MultiChannelTap(clock=clock)
    tap.register_room('synthetic-a')
    pushing = asyncio.create_task(asyncio.to_thread(tap.push, 'synthetic-a', audio))
    try:
        await until(audio.entered.is_set)
        closing = tap.begin_close()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(closing), .03)
    finally:
        audio.release.set()
        await pushing
        await tap.begin_close()
    audio = None
    assert reference() is None
    assert tap.buffered_count() == 0


async def test_buffered_unknown_ambient_input_remains_pending_after_callback_returns(tmp_path):
    async with graph(tmp_path, controllers=False) as g:
        async with ambient(g, tmp_path) as path:
            g.feed()
            assert path.tap.buffered_count() == 1
            assert path.transcriber.conversation_busy(str(uuid4()))
            path.tap.drain('synthetic-a')
            assert not path.transcriber.conversation_busy(str(uuid4()))


async def test_cancelled_stop_observer_cannot_release_ambient_ownership(tmp_path):
    async with graph(tmp_path, controllers=False) as g:
        async with ambient(g, tmp_path) as path:
            audio = HeldInput('copy')
            pushing = asyncio.create_task(asyncio.to_thread(invoke, g, 'multi', audio))
            try:
                await until(audio.entered.is_set)
                observer = asyncio.create_task(path.stop())
                await until(lambda: path._closing is not None)
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert not path.begin_close().done()
                assert path.store._db is not None
            finally:
                audio.release.set()
                await pushing
                await path.begin_close()
            assert path.store._db is None


async def test_closed_ambient_skips_its_copy_while_voice_continues(tmp_path):
    async with graph(tmp_path, controllers=False) as g:
        async with ambient(g, tmp_path) as path:
            await path.stop()

            class Counted(np.ndarray):
                copies = 0

                def copy(self, *args, **kwargs):
                    type(self).copies += 1
                    return np.asarray(self).copy(*args, **kwargs)

            invoke(g, 'multi', np.ones(1280, dtype=np.float32).view(Counted))
            assert Counted.copies == 1
            room = g.loop._capture_rooms['synthetic-a']
            assert room.resource_status()['queued'] == 1
            await room.step()
            assert g.source() is not None
            assert not path.tap.ingress_pending()
