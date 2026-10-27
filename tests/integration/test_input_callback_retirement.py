"""Real input callbacks and idle capture state survive observer cancellation."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
import json
import threading
from uuid import uuid4
from weakref import ref

import numpy as np
import pytest

from src.core.owned_work import WorkCloseTimeout
from src.ambient.tap import MultiChannelTap
from tests.integration.test_capture_producer_privacy import ACTOR, graph as multi_graph, until
from tests.integration.test_single_input_privacy import graph as single_graph

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def graph(path, monkeypatch, transport):
    if transport == 'single':
        async with single_graph(path, monkeypatch, launch=False) as g:
            await g.loop.start()
            await g.loop._input.open()
            yield g, g.loop._room
    else:
        async with multi_graph(path, controllers=False) as g:
            yield g, g.loop._capture_rooms['synthetic-a']


def invoke(g, transport, audio):
    if transport == 'single':
        g.backend.stream.emit(audio)
    else:
        g.callbacks['synthetic-a'](audio[:, None], len(audio), None, None)


class HeldInput(np.ndarray):
    def __new__(cls, stage):
        result = np.ones(1280, dtype=np.float32).view(cls)
        result.stage = stage
        result.entered, result.release = threading.Event(), threading.Event()
        return result

    def __array_finalize__(self, parent):
        if parent is not None:
            self.stage = getattr(parent, 'stage', None)
            self.entered = getattr(parent, 'entered', None)
            self.release = getattr(parent, 'release', None)

    def wait(self, stage):
        if self.stage == stage:
            self.entered.set()
            assert self.release.wait(3)

    def any(self, *args, **kwargs):
        self.wait('any')
        return np.asarray(self).any(*args, **kwargs)

    def copy(self, *args, **kwargs):
        self.wait('copy')
        return np.asarray(self).copy(*args, **kwargs)


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_empty_queue_fence_cannot_receive_an_old_inflight_copy(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        audio = HeldInput('copy')
        pushing = asyncio.create_task(asyncio.to_thread(invoke, g, transport, audio))
        try:
            await until(audio.entered.is_set)
            assert room.resource_status()['queued'] == 0
            assert room.callbacks.status()['active'] == 1
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            room.publish_boundary()
        finally:
            audio.release.set()
            await pushing
        assert room.resource_status()['queued'] == 0
        assert room.callbacks.status()['active'] == 0


@pytest.mark.parametrize('transport', ['single', 'multi'])
@pytest.mark.parametrize('stage', ['copy', 'reserved'])
async def test_writer_lease_waits_for_actual_input_callback_after_driver_close(tmp_path, monkeypatch, transport, stage):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        audio = HeldInput('copy' if stage == 'copy' else None)
        entered, release = audio.entered, audio.release
        if stage == 'reserved':
            class HeldPool(deque):
                def popleft(self):
                    ticket = super().popleft()
                    entered.set()
                    assert release.wait(3)
                    return ticket
            room.callbacks._free = HeldPool(room.callbacks._free)
        pushing = asyncio.create_task(asyncio.to_thread(invoke, g, transport, audio))
        try:
            await until(entered.is_set)
            assert room.callbacks.status()['active'] == 1
            assert g.loop.conversation_busy(str(uuid4()))
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
            assert not g.loop.begin_close().done()
            if transport == 'single':
                assert g.loop._input.resource_status()['state'] == 'closed'
        finally:
            release.set()
            await pushing
            await g.privacy.begin_close()
        assert room.callbacks.status()['active'] == 0
        assert not g.privacy.resource_status()['writer_lease']
        assert room.resource_status()['queued'] == 0


async def test_multichannel_timestamp_precedes_array_work_and_cannot_rename_retired_input(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, 'multi') as (g, room):
        invoke(g, 'multi', np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        audio = HeldInput('any')
        pushing = asyncio.create_task(asyncio.to_thread(invoke, g, 'multi', audio))
        try:
            await until(audio.entered.is_set)
            assert room.callbacks.status()['active'] == 1
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            g.clock.now += 300
            room.publish_boundary()
        finally:
            audio.release.set()
            await pushing
        await room.step()
        assert room.source is None and room.resource_status()['queued'] == 0


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_idle_buffers_remain_busy_until_explicit_native_prune_without_new_input(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        wake = g.wake if transport == 'single' else g.wakes['synthetic-a']
        references = [ref(value) for value in wake.history]
        assert references and all(value() is not None for value in references)
        assert g.loop.conversation_busy(source.conversation_id)
        assert not g.loop.conversation_busy(str(uuid4()))
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        await room.prune_private()
        assert all(value() is None for value in references)
        assert not g.loop.conversation_busy(source.conversation_id)
        assert room.source is None and not room.resource_status()['has_state']
        status = json.dumps(room.resource_status())
        assert source.conversation_id not in status and source.capture_id not in status


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_failed_native_reset_retains_idle_affinity_and_writer_lease_until_retry(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        wake = g.wake if transport == 'single' else g.wakes['synthetic-a']
        references = [ref(value) for value in wake.history]
        reset = wake.reset
        failed = [True]

        def retry_reset():
            if failed[0]:
                raise ValueError('Synthetic private reset failure')
            reset()

        monkeypatch.setattr(wake, 'reset', retry_reset)
        try:
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            await room.prune_private()
            assert room.resource_status()['reset_failed']
            assert room.conversation_busy(source.conversation_id)
            assert any(value() is not None for value in references)
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
        finally:
            failed[0] = False
            await g.privacy.begin_close()
        assert all(value() is None for value in references)
        assert not room.resource_status()['reset_failed']
        assert not g.privacy.resource_status()['writer_lease']


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_cancelled_idle_prune_retains_actual_reset_and_lease(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        gate = g.gates['reset'] if transport == 'single' else g.gates['synthetic-a']['reset']
        gate.block()
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        observer = asyncio.create_task(room.prune_private())
        errors = []
        try:
            await until(gate.entered.is_set)
            observer.cancel()
            try:
                await observer
            except asyncio.CancelledError as error:
                errors.append(error)
            assert errors and room.conversation_busy(source.conversation_id)
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
        finally:
            gate.release.set()
            await g.privacy.begin_close()
        assert not g.privacy.resource_status()['writer_lease']
        assert room.source is None and not room.resource_status()['has_state']


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_callback_capacity_rejects_before_touching_another_input(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        audio = [HeldInput('copy') for _ in range(room.callbacks.status()['capacity'])]
        pending = [asyncio.create_task(asyncio.to_thread(invoke, g, transport, item)) for item in audio]
        touched = []
        class Rejected:
            def __len__(self):
                return 1280
            def __getitem__(self, key):
                return self
            def any(self):
                touched.append(True)
                return True
            def copy(self):
                touched.append(True)
                return np.ones((1280, 1), dtype=np.float32)
        try:
            await until(lambda: all(item.entered.is_set() for item in audio))
            assert room.callbacks.status()['active'] == len(audio)
            await asyncio.wait_for(asyncio.to_thread(invoke, g, transport, Rejected()), 1)
            assert not touched and room.callbacks.status()['drops'] == 1
        finally:
            for item in audio:
                item.release.set()
            await asyncio.gather(*pending)
        assert room.callbacks.status()['active'] == 0


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_callback_failure_releases_slot_and_does_not_export_raw_exception(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        references = []
        class Payload:
            pass
        class Failed:
            def __len__(self):
                return 1280
            def __getitem__(self, key):
                return self
            def any(self):
                return self.copy()
            def copy(self):
                raw = Payload()
                references.append(ref(raw))
                raise ValueError(raw)
        await asyncio.to_thread(invoke, g, transport, Failed())
        assert references and references[0]() is None
        assert room.callbacks.status()['active'] == 0


async def test_ambient_tee_keeps_acquisition_time_across_a_delayed_copy(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, 'multi') as (g, room):
        tap = MultiChannelTap(clock=g.clock)
        tap.register_room('synthetic-a')
        g.loop.attach_ambient(tap)
        audio = HeldInput('copy')
        timestamp = g.clock.now - len(audio)/g.loop.sample_rate
        pending = asyncio.create_task(asyncio.to_thread(invoke, g, 'multi', audio))
        try:
            await until(audio.entered.is_set)
            g.clock.now += 300
        finally:
            audio.release.set()
            await pending
        rows = tap.drain('synthetic-a')
        assert len(rows) == 1 and rows[0][0] == timestamp
        assert room.callbacks.status()['active'] == 0


async def test_single_input_ambient_buffers_remain_busy_after_capture_epoch_reset(tmp_path, monkeypatch):
    async with single_graph(tmp_path, monkeypatch, launch=False, ambient=True) as g:
        await g.loop.start()
        await g.loop._input.open()
        room = g.loop._room
        invoke(g, 'single', np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        references = [ref(audio) for _, audio in g.loop._ambient_frames]
        assert references and all(value() is not None for value in references)
        await room.call(room._reset, sources=())
        assert room.source is None and not room.conversation_busy(source.conversation_id)
        assert g.loop.conversation_busy(source.conversation_id)
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        await g.loop.prune_private()
        assert all(value() is None for value in references)
        assert not g.loop.conversation_busy(source.conversation_id)


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_delayed_callback_clock_cannot_rename_input_after_a_privacy_fence(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        invoke(g, transport, np.ones(1280, dtype=np.float32))
        await room.step()
        source = room.source
        entered, release = threading.Event(), threading.Event()
        main_thread = threading.get_ident()
        def clock():
            if threading.get_ident() != main_thread:
                entered.set()
                assert release.wait(3)
            return g.clock()
        monkeypatch.setattr(g.loop, '_clock' if transport == 'single' else '_microphone_wall_clock', clock)
        pending = asyncio.create_task(asyncio.to_thread(invoke, g, transport, np.ones(1280, dtype=np.float32)))
        try:
            await until(entered.is_set)
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            room.publish_boundary()
            g.clock.now += 300
        finally:
            release.set()
            await pending
        await room.step()
        assert room.source is None and room.resource_status()['queued'] == 0
