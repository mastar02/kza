"""Receiver affinity stays continuous across capture futures and native work."""

import asyncio
from contextlib import asynccontextmanager
import json
from uuid import uuid4
from weakref import ref

import numpy as np
import pytest

from src.core.owned_work import WorkCloseTimeout
from src.pipeline.command_event import CommandEvent
from tests.integration.test_capture_producer_privacy import ACTOR, graph as multi_graph, until
from tests.integration.test_single_input_privacy import graph as single_graph

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def graph(path, monkeypatch, transport, *, blocked=None):
    if transport == 'single':
        async with single_graph(path, monkeypatch, blocked=blocked) as g:
            yield g, g.loop._room
    else:
        async with multi_graph(path, blocked=blocked) as g:
            yield g, g.loop._capture_rooms['synthetic-a']


def status(g, transport):
    return g.loop.resource_status() if transport == 'single' else g.loop.capture_status()


def controller(g, transport):
    return g.loop._controller if transport == 'single' else g.loop._capture_controllers['synthetic-a']


def install_poll(room, monkeypatch):
    emitted = False
    def poll():
        nonlocal emitted
        if emitted:
            return None
        emitted = True
        return CommandEvent(np.ones(1600, dtype=np.float32), room.room_id, capture_complete=True)
    monkeypatch.setattr(room, '_poll', poll)


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_receiver_knows_source_while_native_capture_is_still_blocked(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport, blocked='wake') as (g, room):
        g.feed()
        gate = g.gates['wake'] if transport == 'single' else g.gates['synthetic-a']['wake']
        try:
            await until(gate.entered.is_set)
            source = room.source
            assert g.loop.conversation_busy(source.conversation_id)
            assert not g.loop.conversation_busy(str(uuid4()))
            assert status(g, transport)['source_scopes']['unknown'] == 0
            assert source.conversation_id not in json.dumps(status(g, transport))
        finally:
            gate.release.set()


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_receiver_owns_result_after_kernel_finishes_before_await_returns(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        install_poll(room, monkeypatch)
        entered, release = asyncio.Event(), asyncio.Event()
        arrays, conversations = [], []
        original = room.step
        async def held_step(*, handoff=None):
            result = await original(handoff=handoff)
            if result is not None:
                arrays.append(ref(result.audio))
                conversations.append(result.conversation_id)
                entered.set()
                await release.wait()
            return result
        async def command(event):
            return {'success': False}
        monkeypatch.setattr(room, 'step', held_step)
        g.loop.on_command(command)
        g.feed()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            conversation_id = conversations[0]
            assert not room.conversation_busy(conversation_id)
            assert g.loop.conversation_busy(conversation_id)
            assert not g.loop.conversation_busy(str(uuid4()))
            assert status(g, transport)['source_scopes']['unknown'] == 0
            assert arrays[0]() is not None
            locals_view = controller(g, transport).get_coro().cr_frame.f_locals
            assert not {'event', 'step', 'batches'} & set(locals_view.keys())
            assert isinstance(locals_view['batch'], asyncio.Task)
        finally:
            release.set()
        await until(lambda: arrays[0]() is None and not g.loop.conversation_busy(conversation_id))


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_noncooperative_result_handoff_keeps_lease_and_cannot_publish_after_close(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        install_poll(room, monkeypatch)
        entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        arrays, conversations, delivered = [], [], []
        original = room.step
        async def held_step(*, handoff=None):
            result = None
            was_cancelled = False
            try:
                result = await original(handoff=handoff)
                if result is not None:
                    arrays.append(ref(result.audio))
                    conversations.append(result.conversation_id)
                    entered.set()
                    while not release.is_set():
                        try:
                            await asyncio.shield(release.wait())
                        except asyncio.CancelledError:
                            was_cancelled = True
                            cancelled.set()
                if was_cancelled:
                    raise asyncio.CancelledError
                return result
            finally:
                result = None
        async def command(event):
            delivered.append(True)
            return {'success': False}
        monkeypatch.setattr(room, 'step', held_step)
        g.loop.on_command(command)
        g.feed()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            conversation_id = conversations[0]
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert cancelled.is_set() and arrays[0]() is not None
            assert g.privacy.resource_status()['writer_lease']
            assert g.loop.conversation_busy(conversation_id)
            assert not g.loop.conversation_busy(str(uuid4()))
        finally:
            release.set()
            await g.privacy.begin_close()
        assert arrays[0]() is None and not delivered
        assert not g.loop.snapshot_operations()


@pytest.mark.parametrize('stop', ['cancel', 'revoke'])
async def test_early_native_parser_keeps_receiver_affinity_after_controller_exit(tmp_path, stop):
    async with multi_graph(tmp_path, blocked='early', early=True) as g:
        g.wakes['synthetic-a'].fire = True
        g.feed()
        try:
            await until(g.gates['synthetic-a']['early'].entered.is_set)
            source = g.source()
            if stop == 'revoke':
                await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            else:
                g.loop.room_streams['synthetic-a'].early_task.cancel()
            await until(lambda: not g.loop._early_tasks)
            assert g.loop.conversation_busy(source.conversation_id)
            assert not g.loop.conversation_busy(str(uuid4()))
        finally:
            g.gates['synthetic-a']['early'].release.set()
            await until(lambda: not g.loop._capture_native._tasks)
        assert not g.loop.room_streams['synthetic-a'].transcript


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_noncooperative_callback_keeps_exact_source_scope_until_exit(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        install_poll(room, monkeypatch)
        entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        conversations = []
        async def command(event):
            conversations.append(event.conversation_id)
            entered.set()
            try:
                await asyncio.shield(release.wait())
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            return {'text': 'synthetic private result', 'success': False}
        g.loop.on_command(command)
        g.feed()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            conversation_id = conversations[0]
            await g.privacy.acall(g.privacy.ledger.delete_conversation, conversation_id, ACTOR)
            await asyncio.wait_for(cancelled.wait(), 3)
            assert g.loop.conversation_busy(conversation_id)
            assert not g.loop.conversation_busy(str(uuid4()))
        finally:
            release.set()
        dispatches = g.loop._dispatches if transport == 'single' else g.loop._dispatch_tasks
        await until(lambda: not dispatches and not g.loop.conversation_busy(conversation_id))


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_resetting_known_idle_epoch_does_not_mark_unrelated_work_unknown(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport, blocked='reset') as (g, room):
        g.feed()
        gate = g.gates['reset'] if transport == 'single' else g.gates['synthetic-a']['reset']
        wake = g.wake if transport == 'single' else g.wakes['synthetic-a']
        try:
            await until(lambda: bool(wake.history))
            source = room.source
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            await until(gate.entered.is_set)
            assert g.loop.conversation_busy(source.conversation_id)
            assert not g.loop.conversation_busy(str(uuid4()))
        finally:
            gate.release.set()


@pytest.mark.parametrize('stop', ['cancel', 'revoke'])
async def test_ambient_native_model_keeps_receiver_scope_after_analysis_exit(tmp_path, monkeypatch, stop):
    async with single_graph(tmp_path, monkeypatch, ambient=True, blocked='model') as g:
        for _ in range(13):
            g.feed()
        try:
            await until(g.gates['model'].entered.is_set)
            source = g.loop._room.source
            if stop == 'cancel':
                next(iter(g.loop._analyses)).cancel()
            else:
                await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            await until(lambda: not g.loop._analyses)
            assert g.loop.conversation_busy(source.conversation_id)
            assert not g.loop.conversation_busy(str(uuid4()))
            assert g.audio_refs and g.audio_refs[0]() is not None
        finally:
            g.gates['model'].release.set()
            await until(lambda: not g.detector.snapshot_operations())
        await until(lambda: g.audio_refs[0]() is None)
        assert not g.events


@pytest.mark.parametrize('failure', ['metadata', 'capacity'])
async def test_dispatch_admission_failure_does_not_retain_raw_event(tmp_path, failure):
    async with multi_graph(tmp_path, controllers=False) as g:
        source = g.privacy.source_for_capture(str(uuid4()), g.clock(), 'synthetic-a')
        event = CommandEvent(np.ones(1600, dtype=np.float32), 'synthetic-a')
        reference = ref(event)
        if failure == 'capacity':
            event.capture_id, event.conversation_id = source.capture_id, source.conversation_id
            event.captured_at, event.source_expires_at = source.captured_at, source.expires_at
        async def dispatch(value):
            try:
                g.loop._start_dispatch(value)
            finally:
                value = None
        g.loop._scopes._capacity = 1
        with g.loop._scopes.operation((source,)):
            rejected = asyncio.create_task(dispatch(event))
            del event
            with pytest.raises((ValueError, RuntimeError)) as retained:
                await rejected
            assert retained.value is not None and reference() is None
            assert not g.loop._dispatch_tasks


async def test_barge_scope_admission_failure_retires_its_task_entry(tmp_path):
    async with multi_graph(tmp_path, controllers=False) as g:
        source = g.privacy.source_for_capture(str(uuid4()), g.clock(), 'synthetic-a')
        g.loop._scopes._capacity = 1
        with g.loop._scopes.operation((source,)):
            task = asyncio.create_task(g.loop._tracked_barge_in(g.loop.room_streams['synthetic-a'], source))
            with pytest.raises(RuntimeError, match='producer_scope_capacity'):
                await task
            assert not g.loop._capture_tasks


@pytest.mark.parametrize('transport', ['single', 'multi'])
async def test_revoked_result_does_not_stop_receiver_from_accepting_fresh_capture(tmp_path, monkeypatch, transport):
    async with graph(tmp_path, monkeypatch, transport) as (g, room):
        entered, release = asyncio.Event(), asyncio.Event()
        old, received = [], []
        original = room.step
        def poll():
            return CommandEvent(np.ones(1600, dtype=np.float32), room.room_id, capture_complete=True)
        async def interleaved(*, handoff=None):
            result = None
            try:
                result = await original(handoff=handoff)
                if result is not None and not old:
                    old.append(result.conversation_id)
                    entered.set()
                    await release.wait()
                    from src.memory.prompt_context import SourcesInvalidated
                    raise SourcesInvalidated('synthetic_handoff_invalidated')
                return result
            finally:
                result = None
        async def command(event):
            received.append(event.conversation_id)
            return {'success': False}
        monkeypatch.setattr(room, '_poll', poll)
        monkeypatch.setattr(room, 'step', interleaved)
        g.loop.on_command(command)
        g.feed()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await g.privacy.acall(g.privacy.ledger.delete_conversation, old[0], ACTOR)
        finally:
            release.set()
        await until(lambda: not (g.loop._batches if transport == 'single' else g.loop._capture_batches))
        assert not controller(g, transport).done()
        g.clock.now += 1
        g.feed()
        await until(lambda: bool(received))
        assert received[0] != old[0] and old[0] not in received
