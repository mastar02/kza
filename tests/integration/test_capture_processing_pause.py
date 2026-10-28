"""Reusable processing pause across actual native owners and fresh capture epochs."""

import asyncio
from collections import deque
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from src.conversation.follow_up_mode import FollowUpMode
from src.core.owned_work import WorkCloseTimeout
from src.memory.producer_context import producer_context
from src.memory.prompt_context import SourcesInvalidated
from src.pipeline.command_event import CommandEvent
from src.wakeword.clip_store import WakeClipStore
from src.wakeword.wake_clip_writer import WakeClipWriter
from tests.integration.test_capture_producer_privacy import graph as multi_graph, until
from tests.integration.test_command_producer_privacy import graph as command_graph
from tests.integration.test_input_driver_pause import Driver, portaudio
from tests.integration.test_single_input_privacy import graph as single_graph
from tests.integration.test_wake_clip_privacy import gate_encoder
from tests.mocks.wake_clips import graph as clip_graph

pytestmark = pytest.mark.asyncio


def copying_submission(g, monkeypatch):
    """Hold a real PCM copy without changing NumPy for unrelated modules."""
    entered, release = threading.Event(), threading.Event()
    audio = np.ones(1600, dtype=np.float32)
    class HeldArray:
        def copy(self):
            entered.set()
            assert release.wait(4)
            return audio.copy()
    monkeypatch.setattr('src.wakeword.wake_clip_writer.np', SimpleNamespace(
        asarray=lambda value, **kwargs: HeldArray() if value is audio else np.asarray(value, **kwargs),
        float32=np.float32, clip=np.clip, int16=np.int16))
    submitting = asyncio.create_task(asyncio.to_thread(g.writer.submit, 'synthetic-room', .9, audio, sources=(g.source(),)))
    return entered, release, submitting


@pytest.mark.parametrize('stage', ['wake', 'vad', 'early'])
async def test_multi_pause_drains_native_work_discards_state_and_preserves_reusable_controllers(tmp_path, monkeypatch, stage):
    driver = Driver()
    portaudio(monkeypatch, [driver])
    async with multi_graph(tmp_path, blocked=stage, early=stage == 'early') as g:
        g.wakes['synthetic-a'].fire = stage != 'wake'
        g.feed()
        draining = None
        try:
            await until(g.gates['synthetic-a'][stage].entered.is_set)
            source, epoch = g.source(), g.loop._capture_rooms['synthetic-a'].input_epoch
            g.loop.pause_admission()
            draining = asyncio.create_task(g.loop.drain_pause())
            await asyncio.sleep(.02)
            assert not draining.done() and epoch.retired
            with pytest.raises(RuntimeError, match='still_owned'):
                g.loop.resume_admission()
        finally:
            g.gates['synthetic-a'][stage].release.set()
            if draining is not None:
                await draining
        assert g.privacy.sources_available((source,))
        await g.loop.prune_private()
        assert not g.wakes['synthetic-a'].history and not g.loop.room_streams['synthetic-a'].audio_buffer
        assert all(not task.done() for task in g.loop._capture_controllers.values())
        g.loop.resume_admission()
        await until(lambda: driver.started == 1)
        g.callbacks['synthetic-a'] = driver.callback
        g.clock.now += .2
        g.feed()
        await until(lambda: g.source() is not None)
        assert g.source().conversation_id != source.conversation_id
        assert g.loop._capture_rooms['synthetic-a'].input_epoch is not epoch


async def test_reset_failure_keeps_capture_sealed_and_can_be_repaired_without_restarting(tmp_path, monkeypatch):
    portaudio(monkeypatch, [Driver()])
    async with multi_graph(tmp_path, controllers=False) as g:
        wake = g.wakes['synthetic-a']
        reset = wake.reset
        def failed():
            raise RuntimeError('synthetic_reset_failure')
        wake.reset = failed
        g.loop.pause_admission()
        with pytest.raises(RuntimeError, match='reset_failed'):
            await g.loop.drain_pause()
        assert g.loop.capture_status()['paused'] and not g.loop._inputs.accepting
        with pytest.raises(RuntimeError):
            g.loop.resume_admission()
        wake.reset = reset
        await g.loop.drain_pause()
        g.loop.resume_admission()
        assert not g.loop.capture_status()['paused']


async def test_multi_pause_retains_noncooperative_command_cleanup_and_denies_post_callback(tmp_path):
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    posts = []
    async with multi_graph(tmp_path, controllers=False) as g:
        g.feed()
        await g.loop._capture_rooms['synthetic-a'].step()
        source = g.source()
        async def command(event):
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            return {'success': True}
        async def post(result, event):
            posts.append(result)
        g.loop.on_command(command)
        g.loop.on_post_command(post)
        event = CommandEvent(np.ones(1600), room_id='synthetic-a', capture_id=source.capture_id,
            conversation_id=source.conversation_id, captured_at=source.captured_at, source_expires_at=source.expires_at)
        g.loop._start_dispatch(event)
        await entered.wait()
        g.loop.pause_admission()
        draining = asyncio.create_task(g.loop.drain_pause())
        try:
            await cancelled.wait()
            assert not draining.done()
            with pytest.raises(RuntimeError):
                g.loop.resume_admission()
        finally:
            release.set()
            await draining
        assert not posts and not g.loop.snapshot_operations()


@pytest.mark.parametrize('stage', ['wake', 'endpoint', 'model', 'callback', 'stop', 'close'])
async def test_single_pause_waits_for_actual_models_callbacks_and_driver_close(tmp_path, monkeypatch, stage):
    ambient = stage in ('model', 'callback')
    async with single_graph(tmp_path, monkeypatch, blocked=stage, ambient=ambient) as g:
        g.loop.follow_up = FollowUpMode()
        g.wake.fire = stage == 'endpoint'
        draining = None
        try:
            if stage not in ('stop', 'close'):
                g.feed(frames=16000 if ambient else 1280)
                await until(g.gates[stage].entered.is_set)
            g.loop.pause_admission()
            draining = asyncio.create_task(g.loop.drain_pause())
            await until(g.gates[stage].entered.is_set)
            await asyncio.sleep(.02)
            assert not draining.done()
            with pytest.raises(RuntimeError, match='still_owned'):
                g.loop.resume_admission()
        finally:
            g.gates[stage].release.set()
            if draining is not None:
                await draining
        assert g.backend.stream.closed and not g.loop._ambient_frames and not g.loop._ambient_ready
        assert g.loop._ambient_samples == 0 and g.loop._ambient_end is None
        await g.loop.prune_private()
        assert not g.wake.history and not g.loop._state.audio
        assert not g.loop._controller.done()
        if stage == 'model':
            assert not g.events


async def test_single_resume_uses_fresh_driver_and_epochs_without_restoring_old_preroll(tmp_path, monkeypatch):
    async with single_graph(tmp_path, monkeypatch) as g:
        g.loop.follow_up = FollowUpMode()
        g.feed()
        await until(lambda: g.loop._room.source is not None and bool(g.wake.history))
        old, source, epoch = g.backend.stream, g.loop._room.source, g.loop._room.input_epoch
        g.loop.pause_admission()
        await g.loop.drain_pause()
        assert old.closed and g.privacy.sources_available((source,))
        g.loop.resume_admission()
        await until(lambda: g.backend.stream is not old and g.backend.stream.opened)
        old.feed()
        assert g.loop._room.resource_status()['queued'] == 0
        g.clock.now += .2
        g.feed()
        await until(lambda: g.loop._room.source is not None)
        assert g.loop._room.input_epoch is not epoch and epoch.retired
        assert g.loop._room.source.conversation_id != source.conversation_id


@pytest.mark.parametrize('ambient', [False, True])
async def test_single_can_load_models_and_start_while_capture_is_already_paused(tmp_path, monkeypatch, ambient):
    async with single_graph(tmp_path, monkeypatch, launch=False, ambient=ambient) as g:
        g.loop.follow_up = FollowUpMode()
        g.loop.pause_admission()
        await g.loop.drain_pause()
        await g.loop.start()
        g.running = asyncio.create_task(g.loop.run())
        await until(lambda: g.loop._controller is not None)
        assert not g.running.done() and g.backend.stream is None
        g.loop.resume_admission()
        await until(lambda: g.backend.stream is not None and g.backend.stream.opened)
        g.clock.now += .2
        g.feed()
        await until(lambda: g.loop._room.source is not None)


@pytest.mark.parametrize('stage', ['constructor', 'start'])
async def test_pausing_during_single_initial_open_does_not_permanently_stop_the_runtime(tmp_path, monkeypatch, stage):
    async with single_graph(tmp_path, monkeypatch, blocked=stage, launch=False) as g:
        g.loop.follow_up = FollowUpMode()
        await g.loop.start()
        g.running = asyncio.create_task(g.loop.run())
        draining = None
        try:
            await until(g.gates[stage].entered.is_set)
            g.loop.pause_admission()
            draining = asyncio.create_task(g.loop.drain_pause())
            await asyncio.sleep(.02)
            assert not draining.done()
        finally:
            g.gates[stage].release.set()
            if draining is not None:
                await draining
        assert not g.running.done() and not g.loop.resource_status()['closed']
        previous = g.backend.stream
        g.loop.resume_admission()
        await until(lambda: g.backend.stream is not previous and g.backend.stream.opened)


@pytest.mark.parametrize('stage', ['primary', 'shadow', 'profiles', 'speaker', 'emotion'])
async def test_processor_pause_retains_each_real_model_and_denies_late_identity_or_transcript(tmp_path, stage):
    async with command_graph(tmp_path, blocked=stage, shadow=stage == 'shadow', veto=stage == 'shadow') as g:
        source = g.source()
        command = g.launch(source)
        draining = None
        try:
            await until(g.gates[stage].entered.is_set)
            g.processor.pause_admission()
            draining = asyncio.create_task(g.processor.drain_pause())
            await asyncio.sleep(.02)
            assert not draining.done()
            with pytest.raises(RuntimeError, match='still_owned'):
                g.processor.resume_admission()
        finally:
            g.gates[stage].release.set()
            await asyncio.gather(command, return_exceptions=True)
            if draining is not None:
                await draining
        assert g.processor.get_current_user() is None and g.processor.get_current_emotion() is None
        assert not g.processor._embeddings_cache
        assert g.privacy.sources_available((source,))
        with producer_context('synthetic-room', (source,)):
            with pytest.raises(SourcesInvalidated):
                await g.processor.process_command(np.ones(16000))
        g.processor.resume_admission()
        assert (await g.launch(g.source())).success


async def test_wake_pause_waits_for_real_encoder_then_reuses_the_same_thread_and_original_ttls(tmp_path, monkeypatch):
    async with clip_graph(tmp_path) as g:
        source = g.source()
        entered, release = gate_encoder(g, monkeypatch)
        thread = g.writer._worker
        assert g.writer.submit('synthetic-room', .9, np.ones(1600), sources=(source,))
        draining = None
        try:
            await until(entered.is_set)
            g.writer.pause_admission()
            draining = asyncio.create_task(g.writer.drain_pause())
            await asyncio.sleep(.02)
            assert not draining.done()
            assert not g.writer.submit('synthetic-room', .9, np.ones(1600), sources=(source,))
            with pytest.raises(RuntimeError, match='still_owned'):
                g.writer.resume_admission()
        finally:
            release.set()
            if draining is not None:
                await draining
        assert not await g.rows() and not g.files()
        assert thread is g.writer._worker and thread.is_alive()
        g.writer.resume_admission()
        assert g.writer.submit('synthetic-room', .9, np.ones(1600), sources=(source,))
        await g.idle()
        rows = await g.rows()
        assert len(rows) == 1 and rows[0]['expires_at'] == source.expires_at


async def test_wake_pause_keeps_existing_history_until_its_original_deadline_and_purge_continues(tmp_path):
    async with clip_graph(tmp_path, background=True) as g:
        source = g.source(ttl=2)
        assert g.writer.submit('synthetic-room', .9, np.ones(1600), sources=(source,))
        await g.idle()
        g.writer.pause_admission()
        await g.writer.drain_pause()
        rows = await g.rows()
        assert len(rows) == 1 and rows[0]['expires_at'] == source.expires_at
        g.clock.now += 2
        await until(lambda: not g.files())
        assert not await g.rows() and g.writer.resource_status()['paused']


async def test_wake_pause_and_close_wait_for_a_submitter_still_copying_input(tmp_path, monkeypatch):
    async with clip_graph(tmp_path) as g:
        entered, release, submitting = copying_submission(g, monkeypatch)
        draining = closing = None
        try:
            await until(entered.is_set)
            g.writer.pause_admission()
            draining = asyncio.create_task(g.writer.drain_pause())
            await asyncio.sleep(.02)
            assert not draining.done() and g.writer.resource_status()['submissions'] == 1
            closing = g.privacy.begin_close()
            await asyncio.sleep(.02)
            assert not closing.done() and g.privacy.resource_status()['writer_lease']
        finally:
            release.set()
            assert not await submitting
            if draining is not None:
                await draining
            if closing is not None:
                await closing
        assert not g.privacy.resource_status()['writer_lease']


async def test_wake_close_alone_retains_the_lease_until_the_actual_input_copy_exits(tmp_path, monkeypatch):
    async with clip_graph(tmp_path) as g:
        entered, release, submitting = copying_submission(g, monkeypatch)
        closing = None
        try:
            await until(entered.is_set)
            closing = g.privacy.begin_close()
            await until(lambda: not g.writer._worker.is_alive())
            await asyncio.sleep(.02)
            assert not closing.done(), 'the submitter still owns raw PCM after the writer thread exits'
            assert g.privacy.resource_status()['writer_lease']
        finally:
            release.set()
            assert not await submitting
            if closing is not None:
                await closing


async def test_multi_pause_includes_the_actual_attached_wake_encoder(tmp_path, monkeypatch):
    portaudio(monkeypatch, [Driver()])
    async with multi_graph(tmp_path, controllers=False) as g:
        store = WakeClipStore(g.privacy, close_timeout_s=.03)
        await store._work.run(store.initialize)
        target = await store._work.run(store.register, tmp_path/'captured')
        writer = WakeClipWriter(target.root, store=store, target=target, close_timeout_s=.03)
        g.loop._wake_clip_writer = writer
        g.feed()
        await g.loop._capture_rooms['synthetic-a'].step()
        entered, release = gate_encoder(SimpleNamespace(writer=writer), monkeypatch)
        assert writer.submit('synthetic-a', .9, np.ones(1600), sources=(g.source(),))
        draining = None
        try:
            await until(entered.is_set)
            g.loop.pause_admission()
            draining = asyncio.create_task(g.loop.drain_pause())
            await asyncio.sleep(.02)
            assert not draining.done() and writer.resource_status()['paused']
            with pytest.raises(RuntimeError):
                g.loop.resume_admission()
        finally:
            release.set()
            if draining is not None:
                await draining
        assert not writer.pause_pending() and writer._worker.is_alive()
        g.loop.resume_admission()
        assert not writer.resource_status()['paused']


async def test_processor_pause_includes_background_shadow_after_primary_command_has_returned(tmp_path):
    async with command_graph(tmp_path, blocked='shadow', shadow=True) as g:
        result = await g.launch(g.source())
        assert result.success
        await until(g.gates['shadow'].entered.is_set)
        g.processor.pause_admission()
        draining = asyncio.create_task(g.processor.drain_pause())
        try:
            await asyncio.sleep(.02)
            assert not draining.done()
        finally:
            g.gates['shadow'].release.set()
            await draining
        assert not g.processor.snapshot_operations() and not g.processor._embeddings_cache


async def test_cancelled_processor_pause_observer_cannot_release_native_work_or_privacy_lease(tmp_path):
    async with command_graph(tmp_path, blocked='primary') as g:
        command = g.launch(g.source())
        draining = None
        try:
            await until(g.gates['primary'].entered.is_set)
            g.processor.pause_admission()
            draining = asyncio.create_task(g.processor.drain_pause())
            await asyncio.sleep(.02)
            draining.cancel()
            await asyncio.gather(draining, return_exceptions=True)
            with pytest.raises(WorkCloseTimeout):
                await g.processor.aclose()
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
        finally:
            g.gates['primary'].release.set()
            await asyncio.gather(command, return_exceptions=True)
            await g.privacy.begin_close()
        assert not g.processor.snapshot_operations() and not g.privacy.resource_status()['writer_lease']


@pytest.mark.parametrize('kind', ['multi', 'single'])
async def test_cancelled_loop_pause_observer_still_retains_close_and_shared_lease(tmp_path, monkeypatch, kind):
    context = multi_graph(tmp_path, blocked='wake') if kind == 'multi' else single_graph(tmp_path, monkeypatch, blocked='wake')
    async with context as g:
        g.loop.follow_up = FollowUpMode()
        gate = g.gates['synthetic-a']['wake'] if kind == 'multi' else g.gates['wake']
        g.feed()
        try:
            await until(gate.entered.is_set)
            g.loop.pause_admission()
            observing = asyncio.create_task(g.loop.drain_pause())
            await asyncio.sleep(.02)
            observing.cancel()
            await asyncio.gather(observing, return_exceptions=True)
            with pytest.raises(WorkCloseTimeout):
                await g.loop.stop()
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
        finally:
            gate.release.set()
            await g.privacy.begin_close()
        assert not g.loop.snapshot_operations() and not g.privacy.resource_status()['writer_lease']


async def test_wake_close_cannot_replace_a_queue_during_the_actual_worker_pop_handoff(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    errors = []
    monkeypatch.setattr(threading, 'excepthook', lambda error: errors.append(error.exc_type.__name__))
    async with clip_graph(tmp_path) as g:
        class HeldQueue(deque):
            def __bool__(self):
                available = len(self) > 0
                if available:
                    entered.set()
                    assert release.wait(4)
                return available
        g.writer._queue = HeldQueue()
        assert g.writer.submit('synthetic-room', .9, np.ones(1600), sources=(g.source(),))
        closing = None
        try:
            await until(entered.is_set)
            closing = g.writer.begin_close()
            await asyncio.sleep(.02)
            assert not closing.done()
        finally:
            release.set()
            if closing is not None:
                await closing
        assert not errors and not await g.rows() and not g.files()
