"""Actual callback queues, serialized wake state and original capture provenance."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import threading
from types import SimpleNamespace
from weakref import ref

import numpy as np
import pytest

from src.conversation.follow_up_mode import FollowUpMode
from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.memory.producer_context import PRODUCER_CONTEXT
from src.pipeline.multi_room_audio_loop import MultiRoomAudioLoop, RoomStream
from tests.mocks.ambient_graph import Clock, Gate

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')
STAGES = ('wake', 'vad', 'early', 'reset')


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


class Wake:
    def __init__(self, gates):
        self.gates = gates
        self.history = []
        self.contexts = []
        self.fire = False

    def load(self):
        self.gates['load'].step()

    def detect(self, audio):
        self.contexts.append(PRODUCER_CONTEXT.get())
        self.history.append(audio)
        self.gates['wake'].step()
        if self.fire:
            self.fire = False
            return 'synthetic-wake', .99
        return None

    def reset(self):
        self.history.clear()
        self.gates['reset'].step()


@dataclass
class Graph:
    privacy: PrivacyServices
    loop: MultiRoomAudioLoop
    clock: Clock
    gates: dict
    wakes: dict
    callbacks: dict
    events: list

    def feed(self, room='synthetic-a', value=.2):
        self.callbacks[room](np.full((1280, 1), value, dtype=np.float32), 1280, None, None)

    def source(self, room='synthetic-a'):
        return self.loop._capture_rooms[room].source


@asynccontextmanager
async def graph(tmp_path, *, blocked=None, rooms=('synthetic-a',), controllers=True, early=False):
    clock = Clock()
    privacy = PrivacyServices(str(tmp_path / 'privacy.db'), clock=clock, close_timeout_s=.03)
    await privacy.acall(privacy.initialize)
    clock.now += 1
    gates = {room: {name: Gate() for name in (*STAGES, 'load')} for room in rooms}
    if blocked:
        gates[rooms[0]][blocked].block()
    wakes = {room: Wake(gates[room]) for room in rooms}
    streams = {}
    events = []

    class STT:
        def transcribe(self, audio, sample_rate):
            gates[rooms[0]]['early'].step()
            return 'Nexa prendé la luz', 1.

    for name in rooms:
        echo = SimpleNamespace(is_safe_to_listen=True,
            should_process_audio=lambda audio: (True, 'ok'), is_human_voice=lambda audio: True)
        streams[name] = RoomStream(name, 0, wakes[name], echo)
    loop = MultiRoomAudioLoop(streams, FollowUpMode(), privacy=privacy,
        microphone_wall_clock=clock, wake_preroll_s=.08,
        early_dispatch_enabled=early, early_dispatch_interval_ms=1,
        early_dispatch_min_audio_s=.01, stt=STT(), close_timeout_s=.03)
    loop._loop = asyncio.get_running_loop()
    loop._running = True
    for name in rooms:
        def vad(audio, name=name):
            gates[name]['vad'].step()
            return .9
        vad.reset = lambda: None
        loop._wake_vad_predictors[name] = vad
    callbacks = {name: loop._make_audio_callback(stream) for name, stream in streams.items()}

    async def command(event):
        events.append(event)
        return {'success': False, 'intent': 'synthetic'}

    loop.on_command(command)
    if controllers:
        for name, stream in streams.items():
            loop._capture_controllers[name] = asyncio.create_task(loop._consume_room(stream))
    try:
        yield Graph(privacy, loop, clock, gates, wakes, callbacks, events)
    finally:
        for room in gates.values():
            for gate in room.values():
                gate.release.set()
        await loop.begin_close()
        await privacy.begin_close()


async def test_callback_returns_before_native_wake_and_does_not_query_authority(tmp_path):
    async with graph(tmp_path, blocked='wake') as g:
        callback_thread = threading.get_ident()
        original = (g.privacy.scopes.buffer_boundary, g.privacy.source_for_capture, g.privacy.sources_available)
        def forbidden(*args, **kwargs):
            raise AssertionError('authority queried on callback')
        g.privacy.scopes.buffer_boundary = g.privacy.source_for_capture = g.privacy.sources_available = forbidden
        try:
            g.feed()
        finally:
            g.privacy.scopes.buffer_boundary, g.privacy.source_for_capture, g.privacy.sources_available = original
        assert not g.gates['synthetic-a']['wake'].entered.is_set()
        assert g.loop._microphone_captures['synthetic-a'].observation.frames_received == 1280
        await until(g.gates['synthetic-a']['wake'].entered.is_set)
        assert g.gates['synthetic-a']['wake'].threads == [g.gates['synthetic-a']['wake'].threads[0]]
        assert g.gates['synthetic-a']['wake'].threads[0] != callback_thread
        source = g.source()
        assert source.captured_at == g.clock() - .08 and source.expires_at == source.captured_at + 48*3600
        assert g.wakes['synthetic-a'].contexts[0].sources == (source,)
        assert g.loop.capture_status()['operations'] > 0
        assert source.conversation_id not in json.dumps(g.loop.capture_status())


@pytest.mark.parametrize('stage', ['wake', 'vad', 'early'])
async def test_revoking_inflight_stage_prevents_late_capture_and_hint_publication(tmp_path, stage):
    async with graph(tmp_path, blocked=stage, early=stage == 'early') as g:
        g.wakes['synthetic-a'].fire = stage != 'wake'
        g.feed()
        await until(g.gates['synthetic-a'][stage].entered.is_set)
        source = g.source()
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        assert g.loop.snapshot_operations()
        g.gates['synthetic-a'][stage].release.set()
        await until(lambda: not g.wakes['synthetic-a'].history and not g.loop.room_streams['synthetic-a'].listening)
        assert not g.events and g.loop.room_streams['synthetic-a'].transcript is None
        if stage == 'wake':
            assert g.gates['synthetic-a']['vad'].calls == 0
        assert not g.loop.room_streams['synthetic-a'].audio_buffer


@pytest.mark.parametrize('stage', STAGES)
async def test_native_capture_and_reset_keep_writer_lease_after_bounded_close(tmp_path, stage):
    async with graph(tmp_path, blocked=stage, early=stage == 'early') as g:
        g.wakes['synthetic-a'].fire = stage in ('vad', 'early')
        g.feed()
        if stage == 'reset':
            await until(lambda: g.source() is not None and bool(g.wakes['synthetic-a'].history))
            source = g.source()
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        await until(g.gates['synthetic-a'][stage].entered.is_set)
        with pytest.raises(WorkCloseTimeout):
            await g.loop.stop()
        with pytest.raises(WorkCloseTimeout):
            await g.privacy.aclose()
        assert g.privacy.resource_status()['writer_lease']
        assert not g.loop.begin_close().done()
        g.gates['synthetic-a'][stage].release.set()
        await g.privacy.begin_close()
        assert not g.privacy.resource_status()['writer_lease']
        assert not g.loop.snapshot_operations()


async def test_idle_wake_and_preroll_retire_at_original_deadline_without_input(tmp_path):
    async with graph(tmp_path) as g:
        g.feed()
        await until(lambda: bool(g.wakes['synthetic-a'].history))
        source = g.source()
        retained = [ref(value) for value in g.wakes['synthetic-a'].history]
        g.clock.now = source.expires_at
        await until(lambda: not g.wakes['synthetic-a'].history)
        assert g.loop.room_streams['synthetic-a'].preroll is None
        assert all(value() is None for value in retained)
        assert g.source() is None


async def test_other_room_continues_when_native_wake_is_blocked(tmp_path):
    async with graph(tmp_path, blocked='wake', rooms=('synthetic-a', 'synthetic-b')) as g:
        g.feed('synthetic-a')
        await until(g.gates['synthetic-a']['wake'].entered.is_set)
        g.feed('synthetic-b')
        await until(g.gates['synthetic-b']['wake'].entered.is_set)
        assert g.source('synthetic-a').conversation_id != g.source('synthetic-b').conversation_id
        source = g.source('synthetic-a')
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        g.feed('synthetic-b')
        await until(lambda: g.gates['synthetic-b']['wake'].calls == 2)
        assert g.privacy.sources_available((g.source('synthetic-b'),))


async def test_original_source_reaches_event_instead_of_being_minted_at_dispatch(tmp_path):
    async with graph(tmp_path, controllers=False) as g:
        g.wakes['synthetic-a'].fire = True
        g.feed()
        room = g.loop._capture_rooms['synthetic-a']
        await room.step()
        source = room.source
        stream = g.loop.room_streams['synthetic-a']
        stream.command_start_time -= .5
        g.clock.now += .08
        g.feed(value=0)
        # Enough silence for the unchanged endpoint contract.
        g.clock.now += .3
        g.callbacks['synthetic-a'](np.zeros((4800, 1), dtype=np.float32), 4800, None, None)
        event = await room.step()
        assert event is not None and event.capture_complete
        assert (event.capture_id, event.conversation_id, event.captured_at, event.source_expires_at) == (
            source.capture_id, source.conversation_id, source.captured_at, source.expires_at)
        assert not g.wakes['synthetic-a'].history and not stream.audio_buffer
        g.loop._start_dispatch(event)
        await until(lambda: len(g.events) == 1)
        assert g.events[0] is event


async def test_cancelled_capture_observer_does_not_retain_input_after_real_close(tmp_path):
    async with graph(tmp_path, blocked='wake', controllers=False) as g:
        g.feed()
        room = g.loop._capture_rooms['synthetic-a']
        audio_ref = ref(room._queue[0].audio)
        observer = asyncio.create_task(room.step())
        await until(g.gates['synthetic-a']['wake'].entered.is_set)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await observer
        assert audio_ref() is not None and room.snapshot_operations()
        closing = g.loop.begin_close()
        assert not closing.done()
        g.gates['synthetic-a']['wake'].release.set()
        await closing
        assert caught.value is not None and audio_ref() is None
        assert not g.wakes['synthetic-a'].history


async def test_retired_callback_queue_cannot_revive_input_after_close(tmp_path):
    async with graph(tmp_path, controllers=False) as g:
        room = g.loop._capture_rooms['synthetic-a']
        entered, release = threading.Event(), threading.Event()

        class Delayed:
            def copy(self):
                entered.set()
                assert release.wait(3)
                return np.ones((1280, 1), dtype=np.float32)

        pushing = asyncio.create_task(asyncio.to_thread(room.push, g.clock(), Delayed(), 1280))
        try:
            await until(entered.is_set)
            await g.loop.begin_close()
        finally:
            release.set()
            await pushing
        assert room.resource_status()['queued'] == 0 and not g.wakes['synthetic-a'].history


async def test_actual_wake_warmup_is_owned_off_loop_and_closes_after_late_load(tmp_path):
    async with graph(tmp_path, blocked='load', controllers=False) as g:
        startup = asyncio.create_task(g.loop.start())
        await until(g.gates['synthetic-a']['load'].entered.is_set)
        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup
        with pytest.raises(WorkCloseTimeout):
            await g.privacy.aclose()
        assert g.privacy.resource_status()['writer_lease']
        assert g.gates['synthetic-a']['load'].threads[0] != threading.get_ident()
        g.gates['synthetic-a']['load'].release.set()
        await g.privacy.begin_close()
        assert not g.privacy.resource_status()['writer_lease']


async def test_pre_epoch_audio_is_not_renamed_inside_an_existing_capture(tmp_path):
    async with graph(tmp_path, controllers=False) as g:
        room = g.loop._capture_rooms['synthetic-a']
        g.feed()
        await room.step()
        source = room.source
        room.push(source.captured_at - .01, np.ones((1280, 1), dtype=np.float32), 1280)
        await room.step()
        assert g.gates['synthetic-a']['wake'].calls == 1
        assert room.source == source


async def test_callback_admission_stays_bounded_while_native_wake_is_blocked(tmp_path):
    async with graph(tmp_path, blocked='wake') as g:
        g.feed()
        await until(g.gates['synthetic-a']['wake'].entered.is_set)
        for _ in range(100):
            g.feed()
        room = g.loop._capture_rooms['synthetic-a']
        assert room.resource_status()['queued'] == 32
        assert g.loop._microphone_captures['synthetic-a'].observation.frames_received == 101*1280
        assert g.gates['synthetic-a']['wake'].calls == 1


async def test_callback_failure_has_no_private_exception_trace_and_rejects_late_input(tmp_path, caplog):
    async with graph(tmp_path, controllers=False) as g:
        class Failed:
            def copy(self):
                raise RuntimeError('Synthetic private input error')
        room = g.loop._capture_rooms['synthetic-a']
        room.push(g.clock(), Failed(), 1280)
        assert room.resource_status()['input_drops'] == 1
        assert 'Synthetic private input error' not in caplog.text
        assert room.resource_status()['queued'] == 0


async def test_pending_failed_revocation_discards_transients_and_allows_fresh_input(tmp_path):
    async with graph(tmp_path) as g:
        g.feed()
        await until(lambda: bool(g.wakes['synthetic-a'].history))
        source = g.source()
        entered, release = threading.Event(), threading.Event()
        original = g.privacy.ledger._db

        class FailedCommit:
            def __getattr__(self, key):
                return getattr(original, key)
            def commit(self):
                entered.set()
                assert release.wait(3)
                raise OSError('synthetic marker rollback')

        g.privacy.ledger._db = FailedCommit()
        marking = asyncio.create_task(g.privacy.acall(
            g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR))
        try:
            await until(entered.is_set)
            await until(lambda: not g.wakes['synthetic-a'].history)
            assert not g.events
        finally:
            release.set()
            with pytest.raises(OSError, match='marker rollback'):
                await marking
            g.privacy.ledger._db = original
        assert g.privacy.sources_available((source,))
        calls = g.gates['synthetic-a']['wake'].calls
        g.feed()  # Still before the observed fence; cannot rename old audio.
        await asyncio.sleep(.08)
        assert g.gates['synthetic-a']['wake'].calls == calls
        g.clock.now += .1
        g.feed()
        await until(lambda: g.gates['synthetic-a']['wake'].calls > calls)


@pytest.mark.parametrize('engine', ['utterance', 'streaming'])
async def test_real_whisper_wake_reset_discards_audio_and_pending_transcript(tmp_path, engine):
    from src.wakeword.whisper_wake import WhisperWakeDetector
    from src.wakeword.streaming_whisper_wake import StreamingWhisperWakeDetector

    async with graph(tmp_path, controllers=False) as g:
        detector_type = WhisperWakeDetector if engine == 'utterance' else StreamingWhisperWakeDetector
        # RMS-only mode reports .5; this fixture exercises speech buffers
        # without loading Silero or changing production calibration.
        detector = detector_type(g.loop._stt, ['nexa'], use_silero_vad=False, vad_threshold=.4)
        detector._loaded = True
        stream = g.loop.room_streams['synthetic-a']
        stream.wake_detector = detector
        room = g.loop._capture_rooms['synthetic-a']
        g.feed()
        await room.step()
        retained = ref(detector._utterance[0] if engine == 'utterance' else detector._buffer[0])
        source = room.source
        if engine == 'utterance':
            detector._pending_text = 'Synthetic private pending text'
        detector._pending_command_audio = np.ones(16000, dtype=np.float32)
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        await room.step()
        assert retained() is None and detector._pending_command_audio is None
        if engine == 'utterance':
            assert detector._pending_text is None and not detector._utterance and not detector._pre_roll
        else:
            assert not detector._buffer and detector._buffer_samples == 0


async def test_actual_native_wake_and_primary_command_share_cancellable_model_admission(tmp_path):
    from uuid import uuid4
    from src.memory.producer_context import producer_context
    from src.memory.prompt_context import SourcesInvalidated
    from src.pipeline.command_processor import CommandProcessor

    async with graph(tmp_path, blocked='wake') as g:
        g.wakes['synthetic-a'].whisper = g.loop._stt
        entered = threading.Event()
        def forbidden(*args):
            entered.set()
            raise AssertionError('queued model entered')
        g.loop._stt.transcribe_with_confidence = forbidden
        cp = CommandProcessor(g.loop._stt, privacy=g.privacy, model_locks=g.loop.native_model_locks)
        g.feed()
        await until(g.gates['synthetic-a']['wake'].entered.is_set)
        source = g.privacy.source_for_capture(str(uuid4()), g.clock(), 'synthetic-b')
        with producer_context('synthetic-b', (source,)):
            observer = asyncio.create_task(cp.process_command(np.ones(16000, dtype=np.float32)))
        try:
            await until(lambda: cp.resource_status()['native']['operations'] == 1)
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            with pytest.raises(SourcesInvalidated):
                await observer
            await until(lambda: cp.resource_status()['native']['operations'] == 0)
            assert g.privacy.sources_available((g.source(),))
            assert not g.gates['synthetic-a']['wake'].release.is_set()
            assert not entered.is_set()
        finally:
            g.gates['synthetic-a']['wake'].release.set()
            await asyncio.gather(observer, return_exceptions=True)
            await cp.aclose()


async def test_porcupine_native_handle_delete_remains_owned_until_real_exit(tmp_path):
    from src.wakeword.porcupine_wake import PorcupineWakeDetector

    async with graph(tmp_path, controllers=False) as g:
        gate = Gate()
        gate.block()
        calls = []
        detector = PorcupineWakeDetector('synthetic-access', 'synthetic.ppn')

        class Handle:
            def process(self, frame):
                calls.append(threading.get_ident())
                return -1
            def delete(self):
                gate.step()

        detector._handle, detector._loaded = Handle(), True
        g.loop.room_streams['synthetic-a'].wake_detector = detector
        room = g.loop._capture_rooms['synthetic-a']
        g.feed()
        await room.step()
        assert calls and detector._buf.size
        source = room.source
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        pruning = asyncio.create_task(room.step())
        try:
            await until(gate.entered.is_set)
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease'] and detector._handle is not None
        finally:
            gate.release.set()
            await asyncio.gather(pruning, return_exceptions=True)
            await g.privacy.begin_close()
        assert detector._handle is None and not detector._loaded and detector._buf.size == 0
        assert gate.threads[0] != threading.get_ident()


async def test_one_failed_native_warmup_does_not_disable_other_room(tmp_path, caplog):
    async with graph(tmp_path, controllers=False, rooms=('synthetic-a', 'synthetic-b')) as g:
        def failed():
            raise RuntimeError('Synthetic private loading failure')
        g.wakes['synthetic-a'].load = failed
        await g.loop.start()
        assert g.loop._capture_rooms['synthetic-a'].resource_status()['closed']
        assert not g.loop._capture_rooms['synthetic-b'].resource_status()['closed']
        g.feed('synthetic-a')
        g.feed('synthetic-b')
        await g.loop._capture_rooms['synthetic-b'].step()
        assert g.gates['synthetic-b']['wake'].calls == 1
        assert g.loop._capture_rooms['synthetic-a'].resource_status()['queued'] == 0
        assert 'Synthetic private loading failure' not in caplog.text


async def test_retiring_microphone_cannot_publish_an_inflight_old_callback(tmp_path):
    async with graph(tmp_path, controllers=False) as g:
        entered, release = threading.Event(), threading.Event()
        room = g.loop._capture_rooms['synthetic-a']
        class Delayed:
            def any(self):
                return True
            def copy(self):
                entered.set()
                assert release.wait(3)
                return np.ones((1280, 1), dtype=np.float32)
        pushing = asyncio.create_task(asyncio.to_thread(
            g.callbacks['synthetic-a'], Delayed(), 1280, None, None))
        try:
            await until(entered.is_set)
            g.loop._retire_microphone_capture('synthetic-a')
        finally:
            release.set()
            await pushing
        assert room.resource_status()['queued'] == 0


async def test_retired_callback_cannot_rename_old_input_with_a_later_clock(tmp_path):
    async with graph(tmp_path, controllers=False) as g:
        entered, release = threading.Event(), threading.Event()
        previous = g.loop._microphone_monotonic
        def delayed_clock():
            entered.set()
            assert release.wait(3)
            return previous()
        g.loop._microphone_monotonic = delayed_clock
        pushing = asyncio.create_task(asyncio.to_thread(g.feed))
        try:
            await until(entered.is_set)
            g.loop._retire_microphone_capture('synthetic-a')
            g.clock.now += 1
        finally:
            release.set()
            await pushing
            g.loop._microphone_monotonic = previous
        assert g.loop._capture_rooms['synthetic-a'].resource_status()['queued'] == 0


async def test_retiring_driver_invalidates_actual_native_wake_before_fresh_stream(tmp_path):
    async with graph(tmp_path, blocked='wake') as g:
        g.feed()
        await until(g.gates['synthetic-a']['wake'].entered.is_set)
        previous = g.source()
        g.loop._retire_microphone_capture('synthetic-a')
        g.gates['synthetic-a']['wake'].release.set()
        await until(lambda: g.source() is None and not g.wakes['synthetic-a'].history)
        assert not g.events
        assert g.privacy.sources_available((previous,))  # Driver retirement is not a durable deletion.
        g.clock.now += .16
        stream = g.loop.room_streams['synthetic-a']
        g.callbacks['synthetic-a'] = g.loop._make_audio_callback(stream)
        g.feed()
        await until(lambda: g.gates['synthetic-a']['wake'].calls == 2)
        assert g.source().capture_id != previous.capture_id
        assert g.source().captured_at > previous.captured_at


async def test_delayed_boundary_publisher_cannot_loosen_retired_stream_fence(tmp_path):
    async with graph(tmp_path, controllers=False) as g:
        room = g.loop._capture_rooms['synthetic-a']
        entered, release = threading.Event(), threading.Event()
        original = g.privacy.scopes.buffer_boundary
        def delayed(*args, **kwargs):
            candidate = original(*args, **kwargs)
            entered.set()
            assert release.wait(3)
            return candidate
        g.privacy.scopes.buffer_boundary = delayed
        publishing = asyncio.create_task(asyncio.to_thread(room.publish_boundary))
        try:
            await until(entered.is_set)
            room.retire_stream(g.clock())
        finally:
            release.set()
            await publishing
            g.privacy.scopes.buffer_boundary = original
        room.push(g.clock() - .01, np.ones((1280, 1), dtype=np.float32), 1280)
        assert room.resource_status()['queued'] == 0
        room.push(g.clock() + .01, np.ones((1280, 1), dtype=np.float32), 1280)
        assert room.resource_status()['queued'] == 1
