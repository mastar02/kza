"""Actual single-input loop, manager, driver and optional event classifier ownership."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import threading
from weakref import ref

import numpy as np
import pytest

from src.ambient.audio_event_detector import AudioEventDetector
from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.memory.producer_context import PRODUCER_CONTEXT
from src.pipeline.audio_loop import AudioLoop
from src.pipeline.audio_manager import AudioManager
from tests.mocks.ambient_graph import Clock, Gate

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


class Wake:
    def __init__(self, gates):
        self.gates, self.history, self.contexts = gates, [], []
        self.fire = False
    def load(self):
        self.gates['load'].step()
    def get_active_models(self):
        return ['synthetic-wake']
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


class Backend:
    def __init__(self, gates):
        self.gates, self.stream = gates, None
        self.close_failures = 0
    def InputStream(self, **options):
        backend = self
        class Stream:
            opened = closed = False
            def start(self):
                backend.gates['start'].step()
                self.opened = True
            def stop(self):
                backend.gates['stop'].step()
            def close(self):
                backend.gates['close'].step()
                if backend.close_failures:
                    backend.close_failures -= 1
                    raise RuntimeError('Synthetic private close failure')
                self.closed = True
            def feed(self, value=.2, frames=1280):
                options['callback'](np.full((frames, 1), value, dtype=np.float32), frames, None, None)
            def emit(self, audio):
                options['callback'](audio[:, None], len(audio), None, None)
        self.stream = Stream()
        self.gates['constructor'].step()
        return self.stream


@dataclass
class Graph:
    privacy: PrivacyServices
    loop: AudioLoop
    clock: Clock
    gates: dict
    backend: Backend
    wake: Wake
    detector: AudioEventDetector | None
    events: list
    commands: list
    model_contexts: list
    vectors: list
    audio_refs: list
    running: asyncio.Task | None = None

    def feed(self, value=.2, frames=1280):
        self.backend.stream.feed(value, frames)

    async def launch(self):
        await self.loop.start()
        self.running = asyncio.create_task(self.loop.run())
        await until(lambda: self.backend.stream is not None and self.backend.stream.opened)


@asynccontextmanager
async def graph(tmp_path, monkeypatch, *, blocked=None, ambient=False, launch=True):
    import src.pipeline.audio_loop as module
    gates = {name: Gate() for name in ('load','wake','reset','endpoint','constructor','start','stop','close','model','callback','event_load')}
    if blocked:
        gates[blocked].block()
    clock = Clock()
    privacy = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock, close_timeout_s=.03)
    await privacy.acall(privacy.initialize)
    clock.now += 1
    wake = Wake(gates)
    manager = AudioManager(clock=clock, wake_factory=lambda **kwargs: wake)
    endpoint = manager.capture_command_with_vad
    def gated_endpoint(*args, **kwargs):
        gates['endpoint'].step()
        return endpoint(*args, **kwargs)
    manager.capture_command_with_vad = gated_endpoint
    class Echo:
        is_safe_to_listen = True
        def should_process_audio(self, audio):
            return True, 'ok'
        def is_human_voice(self, audio):
            return True
    class FollowUp:
        is_active = False
        def start_conversation(self):
            pass
    events, commands, contexts, vectors, audio_refs = [], [], [], [], []
    detector = AudioEventDetector(close_timeout_s=.03) if ambient else None
    if detector is not None:
        class Scores:
            def numpy(self):
                return np.array([[.99]], dtype=np.float32)
        class Model:
            def __call__(self, audio):
                audio_refs.append(ref(audio))
                contexts.append(PRODUCER_CONTEXT.get())
                gates['model'].step()
                embedding, spectrum = np.ones(192), np.ones(100)
                vectors.extend((ref(embedding), ref(spectrum)))
                return Scores(), embedding, spectrum
        def load_model():
            gates['event_load'].step()
            detector._model = Model()
            detector._class_names = ['Smoke detector']
        detector._load_model_native = load_model
        def on_event(event):
            gates['callback'].step()
            events.append(event)
        detector.on_event(on_event)
    loop = AudioLoop(manager, Echo(), FollowUp(), ambient_detector=detector,
                     privacy=privacy, clock=clock, close_timeout_s=.03)
    async def command(event):
        commands.append(event)
        return {'success': False}
    loop.on_command(command)
    backend = Backend(gates)
    g = Graph(privacy, loop, clock, gates, backend, wake, detector, events, commands, contexts, vectors, audio_refs)
    with monkeypatch.context() as patch:
        patch.setattr(module, 'sd', backend)
        try:
            if launch:
                await g.launch()
            yield g
        finally:
            for gate in gates.values():
                gate.release.set()
            await loop.begin_close()
            if g.running is not None:
                await asyncio.gather(g.running, return_exceptions=True)
            await privacy.begin_close()


async def test_actual_driver_callback_only_queues_and_wake_runs_off_loop(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, blocked='wake') as g:
        main_thread = threading.get_ident()
        g.feed()
        assert not g.gates['wake'].entered.is_set()
        await until(g.gates['wake'].entered.is_set)
        source = g.loop._room.source
        assert source.captured_at == g.clock()-.08
        assert source.expires_at == source.captured_at+48*3600
        assert g.wake.contexts[0].sources == (source,)
        assert g.gates['wake'].threads[0] != main_thread
        assert source.conversation_id not in json.dumps(g.loop.resource_status())


@pytest.mark.parametrize('stage', ['wake','endpoint'])
async def test_revoking_actual_capture_prevents_late_event_and_post_callback(tmp_path, monkeypatch, stage):
    async with graph(tmp_path, monkeypatch, blocked=stage) as g:
        g.wake.fire = stage == 'endpoint'
        g.feed()
        await until(g.gates[stage].entered.is_set)
        source = g.loop._room.source
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        g.gates[stage].release.set()
        await until(lambda: not g.wake.history and not g.loop._state.listening)
        assert not g.commands


@pytest.mark.parametrize('stage', ['wake','endpoint','reset','stop','close'])
async def test_single_input_native_exit_retains_shared_lease_after_close_timeout(tmp_path, monkeypatch, stage):
    async with graph(tmp_path, monkeypatch, blocked=stage) as g:
        g.wake.fire = stage == 'endpoint'
        if stage not in ('stop','close'):
            g.feed()
            if stage == 'reset':
                await until(lambda: bool(g.wake.history))
                source = g.loop._room.source
                await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            await until(g.gates[stage].entered.is_set)
        with pytest.raises(WorkCloseTimeout):
            await g.loop.stop()
        with pytest.raises(WorkCloseTimeout):
            await g.privacy.aclose()
        assert g.privacy.resource_status()['writer_lease']
        assert not g.loop.begin_close().done()
        g.gates[stage].release.set()
        await g.privacy.begin_close()
        assert g.backend.stream.closed and not g.privacy.resource_status()['writer_lease']


@pytest.mark.parametrize('stage', ['constructor','start'])
async def test_cancelled_native_open_cannot_leak_a_late_driver_handle(tmp_path, monkeypatch, stage):
    async with graph(tmp_path, monkeypatch, blocked=stage, launch=False) as g:
        await g.loop.start()
        g.running = asyncio.create_task(g.loop.run())
        await until(g.gates[stage].entered.is_set)
        g.running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await g.running
        with pytest.raises(WorkCloseTimeout):
            await g.privacy.aclose()
        assert g.privacy.resource_status()['writer_lease'] and not g.backend.stream.closed
        g.gates[stage].release.set()
        await g.privacy.begin_close()
        assert g.backend.stream.closed
        if stage == 'constructor':
            assert g.gates['start'].calls == 0


async def test_wake_warmup_cancel_keeps_manager_reset_after_actual_load(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, blocked='load', launch=False) as g:
        loading = asyncio.create_task(g.loop.start())
        await until(g.gates['load'].entered.is_set)
        loading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loading
        with pytest.raises(WorkCloseTimeout):
            await g.privacy.aclose()
        assert g.gates['reset'].calls == 0 and g.privacy.resource_status()['writer_lease']
        g.gates['load'].release.set()
        await g.privacy.begin_close()
        assert g.gates['reset'].calls == 1
        assert g.loop.audio_manager._wake_model is None
        assert g.gates['load'].threads[0] != threading.get_ident()


async def test_close_failure_retries_the_same_owned_handle_without_reopening(tmp_path, monkeypatch, caplog):
    async with graph(tmp_path, monkeypatch) as g:
        g.backend.close_failures = 2
        with pytest.raises(WorkCloseTimeout):
            await g.loop.stop()
        with pytest.raises(WorkCloseTimeout):
            await g.privacy.aclose()
        assert g.privacy.resource_status()['writer_lease']
        await g.privacy.begin_close()
        assert g.backend.stream.closed and g.gates['constructor'].calls == 1
        assert g.gates['close'].calls == 3
        assert 'Synthetic private close failure' not in caplog.text


async def test_completed_capture_preserves_original_source_and_endpoint_finality(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch) as g:
        g.wake.fire = True
        g.feed()
        await until(lambda: g.loop._state.listening)
        source = g.loop._room.source
        g.clock.now += .4
        g.feed(0, 4800)
        await until(lambda: len(g.commands) == 1)
        event = g.commands[0]
        assert event.capture_complete is True and len(event.audio) == 4800
        assert (event.capture_id,event.conversation_id,event.captured_at,event.source_expires_at) == (
            source.capture_id,source.conversation_id,source.captured_at,source.expires_at)
        assert not g.wake.history and not g.loop._state.audio


async def test_partial_ambient_window_expires_while_capture_epoch_is_idle(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, ambient=True) as g:
        g.feed()
        await until(lambda: g.loop._ambient_samples == 1280)
        retained = ref(g.loop._ambient_frames[0][1])
        source = g.loop._room.source
        g.clock.now = source.expires_at
        await until(lambda: g.loop._ambient_samples == 0)
        assert retained() is None and not g.events


@pytest.mark.parametrize('stage', ['model','callback'])
async def test_actual_event_classifier_and_sync_notification_are_owned_off_loop(tmp_path, monkeypatch, stage):
    async with graph(tmp_path, monkeypatch, ambient=True, blocked=stage) as g:
        for _ in range(13):
            g.feed()
        await until(g.gates[stage].entered.is_set)
        source = g.model_contexts[0].sources[0]
        assert g.gates[stage].threads[0] != threading.get_ident()
        assert source == g.loop._room.source
        with pytest.raises(WorkCloseTimeout):
            await g.loop.stop()
        with pytest.raises(WorkCloseTimeout):
            await g.privacy.aclose()
        assert g.privacy.resource_status()['writer_lease']
        g.gates[stage].release.set()
        await g.privacy.begin_close()
        assert not g.detector.snapshot_operations() and g.detector._model is None
        assert all(value() is None for value in g.vectors)


async def test_revoked_classifier_cannot_notify_after_native_result_arrives(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, ambient=True, blocked='model') as g:
        for _ in range(13):
            g.feed()
        await until(g.gates['model'].entered.is_set)
        source = g.model_contexts[0].sources[0]
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        await until(lambda: not g.loop._analyses)
        assert g.detector.snapshot_operations()
        g.gates['model'].release.set()
        await until(lambda: not g.detector.snapshot_operations())
        assert not g.events and g.gates['callback'].calls == 0
        assert all(value() is None for value in g.vectors)


async def test_queued_frames_keep_acquisition_time_for_endpointing(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch) as g:
        gate = Gate()
        gate.block()
        holding = asyncio.create_task(g.loop._room.call(gate.step))
        g.wake.fire = True
        try:
            await until(gate.entered.is_set)
            g.clock.now += .08
            g.feed()
            g.clock.now += .08
            g.feed()
            g.clock.now += .3
            g.feed(0, 4800)
        finally:
            gate.release.set()
            await holding
        await until(lambda: len(g.commands) == 1)
        assert g.commands[0].capture_complete is True and len(g.commands[0].audio) == 6080


async def test_long_command_callback_does_not_stop_single_device_capture(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch) as g:
        entered, release = asyncio.Event(), asyncio.Event()
        async def slow(event):
            entered.set()
            await release.wait()
            return {}
        g.loop.on_command(slow)
        g.wake.fire = True
        g.feed()
        await until(lambda: g.loop._state.listening)
        g.clock.now += .4
        g.feed(0, 4800)
        try:
            await until(entered.is_set)
            previous = g.gates['wake'].calls
            g.clock.now += .08
            g.feed()
            await until(lambda: g.gates['wake'].calls > previous)
            assert len(g.loop._dispatches) == 1
        finally:
            release.set()


async def test_revocation_during_command_prevents_post_processing(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch) as g:
        entered, release = asyncio.Event(), asyncio.Event()
        post = []
        async def late(event):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return {'text': 'Synthetic private late result'}
        async def after(result, audio):
            post.append(result)
        g.loop.on_command(late)
        g.loop.on_post_command(after)
        g.wake.fire = True
        g.feed()
        await until(lambda: g.loop._state.listening)
        source = g.loop._room.source
        g.clock.now += .4
        g.feed(0, 4800)
        try:
            await until(entered.is_set)
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            await asyncio.sleep(.03)
            assert g.loop._dispatches
        finally:
            release.set()
        await until(lambda: not g.loop._dispatches)
        assert not post


async def test_noncooperative_analysis_callback_keeps_lease_until_real_cleanup(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, ambient=True) as g:
        entered, release = asyncio.Event(), asyncio.Event()
        async def late(event):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
        g.detector.on_event(late)
        for _ in range(13):
            g.feed()
        try:
            await until(entered.is_set)
            with pytest.raises(WorkCloseTimeout):
                await g.loop.stop()
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease'] and g.loop.snapshot_operations()
        finally:
            release.set()
        await g.privacy.begin_close()
        assert not g.detector.snapshot_operations()


async def test_mixed_ambient_window_uses_oldest_dependency_without_renewal(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, ambient=True, blocked='model') as g:
        g.wake.fire = True
        g.feed()
        await until(lambda: g.loop._state.listening)
        first = g.loop._room.source
        g.clock.now += .4
        g.feed(0, 4800)
        await until(lambda: len(g.commands) == 1)
        for _ in range(8):
            g.clock.now += .08
            g.feed()
        await until(g.gates['model'].entered.is_set)
        sources = g.model_contexts[0].sources
        assert len(sources) == 2 and first in sources
        assert min(s.expires_at for s in sources) == first.expires_at
        g.clock.now = first.expires_at
        await until(lambda: not g.loop._analyses)
        g.gates['model'].release.set()
        await until(lambda: not g.detector.snapshot_operations())
        assert not g.events


async def test_ambient_frames_from_a_long_gap_are_never_spliced(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, ambient=True) as g:
        g.feed()
        await until(lambda: g.loop._ambient_samples == 1280)
        first = g.loop._ambient_frames[0][0][0]
        previous = ref(g.loop._ambient_frames[0][1])
        g.clock.now += 10
        g.feed()
        await until(lambda: g.loop._ambient_frames[0][0][0].capture_id != first.capture_id)
        assert g.loop._ambient_samples == 1280 and previous() is None
        assert not g.events


async def test_cancelled_capture_traceback_does_not_keep_retired_audio(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, blocked='wake', launch=False) as g:
        await g.loop.start()
        callback = g.loop._make_audio_callback()
        callback(np.ones((1280, 1), dtype=np.float32), 1280, None, None)
        audio = ref(g.loop._room._queue[0].audio)
        observing = asyncio.create_task(g.loop._room.step())
        await until(g.gates['wake'].entered.is_set)
        observing.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await observing
        assert audio() is not None
        closing = g.loop.begin_close()
        g.gates['wake'].release.set()
        await closing
        assert caught.value is not None and audio() is None and not g.wake.history


async def test_old_callback_with_delayed_clock_cannot_enter_closed_input(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch) as g:
        entered, release = threading.Event(), threading.Event()
        clock = g.loop._clock
        def late():
            entered.set()
            assert release.wait(3)
            return clock()
        g.loop._clock = late
        pushing = asyncio.create_task(asyncio.to_thread(g.feed))
        try:
            await until(entered.is_set)
            # Retire immediately, but do not report close while the admitted
            # callback still owns its input behind the blocked clock.
            g.loop._clock = clock
            with pytest.raises(WorkCloseTimeout):
                await g.loop.stop()
            g.clock.now += 1
        finally:
            release.set()
            await pushing
            await g.loop.begin_close()
        assert g.loop._room.resource_status()['queued'] == 0
        assert g.gates['wake'].calls == 0


async def test_event_model_warmup_retains_lease_and_drops_late_loaded_model(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, ambient=True, blocked='event_load', launch=False) as g:
        loading = asyncio.create_task(g.loop.start())
        await until(g.gates['event_load'].entered.is_set)
        loading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await loading
        with pytest.raises(WorkCloseTimeout):
            await g.privacy.aclose()
        assert g.privacy.resource_status()['writer_lease'] and g.detector.snapshot_operations()
        g.gates['event_load'].release.set()
        await g.privacy.begin_close()
        assert g.detector._model is None and not g.detector.snapshot_operations()


async def test_privacy_bound_event_detector_rejects_unsourced_audio(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, ambient=True) as g:
        result = await g.detector.analyze_chunk(np.ones(16000, dtype=np.float32))
        assert result == [] and g.gates['model'].calls == 0 and not g.events
        assert g.detector._privacy is g.privacy


async def test_real_feature_fallback_is_off_loop_and_notifies_with_original_sources(tmp_path, monkeypatch):
    from src.ambient.audio_event_detector import AudioEventType
    async with graph(tmp_path, monkeypatch, ambient=True) as g:
        g.detector._model = None
        # Exercise the existing feature result (.5 confidence) in a synthetic
        # fixture; production event thresholds and calibration stay unchanged.
        g.detector.configure_event(AudioEventType.GLASS_BREAKING, min_confidence=.4)
        original = g.detector._classify_features
        observed = []
        def features(*args):
            observed.append((threading.get_ident(), PRODUCER_CONTEXT.get().sources))
            return original(*args)
        g.detector._classify_features = features
        for index in range(13):
            audio = np.full(1280, .05, dtype=np.float32)
            if index == 0:
                audio[0] = .99
            g.backend.stream.emit(audio)
        await until(lambda: bool(g.events))
        assert g.events[0].event_type is AudioEventType.GLASS_BREAKING
        assert observed[0][0] != threading.get_ident()
        assert observed[0][1] == g.events[0].sources
        assert g.events[0].sources == (g.loop._room.source,)
        other = AudioEventDetector()
        assert other._event_configs[AudioEventType.GLASS_BREAKING].min_confidence == .7
        await other.stop()


async def test_event_callback_may_request_its_own_retained_detector_close(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, ambient=True) as g:
        ended = asyncio.Event()
        async def on_event(event):
            await g.detector.stop()
            ended.set()
        g.detector.on_event(on_event)
        for _ in range(13):
            g.feed()
        await until(ended.is_set)
        await g.detector.begin_close()
        assert not g.detector.snapshot_operations()


async def test_event_callback_may_signal_loop_close_without_waiting_on_itself(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, ambient=True) as g:
        ended = asyncio.Event()
        async def on_event(event):
            await g.loop.stop()
            ended.set()
        g.detector.on_event(on_event)
        for _ in range(13):
            g.feed()
        await until(ended.is_set)
        await g.loop.stop()
        assert g.backend.stream.closed and not g.loop.snapshot_operations()


async def test_notification_keeps_ownership_without_retaining_unneeded_waveform(tmp_path, monkeypatch):
    async with graph(tmp_path, monkeypatch, ambient=True, blocked='callback') as g:
        security = []
        async def on_security(event):
            security.append(event)
        g.detector.on_security_event(on_security)
        for _ in range(13):
            g.feed()
        await until(g.gates['callback'].entered.is_set)
        assert g.audio_refs and all(value() is None for value in g.audio_refs)
        assert g.detector.snapshot_operations() and not g.gates['callback'].release.is_set()
        source = g.model_contexts[0].sources[0]
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        g.gates['callback'].release.set()
        await until(lambda: not g.detector.snapshot_operations())
        assert not security


async def test_retirement_hook_failure_cannot_skip_real_driver_release(tmp_path, monkeypatch, caplog):
    async with graph(tmp_path, monkeypatch) as g:
        def failed():
            raise RuntimeError('Synthetic private retirement failure')
        g.loop._input._retire = failed
        await g.loop.stop()
        assert g.backend.stream.closed and not g.loop.snapshot_operations()
        assert 'Synthetic private retirement failure' not in caplog.text
