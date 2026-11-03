"""Real selected wake/ambient Silero factories; torch.hub is synthetic only."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import sys
import threading
from types import ModuleType, SimpleNamespace
from weakref import ref

import numpy as np
import pytest

from src.ambient.room_producer import RoomProducer
from src.ambient.segmenter import UtteranceSegmenter, make_silero_predictor
from src.main import build_application
from src.pipeline.multi_room_audio_loop import compute_wake_vad
from tests.integration.test_application_graph import leaves as application_leaves
from tests.mocks.application_graph import OfflineWake, offline_config, offline_home


leaves = application_leaves
pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('dashboard_auth')]
CHUNK = np.full((1280, 3), .125, dtype=np.float32)


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(.002)


def install_torch(monkeypatch):
    sdk = SimpleNamespace(models=[], gates=[], hold_load=set(), hold_eval=set(), windows=[])
    local = threading.local()

    class Tensor:
        def __init__(self, value):
            self.value = value

    class NoGrad:
        def __enter__(self):
            local.no_grad = True

        def __exit__(self, *_args):
            local.no_grad = False

    class Model:
        def __init__(self, gate):
            self.gate = gate

        def eval(self):
            self.gate.eval_entered.set()
            assert self.gate.eval_release.wait(10), 'synthetic Silero eval was not released'
            return self

        def __call__(self, tensor, rate):
            assert rate == 16000 and tensor.value.shape == (512,)
            assert tensor.value.dtype == np.float32 and local.no_grad
            sdk.windows.append((self.gate.index, threading.get_ident(), len(tensor.value)))
            self.gate.infer_entered.set()
            assert self.gate.infer_release.wait(10), 'synthetic Silero inference was not released'
            return SimpleNamespace(item=lambda: .8)

        def reset_states(self):
            self.gate.reset_entered.set()
            assert self.gate.reset_release.wait(10), 'synthetic Silero reset was not released'

    def load(**kwargs):
        assert kwargs == {'repo_or_dir': 'snakers4/silero-vad', 'model': 'silero_vad', 'trust_repo': True}
        index = len(sdk.gates)
        gate = SimpleNamespace(index=index,
            load_entered=threading.Event(), load_release=threading.Event(),
            eval_entered=threading.Event(), eval_release=threading.Event(),
            infer_entered=threading.Event(), infer_release=threading.Event(),
            reset_entered=threading.Event(), reset_release=threading.Event())
        for event in (gate.load_release, gate.eval_release, gate.infer_release, gate.reset_release):
            event.set()
        if index in sdk.hold_load:
            gate.load_release.clear()
        if index in sdk.hold_eval:
            gate.eval_release.clear()
        model = Model(gate)
        sdk.gates.append(gate)
        sdk.models.append(ref(model))
        gate.load_entered.set()
        assert gate.load_release.wait(10), 'synthetic Silero load was not released'
        return model, None

    module = ModuleType('torch')
    module.hub = SimpleNamespace(load=load)
    module.from_numpy, module.no_grad = Tensor, NoGrad
    monkeypatch.setitem(sys.modules, 'torch', module)
    return sdk


def release_sdk(sdk):
    for gate in sdk.gates:
        for event in (gate.load_release, gate.eval_release, gate.infer_release, gate.reset_release):
            event.set()


async def prepare(leaves, tmp_path, monkeypatch, url, stack):
    import src.ambient.segmenter as segmenter_module

    original, created, entries = segmenter_module.make_silero_predictor, [], []

    def observe_factory():
        entries.append(threading.get_ident())
        predictor = original()
        created.append(predictor)
        return predictor

    monkeypatch.setattr(segmenter_module, 'make_silero_predictor', observe_factory)
    config = offline_config(tmp_path, url, rooms=True)
    config['rooms']['living'] = {'name': 'Synthetic living', 'mic_device_index': 0}
    config['ambient'] = {'enabled': True, 'shadow_mode': True, 'poll_interval_s': 60,
        'db_path': str(tmp_path / 'ambient.db'), 'stt': {'engine': 'parakeet'},
        'keep_audio': {'enabled': False, 'dir': str(tmp_path / 'clips')}}
    graph = await build_application(config, shutdown=stack,
                                    factories=replace(leaves, wake_detector=OfflineWake))
    transcriber = graph.ambient_path.transcriber
    await transcriber.start()
    return SimpleNamespace(graph=graph, loop=graph.pipeline.audio_loop,
        transcriber=transcriber, room=transcriber._room_states['living'],
        created=created, entries=entries)


def stamped(g):
    tap = g.graph.ambient_path.tap
    tap.push('living', CHUNK, ts=g.graph.privacy.ledger.clock(), tts_active=False)
    items = tap.drain_owned('living')
    assert len(items) == 1
    return items


def close_graph(close_path, g, stack):
    operation = stack.aclose() if close_path == 'factory_stack' else g.graph._stop_services(None, None)
    return asyncio.create_task(operation)


async def observe_cancelled_close(closing, g):
    await until(lambda: g.transcriber._closed)
    for _ in range(2):
        assert closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
    await until(lambda: g.loop._closed)


def retained(g):
    assert g.graph.privacy.resource_status()['writer_lease']
    assert g.graph.dashboard_authority._lease_fd is not None
    assert not g.graph.ha_client._session.closed


@pytest.mark.parametrize('close_path', ['factory_stack', 'runtime_shutdown'])
async def test_wake_and_ambient_models_wait_for_inference_postlude_and_reset(
    leaves, tmp_path, monkeypatch, close_path,
):
    sdk = install_torch(monkeypatch)
    wake_post, wake_release = threading.Event(), threading.Event()
    ambient_post, ambient_release = threading.Event(), threading.Event()
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            g = await prepare(leaves, tmp_path, monkeypatch, url, stack)
            stream, capture = g.loop.room_streams['living'], g.loop._capture_rooms['living']
            await g.loop._capture_native.run(g.loop._load_capture_models, stream)
            await g.room.feed(stamped(g))
            wake = g.loop._wake_vad_predictors['living']
            segmenter = g.room._segmenter
            ambient = segmenter._vad_predict
            assert len(sdk.models) == len(g.created) == 2 and wake is not ambient
            sdk.gates[0].infer_entered.clear()
            sdk.gates[0].infer_release.clear()
            original_feed = g.room._feed

            def wake_job():
                value = compute_wake_vad(CHUNK, wake)
                wake_post.set()
                assert wake_release.wait(10), 'synthetic wake caller postlude was not released'
                return value

            def ambient_job(items):
                borrowed_model = sdk.models[1]()
                value = original_feed(items)
                ambient_post.set()
                try:
                    assert ambient_release.wait(10), 'synthetic ambient caller postlude was not released'
                    assert borrowed_model is sdk.models[1]()
                    return value
                finally:
                    borrowed_model = value = None

            monkeypatch.setattr(g.room, '_feed', ambient_job)
            wake_call = asyncio.create_task(capture.call(wake_job, sources=(), private=False))
            ambient_call = asyncio.create_task(g.room.feed(stamped(g)))
            closing = None
            try:
                await until(lambda: sdk.gates[0].infer_entered.is_set() and ambient_post.is_set())
                assert all(thread != threading.get_ident() for _, thread, _ in sdk.windows)
                for gate in sdk.gates:
                    gate.reset_entered.clear()
                    gate.reset_release.clear()
                wake_call.cancel()
                ambient_call.cancel()
                await asyncio.gather(wake_call, ambient_call, return_exceptions=True)
                closing = close_graph(close_path, g, stack)
                await observe_cancelled_close(closing, g)
                retained(g)
                assert all(model() is not None for model in sdk.models)
                sdk.gates[0].infer_release.set()
                await until(wake_post.is_set)
                assert not closing.done()
                ambient_release.set()
                await until(sdk.gates[1].reset_entered.is_set)
                retained(g)
                sdk.gates[1].reset_release.set()
                wake_release.set()
                await until(sdk.gates[0].reset_entered.is_set)
                assert not closing.done() and sdk.models[0]() is not None
                retained(g)
            finally:
                wake_release.set()
                ambient_release.set()
                release_sdk(sdk)
                await asyncio.gather(wake_call, ambient_call, return_exceptions=True)
                if closing is not None:
                    await asyncio.gather(closing, return_exceptions=True)
            # Keep both callable wrappers and the segmenter alive deliberately:
            # dropping owner containers alone is not explicit backend retirement.
            assert callable(wake) and callable(ambient) and segmenter is not None
            assert not g.loop._wake_vad_predictors
            assert all(model() is None for model in sdk.models)
            with pytest.raises(RuntimeError, match='^ambient_segmenter_closed$'):
                segmenter.feed(ts=g.graph.privacy.ledger.clock(), chunk=CHUNK, tts_active=False)
        assert not g.graph.privacy.resource_status()['writer_lease']
        assert g.graph.dashboard_authority._lease_fd is None
        assert g.graph.ha_client._session.closed and observations['service_calls'] == 0


@pytest.mark.parametrize('close_path', ['factory_stack', 'runtime_shutdown'])
async def test_late_wake_load_and_independent_ambient_eval_cannot_escape_close(
    leaves, tmp_path, monkeypatch, close_path,
):
    sdk = install_torch(monkeypatch)
    sdk.hold_load.add(0)
    sdk.hold_eval.add(1)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            g = await prepare(leaves, tmp_path, monkeypatch, url, stack)
            stream = g.loop.room_streams['living']
            wake_call = asyncio.create_task(g.loop._capture_native.run(g.loop._load_capture_models, stream))
            ambient_call = closing = None
            try:
                await until(lambda: len(sdk.gates) == 1 and sdk.gates[0].load_entered.is_set())
                ambient_call = asyncio.create_task(g.room._call(g.room._ensure_segmenter))
                # This is an actual second admitted factory waiting for the
                # shared Hub lock, not a substituted recovery/controller.
                await until(lambda: len(g.entries) == 2)
                for observer in (wake_call, ambient_call):
                    observer.cancel()
                await asyncio.gather(wake_call, ambient_call, return_exceptions=True)
                closing = close_graph(close_path, g, stack)
                await observe_cancelled_close(closing, g)
                retained(g)
                assert sdk.models[0]() is not None
                sdk.gates[0].load_release.set()
                await until(lambda: len(sdk.gates) == 2 and sdk.gates[1].eval_entered.is_set())
                # Ambient still owns the global load lock during eval. The
                # old wake instance must retire without taking that lock.
                await until(lambda: g.loop._closing is not None and g.loop._closing.done())
                assert g.loop._closing.exception() is None
                assert sdk.models[0]() is None
                assert sdk.models[1]() is not None and not closing.done()
                assert 'living' not in g.loop._wake_vad_predictors
                retained(g)
            finally:
                release_sdk(sdk)
                await asyncio.gather(wake_call, *([ambient_call] if ambient_call is not None else []),
                                     return_exceptions=True)
                if closing is not None:
                    await asyncio.gather(closing, return_exceptions=True)
            assert len(g.created) == len(sdk.models) == 2
            assert all(callable(predictor) for predictor in g.created)
            assert all(model() is None for model in sdk.models)
            assert not g.loop._wake_vad_predictors and g.room._segmenter is None
        assert not g.graph.privacy.resource_status()['writer_lease']
        assert g.graph.dashboard_authority._lease_fd is None
        assert g.graph.ha_client._session.closed and observations['service_calls'] == 0


async def test_real_ambient_factory_retires_predictor_when_segmenter_constructor_fails(
    leaves, tmp_path, monkeypatch,
):
    sdk = install_torch(monkeypatch)

    def later_failure(self, vad_predict, *args, **kwargs):
        assert callable(vad_predict)
        raise RuntimeError('synthetic segmenter construction failure')

    monkeypatch.setattr(UtteranceSegmenter, '__init__', later_failure)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            g = await prepare(leaves, tmp_path, monkeypatch, url, stack)
            await g.room.feed(stamped(g))
            assert len(g.created) == len(sdk.models) == 1
            assert g.room._segmenter is None
            # Keep the actual callable returned by the successful native model
            # factory. A failed dependent constructor must still retire it.
            assert callable(g.created[0])
            assert sdk.models[0]() is None
        assert not g.graph.privacy.resource_status()['writer_lease']
        assert observations['service_calls'] == 0


async def test_default_borrowed_segmenter_and_predictor_survive_room_discard(monkeypatch):
    sdk = install_torch(monkeypatch)
    predictor = make_silero_predictor()
    segmenter = UtteranceSegmenter(vad_predict=predictor)
    room = RoomProducer('synthetic-borrowed', lambda: segmenter, require_cleanup=True)
    try:
        await room.feed([(0., CHUNK, False)])
        assert sdk.models[0]() is not None
        await room.begin_close()
        assert sdk.models[0]() is not None
        # Both ownership defaults are borrowed; reset stays reusable and
        # neither wrapper may withdraw the caller's independent predictor.
        assert predictor(CHUNK[:, 0]) == pytest.approx(.8)
        assert segmenter.feed(ts=1., chunk=CHUNK, tts_active=False) == []
        assert len(sdk.models) == 1
    finally:
        release_sdk(sdk)
        await room.begin_close()
        close = getattr(predictor, 'close', None)
        if callable(close):
            await asyncio.to_thread(close)
