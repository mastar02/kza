"""Real selected wake wrappers/capture owners; synthetic OpenWakeWord only."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace
from weakref import ref

import numpy as np
import pytest

from src.main import build_application
from src.wakeword.detector import WakeWordDetector
from tests.integration.test_application_graph import leaves as application_leaves
from tests.mocks.application_graph import offline_config, offline_home


leaves = application_leaves
pytestmark = pytest.mark.usefixtures('dashboard_auth')
AUDIO = np.full(1280, .125, dtype=np.float32)


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(.002)


def install_sdk(monkeypatch):
    sdk = SimpleNamespace(gates=[], models=[], hold_constructors=set(), loads=0)

    class Model:
        def __init__(self, *, wakeword_models, inference_framework):
            assert wakeword_models and inference_framework == 'onnx'
            gate = SimpleNamespace(
                ctor_entered=threading.Event(), ctor_release=threading.Event(),
                predict_entered=threading.Event(), predict_release=threading.Event(),
                reset_entered=threading.Event(), reset_release=threading.Event(), threads=[],
            )
            gate.ctor_release.set()
            gate.predict_release.set()
            gate.reset_release.set()
            if sdk.loads in sdk.hold_constructors:
                gate.ctor_release.clear()
            sdk.loads += 1
            sdk.gates.append(gate)
            sdk.models.append(ref(self))
            self.gate = gate
            gate.ctor_entered.set()
            assert gate.ctor_release.wait(10), 'synthetic wake constructor was not released'

        def predict(self, audio):
            assert audio.dtype == np.int16 and audio.shape == (1280,)
            self.gate.threads.append(threading.get_ident())
            self.gate.predict_entered.set()
            assert self.gate.predict_release.wait(10), 'synthetic wake inference was not released'
            return {'synthetic_wake': .95}

        def reset(self):
            self.gate.reset_entered.set()
            assert self.gate.reset_release.wait(10), 'synthetic wake reset was not released'

    module = ModuleType('openwakeword')
    module.Model = Model
    monkeypatch.setitem(sys.modules, 'openwakeword', module)
    return sdk


def configured(tmp_path, url, *, rooms=True):
    config = offline_config(tmp_path, url, rooms=rooms)
    config['wake_word'] = {'model': 'synthetic.onnx', 'threshold': .5}
    if rooms:
        config['rooms']['wake_word'] = {'engine': 'openwakeword', 'model': 'synthetic.onnx', 'threshold': .5}
        for index, name in enumerate(('living', 'hall')):
            config['rooms'][name] = {'name': name, 'mic_device_index': index}
    return config


def close_graph(close_path, graph, stack):
    operation = stack.aclose() if close_path == 'factory_stack' else graph._stop_services(None, None)
    return asyncio.create_task(operation)


def release_sdk(sdk):
    for gate in sdk.gates:
        gate.ctor_release.set()
        gate.predict_release.set()
        gate.reset_release.set()


def retained(graph):
    assert graph.dashboard_authority._lease_fd is not None
    assert graph.privacy.resource_status()['writer_lease']
    assert not graph.ha_client._session.closed


@pytest.mark.asyncio
@pytest.mark.parametrize('close_path', ['factory_stack', 'runtime_shutdown'])
async def test_graph_retains_ctor_callback_postlude_and_final_reset(
    leaves, tmp_path, monkeypatch, close_path,
):
    sdk = install_sdk(monkeypatch)
    callback_entered, callback_release = threading.Event(), threading.Event()
    postlude_entered, postlude_release = threading.Event(), threading.Event()
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(configured(tmp_path, url), shutdown=stack,
                factories=replace(leaves, wake_detector=None))
            loop = graph.pipeline.audio_loop
            room_a, room_b = loop._capture_rooms['living'], loop._capture_rooms['hall']
            detector_a, detector_b = loop.room_streams['living'].wake_detector, loop.room_streams['hall'].wake_detector
            assert type(detector_a) is type(detector_b) is WakeWordDetector
            await room_b.call(detector_b.load, sources=(), private=False)
            gate_b = sdk.gates[0]
            gate_b.predict_release.clear()
            gate_b.reset_release.clear()
            sdk.hold_constructors.add(1)

            def callback(name, score):
                assert name == 'synthetic_wake' and score == .95
                callback_entered.set()
                assert callback_release.wait(10), 'synthetic wake callback was not released'

            detector_b.on_detection(callback)

            def invoke_with_postlude():
                borrowed_model = detector_b._oww_model
                value = detector_b.detect(AUDIO)
                postlude_entered.set()
                try:
                    assert postlude_release.wait(10), 'synthetic wake postlude was not released'
                    assert borrowed_model is sdk.models[0]()
                    return value
                finally:
                    borrowed_model = value = None

            loading = asyncio.create_task(room_a.call(detector_a.load, sources=(), private=False))
            detection = asyncio.create_task(room_b.call(invoke_with_postlude, sources=(), private=False))
            closing = None
            try:
                await until(lambda: len(sdk.gates) == 2 and sdk.gates[1].ctor_entered.is_set()
                            and gate_b.predict_entered.is_set())
                assert all(identity != threading.get_ident() for identity in gate_b.threads)
                for observer in (loading, detection):
                    observer.cancel()
                await asyncio.gather(loading, detection, return_exceptions=True)
                closing = close_graph(close_path, graph, stack)
                await until(lambda: graph.wake_models._closing is not None)
                actual = graph.wake_models.begin_close()
                for _ in range(2):
                    assert closing.cancel()
                    await asyncio.sleep(0)
                    assert not closing.done() and not actual.done()
                retained(graph)
                assert all(model() is not None for model in sdk.models)
                sdk.gates[1].ctor_release.set()
                await until(sdk.gates[1].reset_entered.is_set)
                assert not closing.done() and not actual.done()
                gate_b.predict_release.set()
                await until(callback_entered.is_set)
                retained(graph)
                assert detector_b._oww_model is sdk.models[0]()
                callback_release.set()
                await until(postlude_entered.is_set)
                assert not actual.done() and detector_b._oww_model is sdk.models[0]()
                postlude_release.set()
                await until(gate_b.reset_entered.is_set)
                retained(graph)
                assert not closing.done() and not actual.done()
                assert graph.wake_models.begin_close() is actual
            finally:
                callback_release.set()
                postlude_release.set()
                release_sdk(sdk)
                await asyncio.gather(loading, detection, return_exceptions=True)
                if closing is not None:
                    results = await asyncio.gather(closing, return_exceptions=True)
                    assert isinstance(results[0], asyncio.CancelledError)
                await graph.wake_models.aclose()
            assert all(detector._oww_model is None for detector in (detector_a, detector_b))
            assert all(model() is None for model in sdk.models)
            for detector in (detector_a, detector_b):
                with pytest.raises(RuntimeError, match='^wake_closed$'):
                    await asyncio.to_thread(detector.predict, AUDIO)
                with pytest.raises(RuntimeError, match='^wake_closed$'):
                    await asyncio.to_thread(detector.load)
            assert sdk.loads == 2
        assert not graph.privacy.resource_status()['writer_lease']
        assert graph.dashboard_authority._lease_fd is None
        assert graph.ha_client._session.closed and observations['service_calls'] == 0


@pytest.mark.asyncio
async def test_reset_reuses_owned_backend_and_graph_close_preserves_external_peer(
    leaves, tmp_path, monkeypatch,
):
    sdk = install_sdk(monkeypatch)
    peer = WakeWordDetector(models=['synthetic-peer.onnx'])
    try:
        async with offline_home() as (url, observations):
            async with AsyncExitStack() as stack:
                graph = await build_application(configured(tmp_path, url), shutdown=stack,
                    factories=replace(leaves, wake_detector=None))
                loop = graph.pipeline.audio_loop
                room = loop._capture_rooms['living']
                detector = loop.room_streams['living'].wake_detector
                await room.call(detector.load, sources=(), private=False)
                first = sdk.models[0]
                await room.call(detector.reset, sources=(), private=False)
                assert detector._oww_model is first() and sdk.loads == 1
                scores = await room.call(detector.predict, AUDIO, sources=(), private=False)
                assert scores == {'synthetic_wake': .95} and sdk.loads == 1
                await asyncio.to_thread(peer.load)
                assert peer._oww_model is sdk.models[1]()
            assert detector._oww_model is None and first() is None
            assert peer._oww_model is sdk.models[1]()
            assert await asyncio.to_thread(peer.predict, AUDIO) == {'synthetic_wake': .95}
            assert sdk.loads == 2 and observations['service_calls'] == 0
            assert not graph.privacy.resource_status()['writer_lease']
    finally:
        release_sdk(sdk)
        await peer.aclose()
    assert all(model() is None for model in sdk.models)


@pytest.mark.asyncio
async def test_partial_factory_failure_retires_room_models_before_loop_exists(
    leaves, tmp_path, monkeypatch,
):
    import src.main as main_module

    sdk = install_sdk(monkeypatch)
    captured = []

    def later_failure(*args, **kwargs):
        for stream in kwargs['room_streams'].values():
            assert type(stream.wake_detector) is WakeWordDetector
            stream.wake_detector.load()
            captured.append(stream.wake_detector)
        raise RuntimeError('synthetic MultiRoom construction failure')

    monkeypatch.setattr(main_module, 'MultiRoomAudioLoop', later_failure)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            with pytest.raises(RuntimeError, match='^synthetic MultiRoom construction failure$'):
                await build_application(configured(tmp_path, url), shutdown=stack,
                    factories=replace(leaves, wake_detector=None))
            assert len(captured) == sdk.loads == 2
            assert all(model() is not None for model in sdk.models)
        assert all(detector._oww_model is None for detector in captured)
        assert all(model() is None for model in sdk.models)
        for detector in captured:
            with pytest.raises(RuntimeError, match='^wake_closed$'):
                await asyncio.to_thread(detector.load)
        assert sdk.loads == 2 and observations['service_calls'] == 0


async def failure_child(path, kind):
    """The child process is final owner of an intentionally failed lease."""
    from src.application import ApplicationFactories
    import src.core.owned_work as native_module
    import src.vectordb.chroma_sync as index_module
    from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS

    patch = pytest.MonkeyPatch()
    patch.chdir(path)
    patch.setattr(index_module, 'SentenceTransformer', lambda *args, **kwargs: OfflineEncoder())
    sdk = install_sdk(patch)
    stt, tts = OfflineSTT(), OfflineTTS()
    factories = ApplicationFactories(stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None)
    async with offline_home() as (url, observations):
        stack = AsyncExitStack()
        try:
            graph = await build_application(configured(path, url, rooms=kind == 'multi'),
                                            shutdown=stack, factories=factories)
            loop = graph.pipeline.audio_loop
            if kind == 'multi':
                room = loop._capture_rooms['living']
                detector = loop.room_streams['living'].wake_detector
                await room.call(detector.load, sources=(), private=False)
            else:
                # AudioLoop.start loads the lazy backend without opening a
                # capture device. Its input is opened only by run().
                await loop.start()
                room = loop._room
                detector = loop.audio_manager._wake_model
            assert sdk.loads == 1 and detector._oww_model is sdk.models[0]()
            original = asyncio.to_thread
            rejected = []

            async def reject_reset(function, *args, **kwargs):
                target = args[0] if function is native_module.observe_native and args else None
                if getattr(target, '__self__', None) is room and getattr(target, '__name__', None) == '_reset':
                    rejected.append(True)
                    raise RuntimeError('synthetic_reset_transport_unavailable')
                return await original(function, *args, **kwargs)

            # Reject only the actual native submission of this room's reset;
            # do not replace/cancel any controller, owner or close method.
            patch.setattr(asyncio, 'to_thread', reject_reset)
            closing = graph.privacy.begin_close()
            async with asyncio.timeout(3):
                results = await asyncio.gather(asyncio.shield(closing), return_exceptions=True)
            assert rejected and isinstance(results[0], RuntimeError)
            assert closing.done() and graph.privacy.begin_close() is closing
            assert loop.begin_close().done() and isinstance(loop.begin_close().exception(), RuntimeError)
            assert graph.privacy.resource_status()['writer_lease']
            assert graph.privacy.ledger._db is not None
            assert detector._oww_model is sdk.models[0]() and sdk.models[0]() is not None
            if kind == 'multi':
                owner = graph.wake_models.begin_close()
                assert owner.done() and isinstance(owner.exception(), RuntimeError)
            assert observations['service_calls'] == 0
            print('failed_capture_retains_backend_and_privacy_lease_' + kind, flush=True)
        finally:
            release_sdk(sdk)
            # AsyncExitStack must still retire unrelated resources. Expected
            # failed barriers remain failed; no private state is reset and no
            # failed owner is cancelled to manufacture quiet.
            try:
                await stack.aclose()
            except Exception:
                pass
            patch.undo()


@pytest.mark.parametrize('kind', ['multi', 'single'])
def test_unconfirmed_native_cleanup_cannot_release_backend_or_privacy_lease(tmp_path, kind):
    code = (
        'import asyncio, runpy, sys\n'
        'from pathlib import Path\n'
        'failure_child = runpy.run_path(sys.argv[1])["failure_child"]\n'
        'asyncio.run(failure_child(Path(sys.argv[2]), sys.argv[3]))\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code, str(Path(__file__).resolve()), str(tmp_path), kind],
        cwd=Path(offline_config.__code__.co_filename).resolve().parents[2],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'failed_capture_retains_backend_and_privacy_lease_' + kind in result.stdout
