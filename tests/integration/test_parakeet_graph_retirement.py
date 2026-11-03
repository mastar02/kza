"""Real factory, Parakeet and borrowers; only onnx_asr model leaves are synthetic."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import sys
import threading
from types import ModuleType, SimpleNamespace
from weakref import ref

import numpy as np
import pytest

from src.ambient.ambient_stt import AmbientSTT
from src.main import build_application
from src.memory.producer_context import producer_context
from src.stt.parakeet_stt import ParakeetSTT
from tests.integration.test_application_graph import leaves as application_leaves
from tests.mocks.application_graph import OfflineWake, offline_config, offline_home
from tests.mocks.memory_sources import memory_source


leaves = application_leaves
pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('dashboard_auth')]
AUDIO = np.full(16000, .125, dtype=np.float32)


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(.002)


def install_sdk(monkeypatch):
    sdk = SimpleNamespace(models=[], gates=[], loads=[])

    class Model:
        def __init__(self, gate):
            self.gate = gate

        def recognize(self, audio, *, language):
            assert audio.dtype == np.float32 and audio.ndim == 1 and language == 'es'
            self.gate.threads.append(threading.get_ident())
            self.gate.entered.set()
            try:
                assert self.gate.release.wait(10), 'synthetic ONNX inference was not released'
                return 'Synthetic recognized command'
            finally:
                self.gate.exited.set()

    def load_model(name):
        gate = SimpleNamespace(entered=threading.Event(), release=threading.Event(),
                               exited=threading.Event(), threads=[])
        gate.release.set()
        model = Model(gate)
        sdk.loads.append(name)
        sdk.gates.append(gate)
        sdk.models.append(ref(model))
        return model

    module = ModuleType('onnx_asr')
    module.load_model = load_model
    monkeypatch.setitem(sys.modules, 'onnx_asr', module)
    return sdk


def configured(tmp_path, url):
    config = offline_config(tmp_path, url, rooms=True)
    config['stt'] = {'engine': 'parakeet', 'parakeet_model': 'synthetic-command',
                     'shadow_engine': 'parakeet', 'language': 'es'}
    config['rooms']['living'] = {'name': 'Synthetic living', 'mic_device_index': 0}
    config['ambient'] = {
        'enabled': True, 'db_path': str(tmp_path / 'ambient.db'),
        'stt': {'engine': 'parakeet', 'parakeet_model': 'synthetic-ambient'},
        'keep_audio': {'enabled': False, 'dir': str(tmp_path / 'ambient-audio')},
    }
    return config


def close_graph(close_path, graph, stack):
    operation = stack.aclose() if close_path == 'factory_stack' else graph._stop_services(None, None)
    return asyncio.create_task(operation)


async def load_engine(engine):
    assert type(engine) is ParakeetSTT
    await asyncio.to_thread(engine.load)
    return engine._model.gate


async def cancel_close_twice(closing):
    for _ in range(2):
        assert closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()


def assert_authority_retained(graph):
    assert graph.dashboard_authority._lease_fd is not None
    assert graph.privacy.resource_status()['writer_lease']
    assert not graph.ha_client._session.closed


async def rejects_reload(engine, sdk):
    loads = tuple(sdk.loads)
    for operation, args in ((engine.load, ()), (engine.transcribe, (AUDIO,)),
                            (engine.transcribe_with_confidence, (AUDIO,))):
        with pytest.raises(RuntimeError, match='^parakeet_closed$'):
            await asyncio.to_thread(operation, *args)
    assert tuple(sdk.loads) == loads


@pytest.mark.parametrize('close_path', ['factory_stack', 'runtime_shutdown'])
async def test_shadow_and_ambient_jobs_outlive_observers_and_cancelled_close(
    leaves, tmp_path, monkeypatch, close_path,
):
    sdk = install_sdk(monkeypatch)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(configured(tmp_path, url), shutdown=stack,
                factories=replace(leaves, stt=None, wake_detector=OfflineWake))
            processor = graph.pipeline.command_processor
            assert graph.ambient_path is not None
            ambient = graph.ambient_path.transcriber._stt
            primary, shadow, dedicated = processor.stt, processor.shadow_stt, ambient._stt
            assert len({id(primary), id(shadow), id(dedicated)}) == 3
            await load_engine(primary)
            shadow_gate, ambient_gate = await load_engine(shadow), await load_engine(dedicated)
            shadow_gate.release.clear()
            ambient_gate.release.clear()
            observer = closing = None
            close_results = []
            try:
                source = memory_source(now=graph.privacy.ledger.clock())
                with producer_context('living', (source,)):
                    result = await processor.process_command(AUDIO)
                    assert result.success and result.stt_confidence.no_speech_prob is None
                    observer = asyncio.create_task(ambient.transcribe(AUDIO))
                await until(lambda: shadow_gate.entered.is_set() and ambient_gate.entered.is_set())
                assert all(identifier != threading.get_ident()
                    for gate in (shadow_gate, ambient_gate) for identifier in gate.threads)
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                closing = close_graph(close_path, graph, stack)
                await until(lambda: ambient._work.status()['closed'])
                # Runtime first observes AmbientPath; cancellation must still
                # reach retained pipeline/model-owner cleanup.
                await cancel_close_twice(closing)
                await until(lambda: graph.stt_executors._closing is not None)
                actual = graph.stt_executors.begin_close()
                assert not actual.done() and not actual.cancelled()
                assert_authority_retained(graph)
                assert shadow._model is sdk.models[1]() and dedicated._model is sdk.models[2]()
                shadow_gate.release.set()
                await until(shadow_gate.exited.is_set)
                await until(lambda: processor._work.status()['operations'] == 0)
                assert not ambient_gate.exited.is_set() and not closing.done() and not actual.done()
                assert graph.stt_executors.begin_close() is actual
                assert dedicated._model is sdk.models[2]()
                assert_authority_retained(graph)
                progressed = asyncio.Event()
                asyncio.get_running_loop().call_soon(progressed.set)
                await progressed.wait()
            finally:
                for gate in sdk.gates:
                    gate.release.set()
                if observer is not None:
                    await asyncio.gather(observer, return_exceptions=True)
                if closing is not None:
                    close_results = await asyncio.gather(closing, return_exceptions=True)
                await graph.stt_executors.aclose()
            assert isinstance(close_results[0], asyncio.CancelledError)
            assert shadow_gate.exited.is_set() and ambient_gate.exited.is_set()
            assert len(sdk.loads) == 3
            assert all(engine._model is None for engine in (primary, shadow, dedicated))
            assert all(model() is None for model in sdk.models)
            await rejects_reload(shadow, sdk)
            await rejects_reload(dedicated, sdk)
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()['writer_lease']
        assert graph.ha_client._session.closed and observations['service_calls'] == 0


@pytest.mark.parametrize('close_path', ['factory_stack', 'runtime_shutdown'])
async def test_ambient_postlude_after_model_return_remains_a_graph_borrower(
    leaves, tmp_path, monkeypatch, close_path,
):
    sdk = install_sdk(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(configured(tmp_path, url), shutdown=stack,
                factories=replace(leaves, stt=None, wake_detector=OfflineWake))
            ambient = graph.ambient_path.transcriber._stt
            engine = ambient._stt
            gate = await load_engine(engine)
            invoke = ambient._invoke

            def postlude(*args):
                borrowed_model = engine._model
                value = invoke(*args)
                # Parakeet's public call already returned, but the actual
                # AmbientSTT native invocation has not returned yet.
                assert gate.exited.is_set()
                entered.set()
                try:
                    assert release.wait(10), 'synthetic ambient postlude was not released'
                    assert borrowed_model is sdk.models[0]()
                    return value
                finally:
                    borrowed_model = value = None

            monkeypatch.setattr(ambient, '_invoke', postlude)
            observer = closing = None
            try:
                source = memory_source(now=graph.privacy.ledger.clock())
                with producer_context('living', (source,)):
                    observer = asyncio.create_task(ambient.transcribe(AUDIO))
                await until(entered.is_set)
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                closing = close_graph(close_path, graph, stack)
                await until(lambda: ambient._work.status()['closed'])
                await cancel_close_twice(closing)
                await until(lambda: graph.stt_executors._closing is not None)
                actual = graph.stt_executors.begin_close()
                assert not actual.done() and not closing.done()
                assert engine._model is sdk.models[0]()
                assert_authority_retained(graph)
            finally:
                release.set()
                if observer is not None:
                    await asyncio.gather(observer, return_exceptions=True)
                if closing is not None:
                    await asyncio.gather(closing, return_exceptions=True)
                await graph.stt_executors.aclose()
            assert engine._model is None and sdk.models[0]() is None
            await rejects_reload(engine, sdk)
            assert sdk.loads == ['synthetic-ambient']
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()['writer_lease']
        assert graph.ha_client._session.closed and observations['service_calls'] == 0


async def test_partial_builder_owns_parakeet_before_ambient_wrapper_exists(
    leaves, tmp_path, monkeypatch,
):
    sdk = install_sdk(monkeypatch)
    captured = []

    def later_failure(self, stt, *args, **kwargs):
        assert type(stt) is ParakeetSTT
        stt.load()
        captured.append(stt)
        raise RuntimeError('synthetic AmbientSTT construction failure')

    monkeypatch.setattr(AmbientSTT, '__init__', later_failure)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(configured(tmp_path, url), shutdown=stack,
                factories=replace(leaves, stt=None, wake_detector=OfflineWake))
            assert graph.ambient_path is None and len(captured) == 1
            assert captured[0]._model is sdk.models[0]()
        assert captured[0]._model is None and sdk.models[0]() is None
        await rejects_reload(captured[0], sdk)
        assert sdk.loads == ['synthetic-ambient']
        assert not graph.privacy.resource_status()['writer_lease']
        assert graph.dashboard_authority._lease_fd is None
        assert graph.ha_client._session.closed and observations['service_calls'] == 0


async def test_owned_ambient_engine_does_not_retire_a_borrowed_peer(monkeypatch):
    sdk = install_sdk(monkeypatch)
    owned, peer = ParakeetSTT(model_name='synthetic-owned'), ParakeetSTT(model_name='synthetic-peer')
    own_wrapper, borrowed_wrapper = AmbientSTT(owned, owns_stt=True), AmbientSTT(peer)
    try:
        assert (await own_wrapper.transcribe(AUDIO)).text == 'Synthetic recognized command'
        assert (await borrowed_wrapper.transcribe(AUDIO)).text == 'Synthetic recognized command'
        actual = own_wrapper.begin_close()
        await asyncio.shield(actual)
        assert own_wrapper.begin_close() is actual
        assert owned._model is None and sdk.models[0]() is None
        assert peer._model is sdk.models[1]()
        await borrowed_wrapper.begin_close()
        assert peer._model is sdk.models[1]()
        result = await asyncio.to_thread(peer.transcribe_with_confidence, AUDIO)
        assert result.text == 'Synthetic recognized command'
        await rejects_reload(owned, sdk)
        assert sdk.loads == ['synthetic-owned', 'synthetic-peer']
    finally:
        await asyncio.gather(own_wrapper.begin_close(), borrowed_wrapper.begin_close())
        await asyncio.gather(owned.aclose(), peer.aclose())
    assert all(model() is None for model in sdk.models)
