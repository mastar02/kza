"""The real lazy AudioManager backend remains owned by single capture."""

import asyncio
from contextlib import AsyncExitStack
import threading

import pytest

from src.main import build_application
from src.pipeline.audio_loop import AudioLoop
from src.wakeword.detector import WakeWordDetector
from tests.integration.test_wake_graph_retirement import (
    AUDIO, close_graph, configured, install_sdk, leaves as graph_leaves,
    release_sdk, retained, until,
)
from tests.mocks.application_graph import offline_home


leaves = graph_leaves
pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('dashboard_auth')]


@pytest.mark.parametrize('stage', ['constructor', 'postlude'])
@pytest.mark.parametrize('close_path', ['factory_stack', 'runtime_shutdown'])
async def test_single_lazy_wake_retains_constructor_callback_and_native_postlude(
    leaves, tmp_path, monkeypatch, stage, close_path,
):
    sdk = install_sdk(monkeypatch)
    callback_entered, callback_release = threading.Event(), threading.Event()
    postlude_entered, postlude_release = threading.Event(), threading.Event()
    callback_calls = []
    if stage == 'constructor':
        sdk.hold_constructors.add(0)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(configured(tmp_path, url, rooms=False),
                                            shutdown=stack, factories=leaves)
            loop = graph.pipeline.audio_loop
            assert type(loop) is AudioLoop
            manager = loop.audio_manager
            assert manager._wake_model is None
            caller = closing = None
            close_results = []
            try:
                if stage == 'constructor':
                    caller = asyncio.create_task(loop.start())
                    await until(lambda: len(sdk.gates) == 1 and sdk.gates[0].ctor_entered.is_set())
                    detector = manager._wake_model
                    assert type(detector) is WakeWordDetector
                    assert detector._oww_model is None and sdk.models[0]() is not None
                else:
                    await loop.start()
                    detector = manager._wake_model
                    assert type(detector) is WakeWordDetector

                    def callback(name, score):
                        callback_calls.append((name, score))
                        callback_entered.set()
                        assert callback_release.wait(10), 'synthetic single wake callback was not released'

                    detector.on_detection(callback)

                    def invoke_with_postlude():
                        borrowed_model = detector._oww_model
                        result = manager.detect_wake_word(AUDIO)
                        postlude_entered.set()
                        try:
                            assert postlude_release.wait(10), 'synthetic single postlude was not released'
                            assert borrowed_model is sdk.models[0]()
                            return result
                        finally:
                            borrowed_model = result = None

                    caller = asyncio.create_task(loop._room.call(
                        invoke_with_postlude, sources=(), private=False))
                    await until(callback_entered.is_set)
                    assert detector._oww_model is sdk.models[0]()
                caller.cancel()
                result = await asyncio.gather(caller, return_exceptions=True)
                assert isinstance(result[0], asyncio.CancelledError)
                assert loop._holder is None
                closing = close_graph(close_path, graph, stack)
                await until(lambda: loop._closed and graph.wake_models._closing is not None)
                actual = graph.wake_models.begin_close()
                for _ in range(2):
                    assert closing.cancel()
                    await asyncio.sleep(0)
                    assert not closing.done() and not actual.done()
                assert manager._wake_model is detector
                assert sdk.models[0]() is not None and sdk.loads == 1
                retained(graph)
                if stage == 'postlude':
                    callback_release.set()
                    await until(postlude_entered.is_set)
                    # detect() and its inline callback have now returned; the
                    # actual CaptureRoom native caller still owns the backend.
                    assert manager._wake_model is detector
                    assert detector._oww_model is sdk.models[0]()
                    assert not closing.done() and not actual.done()
                    assert graph.wake_models.begin_close() is actual
                    retained(graph)
                progressed = asyncio.Event()
                asyncio.get_running_loop().call_soon(progressed.set)
                await progressed.wait()
            finally:
                callback_release.set()
                postlude_release.set()
                release_sdk(sdk)
                if caller is not None:
                    await asyncio.gather(caller, return_exceptions=True)
                if closing is not None:
                    close_results = await asyncio.gather(closing, return_exceptions=True)
                await graph.wake_models.aclose()
            assert isinstance(close_results[0], asyncio.CancelledError)
            assert manager._wake_model is None and detector._oww_model is None
            assert sdk.models[0]() is None
            assert loop._closed and loop._holder is None
            with pytest.raises(RuntimeError, match='^wake_backend_closed$'):
                await asyncio.to_thread(manager.load_wake_word)
            assert sdk.loads == 1
            assert callback_calls == ([('synthetic_wake', .95)] if stage == 'postlude' else [])
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()['writer_lease']
        assert observations['service_calls'] == 0
