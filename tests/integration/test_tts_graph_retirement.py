"""Selected Dual/Kokoro lifetime through the real offline application graph."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import sys
import threading
from types import SimpleNamespace
from weakref import ref

import numpy as np
import pytest

from src.application import ApplicationFactories
from src.main import build_application
from src.tts.piper_tts import DualTTS, KokoroTTS, create_tts
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, offline_config, offline_home


pytestmark = pytest.mark.asyncio


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


def selected_config(tmp_path, url):
    config = offline_config(tmp_path, url)
    config["tts"] = {
        "engine": "dual",
        "kokoro": {"model": "synthetic-kokoro", "device": "cpu", "lang_code": "e"},
        "streaming": {"enabled": False, "shutdown_timeout_s": .01},
    }
    return config


@pytest.fixture
def selected_kokoro(monkeypatch, tmp_path):
    """Replace only SDK/model leaves; never initialize a real speech model."""
    import src.vectordb.chroma_sync as index_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(index_module, "SentenceTransformer", lambda *a, **kw: OfflineEncoder())
    sdk = SimpleNamespace(
        entered=threading.Event(), release=threading.Event(),
        iterator_exited=threading.Event(), collected=threading.Event(),
        models=[], constructors=[], native_threads=[],
    )
    sdk.release.set()

    class Pipeline:
        def __init__(self, *, lang_code, device):
            sdk.constructors.append((lang_code, device))
            sdk.models.append(ref(self, lambda _model: sdk.collected.set()))

        def __call__(self, text, *, voice):
            sdk.native_threads.append(threading.get_ident())
            sdk.entered.set()
            try:
                assert sdk.release.wait(5), "synthetic Kokoro iterator was not released"
                yield SimpleNamespace(audio=np.full(8192, .125, dtype=np.float32))
            finally:
                sdk.iterator_exited.set()

    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=Pipeline))
    stt = OfflineSTT()
    factories = ApplicationFactories(stt=lambda _: stt, tts=create_tts, preflight=lambda: None)
    return sdk, factories


def close_graph(close_path, graph, stack):
    # Both paths use actual owners, without starting capture, playback or graph.run.
    return asyncio.create_task(
        stack.aclose() if close_path == "factory_stack" else graph._stop_services(None, None)
    )


async def assert_retained(graph, closing):
    responses = graph.voice_services.responses
    await until(lambda: responses.get_delivery_status()["closed"] or closing.done())
    assert responses.get_delivery_status()["closed"]
    # The graph must retain real exit beyond the bounded public output observer.
    done, _ = await asyncio.wait({closing}, timeout=.05)
    assert not done
    assert not graph.ha_client._session.closed
    assert graph.dashboard_authority._lease_fd is not None
    assert graph.privacy.resource_status()["writer_lease"]


async def cancel_close_twice(closing):
    assert closing.cancel()
    await asyncio.sleep(0)
    assert closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()


@pytest.mark.parametrize("close_path", ["factory_stack", "runtime"])
async def test_selected_kokoro_iterator_outlives_cancelled_observers_until_actual_exit(
    selected_kokoro, tmp_path, close_path,
):
    sdk, factories = selected_kokoro
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(selected_config(tmp_path, url), shutdown=stack, factories=factories)
            tts, responses = graph.models.tts, graph.voice_services.responses
            assert type(tts) is DualTTS and type(tts.kokoro) is KokoroTTS and tts.qwen3 is None
            sdk.release.clear()
            observer = asyncio.create_task(responses.asynthesize("Synthetic held iterator"))
            closing = None
            results = []
            try:
                assert await asyncio.to_thread(sdk.entered.wait, 3)
                assert len(sdk.native_threads) == 1
                assert sdk.native_threads[0] != threading.get_ident()
                assert sdk.constructors == [("e", "cpu")]
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                closing = close_graph(close_path, graph, stack)
                await assert_retained(graph, closing)
                await cancel_close_twice(closing)
                assert not sdk.iterator_exited.is_set()
                assert tts.kokoro._model is sdk.models[0]()
                assert not sdk.collected.is_set()
            finally:
                sdk.release.set()
                results = await asyncio.gather(
                    *[task for task in (observer, closing) if task is not None], return_exceptions=True,
                )
                await responses.aclose()
            assert isinstance(results[-1], asyncio.CancelledError)
            assert sdk.iterator_exited.is_set()
            assert tts.kokoro._model is None
            assert sdk.models[0]() is None and sdk.collected.is_set()
            with pytest.raises(RuntimeError):
                await asyncio.to_thread(tts.synthesize, "Synthetic late synthesis")
            assert sdk.constructors == [("e", "cpu")]
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()["writer_lease"]
        assert observations["service_calls"] == 0


@pytest.mark.parametrize("close_path", ["factory_stack", "runtime"])
async def test_graph_waits_for_actual_synthesis_return_after_worker_slot_is_released(
    selected_kokoro, tmp_path, monkeypatch, close_path,
):
    sdk, factories = selected_kokoro
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(selected_config(tmp_path, url), shutdown=stack, factories=factories)
            tts, responses = graph.models.tts, graph.voice_services.responses
            entered, release = threading.Event(), threading.Event()
            finished = responses._worker_finished

            def held_postlude(session):
                borrowed_model = tts.kokoro._model
                try:
                    finished(session)
                    entered.set()
                    assert release.wait(5), "synthetic worker postlude was not released"
                    assert borrowed_model is sdk.models[0]()
                finally:
                    borrowed_model = None

            monkeypatch.setattr(responses, "_worker_finished", held_postlude)
            observer = asyncio.create_task(responses.asynthesize("Synthetic held return"))
            closing = None
            results = []
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                assert sdk.iterator_exited.is_set()
                assert responses.get_delivery_status()["workers"] == 0
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                closing = close_graph(close_path, graph, stack)
                await assert_retained(graph, closing)
                await cancel_close_twice(closing)
                assert tts.kokoro._model is sdk.models[0]()
                assert not sdk.collected.is_set()
            finally:
                release.set()
                results = await asyncio.gather(
                    *[task for task in (observer, closing) if task is not None], return_exceptions=True,
                )
                await responses.aclose()
            assert isinstance(results[-1], asyncio.CancelledError)
            assert tts.kokoro._model is None
            assert sdk.models[0]() is None and sdk.collected.is_set()
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()["writer_lease"]
        assert observations["service_calls"] == 0


async def test_loaded_selected_kokoro_is_retired_after_later_factory_failure(
    selected_kokoro, tmp_path, monkeypatch,
):
    import src.main as main_module

    sdk, factories = selected_kokoro
    adapters = []

    def loaded_tts(config):
        tts = create_tts(config)
        # KPipeline is already the synthetic constructor; no real model is loaded.
        tts.load()
        adapters.append(tts)
        return tts

    def fail_index(*args, **kwargs):
        raise RuntimeError("synthetic failure after TTS construction")

    monkeypatch.setattr(main_module, "ChromaSync", fail_index)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            with pytest.raises(RuntimeError, match="synthetic failure after TTS construction"):
                await build_application(
                    selected_config(tmp_path, url), shutdown=stack,
                    factories=replace(factories, tts=loaded_tts),
                )
        assert len(adapters) == 1 and sdk.constructors == [("e", "cpu")]
        assert adapters[0].kokoro._model is None
        assert sdk.models[0]() is None and sdk.collected.is_set()
        with pytest.raises(RuntimeError):
            await asyncio.to_thread(adapters[0].synthesize, "Synthetic post-failure synthesis")
        assert sdk.constructors == [("e", "cpu")]
        assert observations["service_calls"] == 0
