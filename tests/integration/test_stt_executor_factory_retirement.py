"""The real graph owns its primary and shadow pinned STT executors."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import threading

import numpy as np
import pytest

from src.main import build_application
from src.stt.whisper_fast import FastWhisperSTT, STTResult
from tests.integration import test_application_graph as application_fixtures
from tests.mocks.application_graph import OfflineWake, offline_config, offline_home


leaves = application_fixtures.leaves


@pytest.mark.asyncio
@pytest.mark.usefixtures("dashboard_auth")
@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
async def test_real_graph_retires_primary_and_shadow_executor_threads(
    leaves, tmp_path, monkeypatch, close_path,
):
    models, threads = [], []

    def create(_config):
        model = FastWhisperSTT(model="synthetic", device="cpu")

        def load():
            model._model = object()
            threads.append(threading.current_thread())

        monkeypatch.setattr(model, "_load_impl", load)
        models.append(model)
        return model

    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        config["stt"] = {"shadow_engine": "whisper"}
        async with AsyncExitStack() as stack:
            graph = await build_application(
                config, shutdown=stack, factories=replace(leaves, stt=create),
            )
            try:
                assert len(models) == 2
                assert graph.models.stt is models[0]
                assert graph.pipeline.command_processor.stt is models[0]
                assert graph.pipeline.command_processor.shadow_stt is models[1]
                for model in models:
                    await asyncio.to_thread(model.load)
                assert len(threads) == 2 and threads[0] is not threads[1]
                assert all(thread.is_alive() for thread in threads)
                if close_path == "factory_stack":
                    await stack.aclose()
                else:
                    # Actual runtime teardown, without starting capture/services.
                    await graph._stop_services(None, None)
                assert [thread.name for thread in threads if thread.is_alive()] == []
                assert observations["service_calls"] == 0
            finally:
                # Only a RED cleanup fallback, after the retirement assertion.
                for model in models:
                    executor = model._executor
                    if executor is not None:
                        await asyncio.to_thread(executor.shutdown, wait=True)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


def synthetic_model(monkeypatch):
    model = FastWhisperSTT(model="synthetic", device="cpu")
    threads = []

    def load():
        if model._model is None:
            model._model = object()
            threads.append(threading.current_thread())

    monkeypatch.setattr(model, "_load_impl", load)
    monkeypatch.setattr(model, "_transcribe_impl", lambda _audio: STTResult("Synthetic", 1.0))
    return model, threads


@pytest.mark.asyncio
@pytest.mark.usefixtures("dashboard_auth")
@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
async def test_real_graph_keeps_authority_until_borrower_and_pinned_thread_exit(
    leaves, tmp_path, monkeypatch, close_path,
):
    model, threads = synthetic_model(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def infer(_audio):
        entered.set()
        assert release.wait(5), "synthetic pinned warmup not released"
        return STTResult("Synthetic retained warmup", 1.0)

    monkeypatch.setattr(model, "_transcribe_impl", infer)
    caller = closing = None
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(
                offline_config(tmp_path, url), shutdown=stack,
                factories=replace(leaves, stt=lambda _: model),
            )
            processor = graph.pipeline.command_processor
            try:
                caller = asyncio.create_task(processor.awarmup_model(
                    model, model.transcribe_with_confidence,
                    np.ones(16000, dtype=np.float32),
                ))
                await until(entered.is_set)
                assert len(threads) == 1 and threads[0].is_alive()
                caller.cancel()
                result = await asyncio.gather(caller, return_exceptions=True)
                assert isinstance(result[0], asyncio.CancelledError)
                operation = stack.aclose() if close_path == "factory_stack" else graph._stop_services(None, None)
                closing = asyncio.create_task(operation)
                await until(lambda: graph.stt_executors._closing is not None)
                retained = graph.stt_executors.begin_close()
                for _ in range(2):
                    closing.cancel()
                    await asyncio.sleep(0)
                    assert not closing.done()
                    assert graph.stt_executors.begin_close() is retained
                    assert not retained.done() and not retained.cancelled()
                # The borrowed invocation must finish before adapter retirement.
                assert model._closing is None and not model._closed
                assert threads[0].is_alive() and model._model is not None
                assert not graph.pipeline.request_router._closed
                assert graph.dashboard_authority._lease_fd is not None
                assert graph.privacy.resource_status()["writer_lease"]
                assert not graph.ha_client._session.closed
                release.set()
                await asyncio.wait_for(asyncio.shield(retained), 3)
                await asyncio.gather(closing, return_exceptions=True)
                assert not threads[0].is_alive() and model._model is None
                assert graph.pipeline.request_router._closed
                assert observations["service_calls"] == 0
            finally:
                release.set()
                if closing is not None:
                    await asyncio.gather(closing, return_exceptions=True)
                if caller is not None:
                    await asyncio.gather(caller, return_exceptions=True)
                if model._executor is not None:
                    await asyncio.to_thread(model._executor.shutdown, wait=True)
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()["writer_lease"]
        assert graph.ha_client._session.closed


@pytest.mark.asyncio
@pytest.mark.usefixtures("dashboard_auth")
async def test_early_factory_failure_retires_already_created_pinned_model(
    leaves, tmp_path, monkeypatch,
):
    model, threads = synthetic_model(monkeypatch)
    await asyncio.to_thread(model.load)

    def broken_tts(_config):
        raise RuntimeError("synthetic later factory failure")

    try:
        async with offline_home() as (url, _observations):
            async with AsyncExitStack() as stack:
                with pytest.raises(RuntimeError, match="synthetic later factory failure"):
                    await build_application(
                        offline_config(tmp_path, url), shutdown=stack,
                        factories=replace(leaves, stt=lambda _: model, tts=broken_tts),
                    )
            assert len(threads) == 1 and not threads[0].is_alive()
            assert model._model is None
    finally:
        if model._executor is not None:
            await asyncio.to_thread(model._executor.shutdown, wait=True)


@pytest.mark.asyncio
@pytest.mark.usefixtures("dashboard_auth")
@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
async def test_thread_exit_receipt_precedes_graph_dependency_and_authority_retirement(
    leaves, tmp_path, monkeypatch, close_path,
):
    model, threads = synthetic_model(monkeypatch)
    exited_worker, release_thread = threading.Event(), threading.Event()
    original_run = threading.Thread.run

    def run_with_held_exit(thread):
        original_run(thread)
        if thread.name.startswith("whisper-cpu_"):
            # The executor worker returned; this real OS Thread is still alive.
            exited_worker.set()
            assert release_thread.wait(5), "synthetic thread postlude not released"

    monkeypatch.setattr(threading.Thread, "run", run_with_held_exit)
    closing = None
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(
                offline_config(tmp_path, url), shutdown=stack,
                factories=replace(leaves, stt=lambda _: model),
            )
            try:
                await asyncio.to_thread(model.load)
                operation = stack.aclose() if close_path == "factory_stack" else graph._stop_services(None, None)
                closing = asyncio.create_task(operation)
                await until(exited_worker.is_set)
                retained = graph.stt_executors.begin_close()
                assert model._model is None and model._executor._shutdown
                assert threads[0].is_alive() and not retained.done()
                for _ in range(2):
                    closing.cancel()
                    await asyncio.sleep(0)
                    assert not closing.done() and not retained.done()
                for _ in range(20):
                    await asyncio.sleep(0)
                assert threads[0].is_alive() and not retained.cancelled()
                assert not graph.pipeline.request_router._closed
                assert graph.dashboard_authority._lease_fd is not None
                assert graph.privacy.resource_status()["writer_lease"]
                assert not graph.ha_client._session.closed
                release_thread.set()
                await asyncio.wait_for(asyncio.shield(retained), 3)
                await asyncio.gather(closing, return_exceptions=True)
                assert not threads[0].is_alive()
                assert graph.pipeline.request_router._closed
                assert observations["service_calls"] == 0
            finally:
                release_thread.set()
                if closing is not None:
                    await asyncio.gather(closing, return_exceptions=True)
                if model._executor is not None:
                    await asyncio.to_thread(model._executor.shutdown, wait=True)


@pytest.mark.asyncio
@pytest.mark.usefixtures("dashboard_auth")
async def test_partial_ambient_builder_keeps_created_engine_owned_without_path(
    leaves, tmp_path, monkeypatch,
):
    import src.ambient.transcriber as ambient_module
    from src.ambient.ambient_stt import AmbientSTT

    model, threads = synthetic_model(monkeypatch)
    await asyncio.to_thread(model.load)
    monkeypatch.setattr(ambient_module, "_build_ambient_stt_engine", lambda *_args: model)
    original_init = AmbientSTT.__init__

    def later_failure(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        raise RuntimeError("synthetic partial ambient failure")

    monkeypatch.setattr(AmbientSTT, "__init__", later_failure)
    try:
        async with offline_home() as (url, observations):
            config = offline_config(tmp_path, url, rooms=True)
            config["rooms"]["living"] = {"name": "Synthetic living", "mic_device_index": 0}
            config["ambient"] = {"enabled": True}
            async with AsyncExitStack() as stack:
                graph = await build_application(
                    config, shutdown=stack, factories=replace(leaves, wake_detector=OfflineWake),
                )
                assert graph.ambient_path is None and threads[0].is_alive()
            assert not threads[0].is_alive() and model._model is None
            assert observations["service_calls"] == 0
    finally:
        if model._executor is not None:
            await asyncio.to_thread(model._executor.shutdown, wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [False, True])
async def test_ambient_wrapper_closes_only_its_owned_engine(monkeypatch, owned):
    from src.ambient.ambient_stt import AmbientSTT

    model, threads = synthetic_model(monkeypatch)
    wrapper = AmbientSTT(model, owns_stt=owned)
    try:
        result = await wrapper.transcribe(np.ones(16000, dtype=np.float32))
        assert result.text == "Synthetic" and len(threads) == 1
        closing = wrapper.begin_close()
        await asyncio.wait_for(asyncio.shield(closing), 3)
        assert wrapper.begin_close() is closing
        assert threads[0].is_alive() == (not owned)
        if owned:
            assert model._model is None
        else:
            value = await asyncio.to_thread(model.transcribe_with_confidence, np.ones(16000, dtype=np.float32))
            assert value.text == "Synthetic" and threads[0].is_alive()
    finally:
        await model.aclose()
