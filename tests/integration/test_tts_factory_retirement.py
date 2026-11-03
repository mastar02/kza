"""Real graph and Dual/Kokoro wrappers with only KPipeline inference synthetic."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import sys
import threading
import types

import numpy as np
import pytest

from src.main import build_application
from src.tts.piper_tts import create_tts
from tests.integration import test_application_graph as application_fixtures
from tests.mocks.application_graph import offline_config, offline_home


leaves = application_fixtures.leaves


def install_pipeline(monkeypatch, *, held=False):
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    if not held:
        release.set()
    instances = []

    class Pipeline:
        def __init__(self, *, lang_code, device):
            assert device == "cpu"
            instances.append(self)

        def __call__(self, text, *, voice):
            entered.set()
            try:
                assert release.wait(5), "synthetic Kokoro inference not released"
                yield types.SimpleNamespace(audio=np.zeros(64, dtype=np.float32))
            finally:
                exited.set()

    module = types.ModuleType("kokoro")
    module.KPipeline = Pipeline
    monkeypatch.setitem(sys.modules, "kokoro", module)
    return entered, release, exited, instances


def config_for_tts(tmp_path, url):
    config = offline_config(tmp_path, url)
    config["tts"] = {
        "engine": "dual", "kokoro": {"device": "cpu"},
        "streaming": {"enabled": False, "shutdown_timeout_s": 0.02},
    }
    return config


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
@pytest.mark.usefixtures("dashboard_auth")
@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
async def test_real_graph_releases_selected_kokoro_reference_after_teardown(
    leaves, tmp_path, monkeypatch, close_path,
):
    entered, _release, exited, instances = install_pipeline(monkeypatch)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(
                config_for_tts(tmp_path, url), shutdown=stack,
                factories=replace(leaves, tts=create_tts),
            )
            tts = graph.models.tts
            assert graph.voice_services.responses.tts is tts and tts.qwen3 is None
            await graph.voice_services.responses.asynthesize("Synthetic startup")
            assert entered.is_set() and exited.is_set() and len(instances) == 1
            assert tts.kokoro._model is instances[0]
            if close_path == "factory_stack":
                await stack.aclose()
            else:
                await graph._stop_services(None, None)
            assert tts.kokoro._model is None
            assert observations["service_calls"] == 0


@pytest.mark.asyncio
@pytest.mark.usefixtures("dashboard_auth")
async def test_factory_keeps_authority_while_selected_native_synthesis_is_running(
    leaves, tmp_path, monkeypatch,
):
    entered, release, exited, instances = install_pipeline(monkeypatch, held=True)
    caller = closing = None
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(
                config_for_tts(tmp_path, url), shutdown=stack,
                factories=replace(leaves, tts=create_tts),
            )
            responses = graph.voice_services.responses
            try:
                caller = asyncio.create_task(responses.asynthesize("Synthetic retained input"))
                await until(entered.is_set)
                caller.cancel()
                result = await asyncio.gather(caller, return_exceptions=True)
                assert isinstance(result[0], asyncio.CancelledError)
                closing = asyncio.create_task(stack.aclose())
                await asyncio.wait({closing}, timeout=0.1)
                assert not exited.is_set() and len(instances) == 1
                assert graph.dashboard_authority._lease_fd is not None
                assert graph.privacy.resource_status()["writer_lease"]
                assert not graph.ha_client._session.closed
                assert not closing.done()
                release.set()
                await asyncio.wait_for(asyncio.shield(closing), 3)
                assert exited.is_set() and graph.models.tts.kokoro._model is None
                assert observations["service_calls"] == 0
            finally:
                release.set()
                if caller is not None:
                    await asyncio.gather(caller, return_exceptions=True)
                if closing is not None:
                    await asyncio.gather(closing, return_exceptions=True)
                await responses.aclose()
