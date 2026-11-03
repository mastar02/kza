"""Actual Parakeet construction and application teardown with an SDK leaf."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import sys
from types import ModuleType
from weakref import ref

import pytest

from src.main import build_application
from src.stt.parakeet_stt import ParakeetSTT
from src.stt.whisper_fast import create_stt
from tests.integration import test_application_graph as application_fixtures
from tests.mocks.application_graph import OfflineSTT, offline_config, offline_home


leaves = application_fixtures.leaves


def install_sdk(monkeypatch):
    models, loads = [], []

    class Recognizer:
        def recognize(self, audio, *, language):
            return "synthetic transcription"

    def load_model(name):
        assert name == "synthetic-parakeet"
        loads.append(name)
        model = Recognizer()
        models.append(ref(model))
        return model

    sdk = ModuleType("onnx_asr")
    sdk.load_model = load_model
    monkeypatch.setitem(sys.modules, "onnx_asr", sdk)
    return models, loads


@pytest.mark.asyncio
@pytest.mark.usefixtures("dashboard_auth")
@pytest.mark.parametrize("role", ["primary", "shadow"])
@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
async def test_graph_retires_selected_parakeet_reference(
    leaves, tmp_path, monkeypatch, role, close_path,
):
    models, loads = install_sdk(monkeypatch)

    def stt_factory(config):
        return create_stt(config) if config.get("engine") == "parakeet" else OfflineSTT()

    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        config.setdefault("stt", {}).update(
            engine="parakeet" if role == "primary" else "whisper",
            shadow_engine="parakeet" if role == "shadow" else None,
            parakeet_model="synthetic-parakeet",
        )
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack,
                factories=replace(leaves, stt=stt_factory))
            model = (graph.models.stt if role == "primary"
                else graph.pipeline.command_processor.shadow_stt)
            assert type(model) is ParakeetSTT
            await asyncio.to_thread(model.load)
            assert len(loads) == 1 and model._model is models[0]()
            if close_path == "factory_stack":
                await stack.aclose()
            else:
                await graph._stop_services(None, None)
            assert model._model is None
            assert models[0]() is None
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()["writer_lease"]
        assert observations["service_calls"] == 0
