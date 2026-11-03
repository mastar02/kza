"""Wake backend ownership through real factory and manager teardown."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import sys
from types import ModuleType
from weakref import ref

import numpy as np
import pytest

from src.main import build_application
from src.pipeline.audio_manager import AudioManager
from src.wakeword.detector import WakeWordDetector
from tests.integration import test_application_graph as application_fixtures
from tests.mocks.application_graph import offline_config, offline_home


leaves = application_fixtures.leaves


def install_sdk(monkeypatch):
    models, loads = [], []

    class Model:
        def __init__(self, **options):
            assert options == {
                "wakeword_models": ["hey_jarvis"], "inference_framework": "onnx",
            }
            loads.append(options)
            models.append(ref(self))

        def predict(self, audio):
            assert audio.dtype == np.int16
            return {"hey_jarvis": 0.0}

        def reset(self):
            pass

    sdk = ModuleType("openwakeword")
    sdk.Model = Model
    monkeypatch.setitem(sys.modules, "openwakeword", sdk)
    return models, loads


@pytest.mark.asyncio
@pytest.mark.usefixtures("dashboard_auth")
@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
async def test_graph_retires_every_created_room_wake_backend(
    leaves, tmp_path, monkeypatch, close_path,
):
    models, loads = install_sdk(monkeypatch)
    detectors = []

    def create_detector(**options):
        detector = WakeWordDetector(**options)
        detectors.append(detector)
        return detector

    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url, rooms=True)
        config["rooms"].update(
            living={"name": "Synthetic living", "mic_device_index": 0},
            kitchen={"name": "Synthetic kitchen", "mic_device_index": 1},
            wake_word={"engine": "openwakeword", "model": "hey_jarvis"},
        )
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack,
                factories=replace(leaves, wake_detector=create_detector))
            assert len(detectors) == 2
            for detector in detectors:
                await asyncio.to_thread(detector.load)
            assert len(loads) == 2 and all(model() is not None for model in models)
            if close_path == "factory_stack":
                await stack.aclose()
            else:
                await graph._stop_services(None, None)
            assert all(detector._oww_model is None for detector in detectors)
            assert all(model() is None for model in models)
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()["writer_lease"]
        assert observations["service_calls"] == 0


def test_audio_manager_terminal_cleanup_retires_retained_detector_reference(monkeypatch):
    models, loads = install_sdk(monkeypatch)
    manager = AudioManager()
    manager.load_wake_word()
    detector = manager._wake_model
    assert type(detector) is WakeWordDetector
    assert detector._oww_model is models[0]() and len(loads) == 1
    manager.close_wake_word()
    assert manager._wake_model is None
    assert detector._oww_model is None and models[0]() is None
