"""Selected speaker adapter retirement through the real application factory."""

import asyncio
from contextlib import AsyncExitStack
import sys
from types import ModuleType
from weakref import ref

import pytest

from src.main import build_application
from tests.integration import test_application_graph as application_fixtures
from tests.mocks.application_graph import offline_config, offline_home


leaves = application_fixtures.leaves


def install_synthetic_speechbrain(monkeypatch):
    models, constructors = [], []

    class Classifier:
        @classmethod
        def from_hparams(cls, **options):
            assert options["source"] == "speechbrain/synthetic-encoder"
            assert options["run_opts"] == {"device": "cpu"}
            constructors.append(options)
            model = cls()
            models.append(ref(model))
            return model

    torchaudio = ModuleType("torchaudio")
    torchaudio.list_audio_backends = lambda: []
    speaker = ModuleType("speechbrain.inference.speaker")
    speaker.EncoderClassifier = Classifier
    for name, module in (
        ("torchaudio", torchaudio),
        ("speechbrain", ModuleType("speechbrain")),
        ("speechbrain.inference", ModuleType("speechbrain.inference")),
        ("speechbrain.inference.speaker", speaker),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return models, constructors


@pytest.mark.asyncio
@pytest.mark.usefixtures("dashboard_auth")
@pytest.mark.parametrize("close_path", ["factory_stack", "runtime_shutdown"])
async def test_selected_speaker_reference_is_retired_after_graph_teardown(
    leaves, tmp_path, monkeypatch, close_path,
):
    models, constructors = install_synthetic_speechbrain(monkeypatch)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url)
        config["speaker_id"].update(
            enabled=True, model="speechbrain/synthetic-encoder", device="cpu",
        )
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            identifier = graph.models.speaker_identifier
            await asyncio.to_thread(identifier.load)
            assert len(constructors) == 1 and identifier._model is models[0]()
            if close_path == "factory_stack":
                await stack.aclose()
            else:
                await graph._stop_services(None, None)
            assert identifier._model is None
            assert models[0]() is None
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()["writer_lease"]
        assert observations["service_calls"] == 0
