"""Selected factory models must retire even when callers retain the adapter."""

from contextlib import AsyncExitStack, nullcontext
from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace
from weakref import ref

import pytest

from src.ambient.segmenter import make_silero_predictor
from src.main import build_application
from tests.integration.test_application_graph import leaves as graph_leaves
from tests.mocks.application_graph import OfflineWake, offline_config, offline_home


leaves = graph_leaves
pytestmark = pytest.mark.usefixtures('dashboard_auth')


def install_torch(monkeypatch, *, fail_eval=False):
    models = []

    class Model:
        def eval(self):
            if fail_eval:
                raise RuntimeError('synthetic_silero_eval_failure')

        def reset_states(self):
            pass

        def __call__(self, audio, rate):
            assert rate == 16000
            return SimpleNamespace(item=lambda: .8)

    def load(**options):
        assert options == {
            'repo_or_dir': 'snakers4/silero-vad',
            'model': 'silero_vad', 'trust_repo': True,
        }
        model = Model()
        models.append(ref(model))
        return model, ()

    module = ModuleType('torch')
    module.hub = SimpleNamespace(load=load)
    module.from_numpy = lambda value: value
    module.no_grad = nullcontext
    monkeypatch.setitem(sys.modules, 'torch', module)
    return models


@pytest.mark.asyncio
@pytest.mark.parametrize('close_path', ['factory_stack', 'runtime_shutdown'])
async def test_factory_retires_created_room_silero_models_with_retained_callables(
    leaves, tmp_path, monkeypatch, close_path,
):
    models = install_torch(monkeypatch)
    async with offline_home() as (url, observations):
        config = offline_config(tmp_path, url, rooms=True)
        for index, room in enumerate(('living', 'kitchen')):
            config['rooms'][room] = {'name': room, 'mic_device_index': index}
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack,
                                             factories=replace(leaves, wake_detector=OfflineWake))
            loop = graph.pipeline.audio_loop
            for room_id, stream in loop.room_streams.items():
                await loop._capture_rooms[room_id].call(
                    loop._load_capture_models, stream, sources=(), private=False)
            retained_predictors = list(loop._wake_vad_predictors.values())
            assert len(retained_predictors) == len(models) == 2
            assert all(model() is not None for model in models)
            if close_path == 'factory_stack':
                await stack.aclose()
            else:
                await graph._stop_services(None, None)
            assert all(model() is None for model in models)
            assert loop._wake_vad_predictors == {}
            assert all(callable(predictor) for predictor in retained_predictors)
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()['writer_lease']
        assert observations['service_calls'] == 0


def test_eval_failure_retires_partial_model_with_exception_traceback_retained(monkeypatch):
    models = install_torch(monkeypatch, fail_eval=True)
    with pytest.raises(RuntimeError) as failure:
        make_silero_predictor()
    assert failure.value.__traceback__ is not None
    assert len(models) == 1
    assert models[0]() is None
    assert str(failure.value) == 'silero_initialization_failed'
    assert failure.value.__cause__ is failure.value.__context__ is None
