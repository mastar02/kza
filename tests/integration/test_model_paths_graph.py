"""main.py model-path sites go through paths.models_root (U3b-0e 2b, F2/N10).

The real graph is built offline: the speaker identifiers (command and
ambient), the openWakeWord custom directory and the Porcupine defaults must
come from the configured root, never from a cwd-relative ./models.
"""

from contextlib import AsyncExitStack
from dataclasses import replace
import sys
from types import ModuleType

import pytest

from src.main import build_application
from tests.integration.test_application_graph import leaves as application_leaves
from tests.integration.test_speaker_graph_retirement import configured, install_sdk
from tests.mocks.application_graph import OfflineSTT, OfflineWake, offline_home


leaves = application_leaves
pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('dashboard_auth')]


def _capturing_wake(seen):
    def factory(**kwargs):
        seen.append(kwargs)
        return OfflineWake(**kwargs)
    return factory


@pytest.mark.parametrize('savedir', ['explicit', 'derived'])
async def test_speaker_and_wake_paths_come_from_the_configured_root(leaves, tmp_path, monkeypatch, savedir):
    import src.ambient.transcriber as ambient_module

    install_sdk(monkeypatch)
    monkeypatch.setattr(ambient_module, '_build_ambient_stt_engine', lambda *_args: OfflineSTT())
    monkeypatch.chdir(tmp_path)
    models = tmp_path / 'models-root'
    cache = tmp_path / 'speaker-cache'
    wake_calls = []
    async with offline_home() as (url, _observations):
        config = configured(tmp_path, url, ambient=True)
        config['paths'] = {'models_root': str(models)}
        config['speaker_id']['savedir'] = str(cache) if savedir == 'explicit' else ''
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack,
                factories=replace(leaves, wake_detector=_capturing_wake(wake_calls)))
            command = graph.models.speaker_identifier
            ambient = graph.ambient_path.transcriber._tagger._identifier
            base = cache if savedir == 'explicit' else models / 'speaker'
            assert command.savedir == str(base / 'speechbrain_synthetic-command')
            assert ambient.savedir == str(base / 'speechbrain_spkrec-ecapa-voxceleb')
            assert [call['custom_models_dir'] for call in wake_calls] == [str(models / 'wakeword')]
    assert not (tmp_path / 'models').exists()


async def test_porcupine_defaults_come_from_the_configured_root(leaves, tmp_path, monkeypatch):
    import src.ambient.transcriber as ambient_module

    install_sdk(monkeypatch)
    monkeypatch.setattr(ambient_module, '_build_ambient_stt_engine', lambda *_args: OfflineSTT())
    seen = []
    porcupine = ModuleType('src.wakeword.porcupine_wake')

    def detector(**kwargs):
        seen.append(kwargs)
        return OfflineWake(**kwargs)
    porcupine.PorcupineWakeDetector = detector
    monkeypatch.setitem(sys.modules, 'src.wakeword.porcupine_wake', porcupine)
    models = tmp_path / 'models-root'
    async with offline_home() as (url, _observations):
        config = configured(tmp_path, url, ambient=True)
        config['paths'] = {'models_root': str(models)}
        config['rooms']['wake_word'] = {'engine': 'porcupine'}
        async with AsyncExitStack() as stack:
            await build_application(config, shutdown=stack, factories=leaves)
    assert [(call['keyword_path'], call['model_path']) for call in seen] == [
        (str(models / 'wakeword' / 'nexa_es.ppn'), str(models / 'wakeword' / 'porcupine_params_es.pv'))]
