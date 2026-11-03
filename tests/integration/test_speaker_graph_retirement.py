"""Real speaker wrappers/graph owners with synthetic SDK/model leaves only."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import sqlite3
import sys
import threading
from types import ModuleType, SimpleNamespace
from weakref import ref

import numpy as np
import pytest

from src.main import build_application
from src.memory.producer_context import producer_context
from tests.integration.test_application_graph import leaves as application_leaves
from tests.mocks.application_graph import OfflineSTT, OfflineWake, offline_config, offline_home
from tests.mocks.memory_sources import memory_source


leaves = application_leaves
pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('dashboard_auth')]
AUDIO = np.full(16000, .125, dtype=np.float32)


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(.002)


def install_sdk(monkeypatch):
    sdk = SimpleNamespace(
        entered=threading.Event(), release=threading.Event(), exited=threading.Event(),
        models=[], constructors=[], native_threads=[],
    )
    sdk.release.set()

    class Tensor:
        def __init__(self, value):
            self.value = value

        def unsqueeze(self, _axis):
            return self

        def squeeze(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value.copy()

    class Classifier:
        @classmethod
        def from_hparams(cls, **options):
            assert options['source'].startswith('speechbrain/')
            assert options['run_opts'] == {'device': 'cpu'}
            sdk.constructors.append(options)
            model = cls()
            sdk.models.append(ref(model))
            return model

        def encode_batch(self, _audio):
            sdk.native_threads.append(threading.get_ident())
            sdk.entered.set()
            try:
                assert sdk.release.wait(10), 'synthetic speaker inference was not released'
                return Tensor(np.ones(192, dtype=np.float32))
            finally:
                sdk.exited.set()

    torch, torchaudio = ModuleType('torch'), ModuleType('torchaudio')
    torch.from_numpy = Tensor
    torchaudio.list_audio_backends = lambda: []
    speaker = ModuleType('speechbrain.inference.speaker')
    speaker.EncoderClassifier = Classifier
    for name, module in (
        ('torch', torch), ('torchaudio', torchaudio),
        ('speechbrain', ModuleType('speechbrain')),
        ('speechbrain.inference', ModuleType('speechbrain.inference')),
        ('speechbrain.inference.speaker', speaker),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return sdk


def configured(tmp_path, url, *, ambient):
    config = offline_config(tmp_path, url, rooms=ambient)
    config['speaker_id'].update(enabled=True, model='speechbrain/synthetic-command', device='cpu')
    if ambient:
        config['rooms']['living'] = {'name': 'Synthetic living', 'mic_device_index': 0}
        config['ambient'] = {
            'enabled': True, 'db_path': str(tmp_path / 'ambient.db'),
            'stt': {'device': 'cpu'},
            'keep_audio': {'enabled': False, 'dir': str(tmp_path / 'ambient-audio')},
        }
    return config


def close_graph(close_path, graph, stack):
    if close_path == 'factory_stack':
        return asyncio.create_task(stack.aclose())
    return asyncio.create_task(graph._stop_services(None, None))


@pytest.mark.parametrize('origin', ['command', 'ambient'])
@pytest.mark.parametrize('close_path', ['factory_stack', 'runtime_shutdown'])
async def test_graph_retains_registry_inference_after_producer_already_returned(
    leaves, tmp_path, monkeypatch, origin, close_path,
):
    import src.ambient.transcriber as ambient_module

    sdk = install_sdk(monkeypatch)
    monkeypatch.setattr(ambient_module, '_build_ambient_stt_engine', lambda *_args: OfflineSTT())
    async with offline_home() as (url, observations):
        config = configured(tmp_path, url, ambient=origin == 'ambient')
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack,
                factories=replace(leaves, wake_detector=OfflineWake))
            sdk.release.clear()
            closing, close_results = None, []
            try:
                source = memory_source(now=graph.privacy.ledger.clock())
                with producer_context('living', (source,)):
                    if origin == 'command':
                        identifier = graph.models.speaker_identifier
                        result = await graph.pipeline.command_processor.process_command(AUDIO)
                        assert result.success and result.user is None
                    else:
                        assert graph.ambient_path is not None
                        tagger = graph.ambient_path.transcriber._tagger
                        identifier = tagger._identifier
                        assert identifier is not graph.models.speaker_identifier
                        assert await tagger.tag(AUDIO, transcript='synthetic', room_id='living') == ('unknown', 0.)
                        assert not tagger.snapshot_operations()
                # Producer returned. Registry's actual worker still borrows
                # the model; draining only tagger/controller is insufficient.
                await until(sdk.entered.is_set)
                assert len(sdk.models) == len(sdk.native_threads) == 1
                assert sdk.native_threads[0] != threading.get_ident()
                assert graph.voice_registry.resource_status()['pending'] == 1
                assert identifier._model is sdk.models[0]()
                closing = close_graph(close_path, graph, stack)
                await until(lambda: graph.speaker_models._closing is not None)
                actual = graph.speaker_models.begin_close()
                assert graph.voice_registry._closed
                assert not actual.done() and not closing.done()
                assert graph.privacy.resource_status()['writer_lease']
                assert graph.dashboard_authority._lease_fd is not None
                assert not graph.ha_client._session.closed
                assert closing.cancel()
                await asyncio.sleep(0)
                assert closing.cancel()
                await asyncio.sleep(0)
                assert not actual.done() and not closing.done()
                assert graph.speaker_models.begin_close() is actual
                assert not sdk.exited.is_set() and sdk.models[0]() is not None
                progressed = asyncio.Event()
                asyncio.get_running_loop().call_soon(progressed.set)
                await progressed.wait()
            finally:
                sdk.release.set()
                if closing is not None:
                    close_results = await asyncio.gather(closing, return_exceptions=True)
                await graph.speaker_models.aclose()
            assert isinstance(close_results[0], asyncio.CancelledError)
            assert sdk.exited.is_set()
            assert identifier._model is None and sdk.models[0]() is None
            assert graph.speaker_models.begin_close() is actual and actual.exception() is None
            constructors = len(sdk.constructors)
            for function, args in ((identifier.load, ()), (identifier.get_embedding, (AUDIO,))):
                with pytest.raises(RuntimeError, match='^speaker_closed$'):
                    await asyncio.to_thread(function, *args)
            assert len(sdk.constructors) == constructors == 1
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()['writer_lease']
        with sqlite3.connect(config['memory']['privacy_db_path']) as db:
            assert db.execute('SELECT count(*) FROM voice_fingerprints').fetchone()[0] == 0
        assert observations['service_calls'] == 0


@pytest.mark.parametrize('close_path', ['factory_stack', 'runtime_shutdown'])
async def test_partial_ambient_registers_encoder_before_tagger_construction(
    leaves, tmp_path, monkeypatch, close_path,
):
    import src.ambient.speaker_tagger as tagger_module
    import src.ambient.transcriber as ambient_module

    sdk = install_sdk(monkeypatch)
    monkeypatch.setattr(ambient_module, '_build_ambient_stt_engine', lambda *_args: OfflineSTT())
    captured = []

    def later_failure(self, *args, **kwargs):
        identifier = kwargs['identifier']
        # No completed tagger or registry registration exists. The early
        # graph model registration must independently retain this instance.
        captured.append(identifier)
        identifier.load()
        raise RuntimeError('synthetic tagger construction failure')

    monkeypatch.setattr(tagger_module.SpeakerTagger, '__init__', later_failure)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(configured(tmp_path, url, ambient=True), shutdown=stack,
                factories=replace(leaves, wake_detector=OfflineWake))
            assert graph.ambient_path is None and len(captured) == 1
            identifier = captured[0]
            assert len(sdk.models) == 1 and identifier._model is sdk.models[0]()
            await close_graph(close_path, graph, stack)
            assert identifier._model is None and sdk.models[0]() is None
            with pytest.raises(RuntimeError, match='^speaker_closed$'):
                await asyncio.to_thread(identifier.load)
            assert len(sdk.constructors) == 1
        assert graph.ha_client._session.closed
        assert graph.dashboard_authority._lease_fd is None
        assert not graph.privacy.resource_status()['writer_lease']
        assert observations['service_calls'] == 0


@pytest.mark.parametrize('close_path', ['factory_stack', 'runtime_shutdown'])
async def test_cancelled_warmup_retains_encoder_through_actual_native_postlude(
    leaves, tmp_path, monkeypatch, close_path,
):
    sdk = install_sdk(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            graph = await build_application(configured(tmp_path, url, ambient=False),
                shutdown=stack, factories=leaves)
            identifier = graph.models.speaker_identifier

            def retained_warmup():
                output = identifier.get_embedding(AUDIO)
                borrowed_model = identifier._model
                try:
                    entered.set()
                    assert release.wait(5), 'synthetic warmup postlude was not released'
                    assert borrowed_model is sdk.models[0]()
                    return output
                finally:
                    borrowed_model = None

            caller = asyncio.create_task(graph.pipeline.command_processor.awarmup_model(
                identifier, retained_warmup,
            ))
            closing = None
            try:
                await until(entered.is_set)
                assert sdk.exited.is_set() and identifier._speaker_active == 0
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await caller
                closing = close_graph(close_path, graph, stack)
                await until(lambda: graph.speaker_models._closing is not None)
                assert not closing.done()
                assert identifier._model is sdk.models[0]()
                assert graph.privacy.resource_status()['writer_lease']
                assert graph.dashboard_authority._lease_fd is not None
                assert not graph.ha_client._session.closed
                release.set()
                await closing
                assert identifier._model is None and sdk.models[0]() is None
            finally:
                release.set()
                await asyncio.gather(
                    *(task for task in (caller, closing) if task is not None),
                    return_exceptions=True,
                )
                await graph.speaker_models.aclose()
        assert observations['service_calls'] == 0


async def test_later_principal_factory_failure_retires_already_loaded_encoder(
    leaves, tmp_path, monkeypatch,
):
    import src.main as main_module

    sdk = install_sdk(monkeypatch)
    captured = []

    def fail_enrollment(*args, **kwargs):
        identifier = kwargs['speaker_identifier']
        captured.append(identifier)
        identifier.load()
        raise RuntimeError('synthetic failure after principal encoder construction')

    monkeypatch.setattr(main_module, 'VoiceEnrollment', fail_enrollment)
    async with offline_home() as (url, observations):
        async with AsyncExitStack() as stack:
            with pytest.raises(RuntimeError, match='synthetic failure after principal encoder'):
                await build_application(configured(tmp_path, url, ambient=False),
                    shutdown=stack, factories=leaves)
        assert len(captured) == len(sdk.constructors) == 1
        assert captured[0]._model is None and sdk.models[0]() is None
        with pytest.raises(RuntimeError, match='^speaker_closed$'):
            await asyncio.to_thread(captured[0].load)
        assert len(sdk.constructors) == 1
        assert observations['service_calls'] == 0
