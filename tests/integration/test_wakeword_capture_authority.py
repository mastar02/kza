"""Real wakeword capture APIs bound to HTTP/SQLite; only SDK leaves are synthetic."""
from __future__ import annotations

import asyncio
import builtins
from contextlib import asynccontextmanager
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import wave

import numpy as np
import pytest

from src.audio.native_capture_authority import WakewordTrainingAuthorityClient
from src.wakeword.recorder import WakeWordRecorder
from tests.integration.test_capture_browser_owners import browser_app
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase, controlled
from tests.mocks.calibration_authority import serve_capture_app
from tests.mocks.wakeword_capture import CaptureCall, TrainingAuthorityFactory, WakeCaptureLeaves


@asynccontextmanager
async def bound_wakeword(tmp_path, monkeypatch, dashboard_auth):
    async with controlled(tmp_path, monkeypatch) as graph:
        async with browser_app(graph.control, tmp_path) as app:
            async with serve_capture_app(app) as url:
                with monkeypatch.context() as patch:
                    sdk = WakeCaptureLeaves(patch)
                    credential = dashboard_auth['Authorization'].split(' ', 1)[1]
                    factory = TrainingAuthorityFactory(url, credential)
                    clients = []
                    initialize = WakewordTrainingAuthorityClient.__init__

                    def observe(client, *args, **kwargs):
                        initialize(client, *args, **kwargs)
                        clients.append(client)

                    patch.setattr(WakewordTrainingAuthorityClient, '__init__', observe)
                    recorder = WakeWordRecorder('synthetic', output_dir=str(tmp_path / 'wav'),
                        sample_duration=.01, authority_factory=factory)
                    calls = []

                    def start(operation):
                        call = CaptureCall(operation)
                        calls.append(call)
                        return call

                    state = SimpleNamespace(graph=graph, sdk=sdk, factory=factory, recorder=recorder,
                        start=start, clients=clients, url=url, credential=credential)
                    try:
                        yield state
                    finally:
                        # HTTP retirement needs the ASGI loop while native callers
                        # return; never park that loop in Thread.join.
                        await asyncio.to_thread(sdk.release)
                        for call in calls:
                            await asyncio.to_thread(call.join)
                        for client in clients:
                            await asyncio.to_thread(client.close)


async def finished(call):
    # The real CLI retains its three-step countdown for both recording kinds;
    # this is containment, not a provider/performance deadline assertion.
    assert await asyncio.to_thread(call.done.wait, 8), 'public capture did not return'
    await asyncio.to_thread(call.join)
    assert call.worker._handle.is_done()


def wav_values(path):
    with wave.open(str(path), 'rb') as output:
        header = [output.getnchannels(), output.getsampwidth(), output.getframerate(), output.getnframes()]
        pcm = output.readframes(output.getnframes())
    return header, np.frombuffer(pcm, dtype=np.int16)


def strict_stream_receipts(sdk):
    assert sdk.streams
    for stream in sdk.streams:
        assert stream.stop_kwargs == [{'ignore_errors': False}]
        assert stream.close_kwargs == [{'ignore_errors': False}]
    assert sdk.stream_closed.is_set()


@pytest.mark.parametrize('kind', ['positive', 'negative'])
@pytest.mark.parametrize('authority_kind', ['none', 'foreign'])
def test_direct_live_recording_requires_nominal_explicit_authority(tmp_path, monkeypatch, kind, authority_kind):
    sdk = WakeCaptureLeaves(monkeypatch)
    factory = None if authority_kind == 'none' else lambda: object()
    try:
        with pytest.raises(RuntimeError):
            recorder = WakeWordRecorder('synthetic', output_dir=str(tmp_path),
                sample_duration=.01, authority_factory=factory)
            getattr(recorder, f'record_{kind}_samples')(n=1, delay_between=0)
        assert not sdk.events and not sdk.streams and not sdk.detectors
        assert not list(tmp_path.rglob('*.wav'))
    finally:
        sdk.release()


@pytest.mark.asyncio
async def test_muted_real_backend_denies_before_wakeword_device_acquisition(tmp_path, monkeypatch, dashboard_auth):
    async with bound_wakeword(tmp_path, monkeypatch, dashboard_auth) as b:
        b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
        await capture_phase(b.graph.control, 'muted')
        call = b.start(lambda: b.recorder.record_positive_samples(n=1, delay_between=0))
        await finished(call)
        assert call.failure is not None
        assert not b.sdk.events and not b.sdk.streams and not b.sdk.detectors
        assert not list(tmp_path.rglob('*.wav'))
        state = b.graph.control.snapshot()
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']


@pytest.mark.asyncio
async def test_positive_negative_preserve_pcm_wav_progress_and_use_fresh_clients(tmp_path, monkeypatch, dashboard_auth):
    async with bound_wakeword(tmp_path, monkeypatch, dashboard_auth) as b:
        progress = []
        for kind in ['positive', 'negative']:
            method = getattr(b.recorder, f'record_{kind}_samples')
            call = b.start(lambda: method(n=1, delay_between=0,
                callback=lambda current, total: progress.append((kind, current, total))))
            await finished(call)
            assert call.failure is None and call.result == 1
            paths = list((b.recorder.output_dir / kind).glob('*.wav'))
            assert len(paths) == 1 and paths[0].name == f'{kind}_0001.wav'
            header, pcm = wav_values(paths[0])
            assert header == [1, 2, 16000, 160]
            assert np.all(pcm == np.int16(.25 * 32767))
            assert b.graph.control.snapshot()['pending_native_clients'] == 0
        assert progress == [('positive', 1, 1), ('negative', 1, 1)]
        assert len(b.factory.clients) == 2 and b.factory.clients[0] is not b.factory.clients[1]
        assert all(not client.current() for client in b.factory.clients)
        assert sum(b.sdk.read_frames) == 320
        assert all({k: stream.kwargs[k] for k in ['samplerate', 'channels', 'dtype']} == {
            'samplerate': 16000, 'channels': 1, 'dtype': 'float32'} for stream in b.sdk.streams)
        strict_stream_receipts(b.sdk)


@pytest.mark.parametrize('boundary', ['start', 'read'])
@pytest.mark.asyncio
async def test_mute_waits_for_owned_start_or_read_and_does_not_auto_resume(tmp_path, monkeypatch, dashboard_auth, boundary):
    async with bound_wakeword(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.hold.add(boundary)
        call = b.start(lambda: b.recorder.record_positive_samples(n=2, delay_between=0))
        try:
            await until(b.sdk.event(boundary).is_set)
            assert len(b.factory.clients) == 1 and b.graph.control.snapshot()['pending_native_clients'] == 1
            b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
            await until(lambda: not b.factory.clients[0].current())
            assert not call.done.is_set() and call.worker.is_alive()
            assert not b.graph.control.snapshot()['total_mute_confirmed']
            assert not list(tmp_path.rglob('*.wav'))
        finally:
            b.sdk.release_event(boundary).set()
        await finished(call)
        assert call.failure is not None and not list(tmp_path.rglob('*.wav'))
        strict_stream_receipts(b.sdk)
        state = await capture_phase(b.graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']
        reads, streams = list(b.sdk.read_frames), len(b.sdk.streams)
        b.graph.control.request(False, state['revision'], actor())
        await capture_phase(b.graph.control, 'running')
        assert call.worker._handle.is_done()
        assert b.sdk.read_frames == reads and len(b.sdk.streams) == streams
        assert len(b.factory.clients) == 1  # Resume is permission, not a new invocation.


@pytest.mark.parametrize('boundary', ['wav-write-return', 'progress'])
@pytest.mark.asyncio
async def test_mute_retains_admitted_wav_write_and_progress_until_actual_return(tmp_path, monkeypatch, dashboard_auth, boundary):
    async with bound_wakeword(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.hold.add(boundary)
        save = WakeWordRecorder._save_wav

        def save_and_hold(recorder, path, audio):
            result = save(recorder, path, audio)
            b.sdk.checkpoint('wav-write-return')
            return result

        monkeypatch.setattr(WakeWordRecorder, '_save_wav', save_and_hold)
        call = b.start(lambda: b.recorder.record_positive_samples(n=1, delay_between=0,
            callback=lambda current, total: b.sdk.checkpoint('progress')))
        try:
            await until(b.sdk.event(boundary).is_set)
            assert len(list(tmp_path.rglob('*.wav'))) == 1
            assert b.graph.control.snapshot()['pending_native_clients'] == 1
            b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
            await until(lambda: not b.factory.clients[0].current())
            assert not call.done.is_set() and not b.graph.control.snapshot()['total_mute_confirmed']
        finally:
            b.sdk.release_event(boundary).set()
        await finished(call)
        assert call.failure is not None
        assert len(list(tmp_path.rglob('*.wav'))) == 1  # Mute is not historical erasure.
        state = await capture_phase(b.graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']
        strict_stream_receipts(b.sdk)


@pytest.mark.parametrize('boundary', ['detect', 'detection-print'])
@pytest.mark.asyncio
async def test_actual_cmd_test_waits_for_callback_or_detection_print_before_ack(tmp_path, monkeypatch, dashboard_auth, boundary):
    async with bound_wakeword(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.emit_callback = True
        b.sdk.hold.add(boundary)
        output = builtins.print

        def print_and_hold(*args, **kwargs):
            result = output(*args, **kwargs)
            if any('Detectado:' in str(value) for value in args):
                b.sdk.checkpoint('detection-print')
            return result

        monkeypatch.setattr(builtins, 'print', print_and_hold)
        monkeypatch.setenv('KZA_WAKEWORD_TEST_OPERATOR', b.credential)
        command = runpy.run_path(str(Path('scripts/train_wakeword.py')))['cmd_test']
        args = SimpleNamespace(model='synthetic.onnx', threshold=.61,
            dashboard_url=b.url, token_env='KZA_WAKEWORD_TEST_OPERATOR')
        call = b.start(lambda: command(args))
        try:
            await until(b.sdk.event(boundary).is_set)
            assert b.graph.control.snapshot()['pending_native_clients'] == 1
            b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
            await until(lambda: bool(b.clients) and not b.clients[0].current())
            assert call.worker.is_alive() and not call.done.is_set()
            assert not b.sdk.detector_closed.is_set()
            assert not b.graph.control.snapshot()['total_mute_confirmed']
        finally:
            b.sdk.release_event(boundary).set()
        await finished(call)
        assert b.sdk.callback_error is None and b.sdk.event('callback-returned').is_set()
        assert b.sdk.detector_closed.is_set()
        assert b.sdk.detectors[0].kwargs['models'] == ['synthetic.onnx']
        assert b.sdk.detectors[0].kwargs['threshold'] == .61
        assert {k: b.sdk.streams[0].kwargs[k] for k in ['samplerate', 'channels', 'dtype', 'blocksize']} == {
            'samplerate': 16000, 'channels': 1, 'dtype': 'float32', 'blocksize': 1280}
        assert all(worker._handle.is_done() for worker in b.sdk.threads)
        strict_stream_receipts(b.sdk)
        state = await capture_phase(b.graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']


@pytest.mark.asyncio
async def test_constructor_without_returned_stream_cannot_ack_native_retirement(tmp_path, monkeypatch, dashboard_auth):
    async with bound_wakeword(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.faults.add('construct')
        call = b.start(lambda: b.recorder.record_positive_samples(n=1, delay_between=0))
        await finished(call)
        assert call.failure is not None and not list(tmp_path.rglob('*.wav'))
        assert b.graph.control.snapshot()['pending_native_clients'] == 1
        await asyncio.to_thread(b.factory.clients[0].close)
        b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
        await capture_phase(b.graph.control, 'muting')
        state = b.graph.control.snapshot()
        assert state['pending_native_clients'] == 1 and not state['total_mute_confirmed']


@pytest.mark.asyncio
async def test_original_operator_revocation_seals_read_without_replacing_actor(tmp_path, monkeypatch, dashboard_auth):
    async with bound_wakeword(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.hold.add('read')
        call = b.start(lambda: b.recorder.record_negative_samples(n=2, delay_between=0))
        try:
            await until(b.sdk.event('read').is_set)
            assert b.graph.control.snapshot()['pending_native_clients'] == 1
            monkeypatch.setenv('KZA_DASHBOARD_TOKEN', 'different-synthetic-operator')
            await until(lambda: not b.factory.clients[0].current())
            assert not call.done.is_set() and call.worker.is_alive()
            assert b.graph.control.snapshot()['pending_native_clients'] == 1
        finally:
            b.sdk.release_event('read').set()
        await finished(call)
        assert call.failure is not None and not list(tmp_path.rglob('*.wav'))
        assert len(b.factory.clients) == 1 and not b.factory.clients[0].current()
        assert b.graph.control.snapshot()['pending_native_clients'] == 0
        strict_stream_receipts(b.sdk)


@pytest.mark.asyncio
async def test_actual_cmd_record_positive_and_negative_bind_fresh_authority(tmp_path, monkeypatch, dashboard_auth):
    async with bound_wakeword(tmp_path, monkeypatch, dashboard_auth) as b:
        monkeypatch.setenv('KZA_WAKEWORD_TEST_OPERATOR', b.credential)
        monkeypatch.setattr(builtins, 'input', lambda *args: '')
        command = runpy.run_path(str(Path('scripts/train_wakeword.py')))['cmd_record']
        args = SimpleNamespace(name='synthetic-cli', output_dir=str(tmp_path / 'cli-wav'),
            positive=True, negative=True, count=1, dashboard_url=b.url, token_env='KZA_WAKEWORD_TEST_OPERATOR')
        call = b.start(lambda: command(args))
        await finished(call)
        assert call.failure is None
        paths = sorted((tmp_path / 'cli-wav').rglob('*.wav'))
        assert len(paths) == 2 and {p.name for p in paths} == {'positive_0001.wav', 'negative_0001.wav'}
        assert len(b.clients) == 2 and b.clients[0] is not b.clients[1]
        assert all(not client.current() for client in b.clients)
        assert b.graph.control.snapshot()['pending_native_clients'] == 0
        strict_stream_receipts(b.sdk)


def test_historical_add_audio_train_and_list_need_no_live_authority(tmp_path, monkeypatch, capsys):
    sdk = WakeCaptureLeaves(monkeypatch)
    recorder = WakeWordRecorder('synthetic', output_dir=str(tmp_path / 'history'))
    for positive in [True, False]:
        for _ in range(30):
            path = Path(recorder.add_sample_from_audio(np.full((2, 1), .25), positive))
            assert wav_values(path)[0] == [1, 2, 16000, 2]
    trained = []

    class Trainer:
        def __init__(self, **kwargs):
            pass

        def train(self, **kwargs):
            trained.append(kwargs)
            return str(tmp_path / 'synthetic.onnx')

        def list_trained_models(self):
            return [{'name': 'synthetic-history', 'size_kb': 1, 'positive_samples': 30}]

    trainer_module = ModuleType('src.wakeword.trainer')
    trainer_module.WakeWordTrainer = Trainer
    monkeypatch.setitem(sys.modules, 'src.wakeword.trainer', trainer_module)
    monkeypatch.delenv('KZA_DASHBOARD_TOKEN', raising=False)
    script = runpy.run_path(str(Path('scripts/train_wakeword.py')))
    monkeypatch.setattr(sys, 'argv', ['train_wakeword.py', 'train', '--name', 'synthetic',
        '--data-dir', str(tmp_path / 'history'), '--output-dir', str(tmp_path / 'models'), '--epochs', '1'])
    assert script['main']() == 0
    monkeypatch.setattr(sys, 'argv', ['train_wakeword.py', 'list'])
    assert script['main']() is None
    assert trained == [{'wake_word_name': 'synthetic', 'epochs': 1}]
    assert 'synthetic-history' in capsys.readouterr().out
    assert not sdk.events and not sdk.streams and not sdk.detectors
