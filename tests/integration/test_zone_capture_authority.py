"""Zone inputs bind the original operator and retire real audio work before ACK."""
from __future__ import annotations

import asyncio
import builtins
from contextlib import asynccontextmanager
from pathlib import Path
import runpy
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.audio.multi_mic_capture import MicrophoneConfig, MultiMicCapture
from src.audio.native_capture_authority import ZoneSetupAuthorityClient
from src.dashboard.authority import DashboardRole, EvaluationModality, EvaluationSample
from tests.integration.test_capture_browser_owners import browser_app
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase, controlled
from tests.mocks.calibration_authority import serve_capture_app
from tests.mocks.wakeword_capture import CaptureCall
from tests.mocks.zone_capture import ZoneCaptureLeaves


SCRIPT = Path('scripts/zone_setup.py')


@asynccontextmanager
async def zone_graph(tmp_path, monkeypatch, dashboard_auth):
    async with controlled(tmp_path, monkeypatch) as graph:
        async with browser_app(graph.control, tmp_path) as app:
            async with serve_capture_app(app) as url:
                with monkeypatch.context() as patch:
                    sdk = ZoneCaptureLeaves(patch)
                    clients, calls, captures = [], [], []
                    credential = dashboard_auth['Authorization'].split(' ', 1)[1]
                    patch.setenv('KZA_ZONE_TEST_OPERATOR', credential)
                    initialize = ZoneSetupAuthorityClient.__init__

                    def observe(client, *args, **kwargs):
                        initialize(client, *args, **kwargs)
                        clients.append(client)

                    patch.setattr(ZoneSetupAuthorityClient, '__init__', observe)

                    def factory():
                        return ZoneSetupAuthorityClient(url, credential, poll_interval_s=.02)

                    def call(operation):
                        pending = CaptureCall(operation)
                        calls.append(pending)
                        return pending

                    def capture(microphones=None, **kwargs):
                        result = MultiMicCapture(microphones or [MicrophoneConfig(17, 'first')],
                            authority_factory=factory, **kwargs)
                        captures.append(result)
                        return result

                    def cli(command, **kwargs):
                        namespace = runpy.run_path(str(SCRIPT))
                        args = SimpleNamespace(device=17, duration=.01, mics=[17, 18],
                            threshold=.02, dashboard_url=url, token_env='KZA_ZONE_TEST_OPERATOR')
                        for key, value in kwargs.items():
                            setattr(args, key, value)
                        return call(lambda: namespace[command](args))

                    try:
                        yield SimpleNamespace(graph=graph, app=app, sdk=sdk, clients=clients,
                            factory=factory, call=call, capture=capture, cli=cli)
                    finally:
                        sdk.release()
                        for capture in captures:
                            await asyncio.to_thread(capture.stop)
                        for pending in calls:
                            await asyncio.to_thread(pending.join)
                        for client in clients:
                            await asyncio.to_thread(client.close)


async def reached(event):
    await until(event.is_set)


async def finished(call):
    await reached(call.done)
    await asyncio.to_thread(call.join)
    assert call.worker._handle.is_done()


def strict_retirement(sdk):
    for stream in sdk.streams:
        assert stream.stop_kwargs == [{'ignore_errors': False}]
        assert stream.close_kwargs == [{'ignore_errors': False}]
        assert stream.exited.is_set()
        assert stream.worker._handle.is_done()


@pytest.mark.parametrize('api', ['microphone', 'multi'])
@pytest.mark.parametrize('foreign', [False, True])
def test_direct_live_apis_require_nominal_explicit_authority(monkeypatch, api, foreign):
    sdk = ZoneCaptureLeaves(monkeypatch)
    factory = (lambda: object()) if foreign else None
    try:
        if api == 'microphone':
            result = MultiMicCapture.test_microphone(17, .01, authority_factory=factory)
            assert result == {'success': False, 'device_index': 17, 'error': 'zone_capture_failed'}
        else:
            with pytest.raises(RuntimeError, match='zone_capture_authority_required'):
                MultiMicCapture([MicrophoneConfig(17, 'first')], authority_factory=factory).start()
        assert not sdk.streams and not sdk.events
    finally:
        sdk.release()


@pytest.mark.parametrize('command', ['cmd_test_mic', 'cmd_test_detection'])
@pytest.mark.parametrize('role', [DashboardRole.HOME, DashboardRole.EVALUATOR])
@pytest.mark.asyncio
async def test_real_cli_never_infers_admin_from_voice_or_household(tmp_path, monkeypatch, dashboard_auth, command, role):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        now = time.time()
        samples = (EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}),
            frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 60),) if role is DashboardRole.EVALUATOR else ()
        authority = b.app.state.dashboard_authority
        grant = await authority.issue_credential(authority.resolve(dashboard_auth['Authorization']),
            subject='synthetic-zone-member', role=role, expires_at=now + 120, samples=samples)
        monkeypatch.setenv('KZA_ZONE_TEST_OPERATOR', grant.token)
        call = b.cli(command)
        await finished(call)
        assert call.failure is not None or call.result != 0
        assert len(b.clients) == 1
        assert not b.sdk.streams and not b.sdk.events
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.parametrize('api', ['microphone', 'multi'])
@pytest.mark.asyncio
async def test_muted_real_backend_denies_both_apis_before_sdk(tmp_path, monkeypatch, dashboard_auth, api):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
        await capture_phase(b.graph.control, 'muted')
        capture = b.capture() if api == 'multi' else None
        call = b.call(capture.start if capture else lambda: MultiMicCapture.test_microphone(
            17, .01, authority_factory=b.factory))
        await finished(call)
        assert call.failure is not None or not call.result['success']
        assert len(b.clients) == 1
        assert not b.sdk.streams and not b.sdk.events
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.parametrize('command', ['cmd_test_mic', 'cmd_test_detection'])
@pytest.mark.asyncio
async def test_real_cli_uses_current_original_actor_and_preserves_report(tmp_path, monkeypatch, dashboard_auth, capsys, command):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.auto_audio = {17: np.full((1600, 1), .25, np.float32),
            18: np.full((1600, 1), .125, np.float32)}
        call = b.cli(command)
        await finished(call)
        assert call.failure is None and call.result == 0
        strict_retirement(b.sdk)
        assert len(b.clients) == 1
        assert b.graph.control.snapshot()['pending_native_clients'] == 0
        output = capsys.readouterr().out
        if command == 'cmd_test_mic':
            assert 'RMS Level: 0.2500' in output and 'Peak Level: 0.2500' in output
            assert sum(frames for _, frames in b.sdk.reads) == 160
        else:
            assert len(b.sdk.streams) == 2
            assert 'Prueba completada' in output


@pytest.mark.asyncio
async def test_valid_admission_retires_partial_delegated_start_and_all_peers(tmp_path, monkeypatch, dashboard_auth):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.start_failure_device = 18
        capture = b.capture([MicrophoneConfig(17, 'first'), MicrophoneConfig(18, 'second')])
        call = b.call(capture.start)
        await finished(call)
        assert len(b.clients) == 1 and len(b.sdk.streams) == 2
        await asyncio.to_thread(capture.stop)
        strict_retirement(b.sdk)
        assert b.graph.control.snapshot()['pending_native_clients'] == 0
        assert all(stream.started.is_set() for stream in b.sdk.streams)


@pytest.mark.parametrize('boundary', ['start', 'read', 'close', 'report'])
@pytest.mark.asyncio
async def test_microphone_ack_waits_for_admitted_sdk_and_report_return(tmp_path, monkeypatch, dashboard_auth, boundary):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.hold.add(boundary)
        reports = []

        def report(result, emit):
            assert result['success']
            reports.append(result.copy())
            b.sdk.checkpoint('report')
            emit(print, 'owned-zone-report')

        call = b.call(lambda: MultiMicCapture.test_microphone(17, .01,
            authority_factory=b.factory, report=report))
        try:
            await reached(b.sdk.event(boundary))
            assert b.graph.control.snapshot()['pending_native_clients'] == 1
            b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
            await until(lambda: bool(b.clients) and not b.clients[0].current())
            assert call.worker.is_alive() and not call.done.is_set()
            state = b.graph.control.snapshot()
            assert state['pending_native_clients'] == 1 and not state['total_mute_confirmed']
            if boundary == 'close':
                assert b.sdk.streams[0].worker._handle.is_done()
        finally:
            b.sdk.release_event(boundary).set()
        await finished(call)
        strict_retirement(b.sdk)
        state = await capture_phase(b.graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']
        if boundary == 'report':
            assert len(reports) == 1


@pytest.mark.parametrize('command', ['cmd_test_mic', 'cmd_test_detection'])
@pytest.mark.asyncio
async def test_cli_report_return_precedes_remote_native_ack(tmp_path, monkeypatch, dashboard_auth, command):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.hold.add('cli-report')
        b.sdk.auto_audio = {17: np.full((1600, 1), .25, np.float32),
            18: np.full((1600, 1), .125, np.float32)}
        original = builtins.print

        def print_and_hold(*args, **kwargs):
            original(*args, **kwargs)
            text = str(args[0]) if args else ''
            if (command == 'cmd_test_mic' and 'RMS Level:' in text) or (
                command == 'cmd_test_detection' and text.startswith('\r')):
                b.sdk.checkpoint('cli-report')

        monkeypatch.setattr(builtins, 'print', print_and_hold)
        call = b.cli(command, duration=.1)
        try:
            await reached(b.sdk.event('cli-report'))
            b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
            await until(lambda: bool(b.clients) and not b.clients[0].current())
            assert call.worker.is_alive() and not call.done.is_set()
            state = b.graph.control.snapshot()
            assert state['pending_native_clients'] == 1 and not state['total_mute_confirmed']
        finally:
            b.sdk.release_event('cli-report').set()
        await finished(call)
        strict_retirement(b.sdk)
        state = await capture_phase(b.graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']


@pytest.mark.asyncio
async def test_mute_retains_native_callback_and_handed_out_chunk_until_return(tmp_path, monkeypatch, dashboard_auth):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.hold.add('user-callback')
        handed = []

        def on_voice(chunk):
            handed.append(chunk)
            b.sdk.checkpoint('user-callback')

        capture = b.capture(on_voice_detected=on_voice)
        start = b.call(capture.start)
        await finished(start)
        audio = np.full((1600, 1), .25, np.float32)
        worker = b.sdk.emit(b.sdk.streams[0], audio)
        stop = None
        try:
            await reached(b.sdk.event('user-callback'))
            b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
            await until(lambda: not b.clients[0].current())
            stop = b.call(capture.stop)
            assert worker.is_alive() and not worker._handle.is_done()
            assert not stop.done.is_set()
            assert not b.graph.control.snapshot()['total_mute_confirmed']
        finally:
            b.sdk.release_event('user-callback').set()
        await finished(stop)
        strict_retirement(b.sdk)
        await capture_phase(b.graph.control, 'muted')
        assert np.array_equal(handed[0].audio_data, np.full(1600, .25, np.float32))
        assert np.array_equal(audio, np.full((1600, 1), .25, np.float32))
        assert not b.sdk.callback_errors


@pytest.mark.asyncio
async def test_unreturned_constructor_stays_pending_without_fabricated_ack(tmp_path, monkeypatch, dashboard_auth):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.faults.add('construct')
        call = b.call(lambda: MultiMicCapture.test_microphone(17, .01, authority_factory=b.factory))
        await finished(call)
        assert call.failure is not None or not call.result['success']
        assert b.sdk.event('construct').is_set() and not b.sdk.threads
        assert b.graph.control.snapshot()['pending_native_clients'] == 1
        await asyncio.to_thread(b.clients[0].close)
        b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
        await capture_phase(b.graph.control, 'muting')
        state = b.graph.control.snapshot()
        assert state['pending_native_clients'] == 1 and not state['total_mute_confirmed']


@pytest.mark.asyncio
async def test_revocation_seals_original_session_and_fresh_start_has_no_stale_queue(tmp_path, monkeypatch, dashboard_auth):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        capture = b.capture()
        start = b.call(capture.start)
        await finished(start)
        worker = b.sdk.emit(b.sdk.streams[0], np.full((1600, 1), .25, np.float32))
        await asyncio.to_thread(b.sdk.join, worker)
        monkeypatch.setenv('KZA_DASHBOARD_TOKEN', 'different-zone-operator')
        await until(lambda: not b.clients[0].current())
        await asyncio.to_thread(capture.stop)
        strict_retirement(b.sdk)
        assert b.graph.control.snapshot()['pending_native_clients'] == 0
        count = len(b.sdk.streams)
        assert len(b.clients) == 1
        monkeypatch.setenv('KZA_DASHBOARD_TOKEN', dashboard_auth['Authorization'].split(' ', 1)[1])
        b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
        muted = await capture_phase(b.graph.control, 'muted')
        b.graph.control.request(False, muted['revision'], actor())
        await capture_phase(b.graph.control, 'running')
        assert len(b.sdk.streams) == count and len(b.clients) == 1
        assert b.sdk.streams[0].worker._handle.is_done()
        start = b.call(capture.start)
        await finished(start)
        assert start.failure is None and len(b.clients) == 2
        assert len(b.sdk.streams) == count + 1
        assert capture.get_next_chunk(timeout=0) is None
        await asyncio.to_thread(capture.stop)
        strict_retirement(b.sdk)


@pytest.mark.asyncio
async def test_ring_wrap_mono_column_rms_and_borrowed_chunk_survive_stop(tmp_path, monkeypatch, dashboard_auth):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        handed = []
        mic = MicrophoneConfig(17, 'first', sample_rate=1000, channels=2, chunk_duration_ms=4)
        capture = b.capture([mic], buffer_duration_sec=.006, vad_threshold=.001,
            on_voice_detected=handed.append)
        start = b.call(capture.start)
        await finished(start)
        stream = b.sdk.streams[0]
        assert stream.kwargs['samplerate'] == 1000 and stream.kwargs['channels'] == 2
        assert stream.kwargs['blocksize'] == 4
        before = time.time()
        chunks = []
        for first in (1, 5):
            mono = np.arange(first, first + 4, dtype=np.float32) / 100
            audio = np.column_stack([mono, np.full(4, 9, np.float32)])
            worker = b.sdk.emit(stream, audio)
            await asyncio.to_thread(b.sdk.join, worker)
            chunk = capture.get_next_chunk(timeout=0)
            chunks.append(chunk)
            assert chunk is handed[-1] and chunk.zone_id == 'first' and chunk.device_index == 17
            assert np.array_equal(chunk.audio_data, mono)
            assert chunk.rms_level == pytest.approx(float(np.sqrt(np.mean(mono ** 2))))
            assert before <= chunk.timestamp <= time.time()
        assert np.allclose(capture.get_recent_audio('first', .006), np.arange(3, 9) / 100)
        preserved = [chunk.audio_data.copy() for chunk in chunks]
        await asyncio.to_thread(capture.stop)
        strict_retirement(b.sdk)
        assert all(np.array_equal(chunk.audio_data, saved) for chunk, saved in zip(chunks, preserved))


@pytest.mark.asyncio
async def test_callback_self_stop_rejects_before_seal_and_external_stop_retires(tmp_path, monkeypatch, dashboard_auth):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        failures = []
        capture = None

        def on_voice(_chunk):
            try:
                capture.stop()
            except RuntimeError as error:
                failures.append(str(error))

        capture = b.capture(on_voice_detected=on_voice)
        start = b.call(capture.start)
        await finished(start)
        worker = b.sdk.emit(b.sdk.streams[0], np.full((1600, 1), .25, np.float32))
        await asyncio.to_thread(b.sdk.join, worker)
        assert failures == ['zone_capture_self_drain']
        assert not b.sdk.streams[0].stop_kwargs and not b.sdk.streams[0].close_kwargs
        assert b.clients[0].current() and b.graph.control.snapshot()['pending_native_clients'] == 1
        await asyncio.to_thread(capture.stop)
        strict_retirement(b.sdk)
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.asyncio
async def test_old_stop_return_cannot_clear_a_fresh_authorized_capture(tmp_path, monkeypatch, dashboard_auth):
    async with zone_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        capture = b.capture()
        start = b.call(capture.start)
        await finished(start)
        assert start.failure is None
        old_owner = capture._capture_owner
        old_stream = b.sdk.streams[0]
        original_finish = old_owner.finish
        b.sdk.hold.add('old-finish-return')

        def finish_then_hold_return():
            original_finish()
            b.sdk.checkpoint('old-finish-return')

        monkeypatch.setattr(old_owner, 'finish', finish_then_hold_return)
        old_stop = b.call(capture.stop)
        try:
            await reached(b.sdk.event('old-finish-return'))
            assert old_owner.retired
            assert old_stream.exited.is_set() and old_stream.worker._handle.is_done()
            assert old_stream.close_kwargs == [{'ignore_errors': False}]
            assert b.graph.control.snapshot()['pending_native_clients'] == 0
            assert not old_stop.done.is_set()

            fresh_start = b.call(capture.start)
            await finished(fresh_start)
            assert fresh_start.failure is None
            assert len(b.clients) == 2 and len(b.sdk.streams) == 2
            fresh_owner = capture._capture_owner
            fresh_stream = b.sdk.streams[1]
            assert fresh_owner is not old_owner
            assert capture._running and capture._streams == {'first': fresh_stream}
            assert not fresh_stream.worker._handle.is_done()
            assert b.clients[1].current()
            assert b.graph.control.snapshot()['pending_native_clients'] == 1

            b.sdk.release_event('old-finish-return').set()
            await finished(old_stop)
            assert old_stop.failure is None
            print({'scenario': 'old_stop_return_after_fresh_start',
                'old_native_handle_done': old_stream.worker._handle.is_done(),
                'fresh_owner_preserved': capture._capture_owner is fresh_owner,
                'fresh_running': capture._running,
                'fresh_stream_map_preserved': capture._streams == {'first': fresh_stream},
                'fresh_native_thread_alive': fresh_stream.worker.is_alive(),
                'fresh_native_handle_done': fresh_stream.worker._handle.is_done()})
            assert capture._capture_owner is fresh_owner
            assert capture._running, 'Old stop return cleared the fresh capture running state'
            assert capture._streams == {'first': fresh_stream}
            assert fresh_stream.worker.is_alive() and not fresh_stream.worker._handle.is_done()
        finally:
            b.sdk.release_event('old-finish-return').set()
            await finished(old_stop)
            await asyncio.to_thread(capture.stop)
            strict_retirement(b.sdk)
