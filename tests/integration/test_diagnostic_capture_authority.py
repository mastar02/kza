"""Real diagnostic CLI/HTTP policy, with synthetic SDK leaves only."""
from __future__ import annotations

import asyncio
import builtins
from contextlib import asynccontextmanager
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.audio.native_capture_authority import WakewordDiagnosticsAuthorityClient
from src.dashboard.authority import DashboardRole, EvaluationModality, EvaluationSample
from tests.integration.test_capture_browser_owners import browser_app
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase, controlled
from tests.mocks.calibration_authority import serve_capture_app
from tests.mocks.wakeword_capture import CaptureCall
from tests.mocks.wakeword_diagnostics import DiagnosticLeaves


SCRIPT = Path('scripts/diagnose_wakeword.py')


@asynccontextmanager
async def diagnostic_graph(tmp_path, monkeypatch, dashboard_auth):
    async with controlled(tmp_path, monkeypatch) as graph:
        async with browser_app(graph.control, tmp_path) as app:
            async with serve_capture_app(app) as url:
                with monkeypatch.context() as patch:
                    sdk = DiagnosticLeaves(patch)
                    clients, calls = [], []
                    initialize = WakewordDiagnosticsAuthorityClient.__init__

                    def observe(client, *args, **kwargs):
                        initialize(client, *args, **kwargs)
                        clients.append(client)

                    patch.setattr(WakewordDiagnosticsAuthorityClient, '__init__', observe)
                    patch.setenv('KZA_DIAGNOSTIC_TEST_OPERATOR', dashboard_auth['Authorization'].split(' ', 1)[1])

                    def start():
                        main = runpy.run_path(str(SCRIPT))['main']
                        call = CaptureCall(lambda: main(['--dashboard-url', url,
                            '--token-env', 'KZA_DIAGNOSTIC_TEST_OPERATOR']))
                        calls.append(call)
                        return call

                    try:
                        yield SimpleNamespace(graph=graph, app=app, sdk=sdk, clients=clients, start=start)
                    finally:
                        sdk.release()
                        for call in calls:
                            await asyncio.to_thread(call.join)
                        for client in clients:
                            await asyncio.to_thread(client.close)


async def reached(event):
    # Containment includes the real3s countdown, not a performance assertion.
    assert await asyncio.to_thread(event.wait, 8), 'diagnostic leaf never entered'


async def finished(call):
    assert await asyncio.to_thread(call.done.wait, 10), 'diagnostic invocation did not return'
    await asyncio.to_thread(call.join)
    assert call.worker._handle.is_done()


def strict_input(sdk):
    assert len(sdk.streams) == 1
    stream = sdk.streams[0]
    assert {key: stream.kwargs[key] for key in ['samplerate', 'channels', 'device', 'dtype']} == {
        'samplerate': 16000, 'channels': 6, 'device': 9, 'dtype': 'float32'}
    assert stream.stop_kwargs == stream.close_kwargs == [{'ignore_errors': False}]
    assert stream.close_returned


def test_main_requires_explicit_locator_before_sdk_imports(monkeypatch):
    imports = []
    original = builtins.__import__

    def no_sdk(name, *args, **kwargs):
        if name.split('.', 1)[0] in {'sounddevice', 'openwakeword', 'onnxruntime'}:
            imports.append(name)
            raise AssertionError('denied invocation attempted SDK import')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_sdk)
    main = runpy.run_path(str(SCRIPT))['main']
    with pytest.raises(SystemExit) as failure:
        main([])
    assert failure.value.code == 2 and not imports


@pytest.mark.parametrize('role', [DashboardRole.HOME, DashboardRole.EVALUATOR])
@pytest.mark.asyncio
async def test_real_diagnostic_cli_cannot_infer_admin_from_household_or_evaluator(tmp_path, monkeypatch, dashboard_auth, role):
    async with diagnostic_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        now = time.time()
        samples = (EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}),
            frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 60),) if role is DashboardRole.EVALUATOR else ()
        authority = b.app.state.dashboard_authority
        grant = await authority.issue_credential(authority.resolve(dashboard_auth['Authorization']),
            subject='synthetic-diagnostic-member', role=role, expires_at=now + 120, samples=samples)
        monkeypatch.setenv('KZA_DIAGNOSTIC_TEST_OPERATOR', grant.token)
        call = b.start()
        await finished(call)
        assert call.failure is not None
        assert not b.sdk.events and not b.sdk.streams and not b.sdk.models
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


@pytest.mark.parametrize('boundary', ['start', 'read', 'predict', 'report', 'close', 'model-release'])
@pytest.mark.asyncio
async def test_global_ack_waits_for_actual_diagnostic_native_model_and_report_return(tmp_path, monkeypatch, dashboard_auth, boundary):
    async with diagnostic_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.hold.add(boundary)
        stdout = sys.stdout

        class ReportSink:
            def write(self, value):
                result = stdout.write(value)
                if 'chunks:' in value:
                    b.sdk.checkpoint('report')
                return result

            def __getattr__(self, name):
                return getattr(stdout, name)

        monkeypatch.setattr(sys, 'stdout', ReportSink())
        call = b.start()
        try:
            await reached(b.sdk.event(boundary))
            assert b.graph.control.snapshot()['pending_native_clients'] == 1
            b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
            await until(lambda: bool(b.clients) and not b.clients[0].current())
            assert call.worker.is_alive() and not call.done.is_set()
            assert not b.graph.control.snapshot()['total_mute_confirmed']
            if boundary in {'start', 'read', 'close'}:
                assert not b.sdk.models
        finally:
            b.sdk.release_event(boundary).set()
        await finished(call)
        strict_input(b.sdk)
        state = await capture_phase(b.graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']
        assert all(ref() is None for ref in b.sdk.model_refs)
        if boundary == 'read':
            before = (list(b.sdk.read_frames), len(b.sdk.models), len(b.clients))
            b.graph.control.request(False, state['revision'], actor())
            await capture_phase(b.graph.control, 'running')
            assert call.worker._handle.is_done()
            assert (b.sdk.read_frames, len(b.sdk.models), len(b.clients)) == before


@pytest.mark.parametrize('constructor', ['stream-construct', 'model-construct'])
@pytest.mark.asyncio
async def test_unreturned_diagnostic_constructor_cannot_ack_stop(tmp_path, monkeypatch, dashboard_auth, constructor):
    async with diagnostic_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.faults.add(constructor)
        call = b.start()
        await finished(call)
        assert call.failure is not None
        assert b.sdk.event(constructor).is_set()
        assert b.graph.control.snapshot()['pending_native_clients'] == 1
        await asyncio.to_thread(b.clients[0].close)
        b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
        await capture_phase(b.graph.control, 'muting')
        state = b.graph.control.snapshot()
        assert state['pending_native_clients'] == 1 and not state['total_mute_confirmed']


@pytest.mark.asyncio
async def test_original_operator_revocation_stops_diagnostic_read_without_new_actor(tmp_path, monkeypatch, dashboard_auth):
    async with diagnostic_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.sdk.hold.add('read')
        call = b.start()
        try:
            await reached(b.sdk.event('read'))
            assert b.graph.control.snapshot()['pending_native_clients'] == 1
            monkeypatch.setenv('KZA_DASHBOARD_TOKEN', 'different-synthetic-operator')
            await until(lambda: bool(b.clients) and not b.clients[0].current())
            assert call.worker.is_alive() and not call.done.is_set()
        finally:
            b.sdk.release_event('read').set()
        await finished(call)
        strict_input(b.sdk)
        assert call.failure is not None and not b.sdk.models
        assert len(b.clients) == 1 and not b.clients[0].current()
        assert b.graph.control.snapshot()['pending_native_clients'] == 0


def test_historical_helpers_use_supplied_arrays_without_live_capture(monkeypatch, capsys):
    sdk = DiagnosticLeaves(monkeypatch)

    def no_authority(*args, **kwargs):
        raise AssertionError('historical helper attempted fresh capture authority')

    monkeypatch.setattr(WakewordDiagnosticsAuthorityClient, '__init__', no_authority)
    script = runpy.run_path(str(SCRIPT))
    audio = np.tile(sdk.values, (2560, 1))
    rows = []
    assert script['analyze_channels'](audio, emit=rows.append) is None
    assert any('ch2' in row and 'RMS=0.2500' in row for row in rows)
    assert script['run_openwakeword'](audio[:, 0].copy(), 'historical', capture=None) is None
    assert 'chunks: 2' in capsys.readouterr().out
    assert not sdk.streams and not sdk.read_frames
    assert len(sdk.models) == 1 and sdk.models[0]['calls'] == 2
    assert all(ref() is None for ref in sdk.model_refs)


@pytest.mark.asyncio
async def test_muted_real_backend_denies_diagnostic_before_input_or_model(tmp_path, monkeypatch, dashboard_auth):
    async with diagnostic_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        b.graph.control.request(True, b.graph.control.snapshot()['revision'], actor())
        await capture_phase(b.graph.control, 'muted')
        call = b.start()
        await finished(call)
        assert call.failure is not None and not b.sdk.events
        state = b.graph.control.snapshot()
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']


@pytest.mark.parametrize('best_channel', [0, 2])
@pytest.mark.asyncio
async def test_real_main_preserves_six_channel_math_raw_models_and_reports(tmp_path, monkeypatch, dashboard_auth, capsys, best_channel):
    async with diagnostic_graph(tmp_path, monkeypatch, dashboard_auth) as b:
        if best_channel == 0:
            b.sdk.values[0] = .3
        call = b.start()
        await finished(call)
        assert call.failure is None
        strict_input(b.sdk)
        assert sum(b.sdk.read_frames) == 64000 and all(0 < frames <= 1280 for frames in b.sdk.read_frames)
        count = 2 if best_channel == 0 else 3
        assert len(b.sdk.models) == count and all(row['calls'] == 50 and row['released'] for row in b.sdk.models)
        expected = [b.sdk.values[0], (np.array([b.sdk.values[0]], dtype=np.float32) * 32767)[0]]
        if best_channel != 0:
            expected.append((np.array([b.sdk.values[best_channel]], dtype=np.float32) * 32767)[0])
        assert [row['first_values'][0] for row in b.sdk.models] == [float(value) for value in expected]
        assert all(ref() is None for ref in b.sdk.model_refs + b.sdk.pcm_refs)
        output = capsys.readouterr().out
        assert f'Canal más activo: ch{best_channel}' in output
        assert output.count('chunks: 50  max: 0.5000  mean: 0.1240') == count
        for block in output.split('chunks: 50  max: 0.5000  mean: 0.1240')[1:]:
            top = [line.strip() for line in block.splitlines() if line.strip().startswith('t=')]
            assert top[:5] == ['t=80ms  score=0.5000', 't=160ms  score=0.5000',
                't=320ms  score=0.4000', 't=240ms  score=0.2000', 't=0ms  score=0.1000']
        assert '=== Interpretación ===' in output
        assert len(b.clients) == 1 and not b.clients[0].current()
        assert b.graph.control.snapshot()['pending_native_clients'] == 0
