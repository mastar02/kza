"""Public calibration bound to real capture policy over loopback HTTP.

Only device/model leaves are synthetic. The earlier unbound observations
remain archived under t023/d/baseline and do not define current permission.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
import threading

import pytest

from tests.integration.test_calibration_retirement import _Harness
from tests.integration.test_capture_browser_owners import browser_app
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_global_capture_control import actor, capture_phase, controlled
from tests.mocks.calibration_authority import serve_capture_app
from tools import acoustic_calibration as calibration
from tools.calibration_authority import CalibrationAuthorityClient


@asynccontextmanager
async def bound_capture(tmp_path, monkeypatch, dashboard_auth):
    async with controlled(tmp_path, monkeypatch) as graph:
        async with browser_app(graph.control, tmp_path) as app:
            async with serve_capture_app(app) as url:
                credential = dashboard_auth['Authorization'].split(' ', 1)[1]
                authority = CalibrationAuthorityClient(url, credential)
                with monkeypatch.context() as patch:
                    capture = _Harness(patch, tmp_path / 'measurements', authority=authority)
                    try:
                        yield graph, capture, authority
                    finally:
                        # Keep ASGI responsive to final HTTP retirement while
                        # joining the actual capture Thread.
                        await asyncio.to_thread(capture.release)
                        await asyncio.to_thread(authority.close)


@pytest.mark.asyncio
async def test_bound_public_capture_retires_before_scalar_success(tmp_path, monkeypatch, dashboard_auth):
    async with bound_capture(tmp_path, monkeypatch, dashboard_auth) as (graph, capture, authority):
        capture.emit_callback = True
        writes = []
        write = Path.write_text
        sdk_before = {name for name in sys.modules if name.startswith(('openwakeword', 'onnxruntime', 'usb.'))}

        def observe(path, text, *args, **kwargs):
            if path.parent == capture.out_dir and path.suffix == '.jsonl':
                assert capture.stream_close_called.is_set()
                assert capture.detector_closed.is_set() and capture.controller_invalidated.is_set()
                assert graph.control.snapshot()['pending_native_clients'] == 1
                writes.append(path)
            return write(path, text, *args, **kwargs)

        monkeypatch.setattr(Path, 'write_text', observe)
        caller = capture.start()
        await until(capture.done.is_set)
        await asyncio.to_thread(caller.join, 3)
        assert not caller.is_alive() and caller._handle.is_done()
        assert capture.failure is None and capture.callback_failure is None
        assert capture.result == writes[0] and len(writes) == 1 and capture.result.is_file()
        rows = capture.rows()
        assert rows[0]['meta'] is True and rows[0]['condition'] == 'silencio'
        assert [(row['kind'], row['value']) for row in rows[1:]] == [('rms', .25), ('wake', .8)]
        assert capture.predict_calls == 1
        assert capture.stream_kwargs['device'] == 17 and capture.stream_kwargs['channels'] == 2
        assert capture.stop_kwargs == capture.close_kwargs == [{'ignore_errors': False}]
        assert capture.controllers[0].usb_port == '5-5.4'
        assert graph.control.snapshot()['pending_native_clients'] == 0
        assert not authority.current()
        assert sdk_before == {name for name in sys.modules if name.startswith(('openwakeword', 'onnxruntime', 'usb.'))}


@pytest.mark.asyncio
async def test_global_mute_waits_for_real_usb_and_callback_exit(tmp_path, monkeypatch, dashboard_auth):
    async with bound_capture(tmp_path, monkeypatch, dashboard_auth) as (graph, capture, authority):
        capture.usb_available = capture.hold_usb = True
        capture.emit_callback = capture.hold_callback = True
        caller = capture.start(duration_s=30)
        try:
            await until(lambda: capture.usb_entered.is_set() and capture.predict_entered.is_set())
            assert graph.control.snapshot()['pending_native_clients'] == 1
            graph.control.request(True, graph.control.snapshot()['revision'], actor())
            await until(lambda: not authority.current())
            await until(capture.stream_close_called.is_set)
            assert caller.is_alive() and not capture.done.is_set()
            assert not capture.detector_closed.is_set()
            assert not list(capture.out_dir.glob('*.jsonl'))
            assert not graph.control.snapshot()['total_mute_confirmed']
            capture.usb_release.set()
            await until(capture.usb_exited.is_set)
            for poller in capture.pollers:
                await asyncio.to_thread(poller.join, 3)
                assert not poller.is_alive() and poller._handle.is_done()
            assert caller.is_alive() and not capture.done.is_set()
            assert not capture.detector_closed.is_set()
            assert graph.control.snapshot()['pending_native_clients'] == 1
            assert not graph.control.snapshot()['total_mute_confirmed']
        finally:
            capture.usb_release.set()
            capture.predict_release.set()
        await until(capture.done.is_set)
        await asyncio.to_thread(caller.join, 3)
        assert not caller.is_alive() and caller._handle.is_done()
        assert capture.failure is not None and capture.callback_failure is None
        assert capture.detector_closed.is_set() and capture.controller_invalidated.is_set()
        assert capture.result is None and not list(capture.out_dir.glob('*.jsonl'))
        state = await capture_phase(graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']


def test_analysis_of_historical_scalar_jsonl_never_acquires_live_capture(tmp_path, monkeypatch, capsys):
    directory = tmp_path / 'historical'
    directory.mkdir()
    rows = [{'meta': True, 'condition': 'silencio'}, {'kind': 'rms', 'value': .25}]
    (directory / 'synthetic_silencio.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))

    def forbidden(*args, **kwargs):
        raise AssertionError('analysis attempted live capture')

    monkeypatch.setattr(calibration, 'run_capture', forbidden)
    monkeypatch.delenv('KZA_DASHBOARD_TOKEN', raising=False)
    assert calibration.main(['--analyze', str(directory)]) == 0
    output = capsys.readouterr().out
    assert 'RMS' in output and 'silencio' in output and '.2500' in output


@pytest.mark.parametrize('authority', [None, object()])
def test_capture_requires_explicit_nominal_authority_before_sdk(tmp_path, monkeypatch, authority):
    capture = _Harness(monkeypatch, tmp_path / 'measurements', authority=object())
    try:
        with pytest.raises(RuntimeError, match='^calibration_authority_required$'):
            calibration.run_capture('silencio', 0, 17, 1, 'synthetic.onnx',
                capture.out_dir, authority=authority)
        assert not capture.events and not capture.detectors and not capture.controllers
        assert not capture.stream_kwargs and not list(capture.out_dir.glob('*.jsonl'))
    finally:
        capture.release()


@pytest.mark.asyncio
async def test_muted_backend_denies_capture_before_sdk_acquisition(tmp_path, monkeypatch, dashboard_auth):
    async with bound_capture(tmp_path, monkeypatch, dashboard_auth) as (graph, capture, authority):
        graph.control.request(True, graph.control.snapshot()['revision'], actor())
        await capture_phase(graph.control, 'muted')
        caller = capture.start()
        await until(capture.done.is_set)
        await asyncio.to_thread(caller.join, 3)
        assert caller._handle.is_done() and capture.failure is not None
        assert not capture.events and not capture.detectors and not capture.controllers
        assert not capture.stream_kwargs and not list(capture.out_dir.glob('*.jsonl'))
        assert not authority.current()
        state = graph.control.snapshot()
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']


@pytest.mark.asyncio
async def test_mute_during_stream_constructor_waits_for_return_and_strict_close(tmp_path, monkeypatch, dashboard_auth):
    async with bound_capture(tmp_path, monkeypatch, dashboard_auth) as (graph, capture, authority):
        entered, release = threading.Event(), threading.Event()
        stream_type = sys.modules['sounddevice'].InputStream

        class HeldStream(stream_type):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                entered.set()
                assert release.wait(5), 'constructor return gate not released'

        monkeypatch.setattr(sys.modules['sounddevice'], 'InputStream', HeldStream)
        caller = capture.start()
        try:
            await until(entered.is_set)
            assert graph.control.snapshot()['pending_native_clients'] == 1
            graph.control.request(True, graph.control.snapshot()['revision'], actor())
            await until(lambda: not authority.current())
            assert caller.is_alive() and not capture.done.is_set()
            assert 'stream-enter' not in capture.events
            assert not capture.detector_closed.is_set()
            assert not graph.control.snapshot()['total_mute_confirmed']
        finally:
            release.set()
        await until(capture.done.is_set)
        await asyncio.to_thread(caller.join, 3)
        assert caller._handle.is_done() and capture.failure is not None
        assert 'stream-enter' not in capture.events
        assert capture.stop_kwargs == capture.close_kwargs == [{'ignore_errors': False}]
        assert capture.detector_closed.is_set() and capture.controller_invalidated.is_set()
        assert not list(capture.out_dir.glob('*.jsonl'))
        state = await capture_phase(graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']


@pytest.mark.asyncio
async def test_scalar_write_return_is_owned_until_remote_retirement(tmp_path, monkeypatch, dashboard_auth, capsys):
    async with bound_capture(tmp_path, monkeypatch, dashboard_auth) as (graph, capture, authority):
        capture.emit_callback = True
        entered, release = threading.Event(), threading.Event()
        written = []
        write = Path.write_text

        def held_write(path, text, *args, **kwargs):
            result = write(path, text, *args, **kwargs)
            if path.parent == capture.out_dir and path.suffix == '.jsonl':
                written.append(path)
                entered.set()
                assert release.wait(5), 'scalar write return gate not released'
            return result

        monkeypatch.setattr(Path, 'write_text', held_write)
        caller = capture.start()
        try:
            await until(entered.is_set)
            assert capture.detector_closed.is_set() and capture.stream_close_called.is_set()
            assert graph.control.snapshot()['pending_native_clients'] == 1
            graph.control.request(True, graph.control.snapshot()['revision'], actor())
            await until(lambda: not authority.current())
            assert caller.is_alive() and not capture.done.is_set()
            assert not graph.control.snapshot()['total_mute_confirmed']
        finally:
            release.set()
        await until(capture.done.is_set)
        await asyncio.to_thread(caller.join, 3)
        assert caller._handle.is_done() and capture.failure is not None and capture.result is None
        # Already written scalar measurements are history; mute is not erasure.
        assert len(written) == 1 and written[0].is_file()
        rows = [json.loads(row) for row in written[0].read_text().splitlines()]
        assert [(row['kind'], row['value']) for row in rows[1:]] == [('rms', .25), ('wake', .8)]
        assert '✔' not in capsys.readouterr().out
        state = await capture_phase(graph.control, 'muted')
        assert state['pending_native_clients'] == 0 and state['total_mute_confirmed']


@pytest.mark.asyncio
async def test_constructor_without_returned_sdk_receipt_cannot_ack_remote_stop(tmp_path, monkeypatch, dashboard_auth):
    async with bound_capture(tmp_path, monkeypatch, dashboard_auth) as (graph, capture, authority):
        capture.faults.add('stream-construct')
        caller = capture.start()
        await until(capture.done.is_set)
        await asyncio.to_thread(caller.join, 3)
        assert caller._handle.is_done() and capture.failure is not None
        assert capture.detector_closed.is_set() and capture.controller_invalidated.is_set()
        assert capture.result is None and not list(capture.out_dir.glob('*.jsonl'))
        assert graph.control.snapshot()['pending_native_clients'] == 1
        graph.control.request(True, graph.control.snapshot()['revision'], actor())
        await until(lambda: graph.control.snapshot()['phase'] == 'muting')
        state = graph.control.snapshot()
        assert state['pending_native_clients'] == 1 and not state['total_mute_confirmed']
        await asyncio.to_thread(authority.close)
        state = graph.control.snapshot()
        assert state['pending_native_clients'] == 1 and not state['total_mute_confirmed']
