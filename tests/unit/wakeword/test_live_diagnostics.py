"""Diagnostic ownership uses real control clients with synthetic native leaves."""
from __future__ import annotations

import gc
import threading
import weakref

import numpy as np
import pytest

from src.audio.native_capture_authority import CalibrationAuthorityClient
from src.wakeword.live_capture import LiveWakewordDiagnosticsCapture
from tests.unit.audio.test_native_capture_authority_client import (
    NativeProtocol, client, diagnostics_client,
)
from tests.unit.tools.test_calibration_authority_client import ADMIN_TOKEN, Protocol, URL
from tests.unit.wakeword.test_live_capture import Stream


def capture():
    protocol = NativeProtocol()
    authority = diagnostics_client(protocol)
    return LiveWakewordDiagnosticsCapture(authority), authority, protocol


def finish_in_thread(owner):
    done = threading.Event()
    failures = []

    def finish():
        try:
            owner.__exit__(None, None, None)
        except BaseException as error:
            failures.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=finish, name='diagnostics-fixture-finish')
    handle = thread._handle
    thread.start()
    return handle, done, failures


def test_diagnostics_rejects_foreign_training_and_calibration_before_registration():
    training_protocol, calibration_protocol = NativeProtocol(), Protocol()
    training = client(training_protocol)
    calibration = CalibrationAuthorityClient(
        URL, ADMIN_TOKEN, session_factory=calibration_protocol.session)
    try:
        for foreign in (object(), training, calibration):
            with pytest.raises(RuntimeError, match='^wakeword_capture_authority_required$'):
                LiveWakewordDiagnosticsCapture(foreign)
        assert training_protocol.calls == calibration_protocol.calls == []
    finally:
        training.close()
        calibration.close()


def test_unreturned_model_factory_preserves_error_and_cannot_ack():
    owner, authority, protocol = capture()
    failure = RuntimeError('synthetic constructor uncertainty')
    constructions = []

    def unreturned():
        constructions.append(True)
        raise failure

    with pytest.raises(RuntimeError) as caught:
        with owner:
            monitor_handle = authority._thread._handle
            owner.own_python_model(unreturned)
    assert caught.value is failure and constructions == [True]
    assert 'retire' not in protocol.calls
    assert protocol.close_entered.is_set()
    assert monitor_handle.is_done()
    assert authority.current() is False


def test_python_model_reference_release_waits_real_destructor_return_before_ack():
    owner, authority, protocol = capture()
    release_entered, release_allowed, release_returned = (
        threading.Event(), threading.Event(), threading.Event())
    timed_out = []
    references = []

    class PythonModel:
        def __del__(self):
            release_entered.set()
            if not release_allowed.wait(3):
                timed_out.append(True)
            release_returned.set()

    def factory():
        model = PythonModel()
        references.append(weakref.ref(model))
        return model

    owner.__enter__()
    monitor_handle = authority._thread._handle
    owner.own_python_model(factory)
    handle, done, failures = finish_in_thread(owner)
    try:
        assert release_entered.wait(3)
        assert not release_returned.is_set() and not done.is_set()
        assert 'retire' not in protocol.calls
        assert not protocol.close_entered.is_set()
    finally:
        release_allowed.set()
        handle.join(5)
        assert handle.is_done(), 'fixture finish native exit unconfirmed'
    assert done.is_set() and failures == [] and timed_out == []
    assert release_returned.is_set() and references[0]() is None
    assert monitor_handle.is_done()
    assert protocol.calls.count('retire') == 1


@pytest.mark.parametrize('kind', ['ordinary', 'keyboard', 'system_exit'])
def test_model_failure_descriptor_releases_pcm_and_model_without_native_trace(kind):
    owner, authority, protocol = capture()

    class PythonModel:
        def predict(self, audio):
            assert audio.shape == (128,)
            if kind == 'ordinary':
                raise ValueError('private diagnostic PCM canary')
            if kind == 'keyboard':
                raise KeyboardInterrupt('private diagnostic PCM canary')
            raise SystemExit(17)

    def predict_and_capture():
        model = owner.own_python_model(PythonModel)
        audio = np.full(128, .25, dtype=np.float32)
        pcm_reference, model_reference = weakref.ref(audio), weakref.ref(model)
        try:
            model.predict(audio)
        except BaseException as error:
            descriptor = owner.model_failure(error)
        return descriptor, pcm_reference, model_reference

    with owner:
        monitor_handle = authority._thread._handle
        descriptor, pcm_reference, model_reference = predict_and_capture()
    gc.collect()
    assert pcm_reference() is None and model_reference() is None
    assert descriptor.__traceback__ is descriptor.__context__ is descriptor.__cause__ is None
    if kind == 'ordinary':
        assert type(descriptor) is RuntimeError
        assert str(descriptor) == 'wakeword_diagnostic_model_failed'
    elif kind == 'keyboard':
        assert type(descriptor) is KeyboardInterrupt and descriptor.args == ()
    else:
        assert type(descriptor) is SystemExit and descriptor.code == 17
        assert owner.model_failure(SystemExit('private diagnostic PCM canary')).code == 1
    assert protocol.calls.count('retire') == 1
    assert monitor_handle.is_done()


def test_diagnostics_inherited_strict_stream_close_failure_cannot_ack_or_retry():
    owner, authority, protocol = capture()
    failure = RuntimeError('synthetic strict stream close failure')
    stream = Stream(close_error=failure)
    with pytest.raises(RuntimeError) as caught:
        with owner:
            monitor_handle = authority._thread._handle
            owner.open_stream(lambda: stream)
            owner.close_stream()
    assert caught.value is failure
    assert stream.starts == stream.stops == stream.closes == 1
    assert 'retire' not in protocol.calls
    with pytest.raises(RuntimeError, match='retirement_already_attempted'):
        owner.__exit__(None, None, None)
    assert stream.closes == 1 and 'retire' not in protocol.calls
    assert monitor_handle.is_done()


def test_diagnostics_ack_waits_for_actual_monitor_http_return_and_native_join():
    owner, authority, protocol = capture()
    protocol.poll_release.clear()
    stopping = threading.Event()
    original_stop = authority.stop_monitor

    def observed_stop():
        stopping.set()
        return original_stop()

    authority.stop_monitor = observed_stop
    owner.__enter__()
    handle = None
    try:
        assert protocol.poll_entered.wait(3)
        monitor_handle = authority._thread._handle
        handle, done, failures = finish_in_thread(owner)
        assert stopping.wait(3)
        assert not monitor_handle.is_done() and not done.is_set()
        assert 'retire' not in protocol.calls
        assert not protocol.close_entered.is_set()
    finally:
        protocol.poll_release.set()
        if handle is not None:
            handle.join(5)
            assert handle.is_done(), 'fixture finish native exit unconfirmed'
        else:
            owner.__exit__(None, None, None)
    assert done.is_set() and failures == [] and monitor_handle.is_done()
    assert protocol.calls.count('retire') == 1
    assert protocol.close_entered.is_set()
