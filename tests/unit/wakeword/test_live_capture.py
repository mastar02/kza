"""Real live-capture owner/authority with only SDK and HTTP leaves replaced."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import weakref

import numpy as np
import pytest

from src.audio import native_capture_authority as authority_module
from src.audio.native_capture_authority import WakewordTrainingAuthorityClient
from src.wakeword.live_capture import CapturePolicyStopped, LiveWakewordCapture
from tests.unit.audio.test_native_capture_authority_client import NativeProtocol
from tests.unit.tools.test_calibration_authority_client import ADMIN_TOKEN, URL


class Stream:
    def __init__(self, *, stop_error=None, close_error=None, held=False):
        self.stop_error, self.close_error = stop_error, close_error
        self.starts = self.stops = self.closes = 0
        self.close_entered = threading.Event()
        self.close_release = threading.Event()
        if not held:
            self.close_release.set()
        self.closed = False

    def start(self):
        self.starts += 1

    def stop(self, *, ignore_errors):
        assert ignore_errors is False
        self.stops += 1
        if self.stop_error is not None:
            raise self.stop_error

    def close(self, *, ignore_errors):
        assert ignore_errors is False
        self.closes += 1
        self.close_entered.set()
        assert self.close_release.wait(3), 'fixture stream close not released'
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class Detector:
    def __init__(self):
        self.loads = self.closes = 0

    def load(self):
        self.loads += 1

    def close(self):
        self.closes += 1


def capture():
    protocol = NativeProtocol()
    authority = WakewordTrainingAuthorityClient(URL, ADMIN_TOKEN, poll_interval_s=.01,
                                               session_factory=protocol.session)
    return LiveWakewordCapture(authority), authority, protocol


def test_strict_stream_close_failure_cannot_ack_or_retry_close():
    owner, authority, protocol = capture()
    failure = RuntimeError('synthetic strict stream close failed')
    stream, detector = Stream(close_error=failure), Detector()
    with pytest.raises(RuntimeError) as caught:
        with owner:
            owner.own_detector(lambda: detector)
            owner.open_stream(lambda: stream)
            owner.close_stream()
    assert caught.value is failure
    assert stream.starts == stream.stops == stream.closes == 1
    assert detector.closes == 1 and authority.current() is False
    assert 'retire' not in protocol.calls
    with pytest.raises(RuntimeError, match='retirement_already_attempted'):
        owner.__exit__(None, None, None)
    assert stream.closes == 1 and 'retire' not in protocol.calls


@pytest.mark.parametrize('lane', ['callback', 'output'])
def test_self_drain_rejects_before_native_close_and_outer_owner_can_finish(lane):
    owner, authority, protocol = capture()
    stream, detector = Stream(), Detector()
    owner.__enter__()
    monitor = authority._thread
    failures = []

    def self_drain():
        try:
            owner.__exit__(None, None, None)
        except RuntimeError as error:
            failures.append(str(error))
        assert stream.stops == stream.closes == detector.closes == 0

    try:
        owner.own_detector(lambda: detector)
        owner.open_stream(lambda: stream)
        if lane == 'callback':
            owner.callback(self_drain)()
        else:
            with pytest.raises(CapturePolicyStopped):
                owner.publish(self_drain)
        assert failures == ['wakeword_capture_self_drain']
        assert 'retire' not in protocol.calls
    finally:
        owner.__exit__(None, None, None)
    assert stream.closed and stream.closes == detector.closes == 1
    assert monitor._handle.is_done()
    assert protocol.calls.count('retire') == 1


def test_stop_control_still_waits_real_close_return_detector_and_monitor():
    class StopControl(BaseException):
        pass

    control = StopControl('synthetic stop control')
    owner, authority, protocol = capture()
    stream, detector = Stream(stop_error=control, held=True), Detector()
    owner.__enter__()
    monitor = authority._thread
    owner.own_detector(lambda: detector)
    owner.open_stream(lambda: stream)
    with ThreadPoolExecutor(max_workers=1) as fixture:
        finishing = fixture.submit(owner.__exit__, None, None, None)
        try:
            assert stream.close_entered.wait(3)
            assert not finishing.done()
            assert stream.closed is False and detector.closes == 0
            assert 'retire' not in protocol.calls
            stream.close_release.set()
            with pytest.raises(StopControl) as caught:
                finishing.result(3)
            assert caught.value is control
        finally:
            stream.close_release.set()
    assert stream.closed and stream.stops == stream.closes == detector.closes == 1
    assert monitor._handle.is_done() and authority.current() is False
    assert protocol.calls.count('retire') == 1
    assert protocol.close_entered.is_set()


def test_unconfirmed_actual_monitor_start_prevents_remote_retirement(monkeypatch):
    owner, authority, protocol = capture()
    original = threading.Thread
    created = []

    class UnconfirmedStart(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def start(self):
            raise RuntimeError('synthetic start without terminal receipt')

    monkeypatch.setattr(authority_module.threading, 'Thread', UnconfirmedStart)
    with pytest.raises(BaseExceptionGroup) as caught:
        owner.__enter__()
    assert 'wakeword_capture_cleanup_failed' in str(caught.value)
    assert len(created) == 1
    assert created[0]._handle.ident == 0 and not created[0]._handle.is_done()
    assert protocol.calls == ['register', 'activate']
    assert 'retire' not in protocol.calls and not protocol.close_entered.is_set()
    assert authority.current() is False
    # A never-confirmed actual handle is retained, not declared an OS actor leak.


@pytest.mark.parametrize('kind', ['ordinary', 'keyboard', 'system_exit'])
def test_quiescent_callback_failure_retains_no_pcm_trace_and_sanitizes_controls(kind):
    owner, authority, protocol = capture()
    owner.__enter__()
    stream = Stream()
    owner.open_stream(lambda: stream)

    def failing(audio):
        assert audio.shape == (128,)
        if kind == 'ordinary':
            raise RuntimeError('private PCM canary')
        if kind == 'keyboard':
            raise KeyboardInterrupt('private PCM canary')
        raise SystemExit('private PCM canary')

    received = owner.callback(failing)
    audio = np.full(128, .25, dtype=np.float32)
    payload = weakref.ref(audio)
    received(audio)
    del audio
    expected = {'ordinary': RuntimeError, 'keyboard': KeyboardInterrupt, 'system_exit': SystemExit}[kind]
    with pytest.raises(expected) as caught:
        owner.__exit__(None, None, None)
    gc.collect()
    assert payload() is None
    assert 'private PCM canary' not in str(caught.value)
    if kind == 'ordinary':
        assert str(caught.value) == 'wakeword_capture_callback_failed'
    elif kind == 'keyboard':
        assert caught.value.args == ()
    else:
        assert caught.value.code == 1
    assert owner._callback_failures == []
    assert stream.closed and protocol.calls.count('retire') == 1
    assert authority.current() is False
