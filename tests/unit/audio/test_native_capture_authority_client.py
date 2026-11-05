"""Closed native namespaces reuse the reviewed calibration protocol owner."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import requests

from src.audio import native_capture_authority as common
from src.audio.native_capture_authority import (
    CalibrationAuthorityClient, NativeCaptureAuthorityClient,
    WakewordDiagnosticsAuthorityClient, WakewordTrainingAuthorityClient,
)
from tests.unit.tools.test_calibration_authority_client import (
    ADMIN_TOKEN, CAPABILITY, Protocol, URL,
)
from tools import calibration_authority as legacy


class NativeProtocol(Protocol):
    """Actual Session with only the fixed-namespace HTTP leaf replaced."""

    def __init__(self):
        super().__init__()
        self.reply_kind = 'wakeword_training'
        self.namespace = 'wakeword-training'
        self.wrong_kind = 'calibration'
        self.wrong_action = None

    def send(self, request, **kwargs):
        assert request.url.startswith(URL + '/api/capture/native/' + self.namespace + '/')
        assert request.headers['Origin'] == URL
        action = request.url.rsplit('/', 1)[-1]
        body = json.loads(request.body or b'{}')
        assert body == ({} if action == 'register' else {'token': CAPABILITY})
        if action == 'retire':
            assert request.headers.get('Authorization') is None
        else:
            assert request.headers['Authorization'] == 'Bearer ' + ADMIN_TOKEN
        self.calls.append(action)
        if action == 'poll':
            self.poll_entered.set()
            assert self.poll_release.wait(3)
        kind = self.wrong_kind if action == self.wrong_action else self.reply_kind
        payload = ({'token': CAPABILITY, 'kind': kind, 'state': 'reserved',
                    'capture_allowed': False, 'generation': 1} if action == 'register'
                   else {'retired': True} if action == 'retire'
                   else {'kind': kind, 'state': 'active', 'capture_allowed': self.allowed,
                         'retirement_required': not self.allowed, 'generation': 1})
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response._content = json.dumps(payload).encode()
        return response


def client(protocol):
    return WakewordTrainingAuthorityClient(URL, ADMIN_TOKEN, poll_interval_s=.01,
                                           session_factory=protocol.session)


def diagnostics_client(protocol):
    protocol.namespace = 'wakeword-diagnostics'
    protocol.reply_kind = 'wakeword_diagnostics'
    return WakewordDiagnosticsAuthorityClient(URL, ADMIN_TOKEN, poll_interval_s=.01,
                                              session_factory=protocol.session)


def test_wakeword_fixed_namespace_original_actor_and_token_only_retirement():
    protocol = NativeProtocol()
    authority = client(protocol)
    stopped = threading.Event()
    try:
        authority.register()
        assert authority.current() is False
        authority.activate()
        assert authority.current() is True
        protocol.allowed = False
        authority.start_monitor(stopped.set)
        assert stopped.wait(3)
        authority.stop_monitor()
        assert authority.current() is False
        authority.retire()
        assert protocol.calls[:2] == ['register', 'activate']
        assert protocol.calls[-1] == 'retire'
        assert all(session.trust_env is False for session in protocol.sessions)
    finally:
        authority.close()


def test_diagnostics_fixed_namespace_original_actor_and_token_only_retirement():
    protocol = NativeProtocol()
    authority = diagnostics_client(protocol)
    stopped = threading.Event()
    assert type(authority) is WakewordDiagnosticsAuthorityClient
    assert not isinstance(authority, (CalibrationAuthorityClient, WakewordTrainingAuthorityClient))
    try:
        authority.register()
        assert authority.current() is False
        authority.activate()
        assert authority.current() is True
        protocol.allowed = False
        authority.start_monitor(stopped.set)
        assert stopped.wait(3)
        authority.stop_monitor()
        assert authority.current() is False
        authority.retire()
        assert protocol.calls[:2] == ['register', 'activate']
        assert 'poll' in protocol.calls
        assert protocol.calls[-1] == 'retire'
        assert all(session.trust_env is False for session in protocol.sessions)
    finally:
        authority.close()


@pytest.mark.parametrize('wrong_kind', ['calibration', 'wakeword_training'])
@pytest.mark.parametrize('action', ['register', 'activate'])
def test_other_namespace_response_cannot_grant_diagnostics_capture(action, wrong_kind):
    protocol = NativeProtocol()
    authority = diagnostics_client(protocol)
    protocol.wrong_kind = wrong_kind
    protocol.wrong_action = action
    try:
        if action == 'activate':
            authority.register()
        with pytest.raises(RuntimeError) as caught:
            getattr(authority, action)()
        assert ADMIN_TOKEN not in str(caught.value)
        assert CAPABILITY not in str(caught.value)
        assert URL not in str(caught.value)
        assert authority.current() is False
        assert 'poll' not in protocol.calls
    finally:
        authority.close()
    assert 'retire' not in protocol.calls


@pytest.mark.parametrize('kind', ['calibration', 'wakeword_training'])
def test_diagnostics_fixed_wrapper_cannot_accept_another_kind(kind):
    with pytest.raises(TypeError):
        WakewordDiagnosticsAuthorityClient(URL, ADMIN_TOKEN, kind=kind)


@pytest.mark.parametrize('action', ['register', 'activate'])
def test_calibration_response_cannot_grant_wakeword_capture(action):
    protocol = NativeProtocol()
    protocol.wrong_action = action
    authority = client(protocol)
    try:
        if action == 'activate':
            authority.register()
        with pytest.raises(RuntimeError) as caught:
            getattr(authority, action)()
        assert ADMIN_TOKEN not in str(caught.value)
        assert URL not in str(caught.value)
        assert authority.current() is False
        assert 'poll' not in protocol.calls
    finally:
        authority.close()
    assert 'retire' not in protocol.calls


@pytest.mark.parametrize('kind', ['wakeword-training', '../calibration', object()])
def test_common_kind_is_closed_before_transport_acquisition(kind):
    acquired = []

    def forbidden():
        acquired.append(True)
        raise AssertionError('invalid kind acquired transport')

    with pytest.raises(ValueError, match='native_capture_authority_kind_invalid'):
        NativeCaptureAuthorityClient(URL, ADMIN_TOKEN, kind=kind, session_factory=forbidden)
    assert acquired == []


@pytest.mark.parametrize('wrapper,kind', [
    (WakewordTrainingAuthorityClient, 'calibration'),
    (CalibrationAuthorityClient, 'wakeword_training'),
])
def test_fixed_wrapper_cannot_accept_another_kind(wrapper, kind):
    with pytest.raises(TypeError):
        wrapper(URL, ADMIN_TOKEN, kind=kind)


def test_legacy_mutable_hooks_and_retained_registry_are_actual_shared_objects(monkeypatch):
    assert legacy.CalibrationAuthorityClient is CalibrationAuthorityClient
    assert legacy.requests is common.requests
    assert legacy.threading is common.threading
    assert legacy._UNCONFIRMED is common._UNCONFIRMED
    assert legacy._UNCONFIRMED_LOCK is common._UNCONFIRMED_LOCK
    original = threading.Thread
    created = []

    class ObservedThread(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    # Patch the historical public module hook and observe actual common dispatch.
    monkeypatch.setattr(legacy.threading, 'Thread', ObservedThread)
    protocol = NativeProtocol()
    protocol.allowed = True
    authority = client(protocol)
    try:
        authority.register()
        authority.activate()
        authority.start_monitor(lambda: None)
        assert protocol.poll_entered.wait(3)
        authority.stop_monitor()
        assert len(created) == 1 and type(created[0]) is ObservedThread
        assert created[0]._handle.is_done()
        authority.retire()
    finally:
        authority.close()


def test_authority_import_does_not_load_audio_or_model_sdk():
    script = """
import sys
from tools.calibration_authority import CalibrationAuthorityClient
from src.audio.native_capture_authority import WakewordTrainingAuthorityClient
assert not any(name in sys.modules for name in ('sounddevice','onnxruntime','kokoro','speechbrain','openwakeword'))
print('NATIVE_AUTHORITY_IMPORT_COMPLETE')
"""
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, '-c', script], cwd=root,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('NATIVE_AUTHORITY_IMPORT_COMPLETE') == 1
