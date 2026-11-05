"""Calibration authority over actual requests Sessions and real monitor Threads."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from requests.adapters import BaseAdapter

from tools.calibration_authority import CalibrationAuthorityClient


URL = 'http://127.0.0.1:9876'
ADMIN_TOKEN = 'synthetic-calibration-operator-secret'
CAPABILITY = 'c' * 43
PREFIX = '/api/capture/native/calibration/'


class Protocol(BaseAdapter):
    """Only the HTTP leaf is synthetic; Session/client/thread ownership is real."""

    def __init__(self):
        self.calls = []
        self.sessions = []
        self.poll_entered = threading.Event()
        self.poll_release = threading.Event()
        self.poll_release.set()
        self.close_entered = threading.Event()
        self.close_release = threading.Event()
        self.close_release.set()
        self.activate_entered = threading.Event()
        self.activate_release = threading.Event()
        self.activate_release.set()
        self.allowed = True
        self.failure = None
        self.close_failure = False
        self.close_calls = 0
        self.hook = None

    def session(self):
        session = requests.Session()
        for adapter in session.adapters.values():
            adapter.close()
        session.mount('http://', self)
        session.mount('https://', self)
        self.sessions.append(session)
        return session

    def send(self, request, **kwargs):
        assert request.url.startswith(URL + PREFIX)
        assert request.headers.get('Origin') == URL
        action = request.url.rsplit('/', 1)[-1]
        body = json.loads(request.body or b'{}')
        if action == 'retire':
            assert request.headers.get('Authorization') is None
            assert body == {'token': CAPABILITY}
        else:
            assert request.headers.get('Authorization') == 'Bearer ' + ADMIN_TOKEN
            assert body == ({} if action == 'register' else {'token': CAPABILITY})
        self.calls.append(action)
        if self.hook is not None:
            self.hook(action)
        if action == 'activate':
            self.activate_entered.set()
            assert self.activate_release.wait(4), 'fixture activation was not released'
        if action == 'poll':
            self.poll_entered.set()
            assert self.poll_release.wait(4), 'fixture poll was not released'
            if self.failure is not None:
                raise self.failure
        payload = ({'token': CAPABILITY, 'kind': 'calibration', 'state': 'reserved',
                    'capture_allowed': False, 'generation': 1} if action == 'register'
                   else {'retired': True} if action == 'retire'
                   else {'kind': 'calibration', 'state': 'active',
                         'capture_allowed': self.allowed, 'retirement_required': not self.allowed,
                         'generation': 1})
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response.headers['Content-Type'] = 'application/json'
        response._content = json.dumps(payload).encode()
        return response

    def close(self):
        self.close_calls += 1
        self.close_entered.set()
        assert self.close_release.wait(4), 'fixture transport close was not released'
        if self.close_failure:
            raise RuntimeError('synthetic transport close failed ' + ADMIN_TOKEN)


def client(protocol):
    return CalibrationAuthorityClient(URL, ADMIN_TOKEN, timeout_s=.2,
                                      poll_interval_s=.01, session_factory=protocol.session)


def stopped_admission(authority):
    deadline = time.monotonic() + 3
    while authority.current() is True:
        if time.monotonic() >= deadline:
            raise AssertionError('actual close never sealed admission')
        time.sleep(.001)


def test_original_bearer_reservation_activation_and_explicit_capability_retirement():
    protocol = Protocol()
    authority = client(protocol)
    try:
        assert authority.current() is False
        authority.register()
        assert authority.current() is False
        authority.activate()
        assert authority.current() is True
        assert protocol.calls == ['register', 'activate']
        assert all(session.trust_env is False for session in protocol.sessions)
        authority.stop_monitor()
        authority.retire()
        assert protocol.calls[-1] == 'retire'
        assert authority.current() is False
    finally:
        authority.close()


def test_close_without_explicit_retire_never_acknowledges_remote_capture():
    protocol = Protocol()
    authority = client(protocol)
    authority.register()
    authority.activate()
    authority.close()
    assert 'retire' not in protocol.calls
    assert protocol.close_entered.is_set()
    assert authority.current() is False


def test_revoked_poll_seals_once_and_monitor_is_joined_before_close():
    protocol = Protocol()
    protocol.allowed = False
    stopped = threading.Event()
    callbacks = []
    authority = client(protocol)
    try:
        authority.register()
        # Activation still grants capture; the actual later poll revokes it.
        protocol.allowed = True
        authority.activate()
        protocol.allowed = False

        def stop():
            callbacks.append(threading.current_thread())
            stopped.set()

        authority.start_monitor(stop)
        assert stopped.wait(3)
        authority.stop_monitor()
        assert len(callbacks) == 1 and callbacks[0] is not threading.current_thread()
        assert authority.current() is False
        assert 'retire' not in protocol.calls
        authority.retire()
        assert protocol.calls[-1] == 'retire'
    finally:
        authority.close()


def test_late_allowed_poll_cannot_reopen_after_stop_or_close_HTTP_early():
    protocol = Protocol()
    protocol.poll_release.clear()
    authority = client(protocol)
    with ThreadPoolExecutor(max_workers=1) as fixture:
        try:
            authority.register()
            authority.activate()
            authority.start_monitor(lambda: None)
            assert protocol.poll_entered.wait(3)
            closing = fixture.submit(authority.close)
            # Observe actual owner progress before asserting the native wait.
            stopped_admission(authority)
            assert not closing.done()
            assert not protocol.close_entered.is_set()
            protocol.poll_release.set()
            closing.result(3)
            assert protocol.close_entered.is_set()
            assert authority.current() is False
            assert 'retire' not in protocol.calls
        finally:
            protocol.poll_release.set()
            authority.close()


def test_actual_transport_close_return_is_part_of_client_close_receipt():
    protocol = Protocol()
    protocol.close_release.clear()
    authority = client(protocol)
    with ThreadPoolExecutor(max_workers=1) as fixture:
        try:
            authority.register()
            authority.activate()
            closing = fixture.submit(authority.close)
            assert protocol.close_entered.wait(3)
            assert not closing.done()
            assert authority.current() is False and 'retire' not in protocol.calls
            protocol.close_release.set()
            closing.result(3)
        finally:
            protocol.close_release.set()
            authority.close()


def test_native_poll_callback_cannot_close_its_own_http_owner():
    protocol = Protocol()
    observed = threading.Event()
    errors = []
    authority = client(protocol)

    def callback(action):
        if action != 'poll':
            return
        try:
            authority.close()
        except RuntimeError as error:
            errors.append(str(error))
        finally:
            observed.set()

    try:
        authority.register()
        authority.activate()
        protocol.hook = callback
        authority.start_monitor(lambda: None)
        assert observed.wait(3)
        assert errors and all(ADMIN_TOKEN not in value for value in errors)
        assert not protocol.close_entered.is_set()
        authority.stop_monitor()
    finally:
        protocol.hook = None
        authority.close()


def test_poll_failure_reports_fixed_state_without_echoing_credentials():
    protocol = Protocol()
    protocol.failure = requests.RequestException('private ' + ADMIN_TOKEN + ' ' + URL)
    stopped = threading.Event()
    authority = client(protocol)
    try:
        authority.register()
        authority.activate()
        authority.start_monitor(stopped.set)
        assert stopped.wait(3)
        authority.stop_monitor()
        assert authority.current() is False
        assert 'retire' not in protocol.calls
    finally:
        authority.close()


def test_transport_close_failure_is_not_a_retirement_ack_or_raw_secret_error():
    protocol = Protocol()
    authority = client(protocol)
    authority.register()
    authority.activate()
    protocol.close_failure = True
    with pytest.raises(RuntimeError) as caught:
        authority.close()
    assert ADMIN_TOKEN not in str(caught.value) and URL not in str(caught.value)
    assert 'retire' not in protocol.calls
    assert authority.current() is False


@pytest.mark.parametrize('url', [
    'http://example.invalid', 'http://user:secret@127.0.0.1',
    'http://127.0.0.1/path', 'http://127.0.0.1?token=secret',
    'http://127.0.0.1#fragment', 'file:///tmp/dashboard',
])
def test_invalid_explicit_locator_cannot_create_a_transport(url):
    created = []

    def forbidden():
        created.append(True)
        raise AssertionError('invalid URL reached transport')

    with pytest.raises((ValueError, RuntimeError)) as caught:
        CalibrationAuthorityClient(url, ADMIN_TOKEN, session_factory=forbidden)
    assert created == []
    assert ADMIN_TOKEN not in str(caught.value) and 'secret' not in str(caught.value)
