"""The action transport has its own actor and never infers an HA effect."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest
import requests

from src.audio.manual_command_action import ManualVoiceActionClient


URL = "http://127.0.0.1:19876"
ACTION_TOKEN = "synthetic-explicit-home-control-actor"


class ActionProtocol(requests.adapters.BaseAdapter):
    def __init__(self):
        self.calls = []
        self.payload = {"request_id": "intent-1", "entity_id": "light.living", "status": "confirmed",
                        "dispatch": "accepted", "effect": "observed", "observation_scope": "ha_state"}
        self.status = 200
        self.error = None
        self.response = None
        self.sessions = []
        self.body_close_calls = self.session_close_calls = 0
        self.body_close_entered = threading.Event()
        self.body_close_release = threading.Event()
        self.body_close_release.set()
        self.body_close_failure = False
        self.hook = None

    def session(self):
        session = requests.Session()
        for adapter in session.adapters.values():
            adapter.close()
        session.mount("http://", self)
        self.sessions.append(session)
        return session

    def send(self, request, **kwargs):
        self.calls.append(request)
        assert request.url == URL + "/api/home/actions"
        assert request.headers["Authorization"] == "Bearer " + ACTION_TOKEN
        assert request.headers["Origin"] == URL
        assert kwargs["timeout"] == 10.0
        assert kwargs["proxies"] == {}
        if self.hook:
            self.hook()
        if self.error:
            raise self.error
        response = requests.Response()
        response.request = request
        response.status_code = self.status
        response._content = json.dumps(self.payload).encode()
        if self.status == 302:
            response.headers["Location"] = "https://other.invalid/api/home/actions"
        original_close = response.close

        def close():
            self.body_close_calls += 1
            self.body_close_entered.set()
            assert self.body_close_release.wait(3)
            original_close()
            if self.body_close_failure:
                raise RuntimeError("synthetic body close failure " + ACTION_TOKEN)

        response.close = close
        self.response = response
        return response

    def close(self):
        self.session_close_calls += 1


def make_client(protocol):
    return ManualVoiceActionClient(URL, ACTION_TOKEN, session_factory=protocol.session)


def test_typed_action_uses_only_explicit_action_actor_and_exact_ha_receipt():
    protocol = ActionProtocol()
    client = make_client(protocol)
    try:
        result = client.execute("light", "turn_on", "light.living", request_id="intent-1")
        assert (result.status, result.dispatch, result.effect, result.observation_scope) == (
            "confirmed", "accepted", "observed", "ha_state")
        assert json.loads(protocol.calls[0].body) == {
            "entity_id": "light.living", "operation": "turn_on", "request_id": "intent-1"}
        assert protocol.body_close_calls == 1
        assert all(session.trust_env is False for session in protocol.sessions)
    finally:
        client.close()
    assert protocol.session_close_calls == 1


def test_unconfirmed_ha_receipt_remains_unconfirmed_after_successful_http():
    protocol = ActionProtocol()
    protocol.payload.update(status="uncertain", effect="unconfirmed", observation_scope=None)
    client = make_client(protocol)
    try:
        result = client.execute("light", "turn_on", "light.living", request_id="intent-1")
        assert (result.status, result.dispatch, result.effect, result.observation_scope) == (
            "uncertain", "accepted", "unconfirmed", None)
        assert len(protocol.calls) == 1
    finally:
        client.close()


def test_nonterminal_server_receipt_cannot_confirm_retirement_or_allow_another_post():
    protocol = ActionProtocol()
    protocol.payload.update(status="in_progress", dispatch="unknown", effect="unknown", observation_scope=None)
    client = make_client(protocol)
    result = client.execute("light", "turn_on", "light.living", request_id="intent-1")
    assert (result.status, result.dispatch, result.effect) == ("in_progress", "unknown", "unknown")
    with pytest.raises(RuntimeError, match="^manual_voice_action_closed$"):
        client.execute("light", "turn_on", "light.living", request_id="intent-2")
    with pytest.raises(RuntimeError, match="^manual_voice_action_close_unconfirmed$"):
        client.close()
    assert len(protocol.calls) == protocol.body_close_calls == protocol.session_close_calls == 1


@pytest.mark.parametrize("mode", ["timeout", "redirect", "wrong_id", "missing_effect", "revoked_delivery"])
def test_unknown_transport_or_receipt_never_retries_or_claims_not_sent(mode):
    protocol = ActionProtocol()
    if mode == "timeout":
        protocol.error = requests.Timeout("synthetic transport secret " + ACTION_TOKEN)
    elif mode == "redirect":
        protocol.status = 302
    elif mode == "wrong_id":
        protocol.payload["request_id"] = "another-intent"
    elif mode == "missing_effect":
        protocol.payload.pop("effect")
    else:
        protocol.status = 403
    client = make_client(protocol)
    try:
        result = client.execute("light", "turn_on", "light.living", request_id="intent-1")
        assert (result.status, result.dispatch, result.effect) == ("uncertain", "unknown", "unknown")
        assert ACTION_TOKEN not in repr(result) and URL not in repr(result)
        assert len(protocol.calls) == 1
    finally:
        with pytest.raises(RuntimeError, match="^manual_voice_action_close_unconfirmed$"):
            client.close()
    assert protocol.session_close_calls == 1


def test_body_close_exception_retains_real_response_without_retry_or_false_retirement():
    protocol = ActionProtocol()
    protocol.body_close_failure = True
    client = make_client(protocol)
    result = client.execute("light", "turn_on", "light.living", request_id="intent-1")
    assert result.status == "uncertain" and result.reason == "manual_voice_action_close_unconfirmed"
    assert client._response is protocol.response
    for _ in range(2):
        with pytest.raises(RuntimeError, match="^manual_voice_action_close_unconfirmed$") as caught:
            client.close()
        assert ACTION_TOKEN not in str(caught.value)
    assert protocol.body_close_calls == protocol.session_close_calls == 1


def test_unconfirmed_body_close_seals_new_actions_without_another_post():
    protocol = ActionProtocol()
    protocol.body_close_failure = True
    client = make_client(protocol)
    result = client.execute("light", "turn_on", "light.living", request_id="intent-1")
    assert result.reason == "manual_voice_action_close_unconfirmed"
    with pytest.raises(RuntimeError, match="^manual_voice_action_closed$"):
        client.execute("light", "turn_off", "light.living", request_id="intent-2")
    assert len(protocol.calls) == protocol.body_close_calls == 1
    assert client._response is protocol.response
    with pytest.raises(RuntimeError, match="^manual_voice_action_close_unconfirmed$"):
        client.close()


def test_failure_after_real_operation_admission_does_not_strand_client_close(monkeypatch):
    protocol = ActionProtocol()
    client = make_client(protocol)
    observed = []

    def failed_outcome(*args):
        observed.append(True)
        raise RuntimeError("synthetic-outcome-construction-fault")

    monkeypatch.setattr(client, "_unknown", failed_outcome)
    with pytest.raises(RuntimeError, match="^synthetic-outcome-construction-fault$"):
        client.execute("light", "turn_on", "light.living", request_id="intent-1")
    assert observed == [True] and protocol.calls == []
    client.close()
    assert protocol.session_close_calls == 1


def test_client_close_waits_for_actual_response_close_return():
    protocol = ActionProtocol()
    protocol.body_close_release.clear()
    client = make_client(protocol)
    with ThreadPoolExecutor(max_workers=2) as pool:
        request = pool.submit(client.execute, "light", "turn_on", "light.living", request_id="intent-1")
        closing = None
        try:
            assert protocol.body_close_entered.wait(3)
            closing = pool.submit(client.close)
            assert not request.done() and not closing.done()
            assert protocol.session_close_calls == 0
        finally:
            protocol.body_close_release.set()
            assert request.result(3).status == "confirmed"
            if closing is not None:
                closing.result(3)
            else:
                client.close()
    assert protocol.body_close_calls == protocol.session_close_calls == 1


def test_close_from_real_request_callback_rejects_before_seal():
    protocol = ActionProtocol()
    client = make_client(protocol)
    observed = []

    def callback():
        with pytest.raises(RuntimeError, match="^manual_voice_action_close_from_worker$"):
            client.close()
        observed.append(True)

    protocol.hook = callback
    try:
        assert client.execute("light", "turn_on", "light.living", request_id="intent-1").status == "confirmed"
        assert observed == [True]
    finally:
        client.close()


@pytest.mark.parametrize("domain,service,entity", [
    ("lock", "unlock", "lock.door"), ("alarm_control_panel", "alarm_disarm", "alarm_control_panel.home"),
    ("switch", "turn_on", "light.living"),
])
def test_existing_action_allowlist_cannot_be_bypassed_by_chroma_metadata(domain, service, entity):
    protocol = ActionProtocol()
    client = make_client(protocol)
    try:
        with pytest.raises(ValueError, match="^manual_voice_action_intent_invalid$"):
            client.execute(domain, service, entity, request_id="intent-1")
        assert protocol.calls == []
    finally:
        client.close()
