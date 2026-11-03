"""Required calls recovery through the real client and controlled transport.

The module-monotonic seam tests admission contracts, not elapsed-time C20.
Complete budget tests must additionally control the event loop's clock.
"""

from __future__ import annotations

import asyncio

import pytest

import src.home_assistant.ha_client as ha_module
from src.home_assistant.ha_client import HAConnectionState
from tests.integration.test_ha_transport_ownership import (
    WIRE_TIME,
    _Transport,
    _client,
    _entered_call,
    _reached,
    _wire_result,
)

pytestmark = pytest.mark.asyncio
ENTITY = "light.synthetic"


@pytest.fixture
def transport(monkeypatch):
    controlled = _Transport()
    monkeypatch.setattr(ha_module.aiohttp, "ClientSession", controlled.session_factory)
    return controlled


@pytest.fixture
def admission_clock(monkeypatch):
    clock = {"now": ha_module.monotonic()}
    monkeypatch.setattr(ha_module, "monotonic", lambda: clock["now"])
    return clock


def _reject_authentication(monkeypatch, socket):
    """Change only the synthetic HA auth result, retaining its actual gates."""
    receive = socket.receive_json

    async def receive_json(**options):
        result = await receive(**options)
        if socket.receive_calls == 2:
            return _wire_result({"type": "auth_invalid", "message": "synthetic_denial"}, options)
        return result

    monkeypatch.setattr(socket, "receive_json", receive_json)


async def _baseline(client, events):
    await client.start_state_sync()
    await _reached(events.stream_waiting)
    value = client.get_entity_observation(ENTITY)
    assert value.exists is True and value.current(client._entity_now())
    assert client._ws_events is events and client._state_frame.initialized


async def _delta(events, sequence):
    events.stream_waiting.clear()
    events.delta({"c": {ENTITY: {"+": {
        "s": "on" if sequence % 2 else "off", "lc": WIRE_TIME + sequence,
    }}}})
    await _reached(events.stream_waiting)


def _assert_same_credentials_and_no_effects(client, transport):
    assert client.token == "synthetic-token"
    for socket in transport.opened:
        for frame in socket.sent:
            if frame.get("type") == "auth":
                assert frame["access_token"] == client.token
    transport.assert_no_effects_or_duplicate_receivers()


async def test_required_calls_failed_at_warmup_recovers_from_events_without_service(
    transport, admission_clock, monkeypatch,
):
    events = transport.socket("events")
    rejected = transport.socket("warmup-calls-rejected")
    _reject_authentication(monkeypatch, rejected)
    restored = transport.socket("restored-calls", hold_auth=True)
    async with _client(transport) as client:
        await _baseline(client, events)
        assert rejected.closed and rejected.close_calls == 1
        assert client._ws_calls is None
        calls = client.get_health_status().calls
        assert calls.required and calls.state is HAConnectionState.AUTH_ERROR
        not_before = client._calls_recovery_not_before
        admission_clock["now"] = not_before - 0.001
        for sequence in range(1, 5):
            await _delta(events, sequence)
        assert transport.opened == [events, rejected]
        assert client._calls_recovery_task is None and not restored.opened.is_set()
        assert client._calls_recovery_not_before == not_before

        admission_clock["now"] = not_before
        await _delta(events, 5)
        await _reached(restored.auth_entered)
        recovery = client._calls_recovery_task
        connection = client._calls_connect_task
        assert recovery is not None and not recovery.done()
        assert connection is not None and not connection.done()
        for sequence in range(6, 11):
            await _delta(events, sequence)
            assert client._calls_recovery_task is recovery
            assert client._calls_connect_task is connection
        assert transport.opened == [events, rejected, restored]
        assert events.max_receivers == restored.max_receivers == 1

        restored.auth_release.set()
        async with asyncio.timeout(5):
            await asyncio.shield(recovery)
        assert client._ws_calls is restored and client._ws_connected
        calls = client.get_health_status().calls
        assert calls.required and calls.state is HAConnectionState.CONNECTED
        assert client.get_entity_observation(ENTITY).current(client._entity_now())
        assert len(events.subscriptions) == 1
        _assert_same_credentials_and_no_effects(client, transport)
    assert all(socket.closed for socket in transport.opened)


async def test_failed_automatic_calls_probe_preserves_next_admission_despite_many_deltas(
    transport, admission_clock, monkeypatch,
):
    events = transport.socket("events")
    warmup = transport.socket("warmup-rejected")
    _reject_authentication(monkeypatch, warmup)
    failed_probe = transport.socket("automatic-rejected", hold_auth=True)
    _reject_authentication(monkeypatch, failed_probe)
    restored = transport.socket("automatic-restored", hold_auth=True)
    async with _client(transport) as client:
        await _baseline(client, events)
        admission_clock["now"] = client._calls_recovery_not_before
        await _delta(events, 1)
        await _reached(failed_probe.auth_entered)
        first_owner = client._calls_recovery_task
        assert first_owner is not None and not first_owner.done()
        failed_probe.auth_release.set()
        async with asyncio.timeout(5):
            await asyncio.shield(first_owner)
        assert failed_probe.closed and client._ws_calls is None
        not_before = client._calls_recovery_not_before
        assert not_before > admission_clock["now"]
        assert client._calls_recovery_backoff > ha_module.STATE_SYNC_BACKOFF_INITIAL_S
        admission_clock["now"] = not_before - 0.001
        for sequence in range(2, 8):
            await _delta(events, sequence)
        assert transport.opened == [events, warmup, failed_probe]
        assert not restored.opened.is_set() and client._calls_recovery_task is None
        assert client._calls_recovery_not_before == not_before

        admission_clock["now"] = not_before
        await _delta(events, 8)
        await _reached(restored.auth_entered)
        second_owner = client._calls_recovery_task
        assert second_owner is not None and second_owner is not first_owner
        restored.auth_release.set()
        async with asyncio.timeout(5):
            await asyncio.shield(second_owner)
        assert client._ws_calls is restored
        assert client._calls_recovery_backoff == ha_module.STATE_SYNC_BACKOFF_INITIAL_S
        _assert_same_credentials_and_no_effects(client, transport)


async def test_calls_recovery_survives_joined_observer_cancellation_and_events_stop(
    transport, admission_clock, monkeypatch,
):
    events = transport.socket("events")
    warmup = transport.socket("warmup-rejected")
    _reject_authentication(monkeypatch, warmup)
    restored = transport.socket("retained-calls", hold_auth=True)
    async with _client(transport) as client:
        await _baseline(client, events)
        admission_clock["now"] = client._calls_recovery_not_before
        await _delta(events, 1)
        await _reached(restored.auth_entered)
        recovery = client._calls_recovery_task
        connection = client._calls_connect_task
        assert recovery is not None and connection is not None
        entered = asyncio.Event()
        observer = transport.spawn(_entered_call(entered, client.connect_websocket))
        await _reached(entered)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert client._calls_recovery_task is recovery and not recovery.done()
        assert client._calls_connect_task is connection and not connection.done()
        assert not restored.auth_cancelled.is_set()

        await client.stop_state_sync()
        assert events.closed and client._ws_events is None
        assert not recovery.done() and not connection.done()
        assert not restored.closed and not restored.auth_cancelled.is_set()
        assert client.get_health_status().calls.required
        assert transport.opened == [events, warmup, restored]
        restored.auth_release.set()
        async with asyncio.timeout(5):
            await asyncio.shield(recovery)
        assert client._ws_calls is restored and client._ws_connected
        assert client.get_entity_state_cached(ENTITY) is None
        calls = client.get_health_status().calls
        assert calls.required and calls.state is HAConnectionState.CONNECTED
        _assert_same_credentials_and_no_effects(client, transport)


async def test_terminal_close_drains_automatic_calls_auth_and_actual_close_after_observer_cancel(
    transport, admission_clock, monkeypatch,
):
    events = transport.socket("events")
    warmup = transport.socket("warmup-rejected")
    _reject_authentication(monkeypatch, warmup)
    candidate = transport.socket("held-calls", hold_auth=True,
        complete_auth_after_cancel=True, hold_close=True)
    successor = transport.socket("must-not-open")
    async with _client(transport) as client:
        await _baseline(client, events)
        admission_clock["now"] = client._calls_recovery_not_before
        await _delta(events, 1)
        await _reached(candidate.auth_entered)
        recovery = client._calls_recovery_task
        assert recovery is not None and not recovery.done()
        entered = asyncio.Event()
        closing = transport.spawn(_entered_call(entered, client.close))
        await _reached(entered)
        assert client._closed
        owner = client._close_task
        assert owner is not None and not owner.done()
        candidate.auth_release.set()
        await _reached(candidate.close_entered)
        assert not candidate.closed and not recovery.done()
        assert client._ws_calls is None and not client._ws_connected
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        second_entered = asyncio.Event()
        joined = transport.spawn(_entered_call(second_entered, client.close))
        await _reached(second_entered)
        assert client._close_task is owner and not owner.done() and not joined.done()
        assert candidate.close_calls == 1 and not successor.opened.is_set()
        candidate.close_release.set()
        async with asyncio.timeout(5):
            await joined
            await asyncio.shield(recovery)
        assert candidate.closed and candidate.close_calls == 1
        assert client._ws_calls is client._ws_events is None
        assert all(session.closed for session in transport.sessions)
        assert not client._transport_tasks
        assert client.get_health_status().calls.phase == "closed"
        assert not await client.connect_websocket()
        client._kick_required_calls_recovery()
        assert not successor.opened.is_set()
        _assert_same_credentials_and_no_effects(client, transport)


async def test_events_do_not_open_calls_when_the_channel_was_never_requested(
    transport, admission_clock,
):
    transport.response([], status=403)
    events = transport.socket("events-without-calls-warmup")
    unrequested = transport.socket("unrequested-calls")
    async with _client(transport) as client:
        # An actual denied preflight makes startup skip optional calls warmup.
        snapshot = await client.get_state_snapshot()
        assert not snapshot.available and client._has_auth_error
        assert not client.get_health_status().calls.required
        await _baseline(client, events)
        assert not client.get_health_status().calls.required
        for sequence in range(1, 6):
            admission_clock["now"] += 20.0
            await _delta(events, sequence)
        assert transport.opened == [events] and not unrequested.opened.is_set()
        assert client._calls_recovery_task is client._calls_connect_task is None
        assert client._ws_calls is None and client.get_health_status().calls.required is False
        _assert_same_credentials_and_no_effects(client, transport)
