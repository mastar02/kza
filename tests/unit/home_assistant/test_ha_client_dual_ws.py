"""Tests for the dual WebSocket architecture (calls + events channels).

Regression cover for the "Concurrent call to receive() is not allowed" race
that caused every voice command to fall back to REST (~288ms). The fix splits
the single WS into two independent connections with their own auth handshake
and message-id counter.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.home_assistant.ha_client import HomeAssistantClient


@pytest.fixture
def client():
    return HomeAssistantClient(url="http://test:8123", token="fake")


def test_init_has_both_ws_slots(client):
    """Both connection slots exist and start as None with independent counters."""
    assert client._ws_calls is None
    assert client._ws_events is None
    assert client._ws_msg_id_calls == 1
    assert client._ws_msg_id_events == 1


def test_ws_connection_alias_reads_calls(client):
    """Backward compat: _ws_connection property returns _ws_calls."""
    mock_ws = MagicMock()
    client._ws_calls = mock_ws
    assert client._ws_connection is mock_ws


def test_ws_connection_alias_setter_writes_calls(client):
    """Legacy callers that assigned to _ws_connection must still reach _ws_calls."""
    mock_ws = MagicMock()
    client._ws_connection = mock_ws
    assert client._ws_calls is mock_ws


def test_msg_id_counters_are_independent(client):
    """HA scopes ids per connection — the two counters must not alias."""
    client._ws_msg_id_calls = 42
    client._ws_msg_id_events = 7
    assert client._ws_msg_id_calls == 42
    assert client._ws_msg_id_events == 7
    # Bumping one must not affect the other.
    client._ws_msg_id_calls += 1
    assert client._ws_msg_id_events == 7


# Nota 2026-04-24: no existe unit test directo para la race condition
# "Concurrent call to receive()". El invariante arquitectural (dos
# conexiones independientes con receivers separados) está cubierto por
# los tests de arriba (init slots, counters, subscribe opens events,
# close shuts both). Un test del race real requeriría un servidor WS
# aiohttp de fixture — overkill para unit. Regresión se detecta en
# producción vía journalctl si el error "Concurrent call to receive()"
# vuelve a aparecer.


@pytest.mark.asyncio
async def test_close_shuts_down_both_connections(client):
    """close() must close _ws_events AND _ws_calls."""
    ws_events = MagicMock()
    ws_events.closed = False
    ws_events.close = AsyncMock()

    ws_calls = MagicMock()
    ws_calls.closed = False
    ws_calls.close = AsyncMock()

    client._ws_events = ws_events
    client._ws_calls = ws_calls
    # Skip the session close path.
    client._session = MagicMock()
    client._session.closed = True

    await client.close()

    ws_events.close.assert_awaited_once()
    ws_calls.close.assert_awaited_once()
    assert client._ws_events is None
    assert client._ws_calls is None
    assert client._ws_connected is False


@pytest.mark.asyncio
async def test_stop_state_sync_closes_events_channel_only(client):
    """stop_state_sync closes _ws_events but leaves _ws_calls alone."""
    ws_events = MagicMock()
    ws_events.closed = False
    ws_events.close = AsyncMock()

    ws_calls = MagicMock()
    ws_calls.closed = False
    ws_calls.close = AsyncMock()

    client._ws_events = ws_events
    client._ws_calls = ws_calls

    await client.stop_state_sync()

    ws_events.close.assert_awaited_once()
    ws_calls.close.assert_not_awaited()
    assert client._ws_events is None
    assert client._ws_calls is ws_calls


@pytest.mark.asyncio
async def test_subscribe_and_sync_opens_events_channel_when_missing(client, monkeypatch):
    """If _ws_events is None, _subscribe_and_sync must open a fresh one via
    _open_ws_authenticated('events') — not reuse the calls channel."""
    client._fetch_all_states_rest = AsyncMock(return_value=[
        {"entity_id": "light.a", "state": "off"},
    ])

    ws_events = MagicMock()
    ws_events.closed = False
    ws_events.send_json = AsyncMock()

    messages = [
        {"id": 2, "type": "result", "success": True},
    ]
    idx = {"i": 0}

    async def receive_json():
        i = idx["i"]
        idx["i"] += 1
        if i < len(messages):
            return messages[i]
        client._state_sync_running = False
        raise asyncio.TimeoutError()

    ws_events.receive_json = AsyncMock(side_effect=receive_json)

    opened = {"purpose": None, "count": 0}

    async def fake_open(purpose: str):
        opened["purpose"] = purpose
        opened["count"] += 1
        return ws_events

    client._open_ws_authenticated = fake_open
    client._state_sync_running = True
    # Pre-condition: no events channel yet.
    assert client._ws_events is None

    await client._subscribe_and_sync()

    assert opened["count"] == 1
    assert opened["purpose"] == "events"
    assert client._ws_events is ws_events
    # Subscribe was sent on the events channel using the events id counter.
    ws_events.send_json.assert_awaited_once()
    sent = ws_events.send_json.await_args[0][0]
    assert sent["type"] == "subscribe_events"
    assert sent["event_type"] == "state_changed"
