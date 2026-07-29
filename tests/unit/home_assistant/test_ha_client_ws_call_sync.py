"""Tests for WebSocket [calls] channel response matching and connect serialization.

Root cause (2026-07-25): the first voice command after an idle gap arrived on a
dead `_ws_calls` socket, triggered a lazy reconnect inside `call_service_ws`,
and returned `success=False` in 8ms. Production evidence (PID 262761):

    15:57:38.182  WebSocket HA[calls] conectado y autenticado   <- boot pre-warm
    16:01:11.680  WebSocket HA[calls] conectado y autenticado   <- lazy RE-connect
    16:01:11.680  [HA-CALL] light.turn_on@... success=False took=8ms

Two structural defects made that failure possible and undiagnosable:

1. `call_service_ws` sent `id=N` then blindly accepted the *next* frame off the
   socket. Any desync (a prior call that hit WS_CALL_TIMEOUT, fell back to REST,
   and left its late response buffered) permanently offsets the stream by one,
   so every later call reads the previous call's result.
2. `connect_websocket()` had no lock, so concurrent callers each ran a full
   handshake, each assigned `self._ws_calls`, and the loser's socket leaked.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.home_assistant.ha_client import HomeAssistantClient


@pytest.fixture
def client():
    return HomeAssistantClient(url="http://test:8123", token="fake")


def _live_ws(frames):
    """A connected WS mock that yields `frames` in order from receive_json()."""
    ws = MagicMock()
    ws.closed = False
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    ws.receive_json = AsyncMock(side_effect=list(frames))
    return ws


# ---------------------------------------------------------------------------
# Defect 1: response id matching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_service_ws_skips_stale_response_and_matches_own_id(client):
    """A buffered response from an earlier call must not be read as ours.

    Fails without id matching: the stale `success: False` frame for id=1 is
    returned as this call's result.
    """
    # Counter is at 1, so this call will send id=2.
    ws = _live_ws([
        {"id": 1, "type": "result", "success": False},   # stale, belongs to a dead call
        {"id": 2, "type": "result", "success": True},    # ours
    ])
    client._ws_calls = ws
    client._ws_connected = True

    result = await client.call_service_ws(
        "light", "turn_on", "light.grupo_escritorio",
    )

    assert result is True, "stale frame for id=1 was accepted as the id=2 result"


@pytest.mark.asyncio
async def test_call_service_ws_skips_event_frames(client):
    """Unsolicited non-result frames (no `id`) must be skipped, not counted."""
    ws = _live_ws([
        {"type": "event", "event": {"event_type": "state_changed"}},
        {"id": 2, "type": "result", "success": True},
    ])
    client._ws_calls = ws
    client._ws_connected = True

    result = await client.call_service_ws("light", "turn_on", "light.x")

    assert result is True


@pytest.mark.asyncio
async def test_call_service_ws_returns_false_on_genuine_ha_error(client):
    """A matching frame with success=False is a real failure — still False."""
    ws = _live_ws([
        {
            "id": 2,
            "type": "result",
            "success": False,
            "error": {"code": "not_found", "message": "Entity not found"},
        },
    ])
    client._ws_calls = ws
    client._ws_connected = True

    result = await client.call_service_ws("light", "turn_on", "light.nope")

    assert result is False


@pytest.mark.asyncio
async def test_call_service_ws_logs_error_body_on_failure(client, caplog):
    """The response body must reach the log — this was the evidence gap.

    Without it, `success=False took=8ms` is unattributable, which is exactly
    why the 2026-07-25 incident could not be root-caused from logs alone.
    """
    ws = _live_ws([
        {
            "id": 2,
            "type": "result",
            "success": False,
            "error": {"code": "not_found", "message": "Entity not found"},
        },
    ])
    client._ws_calls = ws
    client._ws_connected = True

    with caplog.at_level("WARNING"):
        await client.call_service_ws("light", "turn_on", "light.nope")

    assert "not_found" in caplog.text or "Entity not found" in caplog.text, (
        "failure response body was not logged"
    )


# ---------------------------------------------------------------------------
# Defect 2: connect serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_connect_does_one_handshake(client):
    """Two concurrent connects must produce exactly one authenticated socket.

    Fails without a lock: both callers run the handshake and one socket leaks.
    """
    handshakes = 0

    async def slow_open(purpose):
        nonlocal handshakes
        handshakes += 1
        await asyncio.sleep(0.01)  # widen the race window
        ws = MagicMock()
        ws.closed = False
        ws.close = AsyncMock()
        return ws

    client._open_ws_authenticated = AsyncMock(side_effect=slow_open)

    results = await asyncio.gather(
        client.connect_websocket(),
        client.connect_websocket(),
    )

    assert all(results), "both callers should report a connected socket"
    assert handshakes == 1, f"expected 1 handshake, got {handshakes} (no lock)"


@pytest.mark.asyncio
async def test_reconnect_closes_the_dead_socket(client):
    """Reconnecting must close the socket it replaces, not leak it."""
    dead = MagicMock()
    dead.closed = True
    dead.close = AsyncMock()
    client._ws_calls = dead
    client._ws_connected = False

    fresh = MagicMock()
    fresh.closed = False
    fresh.close = AsyncMock()
    client._open_ws_authenticated = AsyncMock(return_value=fresh)

    await client.connect_websocket()

    assert client._ws_calls is fresh
    dead.close.assert_awaited(), "the replaced socket was leaked without close()"


# ---------------------------------------------------------------------------
# Defect 1+2 combined: the actual reported symptom
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_call_on_fresh_connection_retries_once(client):
    """The reported bug: first command after idle fails once, then must retry.

    When the call itself had to open the connection, a single failure is not
    trustworthy — the socket was just built. One retry recovers it instead of
    losing the user's command.
    """
    ws = _live_ws([
        {"id": 2, "type": "result", "success": False},  # fails on the fresh socket
        {"id": 3, "type": "result", "success": True},   # retry succeeds
    ])
    client._ws_connected = False
    client._ws_calls = None
    client._open_ws_authenticated = AsyncMock(return_value=ws)

    result = await client.call_service_ws(
        "light", "turn_on", "light.grupo_escritorio",
    )

    assert result is True, "no retry after failing on a just-opened connection"
    assert ws.send_json.await_count == 2, "the call was not retried"


@pytest.mark.asyncio
async def test_call_on_established_connection_does_not_retry(client):
    """A failure on a warm connection is real — do not double-fire the service.

    Retrying every failure would send `turn_on` twice on a genuine HA error.
    """
    ws = _live_ws([
        {"id": 2, "type": "result", "success": False},
    ])
    client._ws_calls = ws
    client._ws_connected = True  # established, not opened by this call

    result = await client.call_service_ws("light", "turn_on", "light.x")

    assert result is False
    assert ws.send_json.await_count == 1, "warm-connection failure must not retry"
