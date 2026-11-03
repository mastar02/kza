"""Recovery must join actual close completion, including an early closed flag."""
from __future__ import annotations

import asyncio

import aiohttp
import pytest

import src.home_assistant.ha_client as ha_module
from tests.integration.test_ha_transport_ownership import _Transport, _client, _reached

pytestmark = pytest.mark.asyncio


@pytest.fixture
def transport(monkeypatch):
    controlled = _Transport()
    monkeypatch.setattr(ha_module.aiohttp, "ClientSession", controlled.session_factory)
    return controlled


async def test_successor_joins_close_owner_even_after_socket_reports_closed(
    transport, monkeypatch,
):
    previous = transport.socket("previous-calls", hold_close=True)
    successor = transport.socket("successor-calls")
    actual_close = previous.close

    async def early_closed(**kwargs):
        # aiohttp 3.14.0 sets this flag before awaiting writer/peer close.
        previous.closed = True
        return await actual_close(**kwargs)

    monkeypatch.setattr(previous, "close", early_closed)
    async with _client(transport) as client:
        await client.connect_websocket()
        assert client._ws_calls is previous
        retirement = transport.spawn(client._retire_socket("calls"))
        await _reached(previous.close_entered)
        assert previous.closed and not previous.close_finished.is_set()
        close_owner = client._socket_close_tasks[id(previous)]
        joined = asyncio.Event()
        original = client._close_socket

        async def observed_join(socket, purpose):
            if socket is previous:
                joined.set()
            return await original(socket, purpose)

        monkeypatch.setattr(client, "_close_socket", observed_join)
        reconnect = transport.spawn(client.connect_websocket())
        await _reached(joined)
        assert not close_owner.done() and not retirement.done()
        assert not successor.opened.is_set()
        assert id(previous) in client._owned_sockets
        assert previous.close_calls == 1
        previous.close_release.set()
        async with asyncio.timeout(5):
            await retirement
            await reconnect
        assert close_owner.done() and previous.close_finished.is_set()
        assert client._ws_calls is successor and successor.opened.is_set()
        assert id(previous) not in client._owned_sockets
        assert previous.close_calls == 1
        transport.assert_no_effects_or_duplicate_receivers()


@pytest.mark.parametrize("failure", ["failed", "cancelled"])
async def test_early_closed_flag_cannot_discard_a_failed_or_cancelled_close_owner(
    transport, monkeypatch, failure,
):
    previous = transport.socket("unconfirmed-calls")
    successor = transport.socket("successor-calls")
    actual_close = previous.close
    error = asyncio.CancelledError if failure == "cancelled" else OSError
    expected = asyncio.CancelledError if failure == "cancelled" else aiohttp.ClientConnectionError

    async def unconfirmed_close(**_kwargs):
        previous.closed = True
        previous.close_calls += 1
        raise error("synthetic close did not confirm retirement")

    async with _client(transport) as client:
        await client.connect_websocket()
        monkeypatch.setattr(previous, "close", unconfirmed_close)
        try:
            with pytest.raises(expected):
                await client._retire_socket("calls")
            owner = client._socket_close_tasks[id(previous)]
            assert owner.done() and previous.closed
            assert not previous.close_finished.is_set()
            with pytest.raises(expected):
                await client.connect_websocket()
            assert not successor.opened.is_set()
            assert client._socket_close_tasks[id(previous)] is owner
            assert id(previous) in client._owned_sockets
            assert client._ws_calls is None
            transport.assert_no_effects_or_duplicate_receivers()
        finally:
            # Lab cleanup supplies real close completion after the assertions;
            # the production recovery path is never allowed to invent it.
            monkeypatch.setattr(previous, "close", actual_close)
            await actual_close()
            client._socket_close_tasks.pop(id(previous), None)
