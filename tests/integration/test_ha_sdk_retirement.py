"""Exercise SDK-originated retirement with real loopback WebSockets."""

import asyncio
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import web
import pytest

from src.home_assistant.ha_client import HomeAssistantClient
from src.home_assistant import ha_client as ha_module


pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def local_ha(*, close_release=None):
    connections = []

    async def websocket(request):
        hold = close_release is not None and not connections
        ws = web.WebSocketResponse(autoclose=not hold)
        await ws.prepare(request)
        connections.append(ws)
        await ws.send_json({"type": "auth_required"})
        auth = await ws.receive_json()
        assert auth == {"type": "auth", "access_token": "synthetic-only"}
        await ws.send_json({"type": "auth_ok"})
        if hold:
            assert (await ws.receive()).type is aiohttp.WSMsgType.CLOSE
            await close_release.wait()
            return ws
        async for message in ws:
            assert message.type is aiohttp.WSMsgType.TEXT
            body = message.json()
            if body == {"type": "close_probe"}:
                await ws.close()
                break
            assert body == {"type": "ping", "id": 7}
            await ws.send_json({"type": "pong", "id": 7})
        return ws

    app = web.Application()
    app.router.add_get("/api/websocket", websocket)
    runner = web.AppRunner(app)
    await runner.setup()
    server = web.TCPSite(runner, "127.0.0.1", 0)
    await server.start()
    try:
        yield "http://127.0.0.1:" + str(server._server.sockets[0].getsockname()[1]), connections
    finally:
        if close_release is not None:
            close_release.set()
        await runner.cleanup()


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.parametrize("abort_transport", [False, True])
async def test_sdk_autoclose_stays_owned_until_writer_and_connection_exit(monkeypatch, abort_transport):
    async with local_ha() as (url, connections), aiohttp.ClientSession() as session:
        client = HomeAssistantClient(url, "synthetic-only", session=session)
        release, entered = asyncio.Event(), asyncio.Event()
        receiver = successor = None
        other = await session.ws_connect(url + "/api/websocket")
        assert (await other.receive_json())["type"] == "auth_required"
        await other.send_json({"type": "auth", "access_token": "synthetic-only"})
        assert (await other.receive_json())["type"] == "auth_ok"
        try:
            assert await client.connect_websocket()
            old = client._ws_calls
            protocol = old._response.connection.protocol
            lost = protocol.closed
            socket = old.get_extra_info("socket")
            writer_close = old._writer.close

            async def held_writer_close(*args, **kwargs):
                entered.set()
                await release.wait()
                return await writer_close(*args, **kwargs)

            monkeypatch.setattr(old._writer, "close", held_writer_close)
            receiver = client._retain_transport(client._receive_ws_result(old, 99), "calls")
            await old.send_json({"type": "close_probe"})
            await asyncio.wait_for(entered.wait(), 3)
            assert old.closed and not receiver.done()
            assert not lost.done() and socket.fileno() >= 0
            assert id(old) not in client._socket_close_tasks
            sdk_owner = client._sdk_retirements[id(old)]._task
            assert sdk_owner is not None and not sdk_owner.done()
            assert client._transport_tasks[sdk_owner] == 'calls'
            if abort_transport:
                protocol.abort()
                await asyncio.gather(asyncio.shield(lost), return_exceptions=True)
                assert lost.done() and socket.fileno() == -1
                assert not receiver.done()

            successor = asyncio.create_task(client.connect_websocket())
            await until(lambda: id(old) in client._socket_close_tasks
                        or id(old) not in client._owned_sockets or successor.done())
            assert id(old) in client._owned_sockets
            assert not successor.done() and len(connections) == 2
            assert not receiver.done()
            assert lost.done() is abort_transport
            await other.send_json({"type": "ping", "id": 7})
            assert await other.receive_json() == {"type": "pong", "id": 7}

            release.set()
            await asyncio.gather(receiver, return_exceptions=True)
            assert await asyncio.wait_for(successor, 3)
            assert lost.done() and socket.fileno() == -1
            assert sdk_owner.done() and sdk_owner.exception() is None
            assert id(old) not in client._owned_sockets
            assert client._ws_calls is not old and len(connections) == 3
            await client.close()

            assert not session.closed and not other.closed
            await other.send_json({"type": "ping", "id": 7})
            assert await other.receive_json() == {"type": "pong", "id": 7}
        finally:
            release.set()
            await asyncio.gather(*(t for t in (receiver, successor) if t is not None),
                                 return_exceptions=True)
            await client.close()
            await other.close()


@pytest.mark.parametrize("owned_session", [False, True])
async def test_actual_sdk_retirement_survives_observer_cancel_and_observation_timeout(
    monkeypatch, owned_session,
):
    monkeypatch.setattr(ha_module, "STATE_CLOSE_OBSERVATION_TIMEOUT_S", .01)
    async with local_ha() as (url, _), aiohttp.ClientSession() as borrowed:
        client = HomeAssistantClient(url, "synthetic-only",
                                     session=None if owned_session else borrowed)
        release, entered = asyncio.Event(), asyncio.Event()
        observer = joined = None
        try:
            assert await client.connect_websocket()
            session, ws = client._session, client._ws_calls
            lost = ws._response.connection.protocol.closed
            socket = ws.get_extra_info("socket")
            original = ws._writer.close

            async def held_writer_close(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original(*args, **kwargs)

            monkeypatch.setattr(ws._writer, "close", held_writer_close)
            observer = asyncio.create_task(client._retire_socket("calls"))
            await asyncio.wait_for(entered.wait(), 3)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            await until(lambda: client._channel_observations["calls"].reason
                        == "ha_socket_close_pending")
            owner = client._socket_close_tasks[id(ws)]
            assert not owner.done() and id(ws) in client._owned_sockets
            await asyncio.wait_for(asyncio.shield(lost), 3)
            assert lost.done() and socket.fileno() == -1
            joined = asyncio.create_task(client._retire_socket("calls"))
            release.set()
            await asyncio.wait_for(joined, 3)
            assert owner.done() and not owner.cancelled() and owner.exception() is None
            assert lost.done() and socket.fileno() == -1
            assert id(ws) not in client._owned_sockets
            await client.close()
            assert session.closed is owned_session and not borrowed.closed
        finally:
            release.set()
            await asyncio.gather(*(t for t in (observer, joined) if t is not None),
                                 return_exceptions=True)
            await client.close()

async def test_sdk_ping_task_stays_owned_after_field_and_descriptor_are_cleared(monkeypatch):
    monkeypatch.setattr(ha_module, "STATE_CLOSE_OBSERVATION_TIMEOUT_S", .01)
    async with local_ha() as (url, _), aiohttp.ClientSession() as session:
        client = HomeAssistantClient(url, "synthetic-only", session=session)
        release, entered, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        retirement = ping = None
        try:
            assert await client.connect_websocket()
            ws = client._ws_calls
            lost = ws._response.connection.protocol.closed
            socket = ws.get_extra_info("socket")
            original = ws._writer.send_frame

            async def held_ping(message, opcode, *args, **kwargs):
                if opcode == aiohttp.WSMsgType.PING:
                    entered.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        await release.wait()
                return await original(message, opcode, *args, **kwargs)

            monkeypatch.setattr(ws._writer, "send_frame", held_ping)
            ws._heartbeat_cb.cancel()
            ws._heartbeat_cb = None
            ws._heartbeat_when = asyncio.get_running_loop().time() - 1
            ws._send_heartbeat()
            ping = ws._ping_task
            await asyncio.wait_for(entered.wait(), 3)
            retirement = asyncio.create_task(client._retire_socket("calls"))
            await asyncio.wait_for(cancelled.wait(), 3)
            await until(lambda: client._channel_observations["calls"].reason
                        == "ha_socket_close_pending")
            assert ws._ping_task is None and not ping.done()
            assert lost.done() and socket.fileno() == -1
            assert not retirement.done() and id(ws) in client._owned_sockets
            release.set()
            await asyncio.wait_for(retirement, 3)
            assert ping.done() and id(ws) not in client._owned_sockets
        finally:
            release.set()
            await asyncio.gather(*(t for t in (retirement, ping) if t is not None),
                                 return_exceptions=True)
            await client.close()


async def test_observation_expiry_aborts_only_target_connection_and_joins_sdk(monkeypatch):
    monkeypatch.setattr(ha_module, "STATE_CLOSE_OBSERVATION_TIMEOUT_S", .01)
    release = asyncio.Event()
    async with local_ha(close_release=release) as (url, connections), aiohttp.ClientSession() as session:
        client = HomeAssistantClient(url, "synthetic-only", session=session)
        other = None
        try:
            assert await client.connect_websocket()
            ws = client._ws_calls
            lost = ws._response.connection.protocol.closed
            socket = ws.get_extra_info("socket")
            other = await session.ws_connect(url + "/api/websocket")
            await other.receive_json()
            await other.send_json({"type": "auth", "access_token": "synthetic-only"})
            assert (await other.receive_json())["type"] == "auth_ok"
            await asyncio.wait_for(client._retire_socket("calls"), 3)
            assert not release.is_set()
            assert lost.done() and socket.fileno() == -1
            assert id(ws) not in client._owned_sockets
            assert id(ws) not in client._sdk_retirements
            assert await client.connect_websocket()
            assert len(connections) == 3 and client._ws_calls is not ws
            await client.close()
            assert not session.closed and not other.closed
            await other.send_json({"type": "ping", "id": 7})
            assert await other.receive_json() == {"type": "pong", "id": 7}
        finally:
            release.set()
            await client.close()
            if other is not None:
                await other.close()


async def test_failed_sdk_ping_is_already_done_when_heartbeat_cleanup_captures_it(monkeypatch):
    async with local_ha() as (url, _), aiohttp.ClientSession() as session:
        client = HomeAssistantClient(url, "synthetic-only", session=session)
        captured_done = []
        try:
            assert await client.connect_websocket()
            ws = client._ws_calls
            original_frame = ws._writer.send_frame
            original_cancel = ws._cancel_heartbeat

            async def failed_ping(message, opcode, *args, **kwargs):
                if opcode == aiohttp.WSMsgType.PING:
                    await asyncio.sleep(0)
                    raise aiohttp.ClientConnectionError("synthetic_ping_failure")
                return await original_frame(message, opcode, *args, **kwargs)

            def observe_cancel():
                if ws._ping_task is not None:
                    captured_done.append(ws._ping_task.done())
                original_cancel()

            monkeypatch.setattr(ws._writer, "send_frame", failed_ping)
            monkeypatch.setattr(ws, "_cancel_heartbeat", observe_cancel)
            ws._heartbeat_cb.cancel()
            ws._heartbeat_cb = None
            ws._heartbeat_when = asyncio.get_running_loop().time() - 1
            ws._send_heartbeat()
            await until(lambda: ws.closed)
            assert captured_done == [True]
            await asyncio.wait_for(client._retire_socket("calls"), 3)
            assert id(ws) not in client._owned_sockets
            assert not client._transport_tasks
        finally:
            await client.close()
