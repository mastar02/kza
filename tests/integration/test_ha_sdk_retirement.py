"""Exercise SDK-originated retirement with real loopback WebSockets."""

import asyncio
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import web
import pytest

from src.home_assistant.ha_client import HomeAssistantClient
from src.home_assistant import ha_client as ha_module
from src.monitoring.health_aggregator import HealthAggregator


pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def local_ha(*, close_release=None, subscriptions=None, effects=None, service_reply="close"):
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
            if subscriptions is not None and body["type"] == "subscribe_entities":
                subscriptions.append((ws, body["id"]))
                await ws.send_json({"type": "result", "id": body["id"], "success": True})
                await ws.send_json({
                    "type": "event", "id": body["id"],
                    "event": {"a": {"light.synthetic": {
                        "s": "off" if len(subscriptions) == 1 else "on", "a": {},
                        "c": "synthetic", "lc": 1_700_000_000 + len(subscriptions),
                    }}},
                })
                continue
            if effects is not None and body["type"] == "call_service":
                effects.append("ws")
                if service_reply == "close":
                    await ws.close()
                    break
                if service_reply == "ack":
                    await ws.send_json({"type": "result", "id": body["id"], "success": True})
                continue
            assert body == {"type": "ping", "id": 7}
            await ws.send_json({"type": "pong", "id": 7})
        return ws

    app = web.Application()
    app.router.add_get("/api/websocket", websocket)
    if effects is not None:
        async def service_post(_request):
            effects.append("rest")
            return web.json_response([])

        app.router.add_post("/api/services/{domain}/{service}", service_post)
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


@pytest.mark.parametrize("purpose", ["calls", "events"])
async def test_resistant_sdk_close_is_quarantined_and_baseline_recovers_within_60s(monkeypatch, caplog, purpose):
    subscriptions, effects, callbacks = [], [], []
    async with (local_ha(subscriptions=subscriptions, effects=effects) as (url, connections),
                aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit_per_host=2)) as session):
        loop = asyncio.get_running_loop()
        real_time = loop.time
        elapsed = 0.
        monkeypatch.setattr(loop, "time", lambda: real_time() + elapsed)
        monkeypatch.setattr(ha_module, "monotonic", loop.time)
        client = HomeAssistantClient(url, "synthetic-only", session=session)
        entity_now = client._entity_now
        monkeypatch.setattr(client, "_entity_now", lambda: entity_now() + elapsed)
        client.set_websocket_connected_handler(lambda: callbacks.append(loop.time()))
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        owner = effect = None
        try:
            await client.start_state_sync()
            await until(lambda: len(callbacks) == 2)
            assert len(callbacks) == 2 and len(connections) == 2
            baseline = client.get_entity_observation("light.synthetic")
            assert baseline.state["state"] == "off" and baseline.current(client._entity_now())
            old = getattr(client, "_ws_" + purpose)
            native_protocol = old._response.connection.protocol
            lost, descriptor = native_protocol.closed, old.get_extra_info("socket")
            original = old._writer.close

            async def resistant_close(*args, **kwargs):
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
                return await original(*args, **kwargs)

            monkeypatch.setattr(old._writer, "close", resistant_close)
            started = loop.time()
            if purpose == "calls":
                effect = asyncio.create_task(client.call_service_ws("light", "turn_on", "light.synthetic"))
            else:
                await old.send_json({"type": "close_probe"})
            await asyncio.wait_for(entered.wait(), 3)
            owner = client._sdk_retirements[id(old)]._task
            assert old.closed and not lost.done() and descriptor.fileno() >= 0
            assert client.get_health_status().retirements_pending == 1
            assert client.get_health_status().retirements_quarantined == 0

            elapsed = ha_module.STATE_CLOSE_OBSERVATION_TIMEOUT_S + .01
            await until(lambda: lost.done() and descriptor.fileno() == -1)
            assert client._channel_observations[purpose].reason == "ha_socket_close_pending"
            assert len(connections) == 2 and len(callbacks) == 2
            assert not owner.done() and not cancelled.is_set()
            assert client.get_health_status().retirements_quarantined == 0

            elapsed += ha_module.STATE_CLOSE_ABORT_JOIN_TIMEOUT_S + .01
            await asyncio.wait_for(cancelled.wait(), 3)
            await until(lambda: len(callbacks) == 3)
            assert 0 < callbacks[-1] - started <= 60.
            successor = getattr(client, "_ws_" + purpose)
            assert successor is not old and successor._response.connection.protocol is not native_protocol
            assert old._response.connection is None and id(old) not in client._owned_sockets
            assert owner not in client._transport_tasks and not owner.done()
            assert len(connections) == 3
            if purpose == "calls":
                stream, subscription_id = subscriptions[0]
                await stream.send_json({"type": "event", "id": subscription_id,
                    "event": {"c": {"light.synthetic": {"+": {"s": "off", "lc": 1_700_000_005}}}}})
                await until(lambda: client.get_entity_observation("light.synthetic").current(client._entity_now()))
            current = client.get_entity_observation("light.synthetic")
            assert current.state["state"] == ("off" if purpose == "calls" else "on")
            assert current.current(client._entity_now())
            health = client.get_health_status()
            assert health.calls.state is health.events.state is ha_module.HAConnectionState.CONNECTED
            assert health.retirements_pending == health.retirements_quarantined == 1
            assert health.retirements_quarantined_total == 1
            diagnostic = HealthAggregator(ha_client=client)._check_ha().extra
            assert diagnostic["retirements_pending"] == diagnostic["retirements_quarantined"] == 1
            assert diagnostic["retirements_quarantined_total"] == 1
            if effect is not None:
                assert await effect is False
                assert effects == ["ws"]
                assert "ha_calls_socket_closed" in caplog.text
                assert "ha_calls_socket_closed" in client.get_health_status().last_error_message

            elapsed = 61.
            assert loop.time() - started > 60. and not owner.done()
            await client._retire_socket(purpose, old)
            assert getattr(client, "_ws_" + purpose) is successor
            assert client.get_health_status().ws_connected
            if purpose == "events":
                with pytest.raises(aiohttp.ClientConnectionError, match="ha_socket_quarantine_pending"):
                    await client.close()
                assert client.get_health_status().retirements_pending == 1
                assert client.get_health_status().events.phase == "closed"
            channel = client._channel_observations[purpose]
            release.set()
            await asyncio.gather(owner, return_exceptions=True)
            assert client._channel_observations[purpose] == channel
            assert len(callbacks) == 3 and len(connections) == 3
            health = client.get_health_status()
            assert health.retirements_pending == health.retirements_quarantined == 0
            assert health.retirements_quarantined_total == 1
        finally:
            release.set()
            await asyncio.gather(*(task for task in (owner, effect) if task is not None),
                                 return_exceptions=True)
            await client.close()
            assert not session.closed


async def test_abort_without_connection_loss_refuses_successor_and_quarantine(monkeypatch):
    effects = []
    async with local_ha(effects=effects) as (url, connections), aiohttp.ClientSession() as session:
        client = HomeAssistantClient(url, "synthetic-only", session=session)
        loop = asyncio.get_running_loop()
        now = loop.time
        advance = 0.
        monkeypatch.setattr(loop, "time", lambda: now() + advance)
        entered, attempted_abort, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        observer = owner = None
        try:
            assert await client.connect_websocket()
            old = client._ws_calls
            protocol = old._response.connection.protocol
            lost = protocol.closed
            descriptor = old.get_extra_info("socket")
            native_close = old._writer.close

            async def hold_native_close(*args, **kwargs):
                entered.set()
                await release.wait()
                return await native_close(*args, **kwargs)

            monkeypatch.setattr(old._writer, "close", hold_native_close)
            monkeypatch.setattr(protocol, "abort", attempted_abort.set)
            observer = asyncio.create_task(client._retire_socket("calls"))
            await asyncio.wait_for(entered.wait(), 3)
            owner = client._sdk_retirements[id(old)]._task
            advance = ha_module.STATE_CLOSE_OBSERVATION_TIMEOUT_S + .01
            await asyncio.wait_for(attempted_abort.wait(), 3)
            assert not protocol._connection_lost_called and not lost.done()
            assert descriptor.fileno() >= 0 and not owner.done() and not observer.done()

            advance += ha_module.STATE_CLOSE_ABORT_JOIN_TIMEOUT_S + .01
            with pytest.raises(aiohttp.ClientConnectionError, match="ha_socket_abort_unconfirmed"):
                await asyncio.wait_for(observer, 3)
            with pytest.raises(aiohttp.ClientConnectionError, match="ha_socket_abort_unconfirmed"):
                await client.connect_websocket()
            with pytest.raises(aiohttp.ClientConnectionError, match="ha_socket_abort_unconfirmed"):
                await client.call_service_ws("light", "turn_on", "light.synthetic")
            assert effects == []
            assert len(connections) == 1 and client._ws_calls is None and not lost.done()
            assert id(old) in client._owned_sockets and owner in client._transport_tasks
            health = client.get_health_status()
            assert health.retirements_pending == 1
            assert health.retirements_quarantined == health.retirements_quarantined_total == 0
        finally:
            release.set()
            await asyncio.gather(*(task for task in (owner, observer) if task is not None),
                                 return_exceptions=True)
            try:
                await client.close()
            except aiohttp.ClientConnectionError as error:
                assert str(error) == "ha_socket_abort_unconfirmed"


@pytest.mark.parametrize("outcome", ["timeout", "replaced"])
async def test_inflight_calls_outcome_is_observed_without_replaying(monkeypatch, caplog, outcome):
    effects = []
    reply = "hold" if outcome == "timeout" else "ack"
    async with local_ha(effects=effects, service_reply=reply) as (url, connections), aiohttp.ClientSession() as session:
        client = HomeAssistantClient(url, "synthetic-only", session=session)
        release, ack_received = asyncio.Event(), asyncio.Event()
        call = None
        try:
            assert await client.connect_websocket()
            if outcome == "timeout":
                monkeypatch.setattr(ha_module, "WS_CALL_TIMEOUT", .02)
            else:
                receive = client._receive_ws_result

                async def hold_ack(*args):
                    result = await receive(*args)
                    ack_received.set()
                    await release.wait()
                    return result

                monkeypatch.setattr(client, "_receive_ws_result", hold_ack)
            call = asyncio.create_task(client.call_service_ws("light", "turn_on", "light.synthetic"))
            if outcome == "replaced":
                await asyncio.wait_for(ack_received.wait(), 3)
                await client._retire_socket("calls")
                assert await client.connect_websocket() and len(connections) == 2
                successor = client.get_health_status().calls
                release.set()
            assert await asyncio.wait_for(call, 3) is False
            assert effects == ["ws"] and client.get_health_status().error_count == 1
            if outcome == "replaced":
                assert "ha_calls_socket_replaced" in caplog.text
                assert client.get_health_status().calls == successor
            else:
                assert "WebSocket timeout for light.turn_on" in caplog.text
        finally:
            release.set()
            if call is not None:
                await asyncio.gather(call, return_exceptions=True)
            await client.close()
