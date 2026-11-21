"""P5 end to end through the live application graph and a loopback HA transport."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime, timedelta
import sys
import time

from aiohttp import web
from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest
import yaml

from scripts.managed_launcher import path_preflight
from src.application import ApplicationFactories
from src.core.household_clock import HouseholdClock
from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import DashboardRole
from src.integrations.calendar_reader import CalendarAccess
from src.integrations.factory import CameraHealthProvider, build_integrations
from src.main import build_application
from src.pipeline.command_event import CommandEvent
from src.users.user_manager import PermissionLevel, UserManager
from tests.integration.test_global_capture_http import login
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, offline_config
from tests.mocks.audio_output import AudioBackend

METER = "sensor.meter"
POWER = "sensor.power"
TV = "media_player.living"
CAMERA = "camera.front"
DIAGNOSTIC = "binary_sensor.front_link"
TODO = "todo.compras"
PERSONAL = "calendar.personal"
SHARED = "calendar.shared"
OTHER = "calendar.other"
PRIVATE_EVENT = "Private P5 appointment"
FOREIGN_EVENT = "Another person's P5 event"
SHARED_EVENT = "Shared P5 event"
MEDIA_BODY = {"room": "living", "entity_id": TV, "operation": "pause"}


class Home:
    def __init__(self):
        self.states = {
            METER: ("1200", {"device_class": "energy", "unit_of_measurement": "Wh"}),
            POWER: ("0.42", {"device_class": "power", "unit_of_measurement": "kW"}),
            TV: ("playing", {"supported_features": 1025, "media_title": "Nublado", "app_name": "TV local"}),
            CAMERA: ("recording", {"entity_picture": "/secret-camera-still", "stream_url": "rtsp://secret-camera"}),
            DIAGNOSTIC: ("off", {"device_class": "connectivity"}),
            PERSONAL: ("off", {}),
            SHARED: ("off", {}),
            OTHER: ("off", {}),
        }
        self.events = {SHARED: SHARED_EVENT, PERSONAL: PRIVATE_EVENT, OTHER: FOREIGN_EVENT}
        self.items = {}
        self.changed = "2001-01-01T12:00:00+00:00"
        self.reads = []
        self.calls = []
        self.sockets = []
        self.hold = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.day = None

    async def state(self, request):
        entity = request.match_info["entity"]
        self.reads.append(entity)
        if entity not in self.states:
            return web.Response(status=404)
        state, attributes = self.states[entity]
        return web.json_response({
            "entity_id": entity, "state": state, "attributes": attributes, "last_changed": self.changed,
            "last_updated": self.changed, "last_reported": datetime.now(UTC).isoformat(),
        })

    async def service(self, request):
        domain, operation = request.match_info["domain"], request.match_info["operation"]
        data = await request.json()
        entity = data["entity_id"]
        self.calls.append((domain, operation, entity))
        if self.hold == (operation, entity):
            self.entered.set()
            await self.release.wait()
        if domain == "calendar":
            assert operation == "get_events" and request.query["return_response"] == "true"
            if entity not in self.events:
                return web.Response(status=503)
            day = self.day.isoformat()
            events = [{"summary": self.events[entity], "start": day, "end": (self.day + timedelta(days=1)).isoformat()}]
            return web.json_response({"service_response": {entity: {"events": events}}})
        if domain == "todo":
            if operation == "get_items":
                assert request.query["return_response"] == "true"
                return web.json_response({"service_response": {entity: {"items": list(self.items.values())}}})
            assert operation == "add_item"
            self.set_item(data["item"])
        return web.json_response([])

    def set_item(self, text, *, status="needs_action", updated_at=None):
        self.items[text.lower()] = {"uid": text.lower(), "summary": text, "status": status}
        if updated_at is not None:
            self.items[text.lower()]["updated_at"] = updated_at

    async def websocket(self, request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.sockets.append(socket)
        await socket.send_json({"type": "auth_required", "ha_version": "2026.11.0"})
        assert (await socket.receive_json())["type"] == "auth"
        await socket.send_json({"type": "auth_ok", "ha_version": "2026.11.0"})
        async for message in socket:
            data = message.json()
            assert data["type"] == "subscribe_entities"
            await socket.send_json({"id": data["id"], "type": "result", "success": True, "result": None})
            await socket.send_json({"id": data["id"], "type": "event", "event": {"a": {}}})
        return socket

    @asynccontextmanager
    async def serve(self):
        async def api(_request):
            return web.json_response({"message": "API running"})

        async def states(_request):
            return web.json_response([])

        app = web.Application()
        app.add_routes([web.get("/api/", api), web.get("/api/states", states),
                        web.get("/api/states/{entity}", self.state), web.get("/api/websocket", self.websocket),
                        web.post("/api/services/{domain}/{operation}", self.service)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        try:
            yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        finally:
            self.release.set()
            for socket in self.sockets:
                await socket.close()
            await runner.cleanup()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    import src.tts.piper_tts as output
    import src.vectordb.chroma_sync as index

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(index, "SentenceTransformer", lambda *_args, **_kwargs: OfflineEncoder())
    monkeypatch.setattr(output, "sd", AudioBackend())
    users = UserManager(str(tmp_path / "users.json"))
    owner, _ = users.add_user("P5 Owner", PermissionLevel.ADULT)
    other, _ = users.add_user("P5 Other", PermissionLevel.ADULT)
    home = Home()

    @asynccontextmanager
    async def open_graph(*, enabled=True, orchestrated=True):
        async with home.serve() as url, AsyncExitStack() as stack:
            config = offline_config(tmp_path, url)
            config["orchestrator"]["enabled"] = orchestrated
            config["lists"].update(ha_sync_enabled=True, ha_todo_entity=TODO if enabled else "")
            if enabled:
                config["integrations"] = {
                    "energy": {"sensors": [{"entity_id": METER, "kind": "energy"},
                                           {"entity_id": POWER, "kind": "power"}]},
                    "calendar": {
                        "ics_sources": [{"path": "house.ics", "scope": "shared"}],
                        "ha_sources": [
                            {"entity_id": SHARED, "scope": "shared"},
                            {"entity_id": PERSONAL, "scope": "personal", "owner_user_id": owner.user_id},
                            {"entity_id": OTHER, "scope": "personal", "owner_user_id": other.user_id},
                        ],
                    },
                    "media": {"players": [{"room": "living", "entity_id": TV}]},
                    "camera_health": {"cameras": {CAMERA: [{"entity_id": DIAGNOSTIC, "kind": "connectivity"}]}},
                }
            factories = ApplicationFactories(stt=lambda _: OfflineSTT(), tts=lambda _: OfflineTTS(),
                                             preflight=lambda: None)
            graph = await build_application(config, shutdown=stack, factories=factories)
            home.day = graph.household_clock.now_local().date()
            if enabled:
                root = tmp_path / "data" / "calendars"
                root.mkdir(parents=True, exist_ok=True)
                day = home.day.strftime("%Y%m%d")
                (root / "house.ics").write_text("BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:shared-ics\n"
                                                 f"DTSTART;VALUE=DATE:{day}\nSUMMARY:Evento ICS\n"
                                                 "END:VEVENT\nEND:VCALENDAR\n")
            client = await stack.enter_async_context(AsyncClient(
                transport=ASGITransport(app=graph.dashboard.app), base_url="http://testserver",
            ))
            yield graph, client
    return home, owner, open_graph


async def say(graph, text, **capture):
    graph.models.stt.text = text
    turn = await graph.pipeline.process_command(CommandEvent(np.full(16000, 0.1, dtype=np.float32), "", **capture))
    if turn["response"]:
        assert turn["response"] == graph.models.tts.texts[-1]
    return turn


async def session(graph, client, auth, subject, user_id=None):
    authority = graph.dashboard_authority
    issued = await authority.issue_credential(
        authority.resolve(auth["Authorization"]), subject=subject, role=DashboardRole.HOME,
        user_id=user_id, expires_at=time.time() + 600,
    )
    client.cookies.clear()
    await login(client, issued.token)
    return issued.identity, client.cookies[COOKIE_NAME]


@pytest.mark.parametrize("orchestrated", [True, False])
async def test_p5_voice_and_http_through_the_same_graph(rig, dashboard_auth, monkeypatch, orchestrated):
    home, _, open_graph = rig
    async with open_graph(orchestrated=orchestrated) as (graph, client):
        router = graph.pipeline.request_router
        assert router.integrations is graph.dashboard.integrations is graph.integrations
        if orchestrated:
            assert router._orchestrator.dispatcher.integrations is graph.integrations

        async def forbidden(*_args, **_kwargs):
            raise AssertionError("P5 invoked the 7B command classifier")

        monkeypatch.setattr(router, "_household_veto_passes", forbidden)
        monkeypatch.setattr(router, "_classify_command", forbidden)
        assert (await client.get("/api/integrations/energy")).status_code == 401
        await session(graph, client, dashboard_auth, "P5 visitor")

        for phrase, expected in (
            ("qué energía consume la casa", "lecturas acumuladas actuales"),
            ("cuánta potencia consume la casa", "420"),
            ("cuánto cuesta la factura de luz", "tarifa verificada"),
            ("qué hay en el calendario hoy", SHARED_EVENT),
            ("agenda mañana", "No hay eventos"),
            ("qué está pasando en la tele del living", "Nublado"),
            ("cómo están las cámaras", "Alertas técnicas detectadas: 1"),
            ("mostrame video de la cámara", "fuera de este servicio"),
        ):
            turn = await say(graph, phrase)
            assert expected in turn["response"], (phrase, turn)
            assert turn["path"] == "fast_household"
            assert PRIVATE_EVENT not in turn["response"]

        read = (await client.get("/api/integrations/energy")).json()
        assert read["reading_type"] == "accumulated_meter_reading"
        assert read["total"] == pytest.approx(1.2) and read["unit"] == "kWh"
        power = (await client.get("/api/integrations/energy", params={"kind": "power"})).json()
        assert power["reading_type"] == "instantaneous_power" and power["total"] == 420
        cost = await client.get("/api/integrations/energy", params={"kind": "bill"})
        assert cost.status_code == 422 and cost.json()["detail"] == "energy_tariff_unverified"
        calendar = (await client.get("/api/integrations/calendar")).json()
        assert calendar["configured"] and calendar["complete"]
        assert {event["summary"] for event in calendar["events"]} == {SHARED_EVENT, "Evento ICS"}
        home.events.pop(SHARED)
        incomplete = (await client.get("/api/integrations/calendar")).json()
        assert incomplete["configured"] and not incomplete["complete"]
        assert "No pude leer todo" in (await say(graph, "agenda hoy"))["response"]

        state = (await client.get("/api/integrations/media", params={"room": "living"})).json()
        assert state["players"][0]["media_title"] == "Nublado"
        for phrase, operation in (("poné pausa en la tele del living", "media_pause"),
                                  ("subí el volumen de la tele del living", "volume_up"),
                                  ("bajá el volumen de la tele", "volume_down")):
            turn = await say(graph, phrase, capture_trigger="wake", wake_score=0.99)
            assert turn["outcome"]["dispatch"] == "accepted", (phrase, turn)
            assert turn["outcome"]["effect"] == "unconfirmed"
            assert home.calls[-1] == ("media_player", operation, TV)
        count = len(home.calls)
        rejected = await say(graph, "Nexa poné pausa en la tele del living", textual_wake=True)
        assert rejected["intent"] == "gate_rejected:media_capture"
        assert len(home.calls) == count
        receipt = await client.post("/api/integrations/media/control", json=MEDIA_BODY)
        assert receipt.json() == {"status": "accepted", "reason": None}

        for raw in ("idle", "recording", "streaming"):
            home.states[CAMERA] = raw, home.states[CAMERA][1]
            home.changed = f"2001-01-0{len(raw) % 7 + 1}T12:00:00+00:00"
            before = datetime.now(UTC)
            health = await client.get("/api/integrations/cameras/health")
            camera, = health.json()["cameras"]
            assert camera["availability"] == "online"
            assert datetime.fromisoformat(camera["observed_at"]) >= before
            assert camera["diagnostics"][0]["status"] == "issue"
            assert all(secret not in health.text for secret in (raw, "secret-camera", "entity_picture", home.changed))
        snapshot = await CameraHealthProvider(graph.ha_client).read_technical_states((CAMERA, DIAGNOSTIC))
        assert snapshot.states[CAMERA].payload == {"state": "idle"}
        assert snapshot.states[DIAGNOSTIC].payload == {"state": "off", "attributes": {"device_class": "connectivity"}}
        previous = list(home.reads)
        video = await client.get("/api/integrations/cameras/health", params={"operation": "snapshot"})
        assert video.status_code == 422 and video.json()["detail"] == "camera_health_scope_error"
        assert home.reads == previous
        assert not {PERSONAL, OTHER}.intersection(home.reads)


async def test_default_selections_are_inert(rig, dashboard_auth):
    home, _, open_graph = rig
    async with open_graph(enabled=False) as (graph, client):
        await session(graph, client, dashboard_auth, "P5 visitor")
        assert (await client.get("/api/integrations/energy")).json()["selected_count"] == 0
        assert (await client.get("/api/integrations/calendar")).json() == {
            "events": [], "complete": True, "configured": False,
        }
        assert "No hay un calendario compartido configurado" in (await say(graph, "agenda hoy"))["response"]
        assert (await client.get("/api/integrations/media")).json() == {"players": []}
        assert (await client.get("/api/integrations/cameras/health")).json()["source_status"] == "not_configured"
        assert (await client.post("/api/lists/sync")).json()["status"] == "invalid_configuration"
        assert not home.reads and not home.calls


async def test_managed_calendar_root_stays_inside_state_data(tmp_path):
    state, runtime = tmp_path / "state", tmp_path / "runtime"
    calendar = state / "data" / "calendars"
    calendar.mkdir(parents=True)
    runtime.mkdir()
    (runtime / "data").symlink_to(state / "data", target_is_directory=True)
    (state / "config").mkdir()
    settings = state / "config" / "settings.yaml"
    config = {"calendar": {"data_root": "./data/calendars",
                           "ics_sources": [{"path": "house.ics", "scope": "shared"}]}}
    settings.write_text(yaml.safe_dump({"integrations": config}))
    binding = {"runtime_root": str(runtime), "state_root": str(state)}
    assert path_preflight(binding, sys.executable) == []

    reader = build_integrations(config, None, HouseholdClock(), state_root=state).calendar_reader
    (calendar / "house.ics").write_text("BEGIN:VCALENDAR\nEND:VCALENDAR\n")
    assert (await reader.get_day(access=CalendarAccess())).complete
    outside = tmp_path / "outside"
    calendar.rename(outside)
    calendar.symlink_to(outside, target_is_directory=True)
    assert not (await reader.get_day(access=CalendarAccess())).complete
    assert "relative_path:integrations.calendar.data_root" in path_preflight(binding, sys.executable)
    with pytest.raises(ValueError, match="calendar_data_root_outside_data_directory"):
        build_integrations(config, None, HouseholdClock(), state_root=state)


async def test_revoked_session_cannot_receive_personal_calendar_or_reconcile_lists(rig, dashboard_auth):
    home, owner, open_graph = rig
    async with open_graph() as (graph, client):
        shared = await graph.list_manager.create_list(None, "compras", shared=True)
        private = await graph.list_manager.create_list(owner.user_id, "privada")
        await graph.list_manager.add_item_to_list_id(private.id, "Private list item", owner.user_id, owner=owner.user_id)
        await session(graph, client, dashboard_auth, "P5 visitor")
        assert (await client.get(f"/api/lists/{private.id}/items")).status_code == 404
        assert (await client.post("/api/lists/sync")).json()["status"] == "complete"

        identity, cookie = await session(graph, client, dashboard_auth, "P5 owner", owner.user_id)
        personal = await client.get("/api/integrations/calendar")
        assert personal.status_code == 200 and PRIVATE_EVENT in personal.text and FOREIGN_EVENT not in personal.text
        spoofed = await client.get("/api/integrations/calendar", params={"user_id": "other"})
        assert spoofed.status_code == 422
        assert "Private list item" in (await client.get(f"/api/lists/{private.id}/items")).text

        home.hold = ("get_events", PERSONAL)
        async with asyncio.timeout(5):
            request = asyncio.create_task(client.get("/api/integrations/calendar"))
            await home.entered.wait()
            root = graph.dashboard_authority.resolve(dashboard_auth["Authorization"])
            assert await graph.dashboard_authority.revoke_credential(root, identity)
            home.release.set()
            response = await request
        assert response.status_code == 403 and PRIVATE_EVENT not in response.text

        client.cookies.clear()
        stale = {"Cookie": f"{COOKIE_NAME}={cookie}"}
        assert (await client.get(f"/api/lists/{private.id}/items", headers=stale)).status_code == 401
        assert (await client.post("/api/lists/sync", headers=stale)).status_code == 401
        await session(graph, client, dashboard_auth, "P5 second owner", owner.user_id)
        home.hold = ("get_items", TODO)
        home.entered.clear()
        home.release.clear()
        home.set_item("Remote while revoked")
        async with asyncio.timeout(5):
            request = asyncio.create_task(client.post("/api/lists/sync"))
            await home.entered.wait()
            assert (await client.post("/api/admin/auth/logout")).json()["session_revoked"]
            home.release.set()
            response = await request
        assert response.status_code == 403 and "Remote while revoked" not in response.text
        assert await graph.list_manager.get_items_in_scope(shared.id, shared_only=True) == []
        assert not any(call[2] == OTHER for call in home.calls)


async def test_media_revocation_after_post_preserves_receipt_in_http(rig, dashboard_auth):
    home, _, open_graph = rig
    async with open_graph() as (graph, client):
        identity, _ = await session(graph, client, dashboard_auth, "P5 visitor")
        home.hold = ("media_pause", TV)
        async with asyncio.timeout(5):
            request = asyncio.create_task(client.post("/api/integrations/media/control", json=MEDIA_BODY))
            await home.entered.wait()
            root = graph.dashboard_authority.resolve(dashboard_auth["Authorization"])
            assert await graph.dashboard_authority.revoke_credential(root, identity)
            home.release.set()
            response = await request
        assert response.status_code == 200
        assert response.json() == {"status": "uncertain", "reason": "authority_lost_after_dispatch"}
        assert home.calls.count(("media_player", "media_pause", TV)) == 1
        assert (await client.post("/api/integrations/media/control", json=MEDIA_BODY)).status_code == 401


async def test_list_writes_surface_real_reconciliation_conflicts(rig, dashboard_auth):
    home, _, open_graph = rig
    async with open_graph() as (graph, client):
        await session(graph, client, dashboard_auth, "P5 visitor")
        shared = await graph.list_manager.create_list(None, "compras", shared=True)
        voice_item = "leche"
        assert (await say(graph, f"agregá {voice_item} a la lista de compras"))["task_completed"]
        await client.post(f"/api/lists/{shared.id}/items", json={"text": "pan"})
        for item in (voice_item, "pan"):
            home.set_item(item, status="completed", updated_at=time.time() + 60)
        uncertain = await say(graph, f"agregá {voice_item} a la lista de compras")
        assert uncertain["outcome"]["status"] == uncertain["outcome"]["persistence"] == "uncertain"
        assert "después de sincronizar" in uncertain["response"]
        result = await client.post(f"/api/lists/{shared.id}/items", json={"text": "pan"})
        assert result.status_code == 503
        assert result.json()["detail"] == "list_write_uncertain"
        assert result.json()["outcome"]["status"] == result.json()["outcome"]["persistence"] == "uncertain"


async def test_ws_reconnect_reconciles_in_background_and_shutdown_cancels(rig):
    home, _, open_graph = rig
    async with open_graph() as (graph, _):
        shared = await graph.list_manager.create_list(None, "compras", shared=True)
        ha = graph.ha_client
        home.set_item("Primer artículo")
        assert await ha.connect_websocket()
        async with asyncio.timeout(5):
            while not await graph.list_manager.get_items_in_scope(shared.id, shared_only=True):
                await asyncio.sleep(0.01)
        await home.sockets[-1].close()
        # Give aiohttp a receive to observe the server close on the calls socket.
        await ha._ws_calls.receive()
        home.set_item("Segundo artículo")
        assert await ha.connect_websocket()
        async with asyncio.timeout(5):
            while len(await graph.list_manager.get_items_in_scope(shared.id, shared_only=True)) != 2:
                await asyncio.sleep(0.01)
        assert len(home.sockets) == 2
        await home.sockets[-1].close()
        await ha._ws_calls.receive()
        home.hold = ("get_items", TODO)
        async with asyncio.timeout(3):
            assert await ha.connect_websocket()
            await home.entered.wait()
        async with asyncio.timeout(0.3):
            assert (await say(graph, "qué hora es", capture_trigger="wake"))["path"] == "fast_clock"
    assert not any(task.get_name() == "ha-shared-list-reconcile" for task in asyncio.all_tasks())
