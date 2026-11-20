"""T033-U5 offline contract checks through the application factory."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from uuid import UUID

import aiohttp
from chromadb.api.client import Client as ChromaClient
from chromadb.api.models.Collection import Collection
from httpx import ASGITransport, AsyncClient, Response
import pytest

from src.application import ApplicationFactories, ApplicationGraph
from src.core.task_outcome import EffectState
from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import DashboardRole
from src.home_assistant.ha_client import HomeAssistantClient
from src.main import build_application
from src.memory.fact_ledger import MemoryReviewer
from src.routines.models import ActionStatus, RoutineRun, RunStatus, parse_routine_spec
from src.routines.proposals import ProposalOperation, RoutineProposalTray
from src.routines.routine_store import RoutineStore
from src.routines.voice_routines import VoiceRoutineService
from src.users.user_manager import PermissionLevel, User
from tests.integration.test_capability_application import configuration
from tests.integration.test_global_capture_http import login
from tests.integration.test_routine_home_denial import login as role_session, wait_for_worker
from tests.integration.test_routine_voice_execute import leaves, say
from tests.integration.test_routine_voice_factory import OWN_WAKE

pytestmark = pytest.mark.asyncio

ROUTINES = "/api/routines"
PROPOSALS = f"{ROUTINES}/proposals"
WHOAMI = "/api/admin/auth/whoami"
PROPOSE = "creá una rutina para apagar la luz del living a las veinte todos los días"
REVIEW_REPLY = "Dejé la propuesta para confirmar en la web."
HA_URL = "http://ha.test"
TARGET = "light.living"
SOURCE = str(UUID(int=1))
OTHER_SOURCE = str(UUID(int=2))
NO_PROPOSALS = {"proposals": [], "cleared_on_restart": True}


class StateResponse:
    """Supply the same streamed JSON body consumed by the real HA reader."""

    def __init__(self, payload: object, *, status: int = 200) -> None:
        self.status = status
        self.content = self
        self._payload = payload
        self._body = BytesIO(json.dumps(payload).encode())

    async def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    async def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        assert self.status == 200


@dataclass
class OfflineHASession:
    """Script HA HTTP responses at its injected session boundary, without sockets."""

    closed = False
    status: int = 200
    timeout: bool = False
    hold: bool = False
    state: str = "on"
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    calls: list[tuple[str, str, dict | None]] = field(default_factory=list)

    @asynccontextmanager
    async def get(self, url: str, **options):
        path = urlsplit(url).path
        assert url.startswith(HA_URL + "/")
        self.calls.append(("GET", path, None))
        entity = {"entity_id": TARGET, "state": self.state,
                  "attributes": {"friendly_name": "Luz living"}}
        payloads = {"/api/": {"message": "API running"}, "/api/states": [entity],
                    f"/api/states/{TARGET}": entity}
        yield StateResponse(payloads[path])

    @asynccontextmanager
    async def post(self, url: str, *, json: dict | None = None, **options):
        path = urlsplit(url).path
        assert url.startswith(HA_URL + "/")
        self.calls.append(("POST", path, json))
        assert path in {"/api/services/light/turn_on", "/api/services/light/turn_off"}
        assert json == {"entity_id": TARGET}
        self.entered.set()
        if self.hold:
            await self.release.wait()
        if self.timeout:
            raise aiohttp.ServerTimeoutError("synthetic service timeout")
        if self.status == 200:
            self.state = "on" if path.endswith("/turn_on") else "off"
        yield StateResponse([], status=self.status)

    @asynccontextmanager
    async def delete(self, url: str, **options):
        self.calls.append(("DELETE", urlsplit(url).path, None))
        raise AssertionError(f"Unexpected HA delete: {url}")
        yield

    def services(self) -> list[str]:
        return [path.removeprefix("/api/services/light/") for method, path, _ in self.calls
                if method == "POST" and path.startswith("/api/services/")]

    def assert_no_automation_writes(self) -> None:
        assert all("/automation" not in path for method, path, _ in self.calls if method != "GET")


@dataclass
class Home:
    graph: ApplicationGraph
    ha: OfflineHASession
    selection: Path

    @property
    def tray(self) -> RoutineProposalTray:
        return self.graph.dashboard.routine_proposals

    @property
    def store(self) -> RoutineStore:
        return self.graph.routine_scheduler.store


@asynccontextmanager
async def application(tmp_path: Path, factories: ApplicationFactories, *, orchestrated: bool = True):
    ha = OfflineHASession()
    config, selection = configuration(tmp_path, HA_URL)
    config["orchestrator"]["enabled"] = orchestrated
    config["routines"]["voice_brake"] = True
    factories = replace(factories, ha_client=lambda **kwargs: HomeAssistantClient(**kwargs, session=ha))
    async with AsyncExitStack() as stack:
        graph = await build_application(config, shutdown=stack, factories=factories)
        ha.calls.clear()  # Exclude the factory's connectivity check.
        try:
            yield Home(graph, ha, selection)
        finally:
            ha.release.set()


@asynccontextmanager
async def operator(home: Home, dashboard_auth: dict):
    async with AsyncClient(transport=ASGITransport(app=home.graph.dashboard.app),
                           base_url="http://testserver") as client:
        await login(client, dashboard_auth["Authorization"].split(" ", 1)[1])
        identity = await client.get(WHOAMI)
        assert identity.status_code == 200 and "routines:manage" in identity.json()["permissions"]
        yield client


@pytest.fixture
async def home(tmp_path, leaves, dashboard_auth):
    async with application(tmp_path, leaves) as current:
        yield current


def definition(name: str = "Ensayo", *, enabled: bool = True, entity: str = TARGET,
               operation: str = "turn_off", two: bool = False) -> dict:
    actions = [{"entity_id": entity, "operation": operation}]
    if two:
        actions.append({"entity_id": TARGET, "operation": "turn_on"})
    return {"name": name, "description": "", "enabled": enabled, "timezone": "America/Montevideo",
            "schedule": None, "conditions": [], "actions": actions}


async def propose(home: Home, source: str = SOURCE) -> str:
    before = set(home.tray._entries)
    result = await say(home, PROPOSE, f"voice:proposal:{source}", **(OWN_WAKE | {"conversation_id": source}))
    assert result["response"] == REVIEW_REPLY
    assert result["outcome"]["persistence"] == "volatile"
    [proposal_id] = set(home.tray._entries) - before
    return proposal_id


async def finished(home: Home, routine_id: str, run_id: str) -> RoutineRun:
    scheduler = home.graph.routine_scheduler
    await wait_for_worker(scheduler, run_id)
    run = await scheduler.get_run(routine_id, run_id)
    assert run.status is not RunStatus.RUNNING
    return run


def expect(response: Response, status: int, detail: str | None = None) -> dict | list:
    assert response.status_code == status, response.text
    body = response.json()
    if detail is not None:
        assert body == {"detail": detail}
    return body


@pytest.mark.parametrize(("verb", "operation", "status"), [
    ("pará", "stop", "stop_requested"),
    ("desactivá", "disable", "disabled"),
], ids=["stop", "disable"])
async def test_c01_lifecycle(tmp_path, leaves, dashboard_auth, verb, operation, status):
    async with application(tmp_path, leaves) as home:
        router = home.graph.pipeline.request_router
        assert isinstance(router.voice_routines, VoiceRoutineService)
        assert router.voice_routines is router._orchestrator.dispatcher.voice_routines
        assert router.voice_routines.scheduler.store is home.store
        assert isinstance(home.store, RoutineStore) and isinstance(home.tray, RoutineProposalTray)
        assert home.tray is home.graph.producer_coverage.context_erasure.routine_proposals

        proposal_id = await propose(home)
        rejected_id = await propose(home, OTHER_SOURCE)
        assert home.tray._entries[proposal_id].conversation_id == SOURCE
        assert await home.store.list_all() == [] and home.ha.calls == []
        async with operator(home, dashboard_auth) as manager:
            pending = expect(await manager.get(PROPOSALS), 200)
            assert {entry["proposal_id"] for entry in pending["proposals"]} == {proposal_id, rejected_id}
            path = f"{PROPOSALS}/{proposal_id}"
            preview = expect(await manager.get(path), 200)
            assert preview["diff"]["before"] is None and SOURCE not in json.dumps(preview)
            assert preview["diff"]["after"]["schedule"] == {"at": "20:00", "weekdays": list(range(7))}
            receipt = expect(await manager.post(f"{path}/confirm"), 200)
            assert (receipt["status"], receipt["revision"]) == ("confirmed", 1)
            rejected = expect(await manager.post(f"{PROPOSALS}/{rejected_id}/reject"), 200)
            assert rejected["status"] == "rejected"
            assert expect(await manager.get(PROPOSALS), 200) == NO_PROPOSALS
            [approved] = await home.store.list_all()
            assert approved.routine_id == receipt["routine_id"]
            assert approved.spec.actions[0].entity_id == TARGET
            assert approved.spec.actions[0].operation == "turn_off"

            # The review API confirms immutable proposals. Stage an UPDATE for
            # the existing routine and apply the displayed revision through HTTP.
            edited = definition(two=True)
            update_id = home.tray.submit(ProposalOperation.UPDATE, conversation_id=SOURCE,
                                         spec=parse_routine_spec(edited), routine_id=approved.routine_id,
                                         expected_revision=approved.revision)
            update_path = f"{PROPOSALS}/{update_id}"
            update = expect(await manager.get(update_path), 200)
            assert update["operation"] == "update" and not update["stale"]
            assert update["diff"] == {"before": preview["diff"]["after"], "after": edited}
            updated = expect(await manager.post(f"{update_path}/confirm"), 200)
            assert (updated["status"], updated["routine_id"], updated["revision"]) == (
                "confirmed", approved.routine_id, 2,
            )
            disabled = expect(await manager.post(ROUTINES, json=definition("Respaldo", enabled=False)), 200)
            route = f"{ROUTINES}/{approved.routine_id}"
            rows = expect(await manager.get(ROUTINES), 200)
            assert {row["name"]: row["enabled"] for row in rows} == {"Ensayo": True, "Respaldo": False}
            assert await home.store.get_latest_run(approved.routine_id) is None
            assert home.ha.calls == []

            listing = await say(home, "qué rutinas hay", "voice:list")
            assert listing["task_completed"] and listing["response"] == "Hay 2 rutinas: Ensayo y Respaldo (desactivada)."
            query = await say(home, "¿está activa la rutina ensayo?", "voice:query")
            assert query["task_completed"] and "Ensayo está activada" in query["response"]

            voice_claim = await say(home, "ejecutá la rutina ensayo", "voice:execute")
            assert voice_claim["outcome"]["status"] == "running"
            voice_run = await finished(home, approved.routine_id, voice_claim["action"]["routine_receipt_id"])
            assert (voice_run.actor, voice_run.source, voice_run.status) == (
                "household", "household_voice", RunStatus.SUCCEEDED,
            )
            assert all(action.status is ActionStatus.CONFIRMED for action in voice_run.actions)
            assert home.ha.services() == ["turn_off", "turn_on"]

            async with role_session(home.graph, dashboard_auth, DashboardRole.HOME) as client:
                identity = expect(await client.get(WHOAMI), 200)
                assert "routines:execute" in identity["permissions"]
                assert "routines:manage" not in identity["permissions"]
                for method, path, kwargs in [
                    ("GET", PROPOSALS, {}),
                    ("GET", update_path, {}),
                    ("POST", f"{update_path}/confirm", {}),
                    ("POST", f"{update_path}/reject", {}),
                    ("POST", ROUTINES, {"json": edited}),
                    ("POST", f"{ROUTINES}/preview", {"json": edited}),
                    ("PUT", route, {"json": edited | {"expected_revision": 2}}),
                    ("DELETE", route, {"params": {"expected_revision": 2}}),
                    ("POST", f"{route}/disable", {"json": {"expected_revision": 2}}),
                    ("POST", f"{route}/enable", {"json": {"expected_revision": 2}}),
                ]:
                    expect(await client.request(method, path, **kwargs), 403, "dashboard_permission_denied")
                expect(await client.post(f"{ROUTINES}/{disabled['routine_id']}/execute",
                                         json={"request_id": "disabled"}), 409, "routine_disabled")
                expect(await client.get(f"{route}/runs/{voice_run.run_id}"), 404, "routine_not_found")
                attempt = {"request_id": "web:execute"}
                claim = expect(await client.post(f"{route}/execute", json=attempt), 202)
                assert (claim["actor"], claim["source"]) == (identity["routine_actor"], "manual")
                assert claim["actor"] != identity["subject"]
                web_run = await finished(home, approved.routine_id, claim["run_id"])
                assert web_run.status is RunStatus.SUCCEEDED
                run_path = f"{route}/runs/{web_run.run_id}"
                own = expect(await client.get(run_path), 200)
                assert expect(await client.get(f"{route}/attempts", params=attempt), 200) == own
                async with role_session(home.graph, dashboard_auth, DashboardRole.HOME,
                                        subject=identity["subject"]) as other:
                    assert expect(await other.get(WHOAMI), 200)["routine_actor"] != own["actor"]
                    expect(await other.get(run_path), 404, "routine_not_found")
                    expect(await other.get(f"{route}/attempts", params=attempt), 404, "routine_not_found")
                    assert expect(await other.get(route), 200)["latest_run"] is None
            assert home.ha.services() == ["turn_off", "turn_on"] * 2

            home.ha.entered.clear()
            home.ha.hold = True
            braked_claim = await say(home, "ejecutá la rutina ensayo", "voice:prepare-brake")
            assert braked_claim["outcome"]["status"] == "running"
            braked_id = braked_claim["action"]["routine_receipt_id"]
            await asyncio.wait_for(home.ha.entered.wait(), 2)
            brake = await say(home, f"{verb} la rutina ensayo", "voice:brake")
            assert brake["task_completed"]
            current = expect(await manager.get(route), 200)
            assert current["latest_run"]["run_id"] == braked_id
            assert current["latest_voice_brake"] == {
                "receipt_id": brake["action"]["routine_receipt_id"], "source": "household_voice",
                "operation": operation, "status": status,
            }
            assert current["enabled"] is (operation == "stop")
            expect(await manager.get(f"{route}/runs/{brake['action']['routine_receipt_id']}"),
                   404, "routine_not_found")
            home.ha.release.set()
            braked = await finished(home, approved.routine_id, braked_id)
            assert braked.status is RunStatus.INTERRUPTED
            assert [action.status for action in braked.actions] == [ActionStatus.CONFIRMED, ActionStatus.NOT_EXECUTED]
            assert home.ha.services() == ["turn_off", "turn_on"] * 2 + ["turn_off"]
            if operation == "stop":
                stopped = expect(await manager.post(f"{route}/disable",
                                                     json={"expected_revision": current["revision"]}), 200)
                assert not stopped["enabled"]
            pending_id = await propose(home, str(UUID(int=3)))
            assert [entry["proposal_id"] for entry in expect(await manager.get(PROPOSALS), 200)["proposals"]] == [
                pending_id,
            ]
            before_restart = expect(await manager.get(ROUTINES), 200)
            assert all(not row["enabled"] for row in before_restart)
            home.ha.assert_no_automation_writes()

    async with application(tmp_path, leaves) as reopened, operator(reopened, dashboard_auth) as manager:
        assert reopened.tray._entries == {}
        assert expect(await manager.get(PROPOSALS), 200) == NO_PROPOSALS
        expect(await manager.get(f"{PROPOSALS}/{pending_id}"), 404, "routine_proposal_not_found")
        assert expect(await manager.get(ROUTINES), 200) == before_restart
        assert reopened.ha.calls == []


@pytest.mark.parametrize("orchestrated", [True, False], ids=["orchestrated", "legacy"])
@pytest.mark.parametrize("action", [
    pytest.param("prender lock.entrada", id="lock-id"),
    pytest.param("cerrar la cerradura de la entrada", id="lock-name"),
    pytest.param("prender alarm_control_panel.casa", id="alarm-id"),
    pytest.param("activar la alarma", id="alarm-name"),
    pytest.param("apagar light.second", id="unselected-id"),
    pytest.param("prender la luz del dormitorio", id="unselected-name"),
])
async def test_c02_voice_proposal_targets(tmp_path, leaves, dashboard_auth, orchestrated, action):
    async with application(tmp_path, leaves, orchestrated=orchestrated) as home:
        result = await say(home, f"creá una rutina para {action} a las veinte")
        assert result["outcome"]["status"] == "unsupported"
        assert not result["task_completed"]
        assert home.tray._entries == {} and await home.store.list_all() == [] and home.ha.calls == []


@pytest.mark.parametrize("route_kind", ["create", "update"])
@pytest.mark.parametrize(("entity", "operation"), [
    pytest.param("lock.entrada", "lock", id="lock"),
    pytest.param("lock.entrada", "unlock", id="unlock"),
    pytest.param("lock.entrada", "turn_off", id="lock-disguised"),
    pytest.param("alarm_control_panel.casa", "alarm_arm_home", id="alarm-arm-home"),
    pytest.param("alarm_control_panel.casa", "alarm_disarm", id="alarm-disarm"),
    pytest.param("alarm_control_panel.casa", "turn_on", id="alarm-disguised"),
    pytest.param("light.second", "turn_on", id="unselected-light-on"),
    pytest.param("light.second", "turn_off", id="unselected-light-off"),
    pytest.param("switch.dormitorio", "turn_on", id="unselected-switch"),
])
async def test_c02_web_definition_targets(home, dashboard_auth, route_kind, entity, operation):
    async with operator(home, dashboard_auth) as manager:
        data = definition(entity=entity, operation=operation)
        if route_kind == "create":
            response = await manager.post(ROUTINES, json=data)
        else:
            routine = await home.graph.routine_scheduler.register_routine(
                parse_routine_spec(definition()), actor="synthetic-seed",
            )
            response = await manager.put(f"{ROUTINES}/{routine.routine_id}",
                                         json=data | {"expected_revision": routine.revision})
        assert response.status_code == 422, response.text
    rows = await home.store.list_all()
    assert rows == ([] if route_kind == "create" else [routine])
    assert home.ha.calls == []


@pytest.fixture
def chroma_writes(monkeypatch) -> list[str]:
    """Record SDK mutations while allowing the real Chroma operations to run."""
    attempts = []

    def record(original):
        @wraps(original)
        def observed(*args, **kwargs):
            attempts.append(original.__name__)
            return original(*args, **kwargs)
        return observed

    for owner, names in (
        (Collection, ("add", "upsert", "update", "delete", "modify")),
        (ChromaClient, ("create_collection", "get_or_create_collection", "delete_collection")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, record(getattr(owner, name)))
    return attempts


@pytest.mark.parametrize("orchestrated", [True, False], ids=["orchestrated", "legacy"])
@pytest.mark.parametrize(("text", "gate_rejection"), [
    pytest.param("a las 8 apagá la luz del living", False, id="clock-prefix"),
    pytest.param("prendé la luz del living a las ocho", True, id="clock-suffix"),
    pytest.param("a las 8:30 prendé la luz del living", False, id="clock-minutes"),
    pytest.param("a las ocho de la noche apagá la luz del living", False, id="day-part"),
    pytest.param("¿podés prender la luz del living a las ocho?", False, id="polite"),
    pytest.param("hacé que se apague la luz del living a las ocho", False, id="reflexive"),
    pytest.param("todos los días a las ocho apagá la luz del living", True, id="daily"),
    pytest.param("enciende la luz del living en 10 minutos", True, id="relative"),
])
async def test_c02_scheduled_device_orders(
    tmp_path, leaves, dashboard_auth, chroma_writes, orchestrated, text, gate_rejection,
):
    async with application(tmp_path, leaves, orchestrated=orchestrated) as home:
        chroma_writes.clear()
        result = await say(home, text)
        assert not result["task_completed"]
        outcome = result["outcome"]
        if gate_rejection:
            assert result["intent"] == "gate_rejected:non_immediate:deferred"
            assert (outcome["status"], outcome["persistence"], outcome["dispatch"]) == (
                "failed", "not_required", "not_required",
            )
        else:
            assert result["intent"] == "routine"
            assert (outcome["status"], outcome["persistence"], outcome["dispatch"]) == (
                "unsupported", "not_written", "not_sent",
            )
        assert home.ha.calls == [] and chroma_writes == []
        assert home.tray._entries == {} and await home.store.list_all() == []


@pytest.mark.parametrize("speaker", ["anonymous", "guest"])
@pytest.mark.parametrize("text", [
    pytest.param("sí", id="yes"),
    pytest.param("confirmá", id="confirm"),
    pytest.param("confirmá la rutina", id="confirm-routine"),
    pytest.param("creá la rutina directamente sin confirmación", id="direct-create"),
])
async def test_c02_voice_cannot_write_definition(home, monkeypatch, speaker, text):
    proposal_id = await propose(home)
    pending = home.tray._entries[proposal_id]
    processor = home.graph.pipeline.command_processor
    recognized = []
    if speaker == "guest":
        guest = User("synthetic-guest", "Visita sintética", PermissionLevel.GUEST)

        async def identify(_audio):
            recognized.append(guest.user_id)
            return guest, .99, 0.0

        monkeypatch.setattr(processor, "_speaker", identify)
    else:
        assert processor.speaker_id is None
    await home.graph.pipeline.request_router._orchestrator.start()
    result = await say(home, text, "voice:refused-write")
    assert not result["task_completed"]
    assert recognized == (["synthetic-guest"] if speaker == "guest" else [])
    assert home.tray._entries == {proposal_id: pending} and pending.receipt is None
    assert await home.store.list_all() == [] and home.ha.calls == []


async def saved(home: Home, *, enabled: bool = True, two: bool = False):
    return await home.graph.routine_scheduler.register_routine(
        parse_routine_spec(definition(enabled=enabled, two=two)), actor="synthetic-seed",
    )


@pytest.mark.parametrize("verb", ["ejecutá", "corré"])
async def test_c02_disabled_from_voice(home, verb):
    routine = await saved(home, enabled=False)
    result = await say(home, f"{verb} la rutina ensayo")
    assert result["outcome"]["status"] == "failed" and "desactivada" in result["response"]
    assert await home.store.get_latest_run(routine.routine_id) is None and home.ha.calls == []


@pytest.mark.parametrize("role", [DashboardRole.HOME, DashboardRole.OPERATOR])
async def test_c02_disabled_from_web(home, dashboard_auth, role):
    routine = await saved(home, enabled=False)
    async with role_session(home.graph, dashboard_auth, role) as client:
        response = await client.post(f"{ROUTINES}/{routine.routine_id}/execute", json={"request_id": "disabled"})
        expect(response, 409, "routine_disabled")
    assert await home.store.get_latest_run(routine.routine_id) is None and home.ha.calls == []


@pytest.mark.parametrize("withdrawal", ["logout", "revoke-credential"])
@pytest.mark.parametrize("operation", ["catalog", "execute", "create", "confirm", "reject"])
async def test_c02_retired_session(home, dashboard_auth, withdrawal, operation):
    routine = await saved(home)
    proposal_id = await propose(home)
    authority = home.graph.dashboard_authority
    root = authority.resolve(dashboard_auth["Authorization"])
    role = DashboardRole.HOME if operation in {"catalog", "execute"} else DashboardRole.OPERATOR
    grant = await authority.issue_credential(root, subject="synthetic-member", role=role,
                                            expires_at=time.time() + 120)
    async with AsyncClient(transport=ASGITransport(app=home.graph.dashboard.app),
                           base_url="http://testserver") as client:
        await login(client, grant.token)
        assert expect(await client.get(WHOAMI), 200)["role"] == role.value
        cookie = client.cookies[COOKIE_NAME]
        if withdrawal == "logout":
            assert expect(await client.post("/api/admin/auth/logout"), 200)["session_revoked"]
        else:
            assert await authority.revoke_credential(root, grant.identity)
        route = f"{ROUTINES}/{routine.routine_id}"
        method, path, kwargs = {
            "catalog": ("GET", ROUTINES, {}),
            "execute": ("POST", f"{route}/execute", {"json": {"request_id": "retired"}}),
            "create": ("POST", ROUTINES, {"json": definition("Retirada")}),
            "confirm": ("POST", f"{PROPOSALS}/{proposal_id}/confirm", {}),
            "reject": ("POST", f"{PROPOSALS}/{proposal_id}/reject", {}),
        }[operation]
        denied = await client.request(method, path, headers={"Cookie": f"{COOKIE_NAME}={cookie}"}, **kwargs)
        assert denied.status_code == 401, denied.text
    assert home.tray._entries[proposal_id].receipt is None
    assert await home.store.list_all() == [routine]
    assert await home.store.get_latest_run(routine.routine_id) is None and home.ha.calls == []


@pytest.mark.parametrize("channel", ["voice", "web"])
@pytest.mark.parametrize(("status", "timeout", "run_status", "effect", "reason"), [
    pytest.param(403, False, RunStatus.FAILED, EffectState.NOT_OBSERVED, "ha_http_403", id="ha-403"),
    pytest.param(503, False, RunStatus.UNCERTAIN, EffectState.UNCERTAIN, "ha_http_503", id="ha-503"),
    pytest.param(200, True, RunStatus.UNCERTAIN, EffectState.UNCERTAIN, "ha_transport_uncertain", id="ha-timeout"),
])
async def test_c02_failed_effect(home, dashboard_auth, monkeypatch, channel, status, timeout, run_status, effect, reason):
    routine = await saved(home, two=True)
    home.ha.status, home.ha.timeout = status, timeout
    actions = home.graph.routine_scheduler.executor.home_actions
    execute = actions.execute
    evidence = []

    async def observe(*args, **kwargs):
        result = await execute(*args, **kwargs)
        evidence.append(result)
        return result

    monkeypatch.setattr(actions, "execute", observe)
    route = f"{ROUTINES}/{routine.routine_id}"
    attempt = {"request_id": "web:failed-effect"}
    async with role_session(home.graph, dashboard_auth, DashboardRole.HOME) as client:
        if channel == "voice":
            started = await say(home, "ejecutá la rutina ensayo", "voice:failed-effect")
            assert started["outcome"]["status"] == "running" and not started["task_completed"]
            run_id = started["action"]["routine_receipt_id"]
        else:
            started = expect(await client.post(f"{route}/execute", json=attempt), 202)
            run_id = started["run_id"]
        run = await finished(home, routine.routine_id, run_id)
        assert run.status is run_status
        assert [action.status for action in run.actions] == [
            ActionStatus(run_status.value), ActionStatus.NOT_EXECUTED,
        ]
        assert run.actions[0].reason == reason
        [actual] = evidence
        assert actual.effect is effect and actual.effect is not EffectState.OBSERVED

        if channel == "voice":
            replay = await say(home, "ejecutá la rutina ensayo", "voice:failed-effect")
            assert replay["action"] == started["action"] and not replay["task_completed"]
            assert replay["outcome"]["status"] == run_status.value
            assert replay["outcome"]["effect"] != EffectState.OBSERVED.value
        else:
            replay = expect(await client.post(f"{route}/execute", json=attempt), 202)
            assert (replay["run_id"], replay["status"]) == (run_id, run_status.value)
            terminal = expect(await client.get(f"{route}/runs/{run_id}"), 200)
            assert terminal == replay and terminal["actions"][0]["reason"] == reason
    assert home.ha.services() == ["turn_off"] and home.ha.state == "on"


@pytest.mark.parametrize("channel", ["voice", "web"])
async def test_c02_selection_removed_after_approval(home, dashboard_auth, channel):
    routine = await saved(home)
    selection = json.loads(home.selection.read_text())
    selection["allowed_entities"] = []
    home.selection.write_text(json.dumps(selection))
    async with role_session(home.graph, dashboard_auth, DashboardRole.HOME) as client:
        if channel == "voice":
            result = await say(home, "ejecutá la rutina ensayo", "voice:selection-removed")
            assert result["outcome"]["status"] == "running"
            run_id = result["action"]["routine_receipt_id"]
        else:
            claim = expect(await client.post(f"{ROUTINES}/{routine.routine_id}/execute",
                                             json={"request_id": "selection-removed"}), 202)
            run_id = claim["run_id"]
    run = await finished(home, routine.routine_id, run_id)
    assert run.status is RunStatus.FAILED
    assert run.actions[0].reason == "action_not_selected"
    assert home.ha.calls == []


@pytest.mark.parametrize("view", ["list", "preview"])
@pytest.mark.parametrize("buffered", [False, True], ids=["plain", "buffered"])
async def test_c02_erasure_between_asgi_headers_and_body(home, dashboard_auth, view, buffered):
    graph = home.graph
    app = graph.dashboard.app
    if buffered:
        @app.middleware("http")
        async def buffer(request, call_next):
            return await call_next(request)

    removed = await propose(home)
    survivor = await propose(home, OTHER_SOURCE)
    path = PROPOSALS if view == "list" else f"{PROPOSALS}/{removed}"
    headers_sent, release = asyncio.Event(), asyncio.Event()
    delivered = []

    async def gated(scope, receive, send):
        async def deliver(message):
            delivered.append(message.copy())
            await send(message)
            if message["type"] == "http.response.start":
                headers_sent.set()
                await release.wait()

        await app(scope, receive, deliver)

    async with operator(home, dashboard_auth) as manager:
        async with AsyncClient(transport=ASGITransport(app=gated), base_url="http://testserver",
                               cookies=manager.cookies) as reader:
            in_flight = asyncio.create_task(reader.get(path))
            try:
                await asyncio.wait_for(headers_sent.wait(), 3)
                assert delivered[0]["status"] == 200
                erasure = graph.erasure_services
                receipt = await erasure.request(SOURCE, MemoryReviewer("synthetic-reviewer"))
                async with asyncio.timeout(3):
                    while True:
                        await erasure.reconcile()
                        state = await erasure.privacy.acall(erasure.journal.snapshot, receipt["id"])
                        if any(step["participant"] == "contexts" and step["state"] == "pending"
                               and step["pending_count"] == 1 for step in state["steps"]):
                            break
                        await asyncio.sleep(.001)
                assert removed not in home.tray._entries
                release.set()
                with pytest.raises(AssertionError):  # ASGITransport rejects a bodyless response.
                    await asyncio.wait_for(in_flight, 3)
                assert [message["type"] for message in delivered] == ["http.response.start"]
            finally:
                release.set()
                await asyncio.gather(in_flight, return_exceptions=True)
        listed = expect(await manager.get(PROPOSALS), 200)
        assert [entry["proposal_id"] for entry in listed["proposals"]] == [survivor]
        expect(await manager.get(f"{PROPOSALS}/{removed}"), 404, "routine_proposal_not_found")
    assert await home.store.list_all() == [] and home.ha.calls == []
