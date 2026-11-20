"""T035-U4: visitor voice and web sessions through the real application graph."""

import asyncio
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

from httpx import ASGITransport, AsyncClient
import numpy as np
import pytest

from src.application import ApplicationFactories, ApplicationGraph
from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import DashboardRole
from src.lists.list_store import OwnerType
from src.main import build_application
from src.pipeline.command_event import CommandEvent
from src.reminders.reminder_store import ReminderScope, ReminderState
from src.routines.models import ActionStatus, RunStatus
from src.timers.named_timers import TimerState
from src.users.user_manager import PermissionLevel, User
from tests.integration.test_capability_application import configuration
from tests.integration.test_global_capture_http import login
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS
from tests.mocks.audio_output import AudioBackend
from tests.mocks.task_transport import TaskTransport

SESSION_HINT = "necesitás iniciar sesión en la web"
PERSONAL_LIST_REPLY = "No encontré esa lista compartida. Para consultar listas personales " + SESSION_HINT + "."
PRIVATE_ITEM = "MARCADOR_ARTICULO_PRIVADO_721"
PRIVATE_REMINDER_MARKER = "MARCADOR_RECORDATORIO_PRIVADO_354"
PRIVATE_REMINDER = f"trámite lunar {PRIVATE_REMINDER_MARKER}"
PRIVATE_ACCOUNT = "MARCADOR_CUENTA_PRIVADA_963"
SHARED_ITEM = "pan de la casa"
SHARED_REMINDER = "regar las plantas"
SHARED_LIST_CATALOG = "Las listas compartidas son: compras"
ROUTINE = {
    "name": "Mañana", "description": "", "enabled": True, "timezone": "America/Montevideo",
    "schedule": None, "conditions": [],
    "actions": [{"entity_id": "light.living", "operation": operation} for operation in ("turn_on", "turn_off")],
}
PATHS = {"list": "fast_list", "reminder": "fast_reminder", "timer": "fast_timer", "routine": "fast_routine"}


@dataclass(frozen=True)
class Probe:
    """One spoken form and its expected household receipt and reply."""

    text: str
    intent: str
    status: str
    reply: str


AMBIGUOUS = (
    Probe("cancelá el timer", "timer", "needs_input", "¿Cuál cancelo?"),
    Probe("pará el timer", "timer", "needs_input", "¿Cuál cancelo?"),
    Probe("pausá el temporizador", "timer", "needs_input", "¿Cuál pauso?"),
    Probe("borrá el temporizador", "timer", "needs_input", "¿Cuál cancelo?"),
    Probe("sacá el timer", "timer", "needs_input", "¿Cuál cancelo?"),
    Probe("cancelá el recordatorio", "reminder", "needs_input", "¿cuál?"),
    Probe("cancelá los recordatorios", "reminder", "needs_input", "¿cuál?"),
    Probe("cancelá la lista", "list", "needs_input", "¿Querés borrar la lista o vaciarla?"),
    Probe("cancelá la lista de compras", "list", "needs_input", "¿Querés borrar la lista o vaciarla?"),
    Probe("frená la lista", "list", "needs_input", "Eso no se puede hacer con una lista."),
)

# "Mis" is also colloquial for household tasks. These reads must still return
# only the shared catalog, even with a live personal session in a browser.
POSSESSIVES = (
    Probe("cuáles son mis listas", "list", "completed", SHARED_LIST_CATALOG),
    Probe("mostrame mis listas", "list", "completed", SHARED_LIST_CATALOG),
    Probe("decime mis listas", "list", "completed", SHARED_LIST_CATALOG),
    Probe("listame mis listas", "list", "completed", SHARED_LIST_CATALOG),
    Probe("lee mis listas", "list", "completed", SHARED_LIST_CATALOG),
    Probe("mostrame mis recordatorios", "reminder", "completed", SHARED_REMINDER),
    Probe("decime mis recordatorios", "reminder", "completed", SHARED_REMINDER),
    Probe("listame mis recordatorios", "reminder", "completed", SHARED_REMINDER),
    Probe("leeme mis recordatorios", "reminder", "completed", SHARED_REMINDER),
    Probe("contame mis recordatorios", "reminder", "completed", SHARED_REMINDER),
)

NAMES = ("Ana", "Bruno", "Carla", "Diego", "Elena", "Fabio", "Gala", "Hugo", "Iris", "Julia")
OTHER_PERSON = tuple(
    probe
    for name in NAMES
    for probe in (
        Probe(f"lee la lista de {name}", "list", "needs_input", PERSONAL_LIST_REPLY),
        Probe(f"recordale a {name} revisar el correo", "reminder", "unsupported", SESSION_HINT),
    )
)

# Repeat identical requests with and without acoustic metadata, a live or
# retired personal session, and a new graph on the same stores.
SCOPE_PROBES = (
    Probe("lee la lista de Ana", "list", "needs_input", PERSONAL_LIST_REPLY),
    Probe("lee la lista de cuentas personales", "list", "needs_input", PERSONAL_LIST_REPLY),
    Probe("lee la lista de horizonte", "list", "needs_input", PERSONAL_LIST_REPLY),
    Probe("lee la lista de compras", "list", "completed", SHARED_ITEM),
    Probe("mostrame mis listas", "list", "completed", SHARED_LIST_CATALOG),
    Probe("qué recordatorios tengo", "reminder", "completed", SHARED_REMINDER),
    Probe("cancelá el recordatorio de trámite lunar", "reminder", "needs_input", "No encontré un recordatorio"),
    Probe("crea una lista de cuentas", "list", "needs_input", "Indicá que querés crear una lista compartida"),
    Probe("crea una rutina de visitas", "routine", "unsupported", "Todavía no creo rutinas por voz"),
    Probe("qué rutinas hay", "routine", "completed", "Mañana"),
)

NEGATIVES = {
    "ambiguas": tuple((suffix, probe) for suffix in ("", ", por favor") for probe in AMBIGUOUS),
    "posesivos": tuple((suffix, probe) for suffix in ("", ", por favor") for probe in POSSESSIVES),
    "otra_persona": tuple(("", probe) for probe in OTHER_PERSON),
    "id_acustico": tuple((identity, probe) for identity in ("owner", "forged") for probe in SCOPE_PROBES),
    "sesion_retirada": tuple((mode, probe) for mode in ("logout", "revoke") for probe in SCOPE_PROBES),
    "reinicio": tuple((identity, probe) for identity in ("unknown", "owner") for probe in SCOPE_PROBES),
}


def test_negative_matrix_has_coverage_by_category() -> None:
    """Guard the number of distinct spoken inputs and authority scenarios."""
    rows = [(category, variant, probe.text) for category, cases in NEGATIVES.items() for variant, probe in cases]
    assert len(set(rows)) == len(rows) >= 100
    assert Counter(category for category, _, _ in rows) == {
        "ambiguas": 20, "posesivos": 20, "otra_persona": 20,
        "id_acustico": 20, "sesion_retirada": 20, "reinicio": 20,
    }


@dataclass
class VoiceHarness:
    """Drive mic captures and the real web application within one graph lifetime."""

    graph: ApplicationGraph
    stt: OfflineSTT
    tts: OfflineTTS
    client: AsyncClient
    acoustic_user: User | None = None
    acoustic_receipts: list[str | None] = field(default_factory=list)

    async def speaker(self, _audio: np.ndarray) -> tuple[User, float, float] | None:
        """Simulate only the acoustic match, before the real router sees it."""
        user = self.acoustic_user
        self.acoustic_receipts.append(user.user_id if user else None)
        return (user, 0.99, 0.0) if user else None

    async def say(self, text: str, *, user: User | None = None) -> dict:
        self.stt.text = text
        self.acoustic_user = user
        before = len(self.acoustic_receipts)
        result = await self.graph.pipeline.process_command(CommandEvent(
            np.full(16000, .1, dtype=np.float32), room_id="", capture_complete=True,
            capture_trigger="wake", wake_score=.9,
        ))
        assert self.acoustic_receipts[before:] == [user.user_id if user else None]
        assert result["user"] is None, (text, result)
        return result


@pytest.fixture
def offline_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AudioBackend:
    import src.tts.piper_tts as output
    import src.vectordb.chroma_sync as index

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(index, "SentenceTransformer", lambda *args, **kwargs: OfflineEncoder())
    backend = AudioBackend()
    monkeypatch.setattr(output, "sd", backend)
    return backend


@asynccontextmanager
async def voice_graph(config: dict, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[VoiceHarness]:
    stt, tts = OfflineSTT(), OfflineTTS()
    leaves = ApplicationFactories(stt=lambda _: stt, tts=lambda _: tts, preflight=lambda: None)
    async with AsyncExitStack() as stack:
        graph = await build_application(config, shutdown=stack, factories=leaves)
        client = await stack.enter_async_context(AsyncClient(
            transport=ASGITransport(app=graph.dashboard.app), base_url="http://testserver",
        ))
        voice = VoiceHarness(graph, stt, tts, client)
        with monkeypatch.context() as patches:
            patches.setattr(graph.pipeline.command_processor, "_speaker", voice.speaker)
            await graph.pipeline.request_router._orchestrator.start()
            await graph.voice_services.features.start()
            yield voice


def assert_receipt(result: dict, intent: str, status: str, reply: str) -> None:
    assert (result["intent"], result["path"], result["outcome"]["status"]) == (intent, PATHS[intent], status), result
    assert result["task_completed"] is (status == "completed"), result
    assert reply in result["response"], result


async def say_ok(voice: VoiceHarness, text: str, intent: str, reply: str,
                 *, persistence: str | None = None) -> dict:
    result = await voice.say(text)
    assert_receipt(result, intent, "completed", reply)
    if persistence is not None:
        assert result["outcome"]["persistence"] == persistence, result
    return result


async def visit(voice: VoiceHarness, dashboard_auth: dict[str, str]) -> None:
    authority = voice.graph.dashboard_authority
    visitor = await authority.issue_credential(authority.resolve(dashboard_auth["Authorization"]),
        subject="synthetic-visitor", role=DashboardRole.HOME, expires_at=time.time() + 600)
    await login(voice.client, visitor.token)
    for endpoint in ("/api/lists", "/api/reminders"):
        assert (await voice.client.get(endpoint + "?scope=shared")).status_code == 200
        denied = await voice.client.get(endpoint + "?scope=personal")
        assert denied.status_code == 403 and denied.json()["detail"] == "personal_session_required"
    assert (await voice.client.post("/api/routines", json=ROUTINE)).status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("require_known", [False, True])
async def test_visitor_shared_task_lifecycle(tmp_path: Path, offline_audio: AudioBackend,
                                            monkeypatch: pytest.MonkeyPatch, dashboard_auth: dict[str, str],
                                            require_known: bool) -> None:
    """Lists, reminders and routines persist while timers have a volatile receipt."""
    transport = TaskTransport()
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        # This legacy switch can reject unknown speakers in the ordinary HA
        # fallback. The household task paths have their own policy and must
        # keep working for visitors even when that switch is on.
        config["security"] = {"require_known_speaker_for_actions": require_known}
        config["routines"]["voice_brake"] = True
        async with voice_graph(config, monkeypatch) as voice:
            await visit(voice, dashboard_auth)
            await say_ok(voice, "crea una lista compartida de compras", "list", "Creé la lista", persistence="committed")
            await say_ok(voice, "agrega pan a la lista de compras", "list", "Agregué pan", persistence="committed")
            await say_ok(voice, "lee la lista de compras", "list", "pan")
            await say_ok(voice, "crea una lista compartida de viajes", "list", "Creé la lista", persistence="committed")
            await say_ok(voice, "borra la lista de viajes", "list", "Borré la lista viajes", persistence="committed")
            lists = await voice.graph.list_manager.get_all_lists(None, shared_only=True)
            assert [(row.name, row.owner_type, row.owner_id) for row in lists] == [("compras", OwnerType.SHARED, None)]
            assert [row["name"] for row in (await voice.client.get("/api/lists")).json()] == ["compras"]

            for text in (SHARED_REMINDER, "ordenar el patio"):
                await say_ok(voice, f"recordame en 20 minutos {text}", "reminder", text, persistence="committed")
            await say_ok(voice, "qué recordatorios tengo", "reminder", SHARED_REMINDER)
            active = await voice.graph.reminder_manager.get_active(None, shared_only=True)
            assert {(row.user_id, row.scope, row.text) for row in active} == {
                ("household", ReminderScope.SHARED, SHARED_REMINDER),
                ("household", ReminderScope.SHARED, "ordenar el patio"),
            }
            cancelled = next(row.id for row in active if row.text == "ordenar el patio")
            await say_ok(voice, "cancelá el recordatorio de ordenar el patio", "reminder",
                         "Cancelé el recordatorio", persistence="committed")

            for name in ("pasta", "salsa"):
                await say_ok(voice, f"poné un timer de 20 minutos para la {name}", "timer", name, persistence="volatile")
            await say_ok(voice, "qué timers hay", "timer", "pasta")
            await say_ok(voice, "cancelá el timer de la pasta", "timer", "Cancelé el temporizador", persistence="volatile")
            timers = voice.graph.voice_services.features.timer_manager
            assert {t.name: (t.user_id, t.state) for t in timers.get_all_timers()} == {
                "pasta": ("household", TimerState.CANCELLED), "salsa": ("household", TimerState.RUNNING),
            }

            # T033-U2 supports execution and braking. Creation is web-managed
            # at this base, so ask by voice and check that only the admin can save.
            refused = await voice.say("crea una rutina de visitas")
            assert_receipt(refused, "routine", "unsupported", "Podés crearlas desde la web")
            assert await voice.graph.routine_scheduler.get_all_routines() == []
            created = await voice.client.post("/api/routines", json=ROUTINE, headers=dashboard_auth)
            assert created.status_code == 200, created.text
            routine_id = created.json()["routine_id"]
            await say_ok(voice, "qué rutinas hay", "routine", "Mañana")
            scheduler = voice.graph.routine_scheduler
            transport.hold = True
            try:
                started = await voice.say("ejecutá la rutina mañana")
                assert_receipt(started, "routine", "running", "Inicié la rutina Mañana.")
                assert started["outcome"]["persistence"] == "committed"
                assert await asyncio.wait_for(transport.entered.wait(), 2)
                running = await scheduler.get_latest_run(routine_id)
                assert running is not None and running.status is RunStatus.RUNNING
                assert started["action"] == {"routine_receipt_id": running.run_id}
                await say_ok(voice, "pará la rutina mañana", "routine", "Frené las acciones pendientes",
                             persistence="committed")
            finally:
                transport.release.set()
            async with asyncio.timeout(3):
                while (run := await scheduler.get_latest_run(routine_id)).status is RunStatus.RUNNING:
                    await asyncio.sleep(.001)
            assert (run.actor, run.source, run.status) == ("household", "household_voice", RunStatus.INTERRUPTED)
            assert [action.status for action in run.actions] == [ActionStatus.CONFIRMED, ActionStatus.NOT_EXECUTED]
            brake = await scheduler.get_latest_brake(routine_id)
            assert brake is not None and brake.status.value == "stop_requested"

        async with voice_graph(config, monkeypatch) as restarted:
            await say_ok(restarted, "lee la lista de compras", "list", "pan")
            lists = await restarted.graph.list_manager.get_all_lists(None, shared_only=True)
            assert [row.name for row in lists] == ["compras"]
            reminders = await say_ok(restarted, "qué recordatorios tengo", "reminder", SHARED_REMINDER)
            assert "ordenar el patio" not in reminders["response"]
            removed = await restarted.graph.reminder_manager._store.get_by_id(cancelled)
            assert removed is not None and removed.state is ReminderState.CANCELLED
            await say_ok(restarted, "qué timers hay", "timer", "No hay temporizadores activos.")
            assert restarted.graph.voice_services.features.timer_manager.get_all_timers() == []
            await say_ok(restarted, "qué rutinas hay", "routine", "Mañana")
            scheduler = restarted.graph.routine_scheduler
            assert await scheduler.get_latest_run(routine_id) == run
            assert await scheduler.get_latest_brake(routine_id) == brake
        assert [send[1] for send in transport.sends] == ["turn_on"]
        assert offline_audio.heard().size > 0


@dataclass(frozen=True)
class PrivateScope:
    """Synthetic personal data that is actually readable using the owner's session."""

    account: User
    list_id: str
    reminder_id: str

    @property
    def forbidden(self) -> tuple[str, ...]:
        # The request can mention a name or the words "trámite lunar", never
        # the private item, full reminder, owner id or stored receipt ids.
        return (PRIVATE_ITEM, PRIVATE_REMINDER_MARKER, PRIVATE_ACCOUNT, self.account.user_id,
                self.list_id, self.reminder_id)

    @property
    def paths(self) -> tuple[str, ...]:
        return "/api/lists?scope=personal", f"/api/lists/{self.list_id}/items", "/api/reminders?scope=personal"


async def seed_matrix(voice: VoiceHarness, dashboard_auth: dict[str, str]) -> PrivateScope:
    graph = voice.graph
    account, _ = await graph.voice_registry.acall(
        graph.pipeline.command_processor.user_manager.add_user, PRIVATE_ACCOUNT, PermissionLevel.ADULT,
    )
    assert account is not None
    lists = graph.list_manager._store
    private = await lists.create_list("compras", OwnerType.USER, account.user_id)
    await lists.add_item(private.id, PRIVATE_ITEM, added_by=account.user_id)
    for name in ("cuentas personales", *(name.lower() for name in NAMES)):
        private_list = await lists.create_list(name, OwnerType.USER, account.user_id)
        await lists.add_item(private_list.id, PRIVATE_ITEM, added_by=account.user_id)
    shared = await lists.create_list("compras", OwnerType.SHARED, None)
    await lists.add_item(shared.id, SHARED_ITEM)
    reminders = graph.reminder_manager._store
    trigger = time.time() + 1200
    private_reminder = await reminders.create(account.user_id, PRIVATE_REMINDER, trigger)
    for label in (SHARED_REMINDER, "ordenar el patio"):
        await reminders.create("household", label, trigger, scope=ReminderScope.SHARED)
    created = await voice.client.post("/api/routines", json=ROUTINE, headers=dashboard_auth)
    assert created.status_code == 200, created.text
    for name in ("pasta", "salsa"):
        await say_ok(voice, f"poné un timer de 20 minutos para la {name}", "timer", name, persistence="volatile")
    return PrivateScope(account, private.id, private_reminder.id)


async def snapshot(graph: ApplicationGraph, scope: PrivateScope) -> tuple:
    """Capture real native state, including private and shared rows."""
    lists = graph.list_manager._store
    catalog = sorted(await lists.get_lists_for_user(scope.account.user_id), key=lambda row: row.id)
    items = {row.id: await lists.get_items(row.id) for row in catalog}
    reminders = graph.reminder_manager._store
    scheduler = graph.routine_scheduler
    routines = await scheduler.get_all_routines()
    return (
        catalog, items,
        sorted(await reminders.get_active_for_user(scope.account.user_id, include_shared=True), key=lambda row: row.id),
        routines, [await scheduler.get_latest_run(row.routine_id) for row in routines],
        [await scheduler.get_latest_brake(row.routine_id) for row in routines],
        [(t.timer_id, t.user_id, t.state) for t in graph.voice_services.features.timer_manager.get_all_timers()],
    )


async def owner_session(voice: VoiceHarness, auth: dict[str, str], scope: PrivateScope, subject: str) -> tuple[str, str]:
    authority = voice.graph.dashboard_authority
    issued = await authority.issue_credential(authority.resolve(auth["Authorization"]), subject=subject,
        role=DashboardRole.HOME, user_id=scope.account.user_id, expires_at=time.time() + 600)
    await login(voice.client, issued.token)
    for path, marker in zip(scope.paths, (scope.list_id, PRIVATE_ITEM, PRIVATE_REMINDER), strict=True):
        visible = await voice.client.get(path)
        assert visible.status_code == 200 and marker in visible.text, visible.text
    return issued.identity, voice.client.cookies[COOKIE_NAME]


async def assert_cookie_retired(voice: VoiceHarness, scope: PrivateScope, cookie: str) -> None:
    assert voice.graph.dashboard_authority.resolve(None, cookie) is None
    voice.client.cookies.clear()
    for path in scope.paths:
        response = await voice.client.get(path, headers={"Cookie": f"{COOKIE_NAME}={cookie}"})
        assert response.status_code == 401, response.text
        assert all(marker not in response.text for marker in scope.forbidden)


async def run_probes(voice: VoiceHarness, scope: PrivateScope, category: str, transport: TaskTransport,
                     *, variant: str | None = None) -> None:
    before = await snapshot(voice.graph, scope)
    authority = voice.graph.dashboard_authority
    grants = (dict(authority._credentials), dict(authority._sessions))
    tts_start = len(voice.tts.texts)
    # The same id as a real account arrives with a forged admin classification,
    # alongside an unrelated speaker using the household actor's identifier.
    speakers = {
        "owner": User(scope.account.user_id, PRIVATE_ACCOUNT, PermissionLevel.ADMIN),
        "forged": User("household", PRIVATE_ACCOUNT, PermissionLevel.ADMIN),
    }
    for scenario, probe in NEGATIVES[category]:
        if variant is not None and scenario != variant:
            continue
        text = probe.text + scenario if category in {"posesivos", "ambiguas"} else probe.text
        speaker = speakers["owner"] if category == "sesion_retirada" else speakers.get(scenario)
        result = await voice.say(text, user=speaker)
        assert_receipt(result, probe.intent, probe.status, probe.reply)
        assert result["outcome"]["persistence"] in {"not_written", "not_required"}, (text, result)
        assert result["action"] is None, (text, result)
        visible = json.dumps(result, ensure_ascii=False)
        assert all(marker not in visible for marker in scope.forbidden), text
        if probe.reply in {PERSONAL_LIST_REPLY, SHARED_LIST_CATALOG}:
            assert result["response"] == probe.reply, (text, result)
    spoken = voice.tts.texts[tts_start:]
    assert spoken and all(marker not in " ".join(spoken) for marker in scope.forbidden)
    assert await snapshot(voice.graph, scope) == before, category
    assert (authority._credentials, authority._sessions) == grants
    context = voice.graph.pipeline.request_router._orchestrator.context_manager
    assert all(context.get(user_id) is None for user_id in (scope.account.user_id, "household", "unknown"))
    household = context.get_household("")
    assert household.ephemeral and household.permission_level == 0 and household.preferences == {}
    assert transport.sends == []


@pytest.mark.asyncio
async def test_negative_visitor_matrix_with_live_retired_and_restarted_sessions(
    tmp_path: Path, offline_audio: AudioBackend, monkeypatch: pytest.MonkeyPatch, dashboard_auth: dict[str, str],
) -> None:
    """A spoken identity never selects the personal scope of a real browser session."""
    transport = TaskTransport()
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        async with voice_graph(config, monkeypatch) as voice:
            scope = await seed_matrix(voice, dashboard_auth)
            _, logged_out = await owner_session(voice, dashboard_auth, scope, "synthetic-owner-logout")
            for category in ("ambiguas", "posesivos", "otra_persona", "id_acustico"):
                await run_probes(voice, scope, category, transport)
            assert (await voice.client.post("/api/admin/auth/logout")).json()["session_revoked"]
            await assert_cookie_retired(voice, scope, logged_out)
            await run_probes(voice, scope, "sesion_retirada", transport, variant="logout")

            grant_id, revoked = await owner_session(voice, dashboard_auth, scope, "synthetic-owner-revoke")
            authority = voice.graph.dashboard_authority
            assert await authority.revoke_credential(authority.resolve(dashboard_auth["Authorization"]), grant_id)
            await assert_cookie_retired(voice, scope, revoked)
            await run_probes(voice, scope, "sesion_retirada", transport, variant="revoke")
            persisted = (await snapshot(voice.graph, scope))[:-1]

        async with voice_graph(config, monkeypatch) as restarted:
            assert (await snapshot(restarted.graph, scope)) == (*persisted, [])
            await assert_cookie_retired(restarted, scope, logged_out)
            await assert_cookie_retired(restarted, scope, revoked)
            await run_probes(restarted, scope, "reinicio", transport)
    assert offline_audio.heard().size > 0
