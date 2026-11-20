"""Saved-routine voice control through the real router, SQLite and Casa transport."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
import sqlite3
import threading
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient
import pytest

from src.core.task_outcome import CURRENT_REQUEST_ID
from src.main import build_application
from src.pipeline.command_event import CapturedTranscript, TranscriptSource
from src.routines.models import ActionStatus, RunStatus, parse_routine_spec
from src.stt.whisper_fast import STTResult
from tests.integration.test_capability_application import configuration
from tests.integration.test_global_capture_http import URL as CAPTURE_URL, login, settled
from tests.integration.test_global_capture_http import leaves as capture_leaves
from tests.integration.test_routine_voice_factory import OWN_WAKE, WITHOUT_A_CLEAR_OWN_WAKE, captured
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio
leaves = capture_leaves


@asynccontextmanager
async def application(tmp_path, leaves, *, brake=True, orchestrated=True):
    transport = TaskTransport()
    async with transport.serve() as url:
        config, selection = configuration(tmp_path, url)
        config["routines"]["voice_brake"] = brake
        config["orchestrator"]["enabled"] = orchestrated
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            try:
                yield SimpleNamespace(graph=graph, selection=selection, transport=transport,
                                      scheduler=graph.routine_scheduler)
            finally:
                transport.release.set()


@pytest.fixture
async def home(tmp_path, leaves, dashboard_auth):
    async with application(tmp_path, leaves) as home:
        yield home


async def save(home, name="Mañana", *, enabled=True, two=False):
    actions = [{"entity_id": "light.living", "operation": "turn_on"}]
    if two:
        actions.append({"entity_id": "light.living", "operation": "turn_off"})
    return await home.scheduler.register_routine(parse_routine_spec({
        "name": name, "description": "", "enabled": enabled, "timezone": "America/Montevideo",
        "schedule": None, "conditions": [], "actions": actions,
    }), actor="synthetic-operator")


async def say(home, text, request_id="voice:same-turn", **capture):
    home.graph.models.stt._model.recognize = lambda *_args, **_kwargs: text
    return await asyncio.wait_for(home.graph.pipeline.process_command(
        captured(**(capture or OWN_WAKE)), request_id=request_id,
    ), 2)


async def complete(home, routine_id):
    async with asyncio.timeout(3):
        while True:
            run = await home.scheduler.get_latest_run(routine_id)
            if run is not None and run.status is not RunStatus.RUNNING:
                return run
            await asyncio.sleep(.001)


async def test_claim_replies_while_ha_is_pending_and_same_turn_never_runs_twice(home):
    routine = await save(home, "Máñána")
    home.transport.hold = True
    try:
        first = await say(home, "ejecutá la rutina MAÑANA")
        assert first["response"] == "Inicié la rutina Máñána."
        assert first["outcome"]["status"] == "running"
        assert first["outcome"]["persistence"] == "committed"
        assert not first["task_completed"] and first["latency_ms"] < 300
        assert await asyncio.wait_for(home.transport.entered.wait(), 2)
        live = await home.scheduler.get_latest_run(routine.routine_id)
        assert (live.actor, live.source) == ("household", "household_voice")
        assert json.loads(live.occurrence_key)[-1] == first["request_id"]
        repeated = await say(home, "ejecutá la rutina mañana")
        assert repeated["action"] == first["action"] == {"routine_receipt_id": live.run_id}
        assert repeated["outcome"]["status"] == "running"
        other = await say(home, "ejecutá la rutina mañana", "voice:another-turn")
        assert other["outcome"]["status"] == "failed" and "ya se está ejecutando" in other["response"]
    finally:
        home.transport.release.set()
    run = await complete(home, routine.routine_id)
    replay = await say(home, "ejecutá la rutina mañana")
    assert replay["outcome"]["status"] == "completed" and replay["action"] == first["action"]
    assert run.run_id == live.run_id and len(home.transport.sends) == 1


async def test_dispatch_requires_turn_id_and_acoustic_identity_never_becomes_actor(home):
    routine = await save(home)
    dispatcher = home.graph.pipeline.request_router._orchestrator.dispatcher
    args = dict(user_id="synthetic-recognized-admin", text="ejecutá la rutina mañana")
    rejected = await dispatcher.dispatch(**args, routine_guard=lambda: True)
    assert rejected.outcome.status.value == "cancelled"
    token = CURRENT_REQUEST_ID.set("voice:dispatcher")
    try:
        rejected = await dispatcher.dispatch(**args)
        assert rejected.outcome.status.value == "cancelled"
        assert await home.scheduler.get_latest_run(routine.routine_id) is None
        accepted = await dispatcher.dispatch(**args, routine_guard=lambda: True)
        assert accepted.outcome.status.value == "running"
    finally:
        CURRENT_REQUEST_ID.reset(token)
    run = await complete(home, routine.routine_id)
    assert (run.actor, run.status, len(home.transport.sends)) == ("household", RunStatus.SUCCEEDED, 1)


async def test_complete_name_preserves_enye_articles_spacing_and_punctuation(home):
    target = await save(home, "La Mañána-Norte")
    for wrong in ("mañana", "la manana-norte", "mañana-norte", "la mañana norte",
                  "la mañana-nort", "la  mañana-norte", "la mañana-norte."):
        result = await say(home, "ejecutá la rutina " + wrong)
        assert result["response"].startswith("No encontré"), wrong
    assert await home.scheduler.execute_by_name("La Mañána-Norte") is None
    assert await home.scheduler.execute_by_name("mañana", {"request_id": "exact-only"}) is None
    assert await home.scheduler.get_latest_run(target.routine_id) is None
    result = await say(home, "ejecutá la rutina LA MAÑANA-NORTE")
    assert result["outcome"]["status"] == "running"
    assert (await complete(home, target.routine_id)).status is RunStatus.SUCCEEDED
    assert len(home.transport.sends) == 1


async def test_collision_lists_all_candidates_and_disabled_targets_do_not_run(home):
    first = await save(home, "Ríego")
    second = await save(home, "RIEGO", enabled=False)
    disabled = await save(home, "Noche", enabled=False)
    ambiguous = await say(home, "ejecutá la rutina riego")
    assert ambiguous["outcome"]["status"] == "needs_input"
    assert "Ríego y RIEGO" in ambiguous["response"]
    response = await say(home, "ejecutá la rutina noche")
    assert response["outcome"]["status"] == "failed" and "desactivada" in response["response"]
    for routine in (first, second, disabled):
        assert await home.scheduler.get_latest_run(routine.routine_id) is None
    assert home.transport.sends == []


@pytest.mark.parametrize(("verb", "enabled", "status"), [
    ("pará", True, "stop_requested"), ("desactivá", False, "disabled"),
])
async def test_brake_cuts_the_next_send_and_projects_separate_web_receipt(home, dashboard_auth, verb, enabled, status):
    routine = await save(home, two=True)
    home.transport.hold = True
    try:
        started = await say(home, "ejecutá la rutina mañana")
        assert started["outcome"]["status"] == "running"
        await asyncio.wait_for(home.transport.entered.wait(), 2)
        brake = await say(home, f"{verb} la rutina mañana", "voice:brake")
        repeated = await say(home, f"{verb} la rutina mañana", "voice:brake")
        assert brake["outcome"]["status"] == "completed" and brake["action"] == repeated["action"]
        async with AsyncClient(transport=ASGITransport(app=home.graph.dashboard.app),
                               base_url="http://testserver", headers=dashboard_auth) as client:
            path = f"/api/routines/{routine.routine_id}"
            response = await client.get(path)
            assert response.status_code == 200
            view = response.json()
            assert view["enabled"] is enabled
            assert view["revision"] == routine.revision + (not enabled)
            assert view["latest_run"]["run_id"] == started["action"]["routine_receipt_id"]
            assert view["latest_run"]["status"] == "running"
            assert view["latest_voice_brake"] == {
                "receipt_id": brake["action"]["routine_receipt_id"], "source": "household_voice",
                "operation": "stop" if enabled else "disable", "status": status,
            }
            assert (await client.get(path + "/runs/" + view["latest_voice_brake"]["receipt_id"])).status_code == 404
            assert (await client.get(
                f"/api/routines/voice-brake:{routine.routine_id}/runs/{view['latest_voice_brake']['receipt_id']}"
            )).status_code == 404
    finally:
        home.transport.release.set()
    run = await complete(home, routine.routine_id)
    assert run.status is RunStatus.INTERRUPTED
    assert [action.status for action in run.actions] == [ActionStatus.CONFIRMED, ActionStatus.NOT_EXECUTED]
    assert [send[1] for send in home.transport.sends] == ["turn_on"]
    if not enabled:
        refused = await say(home, "ejecutá la rutina mañana", "voice:disabled")
        assert refused["outcome"]["status"] == "failed"
        assert await home.scheduler.get_latest_run(routine.routine_id) == run


async def test_voice_brake_flag_off_preserves_web_start_and_records_no_brake(tmp_path, leaves, dashboard_auth):
    async with application(tmp_path, leaves, brake=False) as home:
        routine = await save(home)
        for verb in ("pará", "desactivá"):
            refused = await say(home, f"{verb} la rutina mañana")
            assert refused["outcome"]["status"] == "unsupported"
        assert await home.scheduler.get_routine(routine.routine_id) == routine
        assert await home.scheduler.get_latest_brake(routine.routine_id) is None
        assert await home.scheduler.get_latest_run(routine.routine_id) is None
        async with AsyncClient(transport=ASGITransport(app=home.graph.dashboard.app),
                               base_url="http://testserver", headers=dashboard_auth) as client:
            result = await client.post(f"/api/routines/{routine.routine_id}/execute", json={"request_id": "web"})
            assert result.status_code == 202 and result.json()["source"] == "manual"
        assert (await complete(home, routine.routine_id)).status is RunStatus.SUCCEEDED


@pytest.mark.parametrize("capture", list(WITHOUT_A_CLEAR_OWN_WAKE) + ["ambient_transcript"])
async def test_weak_or_ambient_voice_never_changes_routines_or_sends(home, capture):
    routine = await save(home)
    fields = WITHOUT_A_CLEAR_OWN_WAKE.get(capture)
    if fields is None:
        event = captured(**OWN_WAKE)
        event.transcript = CapturedTranscript.from_audio("desactivá la rutina mañana", TranscriptSource.AMBIENT, event.audio)
        result = await home.graph.pipeline.process_command(event)
    else:
        result = await say(home, "desactivá la rutina mañana", **fields)
    assert result["intent"] == "gate_rejected:routine_capture" and not result["response"]
    assert await home.scheduler.get_routine(routine.routine_id) == routine
    assert await home.scheduler.get_latest_brake(routine.routine_id) is None
    assert await home.scheduler.get_latest_run(routine.routine_id) is None
    assert home.transport.sends == []


@pytest.mark.parametrize("orchestrated", [False, True])
async def test_legacy_and_compound_cannot_escape_to_a_producer(tmp_path, leaves, orchestrated):
    async with application(tmp_path, leaves, orchestrated=orchestrated) as home:
        routine = await save(home, "Riego")
        if not orchestrated:
            for verb in ("ejecutá", "pará", "desactivá"):
                denied = await say(home, f"{verb} la rutina riego")
                assert denied["intent"] == "gate_rejected:legacy_routine_control"
        denied = await say(home, "ejecutá rutina riego y apagá luz")
        assert denied["intent"] == ("compound" if orchestrated else "gate_rejected:legacy_routine_control")
        assert await home.scheduler.get_routine(routine.routine_id) == routine
        assert await home.scheduler.get_latest_run(routine.routine_id) is None
        assert await home.scheduler.get_latest_brake(routine.routine_id) is None
        assert home.transport.sends == []


async def test_command_and_stt_confidence_gates_run_before_any_claim(home, monkeypatch):
    routine = await save(home)
    deferred = await save(home, "que apaga la luz en 10 minutos")
    invalid = await say(home, "ejecutá la rutina que apaga la luz en 10 minutos")
    assert invalid["intent"].startswith("gate_rejected:non_immediate")
    monkeypatch.setattr(type(home.graph.models.stt), "transcribe_with_confidence",
                        lambda *_: STTResult("ejecutá la rutina mañana", 1, avg_logprob=-2.0, no_speech_prob=.8))
    weak = await say(home, "ejecutá la rutina mañana")
    assert weak["intent"].startswith("gate_rejected:")
    assert await home.scheduler.get_latest_run(routine.routine_id) is None
    assert await home.scheduler.get_latest_run(deferred.routine_id) is None
    assert home.transport.sends == []


@pytest.mark.parametrize("verb", ["ejecutá", "pará", "desactivá"])
async def test_routine_controls_fail_closed_on_command_gate_error(home, monkeypatch, verb):
    routine = await save(home)
    gate = home.graph.pipeline.request_router.command_gate
    text = f"{verb} la rutina mañana"

    def fail_evaluation(*_args):
        raise RuntimeError("Synthetic command gate failure")

    monkeypatch.setattr(gate, "_evaluate", fail_evaluation)
    fallback = gate.evaluate(text)
    assert fallback.accept and fallback.reason == "gate_error"

    result = await say(home, text)
    assert await home.scheduler.get_routine(routine.routine_id) == routine
    assert await home.scheduler.get_latest_run(routine.routine_id) is None
    assert await home.scheduler.get_latest_brake(routine.routine_id) is None
    assert home.transport.sends == []
    assert result["intent"] == "gate_rejected:routine_capture" and not result["response"]


@pytest.mark.parametrize("verb", ["ejecutá", "desactivá"])
async def test_total_mute_retires_authority_before_native_commit(home, dashboard_auth, monkeypatch, verb):
    routine = await save(home)
    store = home.scheduler.store
    original = store._work.write
    entered, release = threading.Event(), threading.Event()

    async def held_write(operation, *, guard=None, publish=None):
        def provisional(db):
            result = operation(db)
            entered.set()
            assert release.wait(3)
            return result

        return await original(provisional, guard=guard, publish=publish)

    monkeypatch.setattr(store._work, "write", held_write)
    pending = asyncio.create_task(say(home, f"{verb} la rutina mañana"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        async with AsyncClient(transport=ASGITransport(app=home.graph.dashboard.app),
                               base_url="http://testserver") as client:
            await login(client, dashboard_auth["Authorization"].split(" ", 1)[1])
            revision = (await client.get(CAPTURE_URL)).json()["revision"]
            assert (await client.post(CAPTURE_URL, json={"muted": True, "expected_revision": revision})).status_code == 202
            assert not home.graph.capture_control.capture_allowed
            release.set()
            assert (await pending)["outcome"]["status"] == "cancelled"
            await settled(client, "muted")
            while_muted = await say(home, "ejecutá la rutina mañana", "voice:muted")
            assert while_muted["outcome"]["status"] == "cancelled" and not while_muted["response"]
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
    assert await home.scheduler.get_routine(routine.routine_id) == routine
    assert await home.scheduler.get_latest_run(routine.routine_id) is None
    assert await home.scheduler.get_latest_brake(routine.routine_id) is None
    assert home.transport.sends == []


async def test_uncertain_ha_receipt_is_never_resent_on_same_turn(home):
    routine = await save(home, two=True)
    home.transport.mode = "uncertain"
    first = await say(home, "ejecutá la rutina mañana")
    run = await complete(home, routine.routine_id)
    assert run.status is RunStatus.UNCERTAIN
    assert [action.status for action in run.actions] == [ActionStatus.UNCERTAIN, ActionStatus.NOT_EXECUTED]
    replay = await say(home, "ejecutá la rutina mañana")
    assert replay["outcome"]["status"] == "uncertain" and replay["action"] == first["action"]
    assert len(home.transport.sends) == 1


async def test_shutdown_and_restart_preserve_uncertainty_and_do_not_replay(home):
    routine = await save(home, two=True)
    home.transport.hold = True
    try:
        first = await say(home, "ejecutá la rutina mañana")
        assert first["outcome"]["status"] == "running"
        await asyncio.wait_for(home.transport.entered.wait(), 2)
        await home.scheduler.stop()
    finally:
        home.transport.release.set()
    await home.scheduler.start()
    resumed = await say(home, "ejecutá la rutina mañana")
    run = await complete(home, routine.routine_id)
    assert resumed["outcome"]["status"] == "uncertain" and resumed["action"] == first["action"]
    assert run.status is RunStatus.INTERRUPTED
    assert [action.status for action in run.actions] == [ActionStatus.UNCERTAIN, ActionStatus.NOT_EXECUTED]
    assert len(home.transport.sends) == 1


async def test_each_action_checks_current_casa_selection_and_security_stays_denied(home, dashboard_auth):
    for text in ("ejecutá la rutina Abrí la cerradura", "ejecutá la rutina Activá la alarma"):
        denied = await say(home, text)
        assert not denied["task_completed"] and denied["response"].startswith("No encontré")
    async with AsyncClient(transport=ASGITransport(app=home.graph.dashboard.app),
                           base_url="http://testserver", headers=dashboard_auth) as client:
        for entity_id, operation in (("lock.front", "unlock"), ("alarm_control_panel.home", "alarm_arm_home")):
            response = await client.post("/api/routines", json={
                "name": "Operación restringida", "description": "", "enabled": True, "timezone": "America/Montevideo",
                "schedule": None, "conditions": [], "actions": [{"entity_id": entity_id, "operation": operation}],
            })
            assert response.status_code == 422
    routine = await save(home, two=True)
    home.transport.hold = True
    try:
        started = await say(home, "ejecutá la rutina mañana")
        assert started["outcome"]["status"] == "running"
        await asyncio.wait_for(home.transport.entered.wait(), 2)
        selection = json.loads(home.selection.read_text())
        selection["allowed_entities"][0]["operations"].remove("turn_off")
        home.selection.write_text(json.dumps(selection))
    finally:
        home.transport.release.set()
    run = await complete(home, routine.routine_id)
    assert run.status is RunStatus.FAILED
    assert [action.status for action in run.actions] == [ActionStatus.CONFIRMED, ActionStatus.FAILED]
    assert [send[1] for send in home.transport.sends] == ["turn_on"]


async def test_idle_brake_keeps_schema_and_execution_history_separate(home):
    routine = await save(home)
    with sqlite3.connect(home.scheduler.store.db_path) as db:
        schema = db.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    idle = await say(home, "pará la rutina mañana")
    assert idle["outcome"]["status"] == "completed" and "No quedaban" in idle["response"]
    with sqlite3.connect(home.scheduler.store.db_path) as db:
        assert db.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == schema
        assert db.execute("SELECT run_id FROM routine_runs WHERE routine_id = ?", (routine.routine_id,)).fetchone() is None
    assert await home.scheduler.get_latest_run(routine.routine_id) is None
    brake = await home.scheduler.get_latest_brake(routine.routine_id)
    assert brake.receipt_id == idle["action"]["routine_receipt_id"]
