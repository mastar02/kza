"""Local voice proposals through capture admission, web review and erasure."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
import pytest

from src.main import build_application
from src.memory.fact_ledger import MemoryReviewer
from src.monitoring.loop_lag import LoopLagConfig, LoopLagProbe
from tests.integration.test_capability_application import configuration
from tests.integration.test_global_capture_http import URL as CAPTURE_URL, login, settled
from tests.integration.test_routine_voice_execute import application, leaves, save, say
from tests.integration.test_routine_voice_factory import OWN_WAKE
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio
PROPOSALS = "/api/routines/proposals"
PROPOSE = "creá una rutina para apagar la luz del living a las veinte todos los días"
REPLY = "Dejé la propuesta para confirmar en la web."
SPEAKER = "synthetic-recognized-person"


@pytest.fixture
async def home(tmp_path, leaves, dashboard_auth):
    async with application(tmp_path, leaves) as home:
        yield home


@pytest.mark.parametrize("orchestrated", [True, False])
async def test_valid_proposal_uses_current_reviewed_selection_and_no_external_producer(
    tmp_path, leaves, dashboard_auth, monkeypatch, orchestrated,
):
    async with application(tmp_path, leaves, orchestrated=orchestrated) as home:
        router = home.graph.pipeline.request_router
        assert router.llm is None and router.routines.llm is None
        veto = AsyncMock(side_effect=AssertionError("proposal reached the model veto"))
        monkeypatch.setattr(router, "_household_veto_passes", veto)
        list_entities = AsyncMock(wraps=router.ha.get_domotics_entities)
        monkeypatch.setattr(router.ha, "get_domotics_entities", list_entities)
        if orchestrated:
            slow = AsyncMock(side_effect=AssertionError("proposal reached the reasoner"))
            monkeypatch.setattr(router._orchestrator.dispatcher, "_slow_path", slow)

        reads = home.transport.state_reads
        source = str(uuid4())
        result = await say(home, PROPOSE, **OWN_WAKE, conversation_id=source)
        assert result["response"] == REPLY
        assert result["task_completed"] and result["outcome"]["persistence"] == "volatile"
        assert home.transport.state_reads == reads and home.transport.entity_reads == []
        assert home.transport.sends == [] and await home.scheduler.get_all_routines() == []
        veto.assert_not_awaited()
        list_entities.assert_not_awaited()
        if orchestrated:
            slow.assert_not_awaited()

        tray = home.graph.dashboard.routine_proposals
        [entry] = tray._entries.values()
        assert entry.conversation_id == source
        assert entry.spec.schedule.at == "20:00" and entry.spec.schedule.weekdays == tuple(range(7))
        assert [(action.entity_id, action.operation) for action in entry.spec.actions] == [("light.living", "turn_off")]

        async with AsyncClient(transport=ASGITransport(app=home.graph.dashboard.app), base_url="http://test") as client:
            await login(client, dashboard_auth["Authorization"].split(" ", 1)[1])
            response = await client.get(PROPOSALS)
            assert response.status_code == 200
            [listed] = response.json()["proposals"]
            assert (listed["proposal_id"], listed["operation"], listed["source"]) == (
                entry.proposal_id, "create", "household_voice",
            )
            assert "conversation_id" not in response.text and source not in response.text


async def test_voice_affirmations_leave_proposal_and_saved_routines_intact(home, dashboard_auth):
    saved = await save(home)
    assert (await say(home, PROPOSE))["response"] == REPLY
    tray = home.graph.dashboard.routine_proposals
    [entry] = tray._entries.values()
    router = home.graph.pipeline.request_router
    await router._orchestrator.start()
    proposal_before = asdict(entry)
    for text in ("sí", "confirmá", "confirmá la rutina"):
        await say(home, text, f"voice:{uuid4()}")
        assert await home.scheduler.get_all_routines() == [saved]
        assert list(tray._entries) == [entry.proposal_id]
        assert asdict(tray._entries[entry.proposal_id]) == proposal_before
        assert router.routines.pending_routine is None and not router.routines.awaiting_confirmation
        assert home.transport.sends == []

    async with AsyncClient(transport=ASGITransport(app=home.graph.dashboard.app), base_url="http://test") as client:
        await login(client, dashboard_auth["Authorization"].split(" ", 1)[1])
        response = await client.post(f"{PROPOSALS}/{entry.proposal_id}/confirm")
        assert response.status_code == 200 and response.json()["status"] == "confirmed"
    assert len(await home.scheduler.get_all_routines()) == 2
    assert home.transport.sends == []


@pytest.mark.parametrize("orchestrated", [True, False])
@pytest.mark.parametrize("missing", ["dashboard", "scheduler"])
async def test_voice_proposals_require_working_web_review(
    tmp_path, leaves, dashboard_auth, orchestrated, missing,
):
    transport = TaskTransport()
    async with transport.serve() as url, AsyncExitStack() as stack:
        config, _ = configuration(tmp_path, url)
        config["orchestrator"]["enabled"] = orchestrated
        if missing == "dashboard":
            config["dashboard"]["enabled"] = False
        else:
            # SQLite initialization fails against a directory, through the real factory.
            config["routines"]["db_path"] = str(tmp_path)
        graph = await build_application(config, shutdown=stack, factories=leaves)
        assert (graph.dashboard is None) == (missing == "dashboard")
        assert (graph.routine_scheduler is None) == (missing == "scheduler")

        result = await say(SimpleNamespace(graph=graph), PROPOSE)
        assert result["response"] == "Ahora no está disponible la revisión de rutinas en la web."
        assert result["outcome"]["status"] == "unsupported" and not result["task_completed"]
        assert result["outcome"]["persistence"] == "not_written"
        assert graph.producer_coverage.context_erasure.routine_proposals._entries == {}
        assert transport.sends == []


@pytest.mark.parametrize(("review_action", "decision", "spoken"), [
    ("confirm", "confirmed", "confirmada"), ("reject", "rejected", "rechazada"),
])
async def test_replaying_a_reviewed_proposal_reports_its_terminal_decision(
    home, dashboard_auth, review_action, decision, spoken,
):
    capture = OWN_WAKE | {"conversation_id": str(uuid4())}
    assert (await say(home, PROPOSE, "voice:first", **capture))["response"] == REPLY
    tray = home.graph.dashboard.routine_proposals
    [proposal_id] = tray._entries
    assert (await say(home, PROPOSE, "voice:pending-replay", **capture))["response"] == REPLY
    assert list(tray._entries) == [proposal_id]
    async with AsyncClient(transport=ASGITransport(app=home.graph.dashboard.app), base_url="http://test") as client:
        await login(client, dashboard_auth["Authorization"].split(" ", 1)[1])
        reviewed = await client.post(f"{PROPOSALS}/{proposal_id}/{review_action}")
        assert reviewed.status_code == 200 and reviewed.json()["status"] == decision
        saved = await home.scheduler.get_all_routines()
        assert len(saved) == (1 if decision == "confirmed" else 0)

        result = await say(home, PROPOSE, "voice:terminal-replay", **capture)
        assert result["response"] == f"Esa propuesta ya fue {spoken} en la web."
        assert not result["task_completed"] and result["outcome"]["persistence"] == "not_written"
        pending = await client.get(PROPOSALS)
        assert pending.status_code == 200 and pending.json()["proposals"] == []
    assert list(tray._entries) == [proposal_id]
    assert await home.scheduler.get_all_routines() == saved and home.transport.sends == []


async def test_voice_proposal_has_no_person_and_is_erased_with_its_conversation(home, monkeypatch):
    source, survivor = str(uuid4()), str(uuid4())
    assert (await say(home, PROPOSE, **OWN_WAKE, conversation_id=survivor))["response"] == REPLY
    graph = home.graph
    router = graph.pipeline.request_router
    captured_sources = []
    process = router._orchestrator.process

    async def identified(**kwargs):
        captured_sources.extend(kwargs["memory_sources"])
        return await process(**(kwargs | {"user_id": SPEAKER}))

    monkeypatch.setattr(router._orchestrator, "process", identified)
    result = await say(home, PROPOSE, **OWN_WAKE, conversation_id=source)
    assert result["response"] == REPLY
    assert [item.conversation_id for item in captured_sources] == [source]
    tray = graph.dashboard.routine_proposals
    [entry] = [item for item in tray._entries.values() if item.conversation_id == source]
    payload = json.dumps(asdict(entry), default=str)
    assert all(value not in payload for value in (SPEAKER, "user_id", "person_id", "created_by", PROPOSE))
    await graph.producer_coverage.context_erasure.retention.begin_close()
    erasure = graph.erasure_services
    await erasure.request(source, MemoryReviewer("synthetic-reviewer"))
    async with asyncio.timeout(3):
        while entry.proposal_id in tray._entries:
            await erasure.reconcile()
            await asyncio.sleep(.001)
    assert [item.conversation_id for item in tray._entries.values()] == [survivor]
    assert await home.scheduler.get_all_routines() == []


@pytest.mark.parametrize("action", [
    "prender la luz del dormitorio",
    "cerrar la cerradura de la entrada",
    "activar la alarma",
    "prender lock.entrada",
    "prender alarm_control_panel.casa",
])
async def test_unsupported_targets_leave_no_proposal(home, action):
    reads = home.transport.state_reads
    result = await say(home, f"creá una rutina para {action} a las veinte")
    assert result["outcome"]["status"] == "unsupported"
    assert home.graph.dashboard.routine_proposals._entries == {}
    assert home.transport.state_reads == reads and home.transport.sends == []


async def test_ambiguous_time_requires_the_full_request_again(home):
    result = await say(home, "creá una rutina para apagar la luz del living a las ocho")
    assert result["outcome"]["status"] == "needs_input" and "¿A qué hora exacta?" in result["response"]
    await home.graph.pipeline.request_router._orchestrator.start()
    await say(home, "de la noche", "voice:new-turn")
    assert home.graph.dashboard.routine_proposals._entries == {}
    result = await say(home, "creá una rutina para apagar la luz del living a las ocho de la noche")
    assert result["response"] == REPLY
    [entry] = home.graph.dashboard.routine_proposals._entries.values()
    assert entry.spec.schedule.at == "20:00"


async def test_clock_and_entity_punctuation_survives_classification(home):
    result = await say(home, "creá una rutina para prender light.living a las 08:30 de lunes a viernes")
    assert result["response"] == REPLY
    [entry] = home.graph.dashboard.routine_proposals._entries.values()
    assert entry.spec.schedule.at == "08:30" and entry.spec.schedule.weekdays == (0, 1, 2, 3, 4)
    assert entry.spec.actions[0].entity_id == "light.living"


async def test_selection_io_keeps_the_event_loop_responsive(home, monkeypatch):
    policy = home.graph.pipeline.request_router.home_actions.policy
    read = policy._read_selection
    threads = []

    def delayed_read():
        threads.append(threading.current_thread())
        time.sleep(.25)
        return read()

    monkeypatch.setattr(policy, "_read_selection", delayed_read)
    probe = LoopLagProbe(LoopLagConfig(interval_s=.02))
    probe.start()
    try:
        assert (await say(home, PROPOSE))["response"] == REPLY
    finally:
        await probe.stop()
    assert threads and all(worker is not threading.main_thread() for worker in threads)
    assert probe.snapshot()["max_ms"] < 200, probe.snapshot()


async def test_total_mute_prevents_voice_proposals(home, dashboard_auth):
    async with AsyncClient(transport=ASGITransport(app=home.graph.dashboard.app), base_url="http://test") as client:
        await login(client, dashboard_auth["Authorization"].split(" ", 1)[1])
        revision = (await client.get(CAPTURE_URL)).json()["revision"]
        response = await client.post(CAPTURE_URL, json={"muted": True, "expected_revision": revision})
        assert response.status_code == 202
        await settled(client, "muted")
        result = await say(home, PROPOSE)
    assert not result["success"] and result["response"] != REPLY
    assert home.graph.dashboard.routine_proposals._entries == {}
    assert await home.scheduler.get_all_routines() == [] and home.transport.sends == []


@pytest.mark.parametrize("orchestrated", [True, False])
async def test_total_mute_retires_an_inflight_proposal_before_submission(
    tmp_path, leaves, dashboard_auth, monkeypatch, orchestrated,
):
    async with application(tmp_path, leaves, orchestrated=orchestrated) as home:
        actions = home.graph.pipeline.request_router.home_actions
        validate = actions.validate_selection
        checked, release = asyncio.Event(), asyncio.Event()

        async def hold_validation(*args, **kwargs):
            await validate(*args, **kwargs)
            checked.set()
            await release.wait()

        monkeypatch.setattr(actions, "validate_selection", hold_validation)
        async with AsyncClient(transport=ASGITransport(app=home.graph.dashboard.app), base_url="http://test") as client:
            await login(client, dashboard_auth["Authorization"].split(" ", 1)[1])
            revision = (await client.get(CAPTURE_URL)).json()["revision"]
            pending = asyncio.create_task(say(home, PROPOSE))
            try:
                await asyncio.wait_for(checked.wait(), 2)
                response = await client.post(CAPTURE_URL, json={"muted": True, "expected_revision": revision})
                assert response.status_code == 202 and not home.graph.capture_control.capture_allowed
            finally:
                release.set()
                result = await pending
            await settled(client, "muted")
        assert not result["success"] and result["response"] == ""
        assert result["outcome"]["status"] == "cancelled" and not result["response_required"]
        assert result["outcome"]["response_requirement"] == "silent"
        assert home.graph.dashboard.routine_proposals._entries == {}
