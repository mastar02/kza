"""Real producer/action/policy graphs with synthetic private selection and HA."""

import asyncio
from dataclasses import replace
import json
import hashlib
import shutil
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.dashboard.api import DashboardAPI
from src.home_assistant.control_policy import ActionOrigin, ActorKind, HomeActor, HomeControlPolicy
from src.home_assistant.ha_client import HomeAssistantClient, ServiceCallOutcome, ServiceCallStatus
from src.home_assistant.home_actions import HomeActionConflict, HomeActionRequest, HomeActionService
from src.hooks import HookRegistry, RewriteResult
from src.orchestrator.dispatcher import MultiUserOrchestrator, RequestDispatcher
from src.pipeline.command_processor import ProcessedCommand
from src.pipeline.request_router import RequestRouter
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_scheduler import ReminderScheduler
from src.reminders.reminder_store import ReminderStore
from src.routines.factory import build_routine_scheduler
from src.routines.models import parse_routine_spec, RoutineValidationError
from src.routines.routine_executor import RoutineExecutor
from src.routines.routine_manager import RoutineManager, Routine
from src.training.command_learner import CommandLearner, CustomCommand


class ObservedHA:
    """Local observation double; confirmation uses the real HA polling method."""

    state_confirmation_timeout_s = 0.03

    def __init__(self):
        self.states = {"light.test_lamp": {"entity_id": "light.test_lamp", "state": "off", "attributes": {"supported_color_modes": ["brightness"]}}}
        self.sends = []
        self.reads = []
        self.observe_effect = True
        self.outcome = ServiceCallOutcome(ServiceCallStatus.ACCEPTED, None)

    def is_entity_available(self, entity_id):
        return None  # Cache is not an authorization decision.

    async def get_entity_state_fresh(self, entity_id, **kwargs):
        self.reads.append(entity_id)
        return self.states.get(entity_id)

    async def call_service_once(self, domain, service, entity_id, data):
        self.sends.append((domain, service, entity_id, data))
        if self.observe_effect and self.outcome.status is ServiceCallStatus.ACCEPTED:
            self.states[entity_id]["state"] = "off" if service == "turn_off" else "on"
            if data:
                self.states[entity_id]["attributes"]["brightness"] = round(data["brightness_pct"] * 255 / 100)
        return self.outcome

    async def confirm_entity_state(self, *args, **kwargs):
        return await HomeAssistantClient.confirm_entity_state(self, *args, **kwargs)


def write_selection(path, *, enabled=True, operations=None, maximum=100):
    review = {
        "selection_approved": True, "live_api_verified": enabled,
        "allowed_entities": [{"entity_id": "light.test_lamp", "live_verified": True,
            "operations": operations or ["turn_on", "turn_off", "brightness"],
            "parameters": {"brightness_pct": {"min": 0, "max": maximum}}}],
    }
    path.write_text(json.dumps(review))
    path.chmod(0o600)


@pytest.fixture
def graph(tmp_path):
    path = tmp_path / "selection.json"
    write_selection(path)
    ha = ObservedHA()
    actions = HomeActionService(ha, policy=HomeControlPolicy(selection_path=str(path)))
    return ha, actions, path


def voice_request(request_id="test", **kwargs):
    return HomeActionRequest("light.test_lamp", "turn_on", request_id, origin=ActionOrigin.VOICE, **kwargs)


async def test_default_selection_denies_before_ha_observation():
    ha = ObservedHA()
    result = await HomeActionService(ha).execute(voice_request(), actor=HomeActor.household())
    assert (result.status, result.reason) == ("failed", "action_not_selected")
    assert ha.sends == ha.reads == []


@pytest.mark.parametrize("state", ["missing", "malformed", "not_live", "unselected", "revoked_operation", "invalid_boolean", "missing_bounds"])
async def test_private_selection_fails_closed_without_reusing_a_previous_grant(graph, state):
    ha, actions, path = graph
    assert (await actions.execute(voice_request("before"), actor=HomeActor.household())).status == "confirmed"
    ha.sends.clear()
    ha.reads.clear()
    if state == "missing":
        path.unlink()
    elif state == "malformed":
        path.write_text("{")
    elif state == "not_live":
        write_selection(path, enabled=False)
    elif state == "revoked_operation":
        write_selection(path, operations=["turn_off"])
    else:
        data = json.loads(path.read_text())
        if state == "unselected":
            data["allowed_entities"] = []
        elif state == "missing_bounds":
            data["allowed_entities"][0]["parameters"] = {}
        else:
            data["live_api_verified"] = "true"
        path.write_text(json.dumps(data))
    assert (await actions.execute(voice_request("after"), actor=HomeActor.household())).status == "failed"
    assert ha.sends == ha.reads == []


async def test_grant_is_rechecked_after_ha_read_and_entity_lane_wait(graph):
    ha, actions, path = graph
    entered, release = asyncio.Event(), asyncio.Event()
    original = ha.get_entity_state_fresh

    async def fresh(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    ha.get_entity_state_fresh = fresh
    first = asyncio.create_task(actions.execute(voice_request("first"), actor=HomeActor.household()))
    await entered.wait()
    second = asyncio.create_task(actions.execute(voice_request("second"), actor=HomeActor.household()))
    await asyncio.sleep(0)
    write_selection(path, enabled=False)
    release.set()
    assert [result.status for result in await asyncio.gather(first, second)] == ["failed", "failed"]
    assert ha.sends == []


@pytest.mark.parametrize("kind", ["string_actor", "no_scope", "http_from_household", "voice_from_dashboard"])
async def test_actor_cannot_be_invented_from_a_speaker_name_or_wrong_producer(graph, kind):
    ha, actions, _ = graph
    actor = HomeActor.household()
    request = voice_request()
    if kind == "string_actor":
        with pytest.raises(ValueError, match="invalid_actor"):
            await actions.execute(request, actor="suggested-speaker-name")
        return
    if kind == "no_scope":
        actor = HomeActor(ActorKind.HOUSEHOLD, "household", frozenset())
    elif kind == "http_from_household":
        request = replace(request, origin=ActionOrigin.HTTP)
    else:
        actor = HomeActor(ActorKind.DASHBOARD, "dashboard-admin", frozenset({"home:control"}))
    assert (await actions.execute(request, actor=actor)).status == "failed"
    assert ha.sends == ha.reads == []


async def test_brightness_receipts_preserve_body_types_and_share_a_single_send(graph):
    ha, actions, _ = graph
    request = replace(voice_request(), operation="brightness", brightness_pct=20)
    results = await asyncio.gather(*(actions.execute(request, actor=HomeActor.household()) for _ in range(3)))
    assert all(result.status == "confirmed" for result in results)
    assert ha.sends == [("light", "turn_on", "light.test_lamp", {"brightness_pct": 20})]
    with pytest.raises(HomeActionConflict):
        await actions.execute(replace(request, brightness_pct=20.0), actor=HomeActor.household())


@pytest.mark.parametrize("case", ["limit", "group", "unavailable", "unconfirmed", "uncertain_transport"])
async def test_selected_target_still_requires_safe_parameters_and_observed_effect(graph, case):
    ha, actions, path = graph
    request = voice_request()
    if case == "limit":
        write_selection(path, maximum=20)
        request = replace(request, operation="brightness", brightness_pct=80)
    elif case == "group":
        ha.states[request.entity_id]["attributes"]["entity_id"] = ["lock.hidden"]
    elif case == "unavailable":
        ha.states[request.entity_id]["state"] = "unavailable"
    elif case == "unconfirmed":
        ha.observe_effect = False
    else:
        ha.outcome = ServiceCallOutcome(ServiceCallStatus.UNCERTAIN, "transport_lost")
    result = await actions.execute(request, actor=HomeActor.household())
    if case in {"unconfirmed", "uncertain_transport"}:
        assert result.status == "uncertain"
        assert len(ha.sends) == 1
    else:
        assert result.status == "failed"
        assert ha.sends == []


@pytest.mark.parametrize("hooks_enabled", [False, True])
@pytest.mark.parametrize("command", [
    {"domain": "lock", "service": "unlock", "entity_id": "lock.test"},
    {"domain": "alarm_control_panel", "service": "alarm_disarm", "entity_id": "alarm_control_panel.test"},
    {"domain": "scene", "service": "turn_on", "entity_id": "scene.opaque"},
    {"domain": "light", "service": "turn_on", "entity_id": "light.unselected"},
    {"domain": "light", "service": "turn_on", "entity_id": "light.test_lamp", "data": {"effect": "unreviewed"}},
])
async def test_dispatcher_cannot_bypass_policy_even_without_hooks(graph, command, hooks_enabled):
    ha, actions, _ = graph
    dispatcher = RequestDispatcher(MagicMock(), ha, MagicMock(), home_actions=actions, hooks=HookRegistry() if hooks_enabled else None)
    assert await dispatcher._fire_and_reconcile_ha(command) is False
    assert ha.sends == []


@pytest.mark.parametrize("rewrite", ["target", "parameters", "limit"])
async def test_hook_rewrite_is_checked_on_final_target_and_parameters(graph, rewrite):
    ha, actions, path = graph
    write_selection(path, maximum=20)
    hooks = HookRegistry()

    def change(call):
        if rewrite == "target":
            modified = replace(call, entity_id="lock.hidden", domain="lock", service="unlock")
        elif rewrite == "parameters":
            modified = call.with_data(target={"entity_id": "lock.hidden"})
        else:
            modified = call.with_data(brightness_pct=80)
        return RewriteResult(modified=modified, rule_name="test_rewrite")

    hooks.register_before("before_ha_action", change, priority=10)
    dispatcher = RequestDispatcher(MagicMock(), ha, MagicMock(), home_actions=actions, hooks=hooks)
    assert not await dispatcher._fire_and_reconcile_ha({"domain": "light", "service": "turn_on", "entity_id": "light.test_lamp", "user_id": "dashboard-admin"})
    assert ha.sends == []


async def test_orchestrator_preserves_shared_service_and_household_actor(graph):
    ha, actions, _ = graph
    orchestrator = MultiUserOrchestrator(MagicMock(), ha, MagicMock(), home_actions=actions)
    assert orchestrator.dispatcher.home_actions is actions
    assert await orchestrator.dispatcher._fire_and_reconcile_ha({"domain": "light", "service": "turn_on", "entity_id": "light.test_lamp", "user_id": "unverified-profile"})
    assert len(ha.sends) == 1


@pytest.mark.parametrize("selected", [True, False])
async def test_legacy_router_uses_policy_and_confirmed_receipt(graph, selected):
    ha, actions, path = graph
    write_selection(path, enabled=selected)
    processor = MagicMock()
    processor.process_command = AsyncMock(return_value=ProcessedCommand(text="nexa prende la luz", user=None, emotion=None, timings={}, success=True))
    chroma = MagicMock()
    chroma.search_command.return_value = {"domain": "light", "service": "turn_on", "entity_id": "light.test_lamp", "description": "luz de prueba", "similarity": 0.95}
    routines = MagicMock()
    routines.handle = AsyncMock(return_value={"handled": False})
    router = RequestRouter(processor, MagicMock(), MagicMock(), orchestrator_enabled=False, chroma_sync=chroma, ha_client=ha, home_actions=actions, routine_manager=routines)
    result = await router.process_command(np.zeros(16000, dtype=np.float32))
    assert result["success"] is selected
    assert len(ha.sends) == int(selected)


def test_authenticated_http_uses_same_policy_and_scopes_receipt(graph, monkeypatch):
    ha, actions, path = graph
    monkeypatch.setenv("KZA_DASHBOARD_TOKEN", "test-session")
    with TestClient(DashboardAPI(home_enabled=True, home_actions=actions).app) as client:
        headers = {"Authorization": "Bearer test-session"}
        body = {"entity_id": "light.test_lamp", "operation": "turn_on", "request_id": "http-one"}
        assert client.post("/api/home/actions", headers=headers, json=body).json()["status"] == "confirmed"
        assert client.get("/api/home/actions", headers=headers, params={"request_id": "http-one"}).json()["status"] == "confirmed"
        write_selection(path, enabled=False)
        body["request_id"] = "http-two"
        assert client.post("/api/home/actions", headers=headers, json=body).json()["status"] == "failed"
        body["actor"] = "dashboard-admin"
        assert client.post("/api/home/actions", headers=headers, json=body).status_code == 422
    assert len(ha.sends) == 1


def routine_spec():
    return parse_routine_spec({"name": "test", "description": "", "enabled": True, "timezone": "America/Montevideo", "schedule": None, "conditions": [], "actions": [{"entity_id": "light.test_lamp", "operation": "turn_on"}]})


async def test_persistent_routine_rechecks_revocation_between_creation_and_execution(graph, tmp_path):
    ha, actions, path = graph
    scheduler = await build_routine_scheduler(db_path=str(tmp_path / "routines.db"), timezone="America/Montevideo", ha_client=ha, home_actions=actions)
    try:
        routine = await scheduler.register_routine(routine_spec(), actor="dashboard-admin")
        write_selection(path, enabled=False)
        with pytest.raises(RoutineValidationError):
            await scheduler.register_routine(routine_spec(), actor="dashboard-admin")
        run = await scheduler.execute_by_id(routine.routine_id, actor="dashboard-admin", request_id="scheduled-test")
        await asyncio.gather(*scheduler._workers.values())
        result = await scheduler.get_run(routine.routine_id, run.run_id)
        assert result.status == "failed"
        assert ha.sends == []
        disabled = await scheduler.disable_routine(routine.routine_id, expected_revision=routine.revision, actor="dashboard-admin")
        assert disabled.spec.enabled is False
    finally:
        await scheduler.stop()
        await scheduler.store.close()


async def test_legacy_routine_executor_rejects_hidden_macro(graph):
    ha, actions, _ = graph
    executor = RoutineExecutor(ha_client=ha, home_actions=actions)
    results = await executor.execute_actions([{"type": "ha_scene", "scene_id": "opaque"}, {"type": "ha_script", "script_id": "opaque"}])
    assert all(not result.success for result in results)
    assert ha.sends == []


async def test_legacy_routine_manager_cannot_install_ha_owned_automation():
    ha = MagicMock()
    chroma = MagicMock()
    manager = RoutineManager(ha, chroma, MagicMock())
    routine = MagicMock(spec=Routine)
    assert manager._create_in_ha(routine)[0] is False
    assert (await manager._handle_delete("test", {}))["success"] is False
    ha.create_automation.assert_not_called()
    ha.delete_automation.assert_not_called()
    chroma.delete_routine.assert_not_called()


async def test_learned_command_cannot_dispatch_an_unselected_effect(graph, tmp_path):
    ha, actions, _ = graph
    learner = CommandLearner(MagicMock(), ha, MagicMock(), str(tmp_path / "custom.json"), home_actions=actions)
    learner._commands["test"] = CustomCommand("test", ["test"], "test", [{"domain": "lock", "service": "unlock", "entity_id": "lock.hidden"}], 1.0)
    assert not await learner.execute_custom_command("test")
    assert ha.sends == []


@pytest.mark.parametrize("delivery", ["due", "arrival"])
async def test_reminders_validate_at_creation_and_again_at_both_delivery_paths(graph, tmp_path, delivery):
    ha, actions, path = graph
    store = ReminderStore(str(tmp_path / "reminders.db"))
    await store.initialize()
    try:
        manager = ReminderManager(store, home_actions=actions)
        action = {"domain": "light", "service": "turn_on", "entity_id": "light.test_lamp"}
        reminder = await manager.create("unverified-profile", "test", 1.0, ha_actions=[action])
        presence = MagicMock()
        presence.get_user_zone.return_value = "test-zone"
        scheduler = ReminderScheduler(store, AsyncMock(), presence, ha, home_actions=actions)
        write_selection(path, enabled=False)
        with pytest.raises(ValueError, match="action_not_selected"):
            await manager.create("profile", "denied", 2.0, ha_actions=[action])
        if delivery == "due":
            await scheduler._fire_reminder(reminder)
        else:
            await scheduler.missed_reminder_on_arrival(reminder.user_id, "test-zone")
        async with store._db.execute("SELECT status, reason FROM reminder_home_actions") as cursor:
            assert [tuple(row) for row in await cursor.fetchall()] == [("failed", "action_not_selected")]
        assert ha.sends == []
    finally:
        await store.close()


async def test_reminder_claim_survives_lost_scheduler_completion_and_restart(graph, tmp_path):
    ha, actions, _ = graph
    db_path = str(tmp_path / "reminders.db")
    store = ReminderStore(db_path)
    await store.initialize()
    action = {"domain": "light", "service": "turn_on", "entity_id": "light.test_lamp"}
    reminder = await store.create("profile", "test", 1.0, ha_actions=[action])
    presence = MagicMock()
    presence.get_user_zone.return_value = "test-zone"
    scheduler = ReminderScheduler(store, AsyncMock(), presence, ha, home_actions=actions)
    ha.outcome = ServiceCallOutcome(ServiceCallStatus.UNCERTAIN, "response_lost")
    store.mark_fired = AsyncMock(side_effect=RuntimeError("scheduler interrupted"))
    with pytest.raises(RuntimeError, match="scheduler interrupted"):
        await scheduler._fire_reminder(reminder)
    await store.close()
    assert len(ha.sends) == 1
    restarted = ReminderStore(db_path)
    await restarted.initialize()
    try:
        scheduler = ReminderScheduler(restarted, AsyncMock(), presence, ha, home_actions=HomeActionService(ha, policy=actions.policy))
        await scheduler._fire_reminder(await restarted.get_by_id(reminder.id))
        assert len(ha.sends) == 1
        async with restarted._db.execute("SELECT status FROM reminder_home_actions") as cursor:
            assert [row[0] for row in await cursor.fetchall()] == ["uncertain"]
    finally:
        await restarted.close()


async def test_interrupted_reminder_claim_is_uncertain_and_body_cannot_change(tmp_path):
    path = str(tmp_path / "journal.db")
    store = ReminderStore(path)
    await store.initialize()
    assert await store.claim_home_action("test", {"entity_id": "light.test"})
    await store.close()
    store = ReminderStore(path)
    await store.initialize()
    try:
        assert not await store.claim_home_action("test", {"entity_id": "light.test"})
        assert (await store.get_home_action_receipt("test")).status == "uncertain"
        with pytest.raises(ValueError, match="body_conflict"):
            await store.claim_home_action("test", {"entity_id": "lock.hidden"})
    finally:
        await store.close()


async def test_due_and_arrival_race_share_one_committed_reminder_effect(graph, tmp_path):
    ha, actions, _ = graph
    store = ReminderStore(str(tmp_path / "race.db"))
    await store.initialize()
    try:
        reminder = await store.create("profile", "test", 1.0, ha_actions=[{"domain": "light", "service": "turn_on", "entity_id": "light.test_lamp"}])
        presence = MagicMock()
        presence.get_user_zone.return_value = "test-zone"
        scheduler = ReminderScheduler(store, AsyncMock(), presence, ha, home_actions=actions)
        await asyncio.gather(scheduler._fire_reminder(reminder), scheduler.missed_reminder_on_arrival("profile", "test-zone"))
        assert len(ha.sends) == 1
    finally:
        await store.close()


@pytest.mark.parametrize("uncertain", [False, True])
async def test_reminder_sequence_advances_only_after_confirmation(graph, tmp_path, uncertain):
    ha, actions, _ = graph
    store = ReminderStore(str(tmp_path / "sequence.db"))
    await store.initialize()
    try:
        reminder = await store.create("profile", "sequence", 1.0, ha_actions=[
            {"domain": "light", "service": "turn_on", "entity_id": "light.test_lamp"},
            {"domain": "light", "service": "turn_off", "entity_id": "light.test_lamp"},
        ])
        if uncertain:
            ha.outcome = ServiceCallOutcome(ServiceCallStatus.UNCERTAIN, "response_lost")
        presence = MagicMock()
        presence.get_user_zone.return_value = "test-zone"
        scheduler = ReminderScheduler(store, AsyncMock(), presence, ha, home_actions=actions)
        await scheduler._fire_reminder(reminder)
        assert [send[1] for send in ha.sends] == (["turn_on"] if uncertain else ["turn_on", "turn_off"])
        async with store._db.execute("SELECT status FROM reminder_home_actions ORDER BY rowid") as cursor:
            assert [row[0] for row in await cursor.fetchall()] == (["uncertain"] if uncertain else ["confirmed", "confirmed"])
    finally:
        await store.close()


async def test_reminder_journal_migration_preserves_legacy_rows_in_an_offline_copy(tmp_path):
    source = tmp_path / "legacy.db"
    store = ReminderStore(str(source))
    await store.initialize()
    reminder = await store.create("legacy-profile", "synthetic reminder", 123.0)
    await store._db.execute("DROP TABLE reminder_home_actions")
    await store._db.commit()
    await store.close()
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    candidate = tmp_path / "candidate.db"
    shutil.copy2(source, candidate)
    migrated = ReminderStore(str(candidate))
    await migrated.initialize()
    try:
        assert await migrated.get_by_id(reminder.id) == reminder
        assert await migrated.claim_home_action("new-effect", {"entity_id": "light.test"})
        assert (await migrated.get_home_action_receipt("new-effect")).status == "sending"
    finally:
        await migrated.close()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
