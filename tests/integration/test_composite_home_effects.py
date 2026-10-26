"""Reviewed synthetic lighting groups/scenes through real producers and policy."""

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
import numpy as np
import pytest

from tests.mocks.responses import response_handler_mock
from src.dashboard.api import DashboardAPI
from src.dashboard.home_catalog import HomeCatalogService
from src.home_assistant.control_policy import ActionOrigin, EntityGrant, HomeActor, HomeControlPolicy
from src.home_assistant.effect_plans import EffectIntent, GroupMembershipSource, scene_effects_from_states
from src.home_assistant.ha_client import HAAreaAssignments, HAStatesSnapshot, ServiceCallOutcome, ServiceCallStatus
from src.home_assistant.home_actions import HomeActionConflict, HomeActionRequest, HomeActionService
from src.home_assistant.voice_resolver import HomeVoiceResolver
from src.hooks import HookRegistry, RewriteResult
from src.nlu.command_gate import CommandAcceptanceGate
from src.orchestrator.dispatcher import MultiUserOrchestrator, RequestDispatcher
from src.pipeline.command_event import CommandEvent
from src.pipeline.command_processor import ProcessedCommand
from src.pipeline.request_router import RequestRouter
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_scheduler import ReminderScheduler
from src.reminders.reminder_store import ReminderStore
from src.rooms.room_context import RoomConfig
from src.routines.factory import build_routine_scheduler
from src.routines.models import parse_routine_spec, spec_to_dict
from tests.integration.test_typed_home_operations import TypedHA


LEAVES = ("light.test_lamp", "light.second_lamp")
GROUP = "light.test_group"
SCENE = "scene.test_evening"
LIGHT_OPS = frozenset({"turn_on", "turn_off", "brightness", "color_temp_kelvin"})


class CompositeHA(TypedHA):
    """Effect confirmation is real polling; no HA macro service may be called."""

    def __init__(self):
        super().__init__()
        self.states[LEAVES[1]] = deepcopy(self.states[LEAVES[0]])
        self.states[LEAVES[1]]["entity_id"] = LEAVES[1]
        self.states[GROUP] = deepcopy(self.states[LEAVES[0]])
        self.states[GROUP].update(entity_id=GROUP)
        self.states[GROUP]["attributes"]["entity_id"] = list(LEAVES)
        self.states[SCENE] = {"entity_id": SCENE, "state": "unknown", "attributes": {"entity_id": [GROUP], "friendly_name": "Synthetic evening"}}

    async def call_service_once(self, domain, service, entity_id, data):
        assert entity_id in (*LEAVES, "climate.test_ac", "cover.test_blind"), "Only explicit selected leaves may reach transport"
        outcome = await super().call_service_once(domain, service, entity_id, data)
        if self.observe_effect and data and "brightness" in data:
            self.states[entity_id]["attributes"]["brightness"] = data["brightness"]
        return outcome


@pytest.fixture
def composite_graph():
    ha = CompositeHA()
    grants = tuple(EntityGrant(entity, LIGHT_OPS, color_temp_min=2000, color_temp_max=6500) for entity in LEAVES)
    grants += (EntityGrant(GROUP, LIGHT_OPS, color_temp_min=2000, color_temp_max=6500, members=LEAVES),
        EntityGrant(SCENE, frozenset({"activate_scene"}), scene_effects=scene_effects_from_states({GROUP: {"state": "on", "color_temp_kelvin": 2700, "brightness": 179}}), display_name="Synthetic evening", aliases=("tarde de prueba",)))
    return ha, HomeActionService(ha, policy=HomeControlPolicy(grants))


def request(entity=SCENE, operation="activate_scene", request_id="composite-test", **parameters):
    return HomeActionRequest(entity, operation, request_id, origin=ActionOrigin.VOICE, **parameters)


@pytest.mark.parametrize("entity,operation,parameters,data", [
    (GROUP, "turn_on", {}, None), (GROUP, "turn_off", {}, None),
    (GROUP, "brightness", {"brightness_pct": 30}, {"brightness_pct": 30}),
    (GROUP, "color_temp_kelvin", {"color_temp_kelvin": 3000, "brightness_pct": 30}, {"color_temp_kelvin": 3000, "brightness_pct": 30}),
    (SCENE, "activate_scene", {}, {"color_temp_kelvin": 2700, "brightness": 179}),
])
async def test_group_or_scene_expands_to_exact_confirmed_leaves(composite_graph, entity, operation, parameters, data):
    ha, actions = composite_graph
    result = await actions.execute(request(entity, operation, **parameters), actor=HomeActor.household())
    assert result.status == "confirmed"
    assert [send[2:] for send in ha.sends] == [(leaf, data) for leaf in LEAVES]
    assert [effect.status for effect in result.effects] == ["confirmed", "confirmed"]
    assert [effect.intent.entity_id for effect in result.effects] == list(LEAVES)
    assert not actions._lanes


@pytest.mark.parametrize("source", [GroupMembershipSource.HA_STATE, GroupMembershipSource.REVIEWED_COPY])
async def test_bridge_group_without_ha_members_requires_an_explicit_reviewed_copy(composite_graph, source):
    ha, actions = composite_graph
    grants = list(await actions.policy.snapshot_grants())
    bridge_group = "light.synthetic_bridge_group"
    ha.states[bridge_group] = deepcopy(ha.states[LEAVES[0]])
    ha.states[bridge_group]["entity_id"] = bridge_group
    ha.states[GROUP]["attributes"]["entity_id"] = [bridge_group]
    grants.append(replace(grants[2], entity_id=bridge_group, membership_source=source))
    grants[2] = replace(grants[2], members=(bridge_group,))
    actions.policy = HomeControlPolicy(tuple(grants))
    result = await actions.execute(request(), actor=HomeActor.household())
    assert result.status == ("confirmed" if source is GroupMembershipSource.REVIEWED_COPY else "failed")
    assert [send[2] for send in ha.sends] == (list(LEAVES) if source is GroupMembershipSource.REVIEWED_COPY else [])


@pytest.mark.parametrize("brightness", [13, 20, 64, 179, 255])
async def test_raw_brightness_is_preserved_instead_of_rounding_through_percent(composite_graph, brightness):
    ha, actions = composite_graph
    result = await actions.execute(request(LEAVES[0], "brightness", brightness=brightness), actor=HomeActor.household())
    assert result.status == "confirmed"
    assert ha.sends == [("light", "turn_on", LEAVES[0], {"brightness": brightness})]
    assert ha.states[LEAVES[0]]["attributes"]["brightness"] == brightness


def test_raw_and_percent_brightness_cannot_conflict_or_bypass_parameter_types():
    with pytest.raises(ValueError, match="conflicting_brightness"):
        EffectIntent(LEAVES[0], "brightness", brightness=179, brightness_pct=70)
    with pytest.raises(ValueError, match="invalid_brightness"):
        EffectIntent(LEAVES[0], "brightness", brightness=True)


async def test_raw_brightness_checks_exact_selected_percentage_limits(composite_graph):
    ha, actions = composite_graph
    grants = list(await actions.policy.snapshot_grants())
    grants[0] = replace(grants[0], brightness_max=70)
    actions.policy = HomeControlPolicy(tuple(grants))
    result = await actions.execute(request(LEAVES[0], "brightness", brightness=179), actor=HomeActor.household())
    assert result.status == "failed" and result.reason == "brightness_outside_selected_range"
    assert ha.sends == []


async def test_internal_leaf_grants_do_not_create_independent_household_controls(composite_graph):
    ha, actions = composite_graph
    grants = tuple(replace(grant, direct_control=False) if grant.entity_id in LEAVES else grant for grant in await actions.policy.snapshot_grants())
    actions.policy = HomeControlPolicy(grants)
    assert (await actions.execute(request(LEAVES[0], "turn_on", "direct-denied"), actor=HomeActor.household())).status == "failed"
    assert ha.sends == []
    assert (await actions.execute(request(), actor=HomeActor.household())).status == "confirmed"
    ha.get_state_snapshot = AsyncMock(return_value=HAStatesSnapshot(True, None, None, tuple(ha.states.values())))
    ha.get_area_assignments = AsyncMock(return_value=HAAreaAssignments(True, None, None))
    catalog = await HomeCatalogService(ha, {}, policy=actions.policy).snapshot()
    assert all(not entity.capabilities for entity in catalog.entities if entity.entity_id in LEAVES)
    assert next(entity for entity in catalog.entities if entity.entity_id == GROUP).capabilities


@pytest.mark.parametrize("case", ["missing_leaf", "missing_leaf_operation", "missing_group_permission", "unreviewed_group", "changed_membership", "missing_membership", "unavailable_last", "unsupported_last", "range_last", "changed_leaf_to_group"])
async def test_all_effects_are_preflighted_before_any_send(composite_graph, case):
    ha, actions = composite_graph
    grants = list(await actions.policy.snapshot_grants())
    if case == "missing_leaf":
        grants = [grant for grant in grants if grant.entity_id != LEAVES[1]]
    elif case == "missing_leaf_operation":
        grants[1] = replace(grants[1], operations=frozenset({"turn_on"}))
    elif case == "missing_group_permission":
        grants[2] = replace(grants[2], operations=frozenset({"turn_on"}))
    elif case == "unreviewed_group":
        grants[2] = replace(grants[2], members=())
    elif case == "changed_membership":
        ha.states[GROUP]["attributes"]["entity_id"].append("light.not_selected")
    elif case == "missing_membership":
        del ha.states[GROUP]["attributes"]["entity_id"]
    elif case == "unavailable_last":
        ha.states[LEAVES[1]]["state"] = "unavailable"
    elif case == "unsupported_last":
        ha.states[LEAVES[1]]["attributes"]["supported_color_modes"] = ["onoff"]
    elif case == "range_last":
        grants[1] = replace(grants[1], brightness_max=60)
    else:
        ha.states[LEAVES[1]]["attributes"]["entity_id"] = ["lock.hidden"]
    actions.policy = HomeControlPolicy(tuple(grants))
    result = await actions.execute(request(), actor=HomeActor.household())
    assert result.status == "failed"
    assert ha.sends == []


@pytest.mark.parametrize("desired", [
    {"lock.hidden": {"state": "unlocked"}}, {"alarm_control_panel.hidden": {"state": "disarmed"}},
    {"script.opaque": {"state": "on"}}, {"scene.nested": {"state": "on"}},
    {GROUP: {"state": "on", "effect": "unreviewed"}}, {GROUP: {"state": "on", "brightness": "{{ value }}"}},
    {GROUP: {"state": "off", "brightness": 20}}, {GROUP: {"state": "on", "brightness": 0}},
    {GROUP: {"state": True}}, {GROUP: {"state": "on", "transition": 10}},
])
def test_restricted_hidden_or_opaque_scene_effects_cannot_be_compiled(desired):
    with pytest.raises(ValueError):
        scene_effects_from_states(desired)


@pytest.mark.parametrize("case", ["cycle", "depth", "conflict", "duplicate_identical"])
async def test_nested_expansion_is_bounded_and_deduplicates_only_identical_intents(composite_graph, case):
    ha, actions = composite_graph
    grants = list(await actions.policy.snapshot_grants())
    if case == "cycle":
        grants[2] = replace(grants[2], members=("light.nested_group",))
        grants.append(replace(grants[2], entity_id="light.nested_group", members=(GROUP,)))
        ha.states[GROUP]["attributes"]["entity_id"] = ["light.nested_group"]
    elif case == "depth":
        grants[2] = replace(grants[2], members=("light.nested_0",))
        for index in range(10):
            grants.append(replace(grants[2], entity_id=f"light.nested_{index}", members=((f"light.nested_{index + 1}" if index < 9 else LEAVES[0]),)))
    else:
        desired = {GROUP: {"state": "on", "color_temp_kelvin": 2700, "brightness": 179},
            LEAVES[1]: {"state": "on", "color_temp_kelvin": 3000 if case == "conflict" else 2700, "brightness": 179}}
        grants[3] = replace(grants[3], scene_effects=scene_effects_from_states(desired))
        ha.states[SCENE]["attributes"]["entity_id"] = list(desired)
    actions.policy = HomeControlPolicy(tuple(grants))
    result = await actions.execute(request(), actor=HomeActor.household())
    assert result.status == ("confirmed" if case == "duplicate_identical" else "failed")
    assert len(ha.sends) == (2 if case == "duplicate_identical" else 0)


@pytest.mark.parametrize("case", ["revocation", "definition_change", "membership_change", "rejection", "unconfirmed", "exception"])
async def test_mid_sequence_failure_is_partial_uncertainty_without_retries(composite_graph, case):
    ha, actions = composite_graph
    original = ha.call_service_once

    async def changing(domain, service, entity_id, data):
        if entity_id == LEAVES[1]:
            if case == "rejection":
                ha.sends.append((domain, service, entity_id, data))
                return ServiceCallOutcome(ServiceCallStatus.REJECTED, "rejected")
            if case == "unconfirmed":
                ha.observe_effect = False
            if case == "exception":
                ha.sends.append((domain, service, entity_id, data))
                raise ConnectionError("response lost")
        result = await original(domain, service, entity_id, data)
        if entity_id == LEAVES[0]:
            if case == "revocation":
                actions.policy = HomeControlPolicy()
            elif case == "membership_change":
                ha.states[GROUP]["attributes"]["entity_id"].pop()
            elif case == "definition_change":
                grants = list(await actions.policy.snapshot_grants())
                grants[3] = replace(grants[3], scene_effects=scene_effects_from_states({GROUP: {"state": "off"}}))
                actions.policy = HomeControlPolicy(tuple(grants))
        return result

    ha.call_service_once = changing
    result = await actions.execute(request(), actor=HomeActor.household())
    assert result.status == "uncertain" and result.reason == "partial_effects"
    assert result.effects[0].status == "confirmed"
    count = 2 if case in {"rejection", "unconfirmed", "exception"} else 1
    assert len(ha.sends) == count
    assert await actions.execute(request(), actor=HomeActor.household()) == result
    assert len(ha.sends) == count


async def test_uncertain_first_effect_stops_and_retains_unexecuted_leaves(composite_graph):
    ha, actions = composite_graph
    ha.observe_effect = False
    result = await actions.execute(request(), actor=HomeActor.household())
    assert result.status == "uncertain"
    assert [effect.status for effect in result.effects] == ["uncertain", "not_executed"]
    assert len(ha.sends) == 1
    assert await actions.execute(request(), actor=HomeActor.household()) == result
    assert len(ha.sends) == 1


async def test_overlapping_composite_and_direct_controls_serialize_without_deadlock(composite_graph):
    ha, actions = composite_graph
    entered, release = asyncio.Event(), asyncio.Event()
    original = ha.call_service_once

    async def delayed(*args):
        if not ha.sends:
            entered.set()
            await release.wait()
        return await original(*args)

    ha.call_service_once = delayed
    first = asyncio.create_task(actions.execute(request(), actor=HomeActor.household()))
    await entered.wait()
    duplicate = asyncio.create_task(actions.execute(request(), actor=HomeActor.household()))
    direct = asyncio.create_task(actions.execute(request(LEAVES[1], "turn_off", "direct"), actor=HomeActor.household()))
    await asyncio.sleep(0)
    assert ha.sends == []
    release.set()
    results = await asyncio.wait_for(asyncio.gather(first, duplicate, direct), timeout=2)
    assert all(result.status == "confirmed" for result in results)
    assert len(ha.sends) == 3 and ha.sends[-1][1] == "turn_off"
    assert not actions._lanes
    with pytest.raises(HomeActionConflict):
        await actions.execute(request(GROUP, "turn_off"), actor=HomeActor.household())


@pytest.mark.parametrize("orchestrated", [False, True])
@pytest.mark.parametrize("text", ["Activá la escena tarde de prueba", "Poné la luz del salón cálida al 30%"])
async def test_both_voice_modes_use_the_same_expanded_effects(composite_graph, orchestrated, text):
    ha, actions = composite_graph
    processor = MagicMock()
    processor.process_command = AsyncMock(return_value=ProcessedCommand(text=text, timings={}, success=True))
    chroma = MagicMock()
    orchestrator = MultiUserOrchestrator(chroma, ha, MagicMock(), home_actions=actions)
    rooms = {"test_room": RoomConfig("test_room", "Test", "el salón", default_light=GROUP, aliases=["salón"])}
    router = RequestRouter(processor, response_handler_mock(), MagicMock(), orchestrator=orchestrator, orchestrator_enabled=orchestrated,
        chroma_sync=chroma, ha_client=ha, home_actions=actions, command_gate=CommandAcceptanceGate(),
        home_voice_resolver=HomeVoiceResolver(actions, rooms))
    result = await router.process_command(CommandEvent(np.zeros(16000, dtype=np.float32), "test_room", capture_complete=True))
    assert result["success"] is True
    assert len(ha.sends) == 2
    assert all(send[2] in LEAVES for send in ha.sends)
    chroma.search_command.assert_not_called()


@pytest.mark.parametrize("text", ["Activá la escena tarde de prueba mañana", "En cinco minutos activá la escena tarde de prueba", "No actives la escena tarde de prueba", "¿Activaste la escena tarde de prueba?"])
@pytest.mark.parametrize("orchestrated", [False, True])
async def test_scene_speech_keeps_shared_non_immediate_blockers(composite_graph, text, orchestrated):
    ha, actions = composite_graph
    processor = MagicMock()
    processor.process_command = AsyncMock(return_value=ProcessedCommand(text=text, timings={}, success=True))
    orchestrator = MultiUserOrchestrator(MagicMock(), ha, MagicMock(), home_actions=actions)
    router = RequestRouter(processor, response_handler_mock(), MagicMock(), orchestrator_enabled=orchestrated, orchestrator=orchestrator,
        ha_client=ha, home_actions=actions, command_gate=CommandAcceptanceGate())
    result = await router.process_command(np.zeros(16000, dtype=np.float32))
    assert result["intent"].startswith("gate_rejected:non_immediate:")
    assert ha.sends == []


def test_authenticated_http_exposes_leaf_receipts_and_preserves_raw_values(composite_graph, monkeypatch):
    ha, actions = composite_graph
    monkeypatch.setenv("KZA_DASHBOARD_TOKEN", "test-session")
    with TestClient(DashboardAPI(home_enabled=True, home_actions=actions).app) as client:
        body = {"entity_id": SCENE, "operation": "activate_scene", "request_id": "http-scene"}
        assert client.post("/api/home/actions", json=body).status_code == 401
        response = client.post("/api/home/actions", headers={"Authorization": "Bearer test-session"}, json=body)
        assert response.status_code == 200 and response.json()["status"] == "confirmed"
        assert [effect["intent"]["brightness"] for effect in response.json()["effects"]] == [179, 179]
        assert len(ha.sends) == 2


async def test_catalog_uses_all_leaf_limits_and_only_verified_scene_effects(composite_graph):
    ha, actions = composite_graph
    ha.get_state_snapshot = AsyncMock(side_effect=lambda: HAStatesSnapshot(True, None, None, tuple(ha.states.values())))
    ha.get_area_assignments = AsyncMock(return_value=HAAreaAssignments(True, None, None))
    catalog = HomeCatalogService(ha, {}, policy=actions.policy)
    entities = {entity.entity_id: entity for entity in (await catalog.snapshot()).entities}
    assert entities[SCENE].capabilities == ("activate_scene",) and entities[SCENE].available
    assert "color_temp_kelvin" in entities[GROUP].capabilities
    ha.states[LEAVES[1]]["attributes"]["min_color_temp_kelvin"] = 3000
    entities = {entity.entity_id: entity for entity in (await catalog.snapshot()).entities}
    assert entities[GROUP].control_limits.color_temp_min == 3000
    assert entities[SCENE].capabilities == ()
    ha.states[LEAVES[1]]["state"] = "unavailable"
    assert next(entity for entity in (await catalog.snapshot()).entities if entity.entity_id == GROUP).capabilities == ()


async def test_scene_and_raw_brightness_survive_persistent_routine_execution(composite_graph, tmp_path):
    ha, actions = composite_graph
    payload = {"name": "test", "description": "", "enabled": True, "timezone": "America/Montevideo", "schedule": None, "conditions": [],
        "actions": [{"entity_id": SCENE, "operation": "activate_scene"}, {"entity_id": GROUP, "operation": "brightness", "brightness": 20}]}
    scheduler = await build_routine_scheduler(db_path=str(tmp_path / "routines.db"), timezone="America/Montevideo", ha_client=ha, home_actions=actions)
    try:
        routine = await scheduler.register_routine(parse_routine_spec(payload), actor="dashboard-admin")
        assert spec_to_dict((await scheduler.get_routine(routine.routine_id)).spec) == payload
        run = await scheduler.execute_by_id(routine.routine_id, actor="dashboard-admin", request_id="test")
        await asyncio.gather(*scheduler._workers.values())
        assert (await scheduler.get_run(routine.routine_id, run.run_id)).status == "succeeded"
        assert [send[3] for send in ha.sends] == [{"color_temp_kelvin": 2700, "brightness": 179}] * 2 + [{"brightness": 20}] * 2
    finally:
        await scheduler.stop()
        await scheduler.store.close()


@pytest.mark.parametrize("delivery", ["due", "arrival"])
async def test_reminder_durable_claim_prevents_replay_of_partial_scene_after_restart(composite_graph, tmp_path, delivery):
    ha, actions = composite_graph
    path = str(tmp_path / "reminders.db")
    store = ReminderStore(path)
    await store.initialize()
    manager = ReminderManager(store, home_actions=actions)
    reminder = await manager.create("profile", "test", 1.0, ha_actions=[{"entity_id": SCENE, "domain": "scene", "service": "turn_on"}])
    original = ha.call_service_once

    async def partial(*args):
        result = await original(*args)
        ha.observe_effect = False
        return result

    ha.call_service_once = partial
    presence = MagicMock()
    presence.get_user_zone.return_value = "test-zone"
    scheduler = ReminderScheduler(store, AsyncMock(), presence, ha, home_actions=actions)
    if delivery == "due":
        await scheduler._fire_reminder(reminder)
    else:
        await scheduler.missed_reminder_on_arrival(reminder.user_id, "test-zone")
    assert len(ha.sends) == 2
    await store.close()
    restarted = ReminderStore(path)
    await restarted.initialize()
    try:
        new_actions = HomeActionService(ha, policy=actions.policy)
        scheduler = ReminderScheduler(restarted, AsyncMock(), presence, ha, home_actions=new_actions)
        await scheduler._fire_reminder(await restarted.get_by_id(reminder.id))
        assert len(ha.sends) == 2
        async with restarted._db.execute("SELECT status FROM reminder_home_actions") as cursor:
            assert [row[0] for row in await cursor.fetchall()] == ["uncertain"]
    finally:
        await restarted.close()


@pytest.mark.parametrize("rewrite", ["target", "parameters"])
async def test_hook_rewrites_cannot_hide_restricted_effects_in_a_group(composite_graph, rewrite):
    ha, actions = composite_graph
    hooks = HookRegistry()

    def change(call):
        modified = replace(call, entity_id="lock.hidden", domain="lock", service="unlock") if rewrite == "target" else call.with_data(target={"entity_id": "lock.hidden"})
        return RewriteResult(modified=modified, rule_name="test")

    hooks.register_before("before_ha_action", change, priority=10)
    dispatcher = RequestDispatcher(MagicMock(), ha, MagicMock(), home_actions=actions, hooks=hooks)
    assert not await dispatcher._fire_and_reconcile_ha({"entity_id": GROUP, "domain": "light", "service": "turn_on"})
    assert ha.sends == []


@pytest.mark.parametrize("verified", [False, True])
async def test_private_review_requires_separate_effect_verification(composite_graph, tmp_path, verified):
    ha, _ = composite_graph
    entries = [{"entity_id": entity, "operations": list(LIGHT_OPS), "live_verified": True, "indirect_effects_verified": True,
        "parameters": {"brightness_pct": {"min": 0, "max": 100}, "color_temp_kelvin": {"min": 2000, "max": 6500}}} for entity in (*LEAVES, GROUP)]
    entries[2].update(kind="light_group", effects_verified=verified, members=list(LEAVES))
    entries.append({"entity_id": SCENE, "operations": ["activate_scene"], "live_verified": True, "indirect_effects_verified": True, "kind": "scene", "effects_verified": verified,
        "name": "Synthetic evening", "aliases": ["tarde de prueba"], "states": {GROUP: {"state": "on", "brightness": 179, "color_temp_kelvin": 2700}}})
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"selection_approved": True, "live_api_verified": True, "allowed_entities": entries}))
    path.chmod(0o600)
    actions = HomeActionService(ha, policy=HomeControlPolicy(selection_path=str(path)))
    result = await actions.execute(request(), actor=HomeActor.household())
    assert result.status == ("confirmed" if verified else "failed")
    assert len(ha.sends) == (2 if verified else 0)
