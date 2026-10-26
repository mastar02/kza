"""Typed controls through real policy, producers, persistence and confirmation."""

import asyncio
from dataclasses import replace
import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
import pytest

from src.dashboard.api import DashboardAPI
from src.home_assistant.control_policy import ActionOrigin, EntityGrant, HomeActor, HomeControlPolicy
from src.home_assistant.ha_client import HomeAssistantClient, ServiceCallOutcome, ServiceCallStatus
from src.home_assistant.home_actions import HomeActionRequest, HomeActionService, request_from_service
from src.home_assistant.operations import action_parameters
from src.orchestrator.dispatcher import RequestDispatcher
from src.reminders.reminder_scheduler import ReminderScheduler
from src.reminders.reminder_store import ReminderStore
from src.routines.factory import build_routine_scheduler
from src.routines.models import parse_routine_spec, spec_to_dict
from tests.integration.test_home_control_policy import ObservedHA


MODES = ["off", "cool", "dry", "fan_only", "auto", "heat"]


class TypedHA(ObservedHA):
    """A synthetic installation; polling and effect matching use the real client."""

    state_confirmation_timeout_s = 0.08

    def __init__(self):
        super().__init__()
        self.unit = "°C"
        self.unit_after_send = None
        self.states["light.test_lamp"]["attributes"].update(supported_color_modes=["color_temp"], min_color_temp_kelvin=2000, max_color_temp_kelvin=6500)
        self.states["climate.test_ac"] = {"entity_id": "climate.test_ac", "state": "cool", "attributes": {"supported_features": 1, "hvac_modes": MODES.copy(), "min_temp": 7, "max_temp": 35, "target_temp_step": 0.5, "temperature": 22, "current_temperature": 31}}
        self.states["cover.test_blind"] = {"entity_id": "cover.test_blind", "state": "open", "attributes": {"supported_features": 15, "device_class": "blind", "current_position": 40}}

    async def get_temperature_unit_fresh(self, **kwargs):
        return self.unit

    async def confirm_home_action(self, *args, **kwargs):
        return await HomeAssistantClient.confirm_home_action(self, *args, **kwargs)

    async def call_service_once(self, domain, service, entity_id, data):
        self.sends.append((domain, service, entity_id, data))
        if self.unit_after_send is not None:
            self.unit = self.unit_after_send
        if self.observe_effect:
            entity = self.states[entity_id]
            if domain == "climate":
                if service == "set_hvac_mode":
                    entity["state"] = data["hvac_mode"]
                else:
                    entity["attributes"]["temperature"] = data["temperature"]
            elif domain == "cover":
                position = {"open_cover": 100, "close_cover": 0}.get(service, (data or {}).get("position", entity["attributes"]["current_position"]))
                entity["attributes"]["current_position"] = position
                entity["state"] = "closed" if position == 0 else "open"
            else:
                entity["state"] = "off" if service == "turn_off" else "on"
                if data and "brightness_pct" in data:
                    entity["attributes"]["brightness"] = round(data["brightness_pct"] * 255 / 100)
                if data and "color_temp_kelvin" in data:
                    entity["attributes"].update(color_mode="color_temp", color_temp_kelvin=data["color_temp_kelvin"])
        return ServiceCallOutcome(ServiceCallStatus.ACCEPTED, None)


@pytest.fixture
def typed_graph():
    ha = TypedHA()
    grants = (
        EntityGrant("light.test_lamp", frozenset({"turn_on", "turn_off", "brightness", "color_temp_kelvin"}), color_temp_min=2000, color_temp_max=6500),
        EntityGrant("climate.test_ac", frozenset({"set_temperature", "set_hvac_mode"}), temperature_min=7, temperature_max=35, hvac_modes=frozenset(MODES)),
        EntityGrant("cover.test_blind", frozenset({"open_cover", "close_cover", "stop_cover", "set_cover_position"}), cover_device_class="blind"),
    )
    return ha, HomeActionService(ha, policy=HomeControlPolicy(grants))


CASES = [
    ("climate.test_ac", "set_temperature", {"temperature": 7}),
    ("climate.test_ac", "set_temperature", {"temperature": 35}),
    ("climate.test_ac", "set_temperature", {"temperature": 23.5}),
    *[("climate.test_ac", "set_hvac_mode", {"hvac_mode": mode}) for mode in MODES],
    ("light.test_lamp", "color_temp_kelvin", {"color_temp_kelvin": 2700, "brightness_pct": 30}),
    ("cover.test_blind", "open_cover", {}),
    ("cover.test_blind", "close_cover", {}),
    ("cover.test_blind", "stop_cover", {}),
    ("cover.test_blind", "set_cover_position", {"position": 25}),
]


@pytest.mark.parametrize("entity,operation,parameters", CASES)
async def test_typed_effect_preserves_parameters_and_confirms_ha_state(typed_graph, entity, operation, parameters):
    ha, actions = typed_graph
    request = HomeActionRequest(entity, operation, "test", origin=ActionOrigin.VOICE, **parameters)
    result = await actions.execute(request, actor=HomeActor.household())
    assert result.status == "confirmed"
    expected_service = "turn_on" if operation == "color_temp_kelvin" else operation
    assert ha.sends == [(entity.partition(".")[0], expected_service, entity, parameters or None)]


@pytest.mark.parametrize("case", ["unit_unknown", "unit_fahrenheit", "unit_changed", "feature_missing", "technical_range", "step", "no_effect", "mode_missing", "cover_class", "cover_feature"])
async def test_capabilities_and_units_cannot_be_assumed_from_selection(typed_graph, case):
    ha, actions = typed_graph
    request = HomeActionRequest("climate.test_ac", "set_temperature", "test", temperature=24, origin=ActionOrigin.VOICE)
    if case == "unit_unknown":
        ha.unit = None
    elif case == "unit_fahrenheit":
        ha.unit = "°F"
    elif case == "unit_changed":
        ha.unit_after_send = "°F"
    elif case == "feature_missing":
        ha.states[request.entity_id]["attributes"]["supported_features"] = 0
    elif case == "technical_range":
        ha.states[request.entity_id]["attributes"]["max_temp"] = 23
    elif case == "step":
        request = replace(request, temperature=24.2)
    elif case == "no_effect":
        ha.observe_effect = False
        ha.states[request.entity_id]["attributes"]["current_temperature"] = request.temperature
    elif case == "mode_missing":
        request = replace(request, operation="set_hvac_mode", temperature=None, hvac_mode="heat")
        ha.states[request.entity_id]["attributes"]["hvac_modes"].remove("heat")
    else:
        request = HomeActionRequest("cover.test_blind", "open_cover", "test", origin=ActionOrigin.VOICE)
        ha.states[request.entity_id]["attributes"]["device_class" if case == "cover_class" else "supported_features"] = "garage" if case == "cover_class" else 0
    result = await actions.execute(request, actor=HomeActor.household())
    assert result.status == ("uncertain" if case in {"unit_changed", "no_effect"} else "failed")
    assert len(ha.sends) == int(case in {"unit_changed", "no_effect"})


@pytest.mark.parametrize("parameters", [{"temperature": True}, {"temperature": float("nan")}, {"temperature": float("inf")}, {"temperature": 6}, {"temperature": 36}, {"temperature": "22"}, {"temperature": 22, "hvac_mode": "heat"}])
async def test_malformed_or_unselected_temperature_fails_before_observation(typed_graph, parameters):
    ha, actions = typed_graph
    result = await actions.execute(HomeActionRequest("climate.test_ac", "set_temperature", "test", origin=ActionOrigin.VOICE, **parameters), actor=HomeActor.household())
    assert result.status == "failed"
    assert ha.sends == ha.reads == []


@pytest.mark.parametrize("entity,service,data", [
    ("climate.test_ac", "set_temperature", {"temperature": 23.5}),
    ("climate.test_ac", "set_hvac_mode", {"hvac_mode": "dry"}),
    ("cover.test_blind", "set_cover_position", {"position": 65}),
    ("light.test_lamp", "turn_on", {"color_temp_kelvin": 3000, "brightness_pct": 70}),
])
async def test_voice_dispatcher_consumes_typed_service_after_policy(typed_graph, entity, service, data):
    ha, actions = typed_graph
    dispatcher = RequestDispatcher(MagicMock(), ha, MagicMock(), home_actions=actions)
    assert await dispatcher._fire_and_reconcile_ha({"entity_id": entity, "domain": entity.partition(".")[0], "service": service, "data": data})
    assert ha.sends[0][3] == data


@pytest.mark.parametrize("parameters", [{"preset_mode": "boost"}, {"fan_mode": "high"}, {"temperature": 22, "hvac_mode": "heat"}, {"target_temp_high": 28}, {"entity_id": "lock.hidden"}])
def test_legacy_adapter_rejects_untyped_climate_parameters(parameters):
    with pytest.raises(ValueError):
        request_from_service("climate", "set_temperature", "climate.test_ac", parameters, request_id="test", origin=ActionOrigin.VOICE)


@pytest.mark.parametrize("entity,operation,parameters", CASES)
def test_authenticated_http_accepts_only_exact_typed_intents(typed_graph, monkeypatch, entity, operation, parameters):
    ha, actions = typed_graph
    monkeypatch.setenv("KZA_DASHBOARD_TOKEN", "test-session")
    with TestClient(DashboardAPI(home_enabled=True, home_actions=actions).app) as client:
        response = client.post("/api/home/actions", headers={"Authorization": "Bearer test-session"}, json={"entity_id": entity, "operation": operation, "request_id": "test", **parameters})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "confirmed"
    assert ha.sends[0][3] == (parameters or None)


async def test_all_extended_fields_survive_routine_storage_and_execution(typed_graph, tmp_path):
    ha, actions = typed_graph
    scheduler = await build_routine_scheduler(db_path=str(tmp_path / "routines.db"), timezone="America/Montevideo", ha_client=ha, home_actions=actions)
    payload = {"name": "typed", "description": "", "enabled": True, "timezone": "America/Montevideo", "schedule": None, "conditions": [], "actions": [{"entity_id": entity, "operation": op, **params} for entity, op, params in CASES]}
    try:
        spec = parse_routine_spec(payload)
        assert spec_to_dict(spec) == payload
        routine = await scheduler.register_routine(spec, actor="dashboard-admin")
        loaded = await scheduler.get_routine(routine.routine_id)
        assert spec_to_dict(loaded.spec) == payload
        run = await scheduler.execute_by_id(routine.routine_id, actor="dashboard-admin", request_id="typed-test")
        await asyncio.gather(*scheduler._workers.values())
        result = await scheduler.get_run(routine.routine_id, run.run_id)
        assert result.status == "succeeded"
        assert [send[3] for send in ha.sends] == [params or None for _, _, params in CASES]
    finally:
        await scheduler.stop()
        await scheduler.store.close()


async def test_reminder_preserves_extended_parameters_and_durable_receipt(typed_graph, tmp_path):
    ha, actions = typed_graph
    store = ReminderStore(str(tmp_path / "reminders.db"))
    await store.initialize()
    try:
        reminder = await store.create("profile", "typed", 1.0, ha_actions=[{"entity_id": "climate.test_ac", "domain": "climate", "service": "set_temperature", "data": {"temperature": 23.5}}])
        presence = MagicMock()
        presence.get_user_zone.return_value = "test-zone"
        await ReminderScheduler(store, AsyncMock(), presence, ha, home_actions=actions)._fire_reminder(reminder)
        assert ha.sends[0][3] == {"temperature": 23.5}
        async with store._db.execute("SELECT status FROM reminder_home_actions") as cursor:
            assert [row[0] for row in await cursor.fetchall()] == ["confirmed"]
    finally:
        await store.close()


async def test_private_review_parser_preserves_authorized_modes_and_celsius_bounds(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"selection_approved": True, "live_api_verified": True, "allowed_entities": [{"entity_id": "climate.test_ac", "operations": ["set_temperature", "set_hvac_mode"], "live_verified": True, "indirect_effects_verified": True, "parameters": {"temperature": {"min": 7, "max": 35, "unit": "C", "unit_confirmed_by_user": True}, "hvac_modes": MODES}}]}))
    ha = TypedHA()
    actions = HomeActionService(ha, policy=HomeControlPolicy(selection_path=str(path)))
    request = HomeActionRequest("climate.test_ac", "set_temperature", "test", origin=ActionOrigin.VOICE, temperature=35)
    assert action_parameters(request) == {"temperature": 35}
    assert (await actions.execute(request, actor=HomeActor.household())).status == "confirmed"


@pytest.mark.parametrize("operation,parameters", [("open_cover", {}), ("stop_cover", {}), ("color_temp_kelvin", {"color_temp_kelvin": 2700})])
async def test_nonbinary_control_requires_observed_requested_effect(typed_graph, operation, parameters):
    ha, actions = typed_graph
    entity = "light.test_lamp" if operation == "color_temp_kelvin" else "cover.test_blind"
    ha.observe_effect = False
    if operation == "stop_cover":
        ha.states[entity]["state"] = "closing"
    if operation == "color_temp_kelvin":
        ha.states[entity]["state"] = "on"
        ha.states[entity]["attributes"].update(color_mode="hs", color_temp_kelvin=2700)
    result = await actions.execute(HomeActionRequest(entity, operation, "test", origin=ActionOrigin.VOICE, **parameters), actor=HomeActor.household())
    assert result.status == "uncertain"
    assert len(ha.sends) == 1


async def test_catalog_uses_shared_selection_and_intersects_technical_bounds(typed_graph):
    from src.dashboard.home_catalog import HomeCatalogService
    from src.home_assistant.ha_client import HAAreaAssignments, HAStatesSnapshot

    ha, actions = typed_graph
    ha.get_state_snapshot = AsyncMock(return_value=HAStatesSnapshot(True, None, None, tuple(ha.states.values())))
    ha.get_area_assignments = AsyncMock(return_value=HAAreaAssignments(True, None, None))
    catalog = HomeCatalogService(ha, {}, policy=actions.policy)
    snapshot = await catalog.snapshot()
    ac = next(entity for entity in snapshot.entities if entity.domain == "climate")
    assert ac.capabilities == ("set_hvac_mode", "set_temperature")
    assert ac.control_limits.temperature_min == 7
    assert ac.control_limits.hvac_modes == tuple(MODES)
    ha.states[ac.entity_id]["attributes"].update(min_temp=18, max_temp=25)
    narrowed = next(entity for entity in (await catalog.snapshot()).entities if entity.domain == "climate")
    assert (narrowed.control_limits.temperature_min, narrowed.control_limits.temperature_max) == (18, 25)
    ha.unit = "°F"
    assert "set_temperature" not in next(entity for entity in (await catalog.snapshot()).entities if entity.domain == "climate").capabilities
    unselected = await HomeCatalogService(ha, {}).snapshot()
    assert all(entity.capabilities == () for entity in unselected.entities)


async def test_temperature_confirmation_cannot_accept_an_unrequested_mode_change(typed_graph):
    ha, actions = typed_graph
    original = ha.call_service_once

    async def changing_mode(*args, **kwargs):
        result = await original(*args, **kwargs)
        ha.states["climate.test_ac"]["state"] = "heat"
        return result

    ha.call_service_once = changing_mode
    request = HomeActionRequest("climate.test_ac", "set_temperature", "test", origin=ActionOrigin.VOICE, temperature=23)
    assert (await actions.execute(request, actor=HomeActor.household())).status == "uncertain"
    assert len(ha.sends) == 1


async def test_cover_uses_its_bounded_travel_budget_instead_of_binary_timeout(typed_graph):
    ha, actions = typed_graph
    ha.state_confirmation_timeout_s = 0.01
    ha._operation_confirmation_timeouts = {"open_cover": 0.2}
    original = ha.call_service_once
    pending = []

    async def traveling(*args, **kwargs):
        result = await original(*args, **kwargs)
        entity = ha.states["cover.test_blind"]
        entity["state"] = "opening"
        entity["attributes"]["current_position"] = 40

        async def finish():
            await asyncio.sleep(0.03)
            entity["state"] = "open"
            entity["attributes"]["current_position"] = 100

        pending.append(asyncio.create_task(finish()))
        return result

    ha.call_service_once = traveling
    request = HomeActionRequest("cover.test_blind", "open_cover", "test", origin=ActionOrigin.VOICE)
    assert (await actions.execute(request, actor=HomeActor.household())).status == "confirmed"
    await asyncio.gather(*pending)
    assert len(ha.sends) == 1


@pytest.mark.parametrize("variant", ["celsius", "fahrenheit", "denied", "malformed", "redirect"])
async def test_temperature_unit_read_uses_real_bounded_authenticated_http(variant):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    redirected = []

    async def config(request):
        assert request.headers["Authorization"] == "Bearer test-only-token"
        if variant == "denied":
            return web.Response(status=401)
        if variant == "malformed":
            return web.json_response({"unit_system": None})
        if variant == "redirect":
            return web.Response(status=307, headers={"Location": "/redirected"})
        return web.json_response({"unit_system": {"temperature": "°C" if variant == "celsius" else "°F"}})

    async def redirect(request):
        redirected.append(True)
        return web.json_response({"unit_system": {"temperature": "°C"}})

    app = web.Application()
    app.router.add_get("/api/config", config)
    app.router.add_get("/redirected", redirect)
    async with TestServer(app) as server:
        client = HomeAssistantClient(str(server.make_url("")).rstrip("/"), "test-only-token")
        try:
            expected = {"celsius": "°C", "fahrenheit": "°F"}.get(variant)
            assert await client.get_temperature_unit_fresh() == expected
            assert not redirected
        finally:
            await client.close()
