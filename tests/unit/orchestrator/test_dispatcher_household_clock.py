"""P3-F1: the dispatcher's injected household clock and origin room."""

import inspect
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.world.weather as weather_module
from src.core.household_clock import HouseholdClock, RoomNamespace
from src.orchestrator.context_manager import ContextManager
from src.orchestrator.dispatcher import MultiUserOrchestrator, PathType, Priority, RequestDispatcher
from src.orchestrator.priority_queue import PriorityRequestQueue

ROOMS = RoomNamespace(("living", "hall", "cocina", "escritorio", "bano"))
# 02:30Z of the 5th is 23:30 of the 4th in Montevideo: "mañana" is the 5th.
FROZEN = datetime(2026, 8, 5, 2, 30, tzinfo=UTC)


def make_dispatcher(**kwargs) -> RequestDispatcher:
    return RequestDispatcher(
        chroma_sync=MagicMock(), ha_client=AsyncMock(), router=None, llm=None,
        context_manager=ContextManager(), priority_queue=PriorityRequestQueue(), **kwargs,
    )


def forecast(*entries):
    return {"service_response": {"weather.forecast_home": {"forecast": [
        {"datetime": iso, "condition": condition, "temperature": high} for iso, condition, high in entries]}}}


@pytest.mark.parametrize(("context_room_id", "zone_id", "expected"), [
    ("hall", "zone_hall", "hall"),                   # the configured room of the mic that heard
    ("hall", "zone_living", "hall"),                 # the resolved/transport zone never replaces the mic
    ("zone_cocina", None, "cocina"),                 # mic in transport form
    (None, "zone_living", None),                     # never the acoustic/transport zone alone
    ("garage", "zone_living", None),                 # unconfigured mic: no origin (P3F1-R3 option A)
])
async def test_dispatch_carries_the_normalized_origin_room(context_room_id, zone_id, expected):
    d = make_dispatcher(room_namespace=ROOMS)
    result = await d.dispatch(user_id="unknown", text="qué recordatorios hay", zone_id=zone_id,
                              context_room_id=context_room_id)
    assert result.path is PathType.FAST_REMINDER
    assert result.origin_room == expected
    assert result.zone_id == zone_id


def test_no_caller_can_name_an_origin_other_than_the_mic():
    """P3F1-R3 option A: dispatch and process take the mic room only, no resolved origin."""
    assert "origin_room" not in inspect.signature(RequestDispatcher.dispatch).parameters
    assert "origin_room" not in inspect.signature(MultiUserOrchestrator.process).parameters


async def test_special_commands_carry_the_origin_room_too():
    d = make_dispatcher(room_namespace=ROOMS)
    result = await d.dispatch(user_id="unknown", text="cancelá eso", zone_id="zone_hall", context_room_id="zone_bano")
    assert result.intent == "cancel"
    assert result.origin_room == "bano"


async def test_without_a_catalog_no_room_is_an_origin():
    d = make_dispatcher()
    assert d.room_namespace.rooms == frozenset()
    result = await d.dispatch(user_id="unknown", text="qué recordatorios hay", context_room_id="living")
    assert result.origin_room is None


async def test_orchestrator_injects_the_same_clock_and_namespace_and_forwards_the_origin():
    clock = HouseholdClock()
    orchestrator = MultiUserOrchestrator(chroma_sync=MagicMock(), ha_client=AsyncMock(),
                                         household_clock=clock, room_namespace=ROOMS)
    assert orchestrator.dispatcher.household_clock is clock
    assert orchestrator.context_manager.household_clock is clock
    assert orchestrator.dispatcher.room_namespace is ROOMS
    result = await orchestrator.process(user_id=None, text="qué recordatorios hay", zone_id="zone_hall",
                                        context_room_id="escritorio")
    assert result.origin_room == "escritorio"


async def test_forecast_day_uses_the_injected_zone_with_no_tz_load_per_turn(monkeypatch):
    clock = HouseholdClock()
    d = make_dispatcher(household_clock=clock)
    d.home_capabilities = SimpleNamespace(now=lambda: FROZEN)
    zones = []
    real_target = weather_module.forecast_target_date

    def target(dia, now, zone):
        zones.append(zone)
        return real_target(dia, now, zone)

    def no_tz_load(*args, **kwargs):
        raise AssertionError("timezone database loaded during a turn")

    monkeypatch.setattr(weather_module, "forecast_target_date", target)
    monkeypatch.setattr("zoneinfo.ZoneInfo", no_tz_load)
    monkeypatch.setattr("src.core.household_clock.ZoneInfo", no_tz_load)
    d.ha.call_service_with_response = AsyncMock(return_value=forecast(
        ("2026-08-06T15:00:00+00:00", "cloudy", 15), ("2026-08-05T15:00:00+00:00", "rainy", 18)))
    result = await d._handle_weather("va a llover mañana", Priority.HIGH)
    assert result.response == "Mañana: lluvioso, 18 grados."
    assert zones == [clock.zone]


async def test_weather_without_capabilities_falls_back_to_the_household_clock():
    d = make_dispatcher(household_clock=HouseholdClock(time_source=FROZEN.timestamp))
    assert d.home_capabilities is None
    d.ha.call_service_with_response = AsyncMock(return_value=forecast(
        ("2026-08-06T15:00:00+00:00", "cloudy", 15), ("2026-08-05T15:00:00+00:00", "rainy", 18)))
    result = await d._handle_weather("va a llover mañana", Priority.HIGH)
    assert result.response == "Mañana: lluvioso, 18 grados."
    assert result.observation.attempted_at == FROZEN == result.observation.observed_at


@pytest.fixture
def foreign_process_tz(monkeypatch):
    """Run with a process TZ that is not the household's (the server's is Montevideo)."""
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_slow_path_prompt_time_is_the_household_wall_time(foreign_process_tz):
    manager = ContextManager(household_clock=HouseholdClock(time_source=FROZEN.timestamp))
    manager.get_or_create("u1", "Juan")
    prompt = manager.build_prompt("u1", "qué hora es")
    stamp = next(line for line in prompt.splitlines() if line.startswith("Fecha y hora: "))
    # 02:30Z of the 5th is 23:30 of the 4th in Montevideo (11:30 of the 5th in Tokyo).
    assert " 04 de " in stamp and stamp.endswith(", 23:30")


def test_one_clock_is_shared_when_none_is_injected():
    orchestrator = MultiUserOrchestrator(chroma_sync=MagicMock(), ha_client=AsyncMock())
    assert isinstance(orchestrator.dispatcher.household_clock, HouseholdClock)
    assert orchestrator.context_manager.household_clock is orchestrator.dispatcher.household_clock
    dispatcher = RequestDispatcher(chroma_sync=MagicMock(), ha_client=AsyncMock(),
                                   router=None, llm=None, priority_queue=PriorityRequestQueue())
    assert dispatcher.context_manager.household_clock is dispatcher.household_clock
