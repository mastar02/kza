"""Voice creation and delivery use the persisted household room."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from src.core.household_clock import HouseholdClock, RoomNamespace
from src.core.task_outcome import PersistenceState, TaskStatus
from src.orchestrator.dispatcher import RequestDispatcher
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_scheduler import ReminderScheduler
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
from src.rooms.room_context import RoomConfig, RoomContextManager
from tests.mocks.responses import delivered

NOW = 1_795_000_000.0


def clock() -> float:
    return NOW


@pytest_asyncio.fixture
async def store(tmp_path):
    database = ReminderStore(str(tmp_path / "reminders.db"))
    await database.initialize()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
def dispatcher(store):
    rooms = RoomContextManager()
    rooms.add_room(RoomConfig("living", "Living", "el living", aliases=["sala", "sector"]))
    rooms.add_room(RoomConfig("cocina", "Cocina", "la cocina", aliases=["fogón", "sector"]))
    household_clock = HouseholdClock(time_source=clock)
    result = RequestDispatcher(
        chroma_sync=None, ha_client=None, household_clock=household_clock,
        reminder_manager=ReminderManager(store, household_clock=household_clock),
        room_namespace=RoomNamespace(rooms.get_all_rooms()),
    )
    result.reminder_rooms = rooms.get_all_rooms()
    return result


@pytest.mark.parametrize("voice_actions", [False, True])
@pytest.mark.parametrize(("text", "origin", "room"), [
    ("recordame en el fogon regar las plantas en diez minutos", "zone_living", "cocina"),
    ("recordame regar las plantas en diez minutos en el fogón", "living", "cocina"),
    ("recordame regar las plantas en diez minutos", "zone_cocina", "cocina"),
    ("recordame regar las plantas en diez minutos", None, None),
])
async def test_voice_creation_routes_only_to_the_saved_room(store, dispatcher, text, origin, room, voice_actions):
    dispatcher.reminder_manager._config["voice_actions"] = voice_actions
    result = await dispatcher.dispatch(user_id="unknown", text=text, context_room_id=origin, zone_id="zone_living")
    if voice_actions:
        assert result.reminder_commit is not None
        assert await store.count_active() == 0
        result = await result.reminder_commit()
    assert result.outcome.status is TaskStatus.COMPLETED
    reminder, = await store.get_active_for_user(None, shared_only=True)
    assert reminder.delivery_room == room
    assert reminder.text == "regar las plantas"

    tts = SimpleNamespace(aspeak=AsyncMock(return_value=delivered()))
    presence = Mock()
    scheduler = ReminderScheduler(store, tts, presence, None, config={"output_zone": "zone_living"},
                                  clock=lambda: reminder.trigger_at)
    await scheduler._deliver_due(reminder.trigger_at)
    tts.aspeak.assert_awaited_once_with("Oye, recuerda regar las plantas",
                                      zone_id="zone_cocina" if room else "zone_living")
    presence.get_user_zone.assert_not_called()


@pytest.mark.parametrize("text", [
    "recordame en el garaje regar las plantas en diez minutos",
    "recordame regar las plantas en diez minutos en el garaje",
    "recordame regar las plantas en diez minutos en la sala y en el fogón",
    "recordame regar las plantas en diez minutos en el fogón y el garaje",
    "recordame en el fogón y la sala regar las plantas en diez minutos",
    "recordame en el fogón y el garaje regar las plantas en diez minutos",
    "recordame en el fogón regar las plantas en la sala en diez minutos",
    "recordame regar las plantas en diez minutos en el sector",
])
async def test_unknown_or_conflicting_delivery_room_needs_input(store, dispatcher, text):
    result = await dispatcher.dispatch(user_id="unknown", text=text, context_room_id="zone_living")
    assert result.outcome.status is TaskStatus.NEEDS_INPUT
    assert result.outcome.persistence is PersistenceState.NOT_WRITTEN
    assert "¿En qué habitación" in result.response
    assert result.reminder_commit is None
    assert await store.count_active() == 0


async def test_due_notices_group_only_within_the_same_zone(store):
    for text, room in [("Aviso legado", None), ("Aviso de cocina", "cocina"), ("Aviso del living", "living")]:
        await store.create("household", text, NOW, scope=ReminderScope.SHARED, delivery_room=room)
    tts = SimpleNamespace(aspeak=AsyncMock(return_value=delivered()))
    scheduler = ReminderScheduler(store, tts, None, None, config={"output_zone": "zone_living"}, clock=clock)

    await scheduler._deliver_due(NOW)

    assert tts.aspeak.await_count == 2
    messages = {call.kwargs["zone_id"]: call.args[0] for call in tts.aspeak.await_args_list}
    assert "Aviso legado" in messages["zone_living"] and "Aviso del living" in messages["zone_living"]
    assert "Aviso de cocina" not in messages["zone_living"]
    assert messages["zone_cocina"] == "Oye, recuerda Aviso de cocina"
    rows = await store._work.read(lambda db: db.execute(
        "SELECT body, status FROM reminder_home_actions WHERE request_id LIKE 'notice:%'").fetchall())
    assert sorted((json.loads(row["body"])["room"], row["status"]) for row in rows) == [
        ("zone_cocina", "confirmed"), ("zone_living", "confirmed"), ("zone_living", "confirmed")]


async def test_targeted_reminder_can_fire_without_a_default_zone(store):
    legacy = await store.create("household", "Aviso legado", NOW - 1, scope=ReminderScope.SHARED)
    targeted = await store.create("household", "Aviso de cocina", NOW, scope=ReminderScope.SHARED,
                                  delivery_room="cocina")
    tts = SimpleNamespace(aspeak=AsyncMock(return_value=delivered()))
    scheduler = ReminderScheduler(store, tts, None, None, clock=clock)

    assert await scheduler._next_timeout() == 0
    await scheduler._deliver_due(NOW)

    tts.aspeak.assert_awaited_once_with("Oye, recuerda Aviso de cocina", zone_id="zone_cocina")
    assert (await store.get_by_id(targeted.id)).state is ReminderState.FIRED
    assert (await store.get_by_id(legacy.id)).state is ReminderState.ACTIVE
    assert await scheduler._next_timeout() is None
