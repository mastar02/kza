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
from tests.mocks.home_control import selected_actions
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
    rooms.add_room(RoomConfig("living", "Living", "el living", aliases=["sala", "sector"],
                              default_light="light.living"))
    rooms.add_room(RoomConfig("cocina", "Cocina", "la cocina", aliases=["fogón", "sector"],
                              default_light="light.cocina"))
    household_clock = HouseholdClock(time_source=clock)
    actions = selected_actions(SimpleNamespace(), ("light.living", "light.cocina"))
    result = RequestDispatcher(
        chroma_sync=None, ha_client=None, household_clock=household_clock, home_actions=actions,
        reminder_manager=ReminderManager(store, home_actions=actions, household_clock=household_clock),
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


@pytest.mark.parametrize("configured_rooms", [False, True])
@pytest.mark.parametrize("voice_actions", [False, True])
@pytest.mark.parametrize(("text", "expected"), [
    ("recordame dejar las llaves en la mesa en diez minutos", "dejar las llaves en la mesa"),
    ("recordame en diez minutos comprar pan en el super", "comprar pan en el super"),
    ("recordame en diez minutos pensar en vos", "pensar en vos"),
    ("recordame en diez minutos poner la ropa en el lavarropas", "poner la ropa en el lavarropas"),
    ("recordame en diez minutos sacar la ropa en el patio", "sacar la ropa en el patio"),
    ("recordame en la mesa dejar las llaves en diez minutos", "en la mesa dejar las llaves"),
    ("recordame dejar la leche en la cocina y sacar la basura en diez minutos",
     "dejar la leche en la cocina y sacar la basura"),
])
async def test_reminder_preserves_ordinary_locatives(store, dispatcher, configured_rooms, voice_actions, text, expected):
    if not configured_rooms:
        dispatcher.room_namespace = RoomNamespace(())
        dispatcher.reminder_rooms = {}
    dispatcher.reminder_manager._config["voice_actions"] = voice_actions

    result = await dispatcher.dispatch(user_id="unknown", text=text, context_room_id="zone_living")
    if voice_actions:
        assert result.reminder_commit is not None
        result = await result.reminder_commit()

    assert result.outcome.status is TaskStatus.COMPLETED
    reminder, = await store.get_active_for_user(None, shared_only=True)
    assert reminder.text == expected
    assert reminder.ha_actions is None
    assert reminder.delivery_room == ("living" if configured_rooms else None)


@pytest.mark.parametrize("voice_actions", [False, True])
@pytest.mark.parametrize("text", [
    "recordame en el garaje regar las plantas en diez minutos",
    "recordame regar las plantas en diez minutos en el garaje",
    "recordame regar las plantas en diez minutos en la terraza",
    "recordame en el dormitorio principal regar las plantas en diez minutos",
    "recordame regar las plantas en diez minutos en la sala y en el fogón",
    "recordame regar las plantas en la cocina y el living en diez minutos",
    "recordame regar las plantas en diez minutos en el fogón y el garaje",
    "recordame en el fogón y la sala regar las plantas en diez minutos",
    "recordame en el fogón y el garaje regar las plantas en diez minutos",
    "recordame en el fogón regar las plantas en la sala en diez minutos",
    "recordame regar las plantas en diez minutos en el sector",
    "recordame apagar las luces en la cocina o el living en diez minutos",
    "recordame apagar las luces en diez minutos en la cocina y el living",
])
async def test_unknown_or_conflicting_delivery_room_needs_input(store, dispatcher, text, voice_actions):
    dispatcher.reminder_manager._config["voice_actions"] = voice_actions
    result = await dispatcher.dispatch(user_id="unknown", text=text, context_room_id="zone_living")
    assert result.outcome.status is TaskStatus.NEEDS_INPUT
    assert result.outcome.persistence is PersistenceState.NOT_WRITTEN
    assert "¿En qué habitación" in result.response
    assert result.reminder_commit is None
    assert await store._work.read(lambda db: db.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]) == 0


@pytest.mark.parametrize("text", [
    "apagar las luces en la cocina y el living",
    "apagar las luces en la cocina y en el living",
    "apagar la luz en el fogón y apagar la luz en el living",
])
async def test_readback_and_commit_keep_action_rooms_separate_from_the_delivery_room(store, dispatcher, text):
    dispatcher.reminder_manager._config["voice_actions"] = True
    result = await dispatcher.dispatch(user_id="unknown", text=f"recordame {text} en diez minutos",
                                       context_room_id="zone_living")

    assert result.reminder_commit is not None
    assert result.response.startswith("Voy a ")
    assert "en la cocina" in result.response and "en el living" in result.response
    assert await store.count_active() == 0
    committed = await result.reminder_commit()

    assert committed.outcome.status is TaskStatus.COMPLETED
    reminder, = await store.get_active_for_user(None, shared_only=True)
    assert reminder.text == text
    assert reminder.delivery_room == "living"
    assert reminder.ha_actions == [
        {"domain": "light", "service": "turn_off", "entity_id": "light.cocina", "data": None},
        {"domain": "light", "service": "turn_off", "entity_id": "light.living", "data": None},
    ]
    dispatcher.home_actions.ha.call_service_once.assert_not_awaited()


@pytest.mark.parametrize("output_zone", ["zone_living", "living"])
async def test_due_notices_group_only_within_the_same_zone(store, output_zone):
    for text, room in [("Aviso legado", None), ("Aviso de cocina", "cocina"), ("Aviso del living", "living")]:
        await store.create("household", text, NOW, scope=ReminderScope.SHARED, delivery_room=room)
    tts = SimpleNamespace(aspeak=AsyncMock(return_value=delivered()))
    scheduler = ReminderScheduler(store, tts, None, None, config={"output_zone": output_zone}, clock=clock)

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
