"""Tests for ReminderManager — CRUD and cancellation logic."""
import logging
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo
from tests.mocks.home_control import selected_actions
import pytest
import pytest_asyncio

from src.core.household_clock import HouseholdClock
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_store import ReminderStore, ReminderScope


@pytest_asyncio.fixture
async def store(tmp_path):
    s = ReminderStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()

@pytest_asyncio.fixture
async def manager(store):
    config = {"default_time": "09:00", "tts_prefix": "Oye, recuerda"}
    return ReminderManager(store=store, config=config, home_actions=selected_actions(MagicMock(), ("light.patio",)))

@pytest.mark.asyncio
async def test_create_one_shot(manager):
    r = await manager.create(user_id="u1", text="dentista", trigger_at=time.time() + 3600)
    assert r.text == "dentista"
    assert r.recurrence is None

@pytest.mark.asyncio
async def test_create_recurring(manager):
    r = await manager.create(user_id="u1", text="pastilla", trigger_at=time.time() + 3600, recurrence="daily")
    assert r.recurrence == "daily"

@pytest.mark.asyncio
async def test_create_with_ha_actions(manager):
    actions = [{"domain": "light", "service": "turn_on", "entity_id": "light.patio"}]
    r = await manager.create(user_id="u1", text="regar", trigger_at=time.time() + 3600, ha_actions=actions)
    assert len(r.ha_actions) == 1

@pytest.mark.asyncio
async def test_cancel_by_text(manager):
    await manager.create(user_id="u1", text="sacar la basura", trigger_at=time.time() + 3600)
    result = await manager.cancel_by_text(user_id="u1", search_text="basura")
    assert result is True

@pytest.mark.asyncio
async def test_cancel_nonexistent(manager):
    result = await manager.cancel_by_text(user_id="u1", search_text="inexistente")
    assert result is False

@pytest.mark.asyncio
async def test_get_active(manager):
    await manager.create(user_id="u1", text="a", trigger_at=time.time() + 100)
    await manager.create(user_id="u1", text="b", trigger_at=time.time() + 200)
    active = await manager.get_active(user_id="u1")
    assert len(active) == 2

MVD = ZoneInfo("America/Montevideo")


@pytest.fixture
def foreign_process_tz(monkeypatch):
    """Run with a process TZ that is not the household's (the server's is Montevideo)."""
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _local(*args) -> datetime:
    """Household wall time (P3-F1), independent of the process's TZ."""
    return datetime(*args, tzinfo=MVD)


def _manager_at(store, local_now: datetime) -> ReminderManager:
    """Manager whose clock is frozen at a household wall-clock instant."""
    return ReminderManager(store=store, config={}, clock=lambda: local_now.timestamp())


@pytest.mark.asyncio
async def test_get_today(store):
    # "now + offset" is only "today" when the injected clock leaves room for
    # the offset before local midnight; the wall clock gave no such guarantee.
    now = _local(2026, 11, 11, 12, 0).timestamp()
    manager = _manager_at(store, _local(2026, 11, 11, 12, 0))
    await manager.create(user_id="u1", text="hoy", trigger_at=now + 100)
    await manager.create(user_id="u1", text="manana", trigger_at=now + 90000)
    today = await manager.get_today(user_id="u1")
    assert len(today) == 1
    assert today[0].text == "hoy"


@pytest.mark.asyncio
async def test_get_today_uses_the_injected_clock_at_the_day_boundary(store):
    """Regression: the shard-a run reached get_today at 23:58:20 local."""
    manager = _manager_at(store, _local(2026, 11, 11, 23, 58, 20))
    await manager.create(user_id="u1", text="antes", trigger_at=_local(2026, 11, 11, 23, 59, 59).timestamp())
    await manager.create(user_id="u1", text="medianoche", trigger_at=_local(2026, 11, 12, 0, 0).timestamp())
    await manager.create(user_id="u1", text="vencido", trigger_at=_local(2026, 11, 10, 8, 0).timestamp())
    assert [r.text for r in await manager.get_today(user_id="u1")] == ["vencido", "antes"]
    after_midnight = _manager_at(store, _local(2026, 11, 12, 0, 0, 1))
    assert [r.text for r in await after_midnight.get_today(user_id="u1")] == ["vencido", "antes", "medianoche"]

@pytest.mark.asyncio
async def test_format_for_voice(manager):
    r = await manager.create(user_id="u1", text="tomar agua", trigger_at=time.time() + 3600, scope=ReminderScope.SHARED)
    voice = manager.format_for_voice(r)
    assert "tomar agua" in voice


# P3-F1: "today" and spoken times are on the injected household clock.
UTC_NEXT_DAY = datetime(2026, 11, 12, 2, 30, tzinfo=UTC)  # 23:30 of the 11th in Montevideo


@pytest.mark.asyncio
async def test_today_is_the_household_day_across_the_utc_midnight(store, foreign_process_tz):
    manager = ReminderManager(store=store, household_clock=HouseholdClock(time_source=UTC_NEXT_DAY.timestamp))
    await manager.create(user_id="u1", text="hoy", trigger_at=_local(2026, 11, 11, 23, 45).timestamp())
    await manager.create(user_id="u1", text="manana", trigger_at=_local(2026, 11, 12, 0, 15).timestamp())
    # On the UTC calendar both would be "the 12th"; the household is still on the 11th.
    assert [r.text for r in await manager.get_today(user_id="u1")] == ["hoy"]


@pytest.mark.asyncio
async def test_spoken_and_logged_times_are_household_wall_times(store, caplog, foreign_process_tz):
    manager = ReminderManager(store=store, config={"tts_prefix": "Recuerda"},
                              household_clock=HouseholdClock(time_source=UTC_NEXT_DAY.timestamp))
    with caplog.at_level(logging.DEBUG, logger="src.reminders.reminder_manager"):
        r = await manager.create(user_id="household", text="sacar la basura", trigger_at=UTC_NEXT_DAY.timestamp(),
                                 scope=ReminderScope.SHARED)
    assert manager.format_for_voice(r) == "Recuerda sacar la basura a las 23:30"
    info = [record.getMessage() for record in caplog.records if record.levelno >= logging.INFO]
    assert len(info) == 1 and r.id in info[0] and "23:30" in info[0]
    # Household text is never logged above DEBUG (design household_conventions.logging).
    assert "basura" not in info[0]


def test_one_clock_source_only(store):
    with pytest.raises(ValueError):
        ReminderManager(store=store, clock=time.time, household_clock=HouseholdClock())
    clock = HouseholdClock()
    assert ReminderManager(store=store, household_clock=clock)._household_clock is clock


# ---------------------------------------------------------------------------
# T032-U1: unique-match resolution replaces the 0.4 similarity threshold
# ---------------------------------------------------------------------------

from src.reminders.reminder_manager import CancelMatch  # noqa: E402


async def _shared(manager, text, trigger_at=None):
    return await manager.create(user_id="household", text=text, trigger_at=trigger_at or time.time() + 3600,
                                scope=ReminderScope.SHARED)


@pytest.mark.asyncio
async def test_resolve_needs_every_content_word(manager):
    water = await _shared(manager, "Tomar agua")
    await _shared(manager, "sacar la basura")
    unique = await manager.resolve(None, "de tomar el agua", shared_only=True)
    assert (unique.match, unique.reminder.id) == (CancelMatch.UNIQUE, water.id)
    assert (await manager.resolve(None, "de tomar agua fría", shared_only=True)).match is CancelMatch.NOT_FOUND


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["tomar aguja", "tomar awa", "toma agua", "agua mineral"])
async def test_a_misheard_word_finds_nothing(manager, query):
    """The old SequenceMatcher >= 0.4 picked the neighbouring reminder for these."""
    await _shared(manager, "tomar agua")
    assert (await manager.resolve(None, query, shared_only=True)).match is CancelMatch.NOT_FOUND
    assert await manager.cancel_by_text(None, query, shared_only=True) is False


@pytest.mark.asyncio
async def test_accents_case_articles_and_plurals_are_ignored(manager):
    pill = await _shared(manager, "Darle la pastilla al perro")
    found = await manager.resolve(None, "que dice darle las pastíllas a los perros", shared_only=True)
    assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, pill.id)


@pytest.mark.asyncio
async def test_several_matches_are_ambiguous_unless_one_is_exact(manager):
    cold = await _shared(manager, "tomar agua fría")
    gas = await _shared(manager, "tomar agua con gas", time.time() + 7200)
    tied = await manager.resolve(None, "de tomar agua", shared_only=True)
    assert tied.match is CancelMatch.AMBIGUOUS and [r.id for r in tied.candidates] == [cold.id, gas.id]
    assert await manager.cancel_by_text(None, "tomar agua", shared_only=True) is False
    plain = await _shared(manager, "tomar agua", time.time() + 9000)
    exact = await manager.resolve(None, "de tomar agua", shared_only=True)
    assert (exact.match, exact.reminder.id) == (CancelMatch.UNIQUE, plain.id)


@pytest.mark.asyncio
async def test_empty_reference_lists_every_candidate_and_no_active_says_so(manager):
    assert (await manager.resolve(None, "", shared_only=True)).match is CancelMatch.NONE_ACTIVE
    a = await _shared(manager, "a uno")
    b = await _shared(manager, "b dos", time.time() + 7200)
    empty = await manager.resolve(None, "de los", shared_only=True)
    assert empty.match is CancelMatch.NO_REFERENCE and [r.id for r in empty.candidates] == [a.id, b.id]


@pytest.mark.asyncio
async def test_resolution_never_sees_a_private_reminder(manager):
    await manager.create(user_id="ana", text="la clave", trigger_at=time.time() + 3600, scope=ReminderScope.PRIVATE)
    assert (await manager.resolve("ana", "la clave", shared_only=True)).match is CancelMatch.NONE_ACTIVE
    await _shared(manager, "tomar agua")
    assert (await manager.resolve("ana", "la clave", shared_only=True)).match is CancelMatch.NOT_FOUND


@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "expected"), [
    ("de las 8", {"08:00", "20:00"}), ("de las 8 de la mañana", {"08:00"}), ("de las 8 de la noche", {"20:00"}),
    ("de las ocho de la noche", {"20:00"}), ("a las 20", {"20:00"}), ("de las 20:00", {"20:00"}),
    ("de las 8 y media", {"08:30"}), ("de la una", {"13:00"}), ("de las 9", set()),
    ("de tomar agua a las 8 de la mañana", {"08:00"}), ("de tomar agua de las 8:30", {"08:30"}),
])
async def test_a_spoken_clock_time_selects_by_the_household_time(store, query, expected):
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)
    manager = ReminderManager(store=store, household_clock=clock)
    by_time = {}
    for hour, minute in ((8, 0), (20, 0), (8, 30), (13, 0)):
        reminder = await _shared(manager, "tomar agua", _local(2026, 11, 11, hour, minute).timestamp())
        by_time[reminder.id] = f"{hour:02d}:{minute:02d}"
    resolution = await manager.resolve(None, query, shared_only=True)
    found = {resolution.reminder.id} if resolution.reminder else {r.id for r in resolution.candidates}
    assert {by_time[i] for i in found} == expected
    assert resolution.match is {0: CancelMatch.NOT_FOUND, 1: CancelMatch.UNIQUE}.get(len(expected),
                                                                                       CancelMatch.AMBIGUOUS)


@pytest.mark.asyncio
async def test_cancel_resolved_reports_only_a_stored_transition(manager, store):
    reminder = await _shared(manager, "tomar agua")
    assert await manager.cancel_resolved(reminder, shared_only=True) is True
    assert await manager.cancel_resolved(reminder, shared_only=True) is False
    assert (await store.get_by_id(reminder.id)).state == "cancelled"


@pytest.mark.asyncio
async def test_voice_label_is_shared_only(manager):
    shared = await _shared(manager, "tomar agua")
    assert manager.voice_label(shared).startswith("tomar agua a las ")
    private = await manager.create(user_id="ana", text="la clave", trigger_at=time.time() + 60)
    with pytest.raises(PermissionError):
        manager.voice_label(private)


@pytest.mark.asyncio
async def test_a_bare_number_is_content_not_a_clock_time(store):
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)
    manager = ReminderManager(store=store, household_clock=clock)
    pills = await _shared(manager, "darle 2 pastillas al perro", _local(2026, 11, 11, 8, 0).timestamp())
    found = await manager.resolve(None, "de darle 2 pastillas al perro", shared_only=True)
    assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, pills.id)
