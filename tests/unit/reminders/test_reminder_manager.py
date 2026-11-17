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
async def test_several_matches_are_ambiguous_even_when_one_is_exact(manager):
    """User answer T032-U1-Q1 = A: another reminder holding every spoken word means ask, never cancel."""
    cold = await _shared(manager, "tomar agua fría")
    gas = await _shared(manager, "tomar agua con gas", time.time() + 7200)
    tied = await manager.resolve(None, "de tomar agua", shared_only=True)
    assert tied.match is CancelMatch.AMBIGUOUS and [r.id for r in tied.candidates] == [cold.id, gas.id]
    assert await manager.cancel_by_text(None, "tomar agua", shared_only=True) is False
    plain = await _shared(manager, "tomar agua", time.time() + 9000)
    exact = await manager.resolve(None, "de tomar agua", shared_only=True)
    assert exact.match is CancelMatch.AMBIGUOUS and exact.reminder is None
    assert [r.id for r in exact.candidates] == [cold.id, gas.id, plain.id]
    assert await manager.cancel_by_text(None, "tomar agua", shared_only=True) is False
    assert [r.id for r in await manager.get_active(None, shared_only=True)] == [cold.id, gas.id, plain.id]
    alone = await manager.resolve(None, "de tomar agua con gas", shared_only=True)
    assert (alone.match, alone.reminder.id) == (CancelMatch.UNIQUE, gas.id)


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


_NIGHT_AND_NOON = ((11, 12, 0), (11, 14, 0), (11, 22, 0), (12, 0, 0), (12, 2, 0))


@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "expected"), [
    ("de tomar la pastilla de las 12 de la noche", {"00:00"}), ("de las doce de la noche", {"00:00"}),
    ("de las 12 de la madrugada", {"00:00"}), ("de las 12 del mediodía", {"12:00"}),
    ("de las 12 de la tarde", {"12:00"}), ("de las 12", {"00:00", "12:00"}),
    ("de las 2 de la noche", {"02:00"}), ("de las 2 de la tarde", {"14:00"}), ("de las 2", {"02:00", "14:00"}),
    ("de las 10 de la noche", {"22:00"}),
])
async def test_midnight_and_small_night_hours_are_past_midnight(store, query, expected):
    """'las 12 de la noche' is 00:00 and 'las 2 de la noche' 02:00 (review T032-U1-R1)."""
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)
    manager = ReminderManager(store=store, household_clock=clock)
    by_time = {}
    for day, hour, minute in _NIGHT_AND_NOON:
        reminder = await _shared(manager, "tomar la pastilla", _local(2026, 11, day, hour, minute).timestamp())
        by_time[reminder.id] = f"{hour:02d}:{minute:02d}"
    resolution = await manager.resolve(None, query, shared_only=True)
    found = {resolution.reminder.id} if resolution.reminder else {r.id for r in resolution.candidates}
    assert {by_time[i] for i in found} == expected
    assert resolution.match is (CancelMatch.UNIQUE if len(expected) == 1 else CancelMatch.AMBIGUOUS)


@pytest.mark.asyncio
@pytest.mark.parametrize(("literal", "other"), [
    ("tomar las 2 pastillas", "tomar las pastillas"), ("regar las 3 macetas", "regar las macetas"),
])
async def test_an_article_and_number_before_a_content_word_is_content(store, literal, other):
    """'tomar las 2 pastillas' names that reminder, not 'tomar las pastillas' at 14:00 (review T032-U1-R2)."""
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)
    manager = ReminderManager(store=store, household_clock=clock)
    named = await _shared(manager, literal, _local(2026, 11, 11, 9, 0).timestamp())
    await _shared(manager, other, _local(2026, 11, 11, 14, 0).timestamp())
    await _shared(manager, other, _local(2026, 11, 11, 15, 0).timestamp())
    found = await manager.resolve(None, f"de {literal}", shared_only=True)
    assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, named.id)
    # A time after the number is still read as one.
    timed = await manager.resolve(None, f"de {other} de las {literal.split()[2]}", shared_only=True)
    assert timed.match is CancelMatch.UNIQUE and timed.reminder.id != named.id


_DOCUMENTED_FORMS = ((12, 3, 0), (11, 15, 0), (11, 8, 15), (11, 8, 30), (11, 20, 30))


@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "expected"), [
    ("de las 3 de la madrugada", {"03:00"}), ("de las 3 de la tarde", {"15:00"}),
    ("de las 8 y cuarto", {"08:15"}), ("de las 8.30", {"08:30", "20:30"}), ("de las 8.30 de la noche", {"20:30"}),
    ("de tomar agua de las 8:30 de la mañana", {"08:30"}),
])
async def test_every_documented_clock_form_selects_its_time(store, query, expected):
    """'de la madrugada', 'y cuarto' and '8.30' are times (review T032-U1-R4)."""
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)
    manager = ReminderManager(store=store, household_clock=clock)
    by_time = {}
    for day, hour, minute in _DOCUMENTED_FORMS:
        reminder = await _shared(manager, "tomar agua", _local(2026, 11, day, hour, minute).timestamp())
        by_time[reminder.id] = f"{hour:02d}:{minute:02d}"
    resolution = await manager.resolve(None, query, shared_only=True)
    found = {resolution.reminder.id} if resolution.reminder else {r.id for r in resolution.candidates}
    assert {by_time[i] for i in found} == expected
    assert resolution.match is (CancelMatch.UNIQUE if len(expected) == 1 else CancelMatch.AMBIGUOUS)


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "query"), [("regar la flor", "de regar las flores"),
                                             ("regar las flores", "de regar la flor")])
async def test_an_es_plural_matches_its_singular(manager, text, query):
    """'flores' and 'flor' are one word (review T032-U1-R4)."""
    flower = await _shared(manager, text)
    await _shared(manager, "regar el pasto")
    found = await manager.resolve(None, query, shared_only=True)
    assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, flower.id)


@pytest.mark.asyncio
@pytest.mark.parametrize("number", ["25", "8:75"])
async def test_an_impossible_clock_time_is_a_word_of_the_reminder(store, number):
    """'las 25' is not a time, so it stays content and still names the reminder (review T032-U1-R4)."""
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)
    manager = ReminderManager(store=store, household_clock=clock)
    quota = await _shared(manager, f"pagar la cuota de las {number}", _local(2026, 11, 11, 9, 0).timestamp())
    await _shared(manager, "pagar la cuota", _local(2026, 11, 11, 9, 0).timestamp())
    found = await manager.resolve(None, f"de pagar la cuota de las {number}", shared_only=True)
    assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, quota.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "expected"), [
    ("de tomar agua de hoy", {11}), ("de tomar agua mañana", {12}), ("de tomar agua de pasado mañana", {13}),
    ("de tomar agua del lunes", {16}), ("de tomar agua de las 9 de hoy", {11}),
    ("de las 9 de la mañana de mañana", {12}), ("de las 9 mañana", {12}), ("de tomar agua de las 9 hoy", {11}),
    ("de tomar agua", {11, 12, 13, 16}),
    # T031 reads "de mañana" as the morning, not tomorrow: it stays a word and finds nothing.
    ("de tomar agua de mañana", set()),
])
async def test_a_spoken_day_selects_by_the_household_date(store, query, expected):
    """'de hoy', 'mañana', 'del lunes' use the T031 day grammar (review T032-U1-R6)."""
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)  # a Wednesday
    manager = ReminderManager(store=store, household_clock=clock)
    by_day = {}
    for day in (11, 12, 13, 16):
        reminder = await _shared(manager, "tomar agua", _local(2026, 11, day, 9, 0).timestamp())
        by_day[reminder.id] = day
    resolution = await manager.resolve(None, query, shared_only=True)
    found = {resolution.reminder.id} if resolution.reminder else {r.id for r in resolution.candidates}
    assert {by_day[i] for i in found} == expected
    assert resolution.match is {0: CancelMatch.NOT_FOUND, 1: CancelMatch.UNIQUE}.get(len(expected),
                                                                                       CancelMatch.AMBIGUOUS)


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "que te pedí de tomar agua", "que puse de tomar agua", "que me pediste de tomar agua", "que hay de tomar agua",
    "que tenía de tomar agua", "que anoté de tomar agua", "que agendé de tomar agua",
])
async def test_a_leading_relative_clause_is_not_content(manager, query):
    """'que te pedí', 'que puse' say how the reminder was made, not what it says (review T032-U1-R6)."""
    water = await _shared(manager, "tomar agua")
    await _shared(manager, "sacar la basura")
    found = await manager.resolve(None, query, shared_only=True)
    assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, water.id)


@pytest.mark.asyncio
async def test_a_relative_clause_verb_inside_the_text_is_still_content(manager):
    """Only a leading clause is dropped: 'pedí' later in the reference is a word to match."""
    await _shared(manager, "tomar agua")
    assert (await manager.resolve(None, "de tomar agua que pedí", shared_only=True)).match is CancelMatch.NOT_FOUND


@pytest.mark.asyncio
async def test_the_spoken_time_reference_resolves_to_exactly_its_reminder(store):
    """Review T032-U1-R7: the time a clarification proposes reads back as one trigger, any hour."""
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 0, 0).timestamp)
    manager = ReminderManager(store=store, household_clock=clock)
    reminders = [await _shared(manager, "tomar agua", _local(2026, 11, 11, hour, minute).timestamp())
                 for hour in range(24) for minute in (0, 30)]
    phrases = {manager.spoken_time_reference(r) for r in reminders}
    assert len(phrases) == len(reminders)
    assert {"las 12 de la noche", "la 1 de la madrugada", "las 8:30 de la mañana", "las 12 del mediodía",
            "las 7 de la tarde", "las 8 de la noche"} <= phrases
    for reminder in reminders:
        phrase = manager.spoken_time_reference(reminder)
        found = await manager.resolve(None, f"de tomar agua de {phrase}", shared_only=True)
        assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, reminder.id), phrase


# ---------------------------------------------------------------------------
# User answer T032-U1-Q1 = A: a clarification proposes only a command that
# selects exactly one reminder; one inside another at the same time has none.
# ---------------------------------------------------------------------------

def _minute_start(hours_ahead: int) -> float:
    return (int(time.time()) // 60 + 1) * 60 + hours_ahead * 3600


@pytest.mark.asyncio
async def test_a_superset_at_the_same_time_leaves_only_the_larger_reminder_nameable(manager):
    base = _minute_start(1)
    plain = await _shared(manager, "tomar agua", base)
    lemon = await _shared(manager, "tomar agua con limón", base + 30)
    candidates = (plain, lemon)
    spoken = manager.spoken_time_reference(lemon)
    assert spoken == manager.spoken_time_reference(plain)
    assert await manager.voice_references(None, candidates, with_time=True, shared_only=True) == {
        lemon.id: f"tomar agua con limón de {spoken}"}
    assert await manager.voice_references(None, candidates, with_time=False, shared_only=True) == {
        lemon.id: "tomar agua con limón"}
    said = await manager.resolve(None, f"de tomar agua de {spoken}", shared_only=True)
    assert said.match is CancelMatch.AMBIGUOUS and [r.id for r in said.candidates] == [plain.id, lemon.id]


@pytest.mark.asyncio
async def test_a_superset_at_another_time_leaves_both_nameable_with_the_time(manager):
    plain = await _shared(manager, "tomar agua", _minute_start(1))
    lemon = await _shared(manager, "tomar agua con limón", _minute_start(4))
    references = await manager.voice_references(None, (plain, lemon), with_time=True, shared_only=True)
    assert references == {plain.id: f"tomar agua de {manager.spoken_time_reference(plain)}",
                          lemon.id: f"tomar agua con limón de {manager.spoken_time_reference(lemon)}"}
    for reminder in (plain, lemon):
        found = await manager.resolve(None, f"de {references[reminder.id]}", shared_only=True)
        assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, reminder.id)
    assert await manager.voice_references(None, (plain, lemon), with_time=False, shared_only=True) == {
        lemon.id: "tomar agua con limón"}


@pytest.mark.asyncio
async def test_same_words_at_the_same_time_have_no_voice_reference(manager):
    base = _minute_start(1)
    one = await _shared(manager, "tomar agua", base)
    other = await _shared(manager, "tomar el agua", base + 30)
    assert await manager.voice_references(None, (one, other), with_time=True, shared_only=True) == {}


@pytest.mark.asyncio
async def test_a_reference_is_checked_against_every_active_reminder_not_only_the_candidates(manager):
    base = _minute_start(1)
    plain = await _shared(manager, "tomar agua", base)
    await _shared(manager, "tomar agua con limón", base + 30)
    assert await manager.voice_references(None, (plain,), with_time=True, shared_only=True) == {}
    assert await manager.voice_references(None, (plain,), with_time=False, shared_only=True) == {}


@pytest.mark.asyncio
async def test_voice_references_never_read_a_private_reminder(manager):
    base = _minute_start(1)
    plain = await _shared(manager, "tomar agua", base)
    await manager.create(user_id="ana", text="tomar agua con limón", trigger_at=base,
                         scope=ReminderScope.PRIVATE)
    assert await manager.voice_references(None, (plain,), with_time=True, shared_only=True) == {
        plain.id: f"tomar agua de {manager.spoken_time_reference(plain)}"}


# ---------------------------------------------------------------------------
# Review T032-U1-RR-F1: a clock or day word inside a reminder's own text
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "content"), [
    ("llamar a mamá a las 3", "llamar a mamá"),
    ("llevar el auto al taller mañana", "llevar el auto al taller"),
])
async def test_a_time_word_in_the_text_still_names_the_only_reminder(store, text, content):
    """The text said whole selects its reminder even when its time words fit no trigger."""
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)
    manager = ReminderManager(store=store, household_clock=clock)
    reminder = await _shared(manager, text, _local(2026, 11, 11, 17, 43).timestamp())
    for query in (f"de {text}", f"de {content}"):
        found = await manager.resolve(None, query, shared_only=True)
        assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, reminder.id), query
    assert await manager.voice_references(None, (reminder,), with_time=False, shared_only=True) == {
        reminder.id: text}
    assert await manager.voice_references(None, (reminder,), with_time=True, shared_only=True) == {
        reminder.id: f"{content} de las 5:43 de la tarde"}
    found = await manager.resolve(None, f"de {content} de las 5:43 de la tarde", shared_only=True)
    assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, reminder.id)


@pytest.mark.asyncio
async def test_time_words_are_text_only_when_no_trigger_fits(store):
    """A spoken time that fits a trigger still selects by time; one that fits none is matched as words."""
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)
    manager = ReminderManager(store=store, household_clock=clock)
    call = await _shared(manager, "llamar a mamá a las 3", _local(2026, 11, 11, 17, 43).timestamp())
    water = await _shared(manager, "tomar agua", _local(2026, 11, 11, 15, 0).timestamp())
    assert (await manager.resolve(None, "de llamar a mamá a las 4", shared_only=True)).match is CancelMatch.NOT_FOUND
    assert (await manager.resolve(None, "de tomar agua de las 9", shared_only=True)).match is CancelMatch.NOT_FOUND
    assert (await manager.resolve(None, "de tomar agua de hoy", shared_only=True)).reminder.id == water.id
    assert (await manager.resolve(None, "de tomar agua mañana", shared_only=True)).match is CancelMatch.NOT_FOUND
    found = await manager.resolve(None, "de llamar a mamá a las 3", shared_only=True)
    assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, call.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "content"), [
    ("llamar a las 3 a mamá", "llamar mamá"),
    ("pagar la luz el lunes a las 9", "pagar la luz"),
    ("tomar la pastilla de las 8 de la noche", "tomar la pastilla"),
    ("tomar agua", None),
    ("tomar las 2 pastillas", None),
    ("a las 3", None),
])
async def test_the_content_reference_drops_only_the_time_words(store, text, content):
    from src.reminders.reminder_manager import _without_time_words
    assert _without_time_words(text) == content


# ---------------------------------------------------------------------------
# Review T032-U1-RR-F2: the day tells apart reminders with the same text and time
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_spoken_day_reference_resolves_to_exactly_its_day(store):
    """'de hoy', the weekday for tomorrow (never 'de mañana'), 'de pasado mañana'."""
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)  # a Wednesday
    manager = ReminderManager(store=store, household_clock=clock)
    by_day = {day: await _shared(manager, "tomar agua", _local(2026, 11, day, 9, 0).timestamp())
              for day in (11, 12, 13, 16, 18)}
    assert {day: manager.spoken_day_reference(r) for day, r in by_day.items()} == {
        11: "de hoy", 12: "del jueves", 13: "de pasado mañana", 16: "del lunes", 18: "del miércoles"}
    for day in (11, 12, 13, 16):
        reminder = by_day[day]
        found = await manager.resolve(None, f"de tomar agua {manager.spoken_day_reference(reminder)}",
                                      shared_only=True)
        assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, reminder.id), day


@pytest.mark.asyncio
async def test_the_same_text_and_time_on_two_days_are_named_by_the_day(store):
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)
    manager = ReminderManager(store=store, household_clock=clock)
    today = await _shared(manager, "tomar agua", _local(2026, 11, 11, 9, 0).timestamp())
    tomorrow = await _shared(manager, "tomar agua", _local(2026, 11, 12, 9, 0).timestamp())
    references = await manager.voice_references(None, (today, tomorrow), with_time=True, shared_only=True)
    assert references == {today.id: "tomar agua de las 9 de la mañana de hoy",
                          tomorrow.id: "tomar agua de las 9 de la mañana del jueves"}
    for reminder in (today, tomorrow):
        found = await manager.resolve(None, f"de {references[reminder.id]}", shared_only=True)
        assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, reminder.id)


@pytest.mark.asyncio
async def test_a_superset_on_another_day_leaves_both_nameable(store):
    clock = HouseholdClock(time_source=_local(2026, 11, 11, 6, 0).timestamp)
    manager = ReminderManager(store=store, household_clock=clock)
    plain = await _shared(manager, "tomar agua", _local(2026, 11, 11, 9, 0).timestamp())
    lemon = await _shared(manager, "tomar agua con limón", _local(2026, 11, 12, 9, 0).timestamp())
    assert await manager.voice_references(None, (plain, lemon), with_time=True, shared_only=True) == {
        plain.id: "tomar agua de las 9 de la mañana de hoy", lemon.id: "tomar agua con limón de las 9 de la mañana"}
