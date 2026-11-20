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
async def test_personal_reminder_creation_info_omits_text_and_user_id(manager, caplog):
    person, text = "synthetic-owner-id", "texto privado del recordatorio"
    with caplog.at_level(logging.INFO, logger="src.reminders.reminder_manager"):
        await manager.create(person, text, time.time())
    messages = [record.getMessage() for record in caplog.records
                if record.name == "src.reminders.reminder_manager"]
    assert messages == ["Created personal reminder"]
    assert person not in caplog.text and text not in caplog.text

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
# T032-U1: unique-match resolution replaces the 0.4 similarity threshold.
# User decision t032_design_20261118: the candidates are every active shared
# reminder holding the base content words; time words and words no reminder
# holds never narrow; one candidate cancels, several are listed by number.
# ---------------------------------------------------------------------------

from src.reminders.reminder_manager import CancelMatch  # noqa: E402


async def _shared(manager, text, trigger_at=None):
    return await manager.create(user_id="household", text=text, trigger_at=trigger_at or time.time() + 3600,
                                scope=ReminderScope.SHARED)


async def _ids(manager, reference):
    resolution = await manager.resolve(None, reference, shared_only=True)
    found = [resolution.reminder.id] if resolution.reminder else [r.id for r in resolution.candidates]
    return resolution.match, found


@pytest.mark.asyncio
async def test_resolve_needs_every_known_content_word(manager):
    water = await _shared(manager, "Tomar agua")
    await _shared(manager, "sacar la basura")
    assert await _ids(manager, "de tomar el agua") == (CancelMatch.UNIQUE, [water.id])
    # Every word is known but no single text holds both.
    assert await _ids(manager, "de tomar la basura") == (CancelMatch.NOT_FOUND, [])


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["tomar aguja", "tomar awa", "toma agua", "agua mineral"])
async def test_a_word_no_reminder_holds_lists_the_similar_one_and_never_cancels_it(manager, query):
    """Review T032-NUM-FV-1: an unknown word is not ignored into a cancel.

    The one reminder holding the other words is listed for a numbered
    confirmation (``PARTIAL``), so a misheard word still finds it but never
    cancels it alone.
    """
    water = await _shared(manager, "tomar agua")
    trash = await _shared(manager, "sacar la basura", time.time() + 7200)
    assert await _ids(manager, query) == (CancelMatch.PARTIAL, [water.id])
    assert await manager.cancel_by_text(None, query, shared_only=True) is False
    assert [r.id for r in await manager.get_active(None, shared_only=True)] == [water.id, trash.id]


@pytest.mark.asyncio
@pytest.mark.parametrize(("query", "texts"), [
    ("de llamar a mamá", ("llamar a papá",)),
    ("de comprar leche", ("comprar pan", "regar las plantas")),
    ("de sacar al perro", ("sacar la basura",)),
    ("de pagar el agua", ("pagar la luz",)),
    ("de tomar pastillas", ("tomar agua",)),
])
async def test_a_word_no_reminder_holds_never_cancels_the_reminder_sharing_the_other_words(manager, query, texts):
    """Review T032-NUM-FV-1 probes: each of these cancelled the other reminder directly at 538ef6a1."""
    created = [await _shared(manager, text, time.time() + 3600 * (n + 1)) for n, text in enumerate(texts)]
    assert await _ids(manager, query) == (CancelMatch.PARTIAL, [created[0].id])
    assert await manager.cancel_by_text(None, query, shared_only=True) is False
    assert {r.id for r in await manager.get_active(None, shared_only=True)} == {r.id for r in created}


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["de comprar pan", "de comprar pan de las 5", "de la clave"])
async def test_a_reference_of_only_unknown_words_finds_nothing(manager, query):
    await _shared(manager, "tomar agua")
    assert await _ids(manager, query) == (CancelMatch.NOT_FOUND, [])
    assert await manager.cancel_by_text(None, query, shared_only=True) is False


@pytest.mark.asyncio
async def test_an_unknown_word_never_hides_a_second_similar_reminder(manager):
    cold = await _shared(manager, "tomar agua fría")
    gas = await _shared(manager, "tomar agua con gas", time.time() + 7200)
    assert await _ids(manager, "de tomar agua tibia") == (CancelMatch.PARTIAL, [cold.id, gas.id])


@pytest.mark.asyncio
async def test_accents_case_articles_and_plurals_are_ignored(manager):
    pill = await _shared(manager, "Darle la pastilla al perro")
    found = await manager.resolve(None, "que dice darle las pastíllas a los perros", shared_only=True)
    assert (found.match, found.reminder.id) == (CancelMatch.UNIQUE, pill.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "query"), [("regar la flor", "de regar las flores"),
                                             ("regar las flores", "de regar la flor")])
async def test_an_es_plural_matches_its_singular(manager, text, query):
    flower = await _shared(manager, text)
    assert await _ids(manager, query) == (CancelMatch.UNIQUE, [flower.id])


@pytest.mark.asyncio
async def test_several_candidates_are_ambiguous_even_when_one_is_exact(manager):
    """An exact text match cancels only when no other active reminder shares its base words."""
    cold = await _shared(manager, "tomar agua fría")
    gas = await _shared(manager, "tomar agua con gas", time.time() + 7200)
    assert await _ids(manager, "de tomar agua") == (CancelMatch.AMBIGUOUS, [cold.id, gas.id])
    assert await manager.cancel_by_text(None, "tomar agua", shared_only=True) is False
    plain = await _shared(manager, "tomar agua", time.time() + 9000)
    exact = await manager.resolve(None, "de tomar agua", shared_only=True)
    assert exact.match is CancelMatch.AMBIGUOUS and exact.reminder is None
    assert [r.id for r in exact.candidates] == [cold.id, gas.id, plain.id]
    assert await manager.cancel_by_text(None, "tomar agua", shared_only=True) is False
    assert [r.id for r in await manager.get_active(None, shared_only=True)] == [cold.id, gas.id, plain.id]
    assert await _ids(manager, "de tomar agua con gas") == (CancelMatch.UNIQUE, [gas.id])


@pytest.mark.asyncio
async def test_empty_reference_lists_every_candidate_and_no_active_says_so(manager):
    assert (await manager.resolve(None, "", shared_only=True)).match is CancelMatch.NONE_ACTIVE
    a = await _shared(manager, "a uno")
    b = await _shared(manager, "b dos", time.time() + 7200)
    assert await _ids(manager, "de los") == (CancelMatch.NO_REFERENCE, [a.id, b.id])


@pytest.mark.asyncio
async def test_resolution_never_sees_a_private_reminder(manager):
    await manager.create(user_id="ana", text="la clave", trigger_at=time.time() + 3600, scope=ReminderScope.PRIVATE)
    assert (await manager.resolve("ana", "la clave", shared_only=True)).match is CancelMatch.NONE_ACTIVE
    await _shared(manager, "tomar agua")
    assert (await manager.resolve("ana", "la clave", shared_only=True)).match is CancelMatch.NOT_FOUND


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "de las 8", "de las 8 de la mañana", "de las ocho de la noche", "a las 20", "de las 20:00", "de las 8 y media",
    "de la una", "de las 12 del mediodía", "de las 2 de la noche", "de hoy", "del lunes", "de mañana",
    "de esta noche", "a la tardecita", "a las 14h", "a las 2 p.m.", "del 20 de noviembre", "de la semana que viene",
])
async def test_a_spoken_time_alone_is_no_reference(store, query):
    """User decision t032_design_20261118: a spoken time never narrows, so it names no reminder."""
    manager = _manager_at(store, _local(2026, 11, 11, 6, 0))
    ids = [(await _shared(manager, "tomar agua", _local(2026, 11, 11, h, m).timestamp())).id
           for h, m in ((8, 0), (8, 30), (13, 0), (20, 0))]
    assert await _ids(manager, query) == (CancelMatch.NO_REFERENCE, ids)


@pytest.mark.asyncio
@pytest.mark.parametrize("said", [
    "a las 2", "de las 2", "a las dos", "de las 14", "a las 2 de la tarde", "a las 2 y 10", "a las 3 menos cuarto",
    "a las 2 en punto", "al mediodía", "a medianoche", "a la tarde", "por la noche", "mañana", "el lunes",
    "el lunes que viene", "el próximo lunes", "del 20", "a la hora de la cena", "a las 2 y pico", "a las 2pm",
    "a la tardecita", "mañana temprano", "el fin de semana", "todos los días",
])
async def test_a_time_phrase_never_narrows_even_when_it_is_a_text(store, said):
    """User decision t032_design_20261118: the reminder whose text holds the phrase and the plain one are both asked."""
    manager = _manager_at(store, _local(2026, 11, 11, 6, 0))
    literal = await _shared(manager, f"tomar pastillas {said}", _local(2026, 11, 11, 9, 45).timestamp())
    plain = await _shared(manager, "tomar pastillas", _local(2026, 11, 11, 14, 0).timestamp())
    assert await _ids(manager, f"de tomar pastillas {said}") == (CancelMatch.AMBIGUOUS, [literal.id, plain.id])
    assert await _ids(manager, "de tomar pastillas") == (CancelMatch.AMBIGUOUS, [literal.id, plain.id])
    assert await manager.cancel_by_text(None, f"tomar pastillas {said}", shared_only=True) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("texts", "query"), [
    (("tomar las 2 pastillas", "tomar las pastillas"), "de tomar las 2 pastillas"),
    (("tomar 10 gotas", "tomar 20 gotas"), "de tomar 10 gotas"),
    (("regar las 3 macetas", "regar las macetas"), "de regar las macetas"),
])
async def test_a_number_never_narrows(manager, texts, query):
    """A number may be a clock ("las 2"), so it never picks one of two similar texts (was review R2)."""
    first = await _shared(manager, texts[0])
    second = await _shared(manager, texts[1], time.time() + 7200)
    assert await _ids(manager, query) == (CancelMatch.AMBIGUOUS, [first.id, second.id])


@pytest.mark.asyncio
async def test_a_time_said_with_a_distinct_word_still_names_that_reminder(store):
    manager = _manager_at(store, _local(2026, 11, 11, 6, 0))
    await _shared(manager, "tomar agua", _local(2026, 11, 11, 8, 0).timestamp())
    gas = await _shared(manager, "tomar agua con gas", _local(2026, 11, 11, 20, 0).timestamp())
    assert await _ids(manager, "de tomar agua con gas de las 8 de la mañana") == (CancelMatch.UNIQUE, [gas.id])


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "que te pedí de tomar agua", "que puse de tomar agua", "que hay de tomar agua", "que tenía de tomar agua",
])
async def test_a_leading_relative_clause_is_not_content(manager, query):
    water = await _shared(manager, "tomar agua")
    await _shared(manager, "puse la mesa")
    assert await _ids(manager, query) == (CancelMatch.UNIQUE, [water.id])


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
@pytest.mark.parametrize(("days", "spoken"), [(0, "de hoy"), (1, "del jueves"), (2, "de pasado mañana"),
                                              (5, "del lunes")])
async def test_the_spoken_day_names_the_household_date(store, days, spoken):
    manager = _manager_at(store, _local(2026, 11, 11, 6, 0))  # a Wednesday
    reminder = await _shared(manager, "tomar agua", _local(2026, 11, 11 + days, 9, 0).timestamp())
    assert manager.spoken_day_reference(reminder) == spoken


@pytest.mark.parametrize("window", [0, -1, float("nan"), float("inf"), "30", True])
def test_the_voice_choice_window_must_be_a_positive_number(store, window):
    with pytest.raises(ValueError, match="voice_choice_window_s"):
        ReminderManager(store=store, config={"voice_choice_window_s": window})


def test_the_voice_choice_window_defaults_and_reads_the_settings(store):
    assert ReminderManager(store=store).choice_window_s == 30.0
    assert ReminderManager(store=store, config={"voice_choice_window_s": 12}).choice_window_s == 12.0


@pytest.mark.asyncio
@pytest.mark.parametrize(("days", "hour", "minute", "recurrence", "spoken"), [
    (0, 20, 0, None, "hoy a las 20:00"),
    (1, 7, 5, None, "mañana a las 07:05"),
    (2, 9, 0, None, "pasado mañana a las 09:00"),
    (3, 9, 0, None, "el sábado a las 09:00"),
    (6, 9, 0, None, "el martes a las 09:00"),
    (7, 9, 0, None, "el miércoles 18 a las 09:00"),
    (1, 8, 0, "daily", "todos los días a las 08:00"),
    (1, 20, 0, "weekdays", "de lunes a viernes a las 20:00"),
    (3, 10, 0, "weekly:6", "todos los sábados a las 10:00"),
    (5, 10, 0, "weekly:1", "todos los lunes a las 10:00"),
    (6, 10, 0, "weekly:7", "todos los domingos a las 10:00"),
    (1, 8, 0, "monthly:12", "mañana a las 08:00 y se repite"),
])
async def test_spoken_schedule_reads_back_the_household_day_and_time(store, foreign_process_tz, days, hour, minute,
                                                                     recurrence, spoken):
    """T031-U2: the voice read-back of a new reminder, on the household calendar (Wednesday 11 at 06:00)."""
    manager = _manager_at(store, _local(2026, 11, 11, 6, 0))
    reminder = await manager.create(user_id="household", text="x", recurrence=recurrence,
                                    trigger_at=_local(2026, 11, 11 + days, hour, minute).timestamp(),
                                    scope=ReminderScope.SHARED)
    assert manager.spoken_schedule(reminder) == spoken


@pytest.mark.asyncio
async def test_spoken_schedule_counts_days_on_the_household_calendar(store, foreign_process_tz):
    """At 23:30 local (02:30 UTC of the next day), 00:15 is "mañana", never "hoy"."""
    manager = ReminderManager(store=store, household_clock=HouseholdClock(time_source=UTC_NEXT_DAY.timestamp))
    reminder = await _shared(manager, "x", _local(2026, 11, 12, 0, 15).timestamp())
    assert manager.spoken_schedule(reminder) == "mañana a las 00:15"
    today = await _shared(manager, "y", _local(2026, 11, 11, 23, 45).timestamp())
    assert manager.spoken_schedule(today) == "hoy a las 23:45"
