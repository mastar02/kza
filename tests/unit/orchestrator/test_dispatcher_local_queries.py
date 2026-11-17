"""T037-U3: time and date answered from the household clock, never by a language model."""

import asyncio
import threading
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from src.core.household_clock import HouseholdClock
from src.core.task_outcome import DispatchState, EffectState, PersistenceState, TaskStatus
from src.llm.reasoner import FastRouter
from src.nlu.command_grammar import WAKE_TERMS
from src.nlu.household_intents import HouseholdIntentClassifier, HouseholdIntentSettings
from src.orchestrator.context_manager import ContextManager
from src.orchestrator.dispatcher import PathType, Priority, RequestDispatcher
from src.orchestrator.priority_queue import PriorityRequestQueue
from src.world.household_time import (
    CLOCK_UNAVAILABLE, ClockQuery, ClockQueryKind, describe_clock, parse_clock_query,
)

# 02:59:59Z on the 18th is 23:59:59 on Tuesday the 17th in Montevideo (UTC-3);
# one second later the household is at midnight of Wednesday the 18th.
BEFORE_MIDNIGHT = datetime(2026, 11, 18, 2, 59, 59, tzinfo=UTC)
AT_MIDNIGHT = BEFORE_MIDNIGHT + timedelta(seconds=1)


# T037-U3-R1: the wake words main configures (rooms.wake_word.words). The
# transcript keeps the wake pre-roll, which Whisper often hears as an alias.
with open("config/settings.yaml") as _fh:
    CONFIGURED_WAKE_WORDS = tuple(yaml.safe_load(_fh)["rooms"]["wake_word"]["words"])
CLOCK_WAKES = tuple(sorted(WAKE_TERMS | set(CONFIGURED_WAKE_WORDS)))


def configured_intents() -> HouseholdIntentClassifier:
    """The classifier main builds from rooms.wake_word.words."""
    return HouseholdIntentClassifier(HouseholdIntentSettings.from_config(None, wake_words=CONFIGURED_WAKE_WORDS))


def clock_at(moment: datetime) -> HouseholdClock:
    return HouseholdClock(time_source=lambda: moment.timestamp())


def spy_router() -> MagicMock:
    router = MagicMock(spec=FastRouter)
    router.complete = AsyncMock(return_value="Son las 12:00, según el modelo.")
    router.generate = MagicMock(return_value=["Son las 12:00, según el modelo."])
    return router


def make_dispatcher(clock: HouseholdClock, router=None, **kwargs) -> RequestDispatcher:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=AssertionError("cloud reasoner called"))
    chroma = MagicMock()
    chroma.asearch_command = AsyncMock(side_effect=AssertionError("vector search called"))
    return RequestDispatcher(
        chroma_sync=chroma, ha_client=AsyncMock(), routine_manager=MagicMock(),
        router=router if router is not None else spy_router(), llm=llm,
        context_manager=ContextManager(household_clock=clock), priority_queue=PriorityRequestQueue(),
        household_clock=clock, **kwargs,
    )


def assert_no_model_and_no_effect(d: RequestDispatcher) -> None:
    d.router.complete.assert_not_called()
    d.router.generate.assert_not_called()
    d.llm.complete.assert_not_called()
    d.chroma.asearch_command.assert_not_called()
    assert d.ha.mock_calls == []
    assert len(d.queue) == 0


TIME_PHRASES = [
    "¿Qué hora es?", "que hora es", "QUÉ HORA ES", "Nexa, ¿qué hora es?", "¿Qué horas son?",
    "decime la hora", "decime la hora, por favor", "¿me decís la hora?", "¿sabés qué hora es?",
    "¿qué hora es ahora?", "che, ¿qué hora tenés?", "¿me podés decir qué hora es?", "dime la hora",
    # Wake pre-roll as Whisper hears it (T037-U3-R1).
    "Alexa, ¿qué hora es?", "alexa que hora es", "nexia, decime la hora", "kaza qué hora es",
    "nexta, decime la hora", "neska qué hora es", "Nexca, ¿qué horas son?", "¿qué hora es, alexa?",
    "che nexia, ¿qué hora es?",
]
DATE_PHRASES = [
    ("¿Qué día es hoy?", 0, False), ("qué día es", 0, False), ("¿Qué día es mañana?", 1, False),
    ("¿qué día es pasado mañana?", 2, False), ("¿Qué día fue ayer?", -1, False), ("¿qué día era ayer?", -1, False),
    ("hoy qué día es", 0, False), ("¿En qué día estamos?", 0, False), ("¿Qué día estamos hoy?", 0, False),
    ("¿A cuánto estamos hoy?", 0, False), ("¿Qué día de la semana es hoy?", 0, False),
    ("¿Qué fecha es hoy?", 0, True), ("¿A qué fecha estamos?", 0, True), ("decime la fecha de hoy", 0, True),
    ("¿qué fecha es mañana?", 1, True), ("Nexa, ¿qué día es hoy?", 0, False),
    ("kaza qué día es hoy", 0, False), ("Nexia, ¿qué día es hoy?", 0, False), ("Alexa, ¿qué día es mañana?", 1, False),
    ("neska, ¿qué fecha es hoy?", 0, True), ("¿qué día es hoy, nexta?", 0, False),
]
# Questions with another subject, place or day stay on their existing paths.
NOT_CLOCK = [
    "¿A qué hora es el partido?", "a qué hora es", "¿qué hora es en Madrid?", "¿qué día es Navidad?",
    "¿qué día es el partido?", "qué día es el 25", "¿qué día juega Peñarol?", "hoy qué día es mañana",
    "la hora", "¿qué hora y qué día es?", "¿cuántas horas faltan?", "hola", "recordame qué hora es",
    "poné un temporizador de una hora", "¿a qué hora sale el sol?", "prendé la luz", "",
    "alexa, ¿qué hora es en Madrid?", "kaza, ¿a qué hora es el partido?", "nexia", "alexa kaza",
]


@pytest.mark.parametrize("text", TIME_PHRASES)
def test_time_phrases_are_time_queries(text):
    assert parse_clock_query(text, CLOCK_WAKES) == ClockQuery(ClockQueryKind.TIME)


@pytest.mark.parametrize(("text", "offset", "with_year"), DATE_PHRASES)
def test_date_phrases_name_the_day_asked(text, offset, with_year):
    assert parse_clock_query(text, CLOCK_WAKES) == ClockQuery(ClockQueryKind.DATE, offset, with_year)


@pytest.mark.parametrize("text", NOT_CLOCK)
def test_other_questions_are_not_clock_queries(text):
    assert parse_clock_query(text, CLOCK_WAKES) is None


def test_the_configured_wake_words_include_the_aliases_whisper_hears():
    assert {"nexa", "alexa", "nexia", "nexta", "kaza"} <= set(CONFIGURED_WAKE_WORDS)


@pytest.mark.parametrize("text", TIME_PHRASES + [phrase for phrase, _, _ in DATE_PHRASES])
def test_dispatcher_built_like_main_routes_every_clock_phrase_to_the_clock(text):
    """T037-U3-R1: wake aliases must not send a clock question to the cloud reasoner."""
    d = make_dispatcher(clock_at(BEFORE_MIDNIGHT), household_intents=configured_intents())
    assert d._classify_request(text.lower()) == (PathType.FAST_CLOCK, Priority.HIGH)


@pytest.mark.parametrize("text", ["Alexa, ¿qué hora es?", "nexia qué día es hoy", "kaza qué hora es",
                                  "nexta, decime la hora", "neska qué hora es"])
async def test_wake_aliases_are_answered_by_the_clock_not_the_cloud(text):
    d = make_dispatcher(clock_at(BEFORE_MIDNIGHT), household_intents=configured_intents())
    result = await d.dispatch(user_id="unknown", text=text, zone_id="zone_living")
    assert (result.path, result.intent, result.success) == (PathType.FAST_CLOCK, "clock", True)
    assert result.response in ("Son las 23:59.", "Hoy es martes 17 de noviembre.")
    assert_no_model_and_no_effect(d)


@pytest.mark.parametrize("text", ["Alexa, ¿qué hora es?", "nexia qué día es hoy", "nexta, decime la hora"])
async def test_grammar_wake_terms_are_stripped_without_configured_words(text):
    d = make_dispatcher(clock_at(BEFORE_MIDNIGHT))
    result = await d.dispatch(user_id="unknown", text=text, zone_id="zone_living")
    assert result.path is PathType.FAST_CLOCK and result.success is True
    assert_no_model_and_no_effect(d)


def test_a_non_string_is_not_a_clock_query():
    assert parse_clock_query(None) is None


@pytest.mark.parametrize(("hour", "minute", "spoken"), [
    (0, 0, "Son las 0:00."), (0, 5, "Son las 0:05."), (1, 0, "Es la 1:00."), (1, 59, "Es la 1:59."),
    (2, 0, "Son las 2:00."), (13, 7, "Son las 13:07."), (23, 59, "Son las 23:59."),
])
def test_time_is_spoken_in_household_24_hour_form(hour, minute, spoken):
    now = datetime(2026, 11, 17, hour, minute, 42, tzinfo=timezone(timedelta(hours=-3)))
    assert describe_clock(ClockQuery(ClockQueryKind.TIME), now) == spoken


@pytest.mark.parametrize(("offset", "with_year", "spoken"), [
    (0, False, "Hoy es martes 17 de noviembre."),
    (1, False, "Mañana es miércoles 18 de noviembre."),
    (2, False, "Pasado mañana es jueves 19 de noviembre."),
    (-1, False, "Ayer fue lunes 16 de noviembre."),
    (0, True, "Hoy es martes 17 de noviembre de 2026."),
])
def test_date_is_spoken_with_fixed_spanish_names(offset, with_year, spoken):
    now = datetime(2026, 11, 17, 12, 0, tzinfo=timezone(timedelta(hours=-3)))
    assert describe_clock(ClockQuery(ClockQueryKind.DATE, offset, with_year), now) == spoken


@pytest.mark.parametrize(("moment", "spoken"), [
    (datetime(2026, 12, 31, 23, 30), "Mañana es viernes 1 de enero de 2027."),
    (datetime(2028, 3, 1, 8, 0), "Ayer fue martes 29 de febrero de 2028."),
])
def test_dates_cross_month_year_and_leap_day(moment, spoken):
    offset = 1 if spoken.startswith("Mañana") else -1
    now = moment.replace(tzinfo=timezone(timedelta(hours=-3)))
    assert describe_clock(ClockQuery(ClockQueryKind.DATE, offset, True), now) == spoken


def test_a_naive_time_is_refused():
    with pytest.raises(ValueError):
        describe_clock(ClockQuery(ClockQueryKind.TIME), datetime(2026, 11, 17, 12, 0))


@pytest.mark.parametrize(("moment", "text", "spoken"), [
    (BEFORE_MIDNIGHT, "¿Qué hora es?", "Son las 23:59."),
    (BEFORE_MIDNIGHT, "¿Qué día es hoy?", "Hoy es martes 17 de noviembre."),
    (BEFORE_MIDNIGHT, "¿qué día es mañana?", "Mañana es miércoles 18 de noviembre."),
    (AT_MIDNIGHT, "¿Qué hora es?", "Son las 0:00."),
    (AT_MIDNIGHT, "¿Qué día es hoy?", "Hoy es miércoles 18 de noviembre."),
    (AT_MIDNIGHT, "¿qué día fue ayer?", "Ayer fue martes 17 de noviembre."),
    (AT_MIDNIGHT, "¿Qué fecha es hoy?", "Hoy es miércoles 18 de noviembre de 2026."),
    # UTC is already the 18th at 00:30Z; the household is still on the 17th at 21:30.
    (datetime(2026, 11, 18, 0, 30, tzinfo=UTC), "qué día es", "Hoy es martes 17 de noviembre."),
])
async def test_frozen_utc_minus_3_clock_answers_near_midnight_without_any_model(moment, text, spoken):
    d = make_dispatcher(clock_at(moment))
    result = await d.dispatch(user_id="unknown", text=text, zone_id="zone_living")
    assert (result.path, result.priority, result.intent) == (PathType.FAST_CLOCK, Priority.HIGH, "clock")
    assert result.success is True and result.response == spoken
    outcome = result.outcome
    assert outcome.status is TaskStatus.COMPLETED and outcome.task_completed
    assert outcome.effect is EffectState.OBSERVED and outcome.observation_scope == "software_state"
    assert outcome.dispatch is DispatchState.NOT_REQUIRED and outcome.persistence is PersistenceState.NOT_REQUIRED
    assert result.to_dict()["task_completed"] is True
    assert_no_model_and_no_effect(d)
    assert d.get_stats()["by_path"][PathType.FAST_CLOCK] == 1


@pytest.mark.parametrize(("moment", "spoken"), [
    # Montevideo's last DST (tz database): 2014-10-05 02:00 -03 jumped to 03:00 -02 ...
    (datetime(2014, 10, 5, 4, 59, 30, tzinfo=UTC), "Es la 1:59."),
    (datetime(2014, 10, 5, 5, 0, 30, tzinfo=UTC), "Son las 3:00."),
    # ... and 2015-03-08 02:00 -02 fell back to 01:00 -03, so 01:30 happens twice.
    (datetime(2015, 3, 8, 3, 30, tzinfo=UTC), "Es la 1:30."),
    (datetime(2015, 3, 8, 4, 30, tzinfo=UTC), "Es la 1:30."),
    (datetime(2015, 3, 8, 4, 0, tzinfo=UTC), "Es la 1:00."),
])
async def test_daylight_saving_changes_follow_the_household_zone(moment, spoken):
    d = make_dispatcher(clock_at(moment))
    result = await d.dispatch(user_id="unknown", text="qué hora es", zone_id=None)
    assert result.path is PathType.FAST_CLOCK and result.response == spoken
    assert_no_model_and_no_effect(d)


async def test_upstream_service_filter_cannot_turn_a_clock_question_into_an_action():
    d = make_dispatcher(clock_at(BEFORE_MIDNIGHT))
    result = await d.dispatch(user_id="unknown", text="¿qué hora es?", zone_id="zone_living",
                              service_filter="turn_on", query_slots={"brightness_pct": 50})
    assert result.path is PathType.FAST_CLOCK and result.response == "Son las 23:59."
    assert result.action is None and result.receipt is None
    assert_no_model_and_no_effect(d)


def broken_clock_source() -> float:
    raise OSError("clock")


@pytest.mark.parametrize("source", [lambda: float("nan"), broken_clock_source])
async def test_unreadable_clock_fails_honestly_without_guessing_or_asking_a_model(source):
    clock = HouseholdClock(time_source=source)
    d = make_dispatcher(clock)
    result = await d.dispatch(user_id="unknown", text="¿Qué día es hoy?", zone_id="zone_living")
    assert result.path is PathType.FAST_CLOCK and result.intent == "clock"
    assert result.success is False and result.response == CLOCK_UNAVAILABLE
    assert result.outcome.status is TaskStatus.FAILED and not result.outcome.task_completed
    assert result.outcome.effect is EffectState.NOT_OBSERVED and result.outcome.observation_scope is None
    assert_no_model_and_no_effect(d)


@pytest.mark.parametrize("text", ["que hora es en madrid", "que dia es el partido", "a que hora es el partido"])
def test_other_time_questions_left_the_7b_fast_router_list(text):
    d = make_dispatcher(clock_at(BEFORE_MIDNIGHT))
    assert d._classify_request(text) == (PathType.SLOW_LLM, Priority.LOW)
    assert not d._is_simple_query(text)


def test_greetings_keep_the_7b_fast_router():
    d = make_dispatcher(clock_at(BEFORE_MIDNIGHT))
    assert d._classify_request("hola") == (PathType.FAST_ROUTER, Priority.MEDIUM)


class BlockingSyncRouter:
    """A legacy router with only a synchronous ``generate`` (no ``complete``)."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread: int | None = None

    def generate(self, prompts, max_tokens=256):
        self.thread = threading.get_ident()
        self.entered.set()
        # On the event loop thread nothing could ever set this event.
        released = self.release.wait(5)
        return ["Hola, ¿en qué te ayudo?" if released else "loop blocked"]


async def test_sync_router_fallback_runs_off_the_event_loop():
    router = BlockingSyncRouter()
    clock = clock_at(BEFORE_MIDNIGHT)
    d = RequestDispatcher(chroma_sync=MagicMock(), ha_client=AsyncMock(), routine_manager=MagicMock(),
                          router=router, llm=None, context_manager=ContextManager(household_clock=clock),
                          priority_queue=PriorityRequestQueue(), household_clock=clock)
    turn = asyncio.create_task(d.dispatch(user_id="unknown", text="hola", zone_id="zone_living"))

    async def generate_entered() -> None:
        while not router.entered.is_set():
            await asyncio.sleep(0.001)

    # Loop-lag probe: this coroutine only runs while generate is blocked if
    # generate is off the loop; on the loop thread it would block for 5 s and
    # the turn would already be done (answering "loop blocked").
    await asyncio.wait_for(generate_entered(), 5)
    assert not turn.done()
    router.release.set()
    result = await asyncio.wait_for(turn, 5)
    assert router.thread != threading.get_ident()
    assert result.path is PathType.FAST_ROUTER and result.response == "Hola, ¿en qué te ayudo?"
    assert result.outcome.status is TaskStatus.COMPLETED


def test_clock_turns_reach_content_free_diagnostics():
    from src.obs.metrics_emitter import project_diagnostic

    projected = project_diagnostic({"event_type": "request", "path": "fast_clock", "intent": "clock"})
    assert projected["path"] == "fast_clock" and projected["intent"] == "clock"
    every_path = [project_diagnostic({"event_type": "request", "path": path.value}) for path in PathType]
    assert all("path" in projected for projected in every_path)
