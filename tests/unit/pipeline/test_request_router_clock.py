"""T037-U4: a clock question on a clear wake skips the 7B veto; a weak wake keeps it.

User answer T037-U3-Q1: a whole-phrase time or date question with a clear
wake (not a textual wake, not ambient_strict, not a low wake score, not a
transcript-based wake engine) is answered by the household clock with no
:8101 call, like a short household request, so it works while :8101 is down.
A weak wake still needs the 7B, which fails closed when it is unavailable.
"""

import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock

sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("pyaudio", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

import pytest  # noqa: E402

from src.core.household_clock import HouseholdClock  # noqa: E402
from src.nlu.household_intents import HouseholdIntentSettings  # noqa: E402
from src.nlu.llm_router import CommandClassification  # noqa: E402
from tests.mocks.c02_matrix import effects  # noqa: E402
from tests.mocks.household_dispatch import CountedLeaves  # noqa: E402
from tests.unit.pipeline.test_request_router_household import Household  # noqa: E402

# 02:59:59Z on the 18th is 23:59:59 on Tuesday the 17th in the household (UTC-3).
MOMENT = datetime(2026, 11, 18, 2, 59, 59, tzinfo=UTC).timestamp()
# What the 7B answers for a time question (llm_router prompt: is_cmd=true, intent=null).
CLOCK_PASS = CommandClassification(is_command=True, confidence=0.9, intent=None)

ANSWERED = [
    ("¿Qué hora es?", "Son las 23:59."),
    ("Nexa, ¿qué día es hoy?", "Hoy es martes 17 de noviembre."),
    # An alias of the grammar's wake terms, as Whisper hears the pre-roll (T037-U3-R1).
    ("Alexa, ¿qué hora es?", "Son las 23:59."),
    # Ten words: the whole-phrase recognizer is the bound, not household_intents.unvetoed_max_words.
    ("che, ¿me podés decir qué día de la semana es mañana, por favor?", "Mañana es miércoles 18 de noviembre."),
]
# Clock questions a TV, a radio or a replay says out loud.
TV_PHRASES = ["¿Qué hora es?", "¿Qué día es hoy?", "decime la hora", "¿A cuánto estamos hoy?"]
WEAK_WAKES = [("textual", "openwakeword"), ("ambient_strict", "openwakeword"), ("low_score", "openwakeword"),
              ("acoustic", "whisper")]


@pytest.fixture
async def leaves(tmp_path):
    counted = CountedLeaves(tmp_path)
    await counted.start()
    yield counted
    await counted.close()


def house_at_moment(leaves: CountedLeaves, text: str, llm: str, engine: str = "openwakeword",
                    settings: HouseholdIntentSettings | None = None) -> Household:
    house = Household(leaves, text=text, llm=llm, engine=engine, settings=settings)
    house.orchestrator.dispatcher.household_clock = HouseholdClock(time_source=lambda: MOMENT)
    return house


def assert_clock_answer(house: Household, result: dict, spoken: str, before, *, veto_calls: int) -> None:
    assert (result["intent"], result["path"], result["response"]) == ("clock", "fast_clock", spoken)
    assert result["success"] and result["outcome"]["status"] == "completed"
    assert house.handler.aspeak.await_args.args[0] == spoken
    assert house.effects(before) == effects(llm_8101_calls=veto_calls, spoken_utterances=1)
    kwargs = house.orchestrator.process.await_args.kwargs
    assert (kwargs["service_filter"], kwargs["query_slots"]) == (None, None)


@pytest.mark.parametrize("llm", ["stub:unavailable", "stub:noise", "down"])
@pytest.mark.parametrize(("text", "spoken"), ANSWERED)
async def test_clear_wake_clock_question_is_answered_with_no_7b_call(leaves, text, spoken, llm):
    """:8101 down, unavailable or rejecting: the clock still answers and the 7B is never asked."""
    house = house_at_moment(leaves, text, "stub:noise" if llm == "down" else llm)
    if llm == "down":
        house.veto.classify.side_effect = ConnectionRefusedError(":8101 down")
    before = leaves.snapshot()
    result = await house.turn("acoustic")
    house.veto.classify.assert_not_awaited()
    assert "llm_router" not in result["timings"]
    assert_clock_answer(house, result, spoken, before, veto_calls=0)


@pytest.mark.parametrize(("wake", "engine"), WEAK_WAKES)
@pytest.mark.parametrize("text", TV_PHRASES)
async def test_weak_wake_clock_question_is_vetoed_by_the_7b(leaves, text, wake, engine):
    """A TV-like clock question on a weak wake asks the 7B, and its rejection ends the turn silently."""
    house = house_at_moment(leaves, text, "stub:noise", engine=engine)
    before = leaves.snapshot()
    result = await house.turn(wake)
    assert house.veto.classify.await_count == 1
    assert result["intent"] == "llm_rejected:noise"
    assert result["success"] is False and result["response"] == ""
    house.orchestrator.process.assert_not_awaited()
    assert house.effects(before) == effects(llm_8101_calls=1)


@pytest.mark.parametrize(("wake", "engine"), WEAK_WAKES)
async def test_weak_wake_clock_question_fails_closed_while_8101_is_unavailable(leaves, wake, engine):
    house = house_at_moment(leaves, "¿Qué hora es?", "stub:unavailable", engine=engine)
    before = leaves.snapshot()
    result = await house.turn(wake)
    assert result["intent"] == "llm_rejected:unavailable"
    house.orchestrator.process.assert_not_awaited()
    assert house.effects(before) == effects(llm_8101_calls=1)


@pytest.mark.parametrize(("wake", "engine"), WEAK_WAKES)
async def test_weak_wake_clock_question_the_7b_accepts_is_answered_by_the_clock(leaves, wake, engine):
    """The 7B only gates the turn; the answer still comes from the household clock, never a model."""
    house = house_at_moment(leaves, "¿Qué hora es?", "stub:noise", engine=engine)
    house.veto.classify.return_value = CLOCK_PASS
    before = leaves.snapshot()
    result = await house.turn(wake)
    assert_clock_answer(house, result, "Son las 23:59.", before, veto_calls=1)


@pytest.mark.parametrize("text", ["¿A qué hora es el partido?", "¿Qué hora es en Madrid?", "¿Qué día es Navidad?"])
async def test_a_question_with_another_subject_keeps_the_7b_gate_on_a_clear_wake(leaves, text):
    house = house_at_moment(leaves, text, "stub:noise")
    result = await house.turn("acoustic")
    assert house.veto.classify.await_count == 1
    assert result["intent"] == "llm_rejected:noise"
    house.orchestrator.process.assert_not_awaited()


async def test_router_strips_the_same_wake_terms_as_the_dispatcher(leaves):
    """Router and dispatcher agree on which turns are clock turns, configured aliases included."""
    settings = HouseholdIntentSettings(wake_words=("nexa", "neska"))
    house = house_at_moment(leaves, "Neska, ¿qué día es hoy?", "stub:noise", settings=settings)
    assert house.router._clock_wake_words == house.orchestrator.dispatcher._clock_wake_words
    assert "neska" in house.router._clock_wake_words and "alexa" in house.router._clock_wake_words
    before = leaves.snapshot()
    result = await house.turn("acoustic")
    house.veto.classify.assert_not_awaited()
    assert_clock_answer(house, result, "Hoy es martes 17 de noviembre.", before, veto_calls=0)

