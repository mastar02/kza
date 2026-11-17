"""T037-U5: a capture without its own wake needs the 7B veto for every shortcut.

User answer T037-U4-Q1 (A): follow-up, barge-in and inline captures carry no
wake of their own; MultiRoomAudioLoop keeps the last real wake's score on
them. Like the T037-U4 clock rule, only a capture opened by its own wake
(``capture_trigger == "wake"``) with no other TV signal skips the 7B for the
household shortcuts (timer, list, reminder claims), the extended home command
and the domotics grammar fast path. While :8101 is down such a turn fails
silently and the user repeats it with the wake word. Own-wake turns and
producers that do not tag their captures keep today's behavior.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("pyaudio", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import src.pipeline.request_router as router_module  # noqa: E402
from src.conversation.follow_up_mode import FollowUpMode  # noqa: E402
from src.pipeline.audio_loop import AudioLoop  # noqa: E402
from src.pipeline.command_event import CommandEvent  # noqa: E402
from src.pipeline.post_wake_endpoint import CaptureTrigger  # noqa: E402
from src.pipeline.request_router import _capture_without_own_wake  # noqa: E402
from tests.mocks.c02_matrix import effects  # noqa: E402
from tests.mocks.household_dispatch import CountedLeaves  # noqa: E402
from tests.unit.pipeline.test_request_router_household import Household  # noqa: E402
from tests.unit.pipeline.test_reminder_follow_up_turns import echo_suppressor  # noqa: E402

# How each capture reaches the router. The borrowed ones carry the last real
# wake's clear 0.9 score, so the trigger is the only thing that tells them apart.
OWN = {
    "wake": {"capture_trigger": "wake", "wake_score": 0.9},
    # Raw audio, the single-room loop's own wake: no trigger, own signals as before.
    "untagged": {"wake_score": 0.9},
}
BORROWED = {
    "follow_up": {"capture_trigger": "follow_up", "wake_score": 0.9},
    "barge_in": {"capture_trigger": "barge_in", "wake_score": 0.9},
    "inline": {"capture_trigger": "inline", "wake_score": 0.9},
    "unknown": {"capture_trigger": "unknown", "wake_score": 0.9},
}
CAPTURES = {**OWN, **BORROWED}

# Short household requests a TV or a radio says in the follow-up window (all
# within unvetoed_max_words, so only the missing wake asks for the veto).
HOUSEHOLD_TV = [
    ("poné un temporizador de 5 minutos", None, "timer"),
    ("pausá el timer de la pasta", "timer_running:pasta", "timer"),
    ("agregá leche a la lista de compras", "shared_list_compras", "list"),
    ("recordame apagar la luz", None, "reminder"),
    ("borrá el recordatorio de tomar agua", "reminder_shared_active:tomar agua", "reminder"),
]
# Domotics the grammar parses fully (0.8 needs no wake bonus; 0.7 needs it).
GRAMMAR_TV = ["apagá la luz del living", "prendé la luz del living", "apagame la lámpara"]


@pytest.fixture
async def leaves(tmp_path):
    counted = CountedLeaves(tmp_path)
    await counted.start()
    yield counted
    await counted.close()


async def house_for(leaves: CountedLeaves, text: str, llm: str, state: str | None = None) -> Household:
    house = Household(leaves, text=text, llm=llm)
    await leaves.apply_state(house.orchestrator.dispatcher, state)
    return house


def down(house: Household) -> None:
    """:8101 refuses the connection; the LLMCommandRouter raises."""
    house.veto.classify.side_effect = ConnectionRefusedError(":8101 down")


async def turn(house: Household, capture: str) -> dict:
    return await house.router.process_command(
        CommandEvent(audio=np.full(16000, .1, dtype=np.float32), room_id="living", **CAPTURES[capture]))


def assert_silent_no_turn(house: Household, result: dict, before, intent: str) -> None:
    assert result["intent"] == intent
    assert result["success"] is False and result["response"] == ""
    house.orchestrator.process.assert_not_awaited()
    house.handler.aspeak.assert_not_awaited()
    assert house.effects(before) == effects(llm_8101_calls=1)


@pytest.mark.parametrize("trigger", list(CaptureTrigger))
def test_only_a_wake_tagged_capture_has_its_own_wake(trigger):
    assert _capture_without_own_wake(trigger.value) is (trigger is not CaptureTrigger.WAKE)
    assert _capture_without_own_wake(None) is False


# --- Household shortcuts -----------------------------------------------------

@pytest.mark.parametrize("capture", list(BORROWED))
@pytest.mark.parametrize(("text", "state", "family"), HOUSEHOLD_TV)
async def test_household_tv_phrase_without_own_wake_is_vetoed(leaves, text, state, family, capture):
    house = await house_for(leaves, text, "stub:noise", state)
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert house.veto.classify.await_count == 1
    assert result["household"]["family"] == family
    assert_silent_no_turn(house, result, before, "llm_rejected:household")


@pytest.mark.parametrize("llm", ["stub:unavailable", "stub:unavailable_command", "down"])
@pytest.mark.parametrize("capture", list(BORROWED))
async def test_household_without_own_wake_fails_silently_while_8101_is_down(leaves, capture, llm):
    house = await house_for(leaves, "poné un temporizador de 5 minutos", "stub:noise" if llm == "down" else llm)
    if llm == "down":
        down(house)
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert_silent_no_turn(house, result, before, "llm_rejected:household")


@pytest.mark.parametrize("capture", list(BORROWED))
async def test_household_without_own_wake_and_no_7b_configured_fails_closed(leaves, capture):
    house = await house_for(leaves, "poné un temporizador de 5 minutos", "absent")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert result["intent"] == "llm_rejected:household"
    house.orchestrator.process.assert_not_awaited()
    assert house.effects(before) == effects()


@pytest.mark.parametrize("capture", list(BORROWED))
async def test_household_without_own_wake_the_7b_accepts_runs(leaves, capture):
    """The 7B only gates: a real follow-up the model accepts still creates the timer."""
    house = await house_for(leaves, "poné un temporizador de 5 minutos", "stub:command@0.9")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert result["path"] == "fast_timer" and result["outcome"]["status"] == "completed"
    assert house.effects(before) == effects(timer_delta=1, llm_8101_calls=1, spoken_utterances=1)


@pytest.mark.parametrize("capture", list(OWN))
@pytest.mark.parametrize("llm", ["stub:unavailable", "down"])
async def test_household_on_own_wake_is_unchanged_with_no_7b_call(leaves, llm, capture):
    """C02-TIM-0201 / C02-TIM-0204 still hold: own wake, 6 words, no 7B even while :8101 is down."""
    house = await house_for(leaves, "poné un temporizador de 5 minutos", "stub:unavailable")
    if llm == "down":
        down(house)
    before = leaves.snapshot()
    result = await turn(house, capture)
    house.veto.classify.assert_not_awaited()
    assert "llm_router" not in result["timings"]
    assert result["path"] == "fast_timer" and result["outcome"]["status"] == "completed"
    assert house.effects(before) == effects(timer_delta=1, spoken_utterances=1)


# --- Domotics grammar fast path ----------------------------------------------

@pytest.mark.parametrize("capture", list(BORROWED))
@pytest.mark.parametrize("text", GRAMMAR_TV)
async def test_grammar_domotics_without_own_wake_is_vetoed(leaves, monkeypatch, text, capture):
    grammar = MagicMock(side_effect=AssertionError("grammar fast path reached"))
    monkeypatch.setattr(router_module, "_grammar_fastpath_classification", grammar)
    house = await house_for(leaves, text, "stub:noise", "light_match")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert house.veto.classify.await_count == 1
    assert_silent_no_turn(house, result, before, "llm_rejected:noise")


@pytest.mark.parametrize("llm", ["stub:unavailable", "stub:unavailable_command"])
@pytest.mark.parametrize("capture", list(BORROWED))
@pytest.mark.parametrize("text", [*GRAMMAR_TV, "la luz del living"])
async def test_grammar_domotics_without_own_wake_fails_silently_while_8101_is_down(leaves, text, capture, llm):
    """No grammar-backed best-effort dispatch and no spoken confirmation question: nothing at all.

    LLMCommandRouter reports a refused or timed-out :8101 as 'unavailable'.
    """
    house = await house_for(leaves, text, llm, "light_match")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert_silent_no_turn(house, result, before, "llm_rejected:unavailable")
    assert "needs_confirmation" not in result and result["action"] is None


@pytest.mark.parametrize("capture", list(BORROWED))
async def test_grammar_domotics_without_own_wake_the_7b_accepts_runs(leaves, capture):
    house = await house_for(leaves, "apagá la luz del living", "stub:turn_off@0.95", "light_match")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert result["success"] is True and result["path"] == "fast_domotics"
    assert house.effects(before) == effects(ha_service_calls=1, vector_searches=1, llm_8101_calls=1)


@pytest.mark.parametrize("capture", list(OWN))
@pytest.mark.parametrize("llm", ["stub:unavailable", "down"])
@pytest.mark.parametrize("text", GRAMMAR_TV)
async def test_grammar_domotics_on_own_wake_is_unchanged_with_no_7b_call(leaves, text, llm, capture):
    """C02-DOM-0201 still holds on its own wake: the grammar decides, :8101 is never asked."""
    house = await house_for(leaves, text, "stub:unavailable", "light_match")
    if llm == "down":
        down(house)
    before = leaves.snapshot()
    result = await turn(house, capture)
    house.veto.classify.assert_not_awaited()
    assert "llm_router" not in result["timings"]
    assert result["success"] is True and result["path"] == "fast_domotics"
    assert house.effects(before) == effects(ha_service_calls=1, vector_searches=1)


@pytest.mark.parametrize("capture", list(OWN))
async def test_partial_grammar_on_own_wake_still_asks_while_8101_is_unavailable(leaves, capture):
    """Unchanged: on its own wake a partial signal with :8101 unavailable asks for confirmation."""
    house = await house_for(leaves, "la luz del living", "stub:unavailable", "light_match")
    result = await turn(house, capture)
    assert result["needs_confirmation"] is True and result["response"]
    house.orchestrator.process.assert_not_awaited()


# --- Extended home command -----------------------------------------------------

@pytest.mark.parametrize("capture", list(BORROWED))
@pytest.mark.parametrize("llm", ["stub:noise", "stub:unavailable", "down"])
async def test_home_command_without_own_wake_is_vetoed(leaves, llm, capture):
    house = await house_for(leaves, "subí la persiana del cuarto", "stub:noise" if llm == "down" else llm)
    if llm == "down":
        down(house)
    house.router.home_voice_resolver.resolve = AsyncMock(side_effect=AssertionError("target resolved"))
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert_silent_no_turn(house, result, before, "llm_rejected:home_command")


@pytest.mark.parametrize("capture", list(OWN))
async def test_home_command_on_own_wake_is_unchanged_with_no_7b_call(leaves, capture):
    house = await house_for(leaves, "subí la persiana del cuarto", "stub:unavailable")
    down(house)
    house.router.home_voice_resolver.resolve = AsyncMock(side_effect=RuntimeError("no selection"))
    result = await turn(house, capture)
    house.veto.classify.assert_not_awaited()
    assert "llm_router" not in result["timings"]
    house.router.home_voice_resolver.resolve.assert_awaited_once()
    assert result["intent"] == "clarification_requested" and result["reason"] == "selection_unavailable"


@pytest.mark.parametrize("capture", list(BORROWED))
async def test_home_command_without_own_wake_the_7b_accepts_reaches_selection(leaves, capture):
    house = await house_for(leaves, "subí la persiana del cuarto", "stub:open@0.9")
    house.router.home_voice_resolver.resolve = AsyncMock(side_effect=RuntimeError("no selection"))
    result = await turn(house, capture)
    assert house.veto.classify.await_count == 1
    house.router.home_voice_resolver.resolve.assert_awaited_once()
    assert result["intent"] == "clarification_requested"


# --- Own-wake fast path latency ----------------------------------------------

@pytest.mark.parametrize(("text", "state", "path"), [
    ("poné un temporizador de 5 minutos", None, "fast_timer"),
    ("apagá la luz del living", "light_match", "fast_domotics"),
])
async def test_own_wake_fast_path_waits_on_no_model_call(leaves, text, state, path):
    """The own-wake fast path adds no 7B round trip: the 7B is never awaited, so its latency never counts."""
    house = await house_for(leaves, text, "stub:noise", state)
    house.veto.classify.side_effect = AssertionError("7B awaited on an own-wake fast path turn")
    result = await turn(house, "wake")
    house.veto.classify.assert_not_awaited()
    assert result["path"] == path and "llm_router" not in result["timings"]


# --- Single-room loop ---------------------------------------------------------

@pytest.mark.parametrize(("wake_heard", "trigger"), [(True, None), (False, "follow_up")])
async def test_single_room_loop_tags_a_follow_up_capture(wake_heard, trigger):
    """The fallback AudioLoop says when a capture was opened in the follow-up window, not by its wake."""
    manager = MagicMock()
    manager.detect_wake_word = MagicMock(return_value=wake_heard)
    manager.capture_command_with_vad = MagicMock(return_value=(True, None, np.zeros(16000, np.float32), False))
    follow_up = FollowUpMode(follow_up_window=600)
    follow_up.start_conversation()
    try:
        loop = AudioLoop(manager, echo_suppressor(), follow_up)
        loop._process_audio(np.full((1280, 1), 0.05, dtype=np.float32), 1280, 0.0, None)
        event = loop._poll()
        assert event.capture_trigger == trigger
        assert _capture_without_own_wake(event.capture_trigger) is (trigger is not None)
        loop._clear_capture()
        assert loop._state.trigger is None
    finally:
        follow_up.end_conversation("test_teardown")
