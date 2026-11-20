"""T037-U5: a capture without its own wake needs the 7B veto for every shortcut.

User answer T037-U4-Q1 (A): follow-up and barge-in captures carry no wake of
their own; MultiRoomAudioLoop keeps the last real wake's score on them. Like
the T037-U4 clock rule, only a capture opened by its own wake
(``capture_trigger`` "wake", or "inline": the wake's own utterance, T037-U5-R2)
with no other TV signal skips the 7B for the
household shortcuts (timer, list, reminder claims), the extended home command
and the domotics grammar fast path. While :8101 is down such a turn fails
silently and the user repeats it with the wake word; with no LLMCommandRouter
configured it fails closed (T037-U5-R1). Own-wake turns and producers that do
not tag their captures keep today's behavior. A light location the rooms cannot
resolve asks "¿En qué ambiente…?" whatever accepted the turn (T037-U5-R1).
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
    # The wake's own utterance handed over by the wake detector (T037-U5-R2).
    "inline": {"capture_trigger": "inline", "wake_score": 0.9},
    # Raw audio, the single-room loop's own wake: no trigger, own signals as before.
    "untagged": {"wake_score": 0.9},
}
BORROWED = {
    "follow_up": {"capture_trigger": "follow_up", "wake_score": 0.9},
    "barge_in": {"capture_trigger": "barge_in", "wake_score": 0.9},
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
    own = trigger in (CaptureTrigger.WAKE, CaptureTrigger.INLINE)
    assert _capture_without_own_wake(trigger.value) is not own
    assert _capture_without_own_wake(trigger) is not own
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


@pytest.mark.parametrize("llm", ["stub:unavailable", "down"])
async def test_inline_on_a_transcript_wake_engine_is_unchanged_with_no_7b_call(leaves, llm):
    """The reviewer's R2 probe: a whisper inline turn with :8101 unavailable dispatches as on base.

    'apagá la luz del living' parses at 0.8, so the grammar decides with no wake
    bonus; an inline capture is the wake's own utterance, not a borrowed wake.
    """
    house = Household(leaves, text="apagá la luz del living", llm="stub:unavailable", engine="whisper")
    await leaves.apply_state(house.orchestrator.dispatcher, "light_match")
    if llm == "down":
        down(house)
    before = leaves.snapshot()
    result = await turn(house, "inline")
    house.veto.classify.assert_not_awaited()
    assert "llm_router" not in result["timings"]
    assert result["success"] is True and result["path"] == "fast_domotics"
    assert house.effects(before) == effects(ha_service_calls=1, vector_searches=1)


# --- Unresolved light location (T037-U5-R1) --------------------------------------

# "quincho" is no configured room: base asked "¿En qué ambiente…?" on a follow-up
# or barge-in; a vector search must never guess the light instead.
UNRESOLVED_LIGHT = ["prendé la luz del quincho", "prendé las luces del quincho"]
ASK_ROOM = "¿En qué ambiente querés controlar la luz?"


def assert_asks_room_no_effect(house: Household, result: dict, before, llm_8101_calls: int) -> None:
    assert result["intent"] == "unresolved_target" and result["response"] == ASK_ROOM
    assert result["action"] is None
    house.orchestrator.process.assert_not_awaited()
    assert house.effects(before) == effects(llm_8101_calls=llm_8101_calls, spoken_utterances=1)


@pytest.mark.parametrize("capture", ["follow_up", "barge_in"])
@pytest.mark.parametrize("text", UNRESOLVED_LIGHT)
async def test_unresolved_light_without_own_wake_the_7b_accepts_asks(leaves, text, capture):
    """The reviewer's probe: the 7B accepts, the room is unknown, so the turn asks and HA is never called."""
    house = await house_for(leaves, text, "stub:turn_on@0.95", "light_match")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert_asks_room_no_effect(house, result, before, llm_8101_calls=1)


@pytest.mark.parametrize("capture", list(BORROWED))
@pytest.mark.parametrize("text", [*UNRESOLVED_LIGHT, "apagá la luz del living"])
async def test_without_own_wake_and_no_7b_configured_fails_closed(leaves, text, capture):
    """No LLMCommandRouter: nothing can veto a capture without its own wake, so nothing happens at all."""
    house = await house_for(leaves, text, "absent", "light_match")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert result["intent"] == "llm_rejected:no_router"
    assert result["success"] is False and result["response"] == "" and result["action"] is None
    house.orchestrator.process.assert_not_awaited()
    house.handler.aspeak.assert_not_awaited()
    assert house.effects(before) == effects()


@pytest.mark.parametrize("capture", list(OWN))
@pytest.mark.parametrize("llm", ["absent", "stub:unavailable"])
@pytest.mark.parametrize("text", UNRESOLVED_LIGHT)
async def test_unresolved_light_on_own_wake_still_asks_with_no_7b_call(leaves, text, llm, capture):
    """Unchanged from base: the grammar fast path asks on its own wake."""
    house = await house_for(leaves, text, llm, "light_match")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert "llm_router" not in result["timings"]
    assert_asks_room_no_effect(house, result, before, llm_8101_calls=0)


@pytest.mark.parametrize("llm", ["absent", "stub:turn_on@0.95"])
@pytest.mark.parametrize("text", UNRESOLVED_LIGHT)
async def test_unresolved_light_without_grammar_fast_path_asks(leaves, text, llm):
    """A transcript wake engine gets no grammar bonus (0.7 < 0.75): the 7B or no model accepts, the turn still asks."""
    house = Household(leaves, text=text, llm=llm, engine="whisper")
    await leaves.apply_state(house.orchestrator.dispatcher, "light_match")
    before = leaves.snapshot()
    result = await turn(house, "wake")
    assert_asks_room_no_effect(house, result, before, llm_8101_calls=0 if llm == "absent" else 1)


@pytest.mark.parametrize("llm", ["absent", "stub:turn_on@0.95"])
@pytest.mark.parametrize("text", UNRESOLVED_LIGHT)
async def test_unresolved_light_on_the_textual_wake_asks(leaves, text, llm):
    """The textual wake channel skips the grammar too; it had the same gap before T037-U5."""
    house = await house_for(leaves, text, llm, "light_match")
    before = leaves.snapshot()
    result = await house.turn("textual")
    assert_asks_room_no_effect(house, result, before, llm_8101_calls=0 if llm == "absent" else 1)


@pytest.mark.parametrize("capture", ["follow_up", "barge_in"])
async def test_resolved_light_without_own_wake_the_7b_accepts_still_runs(leaves, capture):
    """The guard only stops a location the rooms cannot resolve: a known room still runs."""
    house = await house_for(leaves, "prendé la luz del living", "stub:turn_on@0.95", "light_match")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert result["success"] is True and result["path"] == "fast_domotics"
    assert house.effects(before) == effects(ha_service_calls=1, vector_searches=1, llm_8101_calls=1)


# 'bajá' and 'cambiá' are verbs the dispatcher routes to FAST_DOMOTICS, but the
# grammar names no intent for them: only the 7B or the verb says it is an action.
VERB_ONLY_LIGHT = ["bajá la luz del quincho", "cambiá la luz del quincho"]
# What the reviewer's probe had the 7B return (T037-U5-FV1): a room-only hint, or no intent.
ACCEPTS_UNNAMED_LIGHT = {
    "set_brightness:quincho": dict(intent="set_brightness", entity_hint="quincho"),
    "toggle:quincho": dict(intent="toggle", entity_hint="quincho"),
    "intent_none": dict(intent=None, entity_hint=None),
}


@pytest.mark.parametrize("accepted", list(ACCEPTS_UNNAMED_LIGHT))
@pytest.mark.parametrize("capture", ["follow_up", "barge_in", "wake"])
@pytest.mark.parametrize("text", VERB_ONLY_LIGHT)
async def test_unresolved_light_the_grammar_names_no_intent_for_asks(leaves, text, capture, accepted):
    """The 7B accepts, its hint names no light and the grammar finds no intent: the turn still asks."""
    from src.nlu.llm_router import CommandClassification
    house = await house_for(leaves, text, "stub:turn_on@0.95", "light_match")
    house.veto.classify.return_value = CommandClassification(
        is_command=True, confidence=0.95, **ACCEPTS_UNNAMED_LIGHT[accepted])
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert_asks_room_no_effect(house, result, before, llm_8101_calls=1)


# A lamp or bulb in an unknown room asks like 'luz' does (T037-U5-FV2).
UNRESOLVED_FIXTURE = ["prendé la lámpara del quincho", "apagá el foco del quincho"]


@pytest.mark.parametrize("capture", ["follow_up", "barge_in"])
@pytest.mark.parametrize(("text", "llm"), zip(UNRESOLVED_FIXTURE, ["stub:turn_on@0.95", "stub:turn_off@0.95"]))
async def test_unresolved_fixture_without_own_wake_the_7b_accepts_asks(leaves, text, llm, capture):
    house = await house_for(leaves, text, llm, "light_match")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert_asks_room_no_effect(house, result, before, llm_8101_calls=1)


@pytest.mark.parametrize("capture", list(OWN))
@pytest.mark.parametrize("text", UNRESOLVED_FIXTURE)
async def test_unresolved_fixture_on_own_wake_asks_with_no_7b_call(leaves, text, capture):
    house = await house_for(leaves, text, "stub:unavailable", "light_match")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert "llm_router" not in result["timings"]
    assert_asks_room_no_effect(house, result, before, llm_8101_calls=0)


@pytest.mark.parametrize("capture", list(OWN))
async def test_fixture_in_a_known_room_on_own_wake_still_runs(leaves, capture):
    """A fixture qualifier before a known room is no unknown location: the vector search picks the lamp."""
    house = await house_for(leaves, "prendé la lámpara de pie del living", "stub:unavailable", "light_match")
    before = leaves.snapshot()
    result = await turn(house, capture)
    assert result["success"] is True and result["path"] == "fast_domotics"
    assert house.effects(before) == effects(ha_service_calls=1, vector_searches=1)


@pytest.mark.parametrize(("hint", "intent", "text", "expected"), [
    ("light", "turn_on", "dale nomás", True),
    ("luz del quincho", "turn_on", "dale nomás", True),
    ("Lámpara del quincho", "turn_off", "dale nomás", True),
    ("LUCES", "set_brightness", "dale nomás", True),
    ("luz del quincho", None, "dale nomás", False),
    ("cover", "open", "dale nomás", False),
    ("persiana del luzón", "open", "dale nomás", False),
    (None, None, "prendé la luz del quincho", True),
    (None, None, "¿por qué parpadea la luz del quincho?", False),
    (None, None, "la luz del quincho", False),
    (None, None, "subí la persiana del quincho", False),
    # T037-U5-FV1: the text names a light and the 7B accepted an action, or a dispatcher verb says so.
    ("quincho", "set_brightness", "bajá la luz del quincho", True),
    ("quincho", "toggle", "cambiá la luz del quincho", True),
    ("quincho", "turn_on", "la luz del quincho", True),
    (None, None, "bajá la luz del quincho", True),
    (None, None, "cambiá las luces del quincho", True),
    ("quincho", None, "la luz del quincho", False),
    ("quincho", "set_brightness", "bajá la persiana del quincho", False),
    (None, None, "bajá la persiana del quincho", False),
    # T037-U5-FV2: the dispatcher's other light nouns in a 7B hint.
    ("foco del quincho", "turn_on", "dale nomás", True),
    ("veladoras", "turn_off", "dale nomás", True),
])
def test_dispatch_targets_light(hint, intent, text, expected):
    from src.nlu.llm_router import CommandClassification
    cls = None if intent is None and hint is None else CommandClassification(
        is_command=True, confidence=0.9, intent=intent, entity_hint=hint)
    assert router_module._dispatch_targets_light(text, cls) is expected


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
@pytest.mark.parametrize("capture", ["wake", "inline"])
async def test_own_wake_fast_path_waits_on_no_model_call(leaves, text, state, path, capture):
    """The own-wake fast path adds no 7B round trip: the 7B is never awaited, so its latency never counts."""
    house = await house_for(leaves, text, "stub:noise", state)
    house.veto.classify.side_effect = AssertionError("7B awaited on an own-wake fast path turn")
    result = await turn(house, capture)
    house.veto.classify.assert_not_awaited()
    assert result["path"] == path and "llm_router" not in result["timings"]


# --- Single-room loop ---------------------------------------------------------

@pytest.mark.parametrize(("wake_heard", "trigger", "score"), [(True, "wake", 0.9), (False, "follow_up", 0.0)])
async def test_single_room_loop_tags_own_wake_and_follow_up(wake_heard, trigger, score):
    """The fallback loop distinguishes a new acoustic wake from a follow-up."""
    manager = MagicMock()
    manager.detect_wake_word = MagicMock(return_value=("nexa", score) if wake_heard else None)
    manager.capture_command_with_vad = MagicMock(return_value=(True, None, np.zeros(16000, np.float32), False))
    follow_up = FollowUpMode(follow_up_window=600)
    follow_up.start_conversation()
    try:
        loop = AudioLoop(manager, echo_suppressor(), follow_up)
        loop._process_audio(np.full((1280, 1), 0.05, dtype=np.float32), 1280, 0.0, None)
        event = loop._poll()
        assert event.capture_trigger == trigger
        assert event.wake_score == score
        assert _capture_without_own_wake(event.capture_trigger) is (not wake_heard)
        loop._clear_capture()
        assert loop._state.trigger is None
    finally:
        follow_up.end_conversation("test_teardown")
