"""P3-F2 S5/S6: claimed household turns in the orchestrated router, with every C02 counter."""

import sys
from unittest.mock import AsyncMock, MagicMock

sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("pyaudio", MagicMock())
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.cuda", MagicMock())

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.nlu.command_gate import CommandAcceptanceGate  # noqa: E402
from src.nlu.household_intents import HouseholdIntentClassifier, HouseholdIntentSettings  # noqa: E402
from src.nlu.llm_router import CommandClassification  # noqa: E402
from src.orchestrator.dispatcher import MultiUserOrchestrator  # noqa: E402
from src.pipeline.command_event import CommandEvent  # noqa: E402
from src.pipeline.command_processor import ProcessedCommand  # noqa: E402
from src.pipeline.request_router import RequestRouter  # noqa: E402
from tests.mocks.c02_matrix import C02Entry, effects  # noqa: E402
from tests.mocks.household_dispatch import CountedLeaves  # noqa: E402
from tests.mocks.responses import response_handler_mock  # noqa: E402

STUBS = {
    "turn_off@0.95": CommandClassification(is_command=True, confidence=0.95, intent="turn_off", entity_hint="light"),
    "turn_on@0.95": CommandClassification(is_command=True, confidence=0.95, intent="turn_on", entity_hint="light"),
    "open@0.9": CommandClassification(is_command=True, confidence=0.9, intent="open", entity_hint="cover"),
    "command@0.9": CommandClassification(is_command=True, confidence=0.9, intent="timer"),
    "command@0.3": CommandClassification(is_command=True, confidence=0.3, intent="timer"),
    "command@1.5": CommandClassification(is_command=True, confidence=1.5, intent="timer"),
    "noise": CommandClassification(is_command=False, confidence=0.1, intent="noise", rejection_reason="noise"),
    "unavailable": CommandClassification(is_command=None, rejection_reason="unavailable"),
    # Defensive shape: a command claim that also reports 'unavailable' is still not a veto pass.
    "unavailable_command": CommandClassification(is_command=True, confidence=0.9, rejection_reason="unavailable"),
}


def _r(id_, text, family, outcome, path, status, persistence, spoken, *, wake="acoustic", llm="absent",
       engine="openwakeword", counts=None, state=None, effect_class="unauthorized_effect", seed=None, note="",
       unit="P3-F2"):
    conditions = {"wake": wake, "llm": llm, "engine": engine}
    if state:
        conditions["state"] = state
    return C02Entry(id_, text, "router", family, effect_class, outcome, path, status, persistence, spoken,
                    counts or effects(), conditions, unit=unit, seed=seed, note=note)


SPOKEN_ONE = effects(spoken_utterances=1)
ASKS = ("T031-U2 closes the interim UNSUPPORTED: a request without a time asks for the whole request again "
        "(NEEDS_INPUT, the seed's clarification)")

C02_ROUTER = (
    _r("C02-REM-0001", "recordame apagar la luz", "reminder", "needs_input", "fast_reminder", "needs_input",
       "not_written", "clarification", counts=SPOKEN_ONE, note=ASKS + "; CORE-02 closed: no grammar turn_off"),
    _r("C02-REM-0004", "recordame abrir la persiana mañana", "reminder", "needs_input", "fast_reminder",
       "needs_input", "not_written", "clarification", llm="stub:open@0.9", counts=SPOKEN_ONE, note=ASKS),
    _r("C02-REM-0008", "recordale a Ana apagar la luz", "reminder", "unsupported", "fast_reminder", "unsupported",
       "not_written", "refusal", llm="stub:turn_off@0.95", counts=SPOKEN_ONE),
    _r("C02-REM-0013", "recordame apagar la luz", "reminder", "needs_input", "fast_reminder", "needs_input",
       "not_written", "clarification", wake="textual", llm="stub:turn_off@0.95",
       counts=effects(llm_8101_calls=1, spoken_utterances=1),
       note="7B veto accepts; service_filter still forced to None; " + ASKS),
    _r("C02-TIM-0006", "poné un temporizador de 5 minutos", "timer", "no_turn", None, None, None, "none",
       wake="ambient_strict", llm="stub:noise", counts=effects(llm_8101_calls=1), effect_class="stale_audio"),
    _r("C02-QRY-0002", "¿qué recordatorios hay?", "query", "no_turn", None, None, None, "none", wake="textual",
       llm="stub:unavailable", counts=effects(llm_8101_calls=1), effect_class="privacy_disclosure"),
    _r("C02-DEF-0002", "apagá la luz en 10 minutos", "deferred_action", "no_turn", None, None, None, "none",
       llm="stub:turn_off@0.95", seed="C02-DEF-0002",
       note="characterized: the command gate still drops deferred home actions (non_immediate:deferred) before "
            "the household layer, as today; the dispatcher refusal applies to other callers. Open question for "
            "the user (speak the refusal now or keep the earcon until T031-U5)."),
    _r("C02-ACT-0007", "apagá la luz en 10 minutos", "deferred_action", "no_turn", None, None, None, "none",
       wake="textual", llm="stub:unavailable", seed="C02-ACT-0007",
       note="gated before the 7B (the seed counted one veto call); zero effects either way"),
    _r("C02-TIM-0201", "poné un temporizador de 5 minutos", "timer", "completed", "fast_timer", "completed",
       "volatile", "readback", llm="stub:turn_on@0.95", counts=effects(timer_delta=1, spoken_utterances=1),
       note="T030-U1: strong acoustic wake, 6 words: no 7B call, no vector search; the timer is created"),
    _r("C02-TIM-0202", "poné un temporizador de 5 minutos", "timer", "completed", "fast_timer", "completed",
       "volatile", "readback", wake="low_score", llm="stub:command@0.9",
       counts=effects(timer_delta=1, llm_8101_calls=1, spoken_utterances=1),
       note="T030-U1: TV signal, the 7B veto accepts, then the timer is created"),
    _r("C02-TIM-0204", "pausá el timer de la pasta", "timer", "completed", "fast_timer", "completed", "volatile",
       "readback", llm="stub:unavailable", state="timer_running:pasta",
       counts=effects(timer_transitions=1, spoken_utterances=1),
       note="T030-U1: 7B unavailable on a strong acoustic wake: no veto call, the pause still runs"),
    _r("C02-TIM-0205", "poné un temporizador de 5 minutos", "timer", "failed", "fast_timer", "failed",
       "not_written", "refusal", state="feature_manager_start_failed", counts=SPOKEN_ONE,
       effect_class="false_confirmation", seed="C02-TIM-0005",
       note="T030-U1: manager not running: the refusal is spoken, no confirmation, no timer"),
    _r("C02-TIM-0203", "poné un temporizador de 5 minutos", "timer", "no_turn", None, None, None, "none",
       wake="low_score", llm="stub:command@0.3", counts=effects(llm_8101_calls=1), effect_class="stale_audio",
       note="veto below llm_min_command_confidence rejects"),
    _r("C02-REM-0201", "recordame apagar la luz del living a las once de la noche", "reminder", "no_turn", None,
       None, None, "none", llm="stub:unavailable", counts=effects(llm_8101_calls=1),
       note="12 words > unvetoed_max_words: the veto fails closed while :8101 is down"),
    _r("C02-REM-0202", "recordame apagar la luz del living a las once de la noche", "reminder", "completed",
       "fast_reminder", "completed", "committed", "readback", llm="stub:turn_off@0.95",
       state="household_clock_tomorrow_at_1000",
       counts=effects(llm_8101_calls=1, reminder_rows_delta=1, spoken_utterances=1), unit="T031-U2",
       note="the reminder exemption lets the gate pass it; the veto accepts; zero HA sends; T031-U2 stores a "
            "text-only shared reminder and says it is only a notice (home actions are T031-U5)"),
    _r("C02-ALM-0301", "cancelá la alarma de las 7", "reminder", "needs_input", "fast_reminder", "needs_input",
       "not_written", "clarification", state="reminder_shared_active:alarma;timer_running_unnamed;queued_request",
       counts=SPOKEN_ONE, effect_class="wrong_family", unit="T031-U2",
       note="user answer T030-U1-Q8 (A): the alarm with a clock time is the reminder's, listed by number; the "
            "timer and the queue are untouched"),
    _r("C02-LST-0201", "agregá leche a la lista de compras", "list", "no_turn", None, None, None, "none",
       engine="whisper", llm="stub:noise", state="shared_list_compras", counts=effects(llm_8101_calls=1),
       effect_class="stale_audio", note="transcript-based wake engine: every claimed turn needs the veto"),
    _r("C02-LST-0202", "agregá leche a la lista de compras", "list", "completed", "fast_list", "completed",
       "committed", "readback", state="shared_list_compras",
       counts=effects(list_mutations=1, spoken_utterances=1),
       note="7 words and no LLMCommandRouter configured: no veto exists and there is no TV signal"),
    _r("C02-REM-0203", "recordame apagar la luz", "reminder", "no_turn", None, None, None, "none",
       wake="ambient_strict", effect_class="stale_audio",
       note="TV signal and no LLMCommandRouter configured: fails closed"),
    _r("C02-DOM-0201", "apagá la luz", "domotics", "completed", "fast_domotics", "completed", "not_required",
       "none", state="light_match", counts=effects(ha_service_calls=1, vector_searches=1),
       effect_class="wrong_family", note="regression: unclaimed domotics keeps the grammar fast path, no 7B call"),
    _r("C02-CMP-0201", "cancelá el timer y el recordatorio", "compound", "needs_input", "fast_household",
       "needs_input", "not_written", "clarification", state="queued_request", counts=SPOKEN_ONE,
       note="interim compound gate until T037-U6"),
    _r("C02-NTF-0201", "activá el no molestar", "notification", "no_turn", None, None, None, "none",
       llm="stub:turn_on@0.95", state="light_match",
       note="characterized: the gate drops it as non_immediate:negation ('no molestar') before the household "
            "layer, as today; zero effects (no turn_on from the 7B or the grammar)"),
    _r("C02-INT-0201", "avisá en la cocina que la comida está lista", "intercom", "no_turn", None, None, None,
       "none", llm="stub:noise", state="shared_list_compras", counts=effects(llm_8101_calls=1),
       effect_class="stale_audio", note="8 words: the veto runs and rejects"),
    _r("C02-RTN-0201", "quiero crear una nueva rutina que apague las luces", "routine", "unsupported",
       "fast_routine", "unsupported", "not_written", "refusal", llm="stub:turn_off@0.95", state="light_match",
       counts=effects(llm_8101_calls=1, spoken_utterances=1),
       note="P3F2-R1-Q4: before, unclaimed and the grammar fast path turned the lights off; now a routine "
            "proposal refused until T033, and a 7B that reads turn_off only passes the veto"),
    _r("C02-RTN-0202", "una rutina que apague las luces", "routine", "unsupported", "fast_routine", "unsupported",
       "not_written", "refusal", llm="stub:turn_off@0.95", state="light_match", counts=SPOKEN_ONE,
       note="6 words on a strong acoustic wake: no 7B call, no grammar, no HA send"),
    _r("C02-RTN-0203", "che armame una rutina que apague las luces a las once", "routine", "unsupported",
       "fast_routine", "unsupported", "not_written", "refusal", llm="stub:command@0.9", state="light_match",
       counts=effects(llm_8101_calls=1, spoken_utterances=1),
       note="the gate lets a claimed routine creation through like a reminder (its time is the routine's "
            "content), so the refusal is spoken instead of the silent non_immediate:deferred drop"),
    _r("C02-RTN-0204", "quiero crear una nueva rutina que apague las luces", "routine", "no_turn", None, None, None,
       "none", wake="textual", llm="stub:noise", state="light_match", counts=effects(llm_8101_calls=1),
       effect_class="stale_audio", note="TV signal: the 7B veto rejects; zero effects"),
    _r("C02-CAN-0002", "borrá el recordatorio de tomar agua", "reminder", "completed", "fast_reminder", "completed",
       "committed", "readback", state="reminder_shared_active:tomar agua;queued_request",
       counts=effects(reminder_transitions=1, spoken_utterances=1), effect_class="false_confirmation",
       seed="C02-CAN-0002", unit="T032-U1", note="T032-U1: the seed's 'Todavía no puedo crear' branch is gone; the queue is "
                                 "untouched (seed layer graph; the graph journey is in test_task_outcomes)"),
    _r("C02-CAN-0301", "sacá el recordatorio de tomar agua", "reminder", "needs_input", "fast_reminder",
       "needs_input", "not_written", "clarification", state="reminder_shared_fired:tomar agua;queued_request",
       counts=SPOKEN_ONE, effect_class="false_confirmation", unit="T032-U1", note="T032-U1: a fired reminder is never cancelled"),
    _r("C02-CAN-0302", "cancelá el recordatorio de la clave", "reminder", "needs_input", "fast_reminder",
       "needs_input", "not_written", "clarification",
       state="reminder_private_active:la clave;reminder_shared_active:tomar agua", counts=SPOKEN_ONE,
       effect_class="privacy_disclosure", unit="T032-U1", note="T032-U1: private text is never read or spoken by household voice"),
    _r("C02-CAN-0303", "el primero", "reminder", "completed", "fast_reminder", "completed", "committed", "readback",
       state="reminder_shared_active:tomar agua fría;reminder_shared_active:tomar agua con gas;queued_request;"
             "reminder_choice_open@living:cancelá el recordatorio de tomar agua",
       counts=effects(reminder_transitions=1, spoken_utterances=1), unit="T032-U1",
       note="T032-U1 numbered_choice_round: the answer is routed back to the choice open in the mic's room, on the "
            "household path (no producer, no service_filter)"),
    _r("C02-CAN-0304", "el primero", "reminder", "no_turn", None, None, None, "none", wake="textual",
       llm="stub:noise", state="reminder_shared_active:tomar agua fría;reminder_shared_active:tomar agua con gas;"
                               "reminder_choice_open@living:cancelá el recordatorio de tomar agua",
       counts=effects(llm_8101_calls=1), effect_class="stale_audio", unit="T032-U1",
       note="T032-U1 numbered_choice_round: a weakly woken answer takes the household 7B veto; rejected, zero effects"),
)

WAKES = {
    "acoustic": {},
    "textual": {"textual_wake": True},
    "ambient_strict": {"ambient_strict": True},
    "low_score": {"wake_score": 0.42},
}


class Household:
    """A real RequestRouter and MultiUserOrchestrator over counted leaves."""

    def __init__(self, leaves: CountedLeaves, *, text: str, llm: str = "absent", engine: str = "openwakeword",
                 room_context_manager=None, settings: HouseholdIntentSettings | None = None) -> None:
        self.leaves = leaves
        self.classifier = HouseholdIntentClassifier(settings or HouseholdIntentSettings(wake_words=("nexa",)))
        self.handler = response_handler_mock()
        self.orchestrator = MultiUserOrchestrator(
            chroma_sync=leaves.chroma, ha_client=leaves.ha, routine_manager=leaves.routines, router=leaves.router,
            music_dispatcher=leaves.music, list_manager=leaves.list_manager,
            reminder_manager=leaves.reminder_manager, response_handler=self.handler,
            household_intents=self.classifier, timer_manager=leaves.timers,
            timer_fallback_zone=leaves.timer_fallback_zone)
        leaves.attach(self.orchestrator.dispatcher)
        self.orchestrator.process = AsyncMock(side_effect=self.orchestrator.process)
        self.veto = None
        if llm != "absent":
            self.veto = MagicMock()
            self.veto.classify = AsyncMock(return_value=STUBS[llm.removeprefix("stub:")])
        processor = MagicMock()
        processor.process_command = AsyncMock(return_value=ProcessedCommand(text=text, timings={"stt": 1.0},
                                                                           success=True))
        self.audio = MagicMock()
        self.audio.detect_source_zone = MagicMock(return_value=None)
        self.router = RequestRouter(
            command_processor=processor, response_handler=self.handler, audio_manager=self.audio,
            room_context_manager=room_context_manager, orchestrator=self.orchestrator, orchestrator_enabled=True,
            chroma_sync=leaves.chroma, ha_client=leaves.ha, wake_words=("nexa",),
            command_gate=CommandAcceptanceGate(household_intents=self.classifier),
            llm_command_router=self.veto, wake_acoustically_confirmed=engine != "whisper",
            household_intents=self.classifier)

    async def turn(self, wake: str = "acoustic", room_id: str = "living") -> dict:
        event = CommandEvent(audio=np.full(16000, .1, dtype=np.float32), room_id=room_id, **WAKES[wake])
        return await self.router.process_command(event)

    def effects(self, before) -> dict:
        calls = self.veto.classify.await_count if self.veto is not None else 0
        return self.leaves.effects(before, llm_8101_calls=calls, spoken=self.handler.aspeak.await_count)


@pytest.fixture
async def leaves(tmp_path):
    counted = CountedLeaves(tmp_path)
    await counted.start()
    yield counted
    await counted.close()


@pytest.mark.parametrize("entry", C02_ROUTER, ids=lambda e: e.id)
async def test_c02_router_entries(leaves, entry):
    house = Household(leaves, text=entry.text, llm=entry.conditions["llm"], engine=entry.conditions["engine"])
    await leaves.apply_state(house.orchestrator.dispatcher, entry.conditions.get("state"))
    before = leaves.snapshot()
    leaves.cancels.clear()
    result = await house.turn(entry.conditions["wake"])
    assert house.effects(before) == entry.effects
    house.audio.detect_source_zone.assert_not_called() if entry.family != "domotics" else None
    if entry.expected_outcome == "no_turn":
        house.orchestrator.process.assert_not_awaited()
        assert result["success"] is False and result["response"] == ""
        assert result["intent"].startswith(("llm_rejected:household", "gate_rejected:non_immediate:"))
        return
    assert result["path"] == entry.path
    assert result["outcome"]["status"] == entry.task_status
    assert result["outcome"]["persistence"] == entry.persistence
    if entry.family not in ("domotics",):
        kwargs = house.orchestrator.process.await_args.kwargs
        assert (kwargs["service_filter"], kwargs["query_slots"]) == (None, None)
        assert "target_entity_id" not in kwargs
        assert kwargs["context_room_id"] == "living" and kwargs["user_id"] is None


async def test_claimed_turn_is_decided_before_home_command_grammar_and_7b(leaves, monkeypatch):
    """S5 sits before _try_home_command and the grammar fast path (design F26)."""
    import src.pipeline.request_router as module
    grammar = MagicMock(side_effect=AssertionError("grammar reached"))
    monkeypatch.setattr(module, "_grammar_fastpath_classification", grammar)
    house = Household(leaves, text="recordame apagar la luz", llm="stub:turn_off@0.95")
    house.router._try_home_command = AsyncMock(side_effect=AssertionError("home command reached"))
    result = await house.turn()
    assert result["path"] == "fast_reminder"
    assert result["household"] == {"family": "reminder", "operation": "create", "rule": "R3", "reason": None,
                                   "parts": 0}
    house.veto.classify.assert_not_awaited()


@pytest.mark.parametrize(("wake", "engine", "words", "vetoed"), [
    ("acoustic", "openwakeword", "poné un timer de 5 minutos", False),
    ("textual", "openwakeword", "poné un timer de 5 minutos", True),
    ("ambient_strict", "openwakeword", "poné un timer de 5 minutos", True),
    ("low_score", "openwakeword", "poné un timer de 5 minutos", True),
    ("acoustic", "whisper", "poné un timer de 5 minutos", True),
    ("acoustic", "openwakeword", "che poné un timer de 5 minutos", True),  # 7 words
])
async def test_weak_wake_needs_the_7b_veto(leaves, wake, engine, words, vetoed):
    house = Household(leaves, text=words, llm="stub:noise", engine=engine)
    result = await house.turn(wake)
    assert house.veto.classify.await_count == (1 if vetoed else 0)
    assert (result["intent"] == "llm_rejected:household") is vetoed
    assert (result.get("path") == "fast_timer") is not vetoed


async def test_word_limit_comes_from_settings(leaves):
    settings = HouseholdIntentSettings(unvetoed_max_words=8, wake_words=("nexa",))
    house = Household(leaves, text="che poné un timer de 5 minutos", llm="stub:noise", settings=settings)
    result = await house.turn()
    assert result["path"] == "fast_timer" and house.veto.classify.await_count == 0


async def test_unavailable_never_passes_the_veto(leaves):
    house = Household(leaves, text="poné un timer de 5 minutos", llm="stub:unavailable_command")
    result = await house.turn("textual")
    assert result["intent"] == "llm_rejected:household"
    house.orchestrator.process.assert_not_awaited()


async def test_veto_confidence_above_one_never_passes(leaves):
    """Same acceptance test as _try_home_command: a confidence outside [min, 1] is not a pass."""
    house = Household(leaves, text="poné un timer de 5 minutos", llm="stub:command@1.5")
    result = await house.turn("textual")
    assert result["intent"] == "llm_rejected:household"
    house.orchestrator.process.assert_not_awaited()
    accepted = Household(leaves, text="poné un timer de 5 minutos", llm="stub:command@0.9")
    assert (await accepted.turn("textual"))["path"] == "fast_timer"


async def test_veto_exception_fails_closed(leaves):
    house = Household(leaves, text="poné un timer de 5 minutos", llm="stub:noise")
    house.veto.classify.side_effect = RuntimeError(":8101 down")
    result = await house.turn("textual")
    assert result["intent"] == "llm_rejected:household"
    house.orchestrator.process.assert_not_awaited()


async def test_reply_zone_follows_room_context_and_never_audio_detection(leaves):
    from src.core.household_clock import RoomNamespace
    from src.rooms.room_context import RoomConfig, RoomContextManager
    rooms = RoomContextManager(fallback_room="living")
    for room in ("living", "hall"):
        rooms.add_room(RoomConfig(room_id=room, name=room.title(), display_name=f"el {room}"))
    house = Household(leaves, text="nexa qué recordatorios hay", room_context_manager=rooms)
    house.orchestrator.dispatcher.room_namespace = RoomNamespace(("living", "hall"))
    hall = await house.turn(room_id="hall")
    patio = await house.turn(room_id="patio")
    calls = house.orchestrator.process.await_args_list
    assert [c.kwargs["zone_id"] for c in calls] == ["zone_hall", "zone_living"]
    assert [c.kwargs["context_room_id"] for c in calls] == ["hall", "patio"]
    assert hall["path"] == patio["path"] == "fast_reminder"
    house.audio.detect_source_zone.assert_not_called()
    assert house.handler.set_active_zone.call_args_list[-1].args == ("zone_living",)


async def test_router_and_dispatcher_agree_on_every_matrix_text(leaves):
    """Every caller of dispatch gets the same precedence as the router (design D1)."""
    from tests.unit.nlu.test_household_intents import PRECEDENCE
    from tests.unit.orchestrator.test_dispatcher_household_intents import C02_DISPATCHER
    house = Household(leaves, text="x")
    texts = [t for t, _ in PRECEDENCE] + [e.text for e in C02_DISPATCHER + C02_ROUTER]
    for text in texts:
        assert house.router.household_intents.classify(text) == house.orchestrator.dispatcher.household_intents.classify(text)
    assert house.router.household_intents is house.orchestrator.dispatcher.household_intents


async def test_default_router_builds_one_classifier_for_itself_and_its_gate(leaves):
    router = RequestRouter(command_processor=MagicMock(), response_handler=response_handler_mock(),
                           audio_manager=MagicMock(), wake_words=("jarvis",))
    assert router.household_intents.settings.wake_words == ("jarvis",)
    assert router.command_gate._household_intents is router.household_intents


ROUTINE_CREATE = "Todavía no creo rutinas por voz. Podés crearlas desde la web."


@pytest.mark.parametrize("text", [
    "quiero crear una nueva rutina que apague las luces", "quisiera crear una rutina que apague las luces del living",
    "creá una rutina para apagar las luces a la noche", "armame una rutina que apague las luces",
    "haceme una rutina que apague todas las luces", "una rutina que apague las luces",
    "nueva automatización que apague las luces", "quiero que me armes una rutina que apague las luces",
    "me podés armar una rutina que prenda la luz del living", "dale creá una rutina que apague las luces",
    "che armame una rutina que apague las luces a las once",
    "creá una rutina que cuando salga el sol prenda la luz",
])
async def test_routine_creation_request_never_reaches_a_domotics_producer(leaves, monkeypatch, text):
    """P3F2-R1-Q4: claimed before the home command, the grammar and the 7B service_filter; refused honestly."""
    import src.pipeline.request_router as module
    monkeypatch.setattr(module, "_grammar_fastpath_classification",
                        MagicMock(side_effect=AssertionError("grammar reached")))
    house = Household(leaves, text=text, llm="stub:turn_off@0.95")
    house.router._try_home_command = AsyncMock(side_effect=AssertionError("home command reached"))
    await leaves.apply_state(house.orchestrator.dispatcher, "light_match")
    before = leaves.snapshot()
    result = await house.turn()
    vetoes = 1 if len(text.split()) > 6 else 0
    assert house.effects(before) == effects(llm_8101_calls=vetoes, spoken_utterances=1)
    assert (result["path"], result["outcome"]["status"], result["response"]) == (
        "fast_routine", "unsupported", ROUTINE_CREATE)
    assert result["household"] == {"family": "routine", "operation": "propose", "rule": "R6",
                                   "reason": "new_routine", "parts": 0}
    kwargs = house.orchestrator.process.await_args.kwargs
    assert (kwargs["service_filter"], kwargs["query_slots"]) == (None, None)
    house.audio.detect_source_zone.assert_not_called()


def test_gate_passes_a_claimed_routine_creation_but_keeps_the_silent_drops():
    """Q4 speaks the routine refusal; Q1/Q5 keep the silent drop of deferred actions and 'no molestar'."""
    gate = CommandAcceptanceGate()
    for text in ("che armame una rutina que apague las luces a las once",
                 "creá una rutina que cuando salga el sol prenda la luz"):
        assert gate.evaluate(text).accept is True, text
    # Only a routine creation is exempt: running a routine with a deferred action stays gated.
    for text in ("apagá la luz en 10 minutos", "activá el no molestar",
                 "ejecutá la rutina que apaga la luz en 10 minutos"):
        decision = gate.evaluate(text)
        assert decision.accept is False and decision.reason.startswith("non_immediate:"), text
    # Q4-RV-4: a routine PROPOSE that is not a creation (no reason new_routine) keeps the drop too.
    classifier = HouseholdIntentClassifier()
    for text in ("programá la rutina que apaga la luz en 10 minutos",
                 "hacé la rutina de apagar las luces a las siete"):
        claimed = classifier.classify(text)
        assert (claimed.family.value, claimed.operation.value, claimed.reason) == ("routine", "propose", None), text
        decision = gate.evaluate(text)
        assert decision.accept is False and decision.reason.startswith("non_immediate:"), text


# Due at 09:00 and 10:00 tomorrow: the bare 'el dos' is no listed due hour (user answer T032-U1-Q3).
TWO_SIMILAR_OPEN = ("reminder_shared_at_0900:tomar agua fría;reminder_shared_at_1000:tomar agua con gas;"
                    "queued_request;reminder_choice_open@living:cancelá el recordatorio de tomar agua")


@pytest.mark.parametrize("others", [("poné la persiana",), ("gracias",), ("poné la persiana", "gracias")],
                         ids=["router_clarification", "gate_reject", "both"])
async def test_a_turn_the_router_ends_itself_drops_the_open_reminder_choice(leaves, others):
    """Review T032-NUM-FV-2: only the next turn may answer; a turn ended before the dispatcher drops the choice.

    At 538ef6a1 'poné la persiana' (router clarification) left the list open
    and 'el dos' then cancelled 'tomar agua con gas'.
    """
    house = Household(leaves, text="el dos")
    dispatcher = house.orchestrator.dispatcher
    await leaves.apply_state(dispatcher, TWO_SIMILAR_OPEN)
    assert house.orchestrator.awaits_reminder_choice("el dos", "living")
    processor = house.router.command_processor.process_command
    for other in others:
        processor.return_value = ProcessedCommand(text=other, timings={"stt": 1.0}, success=True)
        before = leaves.snapshot()
        await house.turn()
        house.orchestrator.process.assert_not_awaited()  # the router ended the turn itself
        assert house.effects(before)["reminder_transitions"] == 0
    assert not house.orchestrator.awaits_reminder_choice("el dos", "living")
    processor.return_value = ProcessedCommand(text="el dos", timings={"stt": 1.0}, success=True)
    before = leaves.snapshot()
    result = await house.turn()
    assert house.effects(before)["reminder_transitions"] == 0 and result.get("path") != "fast_reminder"
    states = {r.text: r.state.value for r in await leaves.reminder_manager.get_active(None, shared_only=True)}
    assert states == {"tomar agua fría": "active", "tomar agua con gas": "active"}


async def test_a_turn_in_another_room_keeps_the_open_reminder_choice(leaves):
    """The choice waits on the mic's room: a turn ended by the router in another room does not drop it."""
    house = Household(leaves, text="gracias")
    await leaves.apply_state(house.orchestrator.dispatcher, TWO_SIMILAR_OPEN)
    await house.turn(room_id="hall")
    assert house.orchestrator.awaits_reminder_choice("el dos", "living")
    house.router.command_processor.process_command.return_value = ProcessedCommand(
        text="el dos", timings={"stt": 1.0}, success=True)
    before = leaves.snapshot()
    result = await house.turn()
    assert house.effects(before)["reminder_transitions"] == 1 and result["outcome"]["status"] == "completed"
    states = {r.text: r.state.value for r in await leaves.reminder_manager.get_active(None, shared_only=True)}
    assert states == {"tomar agua fría": "active"}


async def test_the_hour_question_and_its_answer_go_through_the_router(leaves):
    """User answer T032-U1-Q3 end to end: 'el dos' with the first listed due at 14:00 asks and cancels nothing;
    'el de las 2' is then routed back to the choice in the mic's room and cancels exactly that one."""
    house = Household(leaves, text="el dos")
    await leaves.apply_state(house.orchestrator.dispatcher,
                             "reminder_shared_at_1400:tomar agua fría;reminder_shared_at_2100:tomar agua con gas;"
                             "queued_request;reminder_choice_open@living:cancelá el recordatorio de tomar agua")
    before = leaves.snapshot()
    asked = await house.turn()
    assert house.effects(before) == effects(spoken_utterances=1)
    assert (asked["response"], asked["outcome"]["status"]) == ("¿El segundo o el de las 2?", "needs_input")
    assert house.orchestrator.awaits_reminder_choice("el de las 2", "living")
    assert not house.orchestrator.awaits_reminder_choice("el de las 2", "hall")
    house.router.command_processor.process_command.return_value = ProcessedCommand(
        text="el de las 2", timings={"stt": 1.0}, success=True)
    before = leaves.snapshot()
    done = await house.turn()
    assert house.effects(before)["reminder_transitions"] == 1 and house.handler.aspeak.await_count == 2
    assert done["outcome"]["status"] == "completed" and done["response"].startswith(
        "Cancelé el recordatorio tomar agua fría a las ")
    states = {r.text: r.state.value for r in await leaves.reminder_manager.get_active(None, shared_only=True)}
    assert states == {"tomar agua con gas": "active"}
