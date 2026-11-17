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
    "noise": CommandClassification(is_command=False, confidence=0.1, intent="noise", rejection_reason="noise"),
    "unavailable": CommandClassification(is_command=None, rejection_reason="unavailable"),
    # Defensive shape: a command claim that also reports 'unavailable' is still not a veto pass.
    "unavailable_command": CommandClassification(is_command=True, confidence=0.9, rejection_reason="unavailable"),
}


def _r(id_, text, family, outcome, path, status, persistence, spoken, *, wake="acoustic", llm="absent",
       engine="openwakeword", counts=None, state=None, effect_class="unauthorized_effect", seed=None, note=""):
    conditions = {"wake": wake, "llm": llm, "engine": engine}
    if state:
        conditions["state"] = state
    return C02Entry(id_, text, "router", family, effect_class, outcome, path, status, persistence, spoken,
                    counts or effects(), conditions, seed=seed, note=note)


SPOKEN_ONE = effects(spoken_utterances=1)
INTERIM = "interim until T031-U2: honest UNSUPPORTED instead of the seed's needs_input clarification"

C02_ROUTER = (
    _r("C02-REM-0001", "recordame apagar la luz", "reminder", "unsupported", "fast_reminder", "unsupported",
       "not_written", "refusal", counts=SPOKEN_ONE, note=INTERIM + "; CORE-02 closed: no grammar turn_off"),
    _r("C02-REM-0004", "recordame abrir la persiana mañana", "reminder", "unsupported", "fast_reminder",
       "unsupported", "not_written", "refusal", llm="stub:open@0.9", counts=SPOKEN_ONE, note=INTERIM),
    _r("C02-REM-0008", "recordale a Ana apagar la luz", "reminder", "unsupported", "fast_reminder", "unsupported",
       "not_written", "refusal", llm="stub:turn_off@0.95", counts=SPOKEN_ONE),
    _r("C02-REM-0013", "recordame apagar la luz", "reminder", "unsupported", "fast_reminder", "unsupported",
       "not_written", "refusal", wake="textual", llm="stub:turn_off@0.95",
       counts=effects(llm_8101_calls=1, spoken_utterances=1),
       note="7B veto accepts; service_filter still forced to None; " + INTERIM),
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
    _r("C02-TIM-0201", "poné un temporizador de 5 minutos", "timer", "unsupported", "fast_timer", "unsupported",
       "not_written", "refusal", llm="stub:turn_on@0.95", counts=SPOKEN_ONE,
       note="strong acoustic wake, 6 words: no 7B call, no vector search"),
    _r("C02-TIM-0202", "poné un temporizador de 5 minutos", "timer", "unsupported", "fast_timer", "unsupported",
       "not_written", "refusal", wake="low_score", llm="stub:command@0.9",
       counts=effects(llm_8101_calls=1, spoken_utterances=1)),
    _r("C02-TIM-0203", "poné un temporizador de 5 minutos", "timer", "no_turn", None, None, None, "none",
       wake="low_score", llm="stub:command@0.3", counts=effects(llm_8101_calls=1), effect_class="stale_audio",
       note="veto below llm_min_command_confidence rejects"),
    _r("C02-REM-0201", "recordame apagar la luz del living a las once de la noche", "reminder", "no_turn", None,
       None, None, "none", llm="stub:unavailable", counts=effects(llm_8101_calls=1),
       note="12 words > unvetoed_max_words: the veto fails closed while :8101 is down"),
    _r("C02-REM-0202", "recordame apagar la luz del living a las once de la noche", "reminder", "unsupported",
       "fast_reminder", "unsupported", "not_written", "refusal", llm="stub:turn_off@0.95",
       counts=effects(llm_8101_calls=1, spoken_utterances=1),
       note="the reminder exemption lets the gate pass it; the veto accepts; zero HA sends; " + INTERIM),
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
            household_intents=self.classifier)
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
