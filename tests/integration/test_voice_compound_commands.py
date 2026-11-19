"""T037-U6: compound voice turns run part by part through the real router, dispatcher and home actions.

User answer Q21: every part of an admissible compound runs now, in spoken
order, through the gates and handler it would meet alone, with its own receipt
and request id; one reply says what was done, what was not and why, what is
uncertain and what was not attempted. Seeds C02-CMP-0001 and 0003..0009
(data/autonomous-roadmap-20261102/p3/design.json) with every C02 counter.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.core.household_clock import RoomNamespace
from src.home_assistant.control_policy import EntityGrant, HomeActor, HomeControlPolicy
from src.home_assistant.home_actions import HomeActionService
from src.home_assistant.voice_resolver import HomeVoiceResolver
from src.nlu.command_gate import CommandAcceptanceGate
from src.nlu.household_intents import HouseholdIntentClassifier, HouseholdIntentSettings
from src.nlu.llm_router import CommandClassification
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.pipeline.command_event import CommandEvent
from src.pipeline.command_processor import ProcessedCommand
from src.pipeline.request_router import RequestRouter
from src.rooms.room_context import RoomConfig, RoomContextManager
from src.timers.named_timers import NamedTimerManager
from tests.integration.test_home_control_policy import ObservedHA
from tests.mocks.c02_matrix import effects
from tests.mocks.responses import response_handler_mock

LIGHTS = ("light.escritorio", "light.cocina")


class CompoundHA(ObservedHA):
    """Two selected lights; an entity in ``unobserved`` accepts the send but never shows the effect."""

    state_confirmation_timeout_s = 0.05

    def __init__(self):
        super().__init__()
        for entity_id in LIGHTS:
            self.states[entity_id] = {"entity_id": entity_id, "state": "on",
                                      "attributes": {"supported_color_modes": ["brightness"]}}
        self.unobserved: set[str] = set()
        self.started = asyncio.Event()
        self.release: asyncio.Event | None = None

    async def call_service_once(self, domain, service, entity_id, data, *, guard=None, on_dispatch=None):
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        observe = self.observe_effect
        self.observe_effect = entity_id not in self.unobserved
        try:
            return await super().call_service_once(domain, service, entity_id, data, guard=guard,
                                                   on_dispatch=on_dispatch)
        finally:
            self.observe_effect = observe


class Graph:
    """A real RequestRouter, MultiUserOrchestrator, HomeActionService and NamedTimerManager."""

    def __init__(self, *, llm: CommandClassification | None = None, settings: HouseholdIntentSettings | None = None):
        self.ha = CompoundHA()
        self.actions = HomeActionService(self.ha, policy=HomeControlPolicy(tuple(
            EntityGrant(entity_id, frozenset({"turn_on", "turn_off"})) for entity_id in LIGHTS)))
        self.rooms = RoomContextManager()
        configs = {"escritorio": RoomConfig("escritorio", "Escritorio", "el escritorio",
                                            default_light="light.escritorio"),
                   "cocina": RoomConfig("cocina", "Cocina", "la cocina", default_light="light.cocina")}
        for config in configs.values():
            self.rooms.add_room(config)
        self.classifier = HouseholdIntentClassifier(settings or HouseholdIntentSettings(wake_words=("nexa",)))
        self.chroma = MagicMock()
        self.chroma.asearch_command = AsyncMock(return_value=None)
        self.chroma.search_command = MagicMock(return_value=None)
        self.music = MagicMock()
        self.music.process = AsyncMock(return_value=SimpleNamespace(
            success=True, response="Pausé la música", details={}, intent=SimpleNamespace(value="pause")))
        self.timers = NamedTimerManager(tts_callback=AsyncMock())
        self.handler = response_handler_mock()
        self.orchestrator = MultiUserOrchestrator(
            self.chroma, self.ha, MagicMock(), home_actions=self.actions, music_dispatcher=self.music,
            response_handler=self.handler, household_intents=self.classifier, timer_manager=self.timers,
            timer_fallback_zone="zone_escritorio", room_namespace=RoomNamespace(tuple(configs)))
        self.orchestrator.process = AsyncMock(side_effect=self.orchestrator.process)
        self.veto = None
        if llm is not None:
            self.veto = MagicMock()
            self.veto.classify = AsyncMock(return_value=llm)
        self.processor = MagicMock()
        self.processor.process_command = AsyncMock()
        self.router = RequestRouter(
            self.processor, self.handler, MagicMock(), room_context_manager=self.rooms,
            orchestrator=self.orchestrator, orchestrator_enabled=True, chroma_sync=self.chroma, ha_client=self.ha,
            home_actions=self.actions, home_voice_resolver=HomeVoiceResolver(self.actions, configs),
            command_gate=CommandAcceptanceGate(household_intents=self.classifier), llm_command_router=self.veto,
            wake_acoustically_confirmed=True, household_intents=self.classifier, wake_words=("nexa",))

    async def turn(self, text: str, *, room_id: str = "escritorio", request_id: str | None = None,
                   **metadata) -> dict:
        self.processor.process_command.return_value = ProcessedCommand(text=text, user=None, emotion=None,
                                                                       timings={}, success=True)
        event = CommandEvent(np.full(16000, .1, dtype=np.float32), room_id, capture_complete=True, **metadata)
        options = {"request_id": request_id} if request_id is not None else {}
        return await self.router.process_command(event, **options)

    def effects(self, *, timers_before: int = 0) -> dict:
        return effects(
            ha_service_calls=len(self.ha.sends), vector_searches=self.chroma.asearch_command.await_count,
            music_calls=self.music.process.await_count, timer_delta=len(self.timers.get_all_timers()) - timers_before,
            llm_8101_calls=self.veto.classify.await_count if self.veto is not None else 0,
            spoken_utterances=self.handler.aspeak.await_count)

    def receipt(self, request_id: str):
        return self.actions.get_receipt(actor=HomeActor.household(), request_id=request_id)

    def spoken(self) -> str:
        return self.handler.aspeak.await_args.args[0]


@pytest.fixture
async def graph():
    built = Graph()
    await built.timers.start()
    yield built
    await built.timers.stop()


async def test_cmp_0001_both_parts_done_each_with_its_own_receipt(graph):
    """C02-CMP-0001: 'apagá la luz y pausá la música' runs both parts; one reply names both as done."""
    result = await graph.turn("apagá la luz y pausá la música", request_id="voice:cmp1")
    assert graph.effects() == effects(ha_service_calls=1, music_calls=1, spoken_utterances=1)
    assert graph.ha.sends == [("light", "turn_off", "light.escritorio", None)]
    assert [(p["verdict"], p["task_status"], p["dispatch"]) for p in result["compound_parts"]] == [
        ("done", "completed", "accepted"), ("done", "completed", "accepted")]
    assert [p["request_id"] for p in result["compound_parts"]] == ["voice:cmp1:part1", "voice:cmp1:part2"]
    assert graph.receipt("voice:cmp1:part1").status == "confirmed"
    assert result["task_completed"] is True and result["outcome"]["status"] == "completed"
    assert result["outcome"]["dispatch"] == "accepted"
    assert [part["status"] for part in result["outcome"]["parts"]] == ["completed", "completed"]
    assert graph.spoken() == "Apagá la luz: hecho. Pausá la música: Pausé la música."
    assert result["path"] == "fast_household" and result["intent"] == "compound"


async def test_cmp_0004_lock_part_refused_and_the_other_done(graph):
    """C02-CMP-0004: the lock part is refused by its own part (never sent); the light part still runs."""
    result = await graph.turn("apagá la luz del escritorio y abrí la cerradura")
    assert graph.effects() == effects(ha_service_calls=1, spoken_utterances=1)
    assert [(p["verdict"], p["task_status"], p["dispatch"]) for p in result["compound_parts"]] == [
        ("done", "completed", "accepted"), ("not_done", "unsupported", "not_sent")]
    assert result["outcome"]["status"] == "failed" and result["outcome"]["dispatch"] == "partial"
    assert graph.spoken() == ("Apagá la luz del escritorio: hecho. Abrí la cerradura: no lo hice. "
                              "Las cerraduras, las puertas y la alarma no las manejo por voz.")
    assert "Listo" not in graph.spoken()


async def test_cmp_0005_uncertain_first_part_does_not_stop_an_independent_second(graph):
    """C02-CMP-0005: an uncertain first part is reported, the second still runs, nothing is resent."""
    graph.ha.unobserved.add("light.escritorio")
    result = await graph.turn("apagá la luz del escritorio y prendé la luz de la cocina")
    assert graph.effects() == effects(ha_service_calls=2, spoken_utterances=1)
    assert graph.ha.sends == [("light", "turn_off", "light.escritorio", None),
                              ("light", "turn_on", "light.cocina", None)]
    assert [(p["verdict"], p["task_status"], p["dispatch"]) for p in result["compound_parts"]] == [
        ("uncertain", "uncertain", "accepted"), ("done", "completed", "accepted")]
    assert result["outcome"]["status"] == "uncertain"
    assert graph.spoken().startswith("Apagá la luz del escritorio: no sé si se hizo; fijate antes de repetirlo.")


async def test_cmp_0006_barge_in_between_parts_stops_the_rest(graph):
    """C02-CMP-0006: another turn accepted in the room while part 1 runs: part 2 is not attempted."""
    graph.ha.release = asyncio.Event()
    compound = asyncio.create_task(graph.turn("apagá la luz del escritorio y pausá la música"))
    await asyncio.wait_for(graph.ha.started.wait(), 5)
    clock = await graph.turn("qué hora es")  # never reaches HA
    graph.ha.release.set()
    result = await compound
    assert clock["path"] == "fast_clock"
    assert graph.music.process.await_count == 0
    assert len(graph.ha.sends) == 1
    assert [(p["verdict"], p["task_status"], p["dispatch"]) for p in result["compound_parts"]] == [
        ("done", "completed", "accepted"), ("interrupted", "cancelled", "not_sent")]
    assert result["outcome"]["status"] == "cancelled" and result["outcome"]["dispatch"] == "partial"
    reply = graph.handler.aspeak.await_args_list[-1].args[0]
    assert reply.endswith("Pausá la música: no lo intenté porque me interrumpiste.")


async def test_cmp_0007_conflicting_parts_on_one_entity_clarify_with_zero_effects(graph):
    """C02-CMP-0007: 'prendé la luz y apagá la luz' is not admissible: NEEDS_INPUT, nothing runs."""
    result = await graph.turn("prendé la luz y apagá la luz")
    assert graph.effects() == effects(spoken_utterances=1)
    assert result["outcome"]["status"] == "needs_input" and "compound_parts" not in result
    assert graph.spoken() == "Decímelo de a una cosa."


async def test_cmp_0003_two_household_objects_each_reach_their_family(graph):
    """C02-CMP-0003: 'cancelá el timer y el recordatorio' runs the timer part and the reminder part."""
    timer = graph.timers.create_timer("", 300, user_id="household", zone_id="zone_escritorio")
    result = await graph.turn("cancelá el timer y el recordatorio")
    assert graph.timers.get_timer(timer.timer_id) is None or graph.timers.get_timer(timer.timer_id).state.value \
        == "cancelled"
    assert graph.effects(timers_before=1)["ha_service_calls"] == 0
    verdicts = [(p["verdict"], p["path"]) for p in result["compound_parts"]]
    assert verdicts[0] == ("done", "fast_timer")
    # No reminder manager in this graph: the reminder part is refused by its own handler, never dropped.
    assert verdicts[1] == ("not_done", "fast_reminder")
    assert result["outcome"]["status"] == "failed"


async def test_cmp_0009_timer_duration_part_and_light_part_both_run(graph):
    """C02-CMP-0009: a timer duration is the timer's own argument; both parts run, timer in the mic's room."""
    result = await graph.turn("poné un temporizador de 5 minutos y apagá la luz")
    assert graph.effects() == effects(ha_service_calls=1, timer_delta=1, spoken_utterances=1)
    (created,) = graph.timers.get_all_timers()
    assert created.zone_id == "zone_escritorio"
    assert [p["verdict"] for p in result["compound_parts"]] == ["done", "done"]
    assert result["outcome"]["status"] == "completed"


async def test_duplicate_target_is_sent_once(graph):
    """Two parts that resolve to the same action on one light: the second is not sent again."""
    result = await graph.turn("apagá la luz y apagá la luz del escritorio")
    assert graph.ha.sends == [("light", "turn_off", "light.escritorio", None)]
    assert [p["verdict"] for p in result["compound_parts"]] == ["done", "duplicate"]
    assert result["outcome"]["status"] == "failed"
    assert graph.spoken().endswith("Apagá la luz del escritorio: no lo repetí porque era lo mismo que lo anterior.")


async def test_opposite_action_on_the_same_resolved_entity_is_not_attempted(graph):
    """Rooms differ for the classifier, but both parts resolve to one light: the undo is not sent."""
    result = await graph.turn("apagá la luz y prendé la luz del escritorio")
    assert graph.ha.sends == [("light", "turn_off", "light.escritorio", None)]
    assert [p["verdict"] for p in result["compound_parts"]] == ["done", "conflict"]


async def test_part_on_the_same_light_as_an_uncertain_part_is_not_attempted(graph):
    graph.ha.unobserved.add("light.escritorio")
    result = await graph.turn("apagá la luz y apagá la luz del escritorio")
    assert len(graph.ha.sends) == 1
    assert [p["verdict"] for p in result["compound_parts"]] == ["uncertain", "same_target"]
    assert result["outcome"]["status"] == "uncertain"


async def test_retried_turn_resends_no_part(graph):
    """The same turn request id again: no part is sent or run again; the reply reports the first run."""
    first = await graph.turn("apagá la luz y pausá la música", request_id="voice:retry")
    second = await graph.turn("apagá la luz y pausá la música", request_id="voice:retry")
    assert len(graph.ha.sends) == 1 and graph.music.process.await_count == 1
    assert [p["retried"] for p in second["compound_parts"]] == [True, True]
    assert second["outcome"]["status"] == first["outcome"]["status"] == "completed"


async def test_parts_run_in_spoken_order(graph):
    sends_before_music = []
    graph.music.process.side_effect = lambda *a, **k: sends_before_music.append(len(graph.ha.sends)) or \
        SimpleNamespace(success=True, response="Pausé la música", details={}, intent=SimpleNamespace(value="pause"))
    result = await graph.turn("pausá la música y apagá la luz del escritorio y prendé la luz de la cocina")
    assert sends_before_music == [0]
    assert [send[2] for send in graph.ha.sends] == ["light.escritorio", "light.cocina"]
    assert [p["index"] for p in result["compound_parts"]] == [1, 2, 3]


async def test_unresolved_light_location_part_asks_and_the_other_runs(graph):
    """T037-U5 guard per part: an unknown room asks for that part; music still runs; no vector search."""
    result = await graph.turn("apagá la luz del quincho y pausá la música")
    assert graph.effects() == effects(music_calls=1, spoken_utterances=1)
    assert [(p["verdict"], p["task_status"]) for p in result["compound_parts"]] == [
        ("not_done", "needs_input"), ("done", "completed")]
    assert graph.spoken().startswith("Apagá la luz del quincho: no lo hice. ¿En qué ambiente querés controlar la luz?")


async def test_sensitive_part_still_needs_its_own_confirmation(graph, monkeypatch):
    import src.pipeline.request_router as module
    monkeypatch.setattr(module, "is_sensitive", lambda intent, entity: (intent, entity) == ("turn_off", "light"))
    result = await graph.turn("apagá la luz y pausá la música")
    assert graph.effects() == effects(music_calls=1, spoken_utterances=1)
    assert [(p["verdict"], p["task_status"]) for p in result["compound_parts"]] == [
        ("not_done", "needs_input"), ("done", "completed")]
    assert "¿Querés apagar la luz? Decí sí o no." in graph.spoken()


async def test_parts_never_carry_a_producer_hint_and_no_7b_chooses_them(graph):
    """Claimed parts reach the dispatcher without hints; unclaimed parts carry only their own grammar evidence."""
    await graph.turn("poné un temporizador de 5 minutos y apagá la luz del escritorio")
    calls = [call.kwargs for call in graph.orchestrator.process.await_args_list]
    assert [c["text"] for c in calls] == ["poné un temporizador de 5 minutos", "apagá la luz del escritorio"]
    assert (calls[0]["service_filter"], calls[0]["query_slots"]) == (None, None)
    assert "target_entity_id" not in calls[0]
    assert (calls[1]["service_filter"], calls[1]["target_entity_id"]) == ("turn_off", "light.escritorio")
    assert all(c["user_id"] is None and c["context_room_id"] == "escritorio" for c in calls)
    assert graph.chroma.asearch_command.await_count == 0


async def test_weak_wake_compound_takes_the_7b_veto_once():
    """A borrowed wake (follow-up capture) needs the 7B veto; it runs once for the whole utterance."""
    graph = Graph(llm=CommandClassification(is_command=True, confidence=0.95, intent="turn_off",
                                            entity_hint="light"))
    await graph.timers.start()
    try:
        result = await graph.turn("apagá la luz del escritorio y prendé la luz de la cocina",
                                  capture_trigger="follow_up")
        assert graph.veto.classify.await_count == 1
        assert [p["verdict"] for p in result["compound_parts"]] == ["done", "done"]
        graph.veto.classify.return_value = CommandClassification(is_command=False, confidence=0.1, intent="noise",
                                                                 rejection_reason="noise")
        vetoed = await graph.turn("apagá la luz del escritorio y prendé la luz de la cocina",
                                  capture_trigger="follow_up")
        assert vetoed["intent"] == "llm_rejected:household" and len(graph.ha.sends) == 2
    finally:
        await graph.timers.stop()


async def test_kill_switch_keeps_the_interim_answer():
    graph = Graph(settings=HouseholdIntentSettings(wake_words=("nexa",), compound_execute=False))
    await graph.timers.start()
    try:
        result = await graph.turn("apagá la luz y pausá la música")
        assert graph.effects() == effects(spoken_utterances=1)
        assert result["outcome"]["status"] == "needs_input" and graph.spoken() == "Decímelo de a una cosa."
    finally:
        await graph.timers.stop()


async def test_failed_part_does_not_stop_an_independent_later_part(graph):
    """A part refused by the selection (not granted) is reported; the next part on another light runs."""
    graph.actions.policy = HomeControlPolicy((EntityGrant("light.cocina", frozenset({"turn_on", "turn_off"})),))
    result = await graph.turn("apagá la luz del escritorio y prendé la luz de la cocina")
    assert graph.ha.sends == [("light", "turn_on", "light.cocina", None)]
    assert [(p["verdict"], p["task_status"], p["dispatch"]) for p in result["compound_parts"]] == [
        ("not_done", "failed", "not_sent"), ("done", "completed", "accepted")]
    assert result["outcome"]["status"] == "failed" and result["outcome"]["dispatch"] == "partial"
    graph.handler.play_earcon.assert_not_called()


async def test_part_without_a_found_device_asks_and_never_falls_back_to_a_model(graph):
    """The vector search finds no fan: that part asks; neither the 7B nor the LLM chooses its target."""
    dispatcher = graph.orchestrator.dispatcher
    dispatcher.router = MagicMock()
    dispatcher.router.complete = AsyncMock(return_value="hola")
    dispatcher._slow_path = AsyncMock(side_effect=AssertionError("LLM reached"))
    result = await graph.turn("apagá el ventilador y pausá la música")
    assert graph.effects() == effects(vector_searches=1, music_calls=1, spoken_utterances=1)
    dispatcher.router.complete.assert_not_awaited()
    assert [(p["verdict"], p["task_status"], p["dispatch"]) for p in result["compound_parts"]] == [
        ("not_done", "needs_input", "not_sent"), ("done", "completed", "accepted")]
    assert graph.spoken().startswith("Apagá el ventilador: no lo hice. No encontré ese dispositivo.")
