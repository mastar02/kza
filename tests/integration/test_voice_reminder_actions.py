"""T031-U5 through the voice router, SQLite reminder ledger and HA policy."""

import asyncio
import json
from contextlib import AsyncExitStack
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import numpy as np
import pytest

from src.core.household_clock import HouseholdClock, RoomNamespace
from src.core.voice_delivery import LocalVoiceDelivery
from src.home_assistant.control_policy import HomeControlPolicy
from src.home_assistant.home_actions import HomeActionService
from src.lists.list_manager import ListManager
from src.lists.list_store import ListStore
from src.nlu.command_gate import CommandAcceptanceGate
from src.nlu.household_intents import HouseholdIntentClassifier, HouseholdIntentSettings
from src.nlu.llm_router import CommandClassification
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.pipeline.command_event import CommandEvent
from src.pipeline.command_processor import ProcessedCommand
from src.pipeline.request_router import RequestRouter
from src.pipeline.response_handler import ResponseHandler
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_scheduler import DEFAULT_LATE_ACTION_GRACE_S, ReminderScheduler
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
from src.rooms.room_context import RoomConfig, RoomContextManager
from src.timers.named_timers import NamedTimerManager
from tests.integration.test_home_control_policy import ObservedHA
from tests.integration.test_household_reminder_scope import ledger_rows
from tests.mocks.audio_output import AudioBackend, GatedZones
from tests.mocks.c02_matrix import effects
from tests.mocks.speech import PCMEngine

LIVING = "zone_living"
LIGHT_OFF = {"domain": "light", "service": "turn_off", "entity_id": "light.living", "data": None}
R3 = "recordame apagar la luz del living a las 11 de la noche"
R4 = "apagá la luz en 10 minutos"


class ReminderVoiceGraph:
    """Real task graph with synthetic STT, audio device and remote transports."""

    async def turn(self, text, **capture):
        self.processor.process_command.return_value = ProcessedCommand(
            text=text, user=None, emotion=None, timings={}, success=True)
        return await self.router.process_command(CommandEvent(
            np.full(16000, .1, dtype=np.float32), room_id="living", capture_complete=True,
            capture_trigger="wake", **capture))

    async def effects(self, *, previous=None, spoken_before=0):
        rows = await self.rows()
        return effects(
            ha_service_calls=len(self.ha.sends), vector_searches=self.chroma.asearch_command.await_count,
            music_calls=self.music.process.await_count, cloud_calls=self.llm.generate.await_count,
            llm_8101_calls=self.veto.classify.await_count, timer_delta=len(self.timers.get_all_timers()),
            list_mutations=len(await self.lists.get_items_by_list_id(self.shopping.id, shared_only=True)),
            reminder_rows_delta=len(rows) - (previous is not None),
            reminder_transitions=sum(row.fire_count for row in rows),
            ledger_claims=len(await ledger_rows(self.store, "")),
            spoken_utterances=self.handler.aspeak.await_count - spoken_before,
        )

    async def rows(self):
        return await self.store._work.read(lambda db: [
            self.store._row_to_reminder(row) for row in db.execute("SELECT * FROM reminders ORDER BY rowid")
        ])

    def spoken(self):
        return [call.args[0] for call in self.handler.aspeak.await_args_list]

    def set_selection(self, entities):
        self.selection.write_text(json.dumps({
            "selection_approved": True, "live_api_verified": True,
            "allowed_entities": [
                {"entity_id": entity_id, "operations": ["turn_on", "turn_off"],
                 "live_verified": True, "indirect_effects_verified": True}
                for entity_id in entities
            ],
        }))
        self.selection.chmod(0o600)


@pytest.fixture
async def graph(tmp_path, monkeypatch):
    import src.tts.piper_tts as audio

    monkeypatch.setattr(audio, "sd", AudioBackend())
    async with AsyncExitStack() as stack:
        g = ReminderVoiceGraph()
        g.selection = tmp_path / "selected.json"
        g.set_selection(("light.living", "light.cocina"))
        g.ha = ObservedHA()
        for entity in ("light.living", "light.cocina"):
            g.ha.states[entity] = {"entity_id": entity, "state": "on", "attributes": {}}
        g.actions = HomeActionService(g.ha, policy=HomeControlPolicy(selection_path=str(g.selection)))
        g.store = ReminderStore(str(tmp_path / "reminders.db"))
        await g.store.initialize()
        stack.push_async_callback(g.store.close)
        list_store = ListStore(str(tmp_path / "lists.db"))
        await list_store.initialize()
        stack.push_async_callback(list_store.close)
        g.lists = ListManager(list_store)
        g.shopping = await g.lists.create_list(None, "compras", shared=True, shared_only=True)
        g.now = [datetime.fromisoformat("2026-11-21T07:00:00-03:00").timestamp()]
        clock = HouseholdClock(time_source=lambda: g.now[0])
        g.config = {"voice_actions": True, "output_zone": LIVING}
        g.manager = ReminderManager(g.store, g.config, home_actions=g.actions, household_clock=clock)
        g.zones = GatedZones()
        g.zones.set_active_zone = Mock()
        g.handler = ResponseHandler(PCMEngine(frames=480), zone_manager=g.zones, streaming_enabled=False)
        g.handler.aspeak = AsyncMock(wraps=g.handler.aspeak)
        stack.push_async_callback(g.handler.aclose)
        g.scheduler = ReminderScheduler(g.store, LocalVoiceDelivery(g.handler), None, g.ha, g.config,
                                        home_actions=g.actions, household_clock=clock)
        stack.push_async_callback(g.scheduler.stop)
        rooms = RoomContextManager()
        for config in (
            RoomConfig("living", "Living", "el living", default_light="light.living", default_climate="climate.living"),
            RoomConfig("cocina", "Cocina", "la cocina", default_light="light.cocina"),
        ):
            rooms.add_room(config)
        classifier = HouseholdIntentClassifier(HouseholdIntentSettings(
            unvetoed_max_words=24, wake_words=("nexa",)))
        g.chroma = SimpleNamespace(asearch_command=AsyncMock(return_value=None))
        g.llm = SimpleNamespace(generate=AsyncMock())
        g.music = SimpleNamespace(process=AsyncMock())
        g.timers = NamedTimerManager(tts_callback=AsyncMock())
        orchestrator = MultiUserOrchestrator(
            g.chroma, g.ha, llm=g.llm, home_actions=g.actions, reminder_manager=g.manager, list_manager=g.lists,
            music_dispatcher=g.music, response_handler=g.handler, timer_manager=g.timers,
            household_clock=clock, household_intents=classifier, room_namespace=RoomNamespace(("living", "cocina")),
        )
        stack.push_async_callback(orchestrator.stop)
        g.veto = SimpleNamespace(classify=AsyncMock(return_value=CommandClassification(
            is_command=False, confidence=0., rejection_reason="unavailable")))
        g.processor = SimpleNamespace(process_command=AsyncMock())
        g.router = RequestRouter(
            g.processor, g.handler, MagicMock(), room_context_manager=rooms, household_clock=clock,
            orchestrator=orchestrator, orchestrator_enabled=True, ha_client=g.ha, chroma_sync=g.chroma,
            home_actions=g.actions, wake_words=("nexa",), household_intents=classifier,
            command_gate=CommandAcceptanceGate(household_intents=classifier), llm_command_router=g.veto,
            wake_acoustically_confirmed=True,
        )
        stack.push_async_callback(g.router.aclose)
        yield g


def assert_outcome(result, status, persistence):
    assert (result["outcome"]["status"], result["outcome"]["persistence"]) == (status, persistence)
    assert result["task_completed"] is (status == "completed")


@pytest.mark.parametrize(
    "phrase,text,hour,seconds",
    [(R3, "apagar la luz del living", "23:00", 16 * 3600),
     (R4, "apagar la luz", "07:10", 600)],
    ids=["C02-ACT-0001", "C02-DEF-0001"],
)
async def test_schedule_commit_only_after_readback(graph, phrase, text, hour, seconds):
    graph.zones.block = "after_first_chunk"
    turn = asyncio.create_task(graph.turn(phrase))
    try:
        assert await asyncio.to_thread(graph.zones.waiting.wait, 2)
        assert await graph.rows() == []
        assert graph.ha.sends == graph.ha.reads == []
    finally:
        graph.zones.release.set()
    result = await asyncio.wait_for(turn, 2)
    assert_outcome(result, "completed", "committed")
    assert result["path"] == "fast_reminder"
    assert await graph.effects() == effects(reminder_rows_delta=1, spoken_utterances=2)
    assert graph.ha.reads == []
    row, = await graph.rows()
    assert (row.user_id, row.scope, row.text, row.trigger_at, row.ha_actions) == (
        "household", ReminderScope.SHARED, text, graph.now[0] + seconds, [LIGHT_OFF])
    readback, receipt = graph.spoken()
    assert "apagar la luz" in readback and "living" in readback and hour in readback
    assert receipt == "Anotado."


@pytest.mark.parametrize(
    "text,denied",
    [("abrir la cerradura", "abrir la cerradura"),
     ("prender el aire", "prender el aire"),
     ("apagar la luz y abrir la cerradura", "abrir la cerradura"),
     ("activar la alarma", "activar la alarma"),
     ("apagar la luz y prender la luz del galpón", "prender la luz del galpón")],
    ids=["C02-ACT-0002", "C02-ACT-0003", "mixto", "alarma", "sin-seleccion"],
)
async def test_uncontrollable_reminder_is_text_only(graph, text, denied):
    result = await graph.turn(f"recordame {text} a las 8 de la mañana")
    assert_outcome(result, "completed", "committed")
    row, = await graph.rows()
    assert row.scope is ReminderScope.SHARED and row.text == text and row.ha_actions is None
    assert await graph.effects() == effects(reminder_rows_delta=1, spoken_utterances=2)
    assert f"No puedo ejecutar por voz: {denied}" in graph.spoken()[0]
    assert graph.spoken()[1] == "Anotado."


@pytest.mark.parametrize("clause,denied", [
    ("abrí la cerradura", "cerradura"), ("activá la alarma", "alarma"),
    ("apagá la luz y abrí la cerradura", "cerradura"), ("apagá el enchufe del galpón", "enchufe del galpón"),
])
async def test_uncontrollable_deferred_action_writes_nothing(graph, clause, denied):
    result = await graph.turn(f"{clause} en 10 minutos")
    assert_outcome(result, "unsupported", "not_written")
    assert denied in graph.spoken()[0]
    assert await graph.effects() == effects(spoken_utterances=1)


async def test_c02_act_0004_barge_in_inside_first_chunk(graph):
    graph.zones.block = "after_first_chunk"
    turn = asyncio.create_task(graph.turn(R3))
    try:
        assert await asyncio.to_thread(graph.zones.waiting.wait, 2)
        assert graph.zones.pulled == 1
        assert await graph.handler.cancel(zone_id=LIVING)
        result = await asyncio.wait_for(turn, 2)
    finally:
        graph.zones.release.set()
    assert_outcome(result, "cancelled", "not_written")
    assert await graph.effects() == effects(spoken_utterances=1)
    assert result["voice_delivery"] == "cancelled"


async def test_incomplete_delivery_writes_nothing(graph):
    graph.handler.tts.failure = RuntimeError("synthetic speech failure")
    result = await graph.turn(R4)
    assert_outcome(result, "needs_input", "not_written")
    assert await graph.effects() == effects(spoken_utterances=1)
    assert result["voice_delivery"] == "failed"


@pytest.mark.parametrize("mode", ["selected", "revoked", "late"], ids=["on-time", "C02-ACT-0005", "after-restart"])
async def test_fire_preserves_notice_and_selection_barriers(graph, mode):
    assert_outcome(await graph.turn(R4), "completed", "committed")
    row, = await graph.rows()
    assert await graph.effects() == effects(reminder_rows_delta=1, spoken_utterances=2)
    if mode == "revoked":
        graph.set_selection(("light.cocina",))
    if mode == "late":
        await graph.store.close()
        await graph.store.initialize()
        graph.scheduler = ReminderScheduler(graph.store, LocalVoiceDelivery(graph.handler), None, graph.ha,
                                            graph.config, home_actions=graph.actions, clock=lambda: graph.now[0])
    graph.now[0] = row.trigger_at + (DEFAULT_LATE_ACTION_GRACE_S + 1 if mode == "late" else 1)
    await graph.scheduler._deliver_due(graph.now[0])
    await graph.scheduler._deliver_due(graph.now[0])
    current, = await graph.rows()
    assert current.state is ReminderState.FIRED and current.fire_count == 1
    assert await ledger_rows(graph.store, "notice:") == [("confirmed", None, 1)]
    receipt = {"selected": ("confirmed", None), "revoked": ("failed", "action_not_selected"),
               "late": ("failed", "late_not_executed")}[mode]
    assert await ledger_rows(graph.store, "reminder:") == [(*receipt, None)]
    assert await graph.effects(previous=row, spoken_before=2) == effects(
        ha_service_calls=int(mode == "selected"), reminder_transitions=1, ledger_claims=2, spoken_utterances=1)
    assert ("atraso" in graph.spoken()[-1]) is (mode == "late")


@pytest.mark.parametrize(
    "text",
    ["agregar leche a la lista", "poner música", "sincronizar los dispositivos", "registrar una persona"],
    ids=["C02-ACT-0006", "musica", "sync", "enrolamiento"],
)
async def test_non_home_complements_remain_text(graph, text):
    result = await graph.turn(f"recordame {text} a las 8 de la noche")
    assert_outcome(result, "completed", "committed")
    row, = await graph.rows()
    assert row.text == text and row.ha_actions is None
    assert await graph.effects() == effects(reminder_rows_delta=1, spoken_utterances=2)
    graph.now[0] = row.trigger_at + 1
    await graph.scheduler._deliver_due(graph.now[0])
    assert await graph.effects(previous=row, spoken_before=2) == effects(
        reminder_transitions=1, ledger_claims=1, spoken_utterances=1)


async def test_c02_act_0007_weak_wake_without_8101(graph):
    result = await graph.turn(R4, textual_wake=True)
    assert result["intent"] == "llm_rejected:household" and result["path"] is None
    assert result["outcome"]["handled"] is False and result["task_completed"] is False
    assert await graph.effects() == effects(llm_8101_calls=1)


async def test_c02_def_0002_disabled(graph):
    graph.config["voice_actions"] = False
    result = await graph.turn(R4)
    assert_outcome(result, "unsupported", "not_written")
    assert "Todavía no programo acciones" in graph.spoken()[0]
    assert await graph.effects() == effects(spoken_utterances=1)
    assert_outcome(await graph.turn(R3), "completed", "committed")
    row, = await graph.rows()
    assert row.ha_actions is None
    assert await graph.effects() == effects(reminder_rows_delta=1, spoken_utterances=2)


@pytest.mark.parametrize("phrase", [
    "recordame apagar la luz del living y prender la luz de la cocina a las 11 de la noche",
    "apagá la luz del living y prendé la luz de la cocina en 10 minutos",
])
async def test_two_admitted_actions_keep_the_spoken_order(graph, phrase):
    assert_outcome(await graph.turn(phrase), "completed", "committed")
    row, = await graph.rows()
    assert row.ha_actions == [LIGHT_OFF, {
        "domain": "light", "service": "turn_on", "entity_id": "light.cocina", "data": None,
    }]
    assert await graph.effects() == effects(reminder_rows_delta=1, spoken_utterances=2)


async def test_room_in_action_qualifies_the_device(graph):
    result = await graph.turn("recordame apagar la luz en la cocina a las 11 de la noche")
    assert_outcome(result, "completed", "committed")
    row, = await graph.rows()
    assert row.ha_actions == [{**LIGHT_OFF, "entity_id": "light.cocina"}]
    assert "cocina" in graph.spoken()[0]


async def test_compound_time_between_actions_needs_clarification(graph):
    result = await graph.turn("apagá la luz en 10 minutos y prendé la luz de la cocina")
    assert_outcome(result, "needs_input", "not_written")
    assert await graph.effects() == effects(spoken_utterances=1)


async def test_compound_limit_keeps_r3_text_and_refuses_r4(graph):
    text = "apagar la luz y prender la luz y apagar la luz y prender la luz"
    assert_outcome(await graph.turn(f"recordame {text} en 10 minutos"), "completed", "committed")
    assert_outcome(await graph.turn(f"{text} en 10 minutos"), "unsupported", "not_written")
    row, = await graph.rows()
    assert row.text == text and row.ha_actions is None
    assert await graph.effects() == effects(reminder_rows_delta=1, spoken_utterances=3)


async def test_reviewed_limits_are_checked_before_storing_brightness(graph):
    selection = json.loads(graph.selection.read_text())
    grant = selection["allowed_entities"][0]
    grant["operations"].append("brightness")
    grant["parameters"] = {"brightness_pct": {"min": 20, "max": 60}}
    graph.selection.write_text(json.dumps(selection))
    result = await graph.turn("recordame poner la luz al 30% a las 8 de la noche")
    assert_outcome(result, "completed", "committed")
    result = await graph.turn("recordame poner la luz al 70% a las 8 de la noche")
    assert_outcome(result, "completed", "committed")
    admitted, fallback = await graph.rows()
    assert admitted.ha_actions == [{
        **LIGHT_OFF, "service": "turn_on", "data": {"brightness_pct": 30},
    }]
    assert fallback.text == "poner la luz al 70%" and fallback.ha_actions is None
    assert "No puedo ejecutar por voz: poner la luz al 70%" in graph.spoken()[-2]
    assert graph.ha.reads == graph.ha.sends == []


async def test_ambient_text_cannot_create_even_with_a_positive_veto(graph):
    graph.veto.classify.return_value = CommandClassification(is_command=True, confidence=.99, intent="turn_off")
    result = await graph.turn(R4, ambient_strict=True)
    assert result["intent"] == "gate_rejected:ambient_reminder"
    assert result["task_completed"] is False
    assert await graph.effects() == effects()
