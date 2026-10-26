"""Spoken text through both real routers, selection, hooks and HA confirmation."""

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from tests.mocks.responses import response_handler_mock
from src.home_assistant.control_policy import EntityGrant, HomeControlPolicy
from src.home_assistant.voice_resolver import HomeVoiceResolver
from src.hooks import HookRegistry, RewriteResult
from src.nlu.command_gate import CommandAcceptanceGate
from src.nlu.llm_router import CommandClassification
from src.orchestrator.dispatcher import MultiUserOrchestrator
from src.pipeline.command_event import CommandEvent
from src.pipeline.command_processor import ProcessedCommand
from src.pipeline.request_router import RequestRouter
from src.rooms.room_context import RoomConfig
from tests.integration import test_typed_home_operations as typed_fixtures


@pytest.fixture(params=[False, True], ids=["legacy", "orchestrated"])
def voice_graph(request):
    ha, actions = typed_fixtures.typed_graph.__wrapped__()
    rooms = {"test_room": RoomConfig("test_room", "Test room", "el salón", aliases=["salón"],
        default_light="light.test_lamp", default_climate="climate.test_ac", default_cover="cover.test_blind")}
    processor = MagicMock()
    processor.process_command = AsyncMock()
    chroma = MagicMock()
    # Real orchestrator: typed controls must be handled before vector/LLM paths.
    orchestrator = MultiUserOrchestrator(chroma, ha, MagicMock(), home_actions=actions)
    manager = MagicMock()
    manager.check_entity_permission.side_effect = AssertionError("Voice profiles are not authorization")
    router = RequestRouter(processor, response_handler_mock(), MagicMock(), orchestrator=orchestrator,
        orchestrator_enabled=request.param, chroma_sync=chroma, ha_client=ha, home_actions=actions,
        home_voice_resolver=HomeVoiceResolver(actions, rooms), user_manager=manager,
        command_gate=CommandAcceptanceGate(), wake_acoustically_confirmed=True)
    return router, ha, actions, processor, chroma


async def speak(graph, text, *, room_id="test_room", **metadata):
    router, _, _, processor, _ = graph
    processor.process_command.return_value = ProcessedCommand(text=text, user=None, emotion=None, timings={}, success=True)
    return await router.process_command(CommandEvent(np.zeros(16000, dtype=np.float32), room_id, capture_complete=True, **metadata))


@pytest.mark.parametrize("text,domain,service,entity,data", [
    ("Nexa poné el aire del salón a veintidós coma cinco grados", "climate", "set_temperature", "climate.test_ac", {"temperature": 22.5}),
    ("Poné el aire a 7 grados", "climate", "set_temperature", "climate.test_ac", {"temperature": 7}),
    ("Poné el aire a 35 °C", "climate", "set_temperature", "climate.test_ac", {"temperature": 35}),
    *[(f"Poné el aire en {word}", "climate", "set_hvac_mode", "climate.test_ac", {"hvac_mode": mode})
      for word, mode in [("apagado", "off"), ("frío", "cool"), ("seco", "dry"), ("ventilación", "fan_only"), ("automático", "auto"), ("calor", "heat")]],
    ("Abrí la persiana", "cover", "open_cover", "cover.test_blind", None),
    ("Cerrá la cortina del salón", "cover", "close_cover", "cover.test_blind", None),
    ("Detené la persiana", "cover", "stop_cover", "cover.test_blind", None),
    ("Bajá la persiana al treinta por ciento", "cover", "set_cover_position", "cover.test_blind", {"position": 30}),
    ("Poné la luz del salón cálida al treinta por ciento", "light", "turn_on", "light.test_lamp", {"color_temp_kelvin": 2700, "brightness_pct": 30}),
])
async def test_spoken_extended_effect_is_single_exact_and_confirmed(voice_graph, text, domain, service, entity, data):
    router, ha, _, _, chroma = voice_graph
    result = await speak(voice_graph, text)
    assert result["success"] is True
    assert result["receipt"]["status"] == "confirmed"
    assert ha.sends == [(domain, service, entity, data)]
    chroma.search_command.assert_not_called()
    chroma.asearch_command.assert_not_called()
    router.user_manager.check_entity_permission.assert_not_called()


@pytest.mark.parametrize("text", [
    "Prendé el aire", "Poné el aire a dos tres grados", "Poné el aire a 70°F",
    "Poné el aire a 22 grados y apagá la luz", "Poné el aire a -20 grados",
    "Poné la persiana al -10%", "Bajá la persiana al 101%", "Poné la luz cálida al 0%",
    "Bajá la persiana un poco", "Poné el aire del balcón a 22 grados",
    "Poné el aire a 36 grados", "Poné el aire a 24,2 grados", "Poné la luz a 9000 kelvin",
    "No pongas el aire a 22 grados", "¿Apagaste el aire?", "Ayer puse el aire a 22 grados",
    "Poné el aire a 22 grados mañana", "Si hace calor poné el aire a 22 grados",
    "Me dijo que apague el aire", "Abrí la persiana después",
    "Dejé el aire a 22 grados", "Configuré el aire a 22 grados", "Ajusté el aire a 22 grados",
    "Activé el aire", "Desactivé el aire", "Paré la persiana",
])
async def test_unresolved_invalid_or_non_immediate_has_no_fallback_or_send(voice_graph, text):
    _, ha, _, _, chroma = voice_graph
    result = await speak(voice_graph, text)
    assert result["success"] is False
    assert ha.sends == []
    chroma.search_command.assert_not_called()
    chroma.asearch_command.assert_not_called()


@pytest.mark.parametrize("case", ["missing_default", "unselected_default", "ambiguous_alias", "unknown_mic", "ambiguous_global", "no_selection"])
async def test_target_evidence_never_substitutes_a_different_entity(voice_graph, case):
    router, ha, actions, _, _ = voice_graph
    room = router.home_voice_resolver.rooms["test_room"]
    text, room_id = "Poné el aire a 23 grados", "test_room"
    if case == "missing_default":
        room.default_climate = None
    elif case == "unselected_default":
        room.default_climate = "climate.not_selected"
    elif case == "ambiguous_alias":
        router.home_voice_resolver.rooms["second"] = RoomConfig("second", "second", "segundo", aliases=["salón"], default_climate=room.default_climate)
        text = "Poné el aire del salón a 23 grados"
    elif case == "unknown_mic":
        room_id = "missing"
    else:
        grants = await actions.policy.snapshot_grants()
        if case == "ambiguous_global":
            actions.policy = HomeControlPolicy((*grants, EntityGrant("climate.other", frozenset({"set_temperature"}), temperature_min=7, temperature_max=35)))
            room_id = None
        else:
            actions.policy = HomeControlPolicy()
    result = await speak(voice_graph, text, room_id=room_id)
    assert result["intent"] == "clarification_requested"
    assert ha.sends == []


async def test_no_room_evidence_may_use_one_selected_domain_target(voice_graph):
    result = await speak(voice_graph, "Poné el aire a 23 grados", room_id=None)
    assert result["success"] is True
    assert voice_graph[1].sends == [("climate", "set_temperature", "climate.test_ac", {"temperature": 23})]


@pytest.mark.parametrize("gate_result", ["missing", "error", "unavailable", "rejected", "low", "nan", "invented_slots"])
@pytest.mark.parametrize("channel", ["textual", "strict", "weak_wake"])
async def test_non_acoustic_or_weak_channels_keep_additional_gate(voice_graph, gate_result, channel):
    router, ha, _, _, chroma = voice_graph
    classification = CommandClassification(True, confidence=0.95, intent="set_temperature", entity_hint="lock",
        slots={"temperature": 35, "entity_id": "lock.hidden"})
    if gate_result != "missing":
        router.llm_command_router = MagicMock()
        router.llm_command_router.classify = AsyncMock(return_value=classification)
    if gate_result == "error":
        router.llm_command_router.classify.side_effect = TimeoutError()
    elif gate_result == "unavailable":
        classification.rejection_reason = "unavailable"
    elif gate_result == "rejected":
        classification.is_command = False
    elif gate_result == "low":
        classification.confidence = 0.2
    elif gate_result == "nan":
        classification.confidence = float("nan")
    metadata = {"textual_wake": True} if channel == "textual" else {"ambient_strict": True} if channel == "strict" else {"wake_score": 0.4}
    result = await speak(voice_graph, "Nexa poné el aire a 23 grados", **metadata)
    assert result["success"] is (gate_result == "invented_slots")
    assert ha.sends == ([("climate", "set_temperature", "climate.test_ac", {"temperature": 23})] if gate_result == "invented_slots" else [])
    chroma.search_command.assert_not_called()
    chroma.asearch_command.assert_not_called()


@pytest.mark.parametrize("case", ["revoked_after_resolve", "revoked_after_read", "hook_target", "hook_hidden_parameter", "no_effect", "unit_unknown"])
async def test_final_action_boundary_remains_mandatory(voice_graph, case):
    router, ha, actions, _, _ = voice_graph
    if case == "revoked_after_resolve":
        original = router.home_voice_resolver.resolve

        async def revoke(*args, **kwargs):
            target = await original(*args, **kwargs)
            actions.policy = HomeControlPolicy()
            return target

        router.home_voice_resolver.resolve = revoke
    elif case == "revoked_after_read":
        original = ha.get_entity_state_fresh

        async def revoke(*args, **kwargs):
            state = await original(*args, **kwargs)
            actions.policy = HomeControlPolicy()
            return state

        ha.get_entity_state_fresh = revoke
    elif case.startswith("hook_"):
        hooks = HookRegistry()

        def rewrite(call):
            changed = replace(call, entity_id="lock.hidden", domain="lock", service="unlock") if case == "hook_target" else call.with_data(target={"entity_id": "lock.hidden"})
            return RewriteResult(modified=changed, rule_name="test")

        hooks.register_before("before_ha_action", rewrite, priority=10)
        router._voice_actions.hooks = hooks
    elif case == "no_effect":
        ha.observe_effect = False
    else:
        ha.unit = None
    result = await speak(voice_graph, "Poné el aire a 23 grados")
    assert result["success"] is False
    assert result["receipt"]["status"] == ("uncertain" if case == "no_effect" else "failed")
    assert len(ha.sends) == int(case == "no_effect")


async def test_incomplete_capture_does_not_run_stt_or_effect(voice_graph):
    router, ha, _, processor, _ = voice_graph
    result = await router.process_command(CommandEvent(np.zeros(16000, dtype=np.float32), "test_room", capture_complete=False))
    assert result["intent"] == "gate_rejected:incomplete_capture"
    processor.process_command.assert_not_called()
    assert ha.sends == []
