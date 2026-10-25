"""Past actions and questions cannot gain authority from an acoustic wake."""

import pytest

from src.nlu.command_gate import CommandAcceptanceGate
from src.nlu.command_grammar import llm_intent_evidenced, parse_command, parse_default_light_target
from tests.command_corpus import IMMEDIATE_ACTIONS, REJECTED_ACTIONS


@pytest.mark.parametrize("text,category", REJECTED_ACTIONS)
def test_non_immediate_corpus_cannot_produce_a_dispatchable_action(text, category):
    decision = CommandAcceptanceGate().evaluate(text)
    assert decision.accept is False, (category, text)
    assert decision.reason.startswith("non_immediate:"), (category, decision)
    assert not parse_command(text).ready_to_dispatch()
    assert parse_default_light_target(text) is None
    for intent in ("turn_on", "turn_off", "set_temperature", "open", "close"):
        assert not llm_intent_evidenced(intent, text), (text, intent)


@pytest.mark.parametrize("text,intent", IMMEDIATE_ACTIONS)
def test_immediate_requests_remain_dispatchable(text, intent):
    decision = CommandAcceptanceGate().evaluate(text)
    assert decision.accept, decision
    command = parse_command(text)
    assert command.ready_to_dispatch(), command
    assert command.intent == intent


@pytest.mark.parametrize("text,requested,opposite", [
    ("Apagá la luz que está prendida", "turn_off", "turn_on"),
    ("Prendé la luz que está apagada", "turn_on", "turn_off"),
])
def test_described_state_cannot_evidence_an_opposite_llm_action(text, requested, opposite):
    assert llm_intent_evidenced(requested, text)
    assert not llm_intent_evidenced(opposite, text)
