"""Tests for hook types and event payloads."""

import pytest
from dataclasses import FrozenInstanceError

from src.hooks.types import (
    HaActionCall, TtsCall, BlockResult, RewriteResult,
    WakePayload, SttPayload, IntentPayload,
    HaActionDispatchedPayload, HaActionBlockedPayload,
    LlmCallPayload, TtsPayload,
    EVENT_NAMES,
)


class TestHaActionCall:
    def test_required_fields(self):
        call = HaActionCall(
            entity_id="light.escritorio",
            domain="light",
            service="turn_on",
            service_data={"brightness_pct": 50},
            user_id="juan",
            user_name="Juan",
            zone_id="zone_escritorio",
            timestamp=1700000000.0,
        )
        assert call.domain == "light"
        assert call.service_data["brightness_pct"] == 50

    def test_immutability(self):
        call = HaActionCall(
            entity_id="light.x", domain="light", service="turn_on",
            service_data={}, user_id=None, user_name=None,
            zone_id=None, timestamp=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            call.service = "turn_off"

    def test_with_data_returns_modified_copy(self):
        call = HaActionCall(
            entity_id="light.x", domain="light", service="turn_on",
            service_data={"brightness_pct": 50}, user_id=None, user_name=None,
            zone_id=None, timestamp=0.0,
        )
        modified = call.with_data(brightness_pct=20)
        assert modified.service_data == {"brightness_pct": 20}
        assert call.service_data == {"brightness_pct": 50}  # original unchanged
        assert modified is not call


class TestTtsCall:
    def test_basic(self):
        c = TtsCall(text="hola", voice=None, lang="es", user_id=None, zone_id=None)
        assert c.text == "hola"

    def test_immutability(self):
        c = TtsCall(text="hola", voice=None, lang="es", user_id=None, zone_id=None)
        with pytest.raises(FrozenInstanceError):
            c.text = "chau"


class TestResults:
    def test_block_result(self):
        b = BlockResult(reason="no se puede", rule_name="protege_alarma")
        assert b.reason == "no se puede"
        assert b.rule_name == "protege_alarma"

    def test_rewrite_result(self):
        call = TtsCall(text="x", voice=None, lang="es", user_id=None, zone_id=None)
        r = RewriteResult(modified=call, rule_name="rule")
        assert r.modified is call


class TestEventPayloads:
    def test_wake_payload(self):
        p = WakePayload(timestamp=0.0, confidence=0.95, zone_id="z")
        assert p.confidence == 0.95

    def test_stt_payload(self):
        p = SttPayload(timestamp=0.0, text="hola", latency_ms=120.0,
                       user_id=None, zone_id=None, success=True)
        assert p.success

    def test_event_names_is_closed_tuple(self):
        # Lista cerrada — type-safe, autocompletable
        assert isinstance(EVENT_NAMES, tuple)
        assert "wake" in EVENT_NAMES
        assert "stt" in EVENT_NAMES
        assert "intent" in EVENT_NAMES
        assert "ha_action_dispatched" in EVENT_NAMES
        assert "ha_action_blocked" in EVENT_NAMES
        assert "llm_call" in EVENT_NAMES
        assert "tts" in EVENT_NAMES
