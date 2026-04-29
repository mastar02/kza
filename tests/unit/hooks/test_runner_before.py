"""Tests for execute_before_chain — block + rewrite + priority + errors."""

import logging
import pytest
from dataclasses import replace

from src.hooks.registry import HookRegistry
from src.hooks.runner import execute_before_chain
from src.hooks.types import HaActionCall, TtsCall, BlockResult, RewriteResult


def _ha_call(**overrides) -> HaActionCall:
    base = dict(
        entity_id="light.x", domain="light", service="turn_on",
        service_data={}, user_id=None, user_name=None,
        zone_id=None, timestamp=0.0,
    )
    base.update(overrides)
    return HaActionCall(**base)


class TestBlockShortCircuit:
    def test_block_stops_chain(self):
        reg = HookRegistry()
        called = []

        def h1(call):
            called.append("h1")
            return BlockResult(reason="nope", rule_name="r1")

        def h2(call):
            called.append("h2")
            return None

        reg.register_before("before_ha_action", h1, priority=10)
        reg.register_before("before_ha_action", h2, priority=20)

        result = execute_before_chain(reg, "before_ha_action", _ha_call())
        assert isinstance(result, BlockResult)
        assert result.reason == "nope"
        assert called == ["h1"]  # h2 NOT called

    def test_no_handlers_returns_call_unchanged(self):
        reg = HookRegistry()
        call = _ha_call()
        result = execute_before_chain(reg, "before_ha_action", call)
        assert result is call


class TestRewriteChain:
    def test_rewrite_passes_modified_to_next_handler(self):
        reg = HookRegistry()
        seen_data = []

        def h1(call):
            return RewriteResult(
                modified=call.with_data(brightness_pct=50),
                rule_name="cap1",
            )

        def h2(call):
            seen_data.append(call.service_data.get("brightness_pct"))
            return RewriteResult(
                modified=call.with_data(brightness_pct=20),
                rule_name="cap2",
            )

        reg.register_before("before_ha_action", h1, priority=10)
        reg.register_before("before_ha_action", h2, priority=20)

        result = execute_before_chain(reg, "before_ha_action", _ha_call())
        assert isinstance(result, HaActionCall)
        assert result.service_data["brightness_pct"] == 20
        # h2 saw the rewritten value from h1
        assert seen_data == [50]

    def test_rewrite_then_block(self):
        reg = HookRegistry()

        def h1(call):
            return RewriteResult(
                modified=call.with_data(brightness_pct=50),
                rule_name="cap1",
            )

        def h2(call):
            if call.service_data.get("brightness_pct") == 50:
                return BlockResult(reason="too bright", rule_name="block1")

        reg.register_before("before_ha_action", h1, priority=10)
        reg.register_before("before_ha_action", h2, priority=20)

        result = execute_before_chain(reg, "before_ha_action", _ha_call())
        assert isinstance(result, BlockResult)


class TestPriorityOrder:
    def test_priority_5_runs_before_priority_10(self):
        reg = HookRegistry()
        order = []

        def lo_prio(call):
            order.append("lo")
            return None

        def hi_prio(call):
            order.append("hi")
            return None

        reg.register_before("before_ha_action", lo_prio, priority=10)
        reg.register_before("before_ha_action", hi_prio, priority=5)
        execute_before_chain(reg, "before_ha_action", _ha_call())
        assert order == ["hi", "lo"]


class TestErrorSwallow:
    def test_handler_exception_logged_and_chain_continues(self, caplog):
        reg = HookRegistry()

        def h1(call):
            raise RuntimeError("kaboom")

        def h2(call):
            return RewriteResult(
                modified=call.with_data(brightness_pct=99),
                rule_name="rescue",
            )

        reg.register_before("before_ha_action", h1, priority=10)
        reg.register_before("before_ha_action", h2, priority=20)

        with caplog.at_level(logging.WARNING):
            result = execute_before_chain(reg, "before_ha_action", _ha_call())

        # Chain continued; h2 ran on original call
        assert isinstance(result, HaActionCall)
        assert result.service_data["brightness_pct"] == 99
        # Counter incremented
        stats = reg.get_stats()
        assert stats["handler_failures"] == 1
        assert "kaboom" in (stats["handler_last_error"] or "")
        # Logged
        assert any("handler error" in rec.message.lower() for rec in caplog.records)


class TestWrongRewriteType:
    def test_rewrite_with_wrong_modified_type_logged_and_skipped(self, caplog):
        reg = HookRegistry()

        def h1(call):  # call is HaActionCall, returns RewriteResult with TtsCall — wrong
            return RewriteResult(
                modified=TtsCall(text="oops", voice=None, lang="es",
                                 user_id=None, zone_id=None),
                rule_name="bad",
            )

        reg.register_before("before_ha_action", h1, priority=10)

        with caplog.at_level(logging.ERROR):
            result = execute_before_chain(reg, "before_ha_action", _ha_call())

        # Result is the ORIGINAL call (rewrite ignored)
        assert isinstance(result, HaActionCall)
        assert result.entity_id == "light.x"
        assert any("rewrite type mismatch" in rec.message.lower() for rec in caplog.records)


class TestTimingWarning:
    def test_slow_handler_logs_warning(self, caplog):
        import time

        reg = HookRegistry()

        def slow(call):
            time.sleep(0.01)  # 10ms — well above 5ms threshold
            return None

        reg.register_before("before_ha_action", slow, priority=10)

        with caplog.at_level(logging.WARNING):
            execute_before_chain(reg, "before_ha_action", _ha_call(), warn_ms=5.0)

        assert any("slow handler" in rec.message.lower() for rec in caplog.records)
