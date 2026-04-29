"""Tests for HookRegistry register/list/clear/stats."""

import pytest

from src.hooks.registry import HookRegistry


class TestRegister:
    def test_register_before_orders_by_priority(self):
        reg = HookRegistry()
        def h1(call): return None
        def h2(call): return None
        def h3(call): return None

        reg.register_before("before_ha_action", h2, priority=20)
        reg.register_before("before_ha_action", h1, priority=5)
        reg.register_before("before_ha_action", h3, priority=10)

        handlers = reg.get_before_handlers("before_ha_action")
        # Priority asc: h1 (5), h3 (10), h2 (20)
        assert [fn for _prio, fn in handlers] == [h1, h3, h2]

    def test_register_before_unknown_hook_raises(self):
        reg = HookRegistry()
        with pytest.raises(ValueError, match="unknown hook"):
            reg.register_before("before_bogus", lambda c: None, priority=10)

    def test_register_after_unknown_event_raises(self):
        reg = HookRegistry()
        with pytest.raises(ValueError, match="unknown event"):
            reg.register_after("bogus_event", lambda p: None)

    def test_register_after_appends(self):
        reg = HookRegistry()
        def h1(p): pass
        def h2(p): pass
        reg.register_after("stt", h1)
        reg.register_after("stt", h2)
        assert reg.get_after_handlers("stt") == [h1, h2]


class TestStatsAndClear:
    def test_stats_initial(self):
        reg = HookRegistry()
        stats = reg.get_stats()
        assert stats["handler_failures"] == 0
        assert stats["handler_last_error"] is None
        assert stats["after_tasks_in_flight"] == 0
        assert stats["before_handler_count"] == {
            "before_ha_action": 0, "before_tts_speak": 0,
        }

    def test_stats_counts_handlers(self):
        reg = HookRegistry()
        reg.register_before("before_ha_action", lambda c: None, priority=10)
        reg.register_before("before_tts_speak", lambda c: None, priority=10)
        reg.register_after("stt", lambda p: None)

        stats = reg.get_stats()
        assert stats["before_handler_count"]["before_ha_action"] == 1
        assert stats["before_handler_count"]["before_tts_speak"] == 1
        assert stats["after_handler_count"]["stt"] == 1

    def test_clear_removes_all_handlers(self):
        reg = HookRegistry()
        reg.register_before("before_ha_action", lambda c: None, priority=10)
        reg.register_after("stt", lambda p: None)
        reg.clear()
        assert reg.get_before_handlers("before_ha_action") == []
        assert reg.get_after_handlers("stt") == []
