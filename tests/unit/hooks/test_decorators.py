"""Tests for the 3 public decorators."""

import pytest

from src.hooks import (
    before_ha_action, before_tts_speak, after_event,
    BlockResult, HookRegistry,
)
from src.hooks.registry import _global_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    """Clear the global registry between tests."""
    _global_registry.clear()
    yield
    _global_registry.clear()


def test_before_ha_action_registers_with_priority():
    @before_ha_action(priority=7)
    def my_handler(call):
        return None

    handlers = _global_registry.get_before_handlers("before_ha_action")
    assert len(handlers) == 1
    assert handlers[0] == (7, my_handler)


def test_before_tts_speak_registers():
    @before_tts_speak(priority=15)
    def my_handler(call):
        return None

    handlers = _global_registry.get_before_handlers("before_tts_speak")
    assert handlers[0][0] == 15


def test_after_event_registers_for_multiple_events():
    @after_event("stt", "intent")
    def my_handler(payload):
        pass

    assert my_handler in _global_registry.get_after_handlers("stt")
    assert my_handler in _global_registry.get_after_handlers("intent")


def test_after_event_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown event"):
        @after_event("bogus")
        def h(p): pass


def test_decorators_return_original_function():
    """Decorator should return fn unchanged so callers can still call it directly."""
    @before_ha_action(priority=10)
    def h(call):
        return "ok"

    assert h.__name__ == "h"
