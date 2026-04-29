"""Public decorator API for plugin hooks.

Plan #3 OpenClaw.
"""

from collections.abc import Callable

from src.hooks.registry import _global_registry


def before_ha_action(priority: int = 100) -> Callable:
    """Register a SYNC handler invoked before each HA action dispatch.

    Handler signature:
        def handler(call: HaActionCall) -> BlockResult | RewriteResult | None

    Lower `priority` runs first. Handlers are expected to complete in <5ms
    (threshold logged as WARNING but not enforced).
    """
    def deco(fn: Callable) -> Callable:
        _global_registry.register_before("before_ha_action", fn, priority)
        return fn
    return deco


def before_tts_speak(priority: int = 100) -> Callable:
    """Register a SYNC handler invoked before each TTS speak.

    Handler signature:
        def handler(call: TtsCall) -> BlockResult | RewriteResult | None
    """
    def deco(fn: Callable) -> Callable:
        _global_registry.register_before("before_tts_speak", fn, priority)
        return fn
    return deco


def after_event(*event_names: str) -> Callable:
    """Register a sync OR async handler for one or more after-events.

    Handler signature:
        def handler(payload) -> None
        # OR
        async def handler(payload) -> None

    Valid event_names: see EVENT_NAMES in src.hooks.types.
    Multiple decorators on the same fn (or repeated names) re-register it.
    """
    def deco(fn: Callable) -> Callable:
        for name in event_names:
            _global_registry.register_after(name, fn)
        return fn
    return deco
