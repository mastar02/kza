"""HookRegistry — central store for before/after hook handlers.

Plan #3 OpenClaw.
"""

import logging
from collections.abc import Callable

from src.hooks.types import EVENT_NAMES

logger = logging.getLogger(__name__)


VALID_BEFORE_HOOKS: tuple[str, ...] = ("before_ha_action", "before_tts_speak")


class HookRegistry:
    """Register and look up plugin hook handlers.

    Single global instance lives in `src.hooks.registry._global_registry`,
    populated by decorators in `src.hooks.decorators` at import-time of
    `src.policies` modules.
    """

    def __init__(self):
        self._before: dict[str, list[tuple[int, Callable]]] = {
            name: [] for name in VALID_BEFORE_HOOKS
        }
        self._after: dict[str, list[Callable]] = {name: [] for name in EVENT_NAMES}

        # Observability counters (consumed by get_stats + ContextManager-style logging)
        self._handler_failures: int = 0
        self._handler_last_error: str | None = None
        self._after_tasks: set = set()  # populated by runner.execute_after_event

    def register_before(self, hook_name: str, fn: Callable, priority: int) -> None:
        if hook_name not in VALID_BEFORE_HOOKS:
            raise ValueError(
                f"unknown hook {hook_name!r}; valid: {VALID_BEFORE_HOOKS}"
            )
        bucket = self._before[hook_name]
        bucket.append((priority, fn))
        bucket.sort(key=lambda item: item[0])  # asc by priority

    def register_after(self, event_name: str, fn: Callable) -> None:
        if event_name not in EVENT_NAMES:
            raise ValueError(
                f"unknown event {event_name!r}; valid: {EVENT_NAMES}"
            )
        self._after[event_name].append(fn)

    def get_before_handlers(self, hook_name: str) -> list[tuple[int, Callable]]:
        return list(self._before.get(hook_name, []))

    def get_after_handlers(self, event_name: str) -> list[Callable]:
        return list(self._after.get(event_name, []))

    def clear(self) -> None:
        """Reset all registrations (used by tests for isolation)."""
        self._before = {name: [] for name in VALID_BEFORE_HOOKS}
        self._after = {name: [] for name in EVENT_NAMES}
        self._handler_failures = 0
        self._handler_last_error = None
        # Don't clear _after_tasks — those are in-flight asyncio tasks.

    def get_stats(self) -> dict:
        return {
            "handler_failures": self._handler_failures,
            "handler_last_error": self._handler_last_error,
            "after_tasks_in_flight": len(self._after_tasks),
            "before_handler_count": {
                name: len(self._before[name]) for name in VALID_BEFORE_HOOKS
            },
            "after_handler_count": {
                name: len(self._after[name]) for name in EVENT_NAMES
            },
        }


# Module-level singleton — decorators register here at import-time.
_global_registry = HookRegistry()
