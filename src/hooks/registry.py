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
        # Ring buffer of recent handler errors (most recent last). Bounded.
        self._handler_recent_errors: list[str] = []
        self._handler_recent_errors_max: int = 10
        # Existing fields kept for backward compat:
        self._handler_failures: int = 0
        self._handler_last_error: str | None = None  # alias for ring[-1]; kept for plan-#2 parity
        self._slow_handler_count: int = 0
        self._after_tasks: set = set()  # populated by runner.execute_after_event

    def _record_failure(self, error_str: str) -> None:
        """Internal: bump failure counter + append to bounded ring buffer."""
        self._handler_failures += 1
        self._handler_last_error = error_str
        self._handler_recent_errors.append(error_str)
        if len(self._handler_recent_errors) > self._handler_recent_errors_max:
            self._handler_recent_errors = self._handler_recent_errors[-self._handler_recent_errors_max:]

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
        self._handler_recent_errors = []
        self._slow_handler_count = 0
        # Don't clear _after_tasks — those are in-flight asyncio tasks.

    def get_stats(self) -> dict:
        return {
            "handler_failures": self._handler_failures,
            "handler_last_error": self._handler_last_error,
            "handler_recent_errors": list(self._handler_recent_errors),
            "slow_handler_count": self._slow_handler_count,
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
