"""Hook runners — execute_before_chain + execute_after_event.

Plan #3 OpenClaw.
"""

import asyncio
import inspect
import logging
import time
from collections.abc import Callable

from src.hooks.registry import HookRegistry
from src.hooks.types import (
    BlockResult, RewriteResult, HaActionCall, TtsCall,
)

logger = logging.getLogger(__name__)


def execute_before_chain(
    registry: HookRegistry,
    hook_name: str,
    call: HaActionCall | TtsCall,
    warn_ms: float = 5.0,
) -> BlockResult | HaActionCall | TtsCall:
    """Run handlers for `hook_name` in priority order.

    Returns:
      - BlockResult: a handler chose to block; chain stopped, caller must NOT execute
      - HaActionCall | TtsCall (same type as input): pass-through or rewritten;
        caller proceeds with this (possibly modified) call

    Errors in handlers are logged + counted, never propagated. Slow handlers
    (> warn_ms) emit a WARNING but do not abort.
    """
    expected_type = type(call)
    handlers = registry.get_before_handlers(hook_name)
    current = call

    for priority, fn in handlers:
        t0 = time.perf_counter()
        try:
            result = fn(current)
        except Exception as e:
            registry._handler_failures += 1
            registry._handler_last_error = f"{type(e).__name__}: {e}"
            logger.warning(
                f"[HookRunner] handler error in {hook_name} ({fn.__name__}): {e}"
            )
            continue
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > warn_ms:
                logger.warning(
                    f"[HookRunner] slow handler {fn.__name__} for {hook_name}: "
                    f"{elapsed_ms:.1f}ms (threshold {warn_ms}ms)"
                )

        if result is None:
            continue
        if isinstance(result, BlockResult):
            return result
        if isinstance(result, RewriteResult):
            if not isinstance(result.modified, expected_type):
                logger.error(
                    f"[HookRunner] rewrite type mismatch in {hook_name} "
                    f"({fn.__name__}): expected {expected_type.__name__}, "
                    f"got {type(result.modified).__name__}; ignoring rewrite"
                )
                continue
            current = result.modified
            continue
        # Unknown return type — treat as None (pass-through) with warning
        logger.warning(
            f"[HookRunner] unexpected return type from {fn.__name__} in "
            f"{hook_name}: {type(result).__name__}; ignoring"
        )

    return current


def execute_after_event(
    registry: HookRegistry,
    event_name: str,
    payload,
) -> None:
    """Fire-and-forget invocation of all `after_event(name)` handlers.

    Sync handlers run inline (must be fast). Async handlers are scheduled
    as `asyncio.create_task` with a strong ref in `registry._after_tasks`
    to prevent GC mid-flight (Python 3.11+ weak-ref task collection).

    Errors are logged + counted, never propagated.
    If no event loop is running and an async handler is registered, the
    handler is skipped with a warning (sync callers can still use this
    function without setting up a loop).
    """
    handlers = registry.get_after_handlers(event_name)
    if not handlers:
        return

    loop = None
    for fn in handlers:
        is_coro = inspect.iscoroutinefunction(fn)
        if is_coro:
            if loop is None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    logger.warning(
                        f"[HookRunner] no event loop running; skipping async "
                        f"after-event handler {fn.__name__} for {event_name}"
                    )
                    continue
            task = loop.create_task(fn(payload))
            registry._after_tasks.add(task)
            task.add_done_callback(_make_after_done_callback(registry, fn, event_name))
        else:
            try:
                fn(payload)
            except Exception as e:
                registry._handler_failures += 1
                registry._handler_last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    f"[HookRunner] after-event handler error in {event_name} "
                    f"({fn.__name__}): {e}"
                )


def _make_after_done_callback(registry: HookRegistry, fn: Callable, event_name: str):
    """Build a done-callback that discards the task ref + logs exceptions."""
    def _on_done(task: asyncio.Task) -> None:
        registry._after_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            registry._handler_failures += 1
            registry._handler_last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                f"[HookRunner] async after-event handler error in {event_name} "
                f"({fn.__name__}): {exc}"
            )
    return _on_done
