"""Hook runners — execute_before_chain + execute_after_event.

Plan #3 OpenClaw.
"""

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
