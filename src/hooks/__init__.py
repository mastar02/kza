"""Plugin hooks system — public API.

Plan #3 OpenClaw — see docs/superpowers/specs/2026-04-29-openclaw-plugin-hooks-design.md
"""

from src.hooks.decorators import before_ha_action, before_tts_speak, after_event
from src.hooks.registry import HookRegistry, _global_registry
from src.hooks.runner import execute_before_chain, execute_after_event
from src.hooks.types import (
    HaActionCall, TtsCall, BlockResult, RewriteResult,
    WakePayload, SttPayload, IntentPayload,
    HaActionDispatchedPayload, HaActionBlockedPayload,
    LlmCallPayload, TtsPayload,
    EVENT_NAMES,
)

__all__ = [
    # Decorators
    "before_ha_action", "before_tts_speak", "after_event",
    # Registry + runner
    "HookRegistry", "_global_registry",
    "execute_before_chain", "execute_after_event",
    # Types
    "HaActionCall", "TtsCall", "BlockResult", "RewriteResult",
    "WakePayload", "SttPayload", "IntentPayload",
    "HaActionDispatchedPayload", "HaActionBlockedPayload",
    "LlmCallPayload", "TtsPayload",
    "EVENT_NAMES",
]
