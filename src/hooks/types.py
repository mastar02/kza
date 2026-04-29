"""Plugin hooks types — frozen dataclasses + enumerated event names.

Plan #3 OpenClaw — see docs/superpowers/specs/2026-04-29-openclaw-plugin-hooks-design.md
"""

from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True, slots=True)
class HaActionCall:
    """Pending HA service call seen by `before_ha_action` hooks."""
    entity_id: str
    domain: str
    service: str
    service_data: dict
    user_id: str | None
    user_name: str | None
    zone_id: str | None
    timestamp: float

    def with_data(self, **changes) -> "HaActionCall":
        """Return a copy with `service_data` updated by `changes`."""
        new_data = {**self.service_data, **changes}
        return replace(self, service_data=new_data)


@dataclass(frozen=True, slots=True)
class TtsCall:
    """Pending TTS speak seen by `before_tts_speak` hooks."""
    text: str
    voice: str | None
    lang: str
    user_id: str | None
    zone_id: str | None


@dataclass(frozen=True, slots=True)
class BlockResult:
    """Returned from a `before_*` hook to short-circuit the chain.

    The reason is spoken to the user via TTS (HA blocks) or logged (TTS blocks).
    rule_name identifies the policy for telemetry/audit.
    """
    reason: str
    rule_name: str


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """Returned from a `before_*` hook to modify the call/text.

    The chain continues with `modified` as the new call. `modified` must be
    of the same dataclass type as the call passed in (HaActionCall or TtsCall).
    """
    modified: object  # HaActionCall | TtsCall
    rule_name: str


# === Event payloads (one per event_name) ===

@dataclass(frozen=True, slots=True)
class WakePayload:
    timestamp: float
    confidence: float
    zone_id: str | None


@dataclass(frozen=True, slots=True)
class SttPayload:
    timestamp: float
    text: str
    latency_ms: float
    user_id: str | None
    zone_id: str | None
    success: bool


@dataclass(frozen=True, slots=True)
class IntentPayload:
    timestamp: float
    text: str
    intent: str   # "domotics" | "music" | "conversation" | ...
    entities: list[str]
    user_id: str | None


@dataclass(frozen=True, slots=True)
class HaActionDispatchedPayload:
    timestamp: float
    call: HaActionCall
    success: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class HaActionBlockedPayload:
    timestamp: float
    call: HaActionCall
    block: BlockResult


@dataclass(frozen=True, slots=True)
class LlmCallPayload:
    timestamp: float
    path: Literal["fast", "slow"]
    endpoint_id: str  # "fast_router_7b" | "reasoner_72b"
    latency_ms: float
    success: bool


@dataclass(frozen=True, slots=True)
class TtsPayload:
    timestamp: float
    text: str
    voice: str | None
    latency_ms: float
    success: bool


# Closed enumeration of event names — handlers register against these.
EVENT_NAMES: tuple[str, ...] = (
    "wake",
    "stt",
    "intent",
    "ha_action_dispatched",
    "ha_action_blocked",
    "llm_call",
    "tts",
)
