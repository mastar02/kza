"""Plugin hooks types — frozen dataclasses + enumerated event names.

Plan #3 OpenClaw — see docs/superpowers/specs/2026-04-29-openclaw-plugin-hooks-design.md
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal


class _FrozenDict(dict):
    """Read-only dict subclass — handlers cannot mutate `service_data`.

    Unlike `MappingProxyType`, this is picklable / deepcopy-able, which is
    needed by `dataclasses.asdict` in downstream consumers (audit log).
    """
    __slots__ = ()

    def _ro(self, *args, **kwargs):
        raise TypeError("HaActionCall.service_data is read-only; use call.with_data(**changes)")

    __setitem__ = _ro
    __delitem__ = _ro
    pop = _ro  # type: ignore[assignment]
    popitem = _ro  # type: ignore[assignment]
    clear = _ro  # type: ignore[assignment]
    update = _ro  # type: ignore[assignment]
    setdefault = _ro  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class HaActionCall:
    """Pending HA service call seen by `before_ha_action` hooks.

    Note: `service_data` is wrapped in a read-only mapping at construction
    so handlers cannot mutate it via `call.service_data[k] = v`. Use
    `with_data(**changes)` to produce a modified copy. The wrapper is a
    `_FrozenDict` (dict subclass with mutation methods raising TypeError),
    chosen over `MappingProxyType` because asdict-deepcopy needs it picklable.

    Attributes:
        entity_id: HA entity id (e.g. "light.kitchen").
        domain: HA service domain (e.g. "light", "switch").
        service: HA service name (e.g. "turn_on", "turn_off").
        service_data: Read-only mapping with extra service args (brightness, etc.).
        user_id: KZA-internal user id of the requester, or None if unknown.
        user_name: Human-readable user name, or None if unknown.
        zone_id: Source zone (room) id, or None if unknown.
        timestamp: Unix epoch seconds (`time.time()`, NOT `perf_counter`).
    """
    entity_id: str
    domain: str
    service: str
    service_data: Mapping  # _FrozenDict wraps a dict copy in __post_init__
    user_id: str | None
    user_name: str | None
    zone_id: str | None
    timestamp: float

    def __post_init__(self):
        # Force a defensive copy + immutable view; bypass frozen via object.__setattr__
        if not isinstance(self.service_data, _FrozenDict):
            object.__setattr__(
                self, "service_data",
                _FrozenDict(self.service_data),
            )

    def with_data(self, **changes) -> "HaActionCall":
        """Return a copy with `service_data` updated by `changes`.

        The returned instance has a fresh _FrozenDict wrapping the new
        data, also immutable.
        """
        new_data = {**self.service_data, **changes}
        return replace(self, service_data=new_data)


@dataclass(frozen=True, slots=True)
class TtsCall:
    """Pending TTS speak seen by `before_tts_speak` hooks.

    Attributes:
        text: Text to speak (Spanish in KZA).
        voice: Voice id, or None to use default.
        lang: BCP-47 language tag (e.g. "es", "es-AR").
        user_id: Target user id, or None if broadcast.
        zone_id: Target zone id, or None if all zones.
    """
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

    Attributes:
        reason: Human-readable Spanish reason (spoken/logged).
        rule_name: Policy/rule identifier for telemetry.
    """
    reason: str
    rule_name: str


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """Returned from a `before_*` hook to modify the call/text.

    The chain continues with `modified` as the new call. `modified` must be
    of the same dataclass type as the call passed in (HaActionCall or TtsCall).

    Attributes:
        modified: The new call (same dataclass type as original input).
        rule_name: Policy/rule identifier for telemetry.
    """
    modified: object  # HaActionCall | TtsCall
    rule_name: str


# === Event payloads (one per event_name) ===

@dataclass(frozen=True, slots=True)
class WakePayload:
    """Payload for `wake` after-event.

    Attributes:
        timestamp: Unix epoch seconds (`time.time()`).
        confidence: Wake-word detector confidence in [0.0, 1.0].
        zone_id: Zone id where wake was detected, or None.
    """
    timestamp: float
    confidence: float
    zone_id: str | None


@dataclass(frozen=True, slots=True)
class SttPayload:
    """Payload for `stt` after-event.

    Attributes:
        timestamp: Unix epoch seconds (`time.time()`).
        text: Transcribed text (may be empty on failure).
        latency_ms: STT latency in milliseconds.
        user_id: Identified user id, or None.
        zone_id: Source zone id, or None.
        success: True if STT produced a transcript.
    """
    timestamp: float
    text: str
    latency_ms: float
    user_id: str | None
    zone_id: str | None
    success: bool


@dataclass(frozen=True, slots=True)
class IntentPayload:
    """Payload for `intent` after-event.

    Attributes:
        timestamp: Unix epoch seconds (`time.time()`).
        text: Source text classified into the intent.
        intent: Intent label ("domotics" | "music" | "conversation" | ...).
        entities: Tuple of extracted entity strings (immutable).
        user_id: User id, or None.
    """
    timestamp: float
    text: str
    intent: str   # "domotics" | "music" | "conversation" | ...
    entities: tuple[str, ...]
    user_id: str | None

    def __post_init__(self):
        if not isinstance(self.entities, tuple):
            object.__setattr__(self, "entities", tuple(self.entities))


@dataclass(frozen=True, slots=True)
class HaActionDispatchedPayload:
    """Payload for `ha_action_dispatched` after-event.

    Attributes:
        timestamp: Unix epoch seconds (`time.time()`).
        call: The HA call that was dispatched.
        success: True if HA accepted the call.
        error: Error string when success is False, else None.
    """
    timestamp: float
    call: HaActionCall
    success: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class HaActionBlockedPayload:
    """Payload for `ha_action_blocked` after-event.

    Attributes:
        timestamp: Unix epoch seconds (`time.time()`).
        call: The HA call that was blocked.
        block: BlockResult emitted by the policy.
    """
    timestamp: float
    call: HaActionCall
    block: BlockResult


@dataclass(frozen=True, slots=True)
class LlmCallPayload:
    """Payload for `llm_call` after-event.

    Attributes:
        timestamp: Unix epoch seconds (`time.time()`).
        path: "fast" (router 7B) or "slow" (reasoner 72B).
        endpoint_id: Stable endpoint id ("fast_router_7b" | "reasoner_72b").
        latency_ms: End-to-end LLM call latency in milliseconds.
        success: True if the call returned a usable response.
    """
    timestamp: float
    path: Literal["fast", "slow"]
    endpoint_id: str  # "fast_router_7b" | "reasoner_72b"
    latency_ms: float
    success: bool


@dataclass(frozen=True, slots=True)
class TtsPayload:
    """Payload for `tts` after-event.

    Attributes:
        timestamp: Unix epoch seconds (`time.time()`).
        text: Spoken text.
        voice: Voice id used, or None for default.
        latency_ms: Synthesis latency in milliseconds.
        success: True if TTS produced audio.
    """
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
