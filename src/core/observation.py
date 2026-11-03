"""Transport-neutral provenance and expiry for observations, never action grants."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import math
import re


class ObservationSource(StrEnum):
    HA_STATES = "ha-rest-states"
    HA_ENTITY = "ha-rest-entity"
    HA_ENTITIES = "ha-ws-entities"
    HA_REGISTRY = "ha-registry"
    HA_CONFIG = "ha-config"
    HA_FORECAST = "ha-weather-forecast"
    HOME_CAPABILITIES = "home-capabilities"
    CONTROL_SELECTION = "control-selection"
    ZONE_MANAGER = "zone-manager"
    ROUTER_STATE = "router-state"
    USER_REGISTRY = "user-registry"
    ALERT_MANAGER = "alert-manager"
    EVENT_STORE = "event-store"
    GPU_PROBE = "nvidia-smi"
    SERVICE_PROBE = "service-probes"
    DEMO = "demo"


class ObservationState(StrEnum):
    FRESH = "fresh"
    PARTIAL = "partial"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    DENIED = "denied"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    DEMO = "demo"


class ObservationReason(StrEnum):
    UNAVAILABLE = "source_unavailable"
    DENIED = "source_denied"
    INVALID = "source_invalid"
    UNCONFIGURED = "source_unconfigured"
    UNKNOWN_TIME = "observation_time_unknown"
    EXPIRED = "observation_expired"
    REFRESH_FAILED = "refresh_failed"
    CLOSED = "source_closed"
    CLOCK_SKEW = "clock_before_observation"
    CHANGED = "source_changed"


def _aware(value: datetime | None) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


@dataclass(frozen=True)
class Observation:
    """Provenance whose successful time is independent of attempts and serialization."""

    source: ObservationSource
    state: ObservationState
    observed_at: datetime | None = None
    expires_at: datetime | None = None
    attempted_at: datetime | None = None
    reason: ObservationReason | None = None
    unobserved_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, ObservationSource) or not isinstance(self.state, ObservationState):
            raise ValueError("invalid_observation_code")
        if self.reason is not None and not isinstance(self.reason, ObservationReason):
            raise ValueError("invalid_observation_reason")
        if any(value is not None and not _aware(value) for value in (self.observed_at, self.expires_at, self.attempted_at)):
            raise ValueError("invalid_observation_time")
        if (self.observed_at is None) != (self.expires_at is None):
            raise ValueError("incomplete_observation_time")
        if self.observed_at is not None and self.expires_at <= self.observed_at:
            raise ValueError("invalid_observation_expiry")
        if self.state in {ObservationState.FRESH, ObservationState.PARTIAL} and self.observed_at is None:
            raise ValueError("missing_observation_time")
        if self.state is ObservationState.DEMO and self.source is not ObservationSource.DEMO:
            raise ValueError("invalid_demo_source")
        if self.source is ObservationSource.DEMO and (self.state is not ObservationState.DEMO or self.observed_at is not None):
            raise ValueError("demo_is_not_an_observation")
        if type(self.unobserved_fields) is not tuple or len(self.unobserved_fields) > 20 or any(
            type(name) is not str or re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name) is None
            for name in self.unobserved_fields
        ):
            raise ValueError("invalid_unobserved_fields")

    @classmethod
    def from_read(cls, source: ObservationSource, *, available: bool, observed_at: datetime | None,
                  ttl_s: float, expires_at: datetime | None = None, attempted_at: datetime | None = None,
                  reason: ObservationReason | None = None, unobserved_fields: tuple[str, ...] = ()) -> "Observation":
        """Preserve original successful timestamps, including after failed reads."""
        if type(available) is not bool or type(ttl_s) not in (int, float) or not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("invalid_observation_read")
        expires_at = expires_at or (observed_at + timedelta(seconds=ttl_s) if observed_at is not None else None)
        state = ObservationState.PARTIAL if unobserved_fields else ObservationState.FRESH
        if not available:
            state = ObservationState.DENIED if reason is ObservationReason.DENIED else ObservationState.UNAVAILABLE
            reason = reason or ObservationReason.UNAVAILABLE
        elif observed_at is None:
            state, reason = ObservationState.UNKNOWN, ObservationReason.UNKNOWN_TIME
        return cls(source, state, observed_at, expires_at, attempted_at, reason, unobserved_fields)

    def as_of(self, now: datetime | None = None) -> "Observation":
        """Expire a successful read without changing its source or original dates."""
        now = datetime.now(UTC) if now is None else now
        if not _aware(now):
            raise ValueError("invalid_observation_clock")
        if self.observed_at is not None and self.observed_at > now:
            return replace(self, state=ObservationState.UNKNOWN, reason=ObservationReason.CLOCK_SKEW)
        if self.state in {ObservationState.FRESH, ObservationState.PARTIAL} and self.expires_at <= now:
            return replace(self, state=ObservationState.STALE, reason=ObservationReason.EXPIRED)
        return self

    def current(self, now: datetime | None = None) -> bool:
        """Return whether this source still has a successful, unexpired observation."""
        return self.as_of(now).state in {ObservationState.FRESH, ObservationState.PARTIAL}

    def failed(self, *, attempted_at: datetime, reason: ObservationReason = ObservationReason.UNAVAILABLE) -> "Observation":
        """Record a failed attempt without refreshing the retained real observation."""
        return replace(self, state=ObservationState.UNAVAILABLE, attempted_at=attempted_at, reason=reason)

    def to_dict(self, now: datetime | None = None) -> dict:
        """Serialize closed metadata without values, private IDs or free-form errors."""
        value = self.as_of(now)
        return {"source": value.source.value, "state": value.state.value,
                "observed_at": value.observed_at.astimezone(UTC).isoformat() if value.observed_at else None,
                "expires_at": value.expires_at.astimezone(UTC).isoformat() if value.expires_at else None,
                "attempted_at": value.attempted_at.astimezone(UTC).isoformat() if value.attempted_at else None,
                "reason": value.reason.value if value.reason else None,
                "unobserved_fields": value.unobserved_fields}


def combine_observations(source: ObservationSource, values: tuple[Observation, ...], *, now: datetime) -> Observation:
    """Intersect dependency lifetimes without extending any successful observation."""
    if not values:
        raise ValueError("missing_observation_dependencies")
    current = tuple(value.as_of(now) for value in values)
    known = all(value.observed_at is not None for value in current)
    observed_at = min(value.observed_at for value in current) if known else None
    expires_at = min(value.expires_at for value in current) if known else None
    attempts = tuple(value.attempted_at for value in current if value.attempted_at is not None)
    unknown_fields = tuple(sorted({name for value in current for name in value.unobserved_fields}))
    missing = next((value for value in current if not value.current(now)), None)
    state = ObservationState.PARTIAL if unknown_fields else ObservationState.FRESH
    reason = None
    if missing is not None:
        state, reason = missing.state, missing.reason
        if state is ObservationState.DEMO:
            state, reason = ObservationState.UNKNOWN, ObservationReason.INVALID
    if not known and state in {ObservationState.FRESH, ObservationState.PARTIAL}:
        state, reason = ObservationState.UNKNOWN, ObservationReason.UNKNOWN_TIME
    return Observation(source, state, observed_at, expires_at, max(attempts) if attempts else None,
                       reason, unknown_fields)
