"""Bounded observation metadata for existing authenticated collection responses."""

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from src.core.observation import Observation, ObservationReason, ObservationSource, ObservationState
from src.home_assistant.state_projection import EntityObservation

COMPONENTS = frozenset({'zones', 'conversations', 'ha_entities', 'ha_actions', 'llm_endpoints',
                        'users', 'alerts', 'gpus', 'services'})


def observation_headers(value: Observation, *, now: datetime) -> dict[str, str]:
    """Project only closed metadata; expiry describes observation freshness."""
    data = value.to_dict(now)
    headers = {'Cache-Control': 'no-store',
        'X-KZA-Source': 'mock' if value.source is ObservationSource.DEMO else 'real' if value.current(now) else 'degraded',
        'X-KZA-Source-Id': data['source'], 'X-KZA-Observation-State': data['state'],
        'X-KZA-Expiry-Scope': 'observation', 'X-KZA-Unobserved-Fields': ','.join(data['unobserved_fields'])}
    for key, header in (('observed_at', 'X-KZA-Observed-At'), ('expires_at', 'X-KZA-Expires-At'),
                        ('attempted_at', 'X-KZA-Attempted-At'), ('reason', 'X-KZA-Observation-Reason')):
        if data[key] is not None:
            headers[header] = data[key]
    return headers

def entity_observation_metadata(value: EntityObservation, *, now: datetime) -> Observation:
    """Describe the original entity evidence without renewing its deadline.

    Args:
        value: Native current or last-known entity observation.
        now: The HTTP projection time, used only to check freshness.

    Returns:
        Closed source metadata and the original observation timestamps.
    """
    if type(value) is not EntityObservation:
        raise ValueError("invalid_entity_observation")
    source = ObservationSource(value.source)
    if source not in {ObservationSource.HA_ENTITY, ObservationSource.HA_ENTITIES,
                      ObservationSource.HA_STATES}:
        raise ValueError("invalid_entity_observation_source")

    def instant(epoch: float | None) -> datetime | None:
        return None if epoch is None else datetime.fromtimestamp(epoch, UTC)

    reason = None if value.reason is None else ObservationReason(value.reason)
    observed, expiry, attempted = (instant(value.observed_at), instant(value.expires_at),
                                  instant(value.attempted_at))
    if reason in {ObservationReason.EXPIRED, ObservationReason.CLOCK_SKEW}:
        state = ObservationState.STALE if reason is ObservationReason.EXPIRED else ObservationState.UNKNOWN
        return Observation(source, state, observed, expiry, attempted, reason)
    return Observation.from_read(source,
        available=value.exists is not None and reason is None,
        observed_at=observed, expires_at=expiry, attempted_at=attempted, ttl_s=6,
        reason=reason)


class ObservationHeaders:
    """Remember only metadata; never retain personal payloads or query parameters."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None, ttl_s: float = 6) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._ttl_s = ttl_s
        self._last: dict[str, Observation] = {}

    def apply(self, response, component: str, source: ObservationSource, *, available: bool = False,
              observation: Observation | None = None, demo: bool = False,
              unobserved_fields: tuple[str, ...] = (), reason: ObservationReason | None = None) -> Observation:
        """Emit provenance without treating retries or serialization as observations."""
        if component not in COMPONENTS:
            raise ValueError('unknown_observation_component')
        now = self._clock()
        if demo:
            value = Observation(ObservationSource.DEMO, ObservationState.DEMO)
        elif observation is not None:
            if observation.source is not source:
                raise ValueError('observation_source_mismatch')
            value = observation.as_of(now)
            previous = self._last.get(component)
            if value.observed_at is None and previous is not None and value.state in {
                ObservationState.UNAVAILABLE, ObservationState.DENIED,
            }:
                value = replace(value, observed_at=previous.observed_at, expires_at=previous.expires_at)
        elif available:
            value = Observation.from_read(source, available=True, observed_at=now,
                ttl_s=self._ttl_s, attempted_at=now, unobserved_fields=unobserved_fields)
        elif component in self._last:
            value = self._last[component].failed(attempted_at=now, reason=reason or ObservationReason.UNAVAILABLE)
        else:
            value = Observation.from_read(source, available=False, observed_at=None,
                ttl_s=self._ttl_s, attempted_at=now, reason=reason or ObservationReason.UNAVAILABLE)
        if unobserved_fields:
            fields = tuple(sorted(set(value.unobserved_fields) | set(unobserved_fields)))
            value = replace(value, unobserved_fields=fields,
                state=ObservationState.PARTIAL if value.state is ObservationState.FRESH else value.state)
        if not demo:
            previous = self._last.get(component)
            minimum = datetime.min.replace(tzinfo=UTC)
            def rank(item):
                return item.observed_at or minimum, item.attempted_at or minimum
            if previous is None or rank(value) >= rank(previous):
                self._last[component] = value
        headers = observation_headers(value, now=now)
        for header in ("X-KZA-Observed-At", "X-KZA-Expires-At", "X-KZA-Attempted-At", "X-KZA-Observation-Reason"):
            if header not in headers and header in response.headers:
                del response.headers[header]
        response.headers.update(headers)
        return value
