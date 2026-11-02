"""Owned, shared observations for voice resolution and home catalog projection."""

import asyncio
from collections.abc import Callable
from contextvars import Context
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import logging

from src.core.observation import Observation, ObservationReason, ObservationSource, ObservationState, combine_observations
from src.home_assistant.capability_projection import HomeControlLimits, projected_capabilities
from src.home_assistant.control_policy import EntityGrant, HomeActionDenied, HomeControlPolicy
from src.home_assistant.ha_client import (
    HAAreaAssignments, HAStatesSnapshot, REGISTRY_OBSERVATION_TTL_S, STATE_OBSERVATION_TTL_S,
    get_home_capabilities,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityCapability:
    """Technical support and selected operations, with an independent observation."""

    entity_id: str
    technical_operations: tuple[str, ...]
    selected: bool
    operations: tuple[str, ...]
    limits: HomeControlLimits
    observation: Observation
    state_available: bool

    def available(self, now: datetime | None = None) -> bool:
        """Require current state evidence, without granting authority to send."""
        return self.state_available and self.observation.current(now)

    def effective_operations(self, now: datetime | None = None) -> tuple[str, ...]:
        """Expose operations only while all of their evidence dependencies remain current."""
        return self.operations if self.available(now) else ()


@dataclass(frozen=True)
class CapabilitySnapshot:
    """One composed observation shared by consumers; selection never authorizes a send."""

    observation: Observation
    registry_observation: Observation
    selection_observation: Observation
    temperature_observation: Observation
    entities: tuple[EntityCapability, ...]
    states: HAStatesSnapshot = field(repr=False)
    areas: HAAreaAssignments = field(repr=False)
    grants: tuple[EntityGrant, ...] = field(repr=False)
    temperature_unit: str | None = None

    def entity(self, entity_id: str) -> EntityCapability | None:
        """Find a projected entity without falling back to a similar target."""
        return next((value for value in self.entities if value.entity_id == entity_id), None)

    def domain_present(self, domain: str, now: datetime | None = None) -> bool | None:
        """Distinguish a real empty domain from missing or expired observation."""
        if not self.observation.current(now):
            return None
        return any(value.entity_id.startswith(domain + ".") for value in self.entities)


def _read_observation(source: ObservationSource, value, ttl_s: float) -> Observation:
    reason = None
    if value.reason:
        reason = (ObservationReason.DENIED if value.reason == "ha_states_denied" else
                  ObservationReason.INVALID if value.reason == "ha_states_invalid" else ObservationReason.UNAVAILABLE)
    return Observation.from_read(source, available=value.available, observed_at=value.observed_at,
        expires_at=value.expires_at, attempted_at=value.attempted_at, ttl_s=ttl_s, reason=reason)


class HomeCapabilityService:
    """Own one on-demand refresh and drain it before the shared HA transport closes."""

    def __init__(self, ha_client, policy: HomeControlPolicy, *, clock: Callable[[], datetime] | None = None,
                 policy_provider: Callable[[], HomeControlPolicy] | None = None) -> None:
        self.ha = ha_client
        self._policy = policy
        self._policy_provider = policy_provider
        self._clock = clock or (lambda: datetime.now(UTC))
        self._refresh_task: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None
        self._last_snapshot: CapabilitySnapshot | None = None
        self._closed = False

    def now(self) -> datetime:
        """Return the observation clock shared by consumers of this service."""
        return self._clock()

    async def snapshot(self) -> CapabilitySnapshot:
        """Share simultaneous reads without hiding subsequent policy changes."""
        if self._closed:
            raise RuntimeError("capabilities_closed")
        if self._refresh_task is None or self._refresh_task.done():
            self._refresh_task = asyncio.create_task(self._refresh(), context=Context())
            self._refresh_task.add_done_callback(self._observe, context=Context())
        return await asyncio.shield(self._refresh_task)

    @staticmethod
    def _observe(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()

    async def _selection(self) -> tuple[tuple[EntityGrant, ...], Observation]:
        try:
            policy = self._policy_provider() if self._policy_provider is not None else self._policy
            grants = await policy.snapshot_grants()
        except HomeActionDenied:
            if self._last_snapshot is not None:
                return (), self._last_snapshot.selection_observation.failed(attempted_at=self._clock())
            return (), Observation.from_read(ObservationSource.CONTROL_SELECTION, available=False,
                observed_at=None, ttl_s=STATE_OBSERVATION_TTL_S, attempted_at=self._clock(),
                reason=ObservationReason.UNAVAILABLE)
        now = self._clock()
        return tuple(grants), Observation.from_read(ObservationSource.CONTROL_SELECTION, available=True,
            observed_at=now, ttl_s=STATE_OBSERVATION_TTL_S, attempted_at=now)

    async def _refresh(self) -> CapabilitySnapshot:
        try:
            if self.ha is None:
                states = HAStatesSnapshot(False, None, "ha_states_unavailable")
                areas = HAAreaAssignments(False, None, "ha_registries_unavailable")
                grants, selection = await self._selection()
            else:
                async with asyncio.TaskGroup() as group:
                    state_task = group.create_task(self.ha.get_state_snapshot())
                    area_task = group.create_task(self.ha.get_area_assignments())
                    selection_task = group.create_task(self._selection())
                states, areas = state_task.result(), area_task.result()
                grants, selection = selection_task.result()
                if not isinstance(states, HAStatesSnapshot) or not isinstance(areas, HAAreaAssignments):
                    raise ValueError("invalid_capability_source")
            needs_unit = self.ha is not None and any("set_temperature" in grant.operations for grant in grants)
            unit_attempted_at = self._clock() if needs_unit else None
            unit = await self.ha.get_temperature_unit_fresh() if needs_unit else None
            now = self._clock()
            previous = self._last_snapshot
            if unit in {"°C", "°F"}:
                temperature = Observation.from_read(ObservationSource.HA_CONFIG, available=True,
                    observed_at=now, ttl_s=STATE_OBSERVATION_TTL_S, attempted_at=unit_attempted_at)
            elif needs_unit:
                temperature = (previous.temperature_observation.failed(attempted_at=unit_attempted_at)
                    if previous is not None else Observation.from_read(ObservationSource.HA_CONFIG,
                        available=False, observed_at=None, ttl_s=STATE_OBSERVATION_TTL_S, attempted_at=unit_attempted_at))
            else:
                temperature = (previous.temperature_observation.as_of(now) if previous is not None
                    else Observation(ObservationSource.HA_CONFIG, ObservationState.UNKNOWN, reason=ObservationReason.UNCONFIGURED))
                unit = previous.temperature_unit if previous is not None and temperature.current(now) else None
            observation = _read_observation(ObservationSource.HA_STATES, states, STATE_OBSERVATION_TTL_S).as_of(now)
            registry = _read_observation(ObservationSource.HA_REGISTRY, areas, REGISTRY_OBSERVATION_TTL_S).as_of(now)
            by_id = {value["entity_id"]: value for value in states.states}
            by_grant = {grant.entity_id: grant for grant in grants}
            entities = []
            for state in states.states:
                entity_id = state["entity_id"]
                operations, limits = projected_capabilities(state, by_grant, by_id, unit)
                grant = by_grant.get(entity_id)
                if not selection.current(now):
                    operations, limits = (), HomeControlLimits()
                available = state["state"] not in {"unknown", "unavailable"} or (entity_id.startswith("scene.") and bool(operations))
                selected = grant is not None and grant.direct_control
                dependencies = (observation, selection) if selected else (observation,)
                if "set_temperature" in operations:
                    dependencies += (temperature,)
                effective_observation = combine_observations(ObservationSource.HOME_CAPABILITIES, dependencies, now=now)
                entities.append(EntityCapability(entity_id, get_home_capabilities(state),
                    selected, operations, limits, effective_observation, available))
            result = CapabilitySnapshot(observation, registry, selection, temperature,
                tuple(entities), states, areas, grants, unit)
        except Exception as error:
            logger.warning("Capability observation unavailable: %s", type(error).__name__)
            result = self._unavailable()
        self._last_snapshot = result
        return result

    def _unavailable(self) -> CapabilitySnapshot:
        now = self._clock()
        previous = self._last_snapshot
        def failed(value):
            return value.failed(attempted_at=now, reason=ObservationReason.REFRESH_FAILED)
        if previous is not None:
            observation = failed(previous.observation)
            return replace(previous, observation=observation, registry_observation=failed(previous.registry_observation),
                selection_observation=failed(previous.selection_observation), temperature_observation=failed(previous.temperature_observation),
                entities=tuple(replace(entity, selected=False, operations=(), limits=HomeControlLimits(),
                                       observation=observation) for entity in previous.entities), grants=())
        def missing(source):
            return Observation.from_read(source, available=False, observed_at=None, ttl_s=STATE_OBSERVATION_TTL_S,
                attempted_at=now, reason=ObservationReason.REFRESH_FAILED)
        return CapabilitySnapshot(missing(ObservationSource.HA_STATES), missing(ObservationSource.HA_REGISTRY),
            missing(ObservationSource.CONTROL_SELECTION), missing(ObservationSource.HA_CONFIG), (),
            HAStatesSnapshot(False, None, "ha_states_unavailable"), HAAreaAssignments(False, None, "ha_registries_unavailable"), ())

    async def aclose(self) -> None:
        """Close admission and retain ownership through repeated waiter cancellation."""
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._drain(), context=Context())
        await asyncio.shield(self._close_task)

    async def _drain(self) -> None:
        if self._refresh_task is not None:
            await asyncio.gather(self._refresh_task, return_exceptions=True)
