"""Read explicitly selected Home Assistant energy and power sensors."""

import asyncio
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, localcontext
from enum import StrEnum
from math import isfinite
from time import time
from typing import Literal, Protocol

from src.home_assistant.control_policy import ActorKind, HomeActionDenied, HomeActor
from src.home_assistant.state_projection import EntityObservation


class MeasurementKind(StrEnum):
    """Home Assistant device classes supported by the monitor."""

    ENERGY = "energy"
    POWER = "power"


class ReadingStatus(StrEnum):
    """Whether a selected sensor can contribute to the summary."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    MISSING = "missing"
    SOURCE_UNAVAILABLE = "source_unavailable"
    INCOMPATIBLE = "incompatible"
    NON_NUMERIC = "non_numeric"
    STALE = "stale"


_UNITS = {MeasurementKind.ENERGY: "kWh", MeasurementKind.POWER: "W"}
_SCALES = {
    MeasurementKind.ENERGY: {"Wh": -3, "kWh": 0, "MWh": 3},
    MeasurementKind.POWER: {"W": 0, "kW": 3},
}


class EnergyStateProvider(Protocol):
    """The read-only subset of the Home Assistant client used here."""

    async def read_entity_observation(self, entity_id: str) -> EntityObservation:
        """Read one entity with freshness and failure metadata."""
        ...


@dataclass(frozen=True, slots=True)
class EnergySensor:
    """An explicitly shared entity and its expected measurement kind."""

    entity_id: str
    kind: MeasurementKind
    room: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"sensor\.[a-z0-9_]+", self.entity_id):
            raise ValueError("invalid_energy_sensor_id")
        if not isinstance(self.kind, MeasurementKind):
            raise ValueError("invalid_energy_sensor_kind")


@dataclass(frozen=True, slots=True)
class EnergyReading:
    """A sensor's value and provenance at the time of the query.

    observed_at is HA's sensor report time, with last_updated as the legacy
    fallback. stale also covers missing or unverifiable observation times.
    """

    sensor: EnergySensor
    status: ReadingStatus
    value: Decimal | None
    unit: str
    original_unit: str | None
    source: str | None
    observed_at: datetime | None
    stale: bool


@dataclass(frozen=True, slots=True)
class EnergySummary:
    """Current selected counters or instantaneous readings for one scope.

    The total sums the available readings from the selected sensors. Configure
    non-overlapping meters to represent a household total. Reading timestamps
    can differ and remain attached to each sensor.
    """

    kind: MeasurementKind
    room: str | None
    entity_id: str | None
    as_of: datetime
    unit: str
    total: Decimal | None
    readings: tuple[EnergyReading, ...]

    @property
    def selected_count(self) -> int:
        """Return the number of configured sensors in this scope."""
        return len(self.readings)

    @property
    def available_count(self) -> int:
        """Return the number of usable sensors contributing to the total."""
        return sum(
            reading.status is ReadingStatus.AVAILABLE for reading in self.readings
        )


class EnergyCostUnavailable(ValueError):
    """Cost and billing queries require a verified tariff source."""


class EnergyMonitor:
    """Expose shared, read-only measurements from a configured selection."""

    def __init__(
        self,
        ha_client: EnergyStateProvider,
        sensors: Iterable[EnergySensor],
        *,
        max_staleness_s: float,
        clock: Callable[[], float] = time,
    ) -> None:
        """Set the shared selection, sensor age limit and POSIX clock."""
        if (
            isinstance(max_staleness_s, bool)
            or not isfinite(max_staleness_s)
            or max_staleness_s <= 0
        ):
            raise ValueError("invalid_energy_max_staleness")
        self._sensors: dict[str, EnergySensor] = {}
        for sensor in sensors:
            if sensor.entity_id in self._sensors:
                raise ValueError("duplicate_energy_sensor")
            self._sensors[sensor.entity_id] = sensor
        self._ha_client = ha_client
        self._clock = clock
        self._max_staleness_s = max_staleness_s

    async def read_sensor(self, entity_id: str, *, actor: HomeActor) -> EnergyReading:
        """Read one configured sensor after authorizing the shared scope."""
        _require_home_read(actor)
        sensor = self._selected_sensor(entity_id)
        observation = await self._ha_client.read_entity_observation(entity_id)
        _require_home_read(actor)
        return self._reading(sensor, observation, self._clock())

    async def summarize(
        self,
        kind: MeasurementKind | Literal["cost", "bill"],
        *,
        actor: HomeActor,
        room: str | None = None,
        entity_id: str | None = None,
    ) -> EnergySummary:
        """Summarize one measurement kind by sensor, room, or household.

        Args:
            kind: The kind of measurement requested.
            actor: The trusted actor requesting the shared data.
            room: The optional configured room to summarize.
            entity_id: The optional configured sensor to summarize.

        Raises:
            PermissionError: The actor lacks current shared read authority.
            EnergyCostUnavailable: A cost or bill was requested.
            ValueError: The kind or scope is invalid for this selection.
        """
        _require_home_read(actor)
        if kind in ("cost", "bill"):
            raise EnergyCostUnavailable("energy_tariff_unverified")
        kind = MeasurementKind(kind)
        if entity_id is not None:
            if room is not None:
                raise ValueError("energy_summary_scope_ambiguous")
            sensor = self._selected_sensor(entity_id)
            if sensor.kind is not kind:
                raise ValueError("energy_sensor_kind_mismatch")
            selected = (sensor,)
        else:
            selected = tuple(
                sensor
                for sensor in self._sensors.values()
                if sensor.kind is kind and (room is None or sensor.room == room)
            )
        observations = await asyncio.gather(
            *(
                self._ha_client.read_entity_observation(sensor.entity_id)
                for sensor in selected
            )
        )
        _require_home_read(actor)
        now = self._clock()
        readings = tuple(
            self._reading(sensor, observation, now)
            for sensor, observation in zip(selected, observations, strict=True)
        )
        values = [reading.value for reading in readings if reading.value is not None]
        total = _sum_values(values) if values else None
        return EnergySummary(
            kind,
            room,
            entity_id,
            datetime.fromtimestamp(now, UTC),
            _UNITS[kind],
            total,
            readings,
        )

    def _selected_sensor(self, entity_id: str) -> EnergySensor:
        sensor = self._sensors.get(entity_id)
        if sensor is None:
            raise ValueError("energy_sensor_not_selected")
        return sensor

    def _reading(
        self, sensor: EnergySensor, observation: EntityObservation, now: float
    ) -> EnergyReading:
        kind = sensor.kind
        if observation.entity_id != sensor.entity_id or observation.exists is None:
            return EnergyReading(
                sensor,
                ReadingStatus.SOURCE_UNAVAILABLE,
                None,
                _UNITS[kind],
                None,
                None,
                None,
                True,
            )
        current = observation.current(now)
        state = observation.state
        if state is None:
            status = ReadingStatus.MISSING if current else ReadingStatus.STALE
            return EnergyReading(
                sensor, status, None, _UNITS[kind], None, observation.source, None, True
            )

        attributes = state["attributes"]
        unit = attributes.get("unit_of_measurement")
        original_unit = unit if isinstance(unit, str) else None
        observed_at = _reported_at(state)
        stale = (
            not current
            or observed_at is None
            or not 0 <= now - observed_at.timestamp() < self._max_staleness_s
        )
        scale = _SCALES[kind].get(original_unit)
        raw_value = state["state"]
        value = None
        if raw_value == "unavailable":
            status = ReadingStatus.UNAVAILABLE
        elif raw_value == "unknown":
            status = ReadingStatus.UNKNOWN
        elif attributes.get("device_class") != kind or scale is None:
            status = ReadingStatus.INCOMPATIBLE
        elif (normalized := _normalized_value(raw_value, scale)) is None:
            status = ReadingStatus.NON_NUMERIC
        elif stale:
            status = ReadingStatus.STALE
        else:
            status = ReadingStatus.AVAILABLE
            value = normalized
        return EnergyReading(
            sensor,
            status,
            value,
            _UNITS[kind],
            original_unit,
            observation.source,
            observed_at,
            stale,
        )


def _require_home_read(actor: HomeActor) -> None:
    if not isinstance(actor, HomeActor) or (
        actor.kind is not ActorKind.HOUSEHOLD and "home:read" not in actor.permissions
    ):
        raise PermissionError("home_read_denied")
    try:
        actor.require_current()
    except HomeActionDenied:
        raise PermissionError("home_read_denied") from None


def _reported_at(state: dict) -> datetime | None:
    reported = state.get("last_reported", state.get("last_updated"))
    if reported is None:
        return None
    # EntityObservation already validates the date and its offset.
    return datetime.fromisoformat(reported)


def _normalized_value(raw_value: str, scale: int) -> Decimal | None:
    try:
        raw = Decimal(raw_value)
        sign, digits, exponent = raw.as_tuple()
        if not isinstance(exponent, int):
            return None
        if raw.is_zero():
            return Decimal(0)
        value = Decimal((sign, digits, exponent + scale))
    except InvalidOperation:
        return None
    # Exclude magnitudes outside the finite numeric range of HA sensor values.
    approximate = float(value)
    return value if approximate and isfinite(approximate) else None


def _sum_values(values: list[Decimal]) -> Decimal:
    # Preserve decimal places across sensors of different magnitudes, including
    # space for carry when adding more than one value.
    leftmost = max(value.adjusted() for value in values)
    rightmost = min(value.as_tuple().exponent for value in values)
    with localcontext(prec=leftmost - rightmost + len(str(len(values))) + 1):
        return sum(values, Decimal(0))
