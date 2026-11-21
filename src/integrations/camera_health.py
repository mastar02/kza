"""Read-only Home Assistant camera availability and technical diagnostics."""

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from src.home_assistant.control_policy import HomeActionDenied, HomeActor

logger = logging.getLogger(__name__)

_CAMERA_ID = re.compile(r"camera\.[a-z0-9_]+")
_DIAGNOSTIC_ID = re.compile(r"(?:binary_sensor|sensor)\.[a-z0-9_]+")
_PRIVATE_SENSOR = re.compile(
    r"motion|moving|movement|person|people|occupan|presence|human|face|"
    r"movimiento|presencia|ocupaci|rostro"
)
_HEALTH_SCOPES = frozenset({"home:read", "admin:manage"})


class SourceStatus(StrEnum):
    """Whether Home Assistant supplied a current observation."""

    NOT_CONFIGURED = "not_configured"
    DISCONNECTED = "disconnected"
    STALE = "stale"
    UNKNOWN = "unknown"
    AVAILABLE = "available"


class CameraAvailability(StrEnum):
    """Technical availability of a single camera."""

    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class DiagnosticKind(StrEnum):
    """Technical categories permitted in a camera configuration."""

    CONNECTIVITY = "connectivity"
    POWER = "power"
    BATTERY = "battery"
    STORAGE = "storage"
    DEVICE_ERROR = "device_error"


class DiagnosticStatus(StrEnum):
    """Evidence reported by the configured technical entity."""

    NORMAL = "normal"
    ISSUE = "issue"
    UNKNOWN = "unknown"


_CAMERA_AVAILABILITY = {
    "idle": CameraAvailability.ONLINE,
    "recording": CameraAvailability.ONLINE,
    "streaming": CameraAvailability.ONLINE,
    "unavailable": CameraAvailability.OFFLINE,
}
_BINARY_ISSUE_STATES = {
    DiagnosticKind.CONNECTIVITY: {"connectivity": "off"},
    DiagnosticKind.POWER: {"power": "off", "plug": "off", "problem": "on"},
    DiagnosticKind.BATTERY: {"battery": "on"},
    DiagnosticKind.STORAGE: {"problem": "on"},
    DiagnosticKind.DEVICE_ERROR: {"problem": "on"},
}
# A scalar sensor must itself report a categorical condition. For example, a
# battery percentage or free-space measurement alone does not report a fault.
_SENSOR_STATES = {
    DiagnosticKind.CONNECTIVITY: ({"connected", "online"}, {"disconnected", "offline"}),
    DiagnosticKind.POWER: (
        {"powered", "plugged_in"},
        {"unpowered", "unplugged", "power_loss"},
    ),
    DiagnosticKind.BATTERY: ({"normal", "ok"}, {"low", "low_battery"}),
    DiagnosticKind.STORAGE: ({"normal", "ok"}, {"full", "storage_full", "error"}),
    DiagnosticKind.DEVICE_ERROR: ({"normal", "ok"}, {"error", "fault", "problem"}),
}


@dataclass(frozen=True, slots=True)
class CameraDiagnostic:
    """Explicit authorization for one technical entity and category."""

    entity_id: str
    kind: DiagnosticKind

    def __post_init__(self) -> None:
        if (
            type(self.entity_id) is not str
            or _DIAGNOSTIC_ID.fullmatch(self.entity_id) is None
        ):
            raise ValueError("invalid_camera_diagnostic_entity")
        object_id = self.entity_id.split(".", 1)[1]
        if _PRIVATE_SENSOR.search(object_id.replace("_", "")):
            raise ValueError("private_camera_diagnostic_entity")
        if not isinstance(self.kind, DiagnosticKind):
            raise ValueError("invalid_camera_diagnostic_kind")


@dataclass(frozen=True, slots=True)
class HAStateReading:
    """An individual HA state read or state event at its actual observation time."""

    observed_at: datetime | None
    payload: Mapping[str, object] = field(repr=False)


@dataclass(frozen=True, slots=True)
class HAHealthSnapshot:
    """Source status and readings from one provider call."""

    connected: bool
    observed_at: datetime | None
    states: Mapping[str, HAStateReading] = field(default_factory=dict, repr=False)


class CameraHealthProvider(Protocol):
    """Adapter for technical state reads or cached technical state events."""

    async def read_technical_states(
        self, entity_ids: tuple[str, ...]
    ) -> HAHealthSnapshot:
        """Read only the given IDs with real source and per-entity observation times.

        Event adapters that cache readings must retain only selected entity IDs,
        technical state evidence, device classes and observation timestamps.
        """
        ...


@dataclass(frozen=True, slots=True)
class DiagnosticHealth:
    """Projected technical condition for an authorized diagnostic."""

    entity_id: str
    kind: DiagnosticKind
    status: DiagnosticStatus
    observed_at: datetime | None
    stale: bool


@dataclass(frozen=True, slots=True)
class CameraHealth:
    """Projected availability and diagnostic evidence for a selected camera."""

    entity_id: str
    availability: CameraAvailability
    observed_at: datetime | None
    stale: bool
    diagnostics: tuple[DiagnosticHealth, ...]


@dataclass(frozen=True, slots=True)
class CameraHealthReport:
    """Technical evidence with source status independent of camera availability."""

    source_status: SourceStatus
    observed_at: datetime | None
    cameras: tuple[CameraHealth, ...]


class CameraHealthScopeError(ValueError):
    """Requested operation falls outside technical health reads."""


def _observation_time(value: datetime | None, now: datetime) -> datetime | None:
    if type(value) is datetime and value.utcoffset() is not None and value <= now:
        return value.astimezone(UTC)
    return None


def _diagnostic_status(
    diagnostic: CameraDiagnostic, payload: Mapping[str, object]
) -> DiagnosticStatus | None:
    attributes = payload.get("attributes", {})
    if not isinstance(attributes, Mapping):
        return None
    device_class = attributes.get("device_class")
    if device_class is not None and type(device_class) is not str:
        return None

    binary_classes = _BINARY_ISSUE_STATES[diagnostic.kind]
    if diagnostic.entity_id.startswith("binary_sensor."):
        if device_class is None or device_class not in binary_classes:
            return None
        state = payload.get("state")
        if type(state) is not str or state not in {"on", "off"}:
            return DiagnosticStatus.UNKNOWN
        return (
            DiagnosticStatus.ISSUE
            if state == binary_classes[device_class]
            else DiagnosticStatus.NORMAL
        )

    if (
        device_class is not None
        and device_class != "enum"
        and device_class not in binary_classes
    ):
        return None
    state = payload.get("state")
    normal_states, issue_states = _SENSOR_STATES[diagnostic.kind]
    if type(state) is str:
        if state in issue_states:
            return DiagnosticStatus.ISSUE
        if state in normal_states:
            return DiagnosticStatus.NORMAL
    # Generic scalar sensors need a recognized technical condition before
    # their timestamps can be exposed as technical evidence.
    return DiagnosticStatus.UNKNOWN if device_class in binary_classes else None


class CameraHealthMonitor:
    """Project allowlisted HA states without retaining raw HA payloads."""

    def __init__(
        self,
        provider: CameraHealthProvider | None,
        cameras: Mapping[str, Sequence[CameraDiagnostic]],
        *,
        clock: Callable[[], datetime],
        max_staleness_s: float,
    ) -> None:
        """Configure the provider, camera allowlist, UTC-aware clock and evidence TTL."""
        if type(max_staleness_s) not in (int, float) or not 0 < max_staleness_s < float(
            "inf"
        ):
            raise ValueError("invalid_camera_health_staleness")
        self._cameras = {
            entity_id: tuple(diagnostics) for entity_id, diagnostics in cameras.items()
        }
        entity_ids: dict[str, None] = {}
        for entity_id, diagnostics in self._cameras.items():
            if type(entity_id) is not str or _CAMERA_ID.fullmatch(entity_id) is None:
                raise ValueError("invalid_camera_health_entity")
            entity_ids[entity_id] = None
            diagnostic_ids: set[str] = set()
            for diagnostic in diagnostics:
                if (
                    type(diagnostic) is not CameraDiagnostic
                    or diagnostic.entity_id in diagnostic_ids
                ):
                    raise ValueError("invalid_camera_health_diagnostics")
                diagnostic_ids.add(diagnostic.entity_id)
                entity_ids[diagnostic.entity_id] = None
        self._entity_ids = tuple(entity_ids)
        self._provider = provider
        self._clock = clock
        self._max_staleness_s = max_staleness_s

    async def read(
        self, actor: HomeActor, *, operation: str = "health"
    ) -> CameraHealthReport:
        """Return technical health for a current, server-authorized actor.

        Raises:
            CameraHealthScopeError: The operation is outside camera health.
            HomeActionDenied: The actor lacks a current health read scope.
        """
        if type(operation) is not str or operation != "health":
            raise CameraHealthScopeError(
                "Solo puedo consultar el estado técnico de las cámaras. "
                "Las imágenes, capturas, grabaciones y el video quedan fuera de este servicio."
            )
        if not isinstance(actor, HomeActor) or _HEALTH_SCOPES.isdisjoint(
            actor.permissions
        ):
            raise HomeActionDenied("home_read_denied")
        actor.require_current()

        if self._provider is None or not self._entity_ids:
            return self._report(SourceStatus.NOT_CONFIGURED)
        try:
            snapshot = await self._provider.read_technical_states(self._entity_ids)
        except Exception:
            logger.warning("Camera health source read failed")
            snapshot = HAHealthSnapshot(False, None)
        actor.require_current()

        if not isinstance(snapshot, HAHealthSnapshot):
            return self._report(SourceStatus.UNKNOWN)
        if snapshot.connected is not True:
            return self._report(SourceStatus.DISCONNECTED)
        now = self._clock()
        observed_at = _observation_time(snapshot.observed_at, now)
        if observed_at is None:
            return self._report(SourceStatus.UNKNOWN)
        source_status = (
            SourceStatus.STALE
            if self._expired(observed_at, now)
            else SourceStatus.AVAILABLE
        )
        return self._report(source_status, observed_at, now, snapshot.states)

    def _expired(self, observed_at: datetime, now: datetime) -> bool:
        return (now - observed_at).total_seconds() >= self._max_staleness_s

    def _report(
        self,
        source_status: SourceStatus,
        observed_at: datetime | None = None,
        now: datetime | None = None,
        states: Mapping[str, HAStateReading] | None = None,
    ) -> CameraHealthReport:
        def sample(
            entity_id: str,
        ) -> tuple[datetime | None, bool, Mapping[str, object]]:
            if states is None or now is None:
                return None, False, {}
            reading = states.get(entity_id)
            if not isinstance(reading, HAStateReading) or not isinstance(
                reading.payload, Mapping
            ):
                return None, False, {}
            timestamp = _observation_time(reading.observed_at, now)
            payload_id = reading.payload.get("entity_id", entity_id)
            if (
                timestamp is None
                or type(payload_id) is not str
                or payload_id != entity_id
            ):
                return None, False, {}
            stale = source_status is SourceStatus.STALE or self._expired(timestamp, now)
            return timestamp, stale, reading.payload

        cameras = []
        for entity_id, diagnostics in self._cameras.items():
            timestamp, stale, payload = sample(entity_id)
            state = payload.get("state") if not stale else None
            availability = CameraAvailability.UNKNOWN
            if type(state) is str:
                availability = _CAMERA_AVAILABILITY.get(
                    state, CameraAvailability.UNKNOWN
                )

            technical = []
            for diagnostic in diagnostics:
                reading_at, reading_stale, payload = sample(diagnostic.entity_id)
                status = _diagnostic_status(diagnostic, payload)
                if status is None:
                    # An unrelated sensor's event time can also reveal activity.
                    reading_at, reading_stale = None, False
                technical.append(
                    DiagnosticHealth(
                        diagnostic.entity_id,
                        diagnostic.kind,
                        DiagnosticStatus.UNKNOWN
                        if reading_stale or status is None
                        else status,
                        reading_at,
                        reading_stale,
                    )
                )
            cameras.append(
                CameraHealth(
                    entity_id, availability, timestamp, stale, tuple(technical)
                )
            )
        return CameraHealthReport(source_status, observed_at, tuple(cameras))
