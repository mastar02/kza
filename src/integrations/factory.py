"""Compose Home Assistant integrations from explicit household configuration."""

import asyncio
import logging
from contextlib import suppress
from contextvars import Context
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.core.household_clock import HouseholdClock
from src.core.settings_schema import IntegrationSettings
from src.core.sqlite_work import AuthorityGuard
from src.home_assistant.ha_client import HomeAssistantClient
from src.integrations.calendar_reader import CalendarReader, CalendarScope, HACalendarSource, ICSCalendarSource
from src.integrations.camera_health import (
    CameraDiagnostic, CameraHealthMonitor, DiagnosticKind, HAHealthSnapshot, HAStateReading, technical_camera_state,
)
from src.integrations.energy_monitor import EnergyMonitor, EnergySensor, MeasurementKind
from src.integrations.media_control import MediaController, MediaSelection, RemoteGrant
from src.lists.list_manager import ListManager

logger = logging.getLogger(__name__)
_LIST_SYNC_TIMEOUT_S = 15.0


class CameraHealthProvider:
    """Read only selected IDs and project technical evidence at receipt time."""

    def __init__(self, ha_client: HomeAssistantClient) -> None:
        self._ha = ha_client

    async def read_technical_states(self, entity_ids: tuple[str, ...]) -> HAHealthSnapshot:
        """Project selected state reads without transition times or media attributes."""
        observations = await asyncio.gather(*(self._ha.read_entity_observation(entity) for entity in entity_ids))
        states = {}
        received_at = []
        for entity, observation in zip(entity_ids, observations, strict=True):
            if observation.entity_id != entity or observation.observed_at is None or observation.reason is not None:
                continue
            timestamp = datetime.fromtimestamp(observation.observed_at, UTC)
            received_at.append(timestamp)
            state = observation.state
            if observation.exists is not True or state is None:
                continue
            # The camera's last_changed/last_updated can reveal recording transitions.
            if entity.startswith("camera."):
                payload = {"state": technical_camera_state(state["state"])}
            else:
                payload = {"state": state["state"], "attributes": {"device_class": state["attributes"].get("device_class")}}
            states[entity] = HAStateReading(timestamp, payload)
        return HAHealthSnapshot(bool(received_at), max(received_at, default=None), states)


@dataclass(frozen=True, slots=True)
class IntegrationServices:
    """One shared set of explicitly configured household integrations."""

    energy_monitor: EnergyMonitor
    calendar_reader: CalendarReader
    media_controller: MediaController
    camera_health: CameraHealthMonitor


def build_integrations(
    config: dict, ha_client: HomeAssistantClient, household_clock: HouseholdClock, *, state_root: Path | None,
) -> IntegrationServices:
    """Build the integrations without contacting Home Assistant or opening files."""
    settings = IntegrationSettings.model_validate(config)
    calendar = settings.calendar
    configured_root = Path(calendar.data_root)
    data_dir = (state_root / "data" if state_root is not None else Path("data")).absolute()
    if not configured_root.is_absolute():
        configured_root = data_dir.joinpath(*configured_root.parts[1:]) if (
            configured_root.parts and configured_root.parts[0] == "data"
        ) else data_dir / configured_root
    data_root = configured_root.resolve()
    if not data_root.is_relative_to(data_dir.resolve()) or (
        state_root is not None and not data_root.is_relative_to(state_root)
    ):
        raise ValueError("calendar_data_root_outside_data_directory")

    sources = [
        ICSCalendarSource(CalendarScope(item.scope), Path(item.path), owner_user_id=item.owner_user_id)
        for item in calendar.ics_sources
    ] + [
        HACalendarSource(CalendarScope(item.scope), item.entity_id, owner_user_id=item.owner_user_id)
        for item in calendar.ha_sources
    ]

    def epoch() -> float:
        return household_clock.now_utc().timestamp()

    return IntegrationServices(
        energy_monitor=EnergyMonitor(
            ha_client,
            (EnergySensor(item.entity_id, MeasurementKind(item.kind), item.room, item.label)
             for item in settings.energy.sensors),
            max_staleness_s=settings.energy.max_staleness_s, clock=epoch,
        ),
        calendar_reader=CalendarReader(
            sources, data_root=data_root, timezone=household_clock.timezone, clock=household_clock.now_utc,
            ha_client=ha_client, ha_timeout=calendar.ha_timeout_s,
        ),
        media_controller=MediaController(
            ha_client,
            (MediaSelection(item.room, item.entity_id, RemoteGrant(**item.remote.model_dump()) if item.remote else None)
             for item in settings.media.players),
            max_staleness_s=settings.media.max_staleness_s, clock=epoch,
        ),
        camera_health=CameraHealthMonitor(
            CameraHealthProvider(ha_client),
            {entity: tuple(CameraDiagnostic(item.entity_id, DiagnosticKind(item.kind)) for item in entries)
             for entity, entries in settings.camera_health.cameras.items()},
            clock=household_clock.now_utc, max_staleness_s=settings.camera_health.max_staleness_s,
        ),
    )


class HouseholdListReconnect:
    """Own at most one shared-list reconciliation after WebSocket recovery."""

    def __init__(self, lists: ListManager) -> None:
        self._lists = lists
        self._active = True
        self._pending = False
        self._task: asyncio.Task[None] | None = None

    def connected(self) -> None:
        """Schedule reconciliation without delaying the HA connection owner."""
        if not self._active:
            return
        self._pending = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._sync(), name="ha-shared-list-reconcile", context=Context())

    async def _sync(self) -> None:
        guard: AuthorityGuard = self._current
        while self._active and self._pending:
            self._pending = False
            try:
                async with asyncio.timeout(_LIST_SYNC_TIMEOUT_S):
                    await self._lists.sync_with_ha(guard=guard)
            except TimeoutError:
                logger.warning("Shared list reconciliation timed out after HA connection")
            except Exception as error:
                logger.warning("Shared list reconciliation failed after HA connection: %s", type(error).__name__)

    def _current(self) -> bool:
        return self._active

    async def aclose(self) -> None:
        """Retire household background authority and drain before stores close."""
        self._active = False
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
