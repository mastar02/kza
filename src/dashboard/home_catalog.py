"""Real Home Assistant catalog with explicit room attribution and provenance."""

import math
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

from src.home_assistant.ha_client import HomeAssistantClient, get_home_capabilities
from src.core.observation import Observation
from src.home_assistant.control_policy import HomeControlPolicy
from src.home_assistant.capability_service import HomeCapabilityService
from src.home_assistant.operations import finite_number
from src.home_assistant.capability_projection import HomeControlLimits
from src.home_assistant.capability_projection import selected_capabilities as selected_capabilities
from src.home_assistant.capability_projection import projected_capabilities as projected_capabilities
from src.rooms.room_context import RoomConfig


@dataclass(frozen=True)
class HomeEntity:
    """Reduced state and action capabilities exposed to the home dashboard."""

    entity_id: str
    domain: str
    name: str
    state: str
    brightness_pct: int | None
    room_id: str | None
    area_id: str | None
    capabilities: tuple[str, ...]
    state_changed_at: str | None
    observed_at: datetime | None
    available: bool
    control_limits: HomeControlLimits = HomeControlLimits()
    temperature: float | None = None
    color_temp_kelvin: int | None = None
    position: int | None = None
    observation: Observation | None = None


@dataclass(frozen=True)
class HomeRoom:
    """Configured room and its explicitly reconciled entity references."""

    room_id: str
    name: str
    entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class HomeCatalogSnapshot:
    """Catalog observation whose age is independent of state change times."""

    observed_at: datetime | None
    stale: bool
    rooms: tuple[HomeRoom, ...]
    entities: tuple[HomeEntity, ...]
    unassigned_entities: tuple[HomeEntity, ...]
    reason: str | None = None
    source: str = "real"
    observation: Observation | None = None
    registry_observation: Observation | None = None
    selection_observation: Observation | None = None


def assign_room(
    *, entity_id: str, explicit_room_ids: tuple[str, ...], ha_area_room_id: str | None,
) -> str | None:
    """Resolve explicit references before areas without entity-name heuristics.

    Args:
        entity_id: Entity being reconciled, retained in the public contract.
        explicit_room_ids: All rooms explicitly referencing the entity.
        ha_area_room_id: Room uniquely matched to the entity's registry area.

    Returns:
        The unique explicit room or the area room, and None on explicit conflict.
    """
    matches = set(explicit_room_ids)
    if len(matches) == 1:
        return next(iter(matches))
    if matches:
        return None
    return ha_area_room_id


def _normalize_alias(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return " ".join("".join(char for char in value if not unicodedata.combining(char)).split())


class HomeCatalogService:
    """Compose typed HA observations and configured rooms without synthetic data."""

    def __init__(self, ha_client: HomeAssistantClient | None, rooms: Mapping[str, RoomConfig], *,
                 policy: HomeControlPolicy | None = None, capabilities: HomeCapabilityService | None = None,
                 policy_provider: Callable[[], HomeControlPolicy] | None = None) -> None:
        self._ha_client = ha_client
        self._rooms = dict(rooms)
        self._policy = policy if policy is not None else HomeControlPolicy()
        self._owns_capabilities = capabilities is None
        self.capabilities = capabilities if capabilities is not None else HomeCapabilityService(
            ha_client, self._policy, policy_provider=policy_provider)

    async def snapshot(self) -> HomeCatalogSnapshot:
        """Build one reduced catalog from bounded bulk HA reads.

        Returns:
            A real catalog, marked stale on failed state or registry observations.
        """
        snapshot = await self.capabilities.snapshot()
        now = self.capabilities.now()
        states, areas = snapshot.states, snapshot.areas
        observation = snapshot.observation.as_of(now)
        registry_observation = snapshot.registry_observation.as_of(now)
        grants = {grant.entity_id: grant for grant in snapshot.grants}
        reasons = [reason for reason in (states.reason, areas.reason) if reason]
        if not snapshot.selection_observation.current(now):
            reasons.append("control_selection_unavailable")
        unit = snapshot.temperature_unit
        explicit: dict[str, list[str]] = defaultdict(list)
        alias_rooms: dict[str, set[str]] = defaultdict(set)
        for room_id, room in self._rooms.items():
            for attribute in (
                "default_light", "default_climate", "default_cover", "default_media_player",
                "default_fan", "motion_sensor", "temperature_sensor", "humidity_sensor", "tts_speaker",
            ):
                if entity_id := getattr(room, attribute):
                    explicit[entity_id].append(room_id)
            for alias in (room.name, *room.aliases):
                if normalized := _normalize_alias(alias):
                    alias_rooms[normalized].add(room_id)

        area_rooms: dict[str, str] = {}
        for area_id, name in areas.area_names.items():
            matches = set(alias_rooms.get(_normalize_alias(name), ()))
            if area_id in self._rooms:
                matches.add(area_id)
            if len(matches) == 1:
                area_rooms[area_id] = next(iter(matches))
            elif matches:
                reasons.append(f"area_room_conflict:{area_id}")

        entities = []
        room_entities: dict[str, list[str]] = defaultdict(list)
        for state in states.states:
            entity_id = state["entity_id"]
            area_id = areas.entity_area_ids.get(entity_id)
            explicit_ids = tuple(explicit.get(entity_id, ()))
            room_id = assign_room(
                entity_id=entity_id, explicit_room_ids=explicit_ids,
                ha_area_room_id=area_rooms.get(area_id),
            )
            if len(set(explicit_ids)) > 1:
                reasons.append(f"explicit_room_conflict:{entity_id}")
            capability = snapshot.entity(entity_id)
            capabilities, limits = capability.effective_operations(now), capability.limits
            attributes = state.get("attributes", {})
            brightness = attributes.get("brightness")
            brightness_pct = None
            if (
                "brightness" in get_home_capabilities(state)
                and type(brightness) in (int, float)
                and math.isfinite(brightness)
                and 0 <= brightness <= 255
            ):
                brightness_pct = round(brightness * 100 / 255)
            name = grants[entity_id].display_name if entity_id in grants and grants[entity_id].scene_effects else attributes.get("friendly_name")
            changed_at = state.get("last_changed")
            entity = HomeEntity(
                entity_id=entity_id, domain=entity_id.partition(".")[0],
                name=name if isinstance(name, str) else entity_id,
                state=state["state"], brightness_pct=brightness_pct,
                room_id=room_id, area_id=area_id, capabilities=capabilities,
                state_changed_at=changed_at if isinstance(changed_at, str) else None,
                observed_at=states.observed_at,
                available=capability.available(now), observation=capability.observation.as_of(now),
                control_limits=limits,
                temperature=attributes.get("temperature") if unit == "°C" and finite_number(attributes.get("temperature")) else None,
                color_temp_kelvin=attributes.get("color_temp_kelvin") if type(attributes.get("color_temp_kelvin")) is int else None,
                position=attributes.get("current_position") if type(attributes.get("current_position")) is int else None,
            )
            entities.append(entity)
            if room_id is not None:
                room_entities[room_id].append(entity_id)
        return HomeCatalogSnapshot(
            observed_at=states.observed_at, stale=not observation.current(now),
            observation=observation, registry_observation=registry_observation,
            selection_observation=snapshot.selection_observation.as_of(now),
            rooms=self._room_dtos(room_entities), entities=tuple(entities),
            unassigned_entities=tuple(entity for entity in entities if entity.room_id is None),
            reason=";".join(dict.fromkeys(reasons)) or None,
        )

    def _room_dtos(self, entity_ids: Mapping[str, list[str]]) -> tuple[HomeRoom, ...]:
        return tuple(
            HomeRoom(room_id, room.name, tuple(entity_ids.get(room_id, ())))
            for room_id, room in self._rooms.items()
        )

    async def aclose(self) -> None:
        """Drain a locally created reader; injected shared readers belong to the graph."""
        if self._owns_capabilities:
            await self.capabilities.aclose()
