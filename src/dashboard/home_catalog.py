"""Real Home Assistant catalog with explicit room attribution and provenance."""

import asyncio
import math
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from src.home_assistant.ha_client import HomeAssistantClient, get_home_capabilities
from src.home_assistant.control_policy import EntityGrant, HomeActionDenied, HomeControlPolicy
from src.home_assistant.operations import finite_number
from src.rooms.room_context import RoomConfig


@dataclass(frozen=True)
class HomeControlLimits:
    """Intersection of reviewed limits and currently observed capabilities."""

    brightness_min: int | None = None
    brightness_max: int | None = None
    color_temp_min: int | None = None
    color_temp_max: int | None = None
    temperature_min: float | None = None
    temperature_max: float | None = None
    temperature_step: float | None = None
    hvac_modes: tuple[str, ...] = ()
    position_min: int | None = None
    position_max: int | None = None


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

    def __init__(self, ha_client: HomeAssistantClient | None, rooms: Mapping[str, RoomConfig], *, policy: HomeControlPolicy | None = None):
        self._ha_client = ha_client
        self._rooms = dict(rooms)
        self._policy = policy if policy is not None else HomeControlPolicy()

    async def snapshot(self) -> HomeCatalogSnapshot:
        """Build one reduced catalog from bounded bulk HA reads.

        Returns:
            A real catalog, marked stale on failed state or registry observations.
        """
        if self._ha_client is None:
            return HomeCatalogSnapshot(None, True, self._room_dtos({}), (), (), "ha_unavailable")

        states, areas = await asyncio.gather(
            self._ha_client.get_state_snapshot(), self._ha_client.get_area_assignments(),
        )
        reasons = [reason for reason in (states.reason, areas.reason) if reason]
        try:
            grants = {grant.entity_id: grant for grant in await self._policy.snapshot_grants()}
        except HomeActionDenied:
            grants = {}
            reasons.append("control_selection_unavailable")
        unit = await self._ha_client.get_temperature_unit_fresh() if any("set_temperature" in grant.operations for grant in grants.values()) else None
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
            capabilities, limits = selected_capabilities(state, grants.get(entity_id), unit)
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
            name = attributes.get("friendly_name")
            changed_at = state.get("last_changed")
            entity = HomeEntity(
                entity_id=entity_id, domain=entity_id.partition(".")[0],
                name=name if isinstance(name, str) else entity_id,
                state=state["state"], brightness_pct=brightness_pct,
                room_id=room_id, area_id=area_id, capabilities=capabilities,
                state_changed_at=changed_at if isinstance(changed_at, str) else None,
                observed_at=states.observed_at,
                available=states.available and state["state"] not in {"unknown", "unavailable"},
                control_limits=limits,
                temperature=attributes.get("temperature") if unit == "°C" and finite_number(attributes.get("temperature")) else None,
                color_temp_kelvin=attributes.get("color_temp_kelvin") if type(attributes.get("color_temp_kelvin")) is int else None,
                position=attributes.get("current_position") if type(attributes.get("current_position")) is int else None,
            )
            entities.append(entity)
            if room_id is not None:
                room_entities[room_id].append(entity_id)
        return HomeCatalogSnapshot(
            observed_at=states.observed_at, stale=not states.available or not areas.available,
            rooms=self._room_dtos(room_entities), entities=tuple(entities),
            unassigned_entities=tuple(entity for entity in entities if entity.room_id is None),
            reason=";".join(dict.fromkeys(reasons)) or None,
        )

    def _room_dtos(self, entity_ids: Mapping[str, list[str]]) -> tuple[HomeRoom, ...]:
        return tuple(
            HomeRoom(room_id, room.name, tuple(entity_ids.get(room_id, ())))
            for room_id, room in self._rooms.items()
        )


def selected_capabilities(state: dict, grant: EntityGrant | None, unit: str | None) -> tuple[tuple[str, ...], HomeControlLimits]:
    """Project controls without exposing an implicit grant for a whole domain."""
    attributes = state.get("attributes", {})
    if grant is None or attributes.get("entity_id") is not None:
        return (), HomeControlLimits()
    capabilities = set(get_home_capabilities(state)) & grant.operations
    values = {}
    if "brightness" in capabilities:
        values.update(brightness_min=grant.brightness_min, brightness_max=grant.brightness_max)
    if "color_temp_kelvin" in capabilities:
        low, high = attributes.get("min_color_temp_kelvin"), attributes.get("max_color_temp_kelvin")
        if not finite_number(low) or not finite_number(high) or max(low, grant.color_temp_min) > min(high, grant.color_temp_max):
            capabilities.remove("color_temp_kelvin")
        else:
            values.update(color_temp_min=max(low, grant.color_temp_min), color_temp_max=min(high, grant.color_temp_max))
    if "set_temperature" in capabilities:
        low, high, step = (attributes.get(name) for name in ("min_temp", "max_temp", "target_temp_step"))
        if unit != "°C" or attributes.get("temperature_unit") not in (None, "°C") or not finite_number(low) or not finite_number(high) or max(low, grant.temperature_min) > min(high, grant.temperature_max) or (step is not None and (not finite_number(step) or step <= 0)):
            capabilities.remove("set_temperature")
        else:
            selected_low, selected_high = max(low, grant.temperature_min), min(high, grant.temperature_max)
            if step is not None:
                selected_low = low + math.ceil((selected_low - low) / step - 1e-9) * step
                selected_high = low + math.floor((selected_high - low) / step + 1e-9) * step
            else:
                selected_low, selected_high = math.ceil(selected_low), math.floor(selected_high)
            if selected_low > selected_high:
                capabilities.remove("set_temperature")
            else:
                values.update(temperature_min=selected_low, temperature_max=selected_high, temperature_step=step if step is not None else 1)
    if "set_hvac_mode" in capabilities:
        modes = tuple(mode for mode in attributes["hvac_modes"] if mode in grant.hvac_modes)
        if not modes:
            capabilities.remove("set_hvac_mode")
        values["hvac_modes"] = modes
    if state["entity_id"].startswith("cover."):
        if attributes.get("device_class") != grant.cover_device_class:
            return (), HomeControlLimits()
        values.update(position_min=grant.position_min, position_max=grant.position_max)
        if grant.position_max < 100:
            capabilities.discard("open_cover")
        if grant.position_min > 0:
            capabilities.discard("close_cover")
    return tuple(operation for operation in get_home_capabilities(state) if operation in capabilities), HomeControlLimits(**values)
