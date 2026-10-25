"""Mandatory control selection and server-owned action provenance."""

import asyncio
from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

from src.home_assistant.operations import BLIND_CLASSES, DOMAIN_OPERATIONS, HVAC_MODES, finite_number

if TYPE_CHECKING:
    from src.home_assistant.home_actions import HomeActionRequest


class ActorKind(StrEnum):
    """Authentication evidence; acoustic identity never supplies this value."""

    DASHBOARD = "dashboard"
    HOUSEHOLD = "household"


class ActionOrigin(StrEnum):
    """Producer of a typed action, independent of any speaker annotation."""

    HTTP = "http"
    VOICE = "voice"
    ROUTINE = "routine"
    REMINDER = "reminder"


@dataclass(frozen=True)
class HomeActor:
    """A principal constructed at a trusted server boundary, never from text."""

    kind: ActorKind
    subject: str
    permissions: frozenset[str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, ActorKind) or type(self.subject) is not str or not self.subject
            or type(self.permissions) is not frozenset
            or any(type(value) is not str for value in self.permissions)
            or (self.kind is ActorKind.HOUSEHOLD and self.subject != "household")
        ):
            raise ValueError("invalid_actor")

    @classmethod
    def household(cls) -> "HomeActor":
        """Return the shared unverified household actor used by voice/jobs."""
        return cls(ActorKind.HOUSEHOLD, "household", frozenset({"home:control"}))

    @property
    def receipt_identity(self) -> str:
        """Namespace receipts by authentication kind as well as subject."""
        return f"{self.kind.value}:{self.subject}"


class HomeActionDenied(ValueError):
    """The current control selection does not authorize the requested effect."""


@dataclass(frozen=True)
class EntityGrant:
    """Explicit operations and limits for one verified leaf entity."""

    entity_id: str
    operations: frozenset[str]
    brightness_min: int = 0
    brightness_max: int = 100
    color_temp_min: int | None = None
    color_temp_max: int | None = None
    temperature_min: float | None = None
    temperature_max: float | None = None
    temperature_unit: str = "C"
    hvac_modes: frozenset[str] = frozenset()
    position_min: int = 0
    position_max: int = 100
    cover_device_class: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.entity_id) is not str
            or re.fullmatch(r"(?:light|switch|climate|cover)\.[a-z0-9_]+", self.entity_id) is None
            or type(self.operations) is not frozenset or not self.operations
            or not self.operations <= DOMAIN_OPERATIONS.get(self.entity_id.partition(".")[0], ())
            or type(self.brightness_min) is not int or type(self.brightness_max) is not int
            or not 0 <= self.brightness_min <= self.brightness_max <= 100
        ):
            raise ValueError("invalid_entity_grant")
        if "color_temp_kelvin" in self.operations and (
            type(self.color_temp_min) is not int or type(self.color_temp_max) is not int
            or not 0 < self.color_temp_min <= self.color_temp_max
        ):
            raise ValueError("invalid_color_temperature_grant")
        if "set_temperature" in self.operations and (
            not finite_number(self.temperature_min) or not finite_number(self.temperature_max)
            or self.temperature_min > self.temperature_max or self.temperature_unit != "C"
        ):
            raise ValueError("invalid_temperature_grant")
        if type(self.hvac_modes) is not frozenset or ("set_hvac_mode" in self.operations and (
            not self.hvac_modes or not self.hvac_modes <= HVAC_MODES
        )):
            raise ValueError("invalid_hvac_modes_grant")
        if self.entity_id.startswith("cover.") and (
            self.cover_device_class not in BLIND_CLASSES or type(self.position_min) is not int
            or type(self.position_max) is not int or not 0 <= self.position_min <= self.position_max <= 100
        ):
            raise ValueError("invalid_cover_grant")


class HomeControlPolicy:
    """Fail closed without selection; re-read private grants before each send.

    In-memory grants are explicit constructor dependencies for controlled
    environments. The application supplies only a private inventory-review path.
    No prior successful read is reused after a missing or malformed file.
    """

    def __init__(self, grants: tuple[EntityGrant, ...] = (), *, selection_path: str | None = None):
        if type(grants) is not tuple or any(not isinstance(grant, EntityGrant) for grant in grants):
            raise ValueError("invalid_control_selection")
        if len({grant.entity_id for grant in grants}) != len(grants):
            raise ValueError("duplicate_control_entity")
        if grants and selection_path:
            raise ValueError("ambiguous_control_selection")
        self._grants = grants
        self._selection_path = Path(selection_path) if selection_path else None

    def _read_selection(self) -> tuple[EntityGrant, ...]:
        if self._selection_path is None:
            return self._grants
        try:
            with self._selection_path.open("rb") as stream:
                raw = stream.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("selection_too_large")
            review = json.loads(raw)
            if review.get("selection_approved") is not True or review.get("live_api_verified") is not True:
                return ()
            entries = review["allowed_entities"]
            if type(entries) is not list:
                raise ValueError("invalid_selection")
            grants = []
            for entry in entries:
                if entry.get("live_verified") is not True:
                    continue
                operations = entry["operations"]
                if type(operations) is not list or any(type(value) is not str for value in operations):
                    raise ValueError("invalid_operations")
                grants.append(grant_from_review(entry))
            if len({grant.entity_id for grant in grants}) != len(grants):
                raise ValueError("duplicate_control_entity")
            return tuple(grants)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            raise HomeActionDenied("control_selection_unavailable") from error

    async def require_allowed(self, request: "HomeActionRequest", actor: HomeActor) -> EntityGrant:
        """Check actor, producer, selected entity, operation and current limits."""
        if not isinstance(actor, HomeActor) or "home:control" not in actor.permissions:
            raise HomeActionDenied("home_control_denied")
        if (
            not isinstance(request.origin, ActionOrigin)
            or (request.origin is ActionOrigin.HTTP) != (actor.kind is ActorKind.DASHBOARD)
        ):
            raise HomeActionDenied("action_provenance_denied")
        grants = await self.snapshot_grants()
        grant = next((item for item in grants if item.entity_id == request.entity_id), None)
        if grant is None or request.operation not in grant.operations:
            raise HomeActionDenied("action_not_selected")
        if request.brightness_pct is not None and (
            "brightness" not in grant.operations or not grant.brightness_min <= request.brightness_pct <= grant.brightness_max
        ):
            raise HomeActionDenied("brightness_outside_selected_range")
        if request.color_temp_kelvin is not None and not grant.color_temp_min <= request.color_temp_kelvin <= grant.color_temp_max:
            raise HomeActionDenied("color_temperature_outside_selected_range")
        if request.temperature is not None and not grant.temperature_min <= request.temperature <= grant.temperature_max:
            raise HomeActionDenied("temperature_outside_selected_range")
        if request.hvac_mode is not None and request.hvac_mode not in grant.hvac_modes:
            raise HomeActionDenied("hvac_mode_not_selected")
        target = {"open_cover": 100, "close_cover": 0}.get(request.operation, request.position)
        if request.entity_id.startswith("cover.") and target is not None and not grant.position_min <= target <= grant.position_max:
            raise HomeActionDenied("position_outside_selected_range")
        return grant

    async def snapshot_grants(self) -> tuple[EntityGrant, ...]:
        """Read current reviewed grants for projection, without freezing admission."""
        return await asyncio.to_thread(self._read_selection) if self._selection_path else self._grants



def grant_from_review(entry: dict) -> EntityGrant:
    """Parse explicit per-operation limits from private reviewed inventory data."""
    parameters = entry.get("parameters", {})
    if type(parameters) is not dict or parameters.keys() - {"brightness_pct", "color_temp_kelvin", "temperature", "hvac_modes", "position", "device_class"}:
        raise ValueError("invalid_parameters")
    operations = frozenset(entry["operations"])

    def bounds(key):
        value = parameters[key]
        if type(value) is not dict or value.keys() != {"min", "max"}:
            raise ValueError("invalid_parameter_bounds")
        return value["min"], value["max"]

    fields = {}
    if "brightness" in operations:
        fields["brightness_min"], fields["brightness_max"] = bounds("brightness_pct")
    if "color_temp_kelvin" in operations:
        fields["color_temp_min"], fields["color_temp_max"] = bounds("color_temp_kelvin")
    if "set_temperature" in operations:
        temperature = parameters["temperature"]
        if type(temperature) is not dict or temperature.keys() != {"min", "max", "unit", "unit_confirmed_by_user"} or temperature["unit_confirmed_by_user"] is not True:
            raise ValueError("temperature_unit_not_verified")
        fields.update(temperature_min=temperature["min"], temperature_max=temperature["max"], temperature_unit=temperature["unit"])
    if "set_hvac_mode" in operations:
        modes = parameters["hvac_modes"]
        if type(modes) is not list or any(type(mode) is not str for mode in modes):
            raise ValueError("invalid_hvac_modes")
        fields["hvac_modes"] = frozenset(modes)
    if entry["entity_id"].startswith("cover."):
        fields["cover_device_class"] = parameters["device_class"]
        if "set_cover_position" in operations or "position" in parameters:
            fields["position_min"], fields["position_max"] = bounds("position")
    return EntityGrant(entry["entity_id"], operations, **fields)
