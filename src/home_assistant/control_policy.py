"""Mandatory control selection and server-owned action provenance."""

import asyncio
from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

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

    def __post_init__(self) -> None:
        if (
            type(self.entity_id) is not str
            or re.fullmatch(r"(?:light|switch)\.[a-z0-9_]+", self.entity_id) is None
            or type(self.operations) is not frozenset or not self.operations
            or not self.operations <= {"turn_on", "turn_off", "brightness"}
            or ("brightness" in self.operations and not self.entity_id.startswith("light."))
            or type(self.brightness_min) is not int or type(self.brightness_max) is not int
            or not 0 <= self.brightness_min <= self.brightness_max <= 100
        ):
            raise ValueError("invalid_entity_grant")


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
                parameters = entry.get("parameters", {})
                if type(parameters) is not dict or parameters.keys() - {"brightness_pct"}:
                    raise ValueError("invalid_parameters")
                bounds = parameters["brightness_pct"] if "brightness" in operations else {"min": 0, "max": 100}
                if type(bounds) is not dict or bounds.keys() != {"min", "max"}:
                    raise ValueError("invalid_brightness_bounds")
                grants.append(EntityGrant(entry["entity_id"], frozenset(operations), bounds["min"], bounds["max"]))
            if len({grant.entity_id for grant in grants}) != len(grants):
                raise ValueError("duplicate_control_entity")
            return tuple(grants)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            raise HomeActionDenied("control_selection_unavailable") from error

    async def require_allowed(self, request: "HomeActionRequest", actor: HomeActor) -> None:
        """Check actor, producer, selected entity, operation and current limits."""
        if not isinstance(actor, HomeActor) or "home:control" not in actor.permissions:
            raise HomeActionDenied("home_control_denied")
        if (
            not isinstance(request.origin, ActionOrigin)
            or (request.origin is ActionOrigin.HTTP) != (actor.kind is ActorKind.DASHBOARD)
        ):
            raise HomeActionDenied("action_provenance_denied")
        grants = await asyncio.to_thread(self._read_selection) if self._selection_path else self._grants
        grant = next((item for item in grants if item.entity_id == request.entity_id), None)
        if grant is None or request.operation not in grant.operations:
            raise HomeActionDenied("action_not_selected")
        if request.brightness_pct is not None and not grant.brightness_min <= request.brightness_pct <= grant.brightness_max:
            raise HomeActionDenied("brightness_outside_selected_range")
