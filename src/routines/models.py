"""Normalized immutable routine definitions and execution receipts."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import re

from src.home_assistant.operations import HomeOperation as Operation, PARAMETER_FIELDS, action_parameters, validate_parameters


class RoutineValidationError(ValueError):
    """A routine definition does not satisfy the supported contract."""


class RoutineNotFound(LookupError):
    """A requested routine or execution receipt does not exist."""


class RoutineConflict(RuntimeError):
    """An operation conflicts with the current durable state.

    Args:
        code: Public conflict code.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class RunStatus(StrEnum):
    """Public execution receipt states."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


class ActionStatus(StrEnum):
    """Public states of an indexed execution action."""

    PENDING = "pending"
    SENDING = "sending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    NOT_EXECUTED = "not_executed"


@dataclass(frozen=True)
class TimeSchedule:
    """One local wall-clock time and its weekdays (Monday is zero)."""

    at: str
    weekdays: tuple[int, ...]


@dataclass(frozen=True)
class StateCondition:
    """Required state of a Home Assistant entity."""

    entity_id: str
    state: str


@dataclass(frozen=True)
class RoutineAction:
    """A supported Casa operation on an entity."""

    entity_id: str
    operation: Operation
    brightness_pct: int | None = None
    color_temp_kelvin: int | None = None
    temperature: float | None = None
    hvac_mode: str | None = None
    position: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", Operation(self.operation))


@dataclass(frozen=True)
class RoutineSpec:
    """Normalized editable definition shared by web, storage and scheduler."""

    name: str
    description: str
    enabled: bool
    timezone: str
    schedule: TimeSchedule | None
    conditions: tuple[StateCondition, ...]
    actions: tuple[RoutineAction, ...]


@dataclass(frozen=True)
class ScheduledRoutine:
    """Persisted definition with server-owned identity and provenance."""

    routine_id: str
    revision: int
    created_by: str
    created_at: datetime
    spec: RoutineSpec


@dataclass(frozen=True)
class RoutineActionResult:
    """Durable outcome of one indexed snapshot action."""

    index: int
    entity_id: str
    status: ActionStatus
    reason: str | None = None


@dataclass(frozen=True)
class RoutineRun:
    """Idempotent execution receipt retaining its immutable definition."""

    run_id: str
    routine_id: str
    occurrence_key: str
    actor: str
    source: str
    status: RunStatus
    snapshot: ScheduledRoutine
    scheduled_for: datetime | None
    actions: tuple[RoutineActionResult, ...]
    reason: str | None = None


@dataclass(frozen=True)
class ClaimResult:
    """Receipt and whether this claim created it."""

    run: RoutineRun
    created: bool


def _object(value: object, required: set[str], optional: set[str] | None = None) -> dict:
    if type(value) is not dict or not required <= value.keys() or value.keys() - required - (optional or set()):
        raise RoutineValidationError("Invalid or unknown definition fields")
    return value


def _text(value: object, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value.strip()):
        raise RoutineValidationError("Expected a nonempty string")
    return value


def _entity(value: object) -> str:
    value = _text(value)
    if not re.fullmatch(r"[a-z_][a-z0-9_]*\.[a-z0-9_]+", value):
        raise RoutineValidationError("Invalid entity_id")
    return value


def parse_routine_spec(payload: dict) -> RoutineSpec:
    """Validate an exact supported web definition without coercing values.

    Args:
        payload: Editable definition without actor or server metadata.

    Returns:
        An immutable normalized definition.

    Raises:
        RoutineValidationError: If a field is missing, unknown or invalid.
    """
    payload = _object(payload, {"name", "description", "enabled", "timezone", "schedule", "conditions", "actions"})
    name = _text(payload["name"])
    description = _text(payload["description"], nonempty=False)
    if type(payload["enabled"]) is not bool or payload["timezone"] != "America/Montevideo":
        raise RoutineValidationError("Invalid enabled or timezone")
    schedule = None
    if payload["schedule"] is not None:
        item = _object(payload["schedule"], {"at", "weekdays"})
        at = _text(item["at"])
        days = item["weekdays"]
        if not re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", at):
            raise RoutineValidationError("Expected HH:MM")
        if type(days) is not list or not days or any(type(day) is not int or not 0 <= day <= 6 for day in days) or len(set(days)) != len(days):
            raise RoutineValidationError("Invalid weekdays")
        schedule = TimeSchedule(at, tuple(days))
    if type(payload["conditions"]) is not list or type(payload["actions"]) is not list or not payload["actions"]:
        raise RoutineValidationError("Expected conditions and nonempty actions lists")
    conditions = []
    for value in payload["conditions"]:
        item = _object(value, {"entity_id", "state"})
        conditions.append(StateCondition(_entity(item["entity_id"]), _text(item["state"])))
    actions = []
    for value in payload["actions"]:
        item = _object(value, {"entity_id", "operation"}, set(PARAMETER_FIELDS))
        if type(item["operation"]) is not str:
            raise RoutineValidationError("Invalid operation")
        try:
            operation = Operation(item["operation"])
            parameters = {name: item[name] for name in PARAMETER_FIELDS if name in item}
            # Explicit null parameters are not discarded by persistence.
            validate_parameters(_entity(item["entity_id"]), operation, parameters)
            action = RoutineAction(item["entity_id"], operation, **parameters)
        except ValueError as error:
            raise RoutineValidationError(str(error)) from error
        actions.append(action)
    return RoutineSpec(name, description, payload["enabled"], payload["timezone"], schedule, tuple(conditions), tuple(actions))


def spec_to_dict(spec: RoutineSpec) -> dict:
    """Serialize the editable fields using the same strict web shape."""
    return {
        "name": spec.name, "description": spec.description, "enabled": spec.enabled,
        "timezone": spec.timezone,
        "schedule": {"at": spec.schedule.at, "weekdays": list(spec.schedule.weekdays)} if spec.schedule else None,
        "conditions": [{"entity_id": item.entity_id, "state": item.state} for item in spec.conditions],
        "actions": [
            {"entity_id": item.entity_id, "operation": Operation(item.operation).value,
             **action_parameters(item)}
            for item in spec.actions
        ],
    }


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise RoutineValidationError("Datetime must have a timezone")
    return value.astimezone(timezone.utc).isoformat()


def routine_to_dict(routine: ScheduledRoutine) -> dict:
    """Serialize a definition with UTC provenance and editable fields."""
    return {
        "routine_id": routine.routine_id, "revision": routine.revision,
        "created_by": routine.created_by, "created_at": _utc_iso(routine.created_at),
        **spec_to_dict(routine.spec),
    }


def run_to_dict(run: RoutineRun) -> dict:
    """Serialize a durable receipt using public English status strings."""
    return {
        "run_id": run.run_id, "routine_id": run.routine_id,
        "occurrence_key": run.occurrence_key, "actor": run.actor, "source": run.source,
        "status": RunStatus(run.status).value, "snapshot": routine_to_dict(run.snapshot),
        "scheduled_for": _utc_iso(run.scheduled_for) if run.scheduled_for else None,
        "actions": [{"index": item.index, "entity_id": item.entity_id,
                     "status": ActionStatus(item.status).value, "reason": item.reason} for item in run.actions],
        "reason": run.reason,
    }
