"""Independent operation and response evidence for domestic request receipts.

Completion refers to the operation's actual boundary, not recognition, queue
admission or generated acknowledgment. It does not certify physical effects,
durability of volatile objects, or the roadmap's overall acceptance criteria.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextvars import ContextVar
from enum import Enum

from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult, ResponseStage

CURRENT_REQUEST_ID: ContextVar[str | None] = ContextVar("task_request_id", default=None)


class TaskStatus(str, Enum):
    """Request handling and domain completion, independent of speech."""

    UNKNOWN = "unknown"
    RECEIVED = "received"
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    UNSUPPORTED = "unsupported"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    CANCELLED = "cancelled"


class PersistenceState(str, Enum):
    """Evidence for object storage; immediate HA receipts are not durable."""

    UNKNOWN = "unknown"
    NOT_REQUIRED = "not_required"
    NOT_WRITTEN = "not_written"
    VOLATILE = "volatile"
    COMMITTED = "committed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class DispatchState(str, Enum):
    """What the transport actually established about sending an operation."""

    UNKNOWN = "unknown"
    NOT_REQUIRED = "not_required"
    NOT_SENT = "not_sent"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    PARTIAL = "partial"


class EffectState(str, Enum):
    """Observed software/HA effect, never inferred physical completion."""

    UNKNOWN = "unknown"
    NOT_REQUIRED = "not_required"
    NOT_OBSERVED = "not_observed"
    OBSERVED = "observed"
    UNCONFIRMED = "unconfirmed"
    UNCERTAIN = "uncertain"
    PARTIAL = "partial"


class ResponseRequirement(str, Enum):
    """A silent operation has no fabricated completed spoken response."""

    REQUIRED = "required"
    SILENT = "silent"


@dataclass(frozen=True)
class TaskOutcome:
    """Domain evidence that remains true even when confirmation audio fails."""

    status: TaskStatus
    handled: bool = True
    persistence: PersistenceState = PersistenceState.NOT_REQUIRED
    dispatch: DispatchState = DispatchState.NOT_REQUIRED
    effect: EffectState = EffectState.NOT_REQUIRED
    observation_scope: str | None = None
    replication: DispatchState = DispatchState.NOT_REQUIRED

    def __post_init__(self) -> None:
        if (not isinstance(self.status, TaskStatus) or type(self.handled) is not bool
                or not isinstance(self.persistence, PersistenceState)
                or not isinstance(self.dispatch, DispatchState)
                or not isinstance(self.effect, EffectState)
                or not isinstance(self.replication, DispatchState)):
            raise ValueError("invalid_task_evidence_type")
        if self.observation_scope not in {None, "software_state", "ha_state"}:
            raise ValueError("invalid_observation_scope")
        if self.status is TaskStatus.COMPLETED and not (
            self.handled
            and self.persistence in {PersistenceState.NOT_REQUIRED, PersistenceState.VOLATILE, PersistenceState.COMMITTED}
            and self.dispatch in {DispatchState.NOT_REQUIRED, DispatchState.ACCEPTED}
            and self.effect in {EffectState.NOT_REQUIRED, EffectState.OBSERVED}
            and (self.effect is not EffectState.OBSERVED or self.observation_scope is not None)
        ):
            raise ValueError("completion_without_required_evidence")

    @property
    def task_completed(self) -> bool:
        """Whether the domain operation completed, separate from its response."""
        return self.status is TaskStatus.COMPLETED

    def to_dict(self) -> dict:
        """Return bounded, content-free evidence suitable for live projection."""
        return {key: value.value if isinstance(value, Enum) else value
                for key, value in asdict(self).items()}


@dataclass(frozen=True)
class DeliveryOutcome:
    """Retain delivery evidence without retaining text or provider exceptions."""

    generation: GenerationState | None = None
    delivery: DeliveryState | None = None
    error: bool = False
    timeout_stage: ResponseStage | None = None
    spoken_segments: int = 0

    @classmethod
    def from_response(cls, response: ResponseResult) -> DeliveryOutcome:
        """Project actual response ownership into content-free evidence."""
        return cls(response.generation, response.delivery, response.error is not None,
                   response.timeout_stage, response.spoken_segments)

    @property
    def response_completed(self) -> bool:
        """Whether generation and delivery both completed without error."""
        return (self.generation is GenerationState.COMPLETE
                and self.delivery is DeliveryState.COMPLETE
                and not self.error and self.timeout_stage is None)

    def to_dict(self) -> dict:
        """Serialize only closed status fields, never audio/text/identity."""
        return {key: value.value if isinstance(value, Enum) else value
                for key, value in asdict(self).items()}


@dataclass
class TurnOutcome:
    """Per-turn recorder shared with owned child tasks, never between requests."""

    task: TaskOutcome | None = None
    response: DeliveryOutcome | None = None
    response_requirement: ResponseRequirement = ResponseRequirement.REQUIRED
    request_id: str | None = None
    capture_meta: dict = field(default_factory=dict, repr=False)

    def record_response(self, response: ResponseResult) -> None:
        """Record actual voice completion without keeping its private text."""
        self.record_delivery(DeliveryOutcome.from_response(response))

    def record_delivery(self, response: DeliveryOutcome) -> None:
        """A later success cannot erase an earlier partial/failed response."""
        old = self.response
        if old is None:
            self.response = response
        else:
            self.response = DeliveryOutcome(
                response.generation if old.generation is GenerationState.COMPLETE else old.generation,
                response.delivery if old.delivery is DeliveryState.COMPLETE else old.delivery,
                old.error or response.error, old.timeout_stage or response.timeout_stage,
                old.spoken_segments + response.spoken_segments,
            )

    def to_dict(self) -> dict:
        """Compose operation, response and overall turn completion truthfully."""
        task = self.task or TaskOutcome(TaskStatus.UNKNOWN, handled=False)
        response_completed = (self.response.response_completed if self.response is not None else False)
        silent = self.response_requirement is ResponseRequirement.SILENT
        completed = task.task_completed and (silent or response_completed)
        return {
            "handled": task.handled,
            "task_completed": task.task_completed,
            "response_required": not silent,
            "response_completed": None if silent else response_completed,
            "turn_completed": completed,
            "success": completed,
            "outcome": {**task.to_dict(), "response_requirement": self.response_requirement.value,
                        "response": self.response.to_dict() if self.response is not None else None},
        }


OUTCOME_FLAGS = frozenset({"handled", "task_completed", "response_required", "response_completed", "turn_completed"})
OUTCOME_CODES = {
    "task_status": frozenset(item.value for item in TaskStatus),
    "persistence_state": frozenset(item.value for item in PersistenceState),
    "dispatch_state": frozenset(item.value for item in DispatchState),
    "effect_state": frozenset(item.value for item in EffectState),
    "replication_state": frozenset(item.value for item in DispatchState),
    "observation_scope": frozenset({"software_state", "ha_state"}),
    "response_requirement": frozenset(item.value for item in ResponseRequirement),
    "generation_state": frozenset(item.value for item in GenerationState),
    "delivery_state": frozenset(item.value for item in DeliveryState),
}


def project_turn_outcome(result: dict) -> dict:
    """Project only bounded status fields, omitting receipts, text and IDs."""
    if type(result) is not dict:
        return {}
    projected = {key: result[key] for key in OUTCOME_FLAGS if type(result.get(key)) is bool}
    task = result.get("outcome")
    task = task if type(task) is dict else {}
    response = task.get("response")
    response = response if type(response) is dict else {}
    selected = {"task_status": task.get("status"), "persistence_state": task.get("persistence"),
                "dispatch_state": task.get("dispatch"), "effect_state": task.get("effect"),
                "replication_state": task.get("replication"), "observation_scope": task.get("observation_scope"),
                "response_requirement": task.get("response_requirement"),
                "generation_state": response.get("generation"), "delivery_state": response.get("delivery")}
    for key, allowed in OUTCOME_CODES.items():
        value = result.get(key, selected.get(key))
        if type(value) is str and value in allowed:
            projected[key] = value
    return projected
