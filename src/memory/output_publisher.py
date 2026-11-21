"""Metadata-only publisher grammar and journal handle for the T024 matrix."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import re
from typing import TYPE_CHECKING

from src.memory import output_registration as base

if TYPE_CHECKING:
    from src.memory.output_registration_v2 import OutputRegistrationV2


# The on-disk catalogue already matches next-census-design-v2.json in order.
ALL_MATRIX_PRODUCERS = base.PRODUCERS
PLAN_KINDS = frozenset({
    "sqlite", "file_tree", "atomic_json", "transport_session", "resource_only",
    "external_effect", "declared_root", "readonly_input",
})
_LABEL = re.compile(r"[a-z][a-z0-9_]{0,63}")
_PRIVATE_FIELDS = frozenset({
    "audio", "embedding", "embeddings", "pcm", "samples", "transcript",
    "content", "payload", "repr", "headers", "authorization", "api_key", "private_key",
})
_PRIVATE_LABELS = frozenset({
    "credential", "credentials", "password", "secret", "secrets", "token", "tokens",
})


class PublisherPhase(StrEnum):
    """Durable observations of the original owner's lifecycle."""

    INTENT = "intent"
    ATTACHED = "attached"
    OPENING = "opening"
    DRAINED = "drained"
    DISABLED = "disabled"


_PREVIOUS_PHASE = {
    PublisherPhase.INTENT: None,
    PublisherPhase.ATTACHED: PublisherPhase.INTENT,
    PublisherPhase.OPENING: PublisherPhase.ATTACHED,
    PublisherPhase.DRAINED: PublisherPhase.OPENING,
    PublisherPhase.DISABLED: None,
}
EventDecoder = Callable[[dict, dict | None, bool, dict], dict]
PlanValidator = Callable[[dict, dict], None]
AggregateValidator = Callable[[dict, dict, dict], None]


@dataclass(frozen=True, eq=False, slots=True)
class GrammarSpec:
    """A fixed producer codec supplied by the owning family.

    A decoder validates family metadata and returns the canonical state from
    ``publisher_event_shape``. Use ``publisher_grammar`` for fixed role sets.
    """

    grammar_id: str
    producer: str
    event_kind: str
    decode_event: EventDecoder

    def __reduce_ex__(self, protocol: int) -> None:
        raise TypeError("output_journal_grammar_not_serializable")


def _roles_shape(roles: object) -> None:
    base._require(type(roles) is list and 0 < len(roles) <= base.MAX_RECORD_BYTES
        and all(type(role) is str and _LABEL.fullmatch(role) for role in roles)
        and roles == sorted(set(roles)), "publisher_roles")


def snapshot_metadata(value: object) -> dict:
    """Detach bounded journal metadata before yielding to owned disk work.

    Accept only concrete JSON data types used by the journal. Family validators
    must also pin the meaning and exact shape of their extra fields.
    """
    remaining = base.MAX_RECORD_BYTES

    def copy(item: object, depth: int, *, roles: bool = False):
        nonlocal remaining
        remaining -= 1
        base._require(depth <= 12 and remaining >= 0, "publisher_metadata")
        if roles:
            _roles_shape(item)
            remaining -= sum(map(len, item))
            base._require(remaining >= 0, "publisher_metadata")
            return list(item)
        if type(item) is dict:
            result = {}
            for key, child in item.items():
                base._require(type(key) is str and _LABEL.fullmatch(key)
                    and key not in _PRIVATE_FIELDS and _PRIVATE_LABELS.isdisjoint(key.split("_")),
                    "publisher_metadata")
                remaining -= len(key)
                result[key] = copy(child, depth + 1, roles=depth == 0 and key == "roles")
            return result
        if type(item) is str:
            remaining -= len(item)
            base._require(remaining >= 0, "publisher_metadata")
            return item
        base._require(item is None or type(item) is bool
                      or type(item) is int and 0 <= item < 2**64, "publisher_metadata")
        return item

    base._require(type(value) is dict, "publisher_metadata")
    return copy(value, 0)


def publisher_plan(kind: str, *, roles: tuple[str, ...] = (), **metadata: object) -> dict:
    """Build a detached plan with role IDs in canonical order."""
    base._require(type(kind) is str and kind in PLAN_KINDS, "publisher_plan")
    base._require(type(roles) is tuple and all(type(role) is str for role in roles), "publisher_roles")
    plan = {"kind": kind, **metadata}
    if roles:
        plan["roles"] = sorted(roles)
    return snapshot_metadata(plan)


def publisher_event_shape(event: dict, prior: dict | None, selected: bool, *,
                          producer: str, event_kind: str) -> dict:
    """Validate a phase transition and return its detached journal state."""
    base._require(type(event) is dict and type(event.get("kind")) is str
                  and event["kind"] == event_kind, "publisher_event")
    phase = event.get("phase")
    prior_phase = None if prior is None else prior["phase"]
    base._require(type(phase) is str and phase in _PREVIOUS_PHASE
                  and _PREVIOUS_PHASE[phase] == prior_phase, "transition")
    base._require(selected is (phase != PublisherPhase.DISABLED), "publisher_selection")
    fields = {"kind", "phase", "plan"}
    if phase == PublisherPhase.DRAINED:
        fields.add("aggregate")
    base._require(set(event) == fields, "publisher_event")
    plan = snapshot_metadata(event["plan"])
    kind = plan.get("kind")
    base._require(type(kind) is str and kind in PLAN_KINDS, "publisher_plan")
    base._require(prior is None or base._encoded(prior["plan"]) == base._encoded(plan), "plan_changed")
    aggregate = snapshot_metadata(event["aggregate"]) if phase == PublisherPhase.DRAINED else None
    return {"phase": phase, "plan": plan, "producer": producer, "aggregate": aggregate}


def publisher_grammar(producer: str, *, roles: tuple[str, ...] = (),
                      plan_shape: PlanValidator | None = None,
                      aggregate_shape: AggregateValidator | None = None) -> GrammarSpec:
    """Build a /1 grammar with fixed roles and closed family metadata shapes.

    Args:
        producer: Matrix row ID.
        roles: Fixed owner IDs for this row, identical at every phase.
        plan_shape: Optional validator for the exact plan keys and values.
            It receives the plan and epoch header, including for disabled rows.
        aggregate_shape: Optional validator for the exact terminal aggregate.
            It receives the aggregate, plan and epoch header.
    """
    base._require(type(producer) is str and producer in ALL_MATRIX_PRODUCERS, "publisher_producer")
    role_ids = tuple(publisher_plan("resource_only", roles=roles).get("roles", ()))
    event_kind = f"{producer}_publisher"

    def decode(event: dict, prior: dict | None, selected: bool, header: dict) -> dict:
        state = publisher_event_shape(event, prior, selected, producer=producer, event_kind=event_kind)
        plan, aggregate = state["plan"], state["aggregate"]
        base._require(tuple(plan.get("roles", ())) == role_ids, "publisher_roles")
        if plan_shape is None:
            base._require(set(plan) <= {"kind", "roles"}, "publisher_plan")
        else:
            plan_shape(plan, header)
        if aggregate is not None:
            if aggregate_shape is None:
                base._require(not aggregate, "publisher_aggregate")
            else:
                aggregate_shape(aggregate, plan, header)
        return state

    return GrammarSpec(f"{event_kind}/1", producer, event_kind, decode)


@dataclass(frozen=True, eq=False, slots=True)
class GenericOutputPublisher:
    """Record one producer's observations through its coordinator.

    The original owner opens and drains its resources under its own authority.
    Callers await the journal observation at each lifecycle boundary.
    """

    _coordinator: OutputRegistrationV2
    producer: str

    async def attached(self) -> None:
        """Record the original owner's attachment."""
        await self._coordinator.publisher_attached(self.producer)

    async def opening(self) -> None:
        """Record opening before requesting work from the original owner."""
        await self._coordinator.publisher_opening(self.producer)

    async def drained(self, aggregate: dict) -> None:
        """Record the aggregate after the original owner has drained."""
        await self._coordinator.publisher_drained(self.producer, aggregate)
