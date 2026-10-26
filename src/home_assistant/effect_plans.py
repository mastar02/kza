"""Reviewed lighting effects expanded into explicit, bounded leaf intents."""

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import TYPE_CHECKING

from src.home_assistant.operations import ActionIntent, action_parameters, validate_parameters

if TYPE_CHECKING:
    from src.home_assistant.control_policy import EntityGrant


MAX_EFFECTS = 128
MAX_GROUP_DEPTH = 8


class GroupMembershipSource(StrEnum):
    """Authority for the pinned definition, never a macro transport choice."""

    HA_STATE = "ha_state"
    REVIEWED_COPY = "reviewed_copy"


@dataclass(frozen=True)
class EffectIntent:
    """One exact requested leaf effect, with no arbitrary service or selector."""

    entity_id: str
    operation: str
    brightness_pct: int | None = None
    color_temp_kelvin: int | None = None
    temperature: float | None = None
    hvac_mode: str | None = None
    position: int | None = None
    brightness: int | None = None

    def __post_init__(self) -> None:
        if type(self.entity_id) is not str or not re.fullmatch(r"(?:light|switch|climate|cover)\.[a-z0-9_]+", self.entity_id):
            raise ValueError("unsupported_effect_target")
        validate_parameters(self.entity_id, self.operation, action_parameters(self))


@dataclass(frozen=True)
class GroupReference:
    """Membership that must still match fresh HA observations before sending."""

    entity_id: str
    members: tuple[str, ...]
    membership_source: GroupMembershipSource = GroupMembershipSource.HA_STATE


@dataclass(frozen=True)
class EffectPlan:
    """A frozen expanded intent; overlapping identical effects execute once."""

    effects: tuple[EffectIntent, ...]
    groups: tuple[GroupReference, ...] = ()
    scene_targets: tuple[str, ...] | None = None
    intermediates: tuple[EffectIntent, ...] = ()


def scene_effects_from_states(states: dict) -> tuple[EffectIntent, ...]:
    """Compile strict reviewed lighting states without rounding brightness.

    Unsupported attributes, templates, sensitive targets and implicit defaults
    are rejected. This describes KZA's reviewed copy, never a call to HA's
    mutable scene, script or automation service.
    """
    if type(states) is not dict or not 0 < len(states) <= MAX_EFFECTS:
        raise ValueError("invalid_scene_states")
    effects = []
    for entity_id, desired in states.items():
        if type(entity_id) is not str or not re.fullmatch(r"light\.[a-z0-9_]+", entity_id):
            raise ValueError("unsupported_scene_target")
        if type(desired) is not dict or "state" not in desired or desired.keys() - {"state", "brightness", "color_temp_kelvin"}:
            raise ValueError("unsupported_scene_parameters")
        state = desired["state"]
        if type(state) is not str or state not in {"on", "off"}:
            raise ValueError("invalid_scene_state")
        parameters = {key: value for key, value in desired.items() if key != "state"}
        if state == "off" and parameters:
            raise ValueError("unsupported_off_scene_parameters")
        if state == "on" and parameters.get("brightness") == 0:
            raise ValueError("contradictory_scene_state")
        operation = "turn_off" if state == "off" else "color_temp_kelvin" if "color_temp_kelvin" in parameters else "brightness" if "brightness" in parameters else "turn_on"
        effects.append(EffectIntent(entity_id, operation, **parameters))
    return tuple(effects)


def expand_effect_plan(request: ActionIntent, grants: tuple["EntityGrant", ...]) -> EffectPlan:
    """Expand reviewed groups/scenes; leaves must be separately selected.

    The caller validates every intermediate/leaf against the same policy and
    current capabilities. Cycles, conflicting overlap and excessive expansion
    fail before effects; a member outside the selection is never silently lost.
    """
    by_id = {grant.entity_id: grant for grant in grants}
    effects: dict[str, EffectIntent] = {}
    references: dict[str, GroupReference] = {}
    intermediates = []
    visited = 0

    def visit(intent: EffectIntent, path: tuple[str, ...]) -> None:
        nonlocal visited
        visited += 1
        if visited > MAX_EFFECTS * 4:
            raise ValueError("too_many_effect_references")
        if intent.entity_id in path or len(path) > MAX_GROUP_DEPTH:
            raise ValueError("cyclic_or_deep_group")
        grant = by_id.get(intent.entity_id)
        if grant is None:
            raise ValueError("effect_not_selected")
        if grant.members:
            reference = GroupReference(intent.entity_id, grant.members, grant.membership_source)
            references[intent.entity_id] = reference
            intermediates.append(intent)
            for member in grant.members:
                visit(EffectIntent(member, intent.operation, **action_parameters(intent)), (*path, intent.entity_id))
        else:
            previous = effects.get(intent.entity_id)
            if previous is not None and previous != intent:
                raise ValueError("conflicting_scene_effects")
            effects[intent.entity_id] = intent
            if len(effects) > MAX_EFFECTS:
                raise ValueError("too_many_effects")

    if request.entity_id.startswith("scene."):
        grant = by_id.get(request.entity_id)
        if grant is None or not grant.scene_effects:
            raise ValueError("scene_effects_unverified")
        targets = tuple(effect.entity_id for effect in grant.scene_effects)
        for effect in grant.scene_effects:
            visit(effect, ())
    else:
        targets = None
        visit(EffectIntent(request.entity_id, request.operation, **action_parameters(request)), ())
    return EffectPlan(tuple(effects.values()), tuple(sorted(references.values(), key=lambda item: item.entity_id)), targets, tuple(intermediates))


def validate_plan_references(request: ActionIntent, plan: EffectPlan, states: dict[str, dict]) -> None:
    """Require exact fresh group/scene membership; aggregate state proves no leaf."""
    references = list(plan.groups)
    if plan.scene_targets is not None:
        references.append(GroupReference(request.entity_id, plan.scene_targets))
    for reference in references:
        state = states.get(reference.entity_id)
        if type(state) is not dict or state.get("entity_id") != reference.entity_id or type(state.get("state")) is not str or state["state"] == "unavailable":
            raise ValueError("effect_reference_unavailable")
        if reference.entity_id.startswith("light.") and state["state"] not in {"on", "off"}:
            raise ValueError("effect_reference_unavailable")
        attributes = state.get("attributes", {})
        if type(attributes) is not dict:
            raise ValueError("effect_reference_unavailable")
        members = attributes.get("entity_id")
        if members is None and reference.membership_source is GroupMembershipSource.REVIEWED_COPY:
            # Some bridge groups expose no membership in HA. Their explicitly
            # reviewed copy is the definition KZA executes. The bridge group's
            # mutable service is never called; no hidden member is forwarded to.
            continue
        if type(members) is not list or any(type(member) is not str for member in members) or len(members) != len(set(members)) or set(members) != set(reference.members):
            raise ValueError("effect_membership_changed")
