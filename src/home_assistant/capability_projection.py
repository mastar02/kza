"""Pure capability/selection intersection shared by voice and dashboard consumers."""

import math
from dataclasses import dataclass

from src.home_assistant.ha_client import get_home_capabilities
from src.home_assistant.control_policy import ActionOrigin, EntityGrant, HomeActor, HomeControlPolicy
from src.home_assistant.operations import action_parameters, finite_number, validate_observation
from src.home_assistant.home_actions import HomeActionRequest
from src.home_assistant.effect_plans import expand_effect_plan, validate_plan_references


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


def selected_capabilities(state: dict, grant: EntityGrant | None, unit: str | None, *, allow_group: bool = False, allow_internal: bool = False) -> tuple[tuple[str, ...], HomeControlLimits]:
    """Project controls without exposing an implicit grant for a whole domain."""
    attributes = state.get("attributes", {})
    if grant is None or (not grant.direct_control and not allow_internal) or (attributes.get("entity_id") is not None and not allow_group):
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


def projected_capabilities(state: dict, grants: dict[str, EntityGrant], states: dict[str, dict], unit: str | None) -> tuple[tuple[str, ...], HomeControlLimits]:
    """Intersect reviewed expansions with every current leaf, never group averages."""
    grant = grants.get(state["entity_id"])
    if grant is not None and not grant.direct_control:
        return (), HomeControlLimits()
    if grant is None or (not grant.members and not grant.scene_effects):
        return selected_capabilities(state, grant, unit)
    root = HomeActionRequest(grant.entity_id, "activate_scene" if grant.scene_effects else "turn_on", "catalog", origin=ActionOrigin.VOICE)
    try:
        plan = expand_effect_plan(root, tuple(grants.values()))
        validate_plan_references(root, plan, states)
        if grant.scene_effects:
            HomeControlPolicy.require_from_grants(root, HomeActor.household(), tuple(grants.values()))
            for intent in (*plan.intermediates, *plan.effects):
                request = HomeActionRequest(intent.entity_id, intent.operation, "catalog", origin=ActionOrigin.VOICE,
                    **action_parameters(intent))
                HomeControlPolicy.require_from_grants(request, HomeActor.household(), tuple(grants.values()), via_expansion=True)
            for intent in plan.effects:
                validate_observation(intent, states.get(intent.entity_id), temperature_unit=unit)
            return ("activate_scene",), HomeControlLimits()
        ids = {grant.entity_id, *(reference.entity_id for reference in plan.groups), *(intent.entity_id for intent in plan.effects)}
        observations = []
        for entity_id in sorted(ids):
            current = states.get(entity_id)
            if current is None or current.get("state") not in {"on", "off"}:
                return (), HomeControlLimits()
            observations.append(selected_capabilities(current, grants.get(entity_id), unit, allow_group=bool(grants[entity_id].members), allow_internal=True))
        common = set.intersection(*(set(operations) for operations, _ in observations))
        values = {}
        for operation, low_key, high_key in (("brightness", "brightness_min", "brightness_max"), ("color_temp_kelvin", "color_temp_min", "color_temp_max")):
            if operation in common:
                low = max(getattr(limits, low_key) for _, limits in observations)
                high = min(getattr(limits, high_key) for _, limits in observations)
                if low > high:
                    common.remove(operation)
                else:
                    values.update({low_key: low, high_key: high})
        return tuple(operation for operation in ("turn_on", "turn_off", "brightness", "color_temp_kelvin") if operation in common), HomeControlLimits(**values)
    except (ValueError, KeyError, TypeError, AttributeError):
        return (), HomeControlLimits()

