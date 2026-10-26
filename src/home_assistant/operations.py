"""Pure contracts for selected controls and their observable HA effects."""

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Protocol


class HomeOperation(StrEnum):
    """Closed operations; no arbitrary HA service may be requested."""

    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    ACTIVATE_SCENE = "activate_scene"
    BRIGHTNESS = "brightness"
    COLOR_TEMP = "color_temp_kelvin"
    SET_TEMPERATURE = "set_temperature"
    SET_HVAC_MODE = "set_hvac_mode"
    OPEN_COVER = "open_cover"
    CLOSE_COVER = "close_cover"
    STOP_COVER = "stop_cover"
    SET_COVER_POSITION = "set_cover_position"


HVAC_MODES = frozenset({"off", "cool", "heat", "heat_cool", "auto", "dry", "fan_only"})
BLIND_CLASSES = frozenset({"blind", "curtain", "shade", "shutter"})
PARAMETER_FIELDS = ("brightness", "brightness_pct", "color_temp_kelvin", "temperature", "hvac_mode", "position")
DOMAIN_OPERATIONS = {
    "scene": frozenset({"activate_scene"}),
    "light": frozenset({"turn_on", "turn_off", "brightness", "color_temp_kelvin"}),
    "switch": frozenset({"turn_on", "turn_off"}),
    "climate": frozenset({"set_temperature", "set_hvac_mode"}),
    "cover": frozenset({"open_cover", "close_cover", "stop_cover", "set_cover_position"}),
}


class ActionIntent(Protocol):
    """Parameters shared by HTTP, voice, storage and action execution."""

    entity_id: str
    operation: str
    brightness_pct: int | None
    brightness: int | None
    color_temp_kelvin: int | None
    temperature: float | None
    hvac_mode: str | None
    position: int | None


def finite_number(value: object) -> bool:
    """Accept finite Python numbers without coercing strings or booleans."""
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def action_parameters(action: ActionIntent) -> dict:
    """Retain every explicitly populated typed parameter."""
    return {name: getattr(action, name) for name in PARAMETER_FIELDS if getattr(action, name, None) is not None}


def validate_parameters(entity_id: str, operation: str, parameters: dict) -> None:
    """Reject wrong-domain operations, unrelated fields and invalid values."""
    domain = entity_id.partition(".")[0]
    if operation not in DOMAIN_OPERATIONS.get(domain, ()):
        raise ValueError("invalid_home_operation")
    required = {
        "brightness": {"brightness_pct"}, "color_temp_kelvin": {"color_temp_kelvin"},
        "set_temperature": {"temperature"}, "set_hvac_mode": {"hvac_mode"},
        "set_cover_position": {"position"},
    }.get(operation, set())
    if operation == "brightness" and "brightness" in parameters:
        required = {"brightness"}
    optional = {"brightness_pct", "brightness"} if operation == "color_temp_kelvin" else set()
    if "brightness" in parameters and "brightness_pct" in parameters:
        raise ValueError("conflicting_brightness")
    if not required <= parameters.keys() or parameters.keys() - required - optional:
        raise ValueError("invalid_action_parameters")
    if "brightness" in parameters and (
        type(parameters["brightness"]) is not int or not 0 <= parameters["brightness"] <= 255
        or (operation == "color_temp_kelvin" and parameters["brightness"] == 0)
    ):
        raise ValueError("invalid_brightness")
    if "brightness_pct" in parameters and (
        type(parameters["brightness_pct"]) is not int or not 0 <= parameters["brightness_pct"] <= 100
        or (operation == "color_temp_kelvin" and parameters["brightness_pct"] == 0)
    ):
        raise ValueError("invalid_brightness")
    if "color_temp_kelvin" in parameters and (type(parameters["color_temp_kelvin"]) is not int or parameters["color_temp_kelvin"] <= 0):
        raise ValueError("invalid_color_temperature")
    if "temperature" in parameters and not finite_number(parameters["temperature"]):
        raise ValueError("invalid_temperature")
    if "hvac_mode" in parameters and (type(parameters["hvac_mode"]) is not str or parameters["hvac_mode"] not in HVAC_MODES):
        raise ValueError("invalid_hvac_mode")
    if "position" in parameters and (type(parameters["position"]) is not int or not 0 <= parameters["position"] <= 100):
        raise ValueError("invalid_cover_position")


def action_capabilities(entity: dict) -> tuple[str, ...]:
    """Derive operations from observed attributes; grants remain independent."""
    entity_id, attributes = entity.get("entity_id"), entity.get("attributes", {})
    if not isinstance(entity_id, str) or not isinstance(attributes, dict):
        return ()
    domain = entity_id.partition(".")[0]
    features = attributes.get("supported_features", 0)
    if type(features) is not int or features < 0:
        return ()
    if domain in {"light", "switch"}:
        result = ("turn_on", "turn_off")
        modes = attributes.get("supported_color_modes") or []
        if domain == "light" and type(modes) is list:
            if any(mode in {"brightness", "color_temp", "hs", "xy", "rgb", "rgbw", "rgbww", "white"} for mode in modes if isinstance(mode, str)):
                result += ("brightness",)
            if "color_temp" in modes:
                result += ("color_temp_kelvin",)
        return result
    if domain == "climate":
        modes = attributes.get("hvac_modes")
        if type(modes) is not list or not modes or any(type(mode) is not str or mode not in HVAC_MODES for mode in modes):
            return ()
        return ("set_hvac_mode", "set_temperature") if features & 1 else ("set_hvac_mode",)
    if domain == "cover" and attributes.get("device_class") in BLIND_CLASSES:
        # HA CoverEntityFeature OPEN=1, CLOSE=2, SET_POSITION=4, STOP=8.
        return tuple(operation for operation, flag in (("open_cover", 1), ("close_cover", 2), ("set_cover_position", 4), ("stop_cover", 8)) if features & flag)
    return ()


def validate_observation(action: ActionIntent, state: dict, *, temperature_unit: str | None = None) -> None:
    """Check current availability, feature flags, bounds and parameter support."""
    if not isinstance(state, dict) or state.get("entity_id") != action.entity_id or action.operation not in action_capabilities(state):
        raise ValueError("entity_not_controllable")
    attributes = state.get("attributes", {})
    if attributes.get("entity_id") is not None:
        raise ValueError("group_effects_unverified")
    domain = action.entity_id.partition(".")[0]
    valid_states = {"light": {"on", "off"}, "switch": {"on", "off"}, "climate": HVAC_MODES, "cover": {"open", "closed", "opening", "closing"}}
    if state.get("state") not in valid_states[domain]:
        raise ValueError("entity_not_controllable")
    if action.color_temp_kelvin is not None:
        low, high = attributes.get("min_color_temp_kelvin"), attributes.get("max_color_temp_kelvin")
        if not finite_number(low) or not finite_number(high) or not 0 < low <= action.color_temp_kelvin <= high:
            raise ValueError("color_temperature_not_supported")
    if action.hvac_mode is not None and action.hvac_mode not in attributes["hvac_modes"]:
        raise ValueError("hvac_mode_not_supported")
    if action.temperature is not None:
        low, high, step = (attributes.get(name) for name in ("min_temp", "max_temp", "target_temp_step"))
        if temperature_unit != "°C" or (attributes.get("temperature_unit") not in (None, "°C")):
            raise ValueError("temperature_unit_not_verified")
        if not finite_number(low) or not finite_number(high) or not low <= action.temperature <= high:
            raise ValueError("temperature_not_supported")
        if step is not None:
            if not finite_number(step) or step <= 0 or not math.isclose((action.temperature - low) / step, round((action.temperature - low) / step), abs_tol=1e-7, rel_tol=0):
                raise ValueError("temperature_step_not_supported")
        elif not float(action.temperature).is_integer():
            raise ValueError("temperature_step_not_verified")


@dataclass(frozen=True)
class HomeEffect:
    """One normalized HA service attempt, without independent retry logic."""

    domain: str
    service: str
    data: dict | None


def compile_effect(action: ActionIntent) -> HomeEffect:
    """Map a validated typed operation to a fixed HA service and exact fields."""
    domain = action.entity_id.partition(".")[0]
    data = action_parameters(action)
    if domain == "scene":
        return HomeEffect(domain, "turn_on", None)
    if domain in {"light", "switch"}:
        service = "turn_off" if action.operation == "turn_off" or (action.brightness_pct == 0 or getattr(action, "brightness", None) == 0) else "turn_on"
        return HomeEffect(domain, service, data if service == "turn_on" and data else None)
    return HomeEffect(domain, str(action.operation), data or None)


def matches_observation(action: ActionIntent, state: dict, *, initial_state: str | None = None) -> bool:
    """Confirm the requested HA state/attributes, never a physical sensor effect."""
    if not isinstance(state, dict) or state.get("entity_id") != action.entity_id:
        return False
    current, attributes = state.get("state"), state.get("attributes", {})
    if not isinstance(attributes, dict):
        return False
    if action.entity_id.startswith(("light.", "switch.")):
        off = action.operation == "turn_off" or (action.brightness_pct == 0 or getattr(action, "brightness", None) == 0)
        if current != ("off" if off else "on"):
            return False
        if not off and action.brightness_pct is not None:
            brightness = attributes.get("brightness")
            if not finite_number(brightness) or abs(brightness - action.brightness_pct * 255 / 100) > 1:
                return False
        if not off and getattr(action, "brightness", None) is not None:
            if type(attributes.get("brightness")) is not int or attributes["brightness"] != action.brightness:
                return False
        return action.color_temp_kelvin is None or (attributes.get("color_mode") == "color_temp" and type(attributes.get("color_temp_kelvin")) is int and attributes["color_temp_kelvin"] == action.color_temp_kelvin)
    if action.operation == "set_hvac_mode":
        return current == action.hvac_mode
    if action.operation == "set_temperature":
        actual = attributes.get("temperature")
        return current == initial_state and current in HVAC_MODES and finite_number(actual) and math.isclose(actual, action.temperature, abs_tol=1e-6, rel_tol=0)
    if action.operation == "stop_cover":
        position = attributes.get("current_position")
        return current in {"open", "closed"} and (position is None or (type(position) is int and 0 <= position <= 100))
    target = {"open_cover": 100, "close_cover": 0}.get(action.operation, action.position)
    if action.operation in {"open_cover", "close_cover"} and "current_position" not in attributes:
        return current == ("closed" if target == 0 else "open")
    actual = attributes.get("current_position")
    return current in {"open", "closed"} and type(actual) is int and actual == target and current == ("closed" if target == 0 else "open")
