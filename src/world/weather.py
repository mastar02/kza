"""Turn Home Assistant weather state into speakable Spanish. Pure, no I/O."""

from __future__ import annotations

# HA condition slugs -> Rioplatense Spanish. Anything unlisted is omitted
# rather than guessed: a wrong condition spoken aloud is worse than silence.
CONDITIONS = {
    "clear-night": "despejado",
    "cloudy": "nublado",
    "exceptional": "condiciones excepcionales",
    "fog": "con niebla",
    "hail": "con granizo",
    "lightning": "con tormenta eléctrica",
    "lightning-rainy": "con tormenta y lluvia",
    "partlycloudy": "parcialmente nublado",
    "pouring": "lloviendo fuerte",
    "rainy": "lluvioso",
    "snowy": "nevando",
    "snowy-rainy": "con aguanieve",
    "sunny": "soleado",
    "windy": "ventoso",
    "windy-variant": "ventoso",
}

UNAVAILABLE = ("unavailable", "unknown", "none")
NO_DATA = "No tengo el dato del clima ahora mismo."


def describe_current(state: dict | None) -> str:
    """One spoken sentence for the current weather, or an honest 'no data'."""
    if not state:
        return NO_DATA
    condition = str(state.get("state") or "").lower()
    if condition in UNAVAILABLE:
        return NO_DATA

    attrs = state.get("attributes") or {}
    parts: list[str] = []

    temp = attrs.get("temperature")
    if temp is not None:
        # Decimals are noise out loud.
        parts.append(f"{round(float(temp))} grados")

    described = CONDITIONS.get(condition)
    if described:
        parts.append(described)

    humidity = attrs.get("humidity")
    if humidity is not None:
        parts.append(f"{round(float(humidity))} por ciento de humedad")

    if not parts:
        return NO_DATA
    return "Hay " + ", ".join(parts) + "."


NO_FORECAST = "No tengo el pronóstico ahora mismo."
_DIA_INDEX = {"hoy": 0, "mañana": 1, "manana": 1}


def describe_forecast(forecast: list[dict], dia: str) -> str:
    """One spoken sentence for a forecast day.

    Never mentions rain probability: `precipitation_probability` comes back
    None from this HA integration, so the condition is the only honest signal.
    """
    index = _DIA_INDEX.get(dia.lower())
    if index is None or not forecast or index >= len(forecast):
        return NO_FORECAST

    day = forecast[index]
    described = CONDITIONS.get(str(day.get("condition") or "").lower())
    low, high = day.get("templow"), day.get("temperature")

    parts: list[str] = []
    if described:
        parts.append(described)
    if low is not None and high is not None:
        parts.append(f"entre {round(float(low))} y {round(float(high))} grados")
    elif high is not None:
        parts.append(f"{round(float(high))} grados")

    if not parts:
        return NO_FORECAST
    return f"{dia.capitalize()}: " + ", ".join(parts) + "."
