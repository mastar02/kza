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

# Entidad de HA por defecto. Vive acá (y no como literal repetido en el
# dispatcher, el orquestador y main.py) para que el override de
# `home_assistant.weather_entity` tenga UN solo default que contradecir.
DEFAULT_ENTITY = "weather.forecast_home"


def _as_number(value: object) -> float | None:
    """Convertir a float, o None si el valor no es numérico.

    HA puede mandar `"unknown"`, `""` o un dict donde se espera un número.
    Un `float()` pelado ahí levanta ValueError/TypeError y —al no atraparla
    nadie hasta cinco capas más arriba— deja el turno de voz MUDO. Devolver
    None hace que el dato simplemente se omita de la frase.

    `bool` se descarta a propósito: `float(True) == 1.0` se locutaría como
    "1 grados".
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def describe_current(state: dict | None) -> str:
    """One spoken sentence for the current weather, or an honest 'no data'."""
    if not isinstance(state, dict) or not state:
        return NO_DATA
    condition = str(state.get("state") or "").lower()
    if condition in UNAVAILABLE:
        return NO_DATA

    attrs = state.get("attributes")
    if not isinstance(attrs, dict):
        attrs = {}
    parts: list[str] = []

    temp = _as_number(attrs.get("temperature"))
    if temp is not None:
        # Decimals are noise out loud.
        parts.append(f"{round(temp)} grados")

    described = CONDITIONS.get(condition)
    if described:
        parts.append(described)

    humidity = _as_number(attrs.get("humidity"))
    if humidity is not None:
        parts.append(f"{round(humidity)} por ciento de humedad")

    if not parts:
        return NO_DATA
    return "Hay " + ", ".join(parts) + "."


NO_FORECAST = "No tengo el pronóstico ahora mismo."
_DIA_INDEX = {"hoy": 0, "mañana": 1, "manana": 1}


def describe_forecast(forecast: list[dict] | None, dia: str) -> str:
    """One spoken sentence for a forecast day.

    Never mentions rain probability: `precipitation_probability` is absent
    from this HA integration's payload, so the condition is the only honest
    signal.

    Cualquier forma inesperada del payload (no-lista, elementos que no son
    dicts, temperaturas no numéricas) degrada a NO_FORECAST hablado. Nunca
    levanta: una excepción acá viaja sin que nadie la atrape hasta el loop de
    audio y el turno queda mudo.
    """
    index = _DIA_INDEX.get(dia.lower())
    if index is None or not isinstance(forecast, list) or index >= len(forecast):
        return NO_FORECAST

    day = forecast[index]
    if not isinstance(day, dict):
        return NO_FORECAST
    described = CONDITIONS.get(str(day.get("condition") or "").lower())
    low, high = _as_number(day.get("templow")), _as_number(day.get("temperature"))

    parts: list[str] = []
    if described:
        parts.append(described)
    if low is not None and high is not None:
        parts.append(f"entre {round(low)} y {round(high)} grados")
    elif high is not None:
        parts.append(f"{round(high)} grados")

    if not parts:
        return NO_FORECAST
    return f"{dia.capitalize()}: " + ", ".join(parts) + "."
