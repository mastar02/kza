"""Turn Home Assistant weather state into speakable Spanish. Pure, no I/O."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from enum import StrEnum

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
_DIA_INDEX = {"hoy": 0, "mañana": 1, "manana": 1, "pasado mañana": 2, "pasado manana": 2}

# Zona horaria del hogar para alinear el pronóstico por fecha local. Es el
# mismo valor que `routines.timezone` en config/settings.yaml (único admitido
# hoy por RoutineScheduler); un test falla si divergen. Interino hasta que
# P3-F1 inyecte el reloj del hogar construido desde esa clave.
HOUSEHOLD_TIMEZONE = "America/Montevideo"


class ForecastAlignment(StrEnum):
    """How the requested local day was matched against the forecast entries."""

    ALIGNED = "aligned"
    NO_ENTRIES = "no_entries"
    MISSING_DAY = "missing_day"
    CONFLICTING_DAY = "conflicting_day"
    INVALID_ENTRIES = "invalid_entries"


@dataclass(frozen=True)
class ForecastAnswer:
    """Spoken forecast sentence and the date alignment that produced it.

    Attributes:
        text: Sentence to speak; NO_FORECAST whenever no single day was selected.
        alignment: Outcome of matching the target local date to entry dates.
    """

    text: str
    alignment: ForecastAlignment

    @property
    def aligned(self) -> bool:
        """True only when exactly one entry matched the requested local date."""
        return self.alignment is ForecastAlignment.ALIGNED


def forecast_target_date(dia: str, now: datetime, zone: tzinfo) -> date | None:
    """Local calendar date a spoken horizon refers to.

    Args:
        dia: Spoken horizon ("hoy", "mañana", "pasado mañana").
        now: Timezone-aware instant of the question.
        zone: Household timezone.

    Returns:
        The household local date, or None for an unsupported horizon.

    Raises:
        ValueError: If `now` is naive (its local day would be a guess).
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("forecast target needs an aware clock")
    offset = _DIA_INDEX.get(dia.lower())
    if offset is None:
        return None
    return now.astimezone(zone).date() + timedelta(days=offset)


def _entry_local_date(entry: object, zone: tzinfo) -> date | None:
    """Household local date of one entry, or None if its datetime is unusable.

    Only aware ISO strings count: a naive time has no defined local day, and
    guessing its offset turns every answer into a confident off-by-one.
    """
    if type(entry) is not dict or type(entry.get("datetime")) is not str:
        return None
    try:
        moment = datetime.fromisoformat(entry["datetime"])
        if moment.tzinfo is None or moment.utcoffset() is None:
            return None
        return moment.astimezone(zone).date()
    except (ValueError, OverflowError):
        return None


def _forecast_parts(day: dict) -> tuple[str, ...]:
    """Speakable fragments of one forecast entry (condition and temperatures)."""
    described = CONDITIONS.get(str(day.get("condition") or "").lower())
    low, high = _as_number(day.get("templow")), _as_number(day.get("temperature"))

    parts: list[str] = []
    if described:
        parts.append(described)
    if low is not None and high is not None:
        parts.append(f"entre {round(low)} y {round(high)} grados")
    elif high is not None:
        parts.append(f"{round(high)} grados")
    return tuple(parts)


def describe_forecast(forecast: list[dict] | None, dia: str, target: date | None,
                      zone: tzinfo) -> ForecastAnswer:
    """One spoken sentence for the forecast entry whose local date is `target`.

    The entry is selected by its own `datetime` converted to the household
    timezone, never by its position. Any entry with a missing, non-string,
    unparseable or naive datetime makes the whole payload unusable (it could
    be the requested day). Two entries for the target day that would be
    spoken differently are a conflict. Every such case answers NO_FORECAST.

    Never mentions rain probability: `precipitation_probability` is absent
    from this HA integration's payload, so the condition is the only honest
    signal.

    Cualquier forma inesperada del payload (no-lista, elementos que no son
    dicts, temperaturas no numéricas) degrada a NO_FORECAST hablado. Nunca
    levanta: una excepción acá viaja sin que nadie la atrape hasta el loop de
    audio y el turno queda mudo.

    Args:
        forecast: Daily entries from `weather.get_forecasts`.
        dia: Spoken horizon, used only as the sentence label.
        target: Household local date requested (see `forecast_target_date`).
        zone: Household timezone.

    Returns:
        The sentence and the alignment outcome.
    """
    if not isinstance(forecast, list) or not forecast:
        return ForecastAnswer(NO_FORECAST, ForecastAlignment.NO_ENTRIES)
    matches: list[tuple[str, ...]] = []
    for entry in forecast:
        local_date = _entry_local_date(entry, zone)
        if local_date is None:
            return ForecastAnswer(NO_FORECAST, ForecastAlignment.INVALID_ENTRIES)
        if local_date == target:
            matches.append(_forecast_parts(entry))
    if not matches:
        return ForecastAnswer(NO_FORECAST, ForecastAlignment.MISSING_DAY)
    if len(set(matches)) > 1:
        return ForecastAnswer(NO_FORECAST, ForecastAlignment.CONFLICTING_DAY)

    parts = matches[0]
    if not parts:
        return ForecastAnswer(NO_FORECAST, ForecastAlignment.ALIGNED)
    return ForecastAnswer(f"{dia.capitalize()}: " + ", ".join(parts) + ".", ForecastAlignment.ALIGNED)
