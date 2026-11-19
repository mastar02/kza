"""Turn Home Assistant weather state into speakable Spanish. Pure, no I/O."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from enum import StrEnum

from src.world.household_time import WEEKDAYS, normalize

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
# T037-U2: a horizon the forecast cannot serve is refused without reading HA,
# and never answered with the current conditions.
NO_FORECAST_HORIZON = ("No tengo ese pronóstico. Puedo decirte el de hoy, mañana, pasado mañana "
                       "o un día de la semana.")
ONE_DAY_AT_A_TIME = "Preguntame por un día a la vez."
# Room temperatures are household state (T037-U4), not the outdoor weather entity.
INDOOR_TEMPERATURE_UNSUPPORTED = "Todavía no puedo decirte la temperatura de adentro de la casa, solo la de afuera."
_DIA_INDEX = {"hoy": 0, "mañana": 1, "manana": 1, "pasado mañana": 2, "pasado manana": 2}
_OFFSET_LABEL = {0: "hoy", 1: "mañana", 2: "pasado mañana"}
_WEEKDAY_INDEX = {normalize(name): index for index, name in enumerate(WEEKDAYS)}
_MONTHS = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
_WEEKDAY_WORDS = "|".join(normalize(name) for name in WEEKDAYS)
# Horizons a daily forecast keyed by date cannot answer honestly: ranges,
# weeks, months, past days and calendar dates (normalized text). A past
# weekday ("el sábado pasado") is past, not the coming one (review T037-U2-RV1);
# "pasado mañana" after a weekday is still a future day.
_UNSUPPORTED_HORIZON_RE = re.compile(
    r"\b(?:la semana|esta semana|proxima semana|semana pasada|fin(?:es)? de semana|finde"
    r"|mes que viene|proximo mes|este mes|mes pasado|ano que viene|proximo ano|ano pasado"
    r"|proximos dias|estos dias|dias que vienen|ayer|anoche|anteayer"
    rf"|(?:{_WEEKDAY_WORDS}) pasado(?! manana)"
    # Fix verification T037-U2-FV2: "el pasado sábado", "la pasada semana".
    rf"|pasad[oa] (?:{_WEEKDAY_WORDS}|semana|mes|ano)"
    r"|(?:en|dentro de) (?:\d+|un|una|dos|tres|cuatro|cinco|seis|siete|diez|quince) (?:dias|semanas?)"
    # Calendar dates (review T037-U2-RV3): "el 25", "el lunes 17", "el 1ro".
    # Fix verification T037-U2-FV2: "el día 25", "el primero de diciembre", "en diciembre".
    rf"|el \d+|\d+ de (?:{_MONTHS})|(?:{_WEEKDAY_WORDS}) \d+|\d+(?:ro|ero|er)"
    rf"|dia \d+|(?:en|de) (?:{_MONTHS}))\b"
)
# Future-tense cues: with no day word, or with today's, the question is about
# today's forecast entry rather than the current conditions (user answer Q23).
_FORECAST_CUE_RE = re.compile(r"\b(?:va|van) a\b|\bpronostico\b|\b(?:llovera|hara|habra|estara)\b")
# "mañana" after these words is the morning ("esta/por la/a la mañana"), not tomorrow.
_MORNING_PREVIOUS = frozenset({"esta", "la", "pasado"})
_TODAY_PERIODS = frozenset({"manana", "tarde", "noche"})
# Question shapes about a temperature (normalized text). "qué temperatura hace
# afuera" matches too: the location decides indoor or outdoor.
_TEMPERATURE_QUESTION_RE = re.compile(
    r"\b(?:que|cual es la|cuanta) temperatura\b|\bcuantos grados\b"
    r"|\b(?:temperatura|grados) (?:hay|hace|tiene|marca)\b|\besta la temperatura\b"
    # T037-U2b: the future form routed to weather must still meet the indoor check.
    r"|\bva a ser la temperatura\b"
)
# A room right after "afuera/fuera" ("afuera de la cocina") is outside, not
# the room (review T037-U2-RV5).
_ROOM_PLACE_RE = re.compile(r"(?<!fuera )\b(?:en|del|de|tiene)(?: (?:el|la|los|las))? (\w+)")
_HOUSE_RE = re.compile(r"\b(?:adentro|dentro|en casa|en la casa|de la casa)\b")
_OUTSIDE_RE = re.compile(r"\b(?:afuera|fuera|en la calle)\b")
# Indoor places named without being a configured room or a grammar alias.
_GENERIC_INDOOR_PLACES = frozenset({"pieza", "comedor", "lavadero", "dormitorio", "cuarto", "habitacion"})


class HorizonKind(StrEnum):
    """What a weather question asks for."""

    CURRENT = "current"
    FORECAST = "forecast"
    UNSUPPORTED = "unsupported"
    SEVERAL_DAYS = "several_days"


@dataclass(frozen=True)
class WeatherHorizon:
    """The day a weather question refers to (T037-U2).

    Attributes:
        kind: Current conditions, one forecast day, an unsupported horizon or
            several different days.
        offset: Days from the household's today for a named day ("hoy",
            "mañana", "pasado mañana"); None otherwise.
        weekday: Monday-based weekday for a named weekday; None otherwise.
        next_week: The weekday was said with "que viene" or "próximo", so the
            same weekday as today means a week from today.
    """

    kind: HorizonKind
    offset: int | None = None
    weekday: int | None = None
    next_week: bool = False

    def spoken_day(self, days_ahead: int) -> str:
        """Spoken label of the resolved day, e.g. "mañana" or "el sábado".

        Args:
            days_ahead: Days from the household's today to the resolved date.
        """
        if self.weekday is None:
            return _OFFSET_LABEL.get(days_ahead, "ese día")
        if days_ahead == 0:
            return "hoy"
        return f"el {WEEKDAYS[self.weekday]}" + (" que viene" if days_ahead == 7 else "")


def parse_weather_horizon(text: str) -> WeatherHorizon:
    """Day a weather question refers to, from the words only (no clock).

    Rules (T037-U2, user answer Q23):

    * "esta/por la/a la mañana", "esta tarde", "esta noche" and "hoy" are today.
    * "mañana" and "pasado mañana" are named days; "mañana por la mañana" is tomorrow.
    * A weekday ("el sábado", "el sábado que viene") is a forecast day by date.
    * Weeks, months, ranges, past days and calendar dates are unsupported.
    * Today (said or implied) is current conditions unless the question is in
      the future ("va a llover", "pronóstico"), which asks for today's entry.
    * Two different days in one question are several days.

    Args:
        text: Utterance already routed to the weather path.

    Returns:
        The horizon; never raises.
    """
    words = normalize(text).split()
    joined = " ".join(words)
    if _UNSUPPORTED_HORIZON_RE.search(joined):
        return WeatherHorizon(HorizonKind.UNSUPPORTED)
    days: set[tuple] = set()
    for i, word in enumerate(words):
        previous = words[i - 1] if i else ""
        following = words[i + 1] if i + 1 < len(words) else ""
        if word == "pasado" and following == "manana":
            days.add(("offset", 2))
        elif word == "manana" and previous not in _MORNING_PREVIOUS:
            days.add(("offset", 1))
        elif word == "hoy" or (word == "esta" and following in _TODAY_PERIODS):
            days.add(("offset", 0))
        elif word in _WEEKDAY_INDEX:
            after = words[i + 1:i + 3]
            next_week = previous == "proximo" or after == ["que", "viene"] or after[:1] == ["proximo"]
            days.add(("weekday", _WEEKDAY_INDEX[word], next_week))
    if len(days) > 1:
        return WeatherHorizon(HorizonKind.SEVERAL_DAYS)
    day = next(iter(days), ("offset", 0))
    if day[0] == "weekday":
        return WeatherHorizon(HorizonKind.FORECAST, weekday=day[1], next_week=day[2])
    if day[1] == 0 and not _FORECAST_CUE_RE.search(joined):
        return WeatherHorizon(HorizonKind.CURRENT)
    return WeatherHorizon(HorizonKind.FORECAST, offset=day[1])


def is_indoor_temperature_question(text: str, room_words: Iterable[str] = ()) -> bool:
    """True for a temperature question about a room or the inside of the house.

    The outdoor weather entity never answers it (T037-U2): "qué temperatura hay
    en el living" is household state for T037-U4. A question about outside
    ("qué temperatura hace afuera") stays weather.

    Args:
        text: Utterance in any case, accents or punctuation.
        room_words: Room ids and aliases (configured rooms plus the grammar's
            room lexicon), compared accent-free.
    """
    plain = normalize(text)
    if not _TEMPERATURE_QUESTION_RE.search(plain):
        return False
    rooms = {normalize(word) for word in room_words if isinstance(word, str)} | _GENERIC_INDOOR_PLACES
    if any(match.group(1) in rooms for match in _ROOM_PLACE_RE.finditer(plain)):
        return True
    return bool(_HOUSE_RE.search(plain)) and not _OUTSIDE_RE.search(plain)


def no_forecast_for(label: str) -> str:
    """Honest refusal naming the requested day (user answer PQ-T037-1)."""
    return f"No tengo el pronóstico para {label}."

# La zona horaria del hogar para alinear el pronóstico por fecha local llega
# inyectada: HouseholdClock.zone (src/core/household_clock.py, P3-F1), construido
# una vez en main desde `routines.timezone`.


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
        text: Sentence to speak; a refusal (NO_FORECAST, or the day-named one
            for a missing or conflicting day) whenever no single day was selected.
        alignment: Outcome of matching the target local date to entry dates.
    """

    text: str
    alignment: ForecastAlignment

    @property
    def aligned(self) -> bool:
        """True only when exactly one entry matched the requested local date."""
        return self.alignment is ForecastAlignment.ALIGNED


def forecast_target_date(dia: str | WeatherHorizon, now: datetime, zone: tzinfo) -> date | None:
    """Local calendar date a spoken horizon refers to.

    A weekday is the next date with that weekday, today included; with "que
    viene" or "próximo" today's weekday means a week from today.

    Args:
        dia: Spoken horizon ("hoy", "mañana", "pasado mañana") or a parsed
            forecast horizon (see `parse_weather_horizon`).
        now: Timezone-aware instant of the question.
        zone: Household timezone.

    Returns:
        The household local date, or None for an unsupported horizon.

    Raises:
        ValueError: If `now` is naive (its local day would be a guess).
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("forecast target needs an aware clock")
    today = now.astimezone(zone).date()
    if isinstance(dia, WeatherHorizon):
        if dia.kind is not HorizonKind.FORECAST:
            return None
        if dia.weekday is not None:
            ahead = (dia.weekday - today.weekday()) % 7
            return today + timedelta(days=7 if ahead == 0 and dia.next_week else ahead)
        offset = dia.offset
    else:
        offset = _DIA_INDEX.get(dia.lower())
    if offset is None:
        return None
    return today + timedelta(days=offset)


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
    spoken differently are a conflict. A missing or conflicting day names the
    requested day ("No tengo el pronóstico para mañana.", user answer
    PQ-T037-1); an empty or invalid payload answers the generic NO_FORECAST.

    Never mentions rain probability: `precipitation_probability` is absent
    from this HA integration's payload, so the condition is the only honest
    signal.

    Cualquier forma inesperada del payload (no-lista, elementos que no son
    dicts, temperaturas no numéricas) degrada a NO_FORECAST hablado. Nunca
    levanta: una excepción acá viaja sin que nadie la atrape hasta el loop de
    audio y el turno queda mudo.

    Args:
        forecast: Daily entries from `weather.get_forecasts`.
        dia: Spoken day label ("mañana", "el sábado"), used only in the sentence.
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
        return ForecastAnswer(no_forecast_for(dia), ForecastAlignment.MISSING_DAY)
    if len(set(matches)) > 1:
        return ForecastAnswer(no_forecast_for(dia), ForecastAlignment.CONFLICTING_DAY)

    parts = matches[0]
    if not parts:
        return ForecastAnswer(NO_FORECAST, ForecastAlignment.ALIGNED)
    return ForecastAnswer(f"{dia.capitalize()}: " + ", ".join(parts) + ".", ForecastAlignment.ALIGNED)
