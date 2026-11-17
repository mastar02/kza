"""Rioplatense voice reminder requests: reminder verb, time, recurrence and zone.

Pure parser (stdlib only, no I/O, no wiring yet). It turns a spoken request such
as "recordame sacar la basura mañana a las 8 de la noche" into a
:class:`VoiceReminderRequest` with the reminder text, the trigger time as a UTC
epoch, an optional recurrence string accepted by
:func:`src.reminders.recurrence.parse_recurrence`, the spoken zone phrase and,
when the request cannot be resolved without guessing, an ambiguity code.

Rules (user answers Q7/Q8, p3/design.json precedence_contract R3/R4):

* It never guesses. A missing hour, a missing meridiem ("a las 8"), a day that
  can be read two ways, a clock time that already passed today or anything it
  cannot read yields an ambiguity code and no trigger time; there is no default
  09:00.
* A successful result always has ``trigger_at`` strictly after the injected
  clock's ``now``.
* Times are resolved on the household's local calendar (the aware clock's
  timezone, America/Montevideo in production) and returned as UTC epochs.
* The raw complement after the reminder verb is kept unmodified so a later unit
  (T031-U5) can parse a home action from it; :func:`extract_time_expression`
  applies the same time grammar to a domotics clause without a reminder verb
  ("apagá la luz en 10 minutos", rule R4).
"""

import logging
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import StrEnum
from functools import lru_cache

logger = logging.getLogger(__name__)


class ReminderAmbiguity(StrEnum):
    """Why a spoken request cannot be scheduled without asking again."""

    UNPARSEABLE = "unparseable"
    MISSING_TEXT = "missing_text"
    MISSING_TIME = "missing_time"
    AMBIGUOUS_DAY = "ambiguous_day"
    AMBIGUOUS_MERIDIEM = "ambiguous_meridiem"
    PAST_TODAY = "past_today"


# Order in which ``ambiguity`` reports the most important issue.
_PRIORITY = tuple(ReminderAmbiguity)


@dataclass(frozen=True)
class VoiceReminderRequest:
    """Result of parsing one spoken reminder request.

    Attributes:
        text: Reminder text in the speaker's words, without the time, day,
            recurrence and zone phrases and without leading connectors.
        trigger_at: First trigger as a UTC epoch; ``None`` unless ``ok``.
        recurrence: Recurrence string for ``parse_recurrence`` or ``None``.
        zone_phrase: Spoken zone phrase ("en la cocina") for a configured alias,
            not resolved to a room here.
        ambiguity: Most important issue, ``None`` when the request is complete.
        complement: Raw text after the reminder verb, unmodified.
        time_phrase: Spoken time words that were recognised, if any.
        issues: Every issue found, most important first.
        detail: Machine-readable reason for logs (never user text).
    """

    text: str
    trigger_at: float | None
    recurrence: str | None
    zone_phrase: str | None
    ambiguity: ReminderAmbiguity | None
    complement: str
    time_phrase: str | None = None
    issues: tuple[ReminderAmbiguity, ...] = ()
    detail: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the request can be scheduled as is."""
        return self.ambiguity is None


@dataclass(frozen=True)
class DeferredClause:
    """Time expression extracted from a clause without a reminder verb (R4).

    Attributes:
        clause: The clause without the time words ("apagá la luz").
        trigger_at: First trigger as a UTC epoch; ``None`` unless ``ok``.
        recurrence: Recurrence string for ``parse_recurrence`` or ``None``.
        ambiguity: Most important issue, ``None`` when complete.
        time_phrase: Spoken time words that were recognised, if any.
        issues: Every issue found, most important first.
        detail: Machine-readable reason for logs (never user text).
    """

    clause: str
    trigger_at: float | None
    recurrence: str | None
    ambiguity: ReminderAmbiguity | None
    time_phrase: str | None = None
    issues: tuple[ReminderAmbiguity, ...] = ()
    detail: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the clause carries one complete, future time."""
        return self.ambiguity is None


# --------------------------------------------------------------------------
# Lexicon (accent-insensitive, lowercase; "ñ" folds to "n").
# --------------------------------------------------------------------------

_VERB = re.compile(
    r"(?:no\s+(?:me|nos)\s+dejes\s+olvidar"
    r"|hace(?:me|nos)\s+acordar"
    r"|(?:(?:me|nos)\s+)?(?:podes|podrias|puedes)\s+(?:recordar|avisar)(?:me|nos|le|les)?"
    r"|(?:me|nos)\s+(?:recordas|avisas)"
    r"|recorda(?:me|nos|le|les)|recuerda(?:me|nos|le|les)"
    r"|acorda(?:me|nos)|acuerda(?:me|nos)"
    r"|avisa(?:me|nos|le|les))\b"
)
_LEAD = re.compile(r"[\s¿¡,.;:!?\"']*(?:(?:che|ey|oye|hola|por\s+favor|porfa)\b[\s,.;:!?]*)*")
_TOKEN = re.compile(r"\d{1,2}[:.]\d{2}(?!\d)|\d+|[ap]\.\s?m\.|[a-z]+")
_CLOCK_TOKEN = re.compile(r"(\d{1,2})[:.](\d{2})")
_UNSUPPORTED = re.compile(
    r"\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre"
    r"|semana\s+que\s+viene|proxima\s+semana|mes\s+que\s+viene|proximo\s+mes|fin(?:es)?\s+de\s+semana"
    r"|todos\s+los\s+(?:meses|anos)|todas\s+las\s+semanas"
    r"|cada\s+(?:mes|semana|ano|hora|\d+|[a-z]+\s+(?:dias|horas|semanas|minutos|meses))"
    r"|(?:en|dentro\s+de)\s+(?:[a-z0-9]+\s+){1,2}(?:dias|semanas|meses|anos)"
    r"|(?:en|dentro\s+de)\s+(?:un|una|1)\s+(?:dia|semana|mes|ano)"
    r"|el\s+dia|ano\s+que\s+viene|proximo\s+ano"
    r"|(?:fin|final|finales|principio|principios|comienzo)\s+de\s+mes"
    r"|el\s+\d{1,2}(?!\s*(?:minutos?|horas?|[:.]\d)))\b"
)
# Date vocabulary that the grammar does not read: left among the reminder words it
# means a date the parser would otherwise ignore, so the request is refused.
_DATE_WORDS = frozenset({
    "dia", "dias", "semana", "semanas", "mes", "meses", "ano", "anos", "finde", "findes",
    "feriado", "feriados", "navidad", "nochebuena", "hoy", "pasado",
})
_CUT_WORDS = {"salvo", "excepto", "menos"}
_SINGULAR_ONE = {"un", "una", "uno"}

_UNITS = {"cero": 0, "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
          "seis": 6, "siete": 7, "ocho": 8, "nueve": 9}
_SMALL = {"diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
          "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
          "veintiun": 21, "veintiuno": 21, "veintiuna": 21, "veintidos": 22, "veintitres": 23,
          "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
          "veintiocho": 28, "veintinueve": 29}
_TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
         "ochenta": 80, "noventa": 90}
_DURATION_UNITS = {"segundo": 1, "segundos": 1, "seg": 1, "minuto": 60, "minutos": 60, "min": 60,
                   "mins": 60, "hora": 3600, "horas": 3600, "hs": 3600, "hrs": 3600}
_VAGUE = {"unos", "unas", "pocos", "pocas", "algunos", "algunas", "varios", "varias"}
_WEEKDAYS = {"lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3, "viernes": 4, "sabado": 5,
             "domingo": 6}
_WEEKDAYS_PLURAL = {"lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3, "viernes": 4,
                    "sabados": 5, "domingos": 6}
_WEEKDAYS_ANY = {**_WEEKDAYS, **_WEEKDAYS_PLURAL}
_PERIODS = {"manana", "tarde", "noche", "madrugada"}
_AM_KINDS = {"am", "manana", "madrugada"}
_ARTICLES = {"el", "la", "los", "las"}

_WEEKDAYS_PHRASES = (
    ("de", "lunes", "a", "viernes"), ("todos", "los", "dias", "de", "semana"),
    ("todos", "los", "dias", "habiles"), ("los", "dias", "de", "semana"),
    ("dias", "de", "semana"), ("dias", "habiles"), ("entre", "semana"),
)
_DAILY_PHRASES = (
    (("todos", "los", "dias"), None), (("cada", "dia"), None), (("diariamente",), None),
    (("todas", "las", "mananas"), "manana"), (("todas", "las", "tardes"), "tarde"),
    (("todas", "las", "noches"), "noche"), (("cada", "manana"), "manana"),
    (("cada", "tarde"), "tarde"), (("cada", "noche"), "noche"),
)

# 12-hour value -> 24-hour value for each period word; a missing key is ambiguous.
_PERIOD_HOURS = {
    "am": {12: 0, **{x: x for x in range(1, 12)}},
    "pm": {12: 12, **{x: x + 12 for x in range(1, 12)}},
    "manana": {x: x for x in range(1, 12)},
    "madrugada": {12: 0, **{x: x for x in range(1, 7)}},
    "mediodia": {12: 12, 1: 13, 2: 14, 3: 15},
    "tarde": {12: 12, **{x: x + 12 for x in range(1, 10)}},
    "noche": {12: 0, **{x: x + 12 for x in range(6, 12)}},
}
# Hour ranges accepted for a 24-hour value (13-23) next to a period word.
_H24_RANGES = {"pm": range(12, 24), "tarde": range(12, 22), "noche": range(18, 24),
               "mediodia": range(12, 16)}


@lru_cache(maxsize=512)
def _fold_char(ch: str) -> str:
    # NFD puts the base letter first ("á" -> "a" + accent, "ñ" -> "n" + tilde).
    return unicodedata.normalize("NFD", ch.lower())[:1] or " "


def _fold(text: str) -> str:
    """Lowercase and strip accents keeping one character per input character."""
    return "".join(_fold_char(ch) for ch in text)


def _number(words: list[str], i: int) -> tuple[int, int] | None:
    """Read a number (digits or words up to 99) at ``i``; return (value, next index)."""
    if i >= len(words):
        return None
    w = words[i]
    if w.isdigit():
        return int(w), i + 1
    if w in _UNITS:
        return _UNITS[w], i + 1
    if w in _SMALL:
        return _SMALL[w], i + 1
    if w in _TENS:
        if i + 2 < len(words) and words[i + 1] == "y" and _UNITS.get(words[i + 2], 0) > 0:
            return _TENS[w] + _UNITS[words[i + 2]], i + 3
        return _TENS[w], i + 1
    return None


def _at(words: list[str], i: int, phrase: tuple[str, ...]) -> bool:
    return tuple(words[i:i + len(phrase)]) == phrase


@dataclass
class _Clock:
    hour: int
    minute: int
    meridiem: str | None
    explicit24: bool
    error: str | None = None


@dataclass
class _Scan:
    """Time constructs found in one token sequence."""

    used: list[bool]
    spans: list[tuple[int, int]] = field(default_factory=list)
    recurrences: list[tuple[str | None, str | None]] = field(default_factory=list)
    relatives: list[int] = field(default_factory=list)
    clocks: list[_Clock] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)
    days: list[tuple[str, int, str | None]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    vague: bool = False

    def take(self, first: int, end: int) -> None:
        for k in range(first, end):
            self.used[k] = True
        self.spans.append((first, end))


def _adjacent_period(words: list[str], k: int) -> tuple[str, int] | None:
    """Period words right after an hour ("de la noche", "p.m.")."""
    n = len(words)
    if k >= n:
        return None
    w = words[k].replace(" ", "").replace(".", "")
    if w in ("am", "pm"):
        return w, k + 1
    if words[k] == "de":
        if k + 2 < n and words[k + 1] == "la" and words[k + 2] in _PERIODS:
            return words[k + 2], k + 3
        if k + 1 < n and words[k + 1] in _PERIODS:
            return words[k + 1], k + 2
    if words[k] == "del" and k + 1 < n and words[k + 1] == "mediodia":
        return "mediodia", k + 2
    return None


def _clock_at(words: list[str], i: int) -> tuple[_Clock, int] | None:
    """Clock time at ``i``: "a las 8 y media de la noche", "al mediodía"."""
    n = len(words)
    if _at(words, i, ("al", "mediodia")):
        return _Clock(12, 0, "mediodia", True), i + 2
    if words[i] not in ("a", "para"):
        return None
    j = i + 1
    if j < n and words[j] == "medianoche":
        return _Clock(0, 0, None, True), j + 1
    if _at(words, j, ("eso", "de")):
        j += 2
    if j >= n or words[j] not in ("la", "las"):
        return None
    j += 1
    if j < n and words[j] == "medianoche":
        return _Clock(0, 0, None, True), j + 1
    if j >= n:
        return None
    token = _CLOCK_TOKEN.fullmatch(words[j])
    if token:
        hour, minute, k = int(token.group(1)), int(token.group(2)), j + 1
    else:
        num = _number(words, j)
        if num is None:
            return None
        hour, k = num
        minute = 0
        if _at(words, k, ("y", "media")):
            minute, k = 30, k + 2
        elif _at(words, k, ("y", "cuarto")):
            minute, k = 15, k + 2
        elif _at(words, k, ("menos", "cuarto")):
            minute, k = -15, k + 2
        elif k < n and words[k] in ("y", "menos") and _number(words, k + 1) is not None:
            value, end = _number(words, k + 1)
            minute, k = (value if words[k] == "y" else -value), end
            if k < n and words[k] in ("minuto", "minutos"):
                k += 1
        elif (later := _number(words, k)) is not None and (later[0] >= 10 or len(words[k]) == 2):
            minute, k = later
    if _at(words, k, ("en", "punto")):
        k += 2
    if k < n and words[k] in ("h", "hs", "hrs", "hora", "horas"):
        k += 1
    period = _adjacent_period(words, k)
    meridiem = None
    if period is not None:
        meridiem, k = period
    explicit24 = hour == 0 or hour >= 13
    error = None
    if hour > 24 or not -59 <= minute <= 59 or (hour == 24 and minute > 0):
        error = "invalid_clock_time"
    elif minute < 0:
        if hour in (0, 24):
            hour = 23
        elif hour == 1 and not explicit24:
            hour = 12
        else:
            hour -= 1
        minute += 60
    return _Clock(hour % 24 if not error else hour, minute, meridiem, explicit24, error), k


def _relative_at(words: list[str], i: int) -> tuple[int | str, int] | None:
    """Relative time at ``i``: seconds, "vague" or "invalid"; with next index."""
    n = len(words)
    if words[i] == "en":
        j = i + 1
    elif _at(words, i, ("dentro", "de")):
        j = i + 2
    else:
        return None
    if j >= n:
        return None
    if words[j] in _VAGUE or _at(words, j, ("un", "par")) or _at(words, j, ("un", "rato")) \
            or _at(words, j, ("un", "ratito")) or words[j] in ("rato", "ratito"):
        return "vague", j
    if _at(words, j, ("media", "hora")):
        return 1800, j + 2
    if _at(words, j, ("un", "cuarto", "de", "hora")):
        return 900, j + 4
    num = _number(words, j)
    if num is None or num[1] >= n or words[num[1]] not in _DURATION_UNITS:
        for look in (j + 1, j + 2):
            if look < n and words[look] in _DURATION_UNITS and _number(words, j) is None:
                return "invalid", look + 1
        return None
    amount, k = num
    unit = _DURATION_UNITS[words[k]]
    seconds, k = amount * unit, k + 1
    if unit == 3600:
        if _at(words, k, ("y", "media")):
            seconds, k = seconds + 1800, k + 2
        elif _at(words, k, ("y", "cuarto")):
            seconds, k = seconds + 900, k + 2
        elif k < n and words[k] == "y" and _number(words, k + 1) is not None:
            extra, end = _number(words, k + 1)
            if end < n and words[end] in ("minuto", "minutos", "min"):
                seconds, k = seconds + extra * 60, end + 1
    if seconds <= 0:
        return "invalid", k
    return seconds, k


def _recurrence_at(words: list[str], i: int) -> tuple[str | None, str | None, int] | None:
    """Recurrence at ``i``: (recurrence or None if unsupported, period, next index)."""
    for phrase in _WEEKDAYS_PHRASES:
        if _at(words, i, phrase):
            return "weekdays", None, i + len(phrase)
    for phrase, period in _DAILY_PHRASES:
        if _at(words, i, phrase):
            return "daily", period, i + len(phrase)
    n = len(words)
    if _at(words, i, ("todos", "los")) and i + 2 < n and words[i + 2] in _WEEKDAYS_ANY:
        weekday, k = _WEEKDAYS_ANY[words[i + 2]], i + 3
    elif words[i] == "cada" and i + 1 < n and words[i + 1] in _WEEKDAYS:
        weekday, k = _WEEKDAYS[words[i + 1]], i + 2
    elif words[i] == "los" and i + 1 < n and words[i + 1] in _WEEKDAYS_PLURAL:
        weekday, k = _WEEKDAYS_PLURAL[words[i + 1]], i + 2
    else:
        return None
    # "los lunes y (los) jueves": several weekdays do not fit one recurrence.
    if k < n and words[k] == "y":
        nxt = k + 2 if k + 1 < n and words[k + 1] == "los" else k + 1
        if nxt < n and words[nxt] in _WEEKDAYS_ANY:
            return None, None, nxt + 1
    return f"weekly:{weekday + 1}", None, k


def _period_at(words: list[str], i: int) -> tuple[str, bool, int] | None:
    """Detached period ("a la noche", "esta tarde"): (period, means today, next index)."""
    n = len(words)
    if words[i] in ("a", "por", "en") and i + 2 < n and words[i + 1] == "la" and words[i + 2] in _PERIODS:
        return words[i + 2], False, i + 3
    if words[i] == "esta" and i + 1 < n and words[i + 1] in ("manana", "tarde", "noche"):
        return words[i + 1], True, i + 2
    if words[i] == "de" and i > 0 and words[i - 1] == "manana" and i + 1 < n and words[i + 1] in _PERIODS:
        return words[i + 1], False, i + 2
    return None


def _day_at(words: list[str], i: int, *, joined: bool = False) -> tuple[str, int, str | None, int] | None:
    """Day word at ``i``: (kind, value, modifier, next index).

    ``joined`` means the word follows "de" right after a time phrase already
    taken ("a las 9 de la noche de mañana"), so "mañana" is the day, not a
    period of the day.
    """
    n = len(words)
    if i >= n:
        return None
    w = words[i]
    if _at(words, i, ("pasado", "manana")):
        return "offset", 2, None, i + 2
    if w == "manana" and (joined or i == 0 or words[i - 1] not in ("la", "esta", "de", "pasado")):
        return "offset", 1, None, i + 1
    if w == "hoy":
        return "today", 0, None, i + 1
    k, modifier = i, None
    if w in ("el", "este"):
        modifier, k = ("este" if w == "este" else None), i + 1
    if k < n and words[k] == "proximo":
        modifier, k = "proximo", k + 1
    if k >= n or words[k] not in _WEEKDAYS:
        return None
    weekday, k = _WEEKDAYS[words[k]], k + 1
    if _at(words, k, ("que", "viene")):
        modifier, k = "proximo", k + 2
    elif k < n and words[k] == "proximo":
        modifier, k = "proximo", k + 1
    return "weekday", weekday, modifier, k


def _scan(words: list[str]) -> _Scan:
    """Find every time construct; later passes skip tokens already taken."""
    scan = _Scan(used=[False] * len(words))

    def recurrence(i: int) -> int | None:
        found = _recurrence_at(words, i)
        if found is None:
            return None
        rec, period, end = found
        scan.recurrences.append((rec, period))
        if rec is None:
            scan.errors.append("unsupported_recurrence")
        return end

    def relative(i: int) -> int | None:
        found = _relative_at(words, i)
        if found is None:
            return None
        value, end = found
        if value == "vague":
            scan.vague = True
            return None
        if value == "invalid":
            scan.errors.append("invalid_duration")
        else:
            scan.relatives.append(int(value))
        return end

    def clock(i: int) -> int | None:
        found = _clock_at(words, i)
        if found is None:
            return None
        parsed, end = found
        scan.clocks.append(parsed)
        if parsed.error:
            scan.errors.append(parsed.error)
        return end

    def period(i: int) -> int | None:
        if words[i] == "de" and i > 0 and scan.used[i - 1]:
            return None          # "de mañana" after a taken time is a day word
        found = _period_at(words, i)
        if found is None:
            return None
        name, today, end = found
        scan.periods.append(name)
        if today:
            scan.days.append(("today", 0, None))
        return end

    def day(i: int) -> int | None:
        if words[i] in ("de", "del") and i > 0 and scan.used[i - 1]:
            found = _day_at(words, i + 1, joined=True)
            if found is not None:
                kind, value, modifier, end = found
                scan.days.append((kind, value, modifier))
                return end
            return None
        found = _day_at(words, i)
        if found is None:
            return None
        kind, value, modifier, end = found
        scan.days.append((kind, value, modifier))
        return end

    for matcher in (recurrence, relative, clock, period, day):
        i = 0
        while i < len(words):
            if scan.used[i]:
                i += 1
                continue
            end = matcher(i)
            if end is None or any(scan.used[i:end]):
                i += 1
                continue
            scan.take(i, end)
            i = end
    return scan


def _hour24(clock: _Clock, meridiem: str | None) -> tuple[int | None, ReminderAmbiguity | None]:
    """Resolve the 24-hour value; never picks between morning and evening."""
    h = clock.hour
    if meridiem is None:
        if clock.explicit24 or h >= 13:
            return h, None
        return None, ReminderAmbiguity.AMBIGUOUS_MERIDIEM
    if clock.explicit24 and h != 12:
        ok = meridiem in ("noche", "madrugada", "am") if h == 0 else h in _H24_RANGES.get(meridiem, ())
        return (h, None) if ok else (None, ReminderAmbiguity.UNPARSEABLE)
    table = _PERIOD_HOURS[meridiem]
    if h in table:
        return table[h], None
    return None, ReminderAmbiguity.AMBIGUOUS_MERIDIEM


def _category(period: str) -> str:
    return "am" if period in _AM_KINDS else "pm"


def _resolve(scan: _Scan, now: datetime) -> tuple[float | None, str | None, set[ReminderAmbiguity], str | None]:
    """Combine the constructs into (trigger_at, recurrence, issues, detail)."""
    issues: set[ReminderAmbiguity] = set()
    if scan.errors:
        return None, None, {ReminderAmbiguity.UNPARSEABLE}, scan.errors[0]
    if len(scan.recurrences) > 1 or len(scan.clocks) > 1 or len(scan.relatives) > 1:
        return None, None, {ReminderAmbiguity.UNPARSEABLE}, "multiple_times"
    if scan.relatives:
        if scan.clocks or scan.recurrences or scan.days or scan.periods:
            return None, None, {ReminderAmbiguity.UNPARSEABLE}, "conflicting_times"
        return now.timestamp() + scan.relatives[0], None, issues, None
    recurrence, rec_period = scan.recurrences[0] if scan.recurrences else (None, None)
    periods = set(scan.periods) | ({rec_period} if rec_period else set())
    if not scan.clocks:
        return None, None, {ReminderAmbiguity.MISSING_TIME}, "vague_time" if scan.vague else "no_time"
    clock = scan.clocks[0]
    meridiem = clock.meridiem
    if periods:
        if len(periods) > 1 or (meridiem and {_category(meridiem)} != {_category(p) for p in periods}):
            issues.add(ReminderAmbiguity.AMBIGUOUS_MERIDIEM)
        elif meridiem is None:
            meridiem = next(iter(periods))
    hour, problem = _hour24(clock, meridiem)
    if problem is ReminderAmbiguity.UNPARSEABLE:
        return None, None, {problem}, "period_contradicts_hour"
    if problem:
        issues.add(problem)

    today = now.date()
    offsets: set[int] = set()
    relative_day = False
    for kind, value, modifier in scan.days:
        if kind == "weekday":
            delta = (value - today.weekday()) % 7
            if delta == 0 and modifier == "proximo":
                delta = 7
            elif delta == 0 and modifier != "este":
                issues.add(ReminderAmbiguity.AMBIGUOUS_DAY)
                continue
            offsets.add(delta)
        else:
            offsets.add(value)
            relative_day = relative_day or kind == "offset"
    if len(offsets) > 1 or (recurrence and scan.days):
        issues.add(ReminderAmbiguity.AMBIGUOUS_DAY)
    if hour is None or issues:
        return None, None, issues, None

    def at(days: int) -> datetime:
        return datetime.combine(today + timedelta(days=days), time(hour, clock.minute), tzinfo=now.tzinfo)

    if recurrence:
        if recurrence == "daily":
            candidates = (0, 1)
        elif recurrence == "weekdays":
            candidates = tuple(d for d in range(8) if (today + timedelta(days=d)).weekday() < 5)
        else:
            first = (int(recurrence.split(":")[1]) - 1 - today.weekday()) % 7
            candidates = (first, first + 7)
        target = next(c for c in map(at, candidates) if c.timestamp() > now.timestamp())
        return target.timestamp(), recurrence, issues, None

    offset = next(iter(offsets)) if offsets else 0
    small_hours = hour < 6
    if offset > 0:
        if hour == 0:
            return None, None, {ReminderAmbiguity.AMBIGUOUS_DAY}, "midnight_with_day"
        if relative_day and now.hour < 6:
            return None, None, {ReminderAmbiguity.AMBIGUOUS_DAY}, "tomorrow_before_dawn"
        target = at(offset)
    else:
        target = at(0)
        if small_hours and target.timestamp() <= now.timestamp():
            target = at(1)
    if target.timestamp() <= now.timestamp():
        return None, None, {ReminderAmbiguity.PAST_TODAY}, None
    return target.timestamp(), None, issues, None


def _clean(text: str, *, connectors: bool) -> str:
    """Collapse whitespace, drop edge punctuation, connectors and politeness."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)
    edge = " ,.;:!?¿¡\"'"
    lead = re.compile(r"^(?:(?:de\s+que|que|de|y)\b[\s,]*)" if connectors else r"^(?:y\b[\s,]*)")
    trail = re.compile(r"[\s,]*\b(?:por\s+favor|porfa|y)$")
    for _ in range(4):
        text = text.strip(edge)
        folded = _fold(text)
        m = lead.match(folded)
        if m and m.end():
            text = text[m.end():]
            continue
        m = trail.search(folded)
        if m:
            text = text[:m.start()]
            continue
        break
    return text.strip(edge)


def _ordered(issues: set[ReminderAmbiguity]) -> tuple[ReminderAmbiguity, ...]:
    return tuple(code for code in _PRIORITY if code in issues)


def _require_aware(now: datetime) -> None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("aware_clock_required")


def _fold_aliases(aliases: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    folded = {tuple(_TOKEN.findall(_fold(alias))) for alias in aliases}
    return tuple(sorted((a for a in folded if a), key=len, reverse=True))


def _zone(words: list[str], used: list[bool], aliases: tuple[tuple[str, ...], ...]) -> tuple[int, int] | None:
    """Zone phrase "en (el|la) <alias>" first or last among the untaken tokens."""
    free = [k for k, taken in enumerate(used) if not taken]
    while len(free) >= 2 and [words[k] for k in free[-2:]] == ["por", "favor"]:
        free = free[:-2]
    if free and words[free[-1]] == "porfa":
        free = free[:-1]
    if not free or not aliases:
        return None

    def phrase_at(pos: int) -> int | None:
        if words[free[pos]] != "en":
            return None
        p = pos + 1
        if p < len(free) and words[free[p]] in _ARTICLES:
            p += 1
        for alias in aliases:
            seg = free[p:p + len(alias)]
            if len(seg) == len(alias) and tuple(words[k] for k in seg) == alias \
                    and seg[-1] - free[pos] == p + len(alias) - 1 - pos:
                return p + len(alias)
        return None

    end = phrase_at(0)
    if end is not None:
        return free[0], free[end - 1] + 1
    for start in range(len(free) - 1, -1, -1):
        end = phrase_at(start)
        if end is not None and end == len(free):
            return free[start], free[end - 1] + 1
    return None


def _is_number_word(word: str) -> bool:
    return _number([word], 0) is not None


def _leftover_problem(words: list[str], used: list[bool], skip: range) -> str | None:
    """Time vocabulary left among the reminder words; the detail code or ``None``.

    Words the scanner did not take are reminder content. When one of them is a
    date or time word, a part of the request was not understood and scheduling
    the rest would be a guess (user answer Q8), so the request is refused.
    """
    n = len(words)
    for k in range(n):
        if used[k] or k in skip:
            continue
        w = words[k]
        prev = words[k - 1] if k > 0 else ""
        nxt = words[k + 1] if k + 1 < n else ""
        after_time = k > 0 and used[k - 1]
        if w in _DATE_WORDS or w in _WEEKDAYS_ANY:
            return "unrecognised_date"
        if w == "manana" and prev != "la":
            return "unrecognised_date"
        if w == "primero" and (prev == "el" or nxt == "de"):
            return "unrecognised_date"
        if w == "el" and _is_number_word(nxt) and (k + 2 >= n or words[k + 2] not in _DURATION_UNITS):
            return "unrecognised_date"
        if w == "y" and nxt == "pico":
            return "unrecognised_time"
        if after_time and w in _CUT_WORDS:
            return "unrecognised_time"
        if after_time and w == "y" and (nxt in ("cuarto", "media")
                                        or (_is_number_word(nxt) and nxt not in _SINGULAR_ONE)):
            return "unrecognised_time"
        if after_time and _is_number_word(w) and w not in _SINGULAR_ONE:
            return "unrecognised_date" if prev in _WEEKDAYS or prev in ("hoy", "manana") else "unrecognised_time"
    return None


def _analyse(original: str, folded: str, start: int, aliases: tuple[tuple[str, ...], ...], *,
             now: datetime, connectors: bool) -> tuple[str, float | None, str | None, str | None, str | None,
                                                       set[ReminderAmbiguity], str | None]:
    """Shared core: (text, trigger_at, recurrence, zone, time_phrase, issues, detail)."""
    tokens = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(folded, start)]
    words = [t[0] for t in tokens]
    if _UNSUPPORTED.search(folded, start):
        scan = _Scan(used=[False] * len(words))
        trigger, recurrence, issues, detail = None, None, {ReminderAmbiguity.UNPARSEABLE}, "unsupported_date"
    else:
        scan = _scan(words)
        trigger, recurrence, issues, detail = _resolve(scan, now)
    removed = [(tokens[a][1], tokens[b - 1][2]) for a, b in scan.spans]
    time_phrase = " ".join(original[a:b] for a, b in sorted(removed)) or None
    zone = None
    zone_span = _zone(words, scan.used, aliases)
    if zone_span is not None:
        a, b = tokens[zone_span[0]][1], tokens[zone_span[1] - 1][2]
        zone = original[a:b]
        removed.append((a, b))
    if ReminderAmbiguity.UNPARSEABLE not in issues:
        leftover = _leftover_problem(words, scan.used, range(*zone_span) if zone_span else range(0))
        if leftover is not None:
            trigger, recurrence, issues, detail = None, None, {ReminderAmbiguity.UNPARSEABLE}, leftover
    pieces, cursor = [], start
    for a, b in sorted(removed):
        pieces.append(original[cursor:a])
        cursor = b
    pieces.append(original[cursor:])
    text = _clean(" ".join(pieces), connectors=connectors)
    if not text:
        issues.add(ReminderAmbiguity.MISSING_TEXT)
    if issues:
        trigger, recurrence = None, None
    return text, trigger, recurrence, zone, time_phrase, issues, detail


def _head(folded: str, wake_words: tuple[str, ...]) -> int:
    """Offset after the leading politeness and wake words."""
    pos = 0
    for _ in range(4):
        pos = _LEAD.match(folded, pos).end()
        for wake in wake_words:
            if folded.startswith(wake, pos) and not folded[pos + len(wake):pos + len(wake) + 1].isalnum():
                pos += len(wake)
                break
        else:
            break
    return _LEAD.match(folded, pos).end()


def parse_reminder_request(text: str, now: datetime, *, zone_aliases: Iterable[str] = (),
                           wake_words: Iterable[str] = ()) -> VoiceReminderRequest | None:
    """Parse a spoken reminder request that starts with a reminder verb.

    Args:
        text: Transcribed utterance.
        now: Aware current time on the household clock.
        zone_aliases: Configured room aliases a zone phrase may name.
        wake_words: Configured wake words to strip from the head.

    Returns:
        The parsed request, or ``None`` when the utterance does not start with
        a reminder verb (recordame, acordame, haceme acordar, avisame, ...).

    Raises:
        ValueError: If ``now`` is naive.
    """
    _require_aware(now)
    return _parse_reminder(text, now, _fold_aliases(zone_aliases),
                           tuple(_fold(w).strip() for w in wake_words if w.strip()))


def _parse_reminder(text: str, now: datetime, aliases: tuple[tuple[str, ...], ...],
                    wake_words: tuple[str, ...]) -> VoiceReminderRequest | None:
    folded = _fold(text)
    head = _head(folded, wake_words)
    verb = _VERB.match(folded, head)
    if verb is None:
        return None
    start = verb.end()
    body, trigger, recurrence, zone, time_phrase, issues, detail = _analyse(
        text, folded, start, aliases, now=now, connectors=True)
    ordered = _ordered(issues)
    logger.debug("[VoiceRequest] reminder parsed: issues=%s detail=%s recurrence=%s",
                 [c.value for c in ordered], detail, recurrence)
    return VoiceReminderRequest(
        text=body, trigger_at=trigger, recurrence=recurrence, zone_phrase=zone,
        ambiguity=ordered[0] if ordered else None, complement=text[start:].strip(),
        time_phrase=time_phrase, issues=ordered, detail=detail,
    )


def extract_time_expression(text: str, now: datetime) -> DeferredClause:
    """Extract the time of a domotics clause without a reminder verb (rule R4).

    Args:
        text: Clause such as "apagá la luz del living en 10 minutos".
        now: Aware current time on the household clock.

    Returns:
        The clause without its time words plus the resolved time or the
        ambiguity code; ``MISSING_TIME`` when the clause has no time.

    Raises:
        ValueError: If ``now`` is naive.
    """
    _require_aware(now)
    folded = _fold(text)
    start = _LEAD.match(folded).end()
    clause, trigger, recurrence, _zone_unused, time_phrase, issues, detail = _analyse(
        text, folded, start, (), now=now, connectors=False)
    ordered = _ordered(issues)
    return DeferredClause(
        clause=clause, trigger_at=trigger, recurrence=recurrence,
        ambiguity=ordered[0] if ordered else None, time_phrase=time_phrase,
        issues=ordered, detail=detail,
    )


class VoiceRequestParser:
    """Parser bound to the household clock, room aliases and wake words."""

    def __init__(self, clock: Callable[[], datetime], *, zone_aliases: Iterable[str] = (),
                 wake_words: Iterable[str] = ()):
        """Build the parser.

        Args:
            clock: Returns the aware current time on the household clock
                (``routines.timezone``).
            zone_aliases: Configured room aliases a zone phrase may name.
            wake_words: Configured wake words to strip from the head.
        """
        self._clock = clock
        self._aliases = _fold_aliases(zone_aliases)
        self._wake_words = tuple(_fold(w).strip() for w in wake_words if w.strip())

    def parse_reminder(self, text: str) -> VoiceReminderRequest | None:
        """Parse a reminder request against the injected clock."""
        now = self._clock()
        _require_aware(now)
        return _parse_reminder(text, now, self._aliases, self._wake_words)

    def parse_deferred(self, text: str) -> DeferredClause:
        """Extract the time of a domotics clause against the injected clock."""
        return extract_time_expression(text, self._clock())
