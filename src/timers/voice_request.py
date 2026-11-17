"""Rioplatense voice timer requests: duration, label and the timer a command names.

Pure parser (stdlib only, no I/O, never logs). The P3-F2 household classifier
already claimed the turn and split off the words after the timer noun (its
``complement``); this module reads that complement:

* :func:`parse_timer_request` for a creation ("de 5 minutos para la pasta",
  "de media hora", "de un cuarto de hora", "en diez minutos"): the duration in
  seconds, an optional label and, when it cannot be read without guessing, an
  issue code instead of a duration.
* :func:`parse_timer_selector` for cancel/stop/pause/resume/query ("de la
  pasta", "de 5 minutos", ""): the label and/or duration that pick a timer.

It never guesses: a missing or vague duration ("unos minutos", "un rato"), a
number without a unit, a number of a hundred or more said in words ("ciento
veinte segundos"), two durations, a clock time ("para las 7", which is a
reminder) or a room ("en la cocina": the timer rings where it was asked, P3F1-R3)
yields an issue code and no duration.

Labels are household content: callers never log them above DEBUG (design
household_conventions.logging).
"""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from src.nlu.command_grammar import ROOM_ALIASES

# Longest timer accepted by voice (24 h). Longer requests are asked again.
MAX_TIMER_SECONDS = 24 * 3600


class TimerRequestIssue(StrEnum):
    """Why a timer creation cannot be done without asking again."""

    MISSING_DURATION = "missing_duration"
    VAGUE_DURATION = "vague_duration"
    UNPARSEABLE = "unparseable"
    CLOCK_TIME = "clock_time"
    TOO_LONG = "too_long"
    ROOM_TARGET = "room_target"


@dataclass(frozen=True)
class TimerRequest:
    """A parsed timer creation.

    Attributes:
        duration_s: Whole seconds, or None when ``issue`` is set.
        label: Words that name the timer as heard (lowercase), or None.
        issue: Why the request must be asked again, or None.
    """

    duration_s: int | None
    label: str | None = None
    issue: TimerRequestIssue | None = None


@dataclass(frozen=True)
class TimerSelector:
    """Which timer(s) a cancel/stop/pause/resume/query names.

    Attributes:
        label: Spoken label, or None.
        duration_s: Spoken duration ("el timer de 5 minutos"), or None.
    """

    label: str | None = None
    duration_s: int | None = None

    @property
    def empty(self) -> bool:
        """Nothing names a particular timer."""
        return self.label is None and self.duration_s is None


# A decimal with one digit ("1,5 horas", "2.5 minutos") is one token; "1.30"/"1:30" stay clock-like tokens.
_WORD_RE = re.compile(r"\d+(?:[:.]\d{2}|[.,]\d(?!\d))?|[^\W\d_]+")
_DECIMAL_RE = re.compile(r"\d+[.,]\d")

_UNITS = {"un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
          "ocho": 8, "nueve": 9}
_SMALL = {"diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
          "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiun": 21, "veintiuno": 21,
          "veintiuna": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
          "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29}
_TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80,
         "noventa": 90}
# Hundreds and thousands are not read: "ciento veinte segundos" must not become 20 seconds named "ciento".
_HUNDREDS = frozenset({"cien", "ciento", "doscientos", "doscientas", "trescientos", "trescientas", "cuatrocientos",
                       "cuatrocientas", "quinientos", "quinientas", "seiscientos", "seiscientas", "setecientos",
                       "setecientas", "ochocientos", "ochocientas", "novecientos", "novecientas", "mil"})
_TIME_UNITS = {"segundo": 1, "segundos": 1, "seg": 1, "segs": 1, "segundito": 1, "segunditos": 1,
               "minuto": 60, "minutos": 60, "min": 60, "mins": 60, "minutito": 60, "minutitos": 60,
               "hora": 3600, "horas": 3600, "h": 3600, "hs": 3600, "hrs": 3600, "horita": 3600, "horitas": 3600}
_VAGUE = frozenset({"unos", "unas", "algunos", "algunas", "pocos", "pocas", "varios", "varias"})
_HALF = frozenset({"medio", "media"})
_HOURS_WORDS = frozenset({"hora", "horas"})
_MINUTE_WORDS = frozenset({"minuto", "minutos", "min", "mins"})
_SECOND_WORDS = frozenset({"segundo", "segundos", "seg", "segs"})

# A clock time stands alone or is followed by what completes one ("a las 8 y media", "para las 7 de la tarde",
# "para las 7 para la pasta"); "para las dos pizzas" is a label, not a clock time.
_CLOCK_RE = re.compile(r"(?:^| )(?:a|para) (?:las? |el )?(?:(?:\d{1,2}(?:[:.]\d{2})?|una|dos|tres|cuatro|cinco|seis"
                       r"|siete|ocho|nueve|diez|once|doce)(?=$| (?:y|menos|en punto|de la (?:manana|tarde|noche|madrugada)"
                       r"|del mediodia|hs|h|horas|am|pm|para|que)(?: |$))|mediodia|medianoche)(?= |$)")
_ARTICLES = frozenset({"el", "la", "los", "las"})
_ROOMS = frozenset(ROOM_ALIASES)

# Words around a label that only connect it to the sentence ("para la pasta", "llamado pasta").
_LEAD = frozenset({"de", "del", "para", "por", "en", "a", "al", "con", "llamado", "llamada", "que", "se", "llame",
                   "nombre", "el", "la", "los", "las", "un", "una", "mi"})
_TRAIL = frozenset({"de", "del", "para", "por", "en", "a", "al", "con", "y", "que", "el", "la", "los", "las"})
# Courtesy and filler words that never name a timer at the edge of a label or selector ("..., gracias",
# "cancelá el timer ya", "pausá el timer un momento", "el timer que puse", "de la pasta nomás",
# "de 5 minutos exactos", "para avisarme", "para que me avise", "para dentro de 10 minutos"). Stripped with the connectors above;
# never a counted noun ("para las 7 gracias" is a clock time, not seven of something).
_FILLER = frozenset({"gracias", "ya", "ahora", "dale", "nomas", "mas", "exacto", "exactos", "exacta", "exactas",
                     "justo", "justos", "justa", "justas", "momento", "momentito", "dentro", "puse", "pusiste",
                     "pusimos", "puso", "pusieron", "recien", "mismo", "porfa", "porfi", "porfis", "favor", "che",
                     "bueno", "listo", "ok", "okey", "avisar", "avisarme", "avisame", "avisarnos", "avisanos",
                     # "para que me avise", "para que nos avises", "para que suene" (T030-U1 RR-4).
                     "me", "nos", "avise", "avises", "avisen", "suene", "suenes", "suenen"})
_LEAD_STRIP = _LEAD | _FILLER
_TRAIL_STRIP = _TRAIL | _FILLER
# Query and state words that never name a timer ("qué timers hay", "el timer que está corriendo").
_STATE_WORDS = frozenset({"hay", "tengo", "tenemos", "quedan", "queda", "son", "tiene", "tienen", "estan", "esta",
                          "activos", "activas", "activo", "activa", "pendientes", "corriendo", "anda", "andan",
                          "andando", "falta", "faltan", "le", "les", "puestos", "puesto", "puestas", "sonando",
                          "todos", "todas", "ahora"})


def _fold(word: str) -> str:
    """Lowercase and strip accents ("ñ" folds to "n")."""
    return "".join(c for c in unicodedata.normalize("NFD", word.lower()) if unicodedata.category(c) != "Mn")


def _words(text: str) -> tuple[list[str], list[str]]:
    original = _WORD_RE.findall((text or "").lower())
    return original, [_fold(w) for w in original]


def _number(w: list[str], i: int) -> tuple[int, int] | None:
    """A number (digits, or words up to 99) at ``i``: (value, next index)."""
    word = w[i] if i < len(w) else ""
    if word.isdigit():
        return int(word), i + 1
    if word in _UNITS:
        return _UNITS[word], i + 1
    if word in _SMALL and word != "veinte":
        return _SMALL[word], i + 1
    tens = 20 if word == "veinte" else _TENS.get(word)
    if tens is not None:
        # "treinta y un minutos", "veinte y cinco minutos" (spoken apart).
        if i + 2 < len(w) and w[i + 1] == "y" and w[i + 2] in _UNITS:
            return tens + _UNITS[w[i + 2]], i + 3
        return tens, i + 1
    return None


def _at(w: list[str], i: int, *phrase: str) -> bool:
    return tuple(w[i:i + len(phrase)]) == phrase


def _extension(w: list[str], k: int, unit: int) -> tuple[int, int]:
    """Seconds added after a unit ("una hora y media", "dos minutos y 30 segundos"): (extra, next index)."""
    if unit == 3600:
        smaller, words = 60, _MINUTE_WORDS
        if _at(w, k, "y", "media") or _at(w, k, "y", "medio"):
            return 1800, k + 2
        if _at(w, k, "y", "cuarto"):
            return 900, k + 2
    elif unit == 60:
        smaller, words = 1, _SECOND_WORDS
        if _at(w, k, "y", "medio") or _at(w, k, "y", "media"):
            return 30, k + 2
    else:
        return 0, k
    start = k + 1 if k < len(w) and w[k] == "y" else k
    found = _number(w, start)
    if found is not None and found[1] < len(w) and w[found[1]] in words:
        return found[0] * smaller, found[1] + 1
    return 0, k


def _duration_at(w: list[str], i: int) -> tuple[int | str, int] | None:
    """A duration starting at token ``i``: (seconds | "vague" | "invalid", next index), or None."""
    n = len(w)
    if w[i] in _VAGUE and i + 1 < n and w[i + 1] in _TIME_UNITS:
        return "vague", i + 2
    if _at(w, i, "un", "par", "de") and i + 3 < n and w[i + 3] in _TIME_UNITS:
        return "vague", i + 4
    if w[i] in ("rato", "ratito") or _at(w, i, "un", "rato") or _at(w, i, "un", "ratito"):
        return "vague", i + (1 if w[i] in ("rato", "ratito") else 2)
    if _at(w, i, "media", "hora"):
        return 1800, i + 2
    if _at(w, i, "medio", "minuto"):
        return 30, i + 2
    if _at(w, i, "minuto", "y", "medio"):
        return 90, i + 3
    if _at(w, i, "hora", "y", "media"):
        return 5400, i + 3
    if _at(w, i, "hora", "y", "cuarto"):
        return 4500, i + 3
    for start in (i, i + 1):
        if start == i + 1 and w[i] not in ("un", "una"):
            break
        if _at(w, start, "cuarto", "de", "hora"):
            return 900, start + 3
    if _at(w, i, "tres", "cuartos", "de", "hora"):
        return 2700, i + 4
    if _DECIMAL_RE.fullmatch(w[i]) and i + 1 < n and w[i + 1] in _TIME_UNITS:
        # "1,5 horas": whole seconds only.
        exact = Decimal(w[i].replace(",", ".")) * _TIME_UNITS[w[i + 1]]
        return (int(exact) if exact == exact.to_integral_value() and exact > 0 else "invalid"), i + 2
    found = _number(w, i)
    if found is None:
        return None
    value, k = found
    if k < n and w[k] in _TIME_UNITS:
        unit = _TIME_UNITS[w[k]]
        extra, k = _extension(w, k + 1, unit)
        seconds = value * unit + extra
    elif k + 2 < n and w[k] == "y" and w[k + 1] in _HALF and w[k + 2] in _TIME_UNITS:
        # "cinco y medio minutos", "dos y media horas"
        unit = _TIME_UNITS[w[k + 2]]
        seconds, k = value * unit + unit // 2, k + 3
    else:
        return None
    return (seconds if seconds > 0 else "invalid"), k


def _bare_number(w: list[str], start: int, end: int) -> bool:
    """A number without a unit ("un timer de 5") among ``w[start:end]``.

    A number that counts a following noun ("para las dos pizzas") is part of a
    label, not a duration.
    """
    for k in range(start, end):
        word = w[k]
        if word.isdigit() or re.fullmatch(r"\d+[:.]\d{2}", word) or _DECIMAL_RE.fullmatch(word) or word in _SMALL \
                or word in _TENS or (word in _UNITS and word not in ("un", "una", "uno")):
            if not _counts_a_noun(w, k + 1, end):
                return True
        elif word in _HUNDREDS and not (k + 1 < end and _number(w, k + 1) is not None):
            # "de ciento", "cien personas" is a label; "ciento veinte" is decided by the next number.
            if not _counts_a_noun(w, k + 1, end):
                return True
    return False


def _counts_a_noun(w: list[str], k: int, end: int) -> bool:
    """The word at ``k`` is a counted noun: a content word, not a unit, number or connector."""
    if k >= end:
        return False
    word = w[k]
    return (word.isalpha() and word not in _TIME_UNITS and word not in _UNITS and word not in _SMALL
            and word not in _TENS and word not in _LEAD and word not in _TRAIL and word not in _HALF
            and word not in _FILLER and word not in _HUNDREDS and word not in ("cuarto", "cuartos", "y", "menos"))


def _trim(original: list[str], folded: list[str], *, state_words: bool) -> list[str]:
    """Strip connector, courtesy and filler words around one piece of a label; optionally drop query/state words."""
    keep = [(o, f) for o, f in zip(original, folded) if not (state_words and f in _STATE_WORDS)]
    while keep and keep[0][1] in _LEAD_STRIP:
        keep.pop(0)
    while keep and keep[-1][1] in _TRAIL_STRIP:
        keep.pop()
    return [o for o, _ in keep]


def _label(original: list[str], folded: list[str], span: tuple[int, int] | None, *,
           state_words: bool = False) -> str | None:
    pieces = [(0, len(folded))] if span is None else [(0, span[0]), (span[1], len(folded))]
    words: list[str] = []
    for start, end in pieces:
        words += _trim(original[start:end], folded[start:end], state_words=state_words)
    return " ".join(words) or None


def _room_target(folded: list[str], span: tuple[int, int] | None) -> bool:
    """'en (el|la) <room>' outside the duration: a ring room the timer cannot honour."""
    for k, word in enumerate(folded):
        if word != "en" or (span is not None and span[0] <= k < span[1]):
            continue
        j = k + 1 + (1 if k + 1 < len(folded) and folded[k + 1] in _ARTICLES else 0)
        if j < len(folded) and folded[j] in _ROOMS:
            return True
    return False


def _find_duration(folded: list[str]) -> tuple[int | str, tuple[int, int]] | None:
    for i in range(len(folded)):
        found = _duration_at(folded, i)
        if found is not None:
            if i > 0 and folded[i - 1] in _HUNDREDS:
                # "ciento veinte segundos": only the tail would be read, so the whole number is asked again.
                return "invalid", (i - 1, found[1])
            return found[0], (i, found[1])
    return None


def parse_timer_request(complement: str) -> TimerRequest:
    """Read a timer creation from the words after the timer (or alarm) noun.

    Args:
        complement: The household intent's complement, e.g. "de 5 minutos para la pasta".

    Returns:
        A :class:`TimerRequest` with ``duration_s`` set and ``issue`` None, or
        ``issue`` set and ``duration_s`` None.
    """
    original, folded = _words(complement)
    found = _find_duration(folded)
    span = found[1] if found is not None else None
    label = _label(original, folded, span)
    rest = folded if span is None else folded[:span[0]] + folded[span[1]:]
    if _room_target(folded, span):
        return TimerRequest(None, label, TimerRequestIssue.ROOM_TARGET)
    # A filler word never hides a clock time ("para las 7 gracias", "de 10 minutos para las 7 ya").
    if _CLOCK_RE.search(" ".join(w for w in rest if w not in _FILLER)):
        return TimerRequest(None, label, TimerRequestIssue.CLOCK_TIME if span is None
                            else TimerRequestIssue.UNPARSEABLE)
    if found is None:
        issue = TimerRequestIssue.UNPARSEABLE if _bare_number(folded, 0, len(folded)) or any(
            w in _TIME_UNITS for w in folded) else TimerRequestIssue.MISSING_DURATION
        return TimerRequest(None, label, issue)
    seconds = found[0]
    if seconds == "vague":
        return TimerRequest(None, label, TimerRequestIssue.VAGUE_DURATION)
    if seconds == "invalid" or _find_duration(rest) is not None or _bare_number(rest, 0, len(rest)) \
            or any(w in _TIME_UNITS for w in rest):
        return TimerRequest(None, label, TimerRequestIssue.UNPARSEABLE)
    if seconds > MAX_TIMER_SECONDS:
        return TimerRequest(None, label, TimerRequestIssue.TOO_LONG)
    return TimerRequest(seconds, label)


def parse_timer_selector(complement: str) -> TimerSelector:
    """Read which timer a cancel/stop/pause/resume/query names.

    Args:
        complement: The household intent's complement, e.g. "de la pasta",
            "de 5 minutos" or "" (no particular timer).
    """
    original, folded = _words(complement)
    found = _find_duration(folded)
    duration = found[0] if found is not None and isinstance(found[0], int) else None
    span = found[1] if duration is not None else None
    return TimerSelector(_label(original, folded, span, state_words=True), duration)


def fold_label(label: str | None) -> str:
    """Comparable form of a label (accent-folded words separated by one space)."""
    return " ".join(_words(label or "")[1])


def format_duration(seconds: float) -> str:
    """Spanish duration for voice, whole seconds rounded up ("1 hora y 30 minutos").

    Args:
        seconds: A non-negative duration.
    """
    total = max(0, int(-(-seconds // 1)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{value} {singular if value == 1 else plural}"
             for value, singular, plural in ((hours, "hora", "horas"), (minutes, "minuto", "minutos"),
                                             (secs, "segundo", "segundos")) if value]
    if not parts:
        return "0 segundos"
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " y " + parts[-1]
