"""Deterministic answers to "¿qué hora es?" and "¿qué día es?". Pure, no I/O.

T037-U3: the household clock (P3-F1, built once from ``routines.timezone``)
answers spoken time and date questions without any language model. Only a
whole utterance that is a time or date question about here and now is
recognized; anything with another subject ("¿a qué hora es el partido?",
"¿qué día es Navidad?", "¿qué hora es en Madrid?") is left to the existing
paths. Weekday and month names are fixed Spanish words, never the process
locale.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
          "agosto", "septiembre", "octubre", "noviembre", "diciembre")
CLOCK_UNAVAILABLE = "No pude leer el reloj de la casa."

# Horizon word (accents stripped) -> days from the household's today.
_HORIZONS = {"hoy": 0, "manana": 1, "pasado manana": 2, "ayer": -1}
# Spoken prefix for each horizon and whether the answer is in the past.
_DAY_PREFIX = {0: "Hoy es", 1: "Mañana es", 2: "Pasado mañana es", -1: "Ayer fue"}

# Wake words used when the caller passes none. The dispatcher passes the
# grammar's WAKE_TERMS plus the configured rooms.wake_word.words, because the
# transcript still carries the wake pre-roll and Whisper often hears "nexa" as
# one of its aliases ("alexa", "nexia", "kaza"...) (review T037-U3-R1).
DEFAULT_WAKE_WORDS = ("nexa",)
# Interjections and courtesy around the question. Stripped only at the edges,
# repeatedly and together with the wake words, so the remaining text must
# still be the whole question.
_LEADING_FILLER = re.compile(r"^(?:che|hola|oye|ey|bueno|a ver|disculpa|perdon)\b ?")
_TRAILING_FILLER = re.compile(r" ?\b(?:che|por favor|porfa|porfavor)$")
# Request lead-ins: "decime/me decís/sabés ... qué hora es".
_LEAD = (r"(?:decime|dime|deci|me decis|me dices|me dirias|me podes decir|me podrias decir|me puedes decir"
         r"|podes decirme|podrias decirme|puedes decirme|sabes|sabe|sabes decirme|queria saber|quiero saber)")
# Harmless here-and-now adverbs after the time question (review T037-U3-R2):
# "¿qué hora es hoy?", "¿qué hora es en este momento?", "¿qué hora es ahora mismo?".
_NOW = r"(?: (?:ahora(?: mismo)?|en este momento|hoy))?"
_TIME = re.compile(
    rf"(?:{_LEAD} )?(?:ahora )?que horas? (?:es|son|tenes|tenemos){_NOW}"
    rf"|{_LEAD} la hora(?: que es)?{_NOW}"
)
_HORIZON = r"(?P<h>hoy|pasado manana|manana|ayer)"
_HORIZON_FIRST = r"(?P<h0>hoy|pasado manana|manana|ayer)"
_DAY_NOUN = r"(?:dia(?: de la semana)?|fecha)"
_DAY_VERB = r"(?:es|seria|sera|fue|era|tenemos|estamos)"
_DATE = re.compile(
    rf"(?:{_LEAD} )?(?:{_HORIZON_FIRST} )?que {_DAY_NOUN} {_DAY_VERB}(?: {_HORIZON})?"
    rf"|(?:{_LEAD} )?(?:a|en) que {_DAY_NOUN} estamos(?: (?P<h1>hoy))?"
    rf"|(?:{_LEAD} )?a cuanto estamos(?: (?P<h2>hoy))?"
    rf"|{_LEAD} (?:la fecha|el dia)(?: de (?P<h3>hoy|pasado manana|manana|ayer))?"
)


class ClockQueryKind(StrEnum):
    """What the household clock is asked for."""

    TIME = "time"
    DATE = "date"


@dataclass(frozen=True)
class ClockQuery:
    """A recognized time or date question.

    Attributes:
        kind: Time of day or calendar date.
        day_offset: Days from the household's today (date questions only).
        with_year: The question asked for "la fecha", so the year is spoken.
    """

    kind: ClockQueryKind
    day_offset: int = 0
    with_year: bool = False


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", plain).split())


def _wake_sequences(wake_words: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    """Normalized word sequences of the wake words; non-strings and blanks are ignored."""
    if isinstance(wake_words, str):
        wake_words = (wake_words,)
    sequences = {tuple(normalize(word).split()) for word in wake_words if isinstance(word, str)}
    return tuple(sorted(sequence for sequence in sequences if sequence))


def _strip_fillers(text: str, wakes: tuple[tuple[str, ...], ...]) -> str:
    previous = None
    while previous != text:
        previous = text
        words = _TRAILING_FILLER.sub("", _LEADING_FILLER.sub("", text)).strip().split()
        for wake in wakes:
            if tuple(words[:len(wake)]) == wake:
                words = words[len(wake):]
            if len(words) >= len(wake) and tuple(words[-len(wake):]) == wake:
                words = words[:-len(wake)]
        text = " ".join(words)
    return text


def parse_clock_query(text: str, wake_words: Iterable[str] = DEFAULT_WAKE_WORDS) -> ClockQuery | None:
    """Recognize a whole-utterance question about the household's time or date.

    Args:
        text: Utterance as transcribed (any case, accents, punctuation).
        wake_words: Wake words (and the aliases the STT hears for them)
            stripped, like interjections, from both edges of the utterance.

    Returns:
        The query, or None when the utterance is anything else.
    """
    if type(text) is not str:
        return None
    core = _strip_fillers(normalize(text), _wake_sequences(wake_words))
    if _TIME.fullmatch(core):
        return ClockQuery(ClockQueryKind.TIME)
    match = _DATE.fullmatch(core)
    if match is None:
        return None
    horizons = [value for value in match.groupdict().values() if value]
    if len(set(horizons)) > 1:
        return None  # "hoy qué día es mañana": two different days asked at once
    return ClockQuery(ClockQueryKind.DATE, _HORIZONS[horizons[0]] if horizons else 0,
                      with_year="fecha" in core.split())


def describe_clock(query: ClockQuery, now_local: datetime) -> str:
    """One spoken Spanish sentence answering ``query`` at ``now_local``.

    Args:
        query: The recognized question.
        now_local: Aware household wall time read once for the turn.

    Raises:
        ValueError: ``now_local`` is naive, so its calendar is unknown.
    """
    if now_local.tzinfo is None or now_local.utcoffset() is None:
        raise ValueError("Household time must be aware")
    if query.kind is ClockQueryKind.TIME:
        lead = "Es la" if now_local.hour == 1 else "Son las"
        return f"{lead} {now_local.hour}:{now_local.minute:02d}."
    # Calendar arithmetic on the local date, never on a fixed 24 h offset.
    day = now_local.date() + timedelta(days=query.day_offset)
    year = f" de {day.year}" if query.with_year else ""
    return f"{_DAY_PREFIX[query.day_offset]} {WEEKDAYS[day.weekday()]} {day.day} de {MONTHS[day.month - 1]}{year}."
