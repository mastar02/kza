"""ReminderManager — CRUD, unique-match cancellation, and voice formatting."""
import logging
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from functools import lru_cache

from src.reminders.reminder_store import ReminderStore, Reminder, ReminderScope
from src.reminders.voice_request import spoken_day_at
from src.core.household_clock import HouseholdClock
from src.core.sqlite_work import AuthorityGuard
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionService, request_from_service

logger = logging.getLogger(__name__)

# Words of a spoken cancel complement that never identify a reminder (T032-U1).
_QUERY_STOPWORDS = frozenset({
    "de", "del", "el", "la", "los", "las", "lo", "un", "una", "unos", "unas", "que", "a", "al", "para", "por",
    "con", "en", "mi", "mis", "tu", "tus", "su", "sus", "y", "o", "ese", "esa", "este", "esta", "eso", "esto",
    "dice", "diga", "decia", "llama", "sobre", "recordatorio", "recordatorios", "todos", "todas", "nuestro",
    "nuestra", "hs", "horas", "hora",
})
_HOUR_WORDS = {"una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8,
               "nueve": 9, "diez": 10, "once": 11, "doce": 12}
_MINUTE_WORDS = {"media": 30, "cuarto": 15}
_CLOCK_DIGITS = re.compile(r"^(\d{1,2})(?:[:.](\d{2}))?$")
_WORD = re.compile(r"\d{1,2}[:.]\d{2}|\d+|[^\W\d_]+")
_AM_PERIODS = frozenset({"manana", "madrugada"})
_PM_PERIODS = frozenset({"tarde", "noche", "mediodia"})
# "las 12 de la noche" is midnight and "las 2 de la noche" is 02:00, not 14:00.
_NIGHT_EARLY_HOURS = frozenset({12, 1, 2, 3, 4})
# T032-U1 fail-safe: temporal words that are never plain content words of a spoken cancel.
# A clock, day or period phrase reads them; any one no phrase reads is text or any time.
_PERIOD_WORDS = frozenset({"manana", "tarde", "noche", "madrugada"})
_PERIOD_LEADS = frozenset({"a", "por", "en", "de", "para", "hasta", "desde", "toda"})
_MINUTE_NUMBER_WORDS = frozenset({
    "trece", "catorce", "quince", "dieciseis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiun",
    "veintiuno", "veintiuna", "veintidos", "veintitres", "veinticuatro", "veinticinco", "veintiseis",
    "veintisiete", "veintiocho", "veintinueve", "treinta", "cuarenta", "cincuenta",
})
_TIME_VOCABULARY = frozenset({
    "manana", "mananas", "tarde", "tardes", "noche", "noches", "madrugada", "madrugadas", "mediodia",
    "medianoche", "hoy", "pasado", "proximo", "proxima", "lunes", "martes", "miercoles", "jueves", "viernes",
    "sabado", "sabados", "domingo", "domingos", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
    "agosto", "septiembre", "setiembre", "octubre", "noviembre", "diciembre",
})
# Connecting words that join temporal words into one unread run ("las 2 y 10 de la tarde").
_TIME_GLUE = frozenset({"y", "de", "del", "la", "las", "a", "al", "el", "en", "hs", "horas", "hora", "por", "para"})
# Weekday names a clarification speaks ("del miércoles"), Monday first.
_SPOKEN_WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
# A leading relative clause names how the reminder was made, not what it says:
# "que (te) pedí", "que puse", "que hay" (review T032-U1-R6).
_RELATIVE_PRONOUNS = frozenset({"te", "le", "les", "me", "nos", "se"})
_RELATIVE_VERBS = frozenset({
    "puse", "pusiste", "puso", "pusimos", "pedi", "pediste", "pidio", "pedimos", "anote", "anotaste", "anoto",
    "anotamos", "agende", "agendaste", "agendo", "agendamos", "programe", "programaste", "guarde", "guardaste",
    "tenia", "tenias", "teniamos", "tengo", "tenemos", "hay", "habia",
})


def _fold(text: str) -> str:
    """Lowercase and strip accents."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _forms(word: str) -> frozenset[str]:
    """The word and its singular candidates ("noches" -> "noche", "flores" -> "flor")."""
    forms = {word}
    for suffix in ("s", "es"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            forms.add(word[:-len(suffix)])
    return frozenset(forms)


def _content(words: list[str]) -> tuple[frozenset[str], ...]:
    """Identifying words of a text, each as its set of plural-insensitive forms."""
    return tuple(dict.fromkeys(_forms(w) for w in words if w not in _QUERY_STOPWORDS))


def _covers(text_words: tuple[frozenset[str], ...], query_words: tuple[frozenset[str], ...]) -> bool:
    """Every query word is a word of the text."""
    return all(any(q & t for t in text_words) for q in query_words)


@dataclass(frozen=True)
class _SpokenClock:
    """Clock time named in a cancel request ("de las 6 de la tarde")."""

    hour: int
    minute: int = 0
    period: str | None = None

    def wall_times(self) -> frozenset[tuple[int, int]]:
        """The household wall times ``(hour, minute)`` this clock can mean.

        Without a period a 1-12 hour is either half of the day. "de la
        noche" moves 12 and 1-4 past midnight ("las 12 de la noche" is
        00:00); "del mediodía" and "de la tarde" keep 12 at noon. An hour
        past 12 is that hour ("las 20" is "las 8 de la noche").
        """
        if self.hour > 12:
            return frozenset({(self.hour, self.minute)})
        base = self.hour % 12
        if self.period in _AM_PERIODS or (self.period == "noche" and self.hour in _NIGHT_EARLY_HOURS):
            return frozenset({(base, self.minute)})
        if self.period in _PM_PERIODS:
            return frozenset({(base + 12, self.minute)})
        return frozenset({(base, self.minute), (base + 12, self.minute)})

    def matches(self, hour: int, minute: int) -> bool:
        """Whether a household wall time ``hour:minute`` is this spoken time (see :meth:`wall_times`)."""
        return (hour, minute) in self.wall_times()


def _in_period(period: str, hour: int) -> bool:
    """Whether a household hour lies in a period said without a clock ("a la tarde").

    Assistant default T032-U1-FS1 (over-asking acceptable): madrugada
    00-05, mañana 00-11 (the morning of accepted default Q3), tarde
    12-20, noche 20-05. The windows overlap (20:xx is tarde and noche),
    so a doubt asks. A clock with its period keeps its own wall times
    (accepted default Q2: "las 5 de la noche" is 17:00).
    """
    if period == "madrugada":
        return hour < 6
    if period == "manana":
        return hour < 12
    if period == "tarde":
        return 12 <= hour <= 20
    return period == "noche" and (hour >= 20 or hour < 6)


def _periods_overlap(first: str, second: str) -> bool:
    """Two periods of the day share an hour ("madrugada" and "noche")."""
    return any(_in_period(first, hour) and _in_period(second, hour) for hour in range(24))


def _clock_at(words: list[str], i: int) -> tuple[_SpokenClock, int, int] | None:
    """A clock time at token ``i``: the clock and the token span it covers.

    Accepted: "10:30" anywhere, and "las 10"/"la una" with an optional
    "y media"/"y cuarto" and "de la mañana/tarde/noche/madrugada" or
    "del mediodía". A bare "las 2" followed by a content word is a number
    of the reminder ("tomar las 2 pastillas"), not a time.
    """
    word = words[i]
    article = i > 0 and words[i - 1] in ("las", "la")
    digits = _CLOCK_DIGITS.match(word)
    if digits is not None and (digits.group(2) is not None or article):
        hour, minute = int(digits.group(1)), int(digits.group(2) or 0)
    elif word in _HOUR_WORDS and article:
        hour, minute = _HOUR_WORDS[word], 0
    else:
        return None
    if hour > 23 or minute > 59:
        return None
    end = i + 1
    if digits is None or digits.group(2) is None:
        if words[end:end + 1] == ["y"] and words[end + 1:end + 2] and words[end + 1] in _MINUTE_WORDS:
            minute = _MINUTE_WORDS[words[end + 1]]
            end += 2
    period = None
    if words[end:end + 2] == ["de", "la"] and words[end + 2:end + 3] \
            and words[end + 2] in _AM_PERIODS | _PM_PERIODS:
        period = words[end + 2]
        end += 3
    elif words[end:end + 2] == ["del", "mediodia"]:
        period = "mediodia"
        end += 2
    bare = end == i + 1 and (digits is None or digits.group(2) is None)
    if bare and words[end:end + 1] and words[end] not in _QUERY_STOPWORDS and spoken_day_at(words, end) is None:
        return None
    return _SpokenClock(hour, minute, period), i - 1 if article else i, end


@dataclass(frozen=True)
class _SpokenDay:
    """Day named in a cancel request ("de hoy", "del lunes")."""

    kind: str
    value: int

    def matches(self, day: date, today: date) -> bool:
        """Whether a reminder's household trigger date is this day.

        A weekday matches any date on that weekday ("el lunes que viene"
        and "este lunes" alike); more than one such reminder asks.
        """
        if self.kind == "weekday":
            return day.weekday() == self.value
        return day == today + timedelta(days=self.value)


@dataclass(frozen=True)
class _SpokenPeriod:
    """Period of the day said without a clock ("a la tarde", "de noche", "esta noche").

    Attributes:
        name: "manana", "tarde", "noche" or "madrugada" (folded).
        today: Said with "esta" ("esta tarde"): the period of today.
    """

    name: str
    today: bool = False

    def matches(self, when: datetime, today: date) -> bool:
        """Whether a household trigger time lies in this period (see :func:`_in_period`)."""
        return _in_period(self.name, when.hour) and (not self.today or when.date() == today)

    @property
    def words(self) -> frozenset[str]:
        """The words a text holds when it says this period ("esta" and "tarde" for "esta tarde")."""
        return frozenset({self.name, "esta"} if self.today else {self.name})


def _is_number(word: str) -> bool:
    """A number token: digits or a number word up to the minutes ("2", "dos", "veinte")."""
    return word.isdecimal() or word in _HOUR_WORDS or word in _MINUTE_NUMBER_WORDS


def _impossible_time(word: str, prev: str) -> bool:
    """Digits no time can be: "8:75", or "las 25" (no hour, and a date is "el/del 25")."""
    clock = _CLOCK_DIGITS.match(word)
    if clock is not None and clock.group(2) is not None:
        return int(clock.group(1)) > 23 or int(clock.group(2)) > 59
    return word.isdecimal() and prev in ("la", "las") and int(word) > 23


def _is_time_word(words: list[str], k: int) -> bool:
    """Token ``k`` belongs to the temporal vocabulary (T032-U1 fail-safe).

    Digits, number words, periods of the day, "mediodía"/"medianoche",
    day words ("hoy", weekdays, "pasado", "próximo") and month names;
    with their context only, "una" after "la(s)", "y/menos cuarto/media",
    "menos" after a number, "en punto" and "que viene".
    """
    word = words[k]
    prev = words[k - 1] if k > 0 else ""
    if any(c.isdigit() for c in word):
        return not _impossible_time(word, prev)
    if word in _TIME_VOCABULARY:
        return True
    if word == "una":
        return prev in ("la", "las")
    if word in _HOUR_WORDS or word in _MINUTE_NUMBER_WORDS:
        return True
    if word in ("cuarto", "media"):
        return prev in ("y", "menos")
    if word == "menos":
        return _is_number(prev)
    if word == "punto":
        return prev == "en"
    return word == "viene" and prev == "que"


@dataclass(frozen=True)
class _Run:
    """Temporal words no clock, day or period phrase reads ("las 2 y 10", "del 20", "mediodía").

    Attributes:
        span: The token span, without connecting words at its ends; a
            clock, day or period phrase inside it is not read on its own.
        number: The run is a single number, not after "el"/"del" ("tomar
            10 gotas", "tomar las 2 pastillas"): in a reminder text it is a
            number of the text (reviews R2, FV2-4), compared only as words.
    """

    span: range
    number: bool


@dataclass(frozen=True)
class _Reading:
    """Every time phrase of folded tokens, each with the token span it covers.

    Attributes:
        clocks: Clock phrases outside any unread run.
        days: Day phrases outside any unread run.
        periods: Periods said without a clock, outside any unread run
            (a bare "de (la) mañana" is the morning, accepted default
            T032-U1-Q3).
        runs: Unread temporal words (T032-U1 fail-safe).
    """

    clocks: tuple[tuple[_SpokenClock, range], ...]
    days: tuple[tuple[_SpokenDay, range], ...]
    periods: tuple[tuple[_SpokenPeriod, range], ...]
    runs: tuple[_Run, ...]

    @property
    def taken(self) -> set[int]:
        """Indexes of the tokens inside any time phrase or unread run."""
        spans = ([span for _, span in self.clocks] + [span for _, span in self.days]
                 + [span for _, span in self.periods] + [run.span for run in self.runs])
        return {k for span in spans for k in span}

    @property
    def unread_time(self) -> bool:
        """Some unread run can be any time (not only a single number)."""
        return any(not run.number for run in self.runs)


def _read(words: list[str]) -> _Reading:
    """Every clock time, day phrase, period and unread temporal word, left to right.

    Each later phrase is read like the first (review T032-U1-PROP-1/2):
    in "tomar pastillas a la una de las 2" both clocks are clocks, so each
    can be the due time or words of the text, and neither becomes a plain
    content word. A period said without a clock ("a la tarde", "de noche",
    "esta noche", the bare "de mañana" of accepted default T032-U1-Q3) is
    a period; right after a clock and "de", "mañana" is the day ("de las 9
    de la noche de mañana").

    Fail-safe (T032-U1): any temporal word no phrase reads ("las 2 y 10",
    "las 3 menos cuarto", "del mediodía", "del 20", "el 20 de noviembre")
    joins its neighbouring time words, through connecting words, into an
    unread run; a phrase inside a run is not read on its own, since the
    run may name another time ("las 2" of "las 2 y 10").
    """
    n = len(words)
    clocks, clock_tokens, i = [], set(), 0
    while i < n:
        found = _clock_at(words, i)
        if found is None:
            i += 1
            continue
        clock, start, end = found
        clocks.append((clock, range(start, end)))
        clock_tokens.update(range(start, end))
        i = end
    clock_ends = {span.stop for _, span in clocks}
    days, i = [], 0
    while i < n:
        joined = i - 1 in clock_ends and words[i - 1] == "de"
        found = None if i in clock_tokens else spoken_day_at(words, i, joined=joined)
        if found is None or clock_tokens & set(range(i, found[2])):
            i += 1
            continue
        kind, value, end = found
        days.append((_SpokenDay(kind, value), range(i, end)))
        i = end
    taken = clock_tokens | {k for _, span in days for k in span}
    periods = []
    for k, word in enumerate(words):
        if word not in _PERIOD_WORDS or k in taken or k == 0 or k - 1 in taken:
            continue
        prev = words[k - 1]
        if prev == "la":
            start = k - 2 if k >= 2 and words[k - 2] in _PERIOD_LEADS and k - 2 not in taken else k - 1
            period = _SpokenPeriod(word)
        elif prev in ("esta", "de"):
            start, period = k - 1, _SpokenPeriod(word, today=prev == "esta")
        else:
            continue
        periods.append((period, range(start, k + 1)))
        taken.update(range(start, k + 1))
    seeds = {k for k in range(n) if k not in taken and _is_time_word(words, k)}
    timeish = [k in taken or k in seeds or words[k] in _TIME_GLUE for k in range(n)]
    runs, k = [], 0
    while k < n:
        if not timeish[k]:
            k += 1
            continue
        end = k
        while end < n and timeish[end]:
            end += 1
        if seeds & set(range(k, end)):
            first, last = k, end
            while first < last and first not in taken and first not in seeds:
                first += 1
            while last > first and last - 1 not in taken and last - 1 not in seeds:
                last -= 1
            number = last - first == 1 and _is_number(words[first]) and (first == 0 or words[first - 1] not in ("el", "del"))
            runs.append(_Run(range(first, last), number))
        k = end
    in_runs = {k for run in runs for k in run.span}
    return _Reading(tuple((c, s) for c, s in clocks if not in_runs & set(s)),
                    tuple((d, s) for d, s in days if not in_runs & set(s)),
                    tuple((p, s) for p, s in periods if not in_runs & set(s)),
                    tuple(runs))


def _without_relative_clause(words: list[str]) -> list[str]:
    """Drop a leading "que (te) pedí/puse/hay" ("que te pedí de tomar agua")."""
    if words[:1] != ["que"]:
        return words
    j = 2 if words[1:2] and words[1] in _RELATIVE_PRONOUNS else 1
    return words[j + 1:] if words[j:j + 1] and words[j] in _RELATIVE_VERBS else words


@dataclass(frozen=True)
class _ParsedQuery:
    """A spoken cancel complement split into content words and time phrases.

    Attributes:
        words: Content words outside the time phrases and unread runs.
        clocks: Every spoken clock time, each with its phrase as spoken
            tokens ("la una", "las 2 de la tarde") for the literal reading.
        days: Every spoken day, each with its phrase from its day word
            ("mañana", "lunes" for "del lunes") for the literal reading.
        periods: Every period said without a clock ("a la tarde", the
            bare "de mañana" of accepted default T032-U1-Q3).
        runs: Unread temporal words as spoken tokens ("las 2 y 10"): text
            or any time (T032-U1 fail-safe).
    """

    words: tuple[frozenset[str], ...]
    clocks: tuple[tuple[_SpokenClock, tuple[str, ...]], ...] = ()
    days: tuple[tuple[_SpokenDay, tuple[str, ...]], ...] = ()
    periods: tuple[_SpokenPeriod, ...] = ()
    runs: tuple[tuple[str, ...], ...] = ()


def _parse_query(query: str) -> _ParsedQuery:
    """Content words, every clock time, day and period, and the unread time words of a spoken cancel complement."""
    words = _without_relative_clause(_WORD.findall(_fold(query)))
    reading = _read(words)
    days = []
    for day, span in reading.days:
        # "del lunes" is said of a text "el lunes" too: the day phrase starts at its day word.
        day_words = [words[k] for k in span]
        while day_words and day_words[0] in _QUERY_STOPWORDS:
            day_words.pop(0)
        days.append((day, tuple(day_words)))
    clocks = tuple((clock, tuple(words[k] for k in span)) for clock, span in reading.clocks)
    runs = tuple(tuple(words[k] for k in run.span) for run in reading.runs)
    taken = reading.taken
    content = _content([w for k, w in enumerate(words) if k not in taken])
    return _ParsedQuery(content, clocks, tuple(days), tuple(p for p, _ in reading.periods), runs)


def _hour_of(word: str) -> int | None:
    """The hour an hour word or plain number names ("una" -> 1, "1" -> 1, "13" -> 13)."""
    if word in _HOUR_WORDS:
        return _HOUR_WORDS[word]
    return int(word) if word.isdecimal() else None


def _same_token(text_word: str, phrase_word: str, after_article: bool) -> bool:
    """Tokens are equal; after "la"/"las" an hour word equals its digits ("la una" = "la 1")."""
    if text_word == phrase_word:
        return True
    if not after_article:
        return False
    hour = _hour_of(phrase_word)
    return hour is not None and hour == _hour_of(text_word)


def _phrase_at(text: list[str], k: int, phrase: tuple[str, ...]) -> bool:
    """The folded text tokens hold ``phrase`` at ``k`` (review FV2-2: "las dos" = "las 2")."""
    return len(text) - k >= len(phrase) and all(
        _same_token(text[k + j], word, j > 0 and phrase[j - 1] in ("la", "las")) for j, word in enumerate(phrase))


def _contains_phrase(text: list[str], phrase: tuple[str, ...]) -> bool:
    """The folded text tokens hold ``phrase`` as contiguous tokens."""
    return len(phrase) > 0 and any(_phrase_at(text, k, phrase) for k in range(len(text) - len(phrase) + 1))


def _text_periods(text: list[str]) -> set[str]:
    """Periods of the day a folded text names ("de la tarde", "por la noche", "mediodía")."""
    return {word for k, word in enumerate(text)
            if word == "mediodia" or (word in _AM_PERIODS | _PM_PERIODS and k > 0 and text[k - 1] == "la")}


def _holds_same_clock(text: list[str], reading: _Reading, clock: _SpokenClock) -> bool:
    """A clock time of the text can be the spoken clock, however either is written.

    Each read clock of the text (not one inside an unread run) is read
    with the query's own grammar, so "8:30", "8.30", "08:30" and "8 y
    media" are one clock (review T032-U1-FV3-1), and both sides are
    compared as the wall times they can mean, so "las 20" is "las 8 de
    la noche" and "las 2" is "las 14" (review T032-U1-FV3-2). A text
    clock without a period of its own takes the periods the text names
    elsewhere ("tomar pastillas a las 2 por la noche" is 02:00, not "las
    2 de la tarde"); with none it is either half of the day ("tomar
    pastillas a las 2" is said "... a las 2 de la tarde" too, review
    FV2-3).
    """
    wanted = clock.wall_times()
    periods = _text_periods(text)
    for own, _ in reading.clocks:
        if own.period is None and periods:
            means = frozenset().union(*(_SpokenClock(own.hour, own.minute, p).wall_times() for p in periods))
        else:
            means = own.wall_times()
        if means & wanted:
            return True
    return False


def _holds_same_day(reading: _Reading, day: _SpokenDay) -> bool:
    """A read day of the text is the spoken day ("el lunes próximo" = "el próximo lunes")."""
    return any(own == day for own, _ in reading.days)


def _holds_same_period(reading: _Reading, period: _SpokenPeriod) -> bool:
    """A period the text says without a clock shares an hour with the spoken one."""
    return any(not own.today and _periods_overlap(own.name, period.name) for own, _ in reading.periods)


def _without_time_words(text: str) -> str | None:
    """A reminder text without the words the resolver would read as a time.

    "llamar a mamá a las 3" -> "llamar a mamá"; "llevar el auto al taller
    mañana" -> "llevar el auto al taller"; "tomar pastillas a las 2 y 10"
    -> "tomar pastillas". Clock, day and period phrases go, and so do
    unread time words, except a single number ("tomar 10 gotas", "tomar
    las 2 pastillas"). Articles and prepositions left hanging at a cut are
    dropped. None when nothing is read as a time or the folded and spoken
    tokens do not line up.
    """
    spoken = _WORD.findall(unicodedata.normalize("NFC", text.lower()))
    folded = ["de"] + [_fold(word) for word in spoken]
    if folded[1:] != _WORD.findall(_fold(text)):
        return None
    reading = _read(folded)
    taken = reading.taken - {k for run in reading.runs if run.number for k in run.span}
    if not taken:
        return None
    runs, run = [], []
    for k, word in enumerate(spoken, start=1):
        if k in taken:
            runs.append(run)
            run = []
        else:
            run.append((word, folded[k]))
    runs.append(run)
    kept = []
    for n, run in enumerate(runs):
        while n > 0 and run and run[0][1] in _QUERY_STOPWORDS:
            run = run[1:]
        while n < len(runs) - 1 and run and run[-1][1] in _QUERY_STOPWORDS:
            run = run[:-1]
        kept.extend(word for word, _ in run)
    return " ".join(kept) or None


@lru_cache(maxsize=4096)
def _text_reading(text: str) -> tuple[tuple[str, ...], _Reading]:
    """Folded tokens of a reminder text and their time reading."""
    words = _WORD.findall(_fold(text))
    return tuple(words), _read(words)


def _reads_as(query: _ParsedQuery, text: list[str], reading: _Reading, local: datetime, today: date) -> bool:
    """The reminder fits the reference with every unread time word read as text.

    Every content word is a word of the text and every unread run is
    contiguous words of it. Each clock, day and period phrase is the
    trigger (clock by wall times, day by household date, period by its
    hours, "esta tarde" today) or words of the text: the contiguous
    phrase, the same clock or day as a read one of the text, the period
    word (or a period of the text sharing an hour). A text's clock or
    day inside an unread run of the text is only read as words.
    """
    if not _covers(_content(text), query.words):
        return False
    if not all(_contains_phrase(text, run) for run in query.runs):
        return False
    if not all(clock.matches(local.hour, local.minute) or _contains_phrase(text, phrase)
               or _holds_same_clock(text, reading, clock) for clock, phrase in query.clocks):
        return False
    if not all(day.matches(local.date(), today) or _contains_phrase(text, phrase) or _holds_same_day(reading, day)
               for day, phrase in query.days):
        return False
    return all(period.matches(local, today) or period.words <= set(text)
               or (not period.today and _holds_same_period(reading, period)) for period in query.periods)


def _may_mean(query: _ParsedQuery, text: list[str], reading: _Reading, local: datetime, today: date) -> bool:
    """Some reading of the reference could mean this reminder (the fail-safe's wider set).

    As :func:`_reads_as`, except that an unread run of the reference fits
    any reminder whose text holds the content words (its words may be any
    time), every phrase fits a text that has unread time words of its own
    (they may be that time), and a period fits a text naming its word or
    any period sharing an hour with it.
    """
    if not _covers(_content(text), query.words):
        return False
    if reading.unread_time:
        return True
    if not all(clock.matches(local.hour, local.minute) or _contains_phrase(text, phrase)
               or _holds_same_clock(text, reading, clock) for clock, phrase in query.clocks):
        return False
    if not all(day.matches(local.date(), today) or _contains_phrase(text, phrase) or _holds_same_day(reading, day)
               for day, phrase in query.days):
        return False
    return all(period.matches(local, today) or period.name in text or _holds_same_period(reading, period)
               for period in query.periods)


class CancelMatch(StrEnum):
    """How a spoken reminder reference resolved among the shared active ones."""

    UNIQUE = "unique"
    NONE_ACTIVE = "none_active"
    NO_REFERENCE = "no_reference"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class CancelResolution:
    """Result of resolving a spoken reference to exactly one reminder.

    Attributes:
        match: How the reference resolved.
        reminder: The single match, only for ``CancelMatch.UNIQUE``.
        candidates: Every eligible candidate for a clarification (all active
            ones for ``NO_REFERENCE``, the tied matches for ``AMBIGUOUS``),
            in trigger order.
    """

    match: CancelMatch
    reminder: Reminder | None = None
    candidates: tuple[Reminder, ...] = field(default_factory=tuple)


class ReminderManager:
    """High-level reminder operations with fuzzy search and voice output."""

    def __init__(self, store: ReminderStore, config: dict | None = None, *,
                 home_actions: HomeActionService | None = None,
                 clock: Callable[[], float] | None = None,
                 household_clock: HouseholdClock | None = None):
        """Build the manager.

        Args:
            store: Durable reminder store.
            config: Reminder section of the settings.
            home_actions: Admission service for reminder home actions.
            clock: Epoch-seconds clock for tests, on the supported household
                timezone. Exclusive with ``household_clock``.
            household_clock: The application's household clock (P3-F1);
                "today" and spoken times are on its local calendar.

        Raises:
            ValueError: Both ``clock`` and ``household_clock`` were given.
        """
        if clock is not None and household_clock is not None:
            raise ValueError("Pass either clock or household_clock, not both")
        self._store = store
        self._household_clock = household_clock or HouseholdClock(time_source=clock or time.time)
        self._config = config or {}
        self.home_actions = home_actions if home_actions is not None else HomeActionService(None)
        self._tts_prefix = self._config.get("tts_prefix", "Recuerda")
        self._default_time = self._config.get("default_time", "09:00")

    async def create(
        self,
        user_id: str,
        text: str,
        trigger_at: float,
        recurrence: str | None = None,
        ha_actions: list[dict] | None = None,
        *, guard: AuthorityGuard | None = None, scope: ReminderScope = ReminderScope.PRIVATE,
    ) -> Reminder:
        """Create a new reminder via the store."""
        scope = ReminderScope(scope)
        for action in ha_actions or []:
            if type(action) is not dict or action.keys() - {"domain", "service", "entity_id", "data"}:
                raise ValueError("invalid_reminder_action")
            request = request_from_service(
                action.get("domain", ""), action.get("service", ""), action.get("entity_id", ""), action.get("data"),
                request_id="admission", origin=ActionOrigin.REMINDER,
            )
            await self.home_actions.validate_selection(request, actor=HomeActor.household())
        reminder = await self._store.create(
            user_id=user_id,
            text=text,
            trigger_at=trigger_at,
            recurrence=recurrence,
            ha_actions=ha_actions,
            guard=guard,
            scope=scope,
        )
        # Household text never goes above DEBUG (design household_conventions.logging).
        logger.info("Created reminder %s (scope=%s) at %s", reminder.id, scope.value,
                    self._household_clock.format_time(trigger_at))
        return reminder

    async def cancel_by_id(self, reminder_id: str, *, guard: AuthorityGuard | None = None,
                           shared_only: bool = False) -> bool:
        """Cancel only within the original authority and selected native scope."""
        return await self._store.cancel(reminder_id, guard=guard, shared_only=shared_only)

    async def cancel_by_text(self, user_id: str | None, search_text: str, *,
                             shared_only: bool = False,
                             guard: AuthorityGuard | None = None) -> bool:
        """Cancel only a unique match within scope; rechecked at cancellation.

        Args:
            user_id: Trusted user for a private view (ignored when ``shared_only``).
            search_text: Spoken or typed reference to the reminder.
            shared_only: Restrict to shared household reminders.
            guard: Authority guard of the caller.

        Returns:
            True only when exactly one active reminder matched and the store
            committed its transition to cancelled.
        """
        resolution = await self.resolve(user_id, search_text, shared_only=shared_only, guard=guard)
        if resolution.match is not CancelMatch.UNIQUE:
            return False
        return await self._store.cancel(resolution.reminder.id, shared_only=shared_only, guard=guard)

    async def resolve(self, user_id: str | None, reference: str, *, shared_only: bool = False,
                      guard: AuthorityGuard | None = None) -> CancelResolution:
        """Resolve a spoken reference to exactly one active reminder (T032-U1).

        Scope is applied by the store before any text is compared. Every
        content word of the reference (accents, articles and plurals ignored)
        must be a word of the reminder's text; every spoken clock time ("de
        las 6 de la tarde") and day ("de hoy", "del lunes", the T031 day
        grammar) must equal its next trigger on the household clock, and a
        bare "de mañana" means a trigger in the morning (T032-U1-Q3). A
        leading "que (te) pedí/puse/hay" clause is not content. The spoken
        clock and day phrases are also read as words of the text, as
        contiguous words ("la una", "las 2 de la tarde"), and "de mañana"
        as the text's word "mañana", each phrase on its own:
        a reminder is a candidate when its clock fits as due time or as
        text, and its day too, mixed or not; when the readings select
        different reminders, all are candidates and KZA asks
        (T032-U1-RR-P1, review FV-1, FV2-1, PROP-1..3). A period said
        without a clock ("a la tarde", "por la noche") is read like "de
        mañana", by the hours of assistant default T032-U1-FS1. Fail-safe:
        a temporal word no phrase reads, in the reference ("de las 2 y 10",
        "del mediodía", "del 20") or in a text ("a las 3 menos cuarto"),
        may be text or any time, so a reminder is cancelled only when it is
        the single one any reading could mean.
        Several matches are always ambiguous, even when one reminder's words
        are exactly the reference's: another active reminder that contains
        every spoken word may be the one meant, so KZA asks rather than
        cancel the wrong one (user answer T032-U1-Q1 = A). There is no
        similarity threshold: a misheard word finds nothing rather than a
        neighbouring reminder.

        Args:
            user_id: Trusted user for a private view (ignored when ``shared_only``).
            reference: The words naming the reminder, e.g. "de tomar agua".
            shared_only: Restrict to shared household reminders.
            guard: Authority guard of the caller.

        Returns:
            The resolution; never a reminder outside the selected scope.
        """
        active = await self._store.get_active_for_user(user_id, shared_only=shared_only, guard=guard)
        return self._resolve_among(active, reference)

    def _resolve_among(self, active: list[Reminder], reference: str) -> CancelResolution:
        """Resolve a reference among already scoped active reminders (see :meth:`resolve`)."""
        if not active:
            return CancelResolution(CancelMatch.NONE_ACTIVE)
        query = _parse_query(reference)
        if not (query.words or query.clocks or query.days or query.periods or query.runs):
            return CancelResolution(CancelMatch.NO_REFERENCE, candidates=tuple(active))
        today = self._household_clock.local_date()
        # The time words may be the reminder's own text ("llamar a mamá a las
        # 3" due at 17:43): read as words they select by text alone (review
        # T032-U1-RR-F1). When that reading and the time reading select
        # different reminders, all are candidates and KZA asks which rather
        # than pick one (user answer T032-U1-RR-P1: "tomar las pastillas de
        # las 2" with "tomar las 2 pastillas" also active asks). The literal
        # reading of a time phrase ("la una", "las 2", "mañana", with its
        # period if one was said) needs it as contiguous words of the text:
        # a repeated number ("tomar 2 pastillas a las 2") or the stopword
        # "una" still count, while "de la una" or "de la una de la tarde"
        # never select a text that only shares "tarde" or no word at all
        # (review T032-U1-FV-1..3). Clock and day are judged apart: "pagar
        # la luz a las 5 del miércoles" is said of "pagar la luz a las 5"
        # due Wednesday at 10 too, so it never cancels "pagar la luz" due
        # Wednesday at 17 alone (review T032-U1-FV2-1). A period said with
        # the clock is read into a text whose own clock has none ("tomar
        # pastillas a las 2" for "... a las 2 de la tarde") unless the text
        # names another period (review T032-U1-FV2-3). The text's clock is read
        # as a clock, so "8:30", "8.30", "08:30" and "8 y media" are one
        # (review T032-U1-FV3-1), and compared by the wall times each side can
        # mean, so "las 20" is "las 8 de la noche" (review T032-U1-FV3-2).
        # Every clock and day phrase is read this way, each on its own, not
        # only the first: "tomar pastillas a la una de las 2" fits "... a las
        # 2" due at 13:00 and "... a la una" due at 14:00 alike, so it asks
        # (review T032-U1-PROP-1/2). A bare "de mañana" is the morning
        # (accepted default T032-U1-Q3): any trigger before noon, or the
        # text's own word "mañana" (review T032-U1-PROP-3); "a la tarde",
        # "por la noche", "de noche", "esta noche" are periods the same way,
        # by the windows of assistant default T032-U1-FS1 (_in_period).
        #
        # Fail-safe (T032-U1, round failsafe_1): a temporal word that no
        # phrase reads, in the reference ("de las 2 y 10", "del mediodía",
        # "del 20") or in a text ("tomar pastillas a las 3 menos cuarto"),
        # may name text or any time. A reminder is cancelled only when it is
        # the single one that fits with every such word read as text
        # (_reads_as) and also the single one that any reading could mean
        # (_may_mean); otherwise KZA asks among every reminder some reading
        # could mean (Q1, RR-P1), and finds nothing only when none could.
        strict, possible = [], []
        for reminder in active:
            text, reading = _text_reading(reminder.text)
            text = list(text)
            local = self._household_clock.to_local(reminder.trigger_at)
            fits = _reads_as(query, text, reading, local, today)
            if fits:
                strict.append(reminder)
            if fits or _may_mean(query, text, reading, local, today):
                possible.append(reminder)
        if not possible:
            return CancelResolution(CancelMatch.NOT_FOUND)
        if strict and len(possible) == 1:
            return CancelResolution(CancelMatch.UNIQUE, strict[0])
        return CancelResolution(CancelMatch.AMBIGUOUS, candidates=tuple(possible))

    async def voice_references(self, user_id: str | None, candidates: tuple[Reminder, ...], *, with_time: bool,
                               shared_only: bool = False,
                               guard: AuthorityGuard | None = None) -> dict[str, str]:
        """Spoken references that :meth:`resolve` maps back to exactly one candidate.

        A clarification proposes "<text>" or "<text> de <time>", or the
        same without the text's own clock and day words when those keep it
        from selecting itself ("llamar a mamá a las 3" due at 17:43); one is
        kept only when, said back now, it selects that candidate and no
        other active reminder. A reminder whose words and time are all
        inside another one's (or identical to it) gets no reference: by
        voice it cannot be told apart (T032-U1-Q1 = A).

        Args:
            user_id: Trusted user for a private view (ignored when ``shared_only``).
            candidates: Reminders to name, from a previous resolution.
            with_time: Add the unambiguous spoken time of the next trigger.
            shared_only: Restrict to shared household reminders.
            guard: Authority guard of the caller.

        Returns:
            Reminder id -> reference (without the leading "de"), for the
            candidates that have one.
        """
        active = await self._store.get_active_for_user(user_id, shared_only=shared_only, guard=guard)
        references = {}
        for reminder in candidates:
            for reference in self._reference_attempts(reminder, with_time=with_time):
                found = self._resolve_among(active, f"de {reference}")
                if found.match is CancelMatch.UNIQUE and found.reminder.id == reminder.id:
                    references[reminder.id] = reference
                    break
        return references

    def _reference_attempts(self, reminder: Reminder, *, with_time: bool) -> list[str]:
        """Spoken references to try for one reminder, most natural first.

        The whole text, then the text without the words the resolver reads
        as a time or day ("llamar a mamá a las 3" -> "llamar a mamá"). With
        ``with_time`` each carries the spoken time, and then the time and
        day ("de las 5 de la tarde de hoy") for reminders the time alone
        does not tell apart (review T032-U1-RR-F2). There is no 24-hour
        attempt: "de las 14" means the same wall time as "de las 2 de la
        tarde", so it selects no fewer reminders (review T032-U1-FV3-2). A
        trigger at another text's clock ("tomar pastillas" at 14:00 next to
        "tomar pastillas a las 2") then has no reference, and the reply
        sends it to the web. An attempt never says two clocks, or two
        days, when one does: when the text holds a clock, the text without
        its own time words comes first with the time ("llamar a mamá de
        las 5:43 de la tarde", not "llamar a mamá a las 3 de las 5:43 de
        la tarde"), and when it holds a clock or a day, also with the time
        and day ("llamar de las 5 de la tarde de hoy", not "llamar mañana
        ... de hoy"). The whole text is the fallback; each of its clocks
        and days is read on its own, so it names the reminder only when no
        reading fits another one (review T032-U1-PROP-1/2/4).
        """
        bases = [reminder.text]
        content = _without_time_words(reminder.text)
        if content is not None and content != reminder.text:
            bases.append(content)
        if not with_time:
            return bases
        spoken_time = self.spoken_time_reference(reminder)
        spoken_day = self.spoken_day_reference(reminder)
        own = _read(["de"] + _WORD.findall(_fold(reminder.text)))
        clocked = bool(own.clocks) or own.unread_time
        timed = bases[::-1] if clocked else bases
        dated = bases[::-1] if clocked or own.days or own.periods else bases
        return ([f"{base} de {spoken_time}" for base in timed]
                + [f"{base} de {spoken_time} {spoken_day}" for base in dated])

    async def cancel_resolved(self, reminder: Reminder, *, shared_only: bool = False,
                              guard: AuthorityGuard | None = None) -> bool:
        """Cancel a resolved reminder; False when it is no longer active."""
        return await self._store.cancel(reminder.id, shared_only=shared_only, guard=guard)

    async def get_active(self, user_id: str | None, *, shared_only: bool = False,
                         guard: AuthorityGuard | None = None) -> list[Reminder]:
        """Project eligible active jobs without deriving authority from a profile."""
        return await self._store.get_active_for_user(user_id, shared_only=shared_only, guard=guard)

    async def get_today(self, user_id: str | None, *, shared_only: bool = False,
                        guard: AuthorityGuard | None = None) -> list[Reminder]:
        """Return eligible active jobs before the end of the current day."""
        cutoff = self._household_clock.next_local_midnight()
        active = await self._store.get_active_for_user(user_id, shared_only=shared_only, guard=guard)
        return [reminder for reminder in active if reminder.trigger_at < cutoff]

    def format_for_voice(self, reminder: Reminder) -> str:
        """Format only explicitly shared material for unauthenticated household speech."""
        if reminder.scope is not ReminderScope.SHARED:
            raise PermissionError("personal_reminder_voice_session_required")
        trigger_time = self._household_clock.format_time(reminder.trigger_at)
        return f"{self._tts_prefix} {reminder.text} a las {trigger_time}"

    def spoken_time_reference(self, reminder: Reminder) -> str:
        """Spoken time that :meth:`resolve` reads back as exactly this trigger.

        The period removes the half-day doubt: "las 8 de la mañana", "las
        8:30 de la noche", "las 12 del mediodía", "la 1 de la madrugada".
        """
        local = self._household_clock.to_local(reminder.trigger_at)
        hour = local.hour % 12 or 12
        if local.hour == 0:
            period = "de la noche"
        elif local.hour < 6:
            period = "de la madrugada"
        elif local.hour < 12:
            period = "de la mañana"
        elif local.hour == 12:
            period = "del mediodía"
        elif local.hour < 20:
            period = "de la tarde"
        else:
            period = "de la noche"
        clock = f"{hour}:{local.minute:02d}" if local.minute else str(hour)
        return f"{'la' if hour == 1 else 'las'} {clock} {period}"

    def spoken_day_reference(self, reminder: Reminder) -> str:
        """Spoken day that :meth:`resolve` reads back as the trigger's household date.

        "de hoy", "de pasado mañana" or "del <weekday>". Never "de mañana",
        which the T031 day grammar reads as the morning; tomorrow is named
        by its weekday.
        """
        day = self._household_clock.to_local(reminder.trigger_at).date()
        offset = (day - self._household_clock.local_date()).days
        if offset == 0:
            return "de hoy"
        if offset == 2:
            return "de pasado mañana"
        return f"del {_SPOKEN_WEEKDAYS[day.weekday()]}"

    def voice_label(self, reminder: Reminder) -> str:
        """Short spoken name of a shared reminder: its text and next time."""
        if reminder.scope is not ReminderScope.SHARED:
            raise PermissionError("personal_reminder_voice_session_required")
        return f"{reminder.text} a las {self._household_clock.format_time(reminder.trigger_at)}"
