"""ReminderManager — CRUD, unique-match cancellation, and voice formatting."""
import logging
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum

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

    def matches(self, hour: int, minute: int) -> bool:
        """Whether a household wall time ``hour:minute`` is this spoken time.

        Without a period a 1-12 hour is either half of the day. "de la
        noche" moves 12 and 1-4 past midnight ("las 12 de la noche" is
        00:00); "del mediodía" and "de la tarde" keep 12 at noon.
        """
        if minute != self.minute:
            return False
        if self.hour > 12:
            return hour == self.hour
        base = self.hour % 12
        if self.period in _AM_PERIODS or (self.period == "noche" and self.hour in _NIGHT_EARLY_HOURS):
            return hour == base
        if self.period in _PM_PERIODS:
            return hour == base + 12
        return hour in (base, base + 12)


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
        words: Content words outside the clock and day phrases.
        clock: The spoken clock time, if any.
        day: The spoken day, if any.
        clock_phrase: The clock phrase as spoken tokens ("la una", "las 2
            de la tarde"), for the literal reading; empty without a clock.
        day_phrase: The day phrase from its day word ("mañana", "lunes"
            for "del lunes"), for the literal reading; empty without a day.
    """

    words: tuple[frozenset[str], ...]
    clock: _SpokenClock | None
    day: _SpokenDay | None
    clock_phrase: tuple[str, ...] = ()
    day_phrase: tuple[str, ...] = ()


def _parse_query(query: str) -> _ParsedQuery:
    """Content words, an optional clock time and day of a spoken cancel complement."""
    words = _without_relative_clause(_WORD.findall(_fold(query)))
    clock, clock_span, day, day_span = _spans(words)
    taken = set(clock_span) | set(day_span)
    clock_phrase = tuple(words[k] for k in clock_span)
    # "del lunes" is said of a text "el lunes" too: the day phrase starts at its day word.
    day_words = [words[k] for k in day_span]
    while day_words and day_words[0] in _QUERY_STOPWORDS:
        day_words.pop(0)
    content = _content([w for k, w in enumerate(words) if k not in taken])
    return _ParsedQuery(content, clock, day, clock_phrase, tuple(day_words))


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


def _spans(words: list[str]) -> tuple[_SpokenClock | None, range, _SpokenDay | None, range]:
    """The first clock time and day phrase, each with the token span it covers."""
    clock, clock_span = None, range(0)
    for i in range(len(words)):
        found = _clock_at(words, i)
        if found is not None:
            clock, start, end = found
            clock_span = range(start, end)
            break
    # "de las 9 de la noche de mañana": "mañana" right after the time is the day.
    clock_end = clock_span.stop if clock_span else None
    day, day_span = None, range(0)
    for i in range(len(words)):
        if i in clock_span:
            continue
        joined = clock_end is not None and i == clock_end + 1 and words[clock_end] == "de"
        found = spoken_day_at(words, i, joined=joined)
        if found is not None and not set(clock_span) & set(range(i, found[2])):
            kind, value, end = found
            day, day_span = _SpokenDay(kind, value), range(i, end)
            break
    return clock, clock_span, day, day_span


def _time_spans(words: list[str]) -> tuple[set[int], _SpokenClock | None, _SpokenDay | None]:
    """Indexes of the first clock time and day phrase, and what they name."""
    clock, clock_span, day, day_span = _spans(words)
    return set(clock_span) | set(day_span), clock, day


def _without_time_words(text: str) -> str | None:
    """A reminder text without the clock and day words the resolver would read as times.

    "llamar a mamá a las 3" -> "llamar a mamá"; "llevar el auto al taller
    mañana" -> "llevar el auto al taller". Articles and prepositions left
    hanging at a cut are dropped. None when nothing is read as a time or
    the folded and spoken tokens do not line up.
    """
    spoken = _WORD.findall(unicodedata.normalize("NFC", text.lower()))
    folded = ["de"] + [_fold(word) for word in spoken]
    if folded[1:] != _WORD.findall(_fold(text)):
        return None
    taken, _, _ = _time_spans(folded)
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
        must be a word of the reminder's text; a spoken clock time ("de las
        6 de la tarde") and day ("de hoy", "del lunes", the T031 day
        grammar) must equal its next trigger on the household clock. A
        leading "que (te) pedí/puse/hay" clause is not content. The spoken
        clock and day phrases are also read as words of the text, as
        contiguous words ("la una", "las 2 de la tarde"), each on its own:
        a reminder is a candidate when its clock fits as due time or as
        text, and its day too, mixed or not; when the readings select
        different reminders, all are candidates and KZA asks
        (T032-U1-RR-P1, review FV-1, FV2-1).
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
        words, clock, day = query.words, query.clock, query.day
        if not words and clock is None and day is None:
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
        # Wednesday at 17 alone (review T032-U1-FV2-1).
        matches = []
        for reminder in active:
            text = _WORD.findall(_fold(reminder.text))
            if not _covers(_content(text), words):
                continue
            local = self._household_clock.to_local(reminder.trigger_at)
            if clock is not None and not (clock.matches(local.hour, local.minute)
                                          or _contains_phrase(text, query.clock_phrase)):
                continue
            if day is not None and not (day.matches(local.date(), today)
                                        or _contains_phrase(text, query.day_phrase)):
                continue
            matches.append(reminder)
        if not matches:
            return CancelResolution(CancelMatch.NOT_FOUND)
        if len(matches) == 1:
            return CancelResolution(CancelMatch.UNIQUE, matches[0])
        return CancelResolution(CancelMatch.AMBIGUOUS, candidates=tuple(matches))

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
        does not tell apart (review T032-U1-RR-F2).
        """
        bases = [reminder.text]
        content = _without_time_words(reminder.text)
        if content is not None and content != reminder.text:
            bases.append(content)
        if not with_time:
            return bases
        spoken_time = self.spoken_time_reference(reminder)
        spoken_day = self.spoken_day_reference(reminder)
        return ([f"{base} de {spoken_time}" for base in bases]
                + [f"{base} de {spoken_time} {spoken_day}" for base in bases])

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
