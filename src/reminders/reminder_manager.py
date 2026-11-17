"""ReminderManager — CRUD, unique-match cancellation, and voice formatting."""
import logging
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from src.reminders.reminder_store import ReminderStore, Reminder, ReminderScope
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
    if bare and words[end:end + 1] and words[end] not in _QUERY_STOPWORDS:
        return None
    return _SpokenClock(hour, minute, period), i - 1 if article else i, end


def _parse_query(query: str) -> tuple[tuple[frozenset[str], ...], _SpokenClock | None]:
    """Content words and an optional clock time of a spoken cancel complement."""
    words = _WORD.findall(_fold(query))
    for i in range(len(words)):
        found = _clock_at(words, i)
        if found is not None:
            clock, start, end = found
            return _content(words[:start] + words[end:]), clock
    return _content(words), None


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
        6 de la tarde") must equal its next trigger on the household clock.
        Among several matches, the single one whose words are exactly the
        reference's wins; otherwise the reference is ambiguous. There is no
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
        if not active:
            return CancelResolution(CancelMatch.NONE_ACTIVE)
        words, clock = _parse_query(reference)
        if not words and clock is None:
            return CancelResolution(CancelMatch.NO_REFERENCE, candidates=tuple(active))
        matches = []
        for reminder in active:
            text_words = _content(_WORD.findall(_fold(reminder.text)))
            if not _covers(text_words, words):
                continue
            if clock is not None:
                local = self._household_clock.to_local(reminder.trigger_at)
                if not clock.matches(local.hour, local.minute):
                    continue
            matches.append((reminder, text_words))
        if not matches:
            return CancelResolution(CancelMatch.NOT_FOUND)
        if len(matches) == 1:
            return CancelResolution(CancelMatch.UNIQUE, matches[0][0])
        exact = [reminder for reminder, text_words in matches if words and _covers(words, text_words)]
        if len(exact) == 1:
            return CancelResolution(CancelMatch.UNIQUE, exact[0])
        return CancelResolution(CancelMatch.AMBIGUOUS, candidates=tuple(reminder for reminder, _ in matches))

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

    def voice_label(self, reminder: Reminder) -> str:
        """Short spoken name of a shared reminder: its text and next time."""
        if reminder.scope is not ReminderScope.SHARED:
            raise PermissionError("personal_reminder_voice_session_required")
        return f"{reminder.text} a las {self._household_clock.format_time(reminder.trigger_at)}"
