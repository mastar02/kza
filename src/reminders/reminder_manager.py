"""ReminderManager — CRUD, unique-match cancellation, and voice formatting."""
import logging
import math
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache

from src.reminders.reminder_store import ReminderStore, Reminder, ReminderScope
from src.core.household_clock import HouseholdClock
from src.core.sqlite_work import AuthorityGuard
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionService, request_from_service

logger = logging.getLogger(__name__)

# Seconds a numbered reminder choice waits for its answer (reminders.voice_choice_window_s).
DEFAULT_VOICE_CHOICE_WINDOW_S = 30.0
# Words of a spoken cancel complement that never identify a reminder (T032-U1).
_QUERY_STOPWORDS = frozenset({
    "de", "del", "el", "la", "los", "las", "lo", "un", "una", "unos", "unas", "que", "a", "al", "para", "por",
    "con", "en", "mi", "mis", "tu", "tus", "su", "sus", "y", "o", "ese", "esa", "este", "esta", "eso", "esto",
    "dice", "diga", "decia", "llama", "sobre", "recordatorio", "recordatorios", "todos", "todas", "nuestro",
    "nuestra", "hs", "horas", "hora",
})
_WORD = re.compile(r"\d{1,2}[:.]\d{2}|\d+|[^\W\d_]+")
# User decision t032_design_20261118: a spoken time never narrows the candidates.
# Besides every token with a digit and every single letter ("14h", "2 p.m."),
# these words of clocks, days, periods and spans are left out of a reference
# (singular forms; plurals are matched by _forms).
_TIME_WORDS = frozenset({
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce",
    "trece", "catorce", "quince", "dieciseis", "diecisiete", "dieciocho", "diecinueve", "veinte", "veintiun",
    "veintiuno", "veintiuna", "veintidos", "veintitres", "veinticuatro", "veinticinco", "veintiseis",
    "veintisiete", "veintiocho", "veintinueve", "treinta", "cuarenta", "cincuenta", "media", "medio", "cuarto",
    "menos", "punto", "pico", "algo", "pasada", "pasadita", "tipo", "eso", "hacia", "antes", "despues", "dentro",
    "luego", "temprano", "tardisimo", "primera", "primer", "manana", "mananita", "tarde", "tardecita", "noche",
    "nochecita", "madrugada", "mediodia", "medianoche", "amanecer", "atardecer", "anochecer", "desayuno",
    "almuerzo", "merienda", "cena", "hoy", "ayer", "pasado", "proximo", "proxima", "viene", "siguiente", "lunes",
    "martes", "miercoles", "jueves", "viernes", "sabado", "domingo", "finde", "fin", "semana", "mes", "ano",
    "dia", "diario", "diaria", "cada", "vez", "rato", "minuto", "segundo", "enero", "febrero", "marzo", "abril",
    "mayo", "junio", "julio", "agosto", "septiembre", "setiembre", "octubre", "noviembre", "diciembre", "am",
    "pm", "hrs",
})
# A leading relative clause names how the reminder was made, not what it says:
# "que (te) pedí", "que puse", "que hay" (review T032-U1-R6).
_RELATIVE_PRONOUNS = frozenset({"te", "le", "les", "me", "nos", "se"})
_RELATIVE_VERBS = frozenset({
    "puse", "pusiste", "puso", "pusimos", "pedi", "pediste", "pidio", "pedimos", "anote", "anotaste", "anoto",
    "anotamos", "agende", "agendaste", "agendo", "agendamos", "programe", "programaste", "guarde", "guardaste",
    "tenia", "tenias", "teniamos", "tengo", "tenemos", "hay", "habia",
})
# Weekday names a clarification speaks ("del miércoles"), Monday first.
_SPOKEN_WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
# A weekly recurrence read back ("todos los sábados"), Monday first (weekly:1 .. weekly:7).
_SPOKEN_WEEKDAYS_PLURAL = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábados", "domingos")


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


def _is_time_word(word: str) -> bool:
    """A folded token that may belong to a spoken time ("14h", "2", "tarde", "lunes", "cena")."""
    return len(word) == 1 or any(c.isdigit() for c in word) or not _forms(word).isdisjoint(_TIME_WORDS)


def _content(words: list[str]) -> tuple[frozenset[str], ...]:
    """Identifying words of a text, each as its set of plural-insensitive forms."""
    return tuple(dict.fromkeys(_forms(w) for w in words if w not in _QUERY_STOPWORDS))


def _covers(text_words: tuple[frozenset[str], ...], query_words: tuple[frozenset[str], ...]) -> bool:
    """Every query word is a word of the text."""
    return all(any(q & t for t in text_words) for q in query_words)


def _without_relative_clause(words: list[str]) -> list[str]:
    """Drop a leading "que (te) pedí/puse/hay" ("que te pedí de tomar agua")."""
    if words[:1] != ["que"]:
        return words
    j = 2 if words[1:2] and words[1] in _RELATIVE_PRONOUNS else 1
    return words[j + 1:] if words[j:j + 1] and words[j] in _RELATIVE_VERBS else words


def _base_words(reference: str) -> tuple[frozenset[str], ...]:
    """Content words of a spoken reference that may narrow: no stopword, no time word."""
    words = _without_relative_clause(_WORD.findall(_fold(reference)))
    return _content([w for w in words if not _is_time_word(w)])


@lru_cache(maxsize=4096)
def _text_words(text: str) -> tuple[frozenset[str], ...]:
    """Content words of a reminder text, its own time words included."""
    return _content(_WORD.findall(_fold(text)))


class CancelMatch(StrEnum):
    """How a spoken reminder reference resolved among the shared active ones."""

    UNIQUE = "unique"
    NONE_ACTIVE = "none_active"
    NO_REFERENCE = "no_reference"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    PARTIAL = "partial"


@dataclass(frozen=True)
class CancelResolution:
    """Result of resolving a spoken reference to exactly one reminder.

    Attributes:
        match: How the reference resolved.
        reminder: The single match, only for ``CancelMatch.UNIQUE``.
        candidates: Every eligible candidate for a clarification (all active
            ones for ``NO_REFERENCE``, every reminder sharing the base words
            for ``AMBIGUOUS``, every reminder holding the known words for
            ``PARTIAL``), in trigger order.
    """

    match: CancelMatch
    reminder: Reminder | None = None
    candidates: tuple[Reminder, ...] = field(default_factory=tuple)


class ReminderManager:
    """High-level reminder operations with unique-match search and voice output."""

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
            ValueError: Both ``clock`` and ``household_clock`` were given, or
                ``voice_choice_window_s`` is not a positive number.
        """
        if clock is not None and household_clock is not None:
            raise ValueError("Pass either clock or household_clock, not both")
        self._store = store
        self._household_clock = household_clock or HouseholdClock(time_source=clock or time.time)
        self._config = config or {}
        self.home_actions = home_actions if home_actions is not None else HomeActionService(None)
        self._tts_prefix = self._config.get("tts_prefix", "Recuerda")
        self._default_time = self._config.get("default_time", "09:00")
        window = self._config.get("voice_choice_window_s", DEFAULT_VOICE_CHOICE_WINDOW_S)
        if type(window) not in (int, float) or not math.isfinite(window) or window <= 0:
            raise ValueError("reminders.voice_choice_window_s must be a positive number of seconds")
        self._choice_window_s = float(window)

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

        User decision t032_design_20261118: the candidates are every active
        reminder in scope whose text holds the reference's base content
        words (accents, articles and plurals ignored). Time words (clocks,
        days, periods, spans, any token with a digit) never narrow: a spoken
        time is never guessed. Exactly one candidate holding every base word
        is ``UNIQUE``; more are ``AMBIGUOUS`` and KZA lists them numbered, so
        an exact text match is cancelled only when no other active reminder
        shares its base words. A base word that no active reminder's text
        holds ("llamar a mamá" with only "llamar a papá") is never ignored
        into a cancel: the reminders holding the other words are
        ``PARTIAL`` and KZA lists them numbered, even a single one, so a
        misheard word costs one confirmation and never cancels a different
        reminder (verification T032-NUM-FV-1). A reference with no base
        word at all ("cancelá el recordatorio", "el de las 5") is
        ``NO_REFERENCE``; one whose words are all unknown is ``NOT_FOUND``.
        A leading "que (te) pedí/puse/hay" clause is not content.

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
        spoken = _base_words(reference)
        if not spoken:
            return CancelResolution(CancelMatch.NO_REFERENCE, candidates=tuple(active))
        texts = [(reminder, _text_words(reminder.text)) for reminder in active]
        vocabulary = frozenset().union(*(form for _, words in texts for form in words))
        known = tuple(word for word in spoken if word & vocabulary)
        candidates = tuple(reminder for reminder, words in texts if known and _covers(words, known))
        if not candidates:
            return CancelResolution(CancelMatch.NOT_FOUND)
        if len(known) < len(spoken):
            return CancelResolution(CancelMatch.PARTIAL, candidates=candidates)
        if len(candidates) == 1:
            return CancelResolution(CancelMatch.UNIQUE, candidates[0])
        return CancelResolution(CancelMatch.AMBIGUOUS, candidates=candidates)

    @property
    def choice_window_s(self) -> float:
        """Seconds a numbered voice choice stays open (``reminders.voice_choice_window_s``)."""
        return self._choice_window_s

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

    def spoken_day_reference(self, reminder: Reminder) -> str:
        """Spoken household date of the trigger, to tell apart listed reminders with the same label.

        "de hoy", "de pasado mañana" or "del <weekday>". Never "de mañana",
        which also means "in the morning"; tomorrow is named by its weekday.
        """
        day = self._household_clock.to_local(reminder.trigger_at).date()
        offset = (day - self._household_clock.local_date()).days
        if offset == 0:
            return "de hoy"
        if offset == 2:
            return "de pasado mañana"
        return f"del {_SPOKEN_WEEKDAYS[day.weekday()]}"

    def spoken_schedule(self, reminder: Reminder) -> str:
        """When a reminder is due, as a voice read-back on the household calendar (T031-U2).

        One-off: "hoy", "mañana", "pasado mañana", "el <weekday>" within the
        week or "el <weekday> <day>" further away, then "a las HH:MM".
        Recurring: "todos los días", "de lunes a viernes" or "todos los
        <weekday>s" with the time; any other recurrence keeps the first day
        and says that it repeats.
        """
        local = self._household_clock.to_local(reminder.trigger_at)
        at = f"a las {self._household_clock.format_time(reminder.trigger_at)}"
        if reminder.recurrence == "daily":
            return f"todos los días {at}"
        if reminder.recurrence == "weekdays":
            return f"de lunes a viernes {at}"
        weekly = re.fullmatch(r"weekly:([1-7])", reminder.recurrence or "")
        if weekly is not None:
            return f"todos los {_SPOKEN_WEEKDAYS_PLURAL[int(weekly.group(1)) - 1]} {at}"
        offset = (local.date() - self._household_clock.local_date()).days
        if offset == 0:
            day = "hoy"
        elif offset == 1:
            day = "mañana"
        elif offset == 2:
            day = "pasado mañana"
        elif 2 < offset < 7:
            day = f"el {_SPOKEN_WEEKDAYS[local.weekday()]}"
        else:
            day = f"el {_SPOKEN_WEEKDAYS[local.weekday()]} {local.day}"
        repeats = " y se repite" if reminder.recurrence else ""
        return f"{day} {at}{repeats}"

    def due_hour(self, reminder: Reminder) -> int:
        """Household-clock hour (0-23) of the next trigger, to tell a listed number from a due hour (T032-U1-Q3)."""
        return self._household_clock.to_local(reminder.trigger_at).hour

    def voice_label(self, reminder: Reminder) -> str:
        """Short spoken name of a shared reminder: its text and next time."""
        if reminder.scope is not ReminderScope.SHARED:
            raise PermissionError("personal_reminder_voice_session_required")
        return f"{reminder.text} a las {self._household_clock.format_time(reminder.trigger_at)}"
