"""Deterministic household intent precedence layer (P3-F2).

A pure, regex/stdlib classifier that claims a turn for a household family
(timer, reminder, list, routine, intercom, notification, deferred action or a
compound) before any domotics producer can act on it: the extended home
command, the grammar fast path, the 7B ``service_filter``, the default light
target, the dispatcher's special commands and its music/list/domotics keywords.

Blocking is unconditional; acting is gated. Once a turn is claimed no later
stage may produce an effect for it. The router decides whether the claimed
turn may proceed (wake guards and the 7B veto); the dispatcher sends it to the
family path, or to an honest refusal when the family has no handler yet.

Rules (design ``precedence_contract`` at data/autonomous-roadmap-20261102/p3/
design.json), evaluated in the order R3, R5, R1, R2, R3a, R4, R6:

* R3: a reminder or deferral verb at the head ("recordame ...").
* R5: a compound pre-split (two imperative clauses, one verb with two
  household objects, or one domotics verb with two targets).
* R1: an object operation (cancel/stop/pause/resume/disable/query) on one
  household object noun; "la alarma"/"el despertador" name the timers here
  (T030-U1), except the security verbs activá/desactivá/armá/desarmá la alarma;
  a cancel or query of an alarm with a clock time ("cancelá la alarma de las 7")
  names the reminders (user answer T030-U1-Q8), and so does a stop of one
  ("apagá/pará la alarma de las 7" is its cancel, user answer T031-U2-Q1); the
  security verbs stay unclaimed even with a clock time ("desactivá la alarma de
  las 7", user answer BH-R1-Q1).
* R2: timer creation, and list item additions/removals.
* R3a: "alarma" with a duration (timer) or a clock time (reminder).
* R4: a domotics verb with a deferral time expression.
* R6: routine, intercom and notification phrases. A request to create a
  routine ("quiero crear una nueva rutina que apague las luces", "armame una
  rutina que ...", "nueva rutina para ...") is a routine proposal with the
  reason ``new_routine``: its content never reaches a domotics producer.

No I/O and no logging of user text: the module never logs.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from functools import lru_cache
from dataclasses import dataclass, field
from enum import Enum

from src.nlu.command_grammar import (
    ENTITY_TERMS,
    ROOM_ALIASES,
    WAKE_TERMS,
    has_deferral_time,
    non_immediate_action_reason,
    parse_command,
    security_order,
)
from src.nlu.home_command import parse_home_command


class HouseholdFamily(str, Enum):
    """Household object family that claimed a turn."""

    TIMER = "timer"
    REMINDER = "reminder"
    LIST = "list"
    ROUTINE = "routine"
    INTERCOM = "intercom"
    NOTIFICATION = "notification"
    DEFERRED_ACTION = "deferred_action"
    COMPOUND = "compound"


class HouseholdOperation(str, Enum):
    """Operation requested on the claimed family."""

    CREATE = "create"
    CANCEL = "cancel"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    DISABLE = "disable"
    QUERY = "query"
    ADD_ITEM = "add_item"
    REMOVE_ITEM = "remove_item"
    EXECUTE = "execute"
    PROPOSE = "propose"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class HouseholdPart:
    """One spoken part of a compound turn, in spoken order.

    Attributes:
        segment: The part's words as heard (lowercase, punctuation removed).
        intent: The part's own classification, or None for an unclaimed
            segment that belongs to the domotics/music path.
    """

    segment: str
    intent: "HouseholdIntent | None" = None


@dataclass(frozen=True)
class HouseholdIntent:
    """A claimed household turn.

    Attributes:
        family: Claimed family.
        operation: Requested operation.
        rule: Rule id (R1..R6, R3a) for audit and the C02 matrix tags.
        complement: Words after the governed object or reminder verb; family
            handlers parse it, the classifier does not.
        verb: Accent-folded head verb as heard (e.g. ``borra``), or "".
        reason: Code for clarifications and refusals (``other_person``,
            ``second_clause``, ``too_many_parts``, ``conflicting_parts``,
            ``deferred_part``, ``unparsed_part``, ``duration``, ``clock_time``,
            ``no_time``, ``new_routine``, ``clock_alarm``,
            ``ring_stop``), or None.
        parts: COMPOUND only: the parts in spoken order.
        literal_span: Intercom message offsets in the original transcript.
    """

    family: HouseholdFamily
    operation: HouseholdOperation
    rule: str
    complement: str = ""
    verb: str = ""
    reason: str | None = None
    parts: tuple[HouseholdPart, ...] = ()
    literal_span: tuple[int, int] | None = field(default=None, repr=False, compare=False)

    @property
    def controls_routine(self) -> bool:
        """Whether the utterance asks for a saved-routine effect."""
        return self.reason == "routine_control_part" or (self.family is HouseholdFamily.ROUTINE and self.operation in {
            HouseholdOperation.EXECUTE, HouseholdOperation.STOP, HouseholdOperation.DISABLE,
        })

    def summary(self) -> dict:
        """Content-free description for logs and turn results (no user text)."""
        return {"family": self.family.value, "operation": self.operation.value, "rule": self.rule,
                "reason": self.reason, "parts": len(self.parts)}


@dataclass(frozen=True)
class HouseholdIntentSettings:
    """``household_intents`` settings (config/settings.yaml).

    Attributes:
        unvetoed_max_words: Longest claimed turn that skips the 7B veto on a
            strong acoustic wake (same limit as the grammar's wake bonus).
        compound_max_parts: More parts make a compound not admissible.
        compound_execute: Compound execution (T037-U6): each part of an
            admissible compound runs in spoken order. False is the kill
            switch: compounds then answer "Decímelo de a una cosa", with no
            effect.
        wake_words: Configured wake words stripped from the head of a turn,
            in addition to the grammar's wake terms.
    """

    unvetoed_max_words: int = 6
    compound_max_parts: int = 3
    compound_execute: bool = True
    wake_words: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.unvetoed_max_words) is not int or self.unvetoed_max_words < 1:
            raise ValueError("household_intents.unvetoed_max_words must be a positive integer")
        if type(self.compound_max_parts) is not int or self.compound_max_parts < 2:
            raise ValueError("household_intents.compound_max_parts must be an integer >= 2")
        if type(self.compound_execute) is not bool:
            raise ValueError("household_intents.compound_execute must be a boolean")

    @classmethod
    def from_config(cls, section: Mapping | None, wake_words: Iterable[str] = ()) -> "HouseholdIntentSettings":
        """Build from the ``household_intents`` section and the configured wake words.

        Args:
            section: The ``household_intents`` mapping, or None for defaults.
            wake_words: ``rooms.wake_word.words`` (or ``wake_word.words``).

        Raises:
            ValueError: A value is out of range or of the wrong type.
        """
        section = section or {}
        if not isinstance(section, Mapping):
            raise ValueError("household_intents must be a mapping")
        words = tuple(_fold(w).strip() for w in wake_words if isinstance(w, str) and w.strip())
        return cls(
            unvetoed_max_words=section.get("unvetoed_max_words", 6),
            compound_max_parts=section.get("compound_max_parts", 3),
            compound_execute=section.get("compound_execute", True),
            wake_words=words,
        )


# Recent utterances a classifier remembers: one turn is classified by the
# command gate, the router and the dispatcher in a row (review F7).
_MEMO_SIZE = 8


class HouseholdIntentClassifier:
    """The one classifier instance shared by the router and the dispatcher.

    Classification is pure, so the instance memoizes the last few utterances:
    the command gate, the router and the dispatcher classify the same turn
    and only the first call pays for it. Results are frozen and nothing is
    logged or persisted.

    Args:
        settings: Household intent settings; defaults when None.
    """

    def __init__(self, settings: HouseholdIntentSettings | None = None) -> None:
        self._settings = settings if settings is not None else HouseholdIntentSettings()
        self._memo = lru_cache(maxsize=_MEMO_SIZE)(self._classify)

    @property
    def settings(self) -> HouseholdIntentSettings:
        """Settings this classifier was built with."""
        return self._settings

    def classify(self, text: str) -> HouseholdIntent | None:
        """Classify one utterance (see :func:`classify_household_intent`)."""
        return self._memo(text if isinstance(text, str) else "")

    def _classify(self, text: str) -> HouseholdIntent | None:
        return classify_household_intent(text, wake_words=self._settings.wake_words,
                                         max_compound_parts=self._settings.compound_max_parts)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

# A one-digit decimal ("1,5 horas") stays one token, as in src.timers.voice_request.
_WORD_RE = re.compile(r"\d+(?:[:.]\d{2}|[.,]\d(?!\d))?|[^\W\d_]+")
_LEAD_WORDS = frozenset({"che", "ey", "oye", "hola", "porfa", "podes", "podrias", "puedes", "podras", "bueno",
                         "bue", "dale"})
# Fillers dropped at the end of a turn ("..., por favor", "... porfa"); "por favor" is matched as a pair
# (review BFR-1). A trailing "dale" may be part of a name ("decirle que dale", "llamar a Dale"), so only
# a caller with a closed vocabulary drops it, through ``extra_trails`` (review BFR1-R1).
_TRAIL_WORDS = frozenset({"porfa"})
# A leading "no" is stripped only when punctuation follows it ("no, cancelá ..."):
# "no cancelar el recordatorio" is a negation (review T032-U1-R3).
_PAUSE_AFTER = re.compile(r"\s*[,;:.!]")
# The pause before a paused trailing word (``paused_trails``: "..., dale"). A colon is left out: it usually
# opens quoted speech, where the word belongs to the text ("decirle a Juan: dale", review BH-R2).
_PAUSE_BEFORE_TRAIL = re.compile(r"\s*[,;.!]")
# Lead words (wake words, fillers, "por favor", a paused "no") stripped at most.
_MAX_LEADS = 6


def _fold(text: str) -> str:
    """Lowercase and strip accents ("ñ" folds to "n")."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


@dataclass(frozen=True)
class _Utterance:
    """Tokens of one utterance: lowercase as heard, and accent-folded."""

    original: tuple[str, ...]
    folded: tuple[str, ...]
    source: str = ""
    word_ends: tuple[int, ...] = ()

    @property
    def text(self) -> str:
        return " ".join(self.folded)

    @property
    def heard(self) -> str:
        return " ".join(self.original)

    def __len__(self) -> int:
        return len(self.folded)

    def at(self, i: int) -> str:
        return self.folded[i] if 0 <= i < len(self.folded) else ""

    def slice(self, start: int, end: int | None = None) -> "_Utterance":
        return _Utterance(self.original[start:end], self.folded[start:end], self.source,
                          self.word_ends[start:end])

    def join(self, other: "_Utterance") -> "_Utterance":
        return _Utterance(self.original + other.original, self.folded + other.folded)


def _tokens(text: str) -> _Utterance:
    source = text or ""
    words = list(_WORD_RE.finditer(source))
    original = tuple(word.group().lower() for word in words)
    return _Utterance(original, tuple(_fold(t) for t in original), source, tuple(word.end() for word in words))


def _filler_bounds(utt: _Utterance, text: str, wake_words: Iterable[str], extra_leads: frozenset[str],
                   extra_trails: frozenset[str], paused_trails: frozenset[str] = frozenset()) -> tuple[int, int]:
    """Token range of ``utt`` (the tokens of ``text``) without its lead and trail fillers.

    A word of ``paused_trails`` is a trailing filler only after a pause
    (",", ";", "." or "!" before it: "..., dale"); a colon is not one
    ("decirle a Juan: dale" keeps it).
    """
    lowered = (text or "").lower()
    spans = [m.span() for m in _WORD_RE.finditer(lowered)]
    leads = WAKE_TERMS | {_fold(w) for w in wake_words} | _LEAD_WORDS | extra_leads
    trails = _TRAIL_WORDS | extra_trails
    start, end = 0, len(utt)
    for _ in range(_MAX_LEADS):
        word = utt.at(start)
        if word in leads:
            start += 1
        elif word == "por" and utt.at(start + 1) == "favor":
            start += 2
        elif word == "no" and _PAUSE_AFTER.match(lowered, spans[start][1]):
            start += 1
        else:
            break
    while end > start:
        if end - start >= 2 and utt.folded[end - 2:end] == ("por", "favor"):
            end -= 2
        elif utt.folded[end - 1] in trails:
            end -= 1
        elif utt.folded[end - 1] in paused_trails and end - start >= 2 \
                and _PAUSE_BEFORE_TRAIL.match(lowered, spans[end - 2][1]):
            end -= 1
        else:
            break
    return start, end


def filler_span(text: str, wake_words: Iterable[str] = (), *, paused_trails: frozenset[str] = frozenset()
                ) -> tuple[int, int]:
    """Character range of ``text`` left by the filler rule of :func:`strip_fillers`.

    For a caller that needs the words as heard, with their punctuation (the
    voice reminder parser, unit BFR1-Q1): ``text[start:end]`` holds every
    token :func:`strip_fillers` keeps and nothing before or after them.

    Args:
        text: The utterance as heard.
        wake_words: Configured wake words, besides ``WAKE_TERMS``.
        paused_trails: Trailing words (accent-folded) dropped only after a
            pause ("tomar agua a las 5, dale"); without the pause, or after
            a colon, they stay, as they may end a free text ("decirle que
            dale", "decirle a Juan: dale").

    Returns:
        ``(start, end)`` offsets into ``text``; ``(0, len(text))`` when
        lowercasing changes the length of ``text`` (offsets would not match),
        and ``(0, 0)`` when only fillers were said.
    """
    text = text or ""
    lowered = text.lower()
    if len(lowered) != len(text):
        return 0, len(text)
    utt = _tokens(text)
    start, end = _filler_bounds(utt, text, wake_words, frozenset(), frozenset(), paused_trails)
    if start >= end:
        return 0, 0
    spans = [m.span() for m in _WORD_RE.finditer(lowered)]
    return spans[start][0], spans[end - 1][1]


def strip_fillers(text: str, wake_words: Iterable[str] = (), *, extra_leads: frozenset[str] = frozenset(),
                  extra_trails: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """Accent-folded tokens of ``text`` without the fillers the household classifier ignores.

    The one filler rule for every short voice answer (review BFR-1): up to
    six leading wake words, fillers ("bueno", "bue", "dale", "hola",
    "porfa", "che", ...), "por favor" and a "no" followed by a pause ("no,
    el segundo"; "no el segundo" keeps it), and any trailing "por favor"
    or "porfa". A trailing "dale" is kept: it may be part of a name
    ("decirle que dale", review BFR1-R1).

    Args:
        text: The utterance as heard.
        wake_words: Configured wake words, besides ``WAKE_TERMS``.
        extra_leads: Further lead words the caller ignores (accent-folded).
        extra_trails: Further trailing words the caller ignores
            (accent-folded); only for a closed vocabulary such as a numbered
            choice answer ("el segundo, dale").

    Returns:
        The remaining tokens, lowercase and accent-folded.
    """
    utt = _tokens(text)
    start, end = _filler_bounds(utt, text, wake_words, extra_leads, extra_trails)
    return utt.folded[start:end]


# T030-U2b (user answer morning_questions_20261119 "T030-U2 bare 'listo'"): the one bare word that
# stops a ringing timer. It is not a classifier rule: the command gate drops it as noise unless a ring
# is sounding or in its stop follow-up, so only the router and the dispatcher claim it, after the gate.
_RING_STOP_WORDS = ("listo",)

RING_STOP_INTENT = HouseholdIntent(HouseholdFamily.TIMER, HouseholdOperation.STOP, rule="R1", verb="listo",
                                   reason="ring_stop")
"""The claim of a bare ring stop ("listo"): stop only a ring, never cancel a timer (T030-U2b)."""


def is_bare_ring_stop(text: str, wake_words: Iterable[str] = ()) -> bool:
    """Whether the turn is a bare "listo", with only the shared fillers of :func:`strip_fillers` (BFR-1).

    "Listo.", "Nexa, listo", "bueno, listo" and "listo, por favor" are; "listo, apagá la luz", "ya
    está listo" and "listo listo" are not. Pure: whether a ring is open is the caller's question.

    Args:
        text: The utterance as heard.
        wake_words: Configured wake words, besides ``WAKE_TERMS``.

    Returns:
        True for a bare ring stop word.
    """
    return isinstance(text, str) and strip_fillers(text, wake_words) == _RING_STOP_WORDS


def _normalize(text: str, wake_words: Iterable[str] = ()) -> _Utterance:
    """Tokenize and strip the fillers of :func:`strip_fillers`.

    Also strips a "ya" right after a cancel verb ("cancelá ya el
    recordatorio ...", "olvidate ya del ..."), so these imperatives are
    claimed.
    """
    utt = _tokens(text)
    start, end = _filler_bounds(utt, text, wake_words, frozenset(), frozenset())
    if start + 1 < end and utt.at(start + 1) == "ya":
        # Checked without the "ya": "olvidate" is a cancel head only before
        # "de/del" (review T032-U1-RR-F4).
        without = utt.slice(start, start + 1).join(utt.slice(start + 2, end))
        head = _verb_at(without, 0)
        if head is not None and _CANCEL in head.classes:
            return without
    return utt.slice(start, end)


# --------------------------------------------------------------------------
# Lexicon (accent-folded)
# --------------------------------------------------------------------------

_ENCLITICS = ("noslo", "nosla", "melo", "mela", "selo", "sela", "me", "te", "le", "lo", "la", "nos", "les",
              "los", "las")

_CANCEL = "cancel"
_CLEAR = "clear"
_STOP = "stop"
_STOP_RINGING = "stop_ringing"
_PAUSE = "pause"
_RESUME = "resume"
_DISABLE = "disable"
_CREATE = "create"
_TIMER_CREATE = "timer_create"
_ADD = "add"
_REMOVE = "remove"
_ROUTINE_EXEC = "routine_exec"
_ROUTINE_PROPOSE = "routine_propose"
_INTERCOM = "intercom"
_DOMOTICS = "domotics"
_AUTOMATE = "automate"

_VERB_FORMS: dict[str, frozenset[str]] = {}


def _forms(cls: str, *words: str) -> None:
    for word in words:
        _VERB_FORMS[word] = _VERB_FORMS.get(word, frozenset()) | {cls}


_forms(_CANCEL, "cancela", "cancele", "cancelen", "cancelar", "borra", "borre", "borren", "borrar", "elimina",
       "elimine", "eliminar", "saca", "saque", "sacar", "quita", "quite", "quitar", "anula", "anule", "anular")
_forms(_CLEAR, "vacia", "vacie", "vaciar", "limpia", "limpie", "limpiar")
_forms(_STOP, "pare", "parar", "detene", "deten", "detenga", "detener", "frena", "frene", "frenar")
_forms(_STOP_RINGING, "apaga", "apague", "apagar", "calla", "calle", "callar", "silencia", "silencie", "silenciar")
_forms(_PAUSE, "pausa", "pause", "pausar")
_forms(_RESUME, "segui", "sigue", "seguir", "continua", "continue", "continuar", "reanuda", "reanude", "reanudar")
_forms(_DISABLE, "desactiva", "desactive", "desactivar")
_forms(_CREATE, "pone", "pon", "ponga", "poner", "crea", "crear", "programa", "programe", "programar", "arranca",
       "arranque", "arrancar", "inicia", "inicie", "iniciar", "activa", "active", "activar", "configura",
       "configure", "configurar", "agenda", "agendar")
# Rioplatense creation verbs that create only a timer ("haceme/metele/tirame un timer").
_forms(_TIMER_CREATE, "hace", "haga", "mete", "meta", "tira", "tire")
_forms(_ADD, "agrega", "agregue", "agregar", "anadi", "anade", "anada", "anadir", "suma", "sume", "sumar", "anota",
       "anote", "anotar")
_forms(_REMOVE, "quita", "quite", "quitar", "saca", "saque", "sacar", "borra", "borre", "borrar", "elimina",
       "elimine", "eliminar", "tacha", "tache", "tachar")
_forms(_ROUTINE_EXEC, "ejecuta", "ejecute", "ejecutar", "corre", "corra", "correr", "lanza", "lance", "lanzar",
       "arranca", "arranque", "arrancar", "inicia", "inicie", "iniciar", "pone", "pon", "ponga", "poner")
_forms(_ROUTINE_PROPOSE, "crea", "crear", "arma", "arme", "armar", "hace", "haga", "hacer", "programa", "programe",
       "programar", "activa", "active", "activar", "configura", "configure", "configurar", "agenda", "agendar")
_forms(_AUTOMATE, "automatiza", "automatice", "automatizar")
_forms(_INTERCOM, "anuncia", "anuncie", "anunciar", "avisa", "avise", "comunica", "comunique", "deci", "di", "diga")
_forms(_DOMOTICS, "prende", "prenda", "prendan", "prender", "encende", "enciende", "encienda", "encender", "apaga",
       "apague", "apaguen", "apagar", "sube", "subi", "suba", "subir", "baja", "baje", "bajen", "bajar", "abri",
       "abre", "abra", "abrir", "cierra", "cerra", "cierre", "cerrar", "pone", "pon", "ponga", "poner", "activa",
       "active", "activar", "desactiva", "desactive", "desactivar", "cambia", "cambie", "cambiar", "pausa",
       "pause", "pausar", "segui", "sigue", "reproduci", "reproduce", "reproducir", "ilumina", "corta", "corte")

_INFINITIVE_ENDINGS = ("ar", "er", "ir")

_OBJECT_NOUNS: dict[str, HouseholdFamily] = {
    "timer": HouseholdFamily.TIMER, "timers": HouseholdFamily.TIMER,
    "temporizador": HouseholdFamily.TIMER, "temporizadores": HouseholdFamily.TIMER,
    "recordatorio": HouseholdFamily.REMINDER, "recordatorios": HouseholdFamily.REMINDER,
    "rutina": HouseholdFamily.ROUTINE, "rutinas": HouseholdFamily.ROUTINE,
    "lista": HouseholdFamily.LIST, "listas": HouseholdFamily.LIST,
}
# "alarma"/"despertador" as the object of a timer command (T030-U1 R3): an alarm with a duration is a
# timer (R3a, user answer Q4), so cancel/stop/pause/resume/query of "la alarma" reach the timers. Claimed in
# R1 only; creation stays in R3a. The security alarm verbs (activá/desactivá/armá/desarmá la alarma) are never
# claimed (design R3a): they stay on the existing path, where alarm panels are not controllable, with or without a
# clock time ("desactivá la alarma de las 7", user answer BH-R1-Q1, 2026-11-18).
_ALARM_NOUNS = frozenset({"alarma", "alarmas", "despertador", "despertadores"})
_SECURITY_ALARM_NOUNS = frozenset({"alarma", "alarmas"})
# User answer T030-U1-Q8 (A, 2026-11-17): an alarm named with a clock time ("la alarma de las 7") was created as
# a reminder (R3a, T031-U2), so its cancel and its query reach the reminders; without a clock time an alarm stays
# a timer. "de las 7 minutos" is a duration, not a clock time; "de las 10 horas" is a clock time (review
# T031-U2-RV2: an article before the hour names the hour of the day).
_ALARM_HOUR = (r"(?:\d{1,2}(?:[:.]\d{2})?|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece"
               r"|catorce|quince|dieciseis|diecisiete|dieciocho|diecinueve|veinte|veintiuna|veintiuno|veintidos"
               r"|veintitres)")
# A day alone ("la alarma de mañana", "de hoy", "del martes") also names an alarm set for a clock time: a timer
# has no day (review T031-U2-RV3), so it reaches the reminders' numbered alarm list too.
_ALARM_CLOCK = re.compile(
    rf"\b(?:de|a|para) (?:la|las) {_ALARM_HOUR}\b(?! (?:segundos?|minutos?|mins?)\b)"
    r"|\b(?:del|al|de la|a la|para el|para la) (?:mediodia|medianoche)\b"
    r"|\b(?:de|para) (?:hoy|pasado manana|manana)\b"
    r"|\b(?:del|de este|para el) (?:proximo )?(?:lunes|martes|miercoles|jueves|viernes|sabado|domingo)\b")
_SECURITY_ALARM_VERBS = frozenset({"activa", "active", "activar", "desactiva", "desactive", "desactivar", "arma",
                                   "arme", "armar", "desarma", "desarme", "desarmar"})
_DETERMINERS = frozenset({
    "el", "la", "los", "las", "un", "una", "mi", "mis", "ese", "esa", "esos", "esas", "este", "esta", "estos",
    "estas", "del", "al", "todos", "todas", "nuestro", "nuestra", "nuestros", "nuestras", "otro", "otra",
})
_ADJECTIVES = frozenset({"nuevo", "nueva", "nuevos", "nuevas"})
_CONJUNCTIONS = frozenset({"y", "tambien", "despues", "luego"})
_CONNECTORS = frozenset({"despues", "luego", "tambien", "ademas"})
_CONTENT_MARKERS = frozenset({"que", "cuando", "si", "porque"})

# Delete and clear verbs that reach the existing whole-list handler (user answer Q30).
LIST_DELETE_VERBS = frozenset({"borra", "borre", "borrar", "elimina", "elimine", "eliminar"})
LIST_CLEAR_VERBS = frozenset({"vacia", "vacie", "vaciar", "limpia", "limpie", "limpiar"})

_REMINDER_HEAD = re.compile(
    r"^(?:no (?:me|nos) dejes olvidar"
    r"|hace(?:me|nos) acordar"
    r"|(?:me|nos) (?:podes|podrias|puedes) (?:recordar|avisar)"
    r"|(?:me|nos) (?:recordas|avisas)"
    r"|recorda(?:me|nos|melo)|recuerda(?:me|nos|melo)|acorda(?:me|nos)|acuerda(?:me|nos)|avisa(?:me|nos)"
    r"|acordate de(?! que\b)|acuerdate de(?! que\b)"
    r"|recordar(?:me|nos)|avisar(?:me|nos)"
    r"|(?P<other>recorda(?:le|les)|recuerda(?:le|les)|avisa(?:le|les)|recordar(?:le|les)|avisar(?:le|les)"
    r"|(?:le|les) (?:podes|podrias|puedes) (?:recordar|avisar)))\b"
)

_NUMBER = (r"(?:\d+|un|una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince"
           r"|dieci(?:seis|siete|ocho|nueve)|veinte|veinti(?:un|uno|una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve)"
           r"|treinta|cuarenta|cincuenta|sesenta|setenta|ochenta|noventa|media|medio|cuartos?)")
_DURATION = re.compile(rf"\b{_NUMBER}(?: y \w+)? (?:de )?(?:segundos?|minutos?|mins?|horas?)\b|\bmedia hora\b"
                       r"|\bhora y (?:media|cuarto)\b|\bminuto y medio\b")
_CLOCK = re.compile(r"\b(?:a|para|cuando sean?) (?:las? )?(?:\d{1,2}(?:[:.]\d{2})?|una|dos|tres|cuatro|cinco"
                    r"|seis|siete|ocho|nueve|diez|once|doce|mediodia|medianoche)\b|\bal mediodia\b")
# Review T031-U2-RV2: "a las 19 horas", "para las 8 hs" and "de las 7 horas de la mañana" name an hour of the
# day (the article before the number), not a duration; R3a reads them before its duration check. "de 19 horas"
# (no article) stays a duration. Likewise "a las 20 y 5 minutos" is 20:05, not a 5-minute timer (review
# T031-U2-RV5: the form the minutes question suggests).
_ARTICLE_CLOCK_UNIT = re.compile(rf"\b(?:a|para|de) (?:la|las) {_ALARM_HOUR} "
                                 rf"(?:h|hs|hrs|horas?|y {_NUMBER} (?:minutos?|mins?))\b")
_INTERCOM_DESTINATION_HEADS = frozenset({"en", "a", "al", "por", "para"})
INTERCOM_BROADCAST_TARGETS = frozenset({
    "toda la casa", "todas las salas", "todas partes", "todos lados", "todos", "parlantes",
    "todos los parlantes",
})
_INTERCOM_PUNCTUATION = re.compile(r"[^\w]*")
_INTERCOM_DIRECTION = re.compile(
    r"(?:" + "|".join(sorted(_INTERCOM_DESTINATION_HEADS)) + r")\b\s*", re.IGNORECASE,
)
_INTERCOM_MESSAGE_MARKER = re.compile(r"\bque\b|:", re.IGNORECASE)
_NOTIFICATION = re.compile(
    r"(?:(?:(?P<disable>desactiva|desactive|quita|quite|saca|saque)|activa|active|pone|pon) )?"
    r"(?:el )?(?:modo )?no molestar"
)
_MUSIC_WORDS = frozenset({"musica", "cancion", "canciones", "tema", "temas", "playlist", "spotify", "reproduccion",
                          "album", "disco"})
_QUERY_HEADS = frozenset({"que", "cuales", "cuantos", "cuantas"})
# "¿tengo recordatorios?", "¿hay algún timer?": the object follows the head.
_HAVE_HEADS = frozenset({"hay", "tengo", "tenemos"})
_SOME = frozenset({"algun", "alguna", "algunos", "algunas"})
_QUERY_VERBS = frozenset({"hay", "tengo", "tenemos", "quedan", "queda", "son", "tiene", "estan", "activos",
                          "activas", "pendientes", "corriendo", "anda", "andan"})
_READ_HEADS = frozenset({"decime", "dime", "leeme", "lee", "mostrame", "listame", "nombrame", "contame"})
_ROOM_PREPS = (("del",), ("de", "la"), ("de", "el"), ("en", "el"), ("en", "la"), ("al",), ("a", "la"))
_ENTITY_WORDS = frozenset(_fold(term) for terms in ENTITY_TERMS.values() for term in terms if " " not in term)
_OPPOSITES = ({"turn_on", "turn_off"}, {"open", "close"}, {"media_play", "media_pause"})


@dataclass(frozen=True)
class _Verb:
    form: str
    classes: frozenset[str]
    end: int
    infinitive: bool = False


def _verb_at(utt: _Utterance, i: int, *, strict: bool = False) -> _Verb | None:
    """Head verb at token ``i``: its classes and the index after it.

    ``strict`` accepts only imperative forms and the accented "pará" (the
    unaccented "para" is a preposition); it is used for clause heads inside
    a turn. Infinitives are accepted only at the head of the turn.
    """
    word = utt.at(i)
    if not word:
        return None
    if word == "olvidate" and utt.at(i + 1) in ("de", "del"):
        return _Verb("olvidate", frozenset({_CANCEL}), i + 2 if utt.at(i + 1) == "de" else i + 1)
    if word == "para":
        # Unaccented "para" is usually the preposition: it is a verb only when
        # spoken "pará", or at the head of a turn with no other imperative.
        if utt.original[i] != "pará" and (strict or any(
                _verb_at(utt, k, strict=True) is not None for k in range(i + 1, len(utt)))):
            return None
        return _Verb("para", frozenset({_STOP}), i + 1)
    candidates = [word] + [word[:-len(suffix)] for suffix in _ENCLITICS
                           if word.endswith(suffix) and len(word) - len(suffix) >= 2]
    for form in candidates:
        classes = _VERB_FORMS.get(form)
        if classes is None:
            continue
        infinitive = form.endswith(_INFINITIVE_ENDINGS)
        if strict and infinitive:
            return None
        return _Verb(form, classes, i + 1, infinitive)
    return None


def _is_playlist(utt: _Utterance, k: int) -> bool:
    """'lista(s) de reproducción/canciones/música' at token ``k`` is music, not a household list."""
    return utt.at(k) in ("lista", "listas") and utt.at(k + 1) == "de" and utt.at(k + 2) in _MUSIC_WORDS


def _object_at(utt: _Utterance, k: int) -> HouseholdFamily | None:
    """The household object family named at token ``k`` (a playlist is not one)."""
    return None if _is_playlist(utt, k) else _OBJECT_NOUNS.get(utt.at(k))


def _mentions_list(utt: _Utterance) -> bool:
    """A household list (not a playlist) is named anywhere in the utterance."""
    return any(_object_at(utt, k) is HouseholdFamily.LIST for k in range(len(utt)))


def _r1_object_at(utt: _Utterance, k: int) -> HouseholdFamily | None:
    """Like :func:`_object_at`, but an alarm noun names the timers (R1 commands and queries only)."""
    return HouseholdFamily.TIMER if utt.at(k) in _ALARM_NOUNS else _object_at(utt, k)


def _clock_alarm(utt: _Utterance, k: int) -> bool:
    """The alarm noun at token ``k`` is named with a clock time or a day after it (user answer T030-U1-Q8)."""
    return utt.at(k) in _ALARM_NOUNS and _ALARM_CLOCK.search(" ".join(utt.folded[k + 1:])) is not None


# User answer T031-U2-Q1 (2026-11-18): "apagá/pará la alarma de las 7" turns off an alarm set for a clock time,
# which is a reminder: it is that reminder's cancel (numbered list first, T032-U1). The other stop verbs
# (detené/frená/callá/silenciá) follow it. A disable reaches this only through "despertador": "desactivá la alarma
# <hora>" is a security alarm verb, never claimed (user answer BH-R1-Q1, design R3a). Pause and resume stay with
# the timers; a stop without a clock time keeps stopping the timers (T030-U1).
_CLOCK_ALARM_CANCELS = frozenset({HouseholdOperation.CANCEL, HouseholdOperation.STOP, HouseholdOperation.DISABLE})


def _r1_intent(utt: _Utterance, k: int, family: HouseholdFamily, operation: HouseholdOperation,
               verb: str) -> HouseholdIntent:
    """An R1 intent on the object at token ``k``.

    A cancel, stop or disable of an alarm with a clock time is the
    reminders' CANCEL, and its query their QUERY, with the reason
    ``clock_alarm`` (user answers T030-U1-Q8 and T031-U2-Q1); any other
    alarm operation stays with ``family``. The disable arrives only for
    "despertador": :func:`_r1_object` never claims "desactivá la alarma"
    (user answer BH-R1-Q1).
    """
    if _clock_alarm(utt, k):
        if operation is HouseholdOperation.QUERY:
            return HouseholdIntent(HouseholdFamily.REMINDER, operation, "R1", _rest(utt, k + 1), verb,
                                   reason="clock_alarm")
        if operation in _CLOCK_ALARM_CANCELS:
            return HouseholdIntent(HouseholdFamily.REMINDER, HouseholdOperation.CANCEL, "R1", _rest(utt, k + 1),
                                   verb, reason="clock_alarm")
    complement = _rest(utt, k + 1)
    if family is HouseholdFamily.ROUTINE and operation in {HouseholdOperation.STOP, HouseholdOperation.DISABLE}:
        complement = _routine_name_after(utt, k)
    return HouseholdIntent(family, operation, "R1", complement, verb)


def _governed_object(utt: _Utterance, j: int, *, allow_adjective: bool = False,
                     alarms: bool = False) -> tuple[HouseholdFamily, int] | None:
    """The object noun right after the verb and its determiners, if any (``alarms``: alarm nouns too)."""
    k = j
    if utt.at(k) == "me":
        k += 1
    for _ in range(2):
        if utt.at(k) in _DETERMINERS:
            k += 1
    if allow_adjective and utt.at(k) in _ADJECTIVES:
        k += 1
    family = _r1_object_at(utt, k) if alarms else _object_at(utt, k)
    return (family, k) if family is not None else None


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

def _rest(utt: _Utterance, start: int, *, skip: Iterable[str] = ()) -> str:
    skip = frozenset(skip)
    while start < len(utt) and utt.at(start) in skip:
        start += 1
    return " ".join(utt.original[start:])


def _routine_name_after(utt: _Utterance, noun: int) -> str:
    """Preserve the exact name tail, including punctuation and internal spacing."""
    return utt.source[utt.word_ends[noun]:].strip() if utt.word_ends else _rest(utt, noun + 1)


def _clause_head(utt: _Utterance, j: int) -> bool:
    """An imperative verb, or a reminder verb, opens a clause at token ``j``."""
    return _verb_at(utt, j, strict=True) is not None or _REMINDER_HEAD.match(" ".join(utt.folded[j:])) is not None


def _has_second_clause(utt: _Utterance, start: int) -> bool:
    for k in range(start, len(utt)):
        if utt.at(k) not in _CONJUNCTIONS:
            continue
        j = k + 1
        while utt.at(j) in _CONNECTORS:
            j += 1
        if _clause_head(utt, j):
            return True
    return False


def _r3(utt: _Utterance) -> HouseholdIntent | None:
    match = _REMINDER_HEAD.match(utt.text)
    if match is None:
        return None
    used = len(match.group(0).split())
    verb = utt.at(0)
    if match.group("other"):
        return HouseholdIntent(HouseholdFamily.REMINDER, HouseholdOperation.CREATE, "R3",
                               _rest(utt, used), verb, reason="other_person")
    if _has_second_clause(utt, used):
        return HouseholdIntent(HouseholdFamily.REMINDER, HouseholdOperation.CLARIFY, "R3",
                               _rest(utt, used, skip=("que", "de")), verb, reason="second_clause")
    return HouseholdIntent(HouseholdFamily.REMINDER, HouseholdOperation.CREATE, "R3",
                           _rest(utt, used, skip=("que", "de")), verb)


_R1_OPERATIONS = (
    (_CANCEL, HouseholdOperation.CANCEL), (_CLEAR, HouseholdOperation.CANCEL), (_STOP, HouseholdOperation.STOP),
    (_STOP_RINGING, HouseholdOperation.STOP), (_PAUSE, HouseholdOperation.PAUSE),
    (_RESUME, HouseholdOperation.RESUME), (_DISABLE, HouseholdOperation.DISABLE),
)


def _r1_object(utt: _Utterance, verb: _Verb) -> tuple[HouseholdFamily, HouseholdOperation, int] | None:
    governed = _governed_object(utt, verb.end, alarms=True)
    if governed is None:
        return None
    family, k = governed
    if utt.at(k) in _SECURITY_ALARM_NOUNS and verb.form in _SECURITY_ALARM_VERBS:
        # Never claimed, with or without a clock time (design R3a; user answer BH-R1-Q1).
        return None
    for cls, operation in _R1_OPERATIONS:
        if cls not in verb.classes:
            continue
        if cls == _CLEAR and family is not HouseholdFamily.LIST:
            continue
        return family, operation, k
    return None


def _query(utt: _Utterance) -> HouseholdIntent | None:
    head = utt.at(0)
    nouns = [(k, family) for k in range(len(utt)) if (family := _r1_object_at(utt, k)) is not None]
    if not nouns:
        return None
    k, family = nouns[0]
    if head == "cuanto" and family is HouseholdFamily.TIMER and {"falta", "queda"} & set(utt.folded[:5]):
        return _r1_intent(utt, k, family, HouseholdOperation.QUERY, head)
    if head in _QUERY_HEADS and _QUERY_VERBS & set(utt.folded):
        return _r1_intent(utt, k, family, HouseholdOperation.QUERY, head)
    if head == "a" and utt.at(1) == "que" and utt.at(2) == "hora" and utt.at(k) in _ALARM_NOUNS:
        # "¿a qué hora suena la alarma?": answered from the timers and the alarms set for a clock time,
        # never by the cloud model, which cannot see either (review T031-U2-RV3).
        return _r1_intent(utt, k, family, HouseholdOperation.QUERY, head)
    if head in _HAVE_HEADS:
        governed = _governed_object(utt, 2 if utt.at(1) in _SOME else 1, alarms=True)
        if governed is not None:
            return _r1_intent(utt, governed[1], governed[0], HouseholdOperation.QUERY, head)
    if head in _READ_HEADS:
        governed = _governed_object(utt, 1, alarms=True)
        if governed is not None:
            return _r1_intent(utt, governed[1], governed[0], HouseholdOperation.QUERY, head)
    return None


def _r1(utt: _Utterance) -> HouseholdIntent | None:
    verb = _verb_at(utt, 0)
    if verb is not None:
        found = _r1_object(utt, verb)
        if found is not None:
            family, operation, k = found
            return _r1_intent(utt, k, family, operation, verb.form)
    return _query(utt)


def _list_target_follows(utt: _Utterance) -> bool:
    """'... a/en la lista' (an item goes into a list), not 'poné la lista'."""
    for k in range(len(utt) - 1):
        if utt.at(k) in ("a", "en") and utt.at(k + 1) in ("la", "mi", "nuestra") \
                and _object_at(utt, k + 2) is HouseholdFamily.LIST:
            return True
    return False


def _r2(utt: _Utterance) -> HouseholdIntent | None:
    verb = _verb_at(utt, 0)
    if verb is None:
        if utt.at(0) in ("timer", "temporizador") and (
                utt.at(1) in ("de", "para", "por") or _DURATION.match(" ".join(utt.folded[1:]))):
            return HouseholdIntent(HouseholdFamily.TIMER, HouseholdOperation.CREATE, "R2", _rest(utt, 1))
        return None
    if _CREATE in verb.classes or _TIMER_CREATE in verb.classes:
        governed = _governed_object(utt, verb.end, allow_adjective=True)
        if governed is not None and governed[0] is HouseholdFamily.TIMER:
            return HouseholdIntent(HouseholdFamily.TIMER, HouseholdOperation.CREATE, "R2",
                                   _rest(utt, governed[1] + 1), verb.form)
        if governed is not None and governed[0] is HouseholdFamily.LIST and _CREATE in verb.classes \
                and verb.form not in ("pone", "pon", "ponga", "poner"):
            return HouseholdIntent(HouseholdFamily.LIST, HouseholdOperation.CREATE, "R2",
                                   _rest(utt, governed[1] + 1), verb.form)
    words = set(utt.folded)
    if not _mentions_list(utt):
        return None
    if _ADD in verb.classes:
        return HouseholdIntent(HouseholdFamily.LIST, HouseholdOperation.ADD_ITEM, "R2", _rest(utt, verb.end), verb.form)
    if verb.form in ("pone", "pon", "ponga", "poner") and _list_target_follows(utt) and not words & _MUSIC_WORDS:
        return HouseholdIntent(HouseholdFamily.LIST, HouseholdOperation.ADD_ITEM, "R2", _rest(utt, verb.end), verb.form)
    if _REMOVE in verb.classes:
        return HouseholdIntent(HouseholdFamily.LIST, HouseholdOperation.REMOVE_ITEM, "R2", _rest(utt, verb.end),
                               verb.form)
    return None


_ALARM_VERBS = frozenset({"pone", "pon", "ponga", "poner", "programa", "programe", "programar", "crea", "crear",
                          "configura", "configure", "configurar", "agenda", "agendar"})


def _r3a(utt: _Utterance) -> HouseholdIntent | None:
    if utt.at(0) in ("alarma", "despertador"):
        k = 0
        verb = ""
    else:
        found = _verb_at(utt, 0)
        if found is None or found.form not in _ALARM_VERBS:
            return None
        k = found.end
        if utt.at(k) == "me":
            k += 1
        if utt.at(k) in _DETERMINERS:
            k += 1
        if utt.at(k) not in ("alarma", "despertador"):
            return None
        verb = found.form
    rest = " ".join(utt.folded[k + 1:])
    complement = _rest(utt, k + 1)
    if _ARTICLE_CLOCK_UNIT.search(rest):
        return HouseholdIntent(HouseholdFamily.REMINDER, HouseholdOperation.CREATE, "R3a", complement, verb,
                               reason="clock_time")
    if _DURATION.search(rest):
        return HouseholdIntent(HouseholdFamily.TIMER, HouseholdOperation.CREATE, "R3a", complement, verb,
                               reason="duration")
    if _CLOCK.search(rest):
        return HouseholdIntent(HouseholdFamily.REMINDER, HouseholdOperation.CREATE, "R3a", complement, verb,
                               reason="clock_time")
    return HouseholdIntent(HouseholdFamily.TIMER, HouseholdOperation.CLARIFY, "R3a", complement, verb,
                           reason="no_time")


def _r4(utt: _Utterance) -> HouseholdIntent | None:
    verb = _verb_at(utt, 0)
    if verb is None or _DOMOTICS not in verb.classes:
        return None
    if non_immediate_action_reason(utt.heard) != "non_immediate:deferred":
        return None
    return HouseholdIntent(HouseholdFamily.DEFERRED_ACTION, HouseholdOperation.CREATE, "R4", utt.heard, verb.form)


# R6 routine creation (user answer P3F2-R1-Q4, 2026-11-17): a request to create a
# routine claims the turn as a routine PROPOSE with reason "new_routine", so the
# routine's content ("que apague las luces") never reaches the grammar, the
# extended home command or the 7B service_filter.
_ROUTINE_NOUNS = frozenset({"rutina", "rutinas", "automatizacion", "automatizaciones", "regla", "reglas"})
_INDEFINITE = frozenset({"un", "una", "unos", "unas", "otro", "otra"})
# Fillers before the request ("dale, creá ...", "bueno, quiero ...", "a ver, armame ..."), also the
# hesitations and the leading 'y' speech recognition keeps ("eh, creá ...", "y creá ..."). No 'pues':
# it is one mishearing from a voice verification word (tests/unit/memory/test_voice_phrase.py).
_ROUTINE_FILLERS = frozenset({"dale", "bueno", "bue", "ahora", "entonces", "yo", "y", "e", "eh", "ehh", "em", "emm",
                              "mm", "mmm", "este"})
# Request heads before the creation verb, longest first ("quiero", "me gustaría",
# "te pido que", "me podés", "vamos a"). Leading podés/podrías/puedes are already
# stripped by _normalize.
_ROUTINE_REQUEST_HEADS = tuple(sorted((
    ("quiero",), ("quisiera",), ("queria",), ("querria",), ("necesito",), ("necesitaria",), ("preciso",),
    ("me", "gustaria"), ("nos", "gustaria"), ("me", "encantaria"), ("vamos", "a"), ("voy", "a"), ("podemos",),
    ("podriamos",), ("hay", "que"), ("habria", "que"), ("se", "puede"), ("me", "podes"), ("me", "podrias"),
    ("me", "puedes"), ("nos", "podes"), ("nos", "podrias"), ("nos", "puedes"), ("te", "pido", "que"),
    ("ayudame", "a"), ("me", "ayudas", "a"), ("me", "ayudarias", "a"), ("a", "ver"),
), key=len, reverse=True))
_DESIRE_HEADS = frozenset({("quiero",), ("quisiera",), ("queria",), ("querria",), ("necesito",), ("necesitaria",),
                           ("preciso",), ("me", "gustaria"), ("nos", "gustaria"), ("me", "encantaria")})
# Creation verbs (voseo/tú/usted imperative, infinitive, the subjunctive of 'quiero
# que me armes', and the questions '¿me armás/armarías ...?'): any determiner
# ("creá la rutina de la mañana"). No 'inventar' literal (one mishearing from a
# voice verification word): after a request head the unlisted-verb slot covers it.
_ROUTINE_CREATE_FORMS = frozenset({
    "crea", "cree", "crear", "crees", "creas", "crearias", "arma", "arme", "armar", "armes", "armas", "armarias",
    "genera", "genere", "generar", "generes", "generas", "generarias", "defini", "define", "defina", "definir",
    "definas", "definis", "definirias", "inventa", "invente", "inventes", "inventas", "inventarias",
    "disena", "disene", "disenar", "disenes", "disenas", "disenarias", "escribi", "escribe", "escriba", "escribir",
    "escribas", "escribis", "escribirias",
})
# Verbs that also run or change an existing routine ("hacé la rutina", "activá la
# rutina X"): a creation only with 'un/una/otra' or a leading 'nueva' ("activá la
# rutina nueva" runs one).
_ROUTINE_NEW_ONLY_FORMS = frozenset({
    "hace", "haz", "haga", "hacer", "hagas", "haces", "agrega", "agregue", "agregar", "agregues",
    "agregas", "agregarias", "suma", "sume", "sumar", "sumes", "sumas", "sumarias", "anadi", "anade", "anada",
    "anadir", "anadas", "programa", "programe", "programar", "programes", "programas", "programarias", "configura",
    "configure", "configurar", "configures", "configuras", "configurarias", "agenda", "agende", "agendar",
    "agendes", "agendas", "activa", "active", "activar", "actives", "activas", "prepara", "prepare", "preparar",
    "prepares", "preparas", "prepararias", "tener", "automatiza", "automatice", "automatizar",
})


# Words the unlisted-verb slot never stands for: 'que' (only a desire head takes it,
# "quiero que me armes ..."), verbs that only look or tell ("quiero ver las rutinas
# nuevas") and the irregular conditional of 'poner'.
_ROUTINE_SLOT_EXCLUDED = frozenset({
    "que", "ver", "mostra", "mostrar", "muestra", "muestre", "muestres", "mostras", "mostrarias", "mira", "mire",
    "mirar", "leer", "deci", "decir", "conta", "contar", "cuenta", "cuente", "saber", "pondria", "pondrias",
})


def _known_command_word(word: str) -> bool:
    """Whether ``word`` is a known command verb in any person, or a word the slot excludes.

    The lexicon lists imperatives, infinitives and some subjunctives; the voseo
    and tú presents ('ponés', 'ejecutás'), the tú subjunctive ('pongas',
    'ejecutes'), plurals ('pongan') and conditionals ('apagarías') are reduced
    to a listed form, so "me ponés una rutina relajante" is not a creation.
    """
    if word in _ROUTINE_SLOT_EXCLUDED:
        return True
    stems = {word} | {word[:-len(suffix)] for suffix in _ENCLITICS
                      if word.endswith(suffix) and len(word) - len(suffix) >= 2}
    for stem in tuple(stems):
        if stem.endswith(("s", "n")):
            stems.add(stem[:-1])
        if stem.endswith("ias"):
            stems.add(stem[:-3])
        elif stem.endswith("ia"):
            stems.add(stem[:-2])
    return any(stem in _VERB_FORMS for stem in stems)


def _routine_create_form(word: str) -> str | None:
    """The creation verb form of ``word``, enclitics stripped ('armame' -> 'arma')."""
    for form in [word] + [word[:-len(suffix)] for suffix in _ENCLITICS
                          if word.endswith(suffix) and len(word) - len(suffix) >= 2]:
        if form in _ROUTINE_CREATE_FORMS or form in _ROUTINE_NEW_ONLY_FORMS:
            return form
    return None


def _routine_creation(utt: _Utterance) -> HouseholdIntent | None:
    """A request to create a routine (P3F2-R1-Q4), or None.

    Shape: [filler] [request head [que]] [me|nos] [creation verb] [determiners]
    [nueva] rutina|automatización|regla [nueva]. Without a creation verb, or with
    a verb that can also run or change a routine, the noun needs 'un/una/otra'
    or a leading 'nueva' ("una rutina que ...", "quiero una nueva rutina ...");
    a trailing 'nueva' counts only with a creation verb or no verb at all
    ("rutina nueva que ...", but not "activá la rutina nueva"). After a request
    head or a clitic, an unlisted word that is not a known command verb in any
    person (``_known_command_word``) stands for the verb under the same
    condition (a conditional such as "¿me ...rías una rutina nueva?").
    """
    k = 0
    while utt.at(k) in _ROUTINE_FILLERS:
        k += 1
    head = next((words for words in _ROUTINE_REQUEST_HEADS if utt.folded[k:k + len(words)] == words), None)
    if head is not None:
        k += len(head)
        if head in _DESIRE_HEADS and utt.at(k) == "que":
            k += 1
    clitic = utt.at(k) in ("me", "nos")
    if clitic:
        k += 1
    form = _routine_create_form(utt.at(k))
    slot = False
    if form is not None:
        k += 1
    elif (head is not None or clitic) and not _known_command_word(utt.at(k)) \
            and utt.at(k) not in _DETERMINERS | _ADJECTIVES | _ROUTINE_NOUNS:
        slot = True
        k += 1
    indefinite = False
    for _ in range(2):
        if utt.at(k) in _DETERMINERS:
            indefinite = indefinite or utt.at(k) in _INDEFINITE
            k += 1
    new = utt.at(k) in _ADJECTIVES
    if new:
        k += 1
    if utt.at(k) not in _ROUTINE_NOUNS:
        return None
    if utt.at(k + 1) in _ADJECTIVES:
        # A trailing 'nueva' marks a creation only without a verb that could run the routine.
        new = new or (form is None and not slot)
        k += 1
    if form in _ROUTINE_CREATE_FORMS or indefinite or new:
        return HouseholdIntent(HouseholdFamily.ROUTINE, HouseholdOperation.PROPOSE, "R6", _rest(utt, k + 1),
                               form or "", reason="new_routine")
    return None


def intercom_slots(complement: str) -> tuple[str | None, int]:
    """Return the explicit destination, if any, and the literal message offset.

    Punctuation between the verb and a destination cannot turn a room
    request into broadcast. "Que" marks the start of an explicitly literal body.
    """
    start = _INTERCOM_PUNCTUATION.match(complement).end()
    direction = _INTERCOM_DIRECTION.match(complement, start)
    marker = _INTERCOM_MESSAGE_MARKER.search(complement, start)
    target_start = direction.end() if direction else start
    target_end = marker.start() if marker else len(complement)
    target = " ".join(complement[target_start:target_end].strip(" ,").split())
    if direction or target.casefold() in INTERCOM_BROADCAST_TARGETS:
        return target, marker.end() if marker else len(complement)
    opening_colon = complement.find(":", 0, start)
    if opening_colon != -1:
        return None, opening_colon + 1
    if marker and marker.start() == start:
        return None, marker.end()
    return None, 0


def _intercom_intent(utt: _Utterance, verb: _Verb | None) -> HouseholdIntent | None:
    """Claim the command head and leave destination resolution to intercom."""
    if verb is None or _INTERCOM not in verb.classes or verb.infinitive:
        return None
    head = utt.at(0)
    next_word = utt.at(verb.end)
    explicit_destination = next_word in _INTERCOM_DESTINATION_HEADS
    announce = head == verb.form and (head.startswith(("anunci", "comunic")) or (
        head in {"avisa", "avise"} and next_word == "que"
    ))
    if not announce and (not explicit_destination or head.endswith(("me", "nos"))):
        return None
    if not utt.word_ends:
        return HouseholdIntent(HouseholdFamily.INTERCOM, HouseholdOperation.EXECUTE, "R6",
                               _rest(utt, verb.end), verb.form)

    start, last_word = utt.word_ends[verb.end - 1], utt.word_ends[-1]
    tail = _INTERCOM_PUNCTUATION.match(utt.source, last_word)
    end = tail.end()
    if end != len(utt.source):
        # A following filler or clause is outside the payload.
        end = last_word + len(tail[0].rstrip(" \t,;:"))
    complement = utt.source[start:end]
    _, message_start = intercom_slots(complement)
    return HouseholdIntent(HouseholdFamily.INTERCOM, HouseholdOperation.EXECUTE, "R6",
                           complement.strip(), verb.form, literal_span=(start + message_start, end))


def _r6(utt: _Utterance) -> HouseholdIntent | None:
    created = _routine_creation(utt)
    if created is not None:
        return created
    verb = _verb_at(utt, 0)
    if verb is not None and _AUTOMATE in verb.classes:
        return HouseholdIntent(HouseholdFamily.ROUTINE, HouseholdOperation.PROPOSE, "R6", _rest(utt, verb.end),
                               verb.form, reason="new_routine")
    if verb is not None:
        governed = _governed_object(utt, verb.end, allow_adjective=True)
        if governed is not None and governed[0] is HouseholdFamily.ROUTINE:
            if _ROUTINE_EXEC in verb.classes:
                operation = HouseholdOperation.EXECUTE
            elif _ROUTINE_PROPOSE in verb.classes:
                operation = HouseholdOperation.PROPOSE
            else:
                operation = None
            if operation is not None:
                noun = governed[1]
                complement = (_routine_name_after(utt, noun) if operation is HouseholdOperation.EXECUTE
                              else _rest(utt, noun + 1))
                return HouseholdIntent(HouseholdFamily.ROUTINE, operation, "R6", complement, verb.form)
    notification = _NOTIFICATION.fullmatch(utt.text)
    if notification is not None:
        return HouseholdIntent(HouseholdFamily.NOTIFICATION,
                               HouseholdOperation.DISABLE if notification["disable"] else HouseholdOperation.EXECUTE, "R6",
                               utt.heard, verb.form if verb is not None else "")
    return _intercom_intent(utt, verb)


_SINGLE_RULES = (_r3, _r1, _r2, _r3a, _r4, _r6)


def _classify_single(utt: _Utterance) -> HouseholdIntent | None:
    for rule in _SINGLE_RULES:
        found = rule(utt)
        if found is not None:
            return found
    return None


# --------------------------------------------------------------------------
# R5: compound pre-split
# --------------------------------------------------------------------------

def _split_clauses(utt: _Utterance) -> list[_Utterance]:
    """(a) Imperative clauses joined by y/también/y después/y luego."""
    segments, start = [], 0
    for k in range(1, len(utt)):
        if utt.at(k) not in _CONJUNCTIONS or k <= start:
            continue
        segment = utt.slice(start)
        if _intercom_intent(segment, _verb_at(segment, 0)) is not None:
            break
        if _CONTENT_MARKERS & set(utt.folded[start:k]):
            return []
        j = k + 1
        while utt.at(j) in _CONNECTORS:
            j += 1
        if not _clause_head(utt, j):
            continue
        segments.append(utt.slice(start, k))
        start = j
    if not segments:
        return []
    segments.append(utt.slice(start))
    return segments


def _split_objects(utt: _Utterance, verb: _Verb) -> list[_Utterance]:
    """(b) One verb governing two or more household object nouns."""
    first = _r1_object(utt, verb)
    if first is None:
        return []
    family, _, k = first
    if family is HouseholdFamily.REMINDER and utt.at(k + 1) in ("de", "para", "que"):
        return []
    splits = []  # (conjunction index, start of the next object phrase)
    for c in range(k + 1, len(utt)):
        if utt.at(c) != "y":
            continue
        j = c + 1
        if utt.at(j) == "tambien":
            j += 1
        if _governed_object(utt, j) is not None:
            splits.append((c, j))
    if not splits:
        return []
    head = utt.slice(0, verb.end)
    segments = [utt.slice(0, splits[0][0])]
    for index, (_, start) in enumerate(splits):
        end = splits[index + 1][0] if index + 1 < len(splits) else None
        segments.append(head.join(utt.slice(start, end)))
    return segments


def _room_prep_at(utt: _Utterance, k: int) -> int:
    for prep in _ROOM_PREPS:
        if utt.folded[k:k + len(prep)] == prep and utt.at(k + len(prep)) in ROOM_ALIASES:
            return len(prep)
    return 0


_R1_CLASSES = frozenset(cls for cls, _ in _R1_OPERATIONS)


def _split_targets(utt: _Utterance, verb: _Verb) -> list[_Utterance]:
    """(c) One verb with two or more targets by deterministic ellipsis.

    A domotics verb splits on a second room or device ("apagá la luz del
    living y de la cocina", "apagá la luz y la tele"). A domotics or object
    operation verb whose own object is not a household object also splits on
    a household object noun ("apagá la luz y el timer", "pará la música y el
    timer"), so the household part is claimed instead of silently dropped.
    Every qualifying 'y' splits, so the compound part limit sees every target.
    """
    domotics = _DOMOTICS in verb.classes
    if not domotics and not verb.classes & _R1_CLASSES:
        return []
    lists = _mentions_list(utt)
    own_object = _governed_object(utt, verb.end) is not None
    segments: list[_Utterance] = []
    head: _Utterance | None = None  # words carried into the current segment by ellipsis
    start = 0

    def current(end: int | None) -> _Utterance:
        words = utt.slice(start, end)
        return head.join(words) if head is not None else words

    for c in range(verb.end, len(utt)):
        if utt.at(c) != "y":
            continue
        if _CONTENT_MARKERS & set(utt.folded[:c]):
            return []
        j = c + 1
        if utt.at(j) == "tambien":
            j += 1
        k = j + 1 if utt.at(j) in _DETERMINERS else j
        if domotics and not lists and _room_prep_at(utt, j):
            previous = current(c)
            room_at = next((r for r in range(verb.end, len(previous)) if _room_prep_at(previous, r)),
                           len(previous))
            segments.append(previous)
            head = previous.slice(0, room_at)
        elif (domotics and not lists and utt.at(k) in _ENTITY_WORDS and k > j) \
                or (not own_object and _governed_object(utt, j) is not None):
            segments.append(current(c))
            head = utt.slice(0, verb.end)
        else:
            continue
        start = j
    if not segments:
        return []
    segments.append(current(None))
    return segments


def _names_object(utt: _Utterance) -> bool:
    """A household object noun appears anywhere in the utterance."""
    return any(_object_at(utt, k) is not None for k in range(len(utt)))


def _deferred(intent: HouseholdIntent | None, segment: _Utterance) -> bool:
    if intent is None:
        verb = _verb_at(segment, 0)
        if verb is not None and _DOMOTICS in verb.classes:
            return False  # R4 already ran on this part and found no deferral
        return non_immediate_action_reason(segment.heard) == "non_immediate:deferred"
    return (intent.family is HouseholdFamily.DEFERRED_ACTION
            or (intent.family is HouseholdFamily.REMINDER and intent.operation is HouseholdOperation.CREATE))


def _conflicting(parses: list) -> bool:
    """Two unclaimed parts ask for opposite operations on the same domain and room."""
    seen: dict[tuple[str | None, str | None], set[str]] = {}
    for parsed in parses:
        if parsed is None or parsed.intent is None:
            continue
        seen.setdefault((parsed.domain, parsed.room), set()).add(parsed.intent)
    return any(pair <= operations for operations in seen.values() for pair in _OPPOSITES)


def _r5(utt: _Utterance, max_parts: int) -> HouseholdIntent | None:
    verb = _verb_at(utt, 0)
    if verb is None:
        return None
    segments = _split_clauses(utt) or _split_objects(utt, verb) or _split_targets(utt, verb)
    if len(segments) < 2:
        return None

    def clarify(reason: str) -> HouseholdIntent:
        return HouseholdIntent(HouseholdFamily.COMPOUND, HouseholdOperation.CLARIFY, "R5", utt.heard, verb.form,
                               reason=reason)

    intents = [_classify_single(segment) for segment in segments]
    if any(intent is not None and intent.controls_routine for intent in intents):
        return clarify("routine_control_part")
    if len(segments) > max_parts:
        return clarify("too_many_parts")
    parses = []  # grammar parse of each unclaimed part, computed once (review F7)
    for segment, intent in zip(segments, intents):
        if _deferred(intent, segment):
            return clarify("deferred_part")
        if intent is not None and intent.operation is HouseholdOperation.CLARIFY:
            return clarify("unparsed_part")
        # T037-U6 (seed C02-CMP-0004): a lock, door or alarm-panel order is a
        # deterministic parse its own part refuses (voice never controls
        # them); it never reaches a domotics producer and never makes the
        # whole compound CLARIFY.
        security = intent is None and security_order(segment.heard) is not None
        if security and has_deferral_time(segment.heard):
            # T037-U6-RV2: "activá la alarma a las 11" has no domotics device, so
            # _deferred misses its time; the time may be meant for every part.
            return clarify("deferred_part")
        if intent is None and not security and _names_object(segment):
            # A household object never reaches a domotics producer as an unclaimed part.
            return clarify("unparsed_part")
        parsed = parse_command(segment.heard) if intent is None and not security else None
        if parsed is not None and parsed.quality != "full" and parse_home_command(segment.heard) is None:
            return clarify("unparsed_part")
        parses.append(parsed)
    if _conflicting(parses):
        return clarify("conflicting_parts")
    parts = tuple(HouseholdPart(segment.heard, intent) for segment, intent in zip(segments, intents))
    return HouseholdIntent(HouseholdFamily.COMPOUND, HouseholdOperation.EXECUTE, "R5", utt.heard, verb.form,
                           parts=parts)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def classify_household_intent(text: str, *, wake_words: Iterable[str] = (),
                              max_compound_parts: int = 3) -> HouseholdIntent | None:
    """Claim a turn for a household family, or return None to leave it unclaimed.

    Pure (no I/O, no logging); the router and the dispatcher call it on the
    same text and must agree.

    Args:
        text: Utterance as transcribed (wake words may still be present).
        wake_words: Configured wake words to strip besides the grammar's.
        max_compound_parts: More compound parts make the compound CLARIFY.

    Returns:
        The claimed intent, or None when no rule claims the turn.
    """
    utt = _normalize(text, wake_words)
    if not len(utt):
        return None
    found = _r3(utt)
    if found is not None:
        return found
    found = _intercom_intent(utt, _verb_at(utt, 0))
    if found is not None:
        return found
    found = _r5(utt, max_compound_parts)
    if found is not None:
        return found
    for rule in (_r1, _r2, _r3a, _r4, _r6):
        found = rule(utt)
        if found is not None:
            return found
    return None


def starts_with_reminder_request(text: str, wake_words: Iterable[str] = ()) -> bool:
    """True when the utterance's head is a reminder or deferral verb (R3).

    Used by the grammar's ``non_immediate_action_reason`` as a second guard
    so "recordame apagar la luz" never becomes an immediate action.
    """
    return _REMINDER_HEAD.match(_normalize(text, wake_words).text) is not None
