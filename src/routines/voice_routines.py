"""Voice access to persistent routines and to the web review tray.

``VoiceRoutineService`` answers "¿qué rutinas hay?" and "¿cómo salió la
rutina de la mañana?" from the RoutineScheduler's store (``routines.db``), the
same definitions and receipts the web catalog shows. Admitted voice commands
can start saved routines, request a stop, or disable them for review.

Read detection is lexical and pure. A routine noun ("rutina", "automatización")
is required at a word boundary ("rutinario" is not one) and a bare "cuando"
never makes a turn a routine (CORE-04).

The scheduler is bound late: ``main`` builds the scheduler after the
orchestrator and the RequestRouter, so both receive this service at
construction and ``bind_scheduler`` attaches the application's single
scheduler once it exists.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
import logging
import re
import unicodedata

from src.core.sqlite_work import AuthorityGuard
from src.core.task_outcome import (
    CURRENT_REQUEST_ID, DispatchState, EffectState, PersistenceState, TaskOutcome, TaskStatus,
)
from src.memory.fact_ledger import MemoryOrigin, MemorySource
from src.nlu.command_grammar import WAKE_TERMS
from src.nlu.household_intents import HouseholdFamily, HouseholdIntent, HouseholdOperation, strip_fillers
from src.routines.models import (
    ActionStatus, BrakeStatus, RoutineBrake, RoutineConflict, RoutineNameConflict, RoutineNotFound, RoutineRun,
    RunStatus, ScheduledRoutine, TimeSchedule,
)
from src.routines.proposals import ProposalDecision, ProposalOperation, ProposalUnavailable, RoutineProposalTray
from src.routines.routine_scheduler import RoutineScheduler
from src.routines.routine_store import HOUSEHOLD_VOICE
from src.routines.voice_proposal_extractor import ProposalIssue, VoiceProposalError, VoiceProposalExtractor

logger = logging.getLogger(__name__)

__all__ = [
    "ROUTINE_CREATE_REFUSAL", "RoutineCatalogEntry", "RoutineProposalResult", "RoutineReadKind", "RoutineReadQuery",
    "RoutineReadResult", "VoiceRoutineService", "is_unclaimed_routine_creation",
]

ROUTINE_CREATE_REFUSAL = "Todavía no creo rutinas por voz. Podés crearlas desde la web."
_UNCLAIMED_CREATION = re.compile(
    r"\b(?:(?:crea|crear) (?:una )?(?:rutina|automatizacion)|nueva (?:rutina|automatizacion)|automatiza)\b"
)
# Keep each lookbehind fixed-width for Python's regex engine.
_CONFIGURATION_CONTEXT = (
    r"(?<!\bse )(?<!\bqu[eé] )(?<!\b(?:c[oó]mo|cu[aá]l) )"
    r"(?<!\b(?:d[oó]nde|qui[eé]n) )(?<!\bcu[aá]ndo )"
)
_ROUTINE_DIRECTIVE = re.compile(
    rf"{_CONFIGURATION_CONTEXT}\b(?:programa|configura) que\b"
    r"|\b(?:quiero|necesito|haz|hace|pon|pone) que (?:cuando|cada vez que)\b"
)
_RECURRENCE_CLAUSE = re.compile(r"\bcada vez que\b")
_RECURRENCE_QUESTION = re.compile(r"\bcuando se\b")
_CONFIGURATION_VERB = re.compile(rf"{_CONFIGURATION_CONTEXT}\b(?:configur(?:a|ame|ar|e)|program(?:a|ame|ar|e))\b")
_VOSEO_PROGRAM_VERB = re.compile(rf"{_CONFIGURATION_CONTEXT}\bprogramá\b", re.I)
_NOMINAL_PROGRAM = re.compile(
    r"\b(?:el|del|al|un|una|algun|alguna|este|ese|aquel|mi|tu|su|nuestro|otro|mismo|cualquier)"
    r"(?: (?:buen|buena|gran|mejor|nuevo|nueva|viejo|mismo|otro|primer|ultimo|pequeno|lindo|unico))? programa\b"
)
_QUESTION_MARKER = re.compile(r"[¿?]|\b(?:cuándo|qué|cuál(?:es)?|dónde|cómo|quién(?:es)?|cuánt[oa]s?)\b", re.I)
_INTERROGATIVE = r"(?:cuando|que|cual(?:es)?|donde|como|quien(?:es)?|cuant[oa]s?)"
_QUESTION_OPENING = re.compile(
    rf"^(?:(?:a|con|de|en|hasta|para|por) )?{_INTERROGATIVE}\b"
    r"|^(?:(?:me )?(?:podes|podrias|puedes) )?(?:decime|dime|decirme|decir|contame|contarme|sabes) "
    rf"(?:si|desde|(?:(?:a|con|de|en|hasta|para|por) )?{_INTERROGATIVE})\b"
)

_SINGULAR_NOUNS = frozenset({"rutina", "automatizacion"})
_PLURAL_NOUNS = frozenset({"rutinas", "automatizaciones"})
_ROUTINE_VOCABULARY = _SINGULAR_NOUNS | _PLURAL_NOUNS | {"rutinario", "rutinaria", "rutinarios", "rutinarias"}
# Words heard before the request itself: wake words, fillers and courtesy.
_LEADS = frozenset(WAKE_TERMS | {"che", "oye", "oi", "hola", "bueno", "dale", "eh", "ehh", "em", "y", "ahora",
                                  "entonces", "porfa", "por", "favor", "disculpa", "perdon"})
# Interrogative heads: "¿qué/cuáles/cuántas rutinas ...?", "¿cómo salió ...?", "¿a qué hora corre ...?".
_WH_HEADS = frozenset({"que", "cuales", "cual", "cuantas", "cuantos", "como", "cuando", "donde"})
# Yes/no question heads and past-tense status verbs that open a question about a routine.
_YES_NO_HEADS = frozenset({"hay", "tengo", "tenemos", "tenes", "tiene", "tienen", "existe", "existen", "esta",
                           "estan", "se", "ya", "funciono", "anduvo", "salio", "corrio", "termino", "quedo",
                           "fallo", "hubo"})
# Verbs that ask to be told or shown something (legacy list phrases included:
# "muestra las rutinas", "lista de rutinas").
_READ_VERBS = frozenset({"decime", "dime", "deci", "decir", "decirme", "decis", "mostrame", "muestrame", "muestra",
                         "mostra", "mostrar", "mostrarme", "listame", "lista", "listar", "listado", "nombrame",
                         "nombra", "contame", "cuentame", "conta", "contar", "leeme", "lee", "leer", "ver", "saber",
                         "sabes", "sabe", "consulta", "consultame", "consultar", "repasame", "repasar",
                         "enumera", "enumerame"})
# Verbs that would create, change, run or stop a routine (or act on the home):
# a turn with one of them before the routine noun is never a read.
_BLOCKING_VERBS = frozenset({
    "crea", "crear", "cree", "creame", "arma", "armar", "arme", "armame", "armes", "haceme", "hacer", "haga",
    "hace", "programa", "programar", "programe", "programame", "configura", "configurar", "configure", "agenda",
    "agendar", "activa", "activar", "active", "activala", "desactiva", "desactivar", "desactive", "deshabilita",
    "deshabilitar", "habilita", "habilitar", "ejecuta", "ejecutar", "ejecute", "ejecutes", "ejecutala", "corre",
    "correr", "corra", "corras", "lanza", "lanzar", "lance", "arranca", "arrancar", "arranque", "inicia",
    "iniciar", "inicie", "pone", "pon", "poner", "ponga", "ponela", "cancela", "cancelar", "cancele", "borra",
    "borrar", "borre", "elimina", "eliminar", "elimine", "saca", "sacar", "saque", "quita", "quitar", "quite",
    "anula", "anular", "para", "parar", "pare", "detene", "detener", "deten", "detenga", "frena", "frenar",
    "pausa", "pausar", "segui", "seguir", "continua", "continuar", "reanuda", "reanudar", "cambia", "cambiar",
    "cambie", "modifica", "modificar", "edita", "editar", "renombra", "renombrar", "agrega", "agregar", "suma",
    "sumar", "automatiza", "automatizar", "automatice", "prende", "prender", "apaga", "apagar", "enciende",
    "encender", "define", "defini", "definir", "genera", "generar", "proba", "probar",
})
# Present-tense forms that are also third-person questions after an interrogative
# head ("¿a qué hora corre la rutina X?", "¿qué hace la rutina X?").
_QUESTION_FORMS = frozenset({"corre", "ejecuta", "activa", "arranca", "inicia", "hace", "prende", "apaga",
                             "pone", "programa"})
# A complement that describes instead of naming ("una rutina que apague ...").
_DESCRIPTION_STARTS = frozenset({"que", "para", "con", "cuando", "si", "y", "o"})
# Words after the name that belong to the question ("la rutina X se ejecutó hoy").
_NAME_TAIL_CUES = frozenset({"se", "salio", "anduvo", "funciono", "corrio", "esta", "estan", "ya", "fue",
                             "termino", "quedo", "fallo", "hoy", "ayer", "anoche", "ahora", "recien", "tiene",
                             "corre", "ejecuta", "activa", "arranca", "hace", "como", "cuando", "bien", "mal",
                             "o", "no", "ejecuto", "activo"})
# Third-person verbs that make the words after the noun a predicate, not a name
# ("¿qué rutina apaga las luces?", "la rutina riego prende el living") (T033-U1-RV3).
_PREDICATE_VERBS = frozenset({"apaga", "prende", "enciende", "sube", "baja", "abre", "cierra", "riega", "desactiva",
                              "manda", "avisa", "cambia", "toca", "suena", "reproduce", "controla", "maneja", "usa",
                              "incluye", "empieza", "comienza", "dura"})
_NAME_BREAKS = _NAME_TAIL_CUES | _QUESTION_FORMS | _PREDICATE_VERBS
# Words that introduce the name itself ("la rutina que se llama riego").
_NAMING_LEADS = (("que", "se", "llama"), ("que", "se", "llamaba"), ("con", "el", "nombre"), ("con", "nombre"),
                 ("de", "nombre"))
# Past-tense status verbs after the noun that make "la rutina X anduvo/falló" a question.
_STATUS_AFTER = frozenset({"salio", "anduvo", "funciono", "corrio", "termino", "fallo", "ejecuto"})
# "¿la rutina riego está activa?": a state copula after the noun followed by one
# of these makes the sentence a question about the routine's state (T033-U1-RV1).
_STATE_COPULAS = frozenset({"esta", "estan", "sigue", "siguen", "quedo", "quedaron"})
_STATE_ADJECTIVES = frozenset({"activa", "activo", "activada", "activado", "activas", "activadas", "desactivada",
                               "desactivado", "desactivadas", "habilitada", "habilitado", "deshabilitada",
                               "deshabilitado", "andando", "corriendo", "funcionando", "programada", "programado",
                               "pausada", "pausado"})
# T033-U1-RV2: "rutina" is everyday Spanish ("una buena rutina de ejercicios", "la
# mejor rutina de skincare"). An indefinite determiner or a qualifying adjective
# just before the noun makes the turn about routines in general, never a saved
# one; only an existence question ("¿hay alguna rutina?") still lists them.
_INDEFINITES = frozenset({"una", "un", "unas", "unos", "alguna", "algun", "algunas", "algunos", "otra", "otro",
                          "otras", "otros", "cualquier", "cierta", "ciertas"})
_QUALIFIERS = frozenset({"buena", "buen", "buenas", "buenos", "mejor", "mejores", "linda", "lindas", "perfecta",
                         "ideal", "recomendada", "recomendadas", "tipica", "clasica", "sana", "simple"})
_EXISTENCE_HEADS = frozenset({"hay", "tengo", "tenemos", "tenes", "tiene", "tienen", "existe", "existen"})
# Words that tie a consult to a saved routine's state, runs or schedule. A consult
# without one whose name matches no saved routine is not a routine question
# ("¿qué rutina de skincare me recomendás?"): callers that did not get it from the
# household layer pass it on instead of answering that the name is unknown.
_ROUTINE_CUES = _STATUS_AFTER | _STATE_ADJECTIVES | frozenset({
    "corre", "ejecuta", "ejecutando", "arranca", "inicia", "hace", "anda", "funciona", "sale", "dispara", "hora",
    "horario", "horarios", "estado", "ultima", "ultimo", "ejecucion", "ejecuciones"})
_SUBJECT_DETERMINERS = frozenset({"la", "el", "las", "los", "mi", "mis", "nuestra", "nuestras"})
# Leading words ignored when two names are compared ("la rutina de la mañana" ~ "Mañana").
_NAME_LEADS = frozenset({"la", "el", "los", "las", "de", "del", "mi", "mis", "rutina", "rutinas",
                         "automatizacion", "llamada", "llamado", "que", "se", "llama", "nombre"})

_WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábados", "domingos")
_MAX_SPOKEN_NAMES = 10
_MAX_CANDIDATES = 3

_UNAVAILABLE = "No puedo consultar las rutinas ahora. Podés verlas desde la web."
_NOT_FOUND = "No encontré una rutina con ese nombre. Podés preguntarme qué rutinas hay."

_LAST_RUN = {
    None: "Todavía no se ejecutó.",
    RunStatus.RUNNING: "Se está ejecutando ahora.",
    RunStatus.SUCCEEDED: "La última ejecución terminó bien.",
    RunStatus.FAILED: "La última ejecución falló.",
    RunStatus.UNCERTAIN: "En la última ejecución no pude confirmar si se aplicaron todas sus acciones.",
    RunStatus.SKIPPED: "La última vez no se ejecutó porque no se cumplían sus condiciones.",
    RunStatus.INTERRUPTED: "La última ejecución se interrumpió.",
}


class RoutineReadKind(StrEnum):
    """Read-only routine operations available by voice."""

    LIST = "list"
    CONSULT = "consult"


@dataclass(frozen=True)
class RoutineReadQuery:
    """A detected routine question.

    Attributes:
        kind: List every routine, or consult one by name.
        names: CONSULT only: the heard name candidates, longest first, each
            already normalized as a comparison key.
        cued: CONSULT only: the turn asks about a saved routine's state, runs
            or schedule ("¿está activa ...?", "¿cómo salió ...?", "¿a qué hora
            corre ...?"), so an unknown name is still a routine question.
    """

    kind: RoutineReadKind
    names: tuple[str, ...] = ()
    cued: bool = False

    @property
    def always_answered(self) -> bool:
        """Whether ``answer`` never declines this query (a list, or a cued consult)."""
        return self.kind is RoutineReadKind.LIST or self.cued


@dataclass(frozen=True)
class RoutineCatalogEntry:
    """One saved routine as the web catalog lists it (GET /api/routines)."""

    routine_id: str
    name: str
    enabled: bool
    revision: int
    latest_run_status: str | None = None


@dataclass(frozen=True)
class RoutineReadResult:
    """The spoken answer of a routine read and its task evidence.

    Attributes:
        response: Spanish reply for the household.
        outcome: COMPLETED for an answer read from the store; NEEDS_INPUT for
            an unknown or ambiguous name; FAILED when the store is unavailable.
        entries: The catalog rows the answer was built from.
        declined: An uncued consult named no saved routine: the turn may be
            a general question that mentions "rutina" (T033-U1-RV2). A caller
            that routed it here by its words passes it on (conversation
            path); the household layer, which claimed it, answers
            ``response`` (NEEDS_INPUT).
    """

    response: str
    outcome: TaskOutcome
    entries: tuple[RoutineCatalogEntry, ...] = ()
    declined: bool = False


@dataclass(frozen=True)
class RoutineControlResult:
    """Spoken acknowledgment and the evidence available before replying."""

    response: str
    outcome: TaskOutcome
    receipt_id: str | None = None


@dataclass(frozen=True)
class RoutineProposalResult:
    """Reply and evidence for an attempted admission to web review."""

    response: str
    outcome: TaskOutcome


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", _fold(text)).split()


def is_unclaimed_routine_creation(text: str, *, wake_words: Iterable[str] = ()) -> bool:
    """Recognize creation or configuration of routines outside the request grammar."""
    original = unicodedata.normalize("NFC", text)
    # The accented imperative "programá" cannot be the noun "programa".
    voseo_program = _VOSEO_PROGRAM_VERB.search(original)
    words = strip_fillers(original, wake_words, extra_leads=_LEADS)
    phrase = " ".join(words)
    if _UNCLAIMED_CREATION.search(phrase):
        return True
    # A question that leads with an interrogative can mention a configuration verb.
    if _QUESTION_OPENING.match(phrase):
        return False
    command_text = _NOMINAL_PROGRAM.sub(" ", phrase)
    names_routine = not _ROUTINE_VOCABULARY.isdisjoint(words)
    # A directive can have an embedded question or a trailing "¿podés?".
    if _ROUTINE_DIRECTIVE.search(command_text) or (
        names_routine and (voseo_program or _CONFIGURATION_VERB.search(command_text))
    ):
        return True
    recurrence = _RECURRENCE_CLAUSE.search(phrase)
    if recurrence is None or (recurrence.start() != 0 and not names_routine):
        return False
    # A bare recurrence clause may introduce a question even if STT dropped the accents.
    return not (_QUESTION_MARKER.search(original) or _RECURRENCE_QUESTION.search(phrase, recurrence.end()))


def name_key(name: str) -> str:
    """Comparison key of a routine name: accent/case folded, leading articles dropped.

    Args:
        name: A stored routine name or a heard name.

    Returns:
        The key; two names are the same routine name when their keys are equal.
    """
    words = _words(name)
    while words and words[0] in _NAME_LEADS:
        words.pop(0)
    return " ".join(words)


def _strip_leads(words: list[str]) -> list[str]:
    k = 0
    while k < len(words):
        if words[k] == "a" and k + 1 < len(words) and words[k + 1] == "ver":
            k += 2
        elif words[k] in _LEADS:
            k += 1
        else:
            break
    return words[k:]


def _states_after(after: list[str]) -> bool:
    """Whether the words after the noun ask about the routine's runs or state."""
    if _STATUS_AFTER & set(after):
        return True
    return any((word in _STATE_COPULAS and after[k + 1] in _STATE_ADJECTIVES)
               or (word == "se" and after[k + 1] in _QUESTION_FORMS) for k, word in enumerate(after[:-1]))


def _name_candidates(complement: list[str]) -> tuple[str, ...]:
    """Name keys heard after the routine noun, longest first; () when it names nothing.

    "la rutina que se llama riego" names Riego (T033-U1-RV3: the naming lead is
    read before "que" makes the complement a description). A complement that
    starts with a tail cue or a predicate verb ("¿qué rutina corre a las 7?",
    "¿qué rutina apaga las luces?") names nothing, so the question lists.
    """
    lead = next((lead for lead in _NAMING_LEADS if tuple(complement[:len(lead)]) == lead), None)
    if lead is not None:
        complement = complement[len(lead):]
    elif complement and complement[0] in _DESCRIPTION_STARTS:
        return ()
    cut = next((k for k, word in enumerate(complement) if word in _NAME_BREAKS), None)
    if not complement or cut == 0:
        return ()
    candidates = [name_key(" ".join(complement))]
    if cut is not None:
        candidates.append(name_key(" ".join(complement[:cut])))
    return tuple(dict.fromkeys(key for key in candidates if key))


def _schedule_phrase(schedule: TimeSchedule) -> str:
    hour, minute = schedule.at.split(":")
    hour_text = str(int(hour))
    at = f"a la {hour_text}:{minute}" if hour_text == "1" else f"a las {hour_text}:{minute}"
    days = sorted(set(schedule.weekdays))
    if days == list(range(7)):
        when = "todos los días"
    elif days == list(range(5)):
        when = "de lunes a viernes"
    elif days == [5, 6]:
        when = "los fines de semana"
    else:
        names = [_WEEKDAYS[day] for day in days]
        when = "los " + (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " y " + names[-1])
    return f"{at} {when}"


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " y " + names[-1]


class VoiceRoutineService:
    """Voice adapter over the application's saved routines and scheduler.

    Args:
        scheduler: The application's RoutineScheduler, or None until
            ``bind_scheduler`` attaches it (late binding in main).
        voice_brake: Whether STOP and DISABLE are available by voice.
        proposal_extractor: Optional local extractor for new definitions.
        proposal_tray: The dashboard's shared, conversation-scoped review tray.
    """

    def __init__(
        self, scheduler: RoutineScheduler | None = None, *, voice_brake: bool = False,
        proposal_extractor: VoiceProposalExtractor | None = None,
        proposal_tray: RoutineProposalTray | None = None,
    ) -> None:
        if (proposal_extractor is None) != (proposal_tray is None):
            raise ValueError("incomplete_voice_proposal_dependencies")
        self._scheduler = scheduler
        self._voice_brake = voice_brake
        self._proposal_extractor = proposal_extractor
        self._proposal_tray = proposal_tray
        self._proposal_review_available: AuthorityGuard | None = None

    def bind_proposal_review(self, available: AuthorityGuard) -> None:
        """Attach the availability check for the application's web review service."""
        self._proposal_review_available = available

    def claims_proposal(self, intent: HouseholdIntent | None) -> bool:
        """Whether this voice request can enter the local proposal path."""
        return (self._proposal_extractor is not None and intent is not None
                and intent.family is HouseholdFamily.ROUTINE
                and intent.operation is HouseholdOperation.PROPOSE and intent.reason == "new_routine")

    async def propose(self, complement: str, *, sources: tuple[MemorySource, ...],
                      guard: AuthorityGuard | None) -> RoutineProposalResult:
        """Admit a volatile creation proposal from a current, unmuted capture."""
        tray, extractor = self._proposal_tray, self._proposal_extractor
        if tray is None or extractor is None:
            return RoutineProposalResult("Podés crear la rutina desde la web.", TaskOutcome(TaskStatus.UNSUPPORTED))

        def current() -> bool:
            return (guard is not None and guard() and len(sources) == 1
                    and sources[0].origin is MemoryOrigin.VOICE and not sources[0].fact_dependencies
                    and tray.source_policy.sources_available(sources))

        def review_available() -> bool:
            review = self._proposal_review_available
            return self._scheduler is not None and review is not None and review()

        cancelled = RoutineProposalResult("", TaskOutcome(TaskStatus.CANCELLED,
                                                           persistence=PersistenceState.NOT_WRITTEN))
        if not current():
            return cancelled
        unavailable = RoutineProposalResult("Ahora no está disponible la revisión de rutinas en la web.",
                                            TaskOutcome(TaskStatus.UNSUPPORTED, persistence=PersistenceState.NOT_WRITTEN))
        if not review_available():
            return unavailable
        source = sources[0]
        try:
            spec = await extractor.extract(complement)
            if not current():
                return cancelled
            if not review_available():
                return unavailable
            submission = tray.submit_for_review(
                ProposalOperation.CREATE, conversation_id=source.conversation_id, spec=spec, expires_at=source.expires_at,
            )
        except PermissionError:
            return cancelled
        except ProposalUnavailable:
            return unavailable if current() else cancelled
        except ValueError as error:
            if not current():
                return cancelled
            if isinstance(error, VoiceProposalError) and error.issue is ProposalIssue.TIME:
                return RoutineProposalResult("¿A qué hora exacta? Decime el horario y la acción de nuevo.",
                                             TaskOutcome(TaskStatus.NEEDS_INPUT))
            return RoutineProposalResult("Para proponer una rutina necesito una acción sobre un dispositivo seleccionado.",
                                         TaskOutcome(TaskStatus.UNSUPPORTED))
        if submission.decision is not None:
            decision = "confirmada" if submission.decision is ProposalDecision.CONFIRMED else "rechazada"
            return RoutineProposalResult(f"Esa propuesta ya fue {decision} en la web.",
                                         TaskOutcome(TaskStatus.UNSUPPORTED, persistence=PersistenceState.NOT_WRITTEN))
        return RoutineProposalResult("Dejé la propuesta para confirmar en la web.", TaskOutcome(
            TaskStatus.COMPLETED, persistence=PersistenceState.VOLATILE, effect=EffectState.OBSERVED,
            observation_scope="software_state",
        ))

    @property
    def scheduler(self):
        """The bound RoutineScheduler, or None while routines are unavailable."""
        return self._scheduler

    def bind_scheduler(self, scheduler) -> None:
        """Attach the application's scheduler once it is built.

        Args:
            scheduler: The RoutineScheduler, or None when the routine subsystem
                failed to start (reads then answer that routines are unavailable).

        Raises:
            ValueError: If a different scheduler is already bound; the router and
                the orchestrator must keep reading the same instance.
        """
        if self._scheduler is not None and scheduler is not self._scheduler:
            raise ValueError("routine_scheduler_already_bound")
        self._scheduler = scheduler

    async def control(self, operation: HouseholdOperation, name: str, *,
                      guard: AuthorityGuard | None) -> RoutineControlResult:
        """Claim or brake one exact saved routine using this capture's authority."""
        request_id = CURRENT_REQUEST_ID.get()
        if guard is None or not isinstance(request_id, str) or not request_id.strip():
            return self._no_control("", TaskStatus.CANCELLED)
        if operation is not HouseholdOperation.EXECUTE and not self._voice_brake:
            return self._no_control("El freno por voz está desactivado. Podés cambiar la rutina desde la web.",
                                    TaskStatus.UNSUPPORTED)
        scheduler = self._scheduler
        if scheduler is None:
            return self._no_control(_UNAVAILABLE, TaskStatus.FAILED)
        if not name:
            return self._no_control("Decime el nombre completo de la rutina.", TaskStatus.NEEDS_INPUT)
        try:
            if operation is HouseholdOperation.EXECUTE:
                routine = await scheduler.resolve_name(name, guard=guard)
                run = await scheduler.execute_by_id(
                    routine.routine_id, actor="household", request_id=request_id, source=HOUSEHOLD_VOICE,
                    expected_name=name, guard=guard,
                )
                return self._run_reply(run)
            brake = await scheduler.brake_by_name(name, RoutineBrake(operation.value), request_id=request_id, guard=guard)
        except RoutineNameConflict as error:
            return self._no_control(f"Hay {len(error.names)} rutinas con ese nombre: {_join(list(error.names))}.",
                                    TaskStatus.NEEDS_INPUT)
        except RoutineNotFound:
            return self._no_control(_NOT_FOUND, TaskStatus.NEEDS_INPUT)
        except PermissionError:
            return self._no_control("", TaskStatus.CANCELLED)
        except RoutineConflict as error:
            response = {
                "routine_disabled": "La rutina está desactivada. Podés revisarla en la web.",
                "routine_busy": "La rutina ya se está ejecutando.",
                "revision_conflict": "La rutina cambió. Podés revisarla en la web.",
                "voice_request_conflict": "Ese turno ya se registró para otra operación.",
            }.get(error.code, _UNAVAILABLE)
            return self._no_control(response, TaskStatus.FAILED)
        except Exception as error:
            logger.warning("Voice routine receipt unavailable: %s", type(error).__name__)
            return RoutineControlResult(
                "No pude confirmar el estado de la rutina. Podés consultarlo en la web.",
                TaskOutcome(TaskStatus.UNCERTAIN, persistence=PersistenceState.UNCERTAIN,
                            dispatch=DispatchState.UNCERTAIN, effect=EffectState.UNCERTAIN),
            )
        response = {
            BrakeStatus.STOP_REQUESTED: "Frené las acciones pendientes. Si había una acción en curso, puede terminar.",
            BrakeStatus.NOTHING_TO_STOP: "No quedaban acciones por frenar.",
            BrakeStatus.DISABLED: "Desactivé la rutina para revisión. Si había una acción en curso, puede terminar.",
            BrakeStatus.ALREADY_DISABLED: "La rutina ya estaba desactivada.",
        }[brake.status]
        return RoutineControlResult(
            response, TaskOutcome(TaskStatus.COMPLETED, persistence=PersistenceState.COMMITTED,
                                  effect=EffectState.OBSERVED, observation_scope="software_state"), brake.receipt_id,
        )

    @staticmethod
    def _no_control(response: str, status: TaskStatus) -> RoutineControlResult:
        return RoutineControlResult(response, TaskOutcome(
            status, persistence=PersistenceState.NOT_WRITTEN, dispatch=DispatchState.NOT_SENT,
            effect=EffectState.NOT_OBSERVED,
        ))

    @staticmethod
    def _run_reply(run: RoutineRun) -> RoutineControlResult:
        status, response, effect = {
            RunStatus.RUNNING: (TaskStatus.RUNNING, f"Inicié la rutina {run.snapshot.spec.name}.", EffectState.UNKNOWN),
            RunStatus.SUCCEEDED: (TaskStatus.COMPLETED, "Esa ejecución terminó bien.", EffectState.OBSERVED),
            RunStatus.FAILED: (TaskStatus.FAILED, "Esa ejecución falló.", EffectState.UNKNOWN),
            RunStatus.UNCERTAIN: (TaskStatus.UNCERTAIN, "No pude confirmar todas las acciones de esa ejecución.",
                                  EffectState.UNCERTAIN),
            RunStatus.SKIPPED: (TaskStatus.FAILED, "La rutina no se ejecutó porque no se cumplían sus condiciones.",
                                EffectState.NOT_OBSERVED),
            RunStatus.INTERRUPTED: (TaskStatus.CANCELLED, "Esa ejecución se interrumpió.", EffectState.UNKNOWN),
        }[run.status]
        if run.status is RunStatus.INTERRUPTED and any(action.status is ActionStatus.UNCERTAIN for action in run.actions):
            status, effect = TaskStatus.UNCERTAIN, EffectState.UNCERTAIN
            response = "Esa ejecución se interrumpió y no pude confirmar la acción en curso."
        return RoutineControlResult(
            response, TaskOutcome(status, persistence=PersistenceState.COMMITTED,
                                  effect=effect, observation_scope="ha_state" if effect is EffectState.OBSERVED else None),
            run.run_id,
        )

    @staticmethod
    def detect(text: str) -> RoutineReadQuery | None:
        """Recognize a routine question from its words alone, with no I/O.

        Args:
            text: The turn's text as heard.

        Returns:
            The read query, or None when the turn is not a routine question
            (including creation, execution, changes and stops, which other
            units handle).
        """
        words = _strip_leads(_words(text))
        noun = next((k for k, word in enumerate(words) if word in _SINGULAR_NOUNS | _PLURAL_NOUNS), None)
        if noun is None:
            return None
        before = words[:noun]
        wh_head = bool(words) and (words[0] in _WH_HEADS or (
            words[0] in {"a", "en", "de", "hasta"} and len(words) > 1 and words[1] in _WH_HEADS))
        for k, word in enumerate(before):
            # "¿está activa la rutina X?": after "está" the form is an adjective, not a command;
            # "¿se activa sola la rutina X?": after "se" a third-person form is a question (RV1).
            adjective = k > 0 and before[k - 1] in {"esta", "estan"}
            reflexive = k > 0 and before[k - 1] == "se" and word in _QUESTION_FORMS
            if (word in _BLOCKING_VERBS and not adjective and not reflexive
                    and not (wh_head and word in _QUESTION_FORMS)):
                return None
        asks = (wh_head or (bool(before) and before[0] in _YES_NO_HEADS)
                or any(word in _READ_VERBS for word in before)
                or (words[noun] in _PLURAL_NOUNS and before[-1:] == ["mis"])
                or (set(before) <= _SUBJECT_DETERMINERS and _states_after(words[noun + 1:])))
        if not asks:
            return None
        if any(word in _INDEFINITES for word in before[-2:]) or before[-1:] and before[-1] in _QUALIFIERS:
            # T033-U1-RV2: routines in general, not a saved one.
            if before[0] in _EXISTENCE_HEADS:
                return RoutineReadQuery(RoutineReadKind.LIST)
            return None
        if words[noun] in _PLURAL_NOUNS:
            return RoutineReadQuery(RoutineReadKind.LIST)
        names = _name_candidates(words[noun + 1:])
        if not names:
            return RoutineReadQuery(RoutineReadKind.LIST)
        return RoutineReadQuery(RoutineReadKind.CONSULT, names, cued=bool(_ROUTINE_CUES & set(words)))

    async def catalog(self) -> tuple[RoutineCatalogEntry, ...]:
        """Read every saved routine and its latest receipt, in the web catalog's order.

        Raises:
            RuntimeError: If no scheduler is bound.
        """
        scheduler = self._scheduler
        if scheduler is None:
            raise RuntimeError("routine_scheduler_unavailable")
        routines = await scheduler.get_all_routines()
        return tuple([await self._entry(scheduler, routine) for routine in routines])

    @staticmethod
    def claims(text: str) -> bool:
        """Whether ``answer`` will answer this turn whatever the store holds.

        True for a detected list, or a consult that asks about a saved
        routine's state, runs or schedule. The command gate's non-immediate
        exemption is limited to these turns, so a declined consult never gets
        further than it would without this service (T033-U1-RV2).
        """
        query = VoiceRoutineService.detect(text)
        return query is not None and query.always_answered

    @staticmethod
    async def _entry(scheduler, routine: ScheduledRoutine) -> RoutineCatalogEntry:
        latest = await scheduler.get_latest_run(routine.routine_id)
        return RoutineCatalogEntry(routine.routine_id, routine.spec.name, routine.spec.enabled, routine.revision,
                                   RunStatus(latest.status).value if latest is not None else None)

    async def answer(self, query: RoutineReadQuery) -> RoutineReadResult:
        """Answer a detected routine question from the scheduler store.

        Args:
            query: The result of ``detect``.

        Returns:
            The spoken reply and its outcome. Reads change nothing, so a
            completed read needs no persistence.
        """
        scheduler = self._scheduler
        if scheduler is None:
            logger.warning("[Routines] voice read refused: no routine scheduler bound")
            return RoutineReadResult(_UNAVAILABLE, TaskOutcome(TaskStatus.FAILED))
        try:
            if query.kind is RoutineReadKind.LIST:
                return await self._list(scheduler)
            return await self._consult(scheduler, query.names, query.cued)
        except Exception:
            # Store errors may carry private paths or names: keep only the class.
            logger.warning("[Routines] voice read failed")
            return RoutineReadResult(_UNAVAILABLE, TaskOutcome(TaskStatus.FAILED))

    async def _list(self, scheduler) -> RoutineReadResult:
        entries = tuple([await self._entry(scheduler, routine) for routine in await scheduler.get_all_routines()])
        logger.info("[Routines] voice list: %d routines", len(entries))
        if not entries:
            return RoutineReadResult("No hay rutinas guardadas. Podés crearlas desde la web.",
                                     TaskOutcome(TaskStatus.COMPLETED), entries)
        spoken = [entry.name if entry.enabled else f"{entry.name} (desactivada)"
                  for entry in entries[:_MAX_SPOKEN_NAMES]]
        rest = len(entries) - len(spoken)
        if rest:
            spoken.append(f"{rest} más")
        head = "Hay una rutina" if len(entries) == 1 else f"Hay {len(entries)} rutinas"
        return RoutineReadResult(f"{head}: {_join(spoken)}.", TaskOutcome(TaskStatus.COMPLETED), entries)

    async def _consult(self, scheduler, names: tuple[str, ...], cued: bool) -> RoutineReadResult:
        routines = await scheduler.get_all_routines()
        matches: list[ScheduledRoutine] = []
        for key in names:
            matches = [routine for routine in routines if name_key(routine.spec.name) == key]
            if matches:
                break
        logger.info("[Routines] voice consult: %d matches", len(matches))
        logger.debug("[Routines] voice consult names: %s", names)
        if not matches:
            return RoutineReadResult(_NOT_FOUND, TaskOutcome(TaskStatus.NEEDS_INPUT), declined=not cued)
        if len(matches) > 1:
            shown = [routine.spec.name for routine in matches[:_MAX_CANDIDATES]]
            return RoutineReadResult(
                f"Hay {len(matches)} rutinas con ese nombre: {_join(shown)}. ¿Cuál querés consultar?",
                TaskOutcome(TaskStatus.NEEDS_INPUT))
        routine = matches[0]
        entry = await self._entry(scheduler, routine)
        spec = routine.spec
        if not spec.enabled:
            state = f"La rutina {spec.name} está desactivada."
        elif spec.schedule is None:
            state = f"La rutina {spec.name} está activada y no tiene horario: se ejecuta solo cuando se pide."
        else:
            state = f"La rutina {spec.name} está activada y corre {_schedule_phrase(spec.schedule)}."
        status = RunStatus(entry.latest_run_status) if entry.latest_run_status is not None else None
        return RoutineReadResult(f"{state} {_LAST_RUN[status]}", TaskOutcome(TaskStatus.COMPLETED), (entry,))
