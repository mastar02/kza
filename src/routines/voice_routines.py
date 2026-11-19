"""Read-only voice access to the persistent routines (T033-U1).

``VoiceRoutineService`` answers "¿qué rutinas hay?" and "¿cómo salió la
rutina de la mañana?" from the RoutineScheduler's store (``routines.db``), the
same definitions and receipts the web catalog shows, never from the legacy
Chroma routine index. It never creates, changes, enables, disables or runs a
routine: execution by voice is T033-U2 and proposals are T033-U3.

Detection is strictly lexical and pure: ``detect`` reads only the words of the
turn, so a turn that is not a routine question never reaches the store, the
7B or the cloud. A routine noun ("rutina", "automatización") is required at a
word boundary ("rutinario" is not one) and a bare "cuando" never makes a turn a
routine (CORE-04).

The scheduler is bound late: ``main`` builds the scheduler after the
orchestrator and the RequestRouter, so both receive this service at
construction and ``bind_scheduler`` attaches the application's single
scheduler once it exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import logging
import re
import unicodedata

from src.core.task_outcome import TaskOutcome, TaskStatus
from src.nlu.command_grammar import WAKE_TERMS
from src.routines.models import RunStatus, ScheduledRoutine, TimeSchedule

logger = logging.getLogger(__name__)

__all__ = [
    "RoutineCatalogEntry", "RoutineReadKind", "RoutineReadQuery", "RoutineReadResult", "VoiceRoutineService",
]

_SINGULAR_NOUNS = frozenset({"rutina", "automatizacion"})
_PLURAL_NOUNS = frozenset({"rutinas", "automatizaciones"})
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
# Past-tense status verbs after the noun that make "la rutina X anduvo/falló" a question.
_STATUS_AFTER = frozenset({"salio", "anduvo", "funciono", "corrio", "termino", "fallo", "ejecuto"})
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
    """

    kind: RoutineReadKind
    names: tuple[str, ...] = ()


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
    """

    response: str
    outcome: TaskOutcome
    entries: tuple[RoutineCatalogEntry, ...] = ()


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", _fold(text)).split()


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


def _name_candidates(complement: list[str]) -> tuple[str, ...]:
    if not complement or complement[0] in _DESCRIPTION_STARTS or all(w in _NAME_TAIL_CUES for w in complement):
        return ()
    candidates = [name_key(" ".join(complement))]
    cut = next((k for k, word in enumerate(complement) if word in _NAME_TAIL_CUES), None)
    if cut:
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
    """Read-only voice adapter over the RoutineScheduler (T033-U1).

    Args:
        scheduler: The application's RoutineScheduler, or None until
            ``bind_scheduler`` attaches it (late binding in main).
    """

    def __init__(self, scheduler=None) -> None:
        self._scheduler = scheduler

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
        for word in before:
            if word in _BLOCKING_VERBS and not (wh_head and word in _QUESTION_FORMS):
                return None
        asks = (wh_head or (bool(before) and before[0] in _YES_NO_HEADS)
                or any(word in _READ_VERBS for word in before)
                or (words[noun] in _PLURAL_NOUNS and before[-1:] == ["mis"])
                or (set(before) <= _SUBJECT_DETERMINERS and bool(_STATUS_AFTER & set(words[noun + 1:]))))
        if not asks:
            return None
        if words[noun] in _PLURAL_NOUNS:
            return RoutineReadQuery(RoutineReadKind.LIST)
        names = _name_candidates(words[noun + 1:])
        if not names:
            return RoutineReadQuery(RoutineReadKind.LIST)
        return RoutineReadQuery(RoutineReadKind.CONSULT, names)

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
            return await self._consult(scheduler, query.names)
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

    async def _consult(self, scheduler, names: tuple[str, ...]) -> RoutineReadResult:
        routines = await scheduler.get_all_routines()
        matches: list[ScheduledRoutine] = []
        for key in names:
            matches = [routine for routine in routines if name_key(routine.spec.name) == key]
            if matches:
                break
        logger.info("[Routines] voice consult: %d matches", len(matches))
        logger.debug("[Routines] voice consult names: %s", names)
        if not matches:
            return RoutineReadResult(_NOT_FOUND, TaskOutcome(TaskStatus.NEEDS_INPUT))
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
