"""Timer commands claimed by the household precedence layer (T030-U1).

The orchestrated dispatcher sends every P3-F2 TIMER turn here: create (R2 and
an "alarma" with a duration, R3a), cancel/stop/disable, pause, resume and
query (R1), and the R3a clarification ("poné una alarma" with no time).

Rules:

* The actor is always the household (``HomeActor.household().subject``), never
  an acoustic speaker label; the ring zone is the origin room's zone chosen by
  the caller (the microphone that heard the turn), never a detected source zone.
* Nothing is confirmed unless the timer manager is running: after a
  ``FeatureManager.start`` failure the reply is a FAILED refusal with no timer.
* A command picks exactly one timer or none: an unknown or ambiguous name is
  NEEDS_INPUT with zero effects; a duplicate label is NEEDS_INPUT.
* Timers are volatile until SCHEMA-V2 (T030-U3): a completed change reports
  ``PersistenceState.VOLATILE``, never COMMITTED.
* Labels are household content and are never logged above DEBUG.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass

from src.core.task_outcome import DispatchState, EffectState, PersistenceState, TaskOutcome, TaskStatus
from src.home_assistant.control_policy import HomeActor
from src.nlu.household_intents import HouseholdOperation
from src.timers.named_timers import NamedTimer, NamedTimerManager, TimerState
from src.timers.voice_request import (
    TimerRequestIssue,
    TimerSelector,
    fold_label,
    format_duration,
    parse_timer_request,
    parse_timer_selector,
)

logger = logging.getLogger(__name__)

# Spoken replies (Spanish, rioplatense).
TIMERS_UNSUPPORTED = "Los temporizadores no están disponibles."
TIMERS_UNAVAILABLE = "Ahora no puedo manejar temporizadores. Probá de nuevo en un rato."
NO_ACTIVE = "No hay temporizadores activos."
ALARM_CLARIFY = "¿Para cuándo es la alarma? Decímelo completo, por ejemplo: poné una alarma de 10 minutos."
_ASK_DURATION = "¿De cuánto tiempo es el temporizador? Decímelo completo, por ejemplo: poné un temporizador de 10 minutos."
_ISSUE_REPLIES = {
    TimerRequestIssue.MISSING_DURATION: _ASK_DURATION,
    TimerRequestIssue.VAGUE_DURATION: _ASK_DURATION,
    TimerRequestIssue.UNPARSEABLE: "No entendí la duración. " + _ASK_DURATION,
    TimerRequestIssue.CLOCK_TIME: ("Un temporizador va con una duración, por ejemplo de 10 minutos. "
                                   "Para una hora fija pedime un recordatorio."),
    TimerRequestIssue.TOO_LONG: "Puedo poner temporizadores de hasta 24 horas.",
    TimerRequestIssue.ROOM_TARGET: ("Por ahora el temporizador suena donde lo pedís. "
                                    "Pedímelo de nuevo sin nombrar la habitación."),
}
_ACTIVE = (TimerState.RUNNING, TimerState.PAUSED)
_EVERY_RE = re.compile(r"\b(?:todos|todas|timers|temporizadores)\b")


@dataclass(frozen=True)
class TimerVoiceReply:
    """Spoken reply and task evidence of one timer command.

    Attributes:
        response: Spanish text to speak.
        outcome: Task evidence (never COMPLETED without an observed change or answer).
    """

    response: str
    outcome: TaskOutcome


def describe(timer: NamedTimer) -> str:
    """How a timer is named aloud: by its label, else by its duration ("de pasta", "de 5 minutos")."""
    return f"de {timer.name}" if timer.name else f"de {format_duration(timer.duration_seconds)}"


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def _refusal(response: str, status: TaskStatus) -> TimerVoiceReply:
    return TimerVoiceReply(response, TaskOutcome(status, persistence=PersistenceState.NOT_WRITTEN,
                                                 dispatch=DispatchState.NOT_SENT, effect=EffectState.NOT_OBSERVED))


def _changed(response: str) -> TimerVoiceReply:
    return TimerVoiceReply(response, TaskOutcome(TaskStatus.COMPLETED, persistence=PersistenceState.VOLATILE,
                                                 effect=EffectState.OBSERVED, observation_scope="software_state"))


def _listing(timers: list[NamedTimer]) -> str:
    def one(timer: NamedTimer) -> str:
        state = ", en pausa" if timer.state is TimerState.PAUSED else ""
        return f"{describe(timer)}, faltan {format_duration(timer.remaining_seconds)}{state}"
    if len(timers) == 1:
        return f"Al temporizador {one(timers[0])}."
    return f"Hay {len(timers)} temporizadores: " + "; ".join(one(t) for t in timers) + "."


class TimerVoiceCommands:
    """Run claimed timer turns against the application's one timer manager.

    Args:
        manager: The feature manager's ``NamedTimerManager`` (the same instance
            the FeatureManager starts and stops), or None when not configured.
    """

    def __init__(self, manager: NamedTimerManager | None) -> None:
        self.manager = manager

    async def execute(self, operation: HouseholdOperation, complement: str, *, text: str,
                      zone_id: str) -> TimerVoiceReply:
        """Run one classified timer command.

        Args:
            operation: The P3-F2 operation (CREATE, CANCEL, STOP, DISABLE, PAUSE,
                RESUME, QUERY or CLARIFY).
            complement: Words after the timer/alarm noun.
            text: The whole dispatch text ("cancelá todos los timers").
            zone_id: Ring zone: the origin room's ``zone_<room>``, or "default"
                (the configured default output) when the turn has no origin.

        Returns:
            The reply to speak and its task evidence.
        """
        if self.manager is None:
            return _refusal(TIMERS_UNSUPPORTED, TaskStatus.UNSUPPORTED)
        if not self.manager.running:
            logger.warning("[Timer] manager not running; %s refused", operation.value)
            return _refusal(TIMERS_UNAVAILABLE, TaskStatus.FAILED)
        if operation is HouseholdOperation.CLARIFY:
            return _refusal(ALARM_CLARIFY, TaskStatus.NEEDS_INPUT)
        if operation is HouseholdOperation.CREATE:
            return self._create(complement, zone_id)
        selector = parse_timer_selector(complement)
        every = _EVERY_RE.search(_fold(text)) is not None
        if operation is HouseholdOperation.QUERY:
            return self._query(selector)
        if operation in (HouseholdOperation.CANCEL, HouseholdOperation.STOP, HouseholdOperation.DISABLE):
            return self._transition(selector, every, _ACTIVE, self.manager.cancel_timer, "cancelo",
                                    lambda t: f"Cancelé el temporizador {describe(t)}.",
                                    lambda n: f"Cancelé {n} temporizadores.")
        if operation is HouseholdOperation.PAUSE:
            return self._transition(selector, every, (TimerState.RUNNING,), self.manager.pause_timer, "pauso",
                                    lambda t: (f"Pausé el temporizador {describe(t)}; "
                                               f"le quedan {format_duration(t.remaining_seconds)}."),
                                    lambda n: f"Pausé {n} temporizadores.")
        if operation is HouseholdOperation.RESUME:
            return self._transition(selector, every, (TimerState.PAUSED,), self.manager.resume_timer, "sigo",
                                    lambda t: (f"Sigue el temporizador {describe(t)}; "
                                               f"faltan {format_duration(t.remaining_seconds)}."),
                                    lambda n: f"Siguen {n} temporizadores.")
        return _refusal("Con los temporizadores puedo ponerlos, pausarlos, seguirlos, consultarlos o cancelarlos.",
                        TaskStatus.NEEDS_INPUT)

    # ------------------------------------------------------------------

    def _active(self, states=_ACTIVE) -> list[NamedTimer]:
        return sorted((t for t in self.manager.get_all_timers() if t.state in states), key=lambda t: t.created_at)

    def _create(self, complement: str, zone_id: str) -> TimerVoiceReply:
        request = parse_timer_request(complement)
        if request.issue is not None:
            logger.info("[Timer] create needs input: %s", request.issue.value)
            return _refusal(_ISSUE_REPLIES[request.issue], TaskStatus.NEEDS_INPUT)
        active = self._active()
        if len(active) >= self.manager.max_timers:
            return _refusal(f"Ya hay {len(active)} temporizadores activos. Cancelá alguno primero.",
                            TaskStatus.NEEDS_INPUT)
        label = request.label or ""
        if label and any(fold_label(t.name) == fold_label(label) for t in active):
            logger.info("[Timer] create needs input: duplicate_label")
            return _refusal(f"Ya hay un temporizador de {label}. Cancelalo primero o usá otro nombre.",
                            TaskStatus.NEEDS_INPUT)
        timer = self.manager.create_timer(label, request.duration_s, user_id=HomeActor.household().subject,
                                          zone_id=zone_id)
        logger.info("[Timer] created id=%s duration_s=%d zone=%s labelled=%s", timer.timer_id,
                    timer.duration_seconds, zone_id, bool(label))
        if label:
            return _changed(f"Temporizador de {label} en marcha: {format_duration(timer.duration_seconds)}.")
        return _changed(f"Temporizador de {format_duration(timer.duration_seconds)} en marcha.")

    def _select(self, selector: TimerSelector, candidates: list[NamedTimer]) -> list[NamedTimer]:
        """Timers the selector names: exact label first, else whole-word label match; duration must agree."""
        if selector.duration_s is not None:
            candidates = [t for t in candidates if t.duration_seconds == selector.duration_s]
        if selector.label is None:
            return candidates
        wanted = fold_label(selector.label)
        exact = [t for t in candidates if t.name and fold_label(t.name) == wanted]
        if exact:
            return exact
        words = set(wanted.split())
        return [t for t in candidates if t.name and words <= set(fold_label(t.name).split())]

    def _not_found(self, selector: TimerSelector) -> TimerVoiceReply:
        if selector.label is not None:
            named = f"de {selector.label}"
        else:
            named = f"de {format_duration(selector.duration_s)}"
        return _refusal(f"No encontré un temporizador {named}.", TaskStatus.NEEDS_INPUT)

    def _query(self, selector: TimerSelector) -> TimerVoiceReply:
        active = self._active()
        if selector.empty:
            if not active:
                return TimerVoiceReply(NO_ACTIVE, TaskOutcome(TaskStatus.COMPLETED))
            return TimerVoiceReply(_listing(active), TaskOutcome(TaskStatus.COMPLETED))
        chosen = self._select(selector, active)
        if not chosen:
            return self._not_found(selector)
        return TimerVoiceReply(_listing(chosen), TaskOutcome(TaskStatus.COMPLETED))

    def _transition(self, selector: TimerSelector, every: bool, states, apply, verb: str, one, many) -> TimerVoiceReply:
        candidates = self._active(states)
        if not candidates:
            if self._active():
                return _refusal("No hay temporizadores " + ("corriendo." if states == (TimerState.RUNNING,)
                                                            else "en pausa."), TaskStatus.NEEDS_INPUT)
            return _refusal(NO_ACTIVE, TaskStatus.NEEDS_INPUT)
        chosen = self._select(selector, candidates)
        if not chosen:
            return self._not_found(selector)
        if len(chosen) > 1 and not every:
            names = ", ".join(describe(t) for t in chosen[:-1]) + " y " + describe(chosen[-1])
            return _refusal(f"Hay {len(chosen)} temporizadores: {names}. ¿Cuál {verb}? Decímelo completo.",
                            TaskStatus.NEEDS_INPUT)
        done = [t for t in chosen if apply(t.timer_id)]
        logger.info("[Timer] %s ids=%s", apply.__name__, [t.timer_id for t in done])  # English op name, no label
        if not done:
            return _refusal(NO_ACTIVE, TaskStatus.NEEDS_INPUT)
        return _changed(one(done[0]) if len(done) == 1 else many(len(done)))
