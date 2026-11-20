"""Sequential execution of a compound voice turn (T037-U6, user answer Q21).

A compound turn ("apagá la luz y pausá la música") is split by the household
classifier (R5). Each part then runs now, in spoken order, through exactly the
gates and handler it would meet alone; this module owns only what is common to
every part: its request id, the order, which later parts are not attempted, the
single reply and the aggregate outcome (design ``compound_contract``).

* Request id: ``<turn request id>:part<n>`` (1-based). Home actions use it as
  their idempotency key, and the runner remembers the parts it ran under each
  id, so a retried turn never sends a part again.
* Independence: a failed, refused or uncertain part never stops independent
  later parts. A later part on the same target as an earlier part that did not
  complete is not attempted; a later part whose action repeats or undoes an
  earlier part's action on the same entity is not sent either (the classifier only sees
  rooms and domains, the entity is known once the part's own gates resolve it).
* Barge-in: a new turn in the same room between parts stops the remaining
  parts (not attempted). Nothing is resent.
* Reply: one reply naming every part as done, not done (with the reason),
  uncertain, or not attempted; never a generic "Listo".
* Aggregate: COMPLETED only if every part completed; else UNCERTAIN if any part
  is uncertain; else CANCELLED if a barge-in stopped it; else FAILED. The
  dispatch is PARTIAL when some parts were sent and others not. Per-part
  outcomes go in ``TaskOutcome.parts``.

No user text is logged.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256

from src.core.task_outcome import (
    CURRENT_REQUEST_ID,
    DispatchState,
    EffectState,
    PersistenceState,
    ResponseRequirement,
    TaskOutcome,
    TaskStatus,
)
from src.nlu.household_intents import HouseholdFamily

logger = logging.getLogger(__name__)

# Parts remembered by request id for retried turns (same bound as HomeActionService receipts).
_LEDGER_SIZE = 256
_MUSIC_PATHS = frozenset({"fast_music", "slow_music"})
# Services that undo each other on one entity.
_OPPOSITE_SERVICES = (
    frozenset({"turn_on", "turn_off"}),
    frozenset({"open_cover", "close_cover"}),
    frozenset({"media_play", "media_pause"}),
    frozenset({"media_play", "media_stop"}),
)


class PartVerdict(StrEnum):
    """How the reply reports one part."""

    DONE = "done"
    NOT_DONE = "not_done"
    UNCERTAIN = "uncertain"
    INTERRUPTED = "interrupted"   # not attempted: a barge-in stopped the turn before it
    SAME_TARGET = "same_target"   # not attempted: an earlier part on the same target did not complete
    CONFLICT = "conflict"         # not attempted: it undoes an earlier part on the same entity
    DUPLICATE = "duplicate"       # not sent: an earlier part already sent the same action to the same entity


def _data_key(data: Mapping | None) -> tuple:
    return tuple(sorted((str(k), repr(v)) for k, v in (data or {}).items()))


@dataclass(frozen=True)
class _Target:
    """What an earlier part did to one target."""

    status: TaskStatus
    service: str | None = None
    data: tuple = ()


@dataclass
class _RunState:
    """Targets touched so far in one compound turn."""

    targets: dict[str, list[_Target]] = field(default_factory=dict)

    def check(self, target: str, service: str | None = None, data: Mapping | None = None) -> PartVerdict | None:
        """The verdict that stops a part on ``target``, or None when it may run."""
        earlier = self.targets.get(target, ())
        if any(t.status is not TaskStatus.COMPLETED for t in earlier):
            return PartVerdict.SAME_TARGET
        if service is None:
            return None
        key = _data_key(data)
        if any(t.service == service and t.data == key for t in earlier):
            return PartVerdict.DUPLICATE
        if any(t.service is not None and frozenset({t.service, service}) in _OPPOSITE_SERVICES for t in earlier):
            return PartVerdict.CONFLICT
        return None

    def record(self, target: str | None, status: TaskStatus, service: str | None = None,
               data: Mapping | None = None) -> None:
        if target:
            self.targets.setdefault(target, []).append(_Target(status, service, _data_key(data)))


@dataclass
class CompoundPartContext:
    """What one part's handler gets: its index, request id, part and the turn's targets.

    A handler that resolves a home entity calls :meth:`check` right before
    sending; when it returns a verdict the handler sends nothing and returns
    :meth:`skip`'s outcome instead.

    Attributes:
        index: 1-based spoken position.
        request_id: The part's own request id (also in ``CURRENT_REQUEST_ID``
            while the part runs).
        part: The classifier's ``HouseholdPart`` (segment and claimed intent).
    """

    index: int
    request_id: str
    part: object
    _state: _RunState = field(repr=False, default_factory=_RunState)
    skipped: PartVerdict | None = None

    def check(self, entity_id: str, service: str | None, data: Mapping | None = None) -> PartVerdict | None:
        """Whether an earlier part stops this action on ``entity_id`` (None: send it)."""
        return self._state.check(entity_id, service, data)

    def skip(self, verdict: PartVerdict) -> TaskOutcome:
        """Record that this part sends nothing, and return its outcome."""
        self.skipped = verdict
        return _skipped_outcome(verdict)


def _skipped_outcome(verdict: PartVerdict) -> TaskOutcome:
    # A repeated action is not sent again either: the part itself did nothing, so it is never
    # reported as completed (the earlier part may have resolved another room to the same device).
    return TaskOutcome(TaskStatus.CANCELLED, dispatch=DispatchState.NOT_SENT, effect=EffectState.NOT_OBSERVED)


@dataclass(frozen=True)
class CompoundPartReport:
    """One part's result, as the reply and the aggregate see it.

    Attributes:
        index: 1-based spoken position.
        request_id: The part's request id.
        segment: The part's words as heard, used for replay and ordinary replies.
        verdict: How the reply reports it.
        outcome: The part's own TaskOutcome.
        response: The part handler's own reply ("" when silent or none).
        reason: A content-free code (receipt reason or refusal intent), or None.
        path: The part's PathType value, or None.
        retried: The report comes from an earlier run of the same turn.
        intercom_receipt: Per-zone playback evidence for an intercom part.
        reply_label: Content-free spoken name for a part with a literal message.
        intercom_fingerprint: Digest of the original complement, preserving
            literal delimiters for idempotency checks.
    """

    index: int
    request_id: str
    segment: str
    verdict: PartVerdict
    outcome: TaskOutcome
    response: str = ""
    reason: str | None = None
    path: str | None = None
    retried: bool = False
    intercom_receipt: dict | None = field(default=None, repr=False)
    reply_label: str | None = None
    intercom_fingerprint: bytes | None = field(default=None, repr=False)

    def summary(self) -> dict:
        """Content-free description for turn results and logs."""
        summary = {"index": self.index, "request_id": self.request_id, "verdict": self.verdict.value,
                "task_status": self.outcome.status.value, "dispatch": self.outcome.dispatch.value,
                "effect": self.outcome.effect.value, "path": self.path, "reason": self.reason,
                "retried": self.retried}
        if self.intercom_receipt is not None:
            summary["intercom_receipt"] = deepcopy(self.intercom_receipt)
        return summary


@dataclass(frozen=True)
class CompoundRun:
    """A finished compound turn: its parts, reply and aggregate outcome."""

    reports: tuple[CompoundPartReport, ...]
    interrupted: bool
    outcome: TaskOutcome
    response: str


def part_outcome(result) -> TaskOutcome:
    """The TaskOutcome of one part's dispatch result.

    A result without an outcome is read conservatively: a music command the
    music service accepted is COMPLETED (its effect is not observed by the
    music path, so it is not required); a failed music command is UNCERTAIN
    with an unknown send, because the music service reports a refusal and a
    call that failed after reaching the player the same way (T037-U6-RV3:
    the reply says "no sé si se hizo", never "no lo hice"); any other result
    without an outcome returned before sending (domain conflict, unknown
    speaker, invalid target) and is FAILED, not sent.
    """
    outcome = getattr(result, "outcome", None)
    if isinstance(outcome, TaskOutcome):
        return outcome
    path = getattr(getattr(result, "path", None), "value", getattr(result, "path", None))
    if path in _MUSIC_PATHS:
        if getattr(result, "success", False) is True:
            return TaskOutcome(TaskStatus.COMPLETED, dispatch=DispatchState.ACCEPTED)
        return TaskOutcome(TaskStatus.UNCERTAIN, dispatch=DispatchState.UNKNOWN, effect=EffectState.UNKNOWN)
    return TaskOutcome(TaskStatus.FAILED, dispatch=DispatchState.NOT_SENT, effect=EffectState.NOT_OBSERVED)


def _verdict(outcome: TaskOutcome) -> PartVerdict:
    if outcome.status is TaskStatus.COMPLETED:
        return PartVerdict.DONE
    if outcome.status in (TaskStatus.UNCERTAIN, TaskStatus.RUNNING, TaskStatus.UNKNOWN, TaskStatus.QUEUED,
                          TaskStatus.RECEIVED):
        return PartVerdict.UNCERTAIN
    return PartVerdict.NOT_DONE


def aggregate_outcome(reports: Iterable[CompoundPartReport], interrupted: bool) -> TaskOutcome:
    """The turn's TaskOutcome from its parts (design ``compound_contract.aggregate_status``)."""
    parts = tuple(report.outcome for report in reports)
    statuses = {part.status for part in parts}
    if statuses == {TaskStatus.COMPLETED}:
        status = TaskStatus.COMPLETED
    elif TaskStatus.UNCERTAIN in statuses or any(_verdict(p) is PartVerdict.UNCERTAIN for p in parts):
        status = TaskStatus.UNCERTAIN
    elif interrupted:
        status = TaskStatus.CANCELLED
    else:
        status = TaskStatus.FAILED

    dispatches = [p.dispatch for p in parts if p.dispatch is not DispatchState.NOT_REQUIRED]
    maybe_sent = {DispatchState.ACCEPTED, DispatchState.PARTIAL, DispatchState.UNCERTAIN, DispatchState.UNKNOWN}
    if not dispatches:
        dispatch = DispatchState.NOT_REQUIRED
    elif all(d is DispatchState.ACCEPTED for d in dispatches):
        dispatch = DispatchState.ACCEPTED
    elif DispatchState.PARTIAL in dispatches or (any(d in maybe_sent for d in dispatches)
                                                 and any(d not in maybe_sent for d in dispatches)):
        dispatch = DispatchState.PARTIAL
    elif any(d in (DispatchState.UNCERTAIN, DispatchState.UNKNOWN) for d in dispatches):
        dispatch = DispatchState.UNCERTAIN
    elif all(d is DispatchState.NOT_SENT for d in dispatches):
        dispatch = DispatchState.NOT_SENT
    else:
        dispatch = DispatchState.REJECTED

    effects = [p.effect for p in parts if p.effect is not EffectState.NOT_REQUIRED]
    if not effects:
        effect = EffectState.NOT_REQUIRED
    elif all(e is EffectState.OBSERVED for e in effects):
        effect = EffectState.OBSERVED
    elif any(e in (EffectState.OBSERVED, EffectState.PARTIAL) for e in effects):
        effect = EffectState.PARTIAL
    elif any(e in (EffectState.UNCERTAIN, EffectState.UNKNOWN) for e in effects):
        effect = EffectState.UNCERTAIN
    elif EffectState.UNCONFIRMED in effects:
        effect = EffectState.UNCONFIRMED
    else:
        effect = EffectState.NOT_OBSERVED
    scope = None
    if effect in (EffectState.OBSERVED, EffectState.PARTIAL):
        scopes = {p.observation_scope for p in parts if p.effect in (EffectState.OBSERVED, EffectState.PARTIAL)}
        # Mixed HA and KZA observations: the aggregate is KZA's own record of each part's evidence.
        scope = "ha_state" if scopes == {"ha_state"} else "software_state"

    stored = {p.persistence for p in parts if p.persistence is not PersistenceState.NOT_REQUIRED}
    persistence = PersistenceState.NOT_REQUIRED
    for candidate in (PersistenceState.UNCERTAIN, PersistenceState.FAILED, PersistenceState.UNKNOWN,
                      PersistenceState.VOLATILE, PersistenceState.COMMITTED, PersistenceState.NOT_WRITTEN):
        if candidate in stored:
            persistence = candidate
            break
    return TaskOutcome(status, persistence=persistence, dispatch=dispatch, effect=effect,
                       observation_scope=scope, parts=parts)


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text if text[-1] in ".?!" else text + "."


_REASONS = {
    "hook_blocked": "Una regla de la casa lo bloqueó.",
    "entity_unavailable": "El dispositivo no está disponible.",
}
_STATUS_REASONS = {
    TaskStatus.NEEDS_INPUT: "Me faltan datos para hacerlo.",
    TaskStatus.UNSUPPORTED: "Eso no lo puedo hacer por voz.",
}


def compose_reply(reports: Iterable[CompoundPartReport]) -> str:
    """One reply naming every part in spoken order (design ``compound_contract.reply``)."""
    sentences = []
    for report in reports:
        head = report.reply_label or (report.segment[:1].upper() + report.segment[1:])
        own = _sentence(report.response)
        if report.verdict is PartVerdict.DONE:
            sentences.append(f"{head}: {own}" if own else f"{head}: hecho.")
        elif report.verdict is PartVerdict.NOT_DONE:
            reason = own or _REASONS.get(report.reason or "") or _STATUS_REASONS.get(
                report.outcome.status, "No pude hacerlo.")
            sentences.append(f"{head}: no lo hice. {reason}")
        elif report.verdict is PartVerdict.UNCERTAIN:
            detail = own if report.intercom_receipt is not None else ""
            sentences.append(f"{head}: {detail or 'no sé si se hizo; fijate antes de repetirlo.'}")
        elif report.verdict is PartVerdict.INTERRUPTED:
            sentences.append(f"{head}: no lo intenté porque me interrumpiste.")
        elif report.verdict is PartVerdict.SAME_TARGET:
            sentences.append(f"{head}: no lo intenté porque lo anterior sobre lo mismo no se completó.")
        elif report.verdict is PartVerdict.CONFLICT:
            sentences.append(f"{head}: no lo hice porque contradice lo que pediste antes.")
        else:
            sentences.append(f"{head}: no lo repetí porque era lo mismo que lo anterior.")
    return " ".join(sentences)


RunPart = Callable[[CompoundPartContext], Awaitable[object]]


def compound_part_request_id(request_id: str, index: int) -> str:
    """Return the identity checked by the runner and the part's effect owner."""
    return f"{request_id}:part{index}"


class CompoundRunner:
    """Run the parts of compound turns; one instance per caller (router or dispatcher).

    The instance keeps a bounded, process-local memory of the parts it ran by
    request id, so a retried turn (same request id) reports them instead of
    running them again. Like home action receipts, a forgotten id is not
    evidence that a part never ran.
    """

    def __init__(self, ledger_size: int = _LEDGER_SIZE) -> None:
        self._ledger: OrderedDict[str, CompoundPartReport] = OrderedDict()
        self._ledger_size = ledger_size

    async def run(self, parts, run_part: RunPart, *, interrupted: Callable[[], bool] = lambda: False,
                  part_target: Callable[[object], str | None] = lambda part: None) -> CompoundRun:
        """Run ``parts`` in order and compose the turn's reply and outcome.

        Args:
            parts: The classifier's ``HouseholdPart`` tuple, in spoken order.
            run_part: Runs one part through its own gates and handler and
                returns its dispatch result (a ``DispatchResult``).
            interrupted: True once a barge-in stopped the turn; checked before
                every part.
            part_target: The target key of a part whose target is known
                before it runs (a household family, the music player), or None.

        Returns:
            The finished turn.
        """
        base = CURRENT_REQUEST_ID.get() or "compound:" + uuid.uuid4().hex
        state = _RunState()
        reports: list[CompoundPartReport] = []
        stopped = False
        for index, part in enumerate(parts, start=1):
            request_id = compound_part_request_id(base, index)
            segment = getattr(part, "segment", "")
            intent = getattr(part, "intent", None)
            reply_label = "El anuncio" if getattr(intent, "family", None) is HouseholdFamily.INTERCOM else None
            if stopped or interrupted():
                stopped = True
                reports.append(CompoundPartReport(index, request_id, segment, PartVerdict.INTERRUPTED,
                                                  _skipped_outcome(PartVerdict.INTERRUPTED), reply_label=reply_label))
                continue
            earlier = self._ledger.get(request_id)
            fingerprint = sha256(intent.complement.encode("utf-8")).digest() if reply_label is not None else None
            if earlier is not None:
                if earlier.segment == segment and earlier.intercom_fingerprint == fingerprint:
                    report = replace(earlier, retried=True)
                else:
                    report = CompoundPartReport(index, request_id, segment, PartVerdict.NOT_DONE,
                                                TaskOutcome(TaskStatus.FAILED, dispatch=DispatchState.NOT_SENT,
                                                            effect=EffectState.NOT_OBSERVED),
                                                reason="request_id_conflict", reply_label=reply_label)
                reports.append(report)
                state.record(part_target(part), report.outcome.status)
                continue
            target = part_target(part)
            blocked = state.check(target) if target else None
            if blocked is not None:
                reports.append(CompoundPartReport(index, request_id, segment, blocked, _skipped_outcome(blocked),
                                                  reply_label=reply_label))
                continue
            context = CompoundPartContext(index, request_id, part, state)
            token = CURRENT_REQUEST_ID.set(request_id)
            try:
                result = await run_part(context)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # The part may have been sent: uncertain, never retried.
                logger.warning("[Compound] part %d failed in its handler: %s", index, type(error).__name__)
                result = None
            finally:
                CURRENT_REQUEST_ID.reset(token)
            report = self._report(context, result, segment)
            remember = True
            if reply_label is not None:
                in_flight = report.intercom_receipt is not None and report.intercom_receipt.get("in_flight") is True
                report = replace(report, reply_label=reply_label, intercom_fingerprint=fingerprint, retried=in_flight)
                # Intercom owns live progress and refuses reuse of an active ID
                # with a different payload. Neither reply can replace its result.
                conflict = (report.outcome.status is TaskStatus.FAILED
                            and report.outcome.persistence is PersistenceState.NOT_WRITTEN)
                remember = not (in_flight or conflict)
            reports.append(report)
            if remember:
                self._remember(report)
            if context.skipped is not None:
                continue  # sent nothing: the target is as the earlier parts left it
            action = getattr(result, "action", None)
            if isinstance(action, Mapping) and isinstance(action.get("entity_id"), str):
                state.record(action["entity_id"], report.outcome.status, action.get("service"), action.get("data"))
            else:
                state.record(target, report.outcome.status)
        outcome = aggregate_outcome(reports, stopped)
        logger.info("[Compound] parts=%s status=%s dispatch=%s", [r.verdict.value for r in reports],
                    outcome.status.value, outcome.dispatch.value)
        return CompoundRun(tuple(reports), stopped, outcome, compose_reply(reports))

    @staticmethod
    def _report(context: CompoundPartContext, result, segment: str) -> CompoundPartReport:
        if result is None:
            outcome = TaskOutcome(TaskStatus.UNCERTAIN, dispatch=DispatchState.UNKNOWN, effect=EffectState.UNKNOWN)
            return CompoundPartReport(context.index, context.request_id, segment, PartVerdict.UNCERTAIN, outcome,
                                      reason="part_exception")
        path = getattr(getattr(result, "path", None), "value", None)
        if context.skipped is not None:
            return CompoundPartReport(context.index, context.request_id, segment, context.skipped,
                                      _skipped_outcome(context.skipped), path=path)
        outcome = part_outcome(result)
        silent = getattr(result, "response_requirement", None) is ResponseRequirement.SILENT
        response = "" if silent else (getattr(result, "response", "") or "")
        receipt = getattr(result, "receipt", None)
        reason = getattr(receipt, "reason", None) or getattr(result, "intent", None)
        intercom_receipt = getattr(result, "intercom_receipt", None)
        return CompoundPartReport(context.index, context.request_id, segment, _verdict(outcome), outcome,
                                  response if isinstance(response, str) else "",
                                  reason if isinstance(reason, str) else None, path,
                                  intercom_receipt=deepcopy(intercom_receipt) if isinstance(intercom_receipt, dict) else None)

    def _remember(self, report: CompoundPartReport) -> None:
        self._ledger[report.request_id] = report
        self._ledger.move_to_end(report.request_id)
        while len(self._ledger) > self._ledger_size:
            self._ledger.popitem(last=False)
