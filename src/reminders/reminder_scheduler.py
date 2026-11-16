"""Asyncio scheduling and delivery engine for reminders."""
import asyncio
import hashlib
import inspect
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, StrEnum
from typing import Any

from src.core.sqlite_work import AuthorityGuard
from src.core.voice_delivery import DeliveryState, ResponseResult, ResponseStage
from src.reminders.recurrence import next_trigger
from src.reminders.reminder_store import (
    NOTICE_ID_PREFIX, Reminder, ReminderActionStatus, ReminderHomeActionReceipt, ReminderLedgerRow,
    ReminderState, ReminderStore,
)
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionService

logger = logging.getLogger(__name__)

DEFAULT_LATE_ACTION_GRACE_S = 120.0
DEFAULT_NOTICE_MAX_ATTEMPTS = 3
DEFAULT_NOTICE_RETRY_MIN_GAP_S = 30.0
DEFAULT_NOTICE_RETENTION_HOURS = 48.0
NOTICE_CLEANUP_INTERVAL_S = 6 * 3600.0
LATE_NOT_EXECUTED = "late_not_executed"
# A recurring catch-up never walks further than this many missed occurrences
# (about eleven years of a daily reminder); beyond it the row is held, not replayed.
MAX_CATCH_UP_STEPS = 4096
MAX_LOOP_ERROR_DELAY_S = 60.0

# Voice-notice receipts (P3-F0/notices/v1). Reasons are diagnostic text for people
# and C16 counts: no code reads them to choose, retry, advance or delete a row.
NOTICE_KIND = "voice_notice"
NOTICE_PARTIAL = "delivery_partial"
NOTICE_CANCELLED = "delivery_cancelled"
NOTICE_UNVERIFIED = "delivery_unverified"
NOTICE_NOT_STARTED = "delivery_not_started:"
NOTICE_DISMISSED = "dismissed"
_NOT_STARTED_STATES = frozenset({DeliveryState.BLOCKED, DeliveryState.FAILED, DeliveryState.EMPTY})
_TERMINAL = frozenset({ReminderActionStatus.CONFIRMED, ReminderActionStatus.UNCERTAIN})


class LateActionOutcome(Enum):
    """What the late barrier found or did for one home action of an occurrence.

    The late notice is built from these outcomes, never from the reminder's
    configured actions alone, so it cannot contradict the durable receipts.
    """

    NOT_EXECUTED = "not_executed"  # receipt failed/late_not_executed (written now or by an earlier barrier)
    CONFIRMED = "confirmed"        # an earlier on-time delivery already ran it
    FAILED = "failed"              # an earlier attempt failed for another reason
    UNCERTAIN = "uncertain"        # an earlier send may or may not have reached the house


class NoticeDismissal(StrEnum):
    """Result of a household web dismiss."""

    DISMISSED = "dismissed"  # a pending occurrence got a confirmed/dismissed attempt and advanced
    PURGED = "purged"        # finished rows of a non-pending occurrence were deleted early
    BUSY = "busy"            # the occurrence is being announced right now
    NOT_FOUND = "not_found"  # no pending occurrence and no finished rows


def _receipt_outcome(receipt: ReminderHomeActionReceipt | None) -> LateActionOutcome:
    """Classify an existing receipt without changing it."""
    if receipt is None:
        return LateActionOutcome.UNCERTAIN
    if receipt.status is ReminderActionStatus.CONFIRMED:
        return LateActionOutcome.CONFIRMED
    if receipt.status is ReminderActionStatus.FAILED:
        return LateActionOutcome.NOT_EXECUTED if receipt.reason == LATE_NOT_EXECUTED else LateActionOutcome.FAILED
    return LateActionOutcome.UNCERTAIN


def _action_request_id(reminder: Reminder, index: int) -> str:
    """Stable receipt key of one action of one occurrence."""
    identity = json.dumps([reminder.id, reminder.trigger_at, index], separators=(",", ":"), allow_nan=False)
    return "reminder:" + hashlib.sha256(identity.encode()).hexdigest()


def notice_request_id(reminder_id: str, trigger_at: float, attempt: int) -> str:
    """Stable receipt key of voice-notice attempt ``attempt`` (1, 2, ...) of one occurrence."""
    identity = json.dumps([reminder_id, trigger_at, attempt], separators=(",", ":"), allow_nan=False)
    return NOTICE_ID_PREFIX + hashlib.sha256(identity.encode()).hexdigest()


def notice_body(reminder_id: str, trigger_at: float, attempt: int, claimed_at: float, room: str | None) -> dict:
    """Immutable claim body of one attempt; it never carries the reminder text."""
    return {"kind": NOTICE_KIND, "reminder_id": reminder_id, "trigger_at": trigger_at, "attempt": attempt,
            "claimed_at": claimed_at, "room": room}


def notice_outcome(result: Any) -> tuple[ReminderActionStatus, str | None]:
    """Map one voice result to the attempt's terminal evidence (delivery_mapping, in order).

    Only a result that proves no output at all is 'failed' (retryable). Any
    result whose output may have started (spoken segments or output_started),
    a timeout after the queue stage, or an unverifiable result is 'uncertain'
    and is never spoken again.

    Args:
        result: What the voice output returned.

    Returns:
        (status, diagnostic reason).
    """
    if not isinstance(result, ResponseResult):
        return ReminderActionStatus.UNCERTAIN, NOTICE_UNVERIFIED
    if result.success:
        return ReminderActionStatus.CONFIRMED, None
    started = result.spoken_segments > 0 or result.output_started
    if result.delivery is DeliveryState.TIMED_OUT and result.timeout_stage is not ResponseStage.QUEUE:
        return ReminderActionStatus.UNCERTAIN, NOTICE_PARTIAL
    if result.delivery is DeliveryState.CANCELLED:
        if started:
            return ReminderActionStatus.UNCERTAIN, NOTICE_CANCELLED
        return ReminderActionStatus.FAILED, NOTICE_NOT_STARTED + DeliveryState.CANCELLED.value
    if started:
        return ReminderActionStatus.UNCERTAIN, NOTICE_PARTIAL
    if result.delivery in _NOT_STARTED_STATES or result.delivery is DeliveryState.TIMED_OUT:
        return ReminderActionStatus.FAILED, NOTICE_NOT_STARTED + result.delivery.value
    return ReminderActionStatus.UNCERTAIN, NOTICE_UNVERIFIED


def _notice_body_of(row: ReminderLedgerRow) -> dict | None:
    try:
        body = json.loads(row.body)
    except ValueError:
        return None
    return body if type(body) is dict and body.get("kind") == NOTICE_KIND else None


@dataclass
class DeliveryHold:
    """In-process state of one occurrence in the current notice round.

    Attributes:
        trigger_at: The occurrence the hold belongs to; a new occurrence is never held.
        failures: Failed attempts of this occurrence in this round.
        retry_at: Earliest next attempt, or None when no further attempt is made in this round.
        reason: Last failure reason, for logs.
    """

    trigger_at: float
    failures: int = 0
    retry_at: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class NoticeClaim:
    """One claimed attempt waiting for the announcement's result."""

    occurrence: Reminder
    attempt: int
    request_id: str
    claimed_at: float
    late: bool
    outcomes: tuple[LateActionOutcome, ...] = ()


@dataclass(frozen=True)
class NoticeEntry:
    """Web projection of one shared notice (pending occurrence or finished attempt).

    Attributes:
        reminder_id: Shared reminder id.
        trigger_at: Scheduled time of the occurrence.
        text: Shared reminder text, or None when the reminder is no longer shared.
        status: 'pending', 'sending', 'confirmed' or 'uncertain'.
        reason: Diagnostic reason of the finished attempt (display only).
        attempts: Attempts claimed for the occurrence so far.
        failed_attempts: Attempts that never started any output.
        claimed_at: Claim time of the latest attempt, if any.
    """

    reminder_id: str
    trigger_at: float
    text: str | None
    status: str
    reason: str | None
    attempts: int
    failed_attempts: int
    claimed_at: float | None

    @property
    def pending(self) -> bool:
        return self.status in ("pending", "sending")


class ReminderScheduler:
    """Schedules and announces shared reminders on the configured output zone.

    Delivery policy (T034-U1, T034-U2 notice receipts):
      * Shared announcements go to the explicit ``output_zone``; a person's BLE
        presence never selects the room. Without a zone nothing is claimed,
        spoken, executed or marked fired.
      * Every announcement is a durable 'notice:' attempt claimed before any
        audio and finished from the voice result: complete -> confirmed;
        possibly heard (output started, post-queue timeout, unverified) ->
        uncertain, never spoken again; no output at all -> failed, retried
        with a bounded backoff. A crash while announcing is recovered as
        uncertain by the store. An occurrence stays pending until one attempt
        is confirmed or uncertain, then advances once.
      * An occurrence later than ``late_action_grace_s`` never runs its home
        actions: each action gets a terminal ``failed``/``late_not_executed``
        receipt before anything is spoken. On time, actions run only after a
        confirmed successful notice of the same pass.
      * A recurring reminder that missed several occurrences is announced once
        and advanced to its first future occurrence.
      * The loop is the only announcer; arrivals only wake it. Several due
        occurrences are claimed together and announced in one summary.
    """

    def __init__(
        self,
        store: ReminderStore,
        tts,
        presence_detector,
        ha_client,
        config: dict = None,
        *, home_actions: HomeActionService | None = None, wake_event: asyncio.Event | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self._store = store
        self._tts = tts
        self._presence_detector = presence_detector
        self._ha_client = ha_client
        self.home_actions = home_actions if home_actions is not None else HomeActionService(ha_client)
        self._config = config or {}
        self._clock = clock
        self._running = False
        self._wake_event = wake_event if wake_event is not None else asyncio.Event()
        self._holds: dict[str, DeliveryHold] = {}
        self._in_flight: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner_task: asyncio.Task | None = None

        self._max_attempts = self._config.get("notice_max_attempts", DEFAULT_NOTICE_MAX_ATTEMPTS)
        self._retry_gap = self._config.get("notice_retry_min_gap_s", DEFAULT_NOTICE_RETRY_MIN_GAP_S)
        self._retention_h = self._config.get("notice_retention_hours", DEFAULT_NOTICE_RETENTION_HOURS)
        self._late_grace = self._config.get("late_action_grace_s", DEFAULT_LATE_ACTION_GRACE_S)
        self._tts_prefix = self._config.get("tts_prefix", "Oye, recuerda")
        self._output_zone = self._config.get("output_zone")
        if type(self._max_attempts) is not int or self._max_attempts < 1:
            raise ValueError("reminders.notice_max_attempts must be a positive integer")
        for name, value in (("notice_retry_min_gap_s", self._retry_gap),
                            ("notice_retention_hours", self._retention_h),
                            ("late_action_grace_s", self._late_grace)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"reminders.{name} must be a finite non-negative number")
        if self._retry_gap <= 0:
            raise ValueError("reminders.notice_retry_min_gap_s must be positive")
        if self._output_zone is None:
            logger.warning("No reminder output zone configured: shared reminders are not announced, executed or "
                           "marked fired until reminders.output_zone or tts.streaming.default_output_zone is set")
        if self._retry_gap >= self._late_grace:
            logger.warning("reminders.notice_retry_min_gap_s (%s) >= late_action_grace_s (%s): every retry after a "
                           "failed first announcement is late, so its home actions will not run",
                           self._retry_gap, self._late_grace)

        # Arrival only wakes the loop; it never selects the room or announces by itself.
        if self._presence_detector is not None:
            self._presence_detector.on_user_arrived(self._on_user_arrived)

    @property
    def output_zone(self) -> str | None:
        """Zone of shared announcements, or None when nothing can be announced."""
        return self._output_zone

    @property
    def _can_announce(self) -> bool:
        """A notice is claimed only with a resolvable room and a voice output."""
        return self._output_zone is not None and getattr(self._tts, "aspeak", None) is not None

    async def start(self) -> None:
        """Main scheduler loop: wait for the next due or retryable occurrence and deliver it."""
        if self._running:
            return
        self._running = True
        self._runner_task = asyncio.current_task()
        self._loop = asyncio.get_running_loop()
        logger.info("ReminderScheduler started")
        error_delay = 1.0
        next_cleanup = self._clock()

        while self._running:
            try:
                self._wake_event.clear()
                if self._clock() >= next_cleanup:
                    next_cleanup = self._clock() + NOTICE_CLEANUP_INTERVAL_S
                    await self._cleanup_notices()
                pending = await self._store.get_next_pending(shared_only=True)
                now = self._clock()
                if pending is None:
                    self._holds.clear()
                    timeout = None
                elif pending.trigger_at > now:
                    timeout = pending.trigger_at - now
                else:
                    await self._deliver_due(now)
                    timeout = await self._next_timeout()
                error_delay = 1.0
                until_cleanup = max(0.0, next_cleanup - self._clock())
                timeout = until_cleanup if timeout is None else min(timeout, until_cleanup)
                if timeout > 0:
                    try:
                        await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        pass
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in scheduler loop")
                await asyncio.sleep(error_delay)
                error_delay = min(error_delay * 2, MAX_LOOP_ERROR_DELAY_S)

        self._running = False
        self._runner_task = None
        self._loop = None
        logger.info("ReminderScheduler stopped")

    async def _cleanup_notices(self) -> None:
        """Retention: drop finished notice rows 48 h after their claim once the occurrence is not pending."""
        cutoff = self._clock() - self._retention_h * 3600
        try:
            removed = await self._store.purge_notice_rows(cutoff)
        except Exception:
            logger.exception("Reminder notice retention cleanup failed")
            return
        if removed:
            logger.info("Reminder notice retention removed %d finished rows", removed)

    async def _deliver_due(self, now: float) -> None:
        """Announce every due occurrence that is not held or already in flight, in one summary."""
        due = await self._store.get_due(now, shared_only=True)
        current = {reminder.id: reminder.trigger_at for reminder in due}
        self._holds = {identity: hold for identity, hold in self._holds.items()
                       if current.get(identity) == hold.trigger_at}
        await self._deliver_batch(due, raise_errors=False)

    async def _next_timeout(self) -> float | None:
        """Seconds until the next claimable occurrence or retry, or None to wait for a wake.

        Due occurrences that cannot be claimed (no room or voice output,
        exhausted in this round, in flight) do not count, so nothing spins.
        """
        if not self._can_announce:
            return None
        now = self._clock()
        deadlines = [hold.retry_at for hold in self._holds.values() if hold.retry_at is not None]
        for reminder in await self._store.get_active_for_user(None, shared_only=True):
            if reminder.trigger_at > now:
                deadlines.append(reminder.trigger_at)
                break
            if not self._is_held(reminder, now):
                deadlines.append(now)
                break
        return max(0.0, min(deadlines) - now) if deadlines else None

    def _is_held(self, reminder: Reminder, now: float) -> bool:
        """True while this occurrence is in flight, backing off, or done for this round."""
        return reminder.id in self._in_flight or self._held_for_occurrence(reminder, now)

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        self._wake_event.set()
        self._loop = None
        task = self._runner_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        logger.info("ReminderScheduler stop requested")

    def notify_new_reminder(self) -> None:
        """Wake the scheduler loop to re-evaluate next pending reminder."""
        self._wake_event.set()

    def notify_output_available(self) -> None:
        """Start a new notice round (zone available again, unmute) and wake the loop.

        Occurrences that used up their attempts in the previous round become
        claimable again; attempt numbers continue in the ledger.
        """
        self._holds = {identity: hold for identity, hold in self._holds.items() if identity in self._in_flight}
        self._wake_event.set()

    async def _shared_occurrence(self, reminder: Reminder, *, guard: AuthorityGuard | None = None) -> Reminder | None:
        """Resolve the same active, due shared occurrence without adopting its successor."""
        identity, selected_trigger = reminder.id, reminder.trigger_at
        current = await self._store.get_by_id(identity, shared_only=True, guard=guard)
        if (current is None or current.state is not ReminderState.ACTIVE
                or current.trigger_at != selected_trigger or current.trigger_at > self._clock()):
            return None
        return current

    def _is_late(self, reminder: Reminder) -> bool:
        """An occurrence past the grace window may only be announced."""
        return self._clock() - reminder.trigger_at > self._late_grace

    async def _attempts(self, reminder: Reminder, *, guard: AuthorityGuard | None = None) -> list[ReminderLedgerRow]:
        """Notice attempts 1..n of one occurrence (n + 1 is the next attempt)."""
        identity, trigger_at = reminder.id, reminder.trigger_at
        return await self._store.probe_ledger(lambda attempt: notice_request_id(identity, trigger_at, attempt),
                                              guard=guard)

    async def _fire_reminder(self, reminder: Reminder) -> bool:
        """Deliver one selected occurrence; return True when its notice was confirmed.

        Errors of this occurrence propagate (after the in-process hold).
        """
        return reminder.id in await self._deliver_batch([reminder], raise_errors=True)

    async def _deliver_batch(self, candidates: list[Reminder], *, raise_errors: bool) -> set[str]:
        """Claim every deliverable occurrence first, speak one summary, finish every claim.

        Single flight is keyed by the reminder id rather than (id, trigger_at):
        a recurring catch-up moves the occurrence to a new trigger_at while it
        is being delivered, and the new key must not be deliverable meanwhile.

        Returns:
            Ids whose notice attempt finished 'confirmed' from a complete delivery.
        """
        owned: list[Reminder] = []
        for reminder in candidates:
            if reminder.id not in self._in_flight:
                self._in_flight.add(reminder.id)
                owned.append(reminder)
        confirmed: set[str] = set()
        try:
            claims: list[NoticeClaim] = []
            for reminder in owned:
                try:
                    claim = await self._prepare(reminder)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._hold_after_failure(reminder, type(error).__name__, self._clock())
                    if raise_errors:
                        raise
                    logger.exception("Reminder %s delivery failed", reminder.id)
                    continue
                if claim is not None:
                    claims.append(claim)
            if not claims:
                return confirmed
            status, reason = await self._announce(self._summary(claims))
            for claim in claims:
                try:
                    if await self._finish(claim, status, reason):
                        confirmed.add(claim.occurrence.id)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    # Never re-spoken in this process; a later round finds the terminal receipt.
                    self._park(claim.occurrence, type(error).__name__)
                    if raise_errors:
                        raise
                    logger.exception("Reminder %s completion failed", claim.occurrence.id)
            return confirmed
        finally:
            for reminder in owned:
                self._in_flight.discard(reminder.id)
            if owned and asyncio.current_task() is not self._runner_task:
                self._wake_event.set()

    async def _prepare(self, reminder: Reminder) -> NoticeClaim | None:
        """Resolve, collapse, bar late actions and claim the next notice attempt of one occurrence."""
        current = await self._shared_occurrence(reminder)
        if current is None or self._held_for_occurrence(current, self._clock()):
            return None
        if not await self._open_for_claim(current):
            return None
        collapsed = await self._collapse_missed(current)
        if collapsed is None:
            return None
        if collapsed.trigger_at != current.trigger_at and not await self._open_for_claim(collapsed):
            return None
        current = collapsed
        late = self._is_late(current)
        outcomes: list[LateActionOutcome] = []
        if late:
            # Durable barrier first: a crash or a failed announcement cannot
            # leave the late actions executable by a later attempt.
            barred = await self._bar_late_actions(current)
            if barred is None:
                return None
            outcomes = barred
        if not self._can_announce:
            # CORE-06: no room or no voice output, no claim; the occurrence stays pending.
            return None
        attempts = await self._attempts(current)
        if any(row.status in _TERMINAL or row.status is ReminderActionStatus.SENDING for row in attempts):
            self._park(current, "notice_changed_while_preparing")
            return None
        attempt, claimed_at = len(attempts) + 1, self._clock()
        request_id = notice_request_id(current.id, current.trigger_at, attempt)
        body = notice_body(current.id, current.trigger_at, attempt, claimed_at, self._output_zone)
        try:
            created = await self._store.claim_home_action(request_id, body)
        except ValueError:
            created = False
        if not created:
            # Another claimant owns this attempt: never speak it here.
            self._park(current, "notice_claimed_elsewhere")
            return None
        return NoticeClaim(current, attempt, request_id, claimed_at, late, tuple(outcomes))

    async def _open_for_claim(self, occurrence: Reminder) -> bool:
        """False when the occurrence already ended (advance it now) or another attempt is in flight."""
        attempts = await self._attempts(occurrence)
        if any(row.status in _TERMINAL for row in attempts):
            # A crash left a terminal notice without its advance: complete it silently.
            await self._advance(occurrence)
            return False
        if attempts and attempts[-1].status is ReminderActionStatus.SENDING:
            # Claimed by a live announcement elsewhere; restart recovery makes it uncertain.
            self._park(occurrence, "notice_in_flight")
            return False
        return True

    async def _announce(self, message: str) -> tuple[ReminderActionStatus, str | None]:
        """Speak once through the output's aspeak and map its ResponseResult."""
        callback = getattr(self._tts, "aspeak", None)
        try:
            if inspect.iscoroutinefunction(callback):
                result = await callback(message, zone_id=self._output_zone)
            else:
                result = await asyncio.to_thread(callback, message, zone_id=self._output_zone)
                if inspect.isawaitable(result):
                    result = await result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("Reminder voice output raised %s; notice is uncertain", type(error).__name__)
            return ReminderActionStatus.UNCERTAIN, NOTICE_UNVERIFIED
        return notice_outcome(result)

    async def _finish(self, claim: NoticeClaim, status: ReminderActionStatus, reason: str | None) -> bool:
        """Record the attempt's evidence, then act and advance only as that evidence allows."""
        occurrence = claim.occurrence
        await self._store.finish_home_action(claim.request_id, status.value, reason)
        logger.info("Reminder %s notice attempt %d in zone %s: %s (%s, late=%s)", occurrence.id, claim.attempt,
                    self._output_zone, status.value, reason, claim.late)
        if status is ReminderActionStatus.FAILED:
            self._hold_after_failure(occurrence, reason or status.value, claim.claimed_at)
            return False
        self._holds.pop(occurrence.id, None)
        if status is ReminderActionStatus.CONFIRMED and not claim.late:
            await self._execute_home_actions(occurrence)
        await self._advance(occurrence)
        return status is ReminderActionStatus.CONFIRMED

    def _held_for_occurrence(self, reminder: Reminder, now: float) -> bool:
        """True while this exact occurrence is backing off or done for this round."""
        hold = self._holds.get(reminder.id)
        if hold is None or hold.trigger_at != reminder.trigger_at:
            return False
        return hold.retry_at is None or hold.retry_at > now

    def _hold(self, reminder: Reminder) -> DeliveryHold:
        hold = self._holds.get(reminder.id)
        if hold is None or hold.trigger_at != reminder.trigger_at:
            hold = self._holds[reminder.id] = DeliveryHold(reminder.trigger_at)
        return hold

    def _hold_after_failure(self, reminder: Reminder, reason: str, failed_at: float) -> None:
        """Back off 30 s, 60 s, ...; after notice_max_attempts no further attempt in this round."""
        hold = self._hold(reminder)
        hold.failures += 1
        hold.reason = reason
        if hold.failures >= self._max_attempts:
            hold.retry_at = None
            logger.warning("Reminder %s not delivered (%s) after %d attempts in this round; kept pending",
                           reminder.id, reason, hold.failures)
        else:
            hold.retry_at = failed_at + self._retry_gap * 2 ** (hold.failures - 1)
            logger.info("Reminder %s not delivered (%s); attempt %d/%d in %.0f s", reminder.id, reason,
                        hold.failures + 1, self._max_attempts, hold.retry_at - self._clock())

    def _park(self, reminder: Reminder, reason: str) -> None:
        """No further attempt for this occurrence in this round."""
        hold = self._hold(reminder)
        hold.retry_at = None
        hold.reason = reason

    async def _collapse_missed(self, reminder: Reminder) -> Reminder | None:
        """Move a recurring reminder to its latest due occurrence so a long outage yields one notice."""
        if not reminder.recurrence:
            return reminder
        now, latest = self._clock(), reminder.trigger_at
        for _ in range(MAX_CATCH_UP_STEPS):
            following = next_trigger(latest, reminder.recurrence)
            if following <= latest:
                raise ValueError("reminder_recurrence_not_advancing")
            if following > now or (reminder.recurrence_end is not None and following > reminder.recurrence_end):
                break
            latest = following
        else:
            raise ValueError("reminder_catch_up_unbounded")
        if latest == reminder.trigger_at:
            return reminder
        if not await self._store.update_trigger(reminder.id, latest, shared_only=True,
                                                expected_trigger_at=reminder.trigger_at):
            return None
        logger.info("Recurring reminder %s collapsed missed occurrences into one notice", reminder.id)
        return replace(reminder, trigger_at=latest)

    def _message(self, reminder: Reminder, late: bool, outcomes: list[LateActionOutcome] | tuple) -> str:
        """Build the announcement; a late notice states only what the receipts prove.

        Args:
            reminder: The occurrence being announced.
            late: Whether the occurrence is past the action grace.
            outcomes: What the late barrier found for each action (empty when on time).
        """
        message = f"{self._tts_prefix} {reminder.text}"
        if not late:
            return message
        notice = "Este aviso llega con atraso"
        if outcomes and all(outcome is LateActionOutcome.NOT_EXECUTED for outcome in outcomes):
            notice += " y no ejecuté sus acciones programadas"
        elif LateActionOutcome.UNCERTAIN in outcomes:
            notice += " y no puedo confirmar si sus acciones programadas se ejecutaron"
        return f"{message.rstrip(' .')}. {notice}."

    def _summary(self, claims: list[NoticeClaim]) -> str:
        """One announcement for every claimed occurrence of the room."""
        parts = [self._message(claim.occurrence, claim.late, claim.outcomes) for claim in claims]
        if len(parts) == 1:
            return parts[0]
        return f"Tienes {len(parts)} avisos pendientes. " + " ".join(part.rstrip(" .") + "." for part in parts)

    async def _advance(self, reminder: Reminder, *, guard: AuthorityGuard | None = None) -> None:
        """Complete an occurrence: mark a one-shot fired, move a recurring one to its first future occurrence."""
        if not reminder.recurrence:
            if await self._store.mark_fired(reminder.id, shared_only=True, expected_trigger_at=reminder.trigger_at,
                                            guard=guard):
                logger.info("One-shot reminder %s marked fired", reminder.id)
            return
        now, following = self._clock(), reminder.trigger_at
        for _ in range(MAX_CATCH_UP_STEPS):
            later = next_trigger(following, reminder.recurrence)
            if later <= following:
                raise ValueError("reminder_recurrence_not_advancing")
            following = later
            if following > now or (reminder.recurrence_end is not None and following > reminder.recurrence_end):
                break
        else:
            raise ValueError("reminder_catch_up_unbounded")
        if reminder.recurrence_end is not None and following > reminder.recurrence_end:
            if await self._store.mark_fired(reminder.id, shared_only=True, expected_trigger_at=reminder.trigger_at,
                                            guard=guard):
                logger.info("Recurring reminder %s past recurrence_end, marked fired", reminder.id)
        elif await self._store.update_trigger(reminder.id, following, shared_only=True,
                                              expected_trigger_at=reminder.trigger_at, guard=guard):
            logger.info("Recurring reminder %s rescheduled to %s", reminder.id, following)

    def _on_user_arrived(self, user_id: str, presence: Any) -> None:
        """Callback from PresenceDetector when a user arrives home: only wake the loop.

        The arrival never chooses the announcement room or announces by itself.
        This callback is invoked synchronously from the presence state update
        loop, so it must not block.

        Args:
            user_id: The arriving user.
            presence: UserPresence dataclass (its zone is not used for routing).
        """
        loop = self._loop
        if loop is None:
            logger.debug("Scheduler loop not available yet, ignoring arrival of %s", user_id)
            return
        logger.info("User %s arrived, waking the reminder loop", user_id)
        try:
            loop.call_soon_threadsafe(self._wake_event.set)
        except RuntimeError:
            logger.debug("Reminder callback arrived after loop shutdown")

    async def missed_reminder_on_arrival(self, user_id: str, zone_id: str | None) -> int:
        """Wake the loop after an arrival; shared reminders are never delivered by presence.

        Args:
            user_id: The arriving user (only triggers the re-check).
            zone_id: Zone where the user was detected; not used for routing.

        Returns:
            Always 0: the scheduler loop is the only announcer (single flight).
        """
        self._wake_event.set()
        return 0

    async def dismiss_notice(self, reminder_id: str, trigger_at: float, *,
                             guard: AuthorityGuard | None = None) -> NoticeDismissal:
        """Household web dismiss of one shared notice.

        A pending occurrence gets its next attempt claimed and finished
        'confirmed'/'dismissed', then advances without home actions. For an
        occurrence that is no longer pending, its finished rows are purged early.

        Args:
            reminder_id: Shared reminder id.
            trigger_at: Scheduled time of the occurrence.
            guard: Authority guard of the web session.
        """
        if reminder_id in self._in_flight:
            return NoticeDismissal.BUSY
        self._in_flight.add(reminder_id)
        try:
            current = await self._store.get_by_id(reminder_id, shared_only=True, guard=guard)
            if (current is not None and current.state is ReminderState.ACTIVE
                    and current.trigger_at == trigger_at and trigger_at <= self._clock()):
                attempts = await self._attempts(current, guard=guard)
                if any(row.status in _TERMINAL for row in attempts):
                    await self._advance(current, guard=guard)
                    return NoticeDismissal.DISMISSED
                if attempts and attempts[-1].status is ReminderActionStatus.SENDING:
                    return NoticeDismissal.BUSY
                attempt = len(attempts) + 1
                request_id = notice_request_id(current.id, current.trigger_at, attempt)
                body = notice_body(current.id, current.trigger_at, attempt, self._clock(), None)
                try:
                    created = await self._store.claim_home_action(request_id, body, guard=guard)
                except ValueError:
                    created = False
                if not created:
                    return NoticeDismissal.BUSY
                # The guarded claim is the authorized act; finishing and advancing it cannot be
                # left half done by a session that expires in between (a stuck 'sending' would
                # keep the occurrence silent until restart).
                await self._store.finish_home_action(request_id, ReminderActionStatus.CONFIRMED.value,
                                                     NOTICE_DISMISSED)
                self._holds.pop(reminder_id, None)
                logger.info("Reminder %s notice dismissed from the web", reminder_id)
                await self._advance(current)
                return NoticeDismissal.DISMISSED
            purged = await self._store.purge_notice_rows(math.inf, reminder_id=reminder_id, trigger_at=trigger_at,
                                                         guard=guard)
            return NoticeDismissal.PURGED if purged else NoticeDismissal.NOT_FOUND
        finally:
            self._in_flight.discard(reminder_id)
            self._wake_event.set()

    async def notice_entries(self, *, guard: AuthorityGuard | None = None) -> list[NoticeEntry]:
        """Shared notices for the web: pending occurrences plus finished attempts within retention."""
        now = self._clock()
        entries: list[NoticeEntry] = []
        for reminder in await self._store.get_due(now, shared_only=True, guard=guard):
            attempts = await self._attempts(reminder, guard=guard)
            if any(row.status in _TERMINAL for row in attempts):
                continue
            last = _notice_body_of(attempts[-1]) if attempts else None
            entries.append(NoticeEntry(
                reminder.id, reminder.trigger_at, reminder.text,
                "sending" if attempts and attempts[-1].status is ReminderActionStatus.SENDING else "pending",
                None, len(attempts), sum(row.status is ReminderActionStatus.FAILED for row in attempts),
                last.get("claimed_at") if last else None))
        cutoff = now - self._retention_h * 3600
        texts: dict[str, str | None] = {}
        for row in await self._store.get_notice_rows(guard=guard):
            body = _notice_body_of(row)
            if row.status not in _TERMINAL or body is None:
                continue
            identity, trigger_at, claimed_at = body.get("reminder_id"), body.get("trigger_at"), body.get("claimed_at")
            if (type(identity) is not str or type(trigger_at) not in (int, float)
                    or type(claimed_at) not in (int, float) or claimed_at < cutoff
                    or row.request_id != notice_request_id(identity, trigger_at, body.get("attempt"))):
                continue
            if identity not in texts:
                shared = await self._store.get_by_id(identity, shared_only=True, guard=guard)
                texts[identity] = shared.text if shared is not None else None
            attempts = await self._store.probe_ledger(
                lambda attempt, i=identity, t=trigger_at: notice_request_id(i, t, attempt), guard=guard)
            entries.append(NoticeEntry(identity, trigger_at, texts[identity], row.status.value, row.reason,
                                       len(attempts),
                                       sum(item.status is ReminderActionStatus.FAILED for item in attempts),
                                       claimed_at))
        return entries

    async def _bar_late_action(self, request_id: str, action: Any) -> LateActionOutcome:
        """Record a terminal late receipt, or report the existing one unchanged."""
        if await self._store.claim_home_action(request_id, action):
            await self._store.finish_home_action(request_id, "failed", LATE_NOT_EXECUTED)
            logger.warning("Late reminder home action not executed")
            return LateActionOutcome.NOT_EXECUTED
        outcome = _receipt_outcome(await self._store.get_home_action_receipt(request_id))
        logger.warning("Late reminder home action already has a receipt (%s); left unchanged", outcome.value)
        return outcome

    async def _bar_late_actions(self, reminder: Reminder) -> list[LateActionOutcome] | None:
        """Bar every action of a late occurrence; None when the occurrence is no longer current.

        An existing receipt (confirmed, failed or uncertain) is never
        overwritten or resent; it is reported so the notice cannot contradict it.
        """
        current = await self._shared_occurrence(reminder)
        if current is None:
            return None
        return [await self._bar_late_action(_action_request_id(current, index), action)
                for index, action in enumerate(current.ha_actions or [])]

    async def _execute_home_actions(self, reminder: Reminder) -> None:
        """Claim durable effects and apply current policy for an on-time occurrence.

        Lateness is re-checked before every action: once the occurrence is past
        the grace, the remaining actions only get late receipts.
        """
        reminder = await self._shared_occurrence(reminder)
        if reminder is None:
            return
        for index, action in enumerate(reminder.ha_actions or []):
            request_id = _action_request_id(reminder, index)
            if self._is_late(reminder):
                await self._bar_late_action(request_id, action)
                continue
            if not await self._store.claim_home_action(request_id, action):
                logger.warning("Reminder home action already claimed; no automatic resend")
                receipt = await self._store.get_home_action_receipt(request_id)
                if receipt is not None and receipt.status == "confirmed":
                    continue
                # Do not advance a partially executed sequence after uncertainty.
                break
            if type(action) is not dict or action.keys() - {"domain", "service", "entity_id", "data"}:
                await self._store.finish_home_action(request_id, "failed", "invalid_reminder_action")
                break
            result = await self.home_actions.execute_service(
                action.get("domain", ""), action.get("service", ""), action.get("entity_id", ""), action.get("data"),
                actor=HomeActor.household(), origin=ActionOrigin.REMINDER, request_id=request_id,
            )
            await self._store.finish_home_action(request_id, result.status, result.reason)
            logger.info("Reminder home action finished: status=%s", result.status)
            if result.status != "confirmed":
                break
