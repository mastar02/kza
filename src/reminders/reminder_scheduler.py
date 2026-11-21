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

from src.core.household_clock import ZONE_PREFIX, HouseholdClock
from src.core.sqlite_work import AuthorityGuard
from src.core.voice_delivery import SIGNALS_OUTPUT_FREE, DeliveryState, ResponseResult, ResponseStage
from src.reminders.recurrence import next_trigger
from src.reminders.reminder_store import (
    NOTICE_ID_PREFIX, Reminder, ReminderActionStatus, ReminderHomeActionReceipt, ReminderLedgerRow,
    ReminderScope, ReminderState, ReminderStore,
)
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionService

logger = logging.getLogger(__name__)

DEFAULT_LATE_ACTION_GRACE_S = 120.0
DEFAULT_NOTICE_MAX_ATTEMPTS = 3
DEFAULT_NOTICE_RETRY_MIN_GAP_S = 30.0
DEFAULT_NOTICE_RETENTION_HOURS = 48.0
# User answer Q31 (2026-11-16): after a round is exhausted, a fallback round every 15 min,
# at most 4 per occurrence, in case the output-free signal never comes.
DEFAULT_NOTICE_FALLBACK_ROUND_S = 900.0
DEFAULT_NOTICE_FALLBACK_MAX_ROUNDS = 4
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
# Q32b durable mark of a grouped occurrence (one per grouping, in the action ledger, never purged
# like the action receipts): a restart or a further catch-up collapse still knows the notice
# covers occurrences missed after a terminal notice, so it stays late and never runs actions.
GROUPED_ID_PREFIX = "grouped:"
GROUPED_KIND = "grouped_notice"
GROUPED_NOTICE = "grouped_late_notice"
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


def grouped_request_id(reminder_id: str, trigger_at: float) -> str:
    """Stable ledger key of the Q32b mark of one grouped occurrence."""
    identity = json.dumps([reminder_id, trigger_at], separators=(",", ":"), allow_nan=False)
    return GROUPED_ID_PREFIX + hashlib.sha256(identity.encode()).hexdigest()


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


def notice_contended(result: Any) -> bool:
    """Whether a notice that started no output lost the output to other activity.

    It waited past the queue budget behind another response, or was cancelled
    before any audio. Only then can a free output help it; a notice the output
    refused by itself (failed, blocked, empty) is retried by its backoff and the
    fallback rounds, never by every finished response (review RV-Q31-1). Decided
    from the typed result, never from the diagnostic reason text.

    Args:
        result: What the voice output returned.

    Returns:
        True for a queue timeout or a cancellation before any output.
    """
    if not isinstance(result, ResponseResult) or result.spoken_segments > 0 or result.output_started:
        return False
    if result.delivery is DeliveryState.CANCELLED:
        return True
    return result.delivery is DeliveryState.TIMED_OUT and result.timeout_stage is ResponseStage.QUEUE


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
        completion_errors: Failed tries to complete (advance) an occurrence whose notice is
            already terminal; such a retry never speaks.
        fallback_rounds: Timed fallback rounds already scheduled for this occurrence in this
            process (bounded by notice_fallback_max_rounds; an output-free round keeps the count).
        successor_round: Latest later occurrence of a recurring reminder for which a round was
            already opened while this occurrence was held (one round per newly due occurrence).
        output_refused: The last failed attempt started no output although no other activity
            held the output (failed, blocked, empty); an output-free signal does not reopen it,
            so a fast-failing output cannot add one attempt per finished response (RV-Q31-1).
        output_round: The current round was opened by the output-free signal; it is not claimed
            while a follow-up conversation is active (user answer Q33).
        timed_retry_at: The backoff or fallback time the output-free round replaced; it still
            applies while that round waits for the conversation to end (bounded wait).
    """

    trigger_at: float
    failures: int = 0
    completion_errors: int = 0
    retry_at: float | None = None
    reason: str = ""
    fallback_rounds: int = 0
    successor_round: float | None = None
    output_refused: bool = False
    output_round: bool = False
    timed_retry_at: float | None = None


@dataclass(frozen=True)
class NoticeClaim:
    """One claimed attempt waiting for the announcement's result."""

    occurrence: Reminder
    attempt: int
    request_id: str
    claimed_at: float
    late: bool
    zone: str
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
    """Schedules and announces shared reminders in their delivery zones.

    Delivery policy (T034-U1, T034-U2 notice receipts):
      * Shared announcements use the saved delivery room, or ``output_zone``
        for rows without one. Without either, the notice stays pending.
      * Every announcement is a durable 'notice:' attempt claimed before any
        audio and finished from the voice result: complete -> confirmed;
        possibly heard (output started, post-queue timeout, unverified) ->
        uncertain, never spoken again; no output at all -> failed, retried
        with a bounded backoff. A crash while announcing is recovered as
        uncertain by the store. An occurrence stays pending until one attempt
        is confirmed or uncertain, then advances once.
      * A round has ``notice_max_attempts`` attempts. A new round starts at
        startup, when other activity frees the output (``notify_output_available``,
        wired to ResponseHandler), every ``notice_fallback_round_s`` after an
        exhausted round (at most ``notice_fallback_max_rounds``), and when a later
        occurrence of a held recurring reminder becomes due, which the catch-up
        collapse then announces instead. The output-free round skips an
        occurrence whose last attempt the output itself refused (no other
        activity held it), and a signal that finds an occurrence in flight is
        applied when that flight ends. A round opened by that signal is not
        claimed while a follow-up conversation is active (``watch_conversation``,
        user answer Q33): it waits for the conversation to end, while the
        occurrence's own backoff or fallback time still applies.
      * An occurrence later than ``late_action_grace_s`` never runs its home
        actions: each action gets a terminal ``failed``/``late_not_executed``
        receipt before anything is spoken. On time, actions run only after a
        confirmed successful notice of the same pass.
      * A recurring reminder that missed several occurrences is announced once
        and advanced to its first future occurrence; after a terminal notice
        whose advance was lost (crash), missed later occurrences still get that
        one notice (user answer Q32), which is always a late notice that never
        runs home actions, even inside the grace (user answer Q32b; a durable
        grouped mark in the action ledger keeps that across a restart and a
        further catch-up collapse).
      * The loop is the only announcer; arrivals only wake it. Several due
        occurrences for the same zone are claimed together and announced in one summary.
    """

    def __init__(
        self,
        store: ReminderStore,
        tts,
        presence_detector,
        ha_client,
        config: dict = None,
        *, home_actions: HomeActionService | None = None, wake_event: asyncio.Event | None = None,
        clock: Callable[[], float] | None = None,
        household_clock: HouseholdClock | None = None,
    ):
        """Build the scheduler.

        Args:
            clock: Epoch-seconds clock for tests; the household timezone is
                then the supported default. Exclusive with ``household_clock``.
            household_clock: The application's household clock (P3-F1); its
                zone counts recurring days and its epoch drives due times.

        Raises:
            ValueError: Both ``clock`` and ``household_clock`` were given, or a
                notice setting is invalid.
        """
        if clock is not None and household_clock is not None:
            raise ValueError("Pass either clock or household_clock, not both")
        self._store = store
        self._tts = tts
        self._presence_detector = presence_detector
        self._ha_client = ha_client
        self.home_actions = home_actions if home_actions is not None else HomeActionService(ha_client)
        self._config = config or {}
        self._household_clock = household_clock or HouseholdClock(time_source=clock or time.time)
        self._clock = clock or self._household_clock.epoch
        self._running = False
        self._wake_event = wake_event if wake_event is not None else asyncio.Event()
        self._holds: dict[str, DeliveryHold] = {}
        self._in_flight: set[str] = set()
        # Output-free signals that found the occurrence in flight; applied when its flight ends (RV-Q31-2).
        self._signal_pending: set[str] = set()
        # Q33: the follow-up conversation (FollowUpMode) and the audio loop's command turns
        # (CommandTurns) whose activity holds output-free rounds.
        self._conversation = None
        self._turns = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner_task: asyncio.Task | None = None

        self._max_attempts = self._config.get("notice_max_attempts", DEFAULT_NOTICE_MAX_ATTEMPTS)
        self._retry_gap = self._config.get("notice_retry_min_gap_s", DEFAULT_NOTICE_RETRY_MIN_GAP_S)
        self._retention_h = self._config.get("notice_retention_hours", DEFAULT_NOTICE_RETENTION_HOURS)
        self._late_grace = self._config.get("late_action_grace_s", DEFAULT_LATE_ACTION_GRACE_S)
        self._fallback_round_s = self._config.get("notice_fallback_round_s", DEFAULT_NOTICE_FALLBACK_ROUND_S)
        self._fallback_max_rounds = self._config.get("notice_fallback_max_rounds",
                                                     DEFAULT_NOTICE_FALLBACK_MAX_ROUNDS)
        self._tts_prefix = self._config.get("tts_prefix", "Oye, recuerda")
        self._output_zone = self._config.get("output_zone")
        if type(self._max_attempts) is not int or self._max_attempts < 1:
            raise ValueError("reminders.notice_max_attempts must be a positive integer")
        if type(self._fallback_max_rounds) is not int or self._fallback_max_rounds < 0:
            raise ValueError("reminders.notice_fallback_max_rounds must be a non-negative integer")
        for name, value in (("notice_retry_min_gap_s", self._retry_gap),
                            ("notice_retention_hours", self._retention_h),
                            ("late_action_grace_s", self._late_grace),
                            ("notice_fallback_round_s", self._fallback_round_s)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"reminders.{name} must be a finite non-negative number")
        if self._retry_gap <= 0:
            raise ValueError("reminders.notice_retry_min_gap_s must be positive")
        if self._fallback_round_s <= 0:
            raise ValueError("reminders.notice_fallback_round_s must be positive")
        if self._output_zone is None:
            logger.warning("No reminder output zone configured: shared reminders without a delivery room stay "
                           "pending until reminders.output_zone or tts.streaming.default_output_zone is set")
        if self._retry_gap >= self._late_grace:
            logger.warning("reminders.notice_retry_min_gap_s (%s) >= late_action_grace_s (%s): every retry after a "
                           "failed first announcement is late, so its home actions will not run",
                           self._retry_gap, self._late_grace)

        # Arrival only wakes the loop; it never selects the room or announces by itself.
        if self._presence_detector is not None:
            self._presence_detector.on_user_arrived(self._on_user_arrived)

    @property
    def output_zone(self) -> str | None:
        """Default zone of shared announcements without a delivery room."""
        return self._output_zone

    def _announcement_zone(self, reminder: Reminder) -> str | None:
        if reminder.delivery_room is None:
            return self._output_zone
        return ZONE_PREFIX + reminder.delivery_room

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
        """Announce due occurrences that are claimable, with one summary per destination."""
        due = await self._store.get_due(now, shared_only=True)
        current = {reminder.id: reminder.trigger_at for reminder in due}
        self._holds = {identity: hold for identity, hold in self._holds.items()
                       if current.get(identity) == hold.trigger_at}
        by_zone: dict[str | None, list[Reminder]] = {}
        for reminder in due:
            self._open_successor_round(reminder, now)
            zone = self._announcement_zone(reminder)
            key = zone.removeprefix(ZONE_PREFIX) if zone is not None else None
            by_zone.setdefault(key, []).append(reminder)
        for reminders in by_zone.values():
            preferred_zone = next((ZONE_PREFIX + r.delivery_room for r in reminders if r.delivery_room is not None),
                                  None)
            await self._deliver_batch(reminders, raise_errors=False, preferred_zone=preferred_zone)

    def _open_successor_round(self, reminder: Reminder, now: float) -> None:
        """A held recurring occurrence gets one new round when a later occurrence becomes due.

        Q31: a recurring reminder never loses its later occurrences while one is
        held (exhausted, waiting for a fallback round, parked). The round goes
        through _prepare, so the catch-up collapse announces the latest due
        occurrence once. One round per newly due occurrence: a round that
        cannot claim again is not reopened until the next occurrence is due.
        """
        hold = self._holds.get(reminder.id)
        if (hold is None or reminder.id in self._in_flight or hold.trigger_at != reminder.trigger_at
                or not self._held_for_occurrence(reminder, now)):
            return
        try:
            following = self._next_occurrence(reminder, hold.successor_round or reminder.trigger_at)
            if following is None or following > now:
                return
            latest = self._latest_due(reminder, following, now)
        except ValueError:
            # A recurrence the catch-up cannot walk stays held (MAX_CATCH_UP_STEPS); the
            # marker keeps _next_timeout from waking the loop for it again and again.
            hold.successor_round = now
            return
        hold.successor_round, hold.failures, hold.retry_at = latest, 0, now
        hold.output_round = False  # a newly due occurrence is scheduled, not an output-free round
        logger.info("Reminder %s: a later occurrence is due while a notice is held; new round", reminder.id)

    async def _next_timeout(self) -> float | None:
        """Seconds until the next claimable occurrence or retry, or None to wait for a wake.

        Due occurrences that cannot be claimed (no room or voice output,
        exhausted in this round, in flight) do not count, so nothing spins.
        """
        if getattr(self._tts, "aspeak", None) is None:
            return None
        now = self._clock()
        deadlines = [deadline for deadline in map(self._retry_deadline, self._holds.values())
                     if deadline is not None]
        for reminder in await self._store.get_active_for_user(None, shared_only=True):
            if self._announcement_zone(reminder) is None:
                continue
            if reminder.trigger_at > now:
                deadlines.append(reminder.trigger_at)
                break
            if not self._is_held(reminder, now):
                deadlines.append(now)
                break
            hold = self._holds.get(reminder.id)
            if reminder.id not in self._in_flight and hold is not None and hold.trigger_at == reminder.trigger_at:
                # A held recurring occurrence wakes the loop when its next occurrence is due (Q31).
                try:
                    following = self._next_occurrence(reminder, hold.successor_round or reminder.trigger_at)
                except ValueError:
                    following = None
                if following is not None:
                    deadlines.append(following)
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
        """Start a new notice round because the output became free again; thread-safe.

        Wired to ResponseHandler.add_output_free_listener (user answer Q31): it
        fires when other activity (not a reminder notice) stops owning the
        output. It may be called from any thread and never blocks; the round
        itself is opened on the scheduler loop.
        """
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._start_output_round)
            except RuntimeError:
                logger.debug("Output-free signal arrived after the reminder loop stopped")
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no scheduler loop and not on an event loop: nothing to wake
        self._start_output_round()

    def watch_conversation(self, conversation, turns=None) -> None:
        """Hold output-free rounds while this follow-up conversation is active (user answer Q33).

        A held notice reopened by the output-free signal waits until the
        follow-up window ends and, if the person keeps talking, until the
        conversation ends, so it never interrupts it. The window's timer can end
        the FollowUpMode conversation while the person is still saying a
        follow-up command, or while it is being answered (review round 3,
        Q33-1), so the audio loop's command turns hold the round too. Their own
        ends (add_end_listener) wake the loop; nothing is timed here.

        Args:
            conversation: The active audio loop's FollowUpMode (``is_active`` and
                ``add_end_listener``).
            turns: The same loop's CommandTurns (``active`` while a command is
                being captured or dispatched, and ``add_end_listener``), or None.
        """
        self._conversation = conversation
        conversation.add_end_listener(self.notify_conversation_ended)
        if turns is not None:
            self._turns = turns
            turns.add_end_listener(self.notify_conversation_ended)

    def _conversation_active(self) -> bool:
        """Whether the person may still be talking with KZA: the watched follow-up conversation is
        active or a command is being captured or dispatched; False when nothing is watched."""
        return (self._watched_state(self._conversation, "is_active")
                or self._watched_state(self._turns, "active"))

    @staticmethod
    def _watched_state(source, attribute: str) -> bool:
        if source is None:
            return False
        try:
            return bool(getattr(source, attribute))
        except Exception as error:
            logger.warning("%s.%s unavailable (%s); not holding notices for it", type(source).__name__,
                           attribute, type(error).__name__)
            return False

    def notify_conversation_ended(self) -> None:
        """Wake the loop for output-free rounds held by the conversation; thread-safe, never blocks.

        Wired to FollowUpMode.add_end_listener, which may call it from the
        event loop, the audio thread or the follow-up timer thread, and to
        CommandTurns.add_end_listener (the audio loop's event loop). The loop
        re-reads both states, so an end while the other still holds is harmless.
        """
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._conversation_ended)
            except RuntimeError:
                logger.debug("Conversation end arrived after the reminder loop stopped")
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no scheduler loop and not on an event loop: nothing to wake
        self._conversation_ended()

    def _conversation_ended(self) -> None:
        if any(hold.output_round for hold in self._holds.values()):
            logger.info("Follow-up conversation ended; held reminder notice round may speak")
            self._wake_event.set()

    def _retry_deadline(self, hold: DeliveryHold) -> float | None:
        """When a held occurrence becomes claimable again, or None to wait for a wake.

        An output-free round waits while a follow-up conversation is active
        (Q33); meanwhile only the backoff or fallback time it replaced applies.
        """
        if hold.output_round and self._conversation_active():
            return hold.timed_retry_at
        return hold.retry_at

    def _start_output_round(self) -> None:
        """Every held occurrence not in flight becomes claimable now with a fresh round.

        Attempt numbers continue in the ledger; the fallback round count is kept,
        so the bound on timed fallback rounds holds across output-free rounds.
        An occurrence the output itself refused is not reopened (RV-Q31-1). An
        occurrence in flight is left untouched; the signal is kept and applied
        when its flight ends (RV-Q31-2). The loop is woken only when some
        occurrence was reopened. During a follow-up conversation the reopened
        rounds are not claimed until it ends (Q33, _retry_deadline).
        """
        self._signal_pending.update(self._in_flight)
        now, opened = self._clock(), False
        for identity, hold in self._holds.items():
            if identity not in self._in_flight:
                opened = self._reopen_for_output(hold, now) or opened
        if opened:
            logger.info("Reminder output is free again; new notice round")
            self._wake_event.set()

    @staticmethod
    def _reopen_for_output(hold: DeliveryHold, now: float) -> bool:
        """Open a fresh round now for one held occurrence, unless the output itself refused it.

        The round is marked as an output-free round (Q33): it is claimed only
        while no follow-up conversation is active; the timed retry it replaces
        is kept for that wait.
        """
        if hold.output_refused:
            return False
        if not hold.output_round:
            hold.timed_retry_at = hold.retry_at
        hold.output_round = True
        hold.failures, hold.completion_errors, hold.retry_at = 0, 0, now
        return True

    def _end_flight(self, identities: list[str]) -> None:
        """Release single flight; an output-free signal received meanwhile opens its round now."""
        now, opened = self._clock(), False
        for identity in identities:
            self._in_flight.discard(identity)
            if identity in self._signal_pending:
                self._signal_pending.discard(identity)
                hold = self._holds.get(identity)
                if hold is not None:
                    opened = self._reopen_for_output(hold, now) or opened
        if opened:
            logger.info("Reminder output became free during a delivery; new notice round")
            self._wake_event.set()

    async def _shared_occurrence(self, reminder: Reminder, *, guard: AuthorityGuard | None = None) -> Reminder | None:
        """Resolve the same active, due shared occurrence without adopting its successor."""
        if reminder.scope is not ReminderScope.SHARED:
            return None  # Q18: personal reminders stay pending for the person's web session.
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

    async def _deliver_batch(self, candidates: list[Reminder], *, raise_errors: bool,
                             preferred_zone: str | None = None) -> set[str]:
        """Claim deliverable occurrences for one zone, speak, and finish every claim.

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
                    claim = await self._prepare(reminder, preferred_zone=preferred_zone)
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
            status, reason, contended = await self._announce(self._summary(claims), claims[0].zone)
            for claim in claims:
                try:
                    if await self._finish(claim, status, reason, contended):
                        confirmed.add(claim.occurrence.id)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    # Never re-spoken: a later pass finds the terminal receipt and only retries the advance.
                    self._retry_completion(claim.occurrence, type(error).__name__)
                    if raise_errors:
                        raise
                    logger.exception("Reminder %s completion failed", claim.occurrence.id)
            return confirmed
        finally:
            self._end_flight([reminder.id for reminder in owned])
            if owned and asyncio.current_task() is not self._runner_task:
                self._wake_event.set()

    async def _prepare(self, reminder: Reminder, *, preferred_zone: str | None = None) -> NoticeClaim | None:
        """Resolve, collapse, bar late actions and claim the next notice attempt of one occurrence."""
        current = await self._shared_occurrence(reminder)
        if current is None or self._backing_off(current, self._clock()):
            return None
        # Completing a terminal notice comes before the round's park: an occurrence that is
        # done for this round but whose notice already ended still advances on the next wake.
        if not await self._open_for_claim(current):
            return None
        if self._held_for_occurrence(current, self._clock()):
            return None
        # Q32b: the durable mark, not process memory, says whether this occurrence groups
        # occurrences missed after a terminal notice (it survives a restart).
        grouped = await self._is_grouped(current)
        collapsed = await self._collapse_missed(current, grouped=grouped)
        if collapsed is None:
            return None
        if collapsed.trigger_at != current.trigger_at:
            if not await self._open_for_claim(collapsed):
                return None
        current = collapsed
        # Q32b: a grouped notice after a terminal one is late even inside the grace; its actions
        # were already barred durably before the reminder moved to it, so nothing can run them.
        late = self._is_late(current) or grouped
        outcomes: list[LateActionOutcome] = []
        if late:
            # Durable barrier first: a crash or a failed announcement cannot
            # leave the late actions executable by a later attempt.
            barred = await self._bar_late_actions(current)
            if barred is None:
                return None
            outcomes = barred
        zone = self._announcement_zone(current)
        if zone is None or getattr(self._tts, "aspeak", None) is None:
            # CORE-06: no room or no voice output, no claim; the occurrence stays pending.
            return None
        if preferred_zone == ZONE_PREFIX + zone:
            zone = preferred_zone
        attempts = await self._attempts(current)
        if any(row.status in _TERMINAL or row.status is ReminderActionStatus.SENDING for row in attempts):
            self._park(current, "notice_changed_while_preparing")
            return None
        attempt, claimed_at = len(attempts) + 1, self._clock()
        request_id = notice_request_id(current.id, current.trigger_at, attempt)
        body = notice_body(current.id, current.trigger_at, attempt, claimed_at, zone)
        try:
            created = await self._store.claim_home_action(request_id, body)
        except ValueError:
            created = False
        if not created:
            # Another claimant owns this attempt: never speak it here.
            self._park(current, "notice_claimed_elsewhere")
            return None
        return NoticeClaim(current, attempt, request_id, claimed_at, late, zone, tuple(outcomes))

    async def _open_for_claim(self, occurrence: Reminder) -> bool:
        """False when the occurrence already ended (advance it now) or another attempt is in flight."""
        attempts = await self._attempts(occurrence)
        if any(row.status in _TERMINAL for row in attempts):
            # A crash or a failed completion left a terminal notice without its advance:
            # complete it silently; on error only the advance is retried, never the speech.
            try:
                await self._complete_terminal(occurrence)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._retry_completion(occurrence, type(error).__name__)
                logger.exception("Reminder %s completion retry failed", occurrence.id)
                return False
            self._holds.pop(occurrence.id, None)
            return False
        if attempts and attempts[-1].status is ReminderActionStatus.SENDING:
            # Claimed by a live announcement elsewhere; restart recovery makes it uncertain.
            self._park(occurrence, "notice_in_flight")
            return False
        return True

    async def _complete_terminal(self, occurrence: Reminder) -> None:
        """Silently complete an occurrence whose notice was already terminal (crash, lost advance).

        User answer Q32: a recurring reminder whose later occurrences were missed
        meanwhile (e.g. a crash while announcing, then a multi-day outage) moves to
        the latest due one, which the next pass announces once as a late notice.
        User answer Q32b: that grouped notice never runs home actions, even when
        the latest occurrence is still inside the grace: every action of the
        grouped occurrence gets its durable late_not_executed barrier, and the
        occurrence its durable grouped mark, BEFORE the reminder moves to it, so
        neither this process nor a restart can run them; a further catch-up
        collapse passes both on (_collapse_missed). Otherwise it advances as usual. The terminal notice itself is never
        spoken again.
        """
        if occurrence.recurrence:
            latest = self._latest_due(occurrence, occurrence.trigger_at, self._clock())
            if latest != occurrence.trigger_at:
                await self._mark_grouped(replace(occurrence, trigger_at=latest))
                if await self._store.update_trigger(occurrence.id, latest, shared_only=True,
                                                    expected_trigger_at=occurrence.trigger_at):
                    logger.info("Recurring reminder %s: missed occurrences after a terminal notice "
                                "grouped into one late notice without home actions", occurrence.id)
                return
        await self._advance(occurrence)

    async def _mark_grouped(self, occurrence: Reminder) -> None:
        """Durably make ``occurrence`` a grouped notice (Q32b) BEFORE the reminder moves to it.

        Every action gets its late_not_executed barrier and the occurrence gets
        its grouped mark; both are idempotent, so a crash or an error before
        the move leaves the reminder where it was and the next pass repeats this.
        """
        for index, action in enumerate(occurrence.ha_actions or []):
            await self._bar_late_action(_action_request_id(occurrence, index), action)
        request_id = grouped_request_id(occurrence.id, occurrence.trigger_at)
        body = {"kind": GROUPED_KIND, "reminder_id": occurrence.id, "trigger_at": occurrence.trigger_at}
        if await self._store.claim_home_action(request_id, body):
            await self._store.finish_home_action(request_id, ReminderActionStatus.CONFIRMED.value, GROUPED_NOTICE)

    async def _is_grouped(self, occurrence: Reminder) -> bool:
        """Whether this occurrence of a recurring reminder carries the durable grouped mark (Q32b)."""
        if not occurrence.recurrence:
            return False
        request_id = grouped_request_id(occurrence.id, occurrence.trigger_at)
        return await self._store.get_home_action_receipt(request_id) is not None

    async def _announce(self, message: str, zone: str) -> tuple[ReminderActionStatus, str | None, bool]:
        """Speak once through the output's aspeak and map its ResponseResult.

        The notice's own output never counts as the output becoming free
        (SIGNALS_OUTPUT_FREE), so its end cannot start another round.

        Returns:
            (status, diagnostic reason, whether it lost the output to other activity).
        """
        callback = getattr(self._tts, "aspeak", None)
        token = SIGNALS_OUTPUT_FREE.set(False)
        try:
            if inspect.iscoroutinefunction(callback):
                result = await callback(message, zone_id=zone)
            else:
                result = await asyncio.to_thread(callback, message, zone_id=zone)
                if inspect.isawaitable(result):
                    result = await result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("Reminder voice output raised %s; notice is uncertain", type(error).__name__)
            return ReminderActionStatus.UNCERTAIN, NOTICE_UNVERIFIED, False
        finally:
            SIGNALS_OUTPUT_FREE.reset(token)
        return (*notice_outcome(result), notice_contended(result))

    async def _finish(self, claim: NoticeClaim, status: ReminderActionStatus, reason: str | None,
                      contended: bool = False) -> bool:
        """Record the attempt's evidence, then act and advance only as that evidence allows.

        Args:
            claim: The claimed attempt.
            status: Terminal or failed status mapped from the voice result.
            reason: Diagnostic reason (display only).
            contended: The attempt lost the output to other activity (notice_contended).
        """
        occurrence = claim.occurrence
        await self._store.finish_home_action(claim.request_id, status.value, reason)
        logger.info("Reminder %s notice attempt %d in zone %s: %s (%s, late=%s)", occurrence.id, claim.attempt,
                    claim.zone, status.value, reason, claim.late)
        if status is ReminderActionStatus.FAILED:
            self._hold_after_failure(occurrence, reason or status.value, claim.claimed_at, refused=not contended)
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
        deadline = self._retry_deadline(hold)
        return deadline is None or deadline > now

    def _backing_off(self, reminder: Reminder, now: float) -> bool:
        """True while this exact occurrence waits for a timed retry (not when parked for the round)."""
        hold = self._holds.get(reminder.id)
        return (hold is not None and hold.trigger_at == reminder.trigger_at
                and hold.retry_at is not None and hold.retry_at > now)

    def _hold(self, reminder: Reminder) -> DeliveryHold:
        hold = self._holds.get(reminder.id)
        if hold is None or hold.trigger_at != reminder.trigger_at:
            hold = self._holds[reminder.id] = DeliveryHold(reminder.trigger_at)
        return hold

    def _hold_after_failure(self, reminder: Reminder, reason: str, failed_at: float, *,
                            refused: bool = False) -> None:
        """Back off 30 s, 60 s, ...; after notice_max_attempts the round ends.

        An exhausted round schedules a fallback round notice_fallback_round_s
        later, at most notice_fallback_max_rounds times per occurrence (Q31);
        after that the occurrence waits for the output to become free, a later
        occurrence, a restart or a web dismiss. When the output became free
        while this attempt was in flight and the attempt lost the output to
        other activity, a new round opens now instead (RV-Q31-2), without
        using a fallback round; during a follow-up conversation that round
        waits for the conversation to end and the timed retry set here still
        applies (Q33). Its bound is always this failure's backoff or fallback
        time, also when the round opens at once and a conversation starts
        before it is claimed (review round 3, Q33-3).

        Args:
            reminder: The occurrence.
            reason: Diagnostic reason (display only).
            failed_at: Claim time of the failed attempt, or the failure time.
            refused: The output itself refused the attempt (see DeliveryHold.output_refused).
        """
        hold = self._hold(reminder)
        if hold.failures >= self._max_attempts:
            hold.failures = 0  # the first failure of a new (fallback) round
        hold.failures += 1
        hold.reason = reason
        hold.output_refused = refused
        hold.output_round = False  # the attempt ended the round it was claimed in
        signalled = reminder.id in self._signal_pending and not refused
        if signalled:
            self._signal_pending.discard(reminder.id)
            if not self._conversation_active():
                # The bound a conversation that opens before the claim would use: this failure's
                # backoff or fallback time, computed now (Q33-3); no fallback round is used (RV-Q31-2).
                hold.retry_at = self._timed_retry(hold, failed_at)
                self._reopen_for_output(hold, self._clock())
                logger.info("Reminder %s not delivered (%s) while the output became free; new round now",
                            reminder.id, reason)
                self._wake_event.set()
                return
        self._schedule_after_failure(hold, reminder, reason, failed_at)
        if signalled:
            # Q33: the new round waits for the follow-up conversation; the backoff just set still applies.
            self._reopen_for_output(hold, self._clock())
            logger.info("Reminder %s not delivered (%s) while the output became free; new round after the "
                        "follow-up conversation", reminder.id, reason)

    def _schedule_after_failure(self, hold: DeliveryHold, reminder: Reminder, reason: str,
                                failed_at: float) -> None:
        """Set the next timed retry of a failed attempt: in-round backoff or a bounded fallback round."""
        retry_at = self._timed_retry(hold, failed_at)
        if hold.failures >= self._max_attempts:
            if retry_at is not None:
                hold.fallback_rounds += 1
                hold.retry_at = retry_at
                logger.warning("Reminder %s not delivered (%s) after %d attempts in this round; fallback round "
                               "%d/%d in %.0f s", reminder.id, reason, hold.failures, hold.fallback_rounds,
                               self._fallback_max_rounds, hold.retry_at - self._clock())
            else:
                hold.retry_at = None
                logger.warning("Reminder %s not delivered (%s) after %d attempts in this round; kept pending "
                               "until the output is free again", reminder.id, reason, hold.failures)
        else:
            hold.retry_at = retry_at
            logger.info("Reminder %s not delivered (%s); attempt %d/%d in %.0f s", reminder.id, reason,
                        hold.failures + 1, self._max_attempts, hold.retry_at - self._clock())

    def _timed_retry(self, hold: DeliveryHold, failed_at: float) -> float | None:
        """Time of the next timed retry after a failure, without using a fallback round.

        In-round backoff (30 s, 60 s, ...) or, once the round is exhausted, the
        fallback round notice_fallback_round_s later while any is left; None
        when only an output-free signal, a later occurrence, a restart or a
        dismiss can retry it.
        """
        if hold.failures < self._max_attempts:
            return failed_at + self._retry_gap * 2 ** (hold.failures - 1)
        if hold.fallback_rounds < self._fallback_max_rounds:
            return failed_at + self._fallback_round_s
        return None

    def _retry_completion(self, reminder: Reminder, reason: str) -> None:
        """Schedule a silent retry of the advance of a terminal notice (30 s, 60 s, ...), bounded per round.

        The retry passes through _open_for_claim, which only advances an
        occurrence with a terminal attempt and parks one still 'sending'.
        """
        hold = self._hold(reminder)
        hold.completion_errors += 1
        hold.reason = reason
        hold.output_refused = False
        hold.output_round = False
        if hold.completion_errors >= self._max_attempts:
            hold.retry_at = None
            logger.warning("Reminder %s completion failed (%s) %d times; retried on the next wake", reminder.id,
                           reason, hold.completion_errors)
        else:
            hold.retry_at = self._clock() + self._retry_gap * 2 ** (hold.completion_errors - 1)
            logger.info("Reminder %s completion failed (%s); silent retry in %.0f s", reminder.id, reason,
                        hold.retry_at - self._clock())

    def _park(self, reminder: Reminder, reason: str) -> None:
        """No further attempt for this occurrence in this round."""
        hold = self._hold(reminder)
        hold.retry_at = None
        hold.reason = reason
        hold.output_refused = False
        hold.output_round = False

    async def _collapse_missed(self, reminder: Reminder, *, grouped: bool = False) -> Reminder | None:
        """Move a recurring reminder to its latest due occurrence so a long outage yields one notice.

        A grouped occurrence (Q32b) passes its durable mark and late barriers to
        the occurrence it collapses into BEFORE the reminder moves, so the one
        notice that also covers the grouped missed occurrences never runs actions,
        also after a restart or a crash between the barrier and the move.
        """
        if not reminder.recurrence:
            return reminder
        latest = self._latest_due(reminder, reminder.trigger_at, self._clock())
        if latest == reminder.trigger_at:
            return reminder
        if grouped:
            await self._mark_grouped(replace(reminder, trigger_at=latest))
        if not await self._store.update_trigger(reminder.id, latest, shared_only=True,
                                                expected_trigger_at=reminder.trigger_at):
            return None
        logger.info("Recurring reminder %s collapsed missed occurrences into one notice", reminder.id)
        return replace(reminder, trigger_at=latest)

    def _next_occurrence(self, reminder: Reminder, after: float) -> float | None:
        """The recurring reminder's occurrence after ``after``, or None (one-shot or past recurrence_end)."""
        if not reminder.recurrence:
            return None
        following = next_trigger(after, reminder.recurrence, zone=self._household_clock.zone)
        if following <= after:
            raise ValueError("reminder_recurrence_not_advancing")
        if reminder.recurrence_end is not None and following > reminder.recurrence_end:
            return None
        return following

    def _latest_due(self, reminder: Reminder, start: float, now: float) -> float:
        """Latest occurrence from ``start`` on that is due by ``now`` (``start`` when none is later)."""
        latest = start
        for _ in range(MAX_CATCH_UP_STEPS):
            following = self._next_occurrence(reminder, latest)
            if following is None or following > now:
                return latest
            latest = following
        raise ValueError("reminder_catch_up_unbounded")

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
            later = next_trigger(following, reminder.recurrence, zone=self._household_clock.zone)
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
            logger.debug("Scheduler loop not available yet, ignoring arrival")
            return
        logger.info("Presence arrival woke the reminder loop")
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
            if current is None:
                return NoticeDismissal.NOT_FOUND
            if (current.state is ReminderState.ACTIVE and current.trigger_at == trigger_at
                    and trigger_at <= self._clock()):
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
            self._end_flight([reminder_id])
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
            text = texts[identity]
            if text is None:
                continue
            attempts = await self._store.probe_ledger(
                lambda attempt, i=identity, t=trigger_at: notice_request_id(i, t, attempt), guard=guard)
            entries.append(NoticeEntry(identity, trigger_at, text, row.status.value, row.reason,
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
