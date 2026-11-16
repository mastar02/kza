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
from typing import Any

from src.core.voice_delivery import ResponseResult, VoiceDeliveryError, invoke_voice_callback
from src.reminders.recurrence import next_trigger
from src.reminders.reminder_store import Reminder, ReminderState, ReminderStore
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionService

logger = logging.getLogger(__name__)

DEFAULT_LATE_ACTION_GRACE_S = 120.0
LATE_NOT_EXECUTED = "late_not_executed"
# A recurring catch-up never walks further than this many missed occurrences
# (about eleven years of a daily reminder); beyond it the row is held, not replayed.
MAX_CATCH_UP_STEPS = 4096
MAX_LOOP_ERROR_DELAY_S = 60.0


@dataclass
class DeliveryHold:
    """In-process delivery state of one occurrence after a failed attempt.

    Attributes:
        trigger_at: The occurrence the hold belongs to; a new occurrence is never held.
        failures: Failed delivery attempts of this occurrence in this process.
        retry_at: Earliest next attempt, or None when no further in-process attempt is made.
        reason: Last failure reason, for logs and status.
    """

    trigger_at: float
    failures: int = 0
    retry_at: float | None = None
    reason: str = ""


class SpeechNotDelivered(VoiceDeliveryError):
    """The voice output did not confirm the announcement."""

    def __init__(self, reason: str, *, heard: bool):
        super().__init__(reason)
        self.heard = heard


class ReminderScheduler:
    """Schedules and announces shared reminders on the configured output zone.

    Delivery policy (T034-U1):
      * Shared announcements go to the explicit ``output_zone``; a person's BLE
        presence never selects the room. Without a zone nothing is spoken,
        executed or marked fired.
      * An occurrence later than ``late_action_grace_s`` never runs its home
        actions: each action gets a terminal ``failed``/``late_not_executed``
        receipt before anything is spoken, so it can never be sent later.
      * A recurring reminder that missed several occurrences is announced once
        and advanced to its next future occurrence.
      * One delivery per reminder at a time across the loop and arrival tasks.
      * Failed voice delivery is retried with bounded exponential backoff; a
        partially heard announcement is never repeated in this process.
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
        self._arrival_tasks: set[asyncio.Task] = set()

        self._max_retries = self._config.get("max_retries", 3)
        self._retry_interval = self._config.get("retry_interval_seconds", 60)
        self._late_grace = self._config.get("late_action_grace_s", DEFAULT_LATE_ACTION_GRACE_S)
        self._tts_prefix = self._config.get("tts_prefix", "Oye, recuerda")
        self._output_zone = self._config.get("output_zone")
        if type(self._max_retries) is not int or self._max_retries < 0:
            raise ValueError("reminders.max_retries must be a non-negative integer")
        for name, value in (("retry_interval_seconds", self._retry_interval),
                            ("late_action_grace_s", self._late_grace)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"reminders.{name} must be a finite non-negative number")
        if self._retry_interval <= 0:
            raise ValueError("reminders.retry_interval_seconds must be positive")

        # Arrival only re-checks due shared reminders; it never selects their room.
        if self._presence_detector is not None:
            self._presence_detector.on_user_arrived(self._on_user_arrived)

    @property
    def _retry_counts(self) -> dict[str, int]:
        """Failed attempts per held reminder (read by the dashboard status)."""
        return {identity: hold.failures for identity, hold in self._holds.items()}

    async def start(self) -> None:
        """Main scheduler loop: wait for the next due or retryable occurrence and deliver it."""
        if self._running:
            return
        self._running = True
        self._runner_task = asyncio.current_task()
        self._loop = asyncio.get_running_loop()
        logger.info("ReminderScheduler started")
        error_delay = 1.0

        while self._running:
            try:
                self._wake_event.clear()
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
                if timeout is None:
                    await self._wake_event.wait()
                elif timeout > 0:
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

    async def _deliver_due(self, now: float) -> None:
        """Deliver every due occurrence that is not held or already in flight."""
        due = await self._store.get_due(now, shared_only=True)
        current = {reminder.id: reminder.trigger_at for reminder in due}
        self._holds = {identity: hold for identity, hold in self._holds.items()
                       if current.get(identity) == hold.trigger_at}
        for reminder in due:
            if self._is_held(reminder, self._clock()):
                continue
            try:
                await self._fire_reminder(reminder)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Reminder %s delivery failed", reminder.id)

    async def _next_timeout(self) -> float | None:
        """Seconds until the next unheld occurrence or retry, or None to wait for a wake."""
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
        """True while this occurrence is in flight, backing off, or parked until restart."""
        return reminder.id in self._in_flight or self._held_for_occurrence(reminder, now)

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        self._wake_event.set()
        self._loop = None
        tasks = set(self._arrival_tasks)
        if self._runner_task is not None and self._runner_task is not asyncio.current_task():
            tasks.add(self._runner_task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._arrival_tasks.clear()
        logger.info("ReminderScheduler stop requested")

    def notify_new_reminder(self) -> None:
        """Wake the scheduler loop to re-evaluate next pending reminder."""
        self._wake_event.set()

    async def _shared_occurrence(self, reminder: Reminder) -> Reminder | None:
        """Resolve the same active, due shared occurrence without adopting its successor."""
        identity, selected_trigger = reminder.id, reminder.trigger_at
        current = await self._store.get_by_id(identity, shared_only=True)
        if (current is None or current.state is not ReminderState.ACTIVE
                or current.trigger_at != selected_trigger or current.trigger_at > self._clock()):
            return None
        return current

    def _is_late(self, reminder: Reminder) -> bool:
        """An occurrence past the grace window may only be announced."""
        return self._clock() - reminder.trigger_at > self._late_grace

    async def _fire_reminder(self, reminder: Reminder) -> bool:
        """Deliver the selected occurrence once; return True when its announcement was confirmed.

        Single-flight is keyed by the reminder id rather than (id, trigger_at):
        a recurring catch-up moves the occurrence to a new trigger_at while it
        is being delivered, and the new key must not be deliverable meanwhile.
        """
        if reminder.id in self._in_flight:
            return False
        self._in_flight.add(reminder.id)
        try:
            return await self._deliver_occurrence(reminder)
        finally:
            self._in_flight.discard(reminder.id)
            if asyncio.current_task() is not self._runner_task:
                # The loop skipped this reminder while it was in flight.
                self._wake_event.set()

    async def _deliver_occurrence(self, reminder: Reminder) -> bool:
        """Barrier, announce, act and advance one occurrence; hold it after any failure."""
        current, spoken = reminder, False
        try:
            resolved = await self._shared_occurrence(reminder)
            if resolved is None or self._held_for_occurrence(resolved, self._clock()):
                return False
            current = resolved
            collapsed = await self._collapse_missed(current)
            if collapsed is None:
                return False
            current = collapsed
            late = self._is_late(current)
            if late:
                # Durable barrier first: a crash or a failed announcement cannot
                # leave the late actions executable by a later attempt.
                await self._execute_home_actions(current)
            if self._output_zone is None:
                raise SpeechNotDelivered("reminder_output_zone_required", heard=False)
            await self._speak(self._message(current, late), self._output_zone)
            spoken = True
            logger.info("TTS fired for reminder %s in zone %s (late=%s)", current.id, self._output_zone, late)
            self._holds.pop(current.id, None)
            if not late:
                await self._execute_home_actions(current)
            await self._advance(current)
            return True
        except SpeechNotDelivered as failure:
            self._hold_after_failure(current, str(failure), heard=failure.heard)
            return False
        except Exception as error:
            self._hold_after_failure(current, type(error).__name__, heard=spoken)
            raise

    def _held_for_occurrence(self, reminder: Reminder, now: float) -> bool:
        """True while this exact occurrence is backing off or parked until restart."""
        hold = self._holds.get(reminder.id)
        if hold is None or hold.trigger_at != reminder.trigger_at:
            return False
        return hold.retry_at is None or hold.retry_at > now

    def _hold_after_failure(self, reminder: Reminder, reason: str, *, heard: bool) -> None:
        """Back off exponentially; stop in-process attempts after max_retries or any heard audio."""
        hold = self._holds.get(reminder.id)
        if hold is None or hold.trigger_at != reminder.trigger_at:
            hold = self._holds[reminder.id] = DeliveryHold(reminder.trigger_at)
        hold.failures += 1
        hold.reason = reason
        if heard or hold.failures > self._max_retries:
            hold.retry_at = None
            logger.warning("Reminder %s not delivered (%s, heard=%s); kept active, no further attempt "
                           "until restart", reminder.id, reason, heard)
        else:
            hold.retry_at = self._clock() + self._retry_interval * 2 ** (hold.failures - 1)
            logger.info("Reminder %s not delivered (%s); attempt %d/%d in %.0f s", reminder.id, reason,
                        hold.failures, self._max_retries + 1, hold.retry_at - self._clock())

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

    def _message(self, reminder: Reminder, late: bool) -> str:
        message = f"{self._tts_prefix} {reminder.text}"
        if not late:
            return message
        notice = "Este aviso llega con atraso"
        if reminder.ha_actions:
            notice += " y no ejecuté sus acciones programadas"
        return f"{message.rstrip(' .')}. {notice}."

    async def _speak(self, message: str, zone: str) -> None:
        """Announce once; raise SpeechNotDelivered saying whether any segment was heard."""
        callback = getattr(self._tts, "aspeak", None) if self._tts is not None else None
        results: list[Any] = []

        async def observed(*args, **kwargs):
            if inspect.iscoroutinefunction(callback):
                result = await callback(*args, **kwargs)
            else:
                result = await asyncio.to_thread(callback, *args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            results.append(result)
            return result

        try:
            await invoke_voice_callback(observed if callback is not None else None, message, zone_id=zone)
        except Exception as error:
            heard = any(isinstance(result, ResponseResult) and result.spoken_segments > 0 for result in results)
            raise SpeechNotDelivered(str(error) or type(error).__name__, heard=heard) from error

    async def _advance(self, reminder: Reminder) -> None:
        """Mark a one-shot fired or move a recurring reminder past the delivered occurrence."""
        if reminder.recurrence:
            new_trigger_at = next_trigger(reminder.trigger_at, reminder.recurrence)
            if reminder.recurrence_end is not None and new_trigger_at > reminder.recurrence_end:
                if await self._store.mark_fired(reminder.id, shared_only=True,
                                                expected_trigger_at=reminder.trigger_at):
                    logger.info("Recurring reminder %s past recurrence_end, marked fired", reminder.id)
            elif await self._store.update_trigger(reminder.id, new_trigger_at, shared_only=True,
                                                  expected_trigger_at=reminder.trigger_at):
                logger.info("Recurring reminder %s rescheduled to %s", reminder.id, new_trigger_at)
        elif await self._store.mark_fired(reminder.id, shared_only=True, expected_trigger_at=reminder.trigger_at):
            logger.info("One-shot reminder %s marked fired", reminder.id)

    def _on_user_arrived(self, user_id: str, presence: Any) -> None:
        """Callback from PresenceDetector when a user arrives home.

        Schedules a re-check of overdue shared reminders on the event loop.
        The arrival never chooses the announcement room. This callback is
        invoked synchronously from the presence state update loop, so it
        must not block.

        Args:
            user_id: The arriving user.
            presence: UserPresence dataclass (its zone is not used for routing).
        """
        loop = self._loop
        if loop is None:
            logger.debug(
                "Scheduler loop not available yet, skipping missed delivery for %s",
                user_id,
            )
            return

        logger.info("User %s arrived, checking for missed reminders", user_id)
        def launch() -> None:
            if not self._running:
                return
            task = asyncio.create_task(self.missed_reminder_on_arrival(user_id, presence.current_zone))
            self._arrival_tasks.add(task)

            def finished(completed):
                self._arrival_tasks.discard(completed)
                if not completed.cancelled() and completed.exception() is not None:
                    logger.error("Missed reminder voice delivery failed")

            task.add_done_callback(finished)

        try:
            loop.call_soon_threadsafe(launch)
        except RuntimeError:
            logger.debug("Reminder callback arrived after loop shutdown")

    async def missed_reminder_on_arrival(
        self, user_id: str, zone_id: str | None
    ) -> int:
        """Deliver overdue shared reminders when someone arrives home.

        Shared announcements use the configured output zone, never the
        arriving person's room, and go through the same single-flight,
        late-action and backoff policy as the main loop.

        Args:
            user_id: The arriving user (only triggers the re-check).
            zone_id: Zone where the user was detected; not used for routing.

        Returns:
            Number of missed reminders whose announcement was confirmed.
        """
        overdue = await self._store.get_overdue_for_user(user_id, self._clock(), shared_only=True)
        if not overdue:
            return 0
        logger.info("Checking %d missed shared reminders after arrival of %s", len(overdue), user_id)
        delivered = 0
        for reminder in overdue:
            if await self._fire_reminder(reminder):
                delivered += 1
        return delivered

    async def _execute_home_actions(self, reminder: Reminder) -> None:
        """Claim durable effects and apply current policy on every delivery path.

        A late occurrence only records terminal ``failed``/``late_not_executed``
        receipts; an existing receipt (confirmed, failed or uncertain) is never
        overwritten or resent.
        """
        reminder = await self._shared_occurrence(reminder)
        if reminder is None:
            return
        for index, action in enumerate(reminder.ha_actions or []):
            identity = json.dumps([reminder.id, reminder.trigger_at, index], separators=(",", ":"), allow_nan=False)
            request_id = "reminder:" + hashlib.sha256(identity.encode()).hexdigest()
            if self._is_late(reminder):
                if await self._store.claim_home_action(request_id, action):
                    await self._store.finish_home_action(request_id, "failed", LATE_NOT_EXECUTED)
                logger.warning("Late reminder home action not executed")
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
