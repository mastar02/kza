"""Asyncio scheduling and delivery engine for reminders."""
import asyncio
import hashlib
import json
import logging
import time
from typing import Any

from src.core.voice_delivery import VoiceDeliveryError, invoke_voice_callback
from src.reminders.recurrence import next_trigger
from src.reminders.reminder_store import Reminder, ReminderStore
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionService

logger = logging.getLogger(__name__)


class ReminderScheduler:
    """Schedules and fires reminders using asyncio sleep/wake.

    Fires TTS announcements when reminders are due, executes
    HA actions, handles retries when user is not present, and
    reschedules recurring reminders.

    When a user arrives home (presence detected), any missed
    reminders are delivered immediately via TTS.
    """

    def __init__(
        self,
        store: ReminderStore,
        tts,
        presence_detector,
        ha_client,
        config: dict = None,
        *, home_actions: HomeActionService | None = None,
    ):
        self._store = store
        self._tts = tts
        self._presence_detector = presence_detector
        self._ha_client = ha_client
        self.home_actions = home_actions if home_actions is not None else HomeActionService(ha_client)
        self._config = config or {}
        self._running = False
        self._wake_event = asyncio.Event()
        self._retry_counts: dict[str, int] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner_task: asyncio.Task | None = None
        self._arrival_tasks: set[asyncio.Task] = set()

        self._max_retries = self._config.get("max_retries", 3)
        self._retry_interval = self._config.get("retry_interval_seconds", 60)
        self._tts_prefix = self._config.get("tts_prefix", "Oye, recuerda")

        # Wire presence-based missed reminder delivery
        if self._presence_detector is not None:
            self._presence_detector.on_user_arrived(self._on_user_arrived)

    async def start(self) -> None:
        """Main scheduler loop: poll for due reminders and fire them."""
        if self._running:
            return
        self._running = True
        self._runner_task = asyncio.current_task()
        self._loop = asyncio.get_running_loop()
        logger.info("ReminderScheduler started")

        while self._running:
            try:
                pending = await self._store.get_next_pending()

                if pending is None:
                    self._wake_event.clear()
                    await self._wake_event.wait()
                    continue

                now = time.time()
                delay = pending.trigger_at - now

                if delay > 0:
                    self._wake_event.clear()
                    try:
                        await asyncio.wait_for(
                            self._wake_event.wait(), timeout=delay
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                due = await self._store.get_due(time.time())
                for reminder in due:
                    await self._fire_reminder(reminder)

                # Prevent unbounded growth of retry tracking
                if len(self._retry_counts) > 100:
                    active_ids = {r.id for r in due}
                    self._retry_counts = {
                        k: v for k, v in self._retry_counts.items()
                        if k in active_ids
                    }

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in scheduler loop")
                await asyncio.sleep(1)

        self._running = False
        self._runner_task = None
        self._loop = None
        logger.info("ReminderScheduler stopped")

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

    async def _fire_reminder(self, reminder: Reminder) -> None:
        """Fire a single reminder: TTS, HA actions, reschedule or mark fired.

        Args:
            reminder: The reminder to fire.
        """
        zone = self._presence_detector.get_user_zone(reminder.user_id) if self._presence_detector is not None else self._config.get("output_zone")
        if self._presence_detector is None and zone is None:
            raise VoiceDeliveryError("reminder_output_zone_required")

        if zone is None:
            retries = self._retry_counts.get(reminder.id, 0)
            if retries < self._max_retries:
                self._retry_counts[reminder.id] = retries + 1
                new_trigger = time.time() + self._retry_interval
                await self._store.update_trigger(reminder.id, new_trigger)
                logger.info(
                    "User %s not present, retry %d/%d for reminder %s",
                    reminder.user_id,
                    retries + 1,
                    self._max_retries,
                    reminder.id,
                )
                return
            logger.warning(
                "Max retries reached for reminder %s, firing anyway",
                reminder.id,
            )

        if zone is not None:
            message = f"{self._tts_prefix} {reminder.text}"
            await invoke_voice_callback(self._tts.aspeak, message, zone_id=zone)
            logger.info(
                "TTS fired for reminder %s in zone %s", reminder.id, zone
            )

        await self._execute_home_actions(reminder)

        if reminder.recurrence:
            new_trigger_at = next_trigger(reminder.trigger_at, reminder.recurrence)

            if (
                reminder.recurrence_end is not None
                and new_trigger_at > reminder.recurrence_end
            ):
                await self._store.mark_fired(reminder.id)
                logger.info(
                    "Recurring reminder %s past recurrence_end, marked fired",
                    reminder.id,
                )
            else:
                await self._store.update_trigger(reminder.id, new_trigger_at)
                logger.info(
                    "Recurring reminder %s rescheduled to %s",
                    reminder.id,
                    new_trigger_at,
                )
        else:
            await self._store.mark_fired(reminder.id)
            logger.info("One-shot reminder %s marked fired", reminder.id)

        self._retry_counts.pop(reminder.id, None)

    def _on_user_arrived(self, user_id: str, presence: Any) -> None:
        """Callback from PresenceDetector when a user arrives home.

        Schedules delivery of any overdue reminders on the event loop.
        This callback is invoked synchronously from the presence state
        update loop, so it must not block.

        Args:
            user_id: The arriving user.
            presence: UserPresence dataclass (unused beyond zone).
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
        """Deliver overdue reminders when a user arrives home.

        Finds all active reminders whose trigger_at has passed and
        announces them via TTS in the user's current zone.

        Args:
            user_id: The arriving user.
            zone_id: Zone where the user was detected.

        Returns:
            Number of missed reminders delivered.
        """
        now = time.time()
        overdue = await self._store.get_overdue_for_user(user_id, now)

        if not overdue:
            return 0

        logger.info(
            "Delivering %d missed reminders for user %s in zone %s",
            len(overdue),
            user_id,
            zone_id,
        )

        delivered = 0
        for reminder in overdue:
            target_zone = zone_id or (self._presence_detector.get_user_zone(user_id) if self._presence_detector is not None else self._config.get("output_zone"))
            if not target_zone or self._tts is None:
                raise VoiceDeliveryError("reminder_output_zone_required")
            message = f"{self._tts_prefix} {reminder.text}"
            await invoke_voice_callback(self._tts.aspeak, message, zone_id=target_zone)
            logger.info(
                "Missed reminder %s delivered in zone %s", reminder.id, target_zone,
            )

            await self._execute_home_actions(reminder)

            # Handle recurrence or mark fired
            if reminder.recurrence:
                new_trigger = next_trigger(reminder.trigger_at, reminder.recurrence)
                if (
                    reminder.recurrence_end is not None
                    and new_trigger > reminder.recurrence_end
                ):
                    await self._store.mark_fired(reminder.id)
                else:
                    await self._store.update_trigger(reminder.id, new_trigger)
            else:
                await self._store.mark_fired(reminder.id)

            self._retry_counts.pop(reminder.id, None)
            delivered += 1

        return delivered

    async def _execute_home_actions(self, reminder: Reminder) -> None:
        """Claim durable effects and apply current policy on every delivery path."""
        for index, action in enumerate(reminder.ha_actions or []):
            identity = json.dumps([reminder.id, reminder.trigger_at, index], separators=(",", ":"), allow_nan=False)
            request_id = "reminder:" + hashlib.sha256(identity.encode()).hexdigest()
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
