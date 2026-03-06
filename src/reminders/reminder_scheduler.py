"""Asyncio scheduling and delivery engine for reminders."""
import asyncio
import logging
import time

from src.reminders.recurrence import next_trigger
from src.reminders.reminder_store import Reminder, ReminderStore

logger = logging.getLogger(__name__)


class ReminderScheduler:
    """Schedules and fires reminders using asyncio sleep/wake.

    Fires TTS announcements when reminders are due, executes
    HA actions, handles retries when user is not present, and
    reschedules recurring reminders.
    """

    # TODO: Implement missed_reminder_on_arrival — subscribe to PresenceDetector
    # user_entered_zone events and deliver missed reminders on arrival

    def __init__(
        self,
        store: ReminderStore,
        tts,
        presence_detector,
        ha_client,
        config: dict = None,
    ):
        self._store = store
        self._tts = tts
        self._presence_detector = presence_detector
        self._ha_client = ha_client
        self._config = config or {}
        self._running = False
        self._wake_event = asyncio.Event()
        self._retry_counts: dict[str, int] = {}

        self._max_retries = self._config.get("max_retries", 3)
        self._retry_interval = self._config.get("retry_interval_seconds", 60)
        self._tts_prefix = self._config.get("tts_prefix", "Oye, recuerda")

    async def start(self) -> None:
        """Main scheduler loop: poll for due reminders and fire them."""
        self._running = True
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

        logger.info("ReminderScheduler stopped")

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        self._wake_event.set()
        logger.info("ReminderScheduler stop requested")

    def notify_new_reminder(self) -> None:
        """Wake the scheduler loop to re-evaluate next pending reminder."""
        self._wake_event.set()

    async def _fire_reminder(self, reminder: Reminder) -> None:
        """Fire a single reminder: TTS, HA actions, reschedule or mark fired.

        Args:
            reminder: The reminder to fire.
        """
        zone = self._presence_detector.get_user_zone(reminder.user_id)

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
            await self._tts.speak(message, zone_id=zone)
            logger.info(
                "TTS fired for reminder %s in zone %s", reminder.id, zone
            )

        if reminder.ha_actions:
            for action in reminder.ha_actions:
                domain = action.get("domain", "")
                service = action.get("service", "")
                entity_id = action.get("entity_id", "")
                data = action.get("data", {})
                await self._ha_client.call_service(
                    domain, service, entity_id, data
                )
                logger.info(
                    "HA action executed: %s.%s on %s",
                    domain,
                    service,
                    entity_id,
                )

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
