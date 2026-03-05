"""ReminderManager — CRUD, fuzzy cancellation, and voice formatting."""
import logging
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from src.reminders.reminder_store import ReminderStore, Reminder

logger = logging.getLogger(__name__)


class ReminderManager:
    """High-level reminder operations with fuzzy search and voice output."""

    def __init__(self, store: ReminderStore, config: dict | None = None):
        self._store = store
        self._config = config or {}
        self._tts_prefix = self._config.get("tts_prefix", "Recuerda")
        self._default_time = self._config.get("default_time", "09:00")

    async def create(
        self,
        user_id: str,
        text: str,
        trigger_at: float,
        recurrence: str | None = None,
        ha_actions: list[dict] | None = None,
    ) -> Reminder:
        """Create a new reminder via the store."""
        reminder = await self._store.create(
            user_id=user_id,
            text=text,
            trigger_at=trigger_at,
            recurrence=recurrence,
            ha_actions=ha_actions,
        )
        logger.info(
            "Created reminder '%s' for user %s at %s",
            text,
            user_id,
            datetime.fromtimestamp(trigger_at).strftime("%H:%M"),
        )
        return reminder

    async def cancel_by_text(self, user_id: str, search_text: str) -> bool:
        """Cancel the best fuzzy-matched active reminder. Returns True if cancelled."""
        active = await self._store.get_active_for_user(user_id)
        if not active:
            return False

        match = self._fuzzy_find(search_text, active)
        if match is None:
            return False

        await self._store.cancel(match.id)
        logger.info("Cancelled reminder '%s' (id=%s) via fuzzy match on '%s'", match.text, match.id, search_text)
        return True

    async def get_active(self, user_id: str) -> list[Reminder]:
        """Return all active reminders for a user."""
        return await self._store.get_active_for_user(user_id)

    async def get_today(self, user_id: str) -> list[Reminder]:
        """Return active reminders with trigger_at before end of today."""
        now = datetime.now()
        end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        cutoff = end_of_day.timestamp()

        active = await self._store.get_active_for_user(user_id)
        return [r for r in active if r.trigger_at <= cutoff]

    def format_for_voice(self, reminder: Reminder) -> str:
        """Format a reminder for TTS output.

        Returns:
            String like "Oye, recuerda tomar agua a las 14:30".
        """
        trigger_time = datetime.fromtimestamp(reminder.trigger_at).strftime("%H:%M")
        return f"{self._tts_prefix} {reminder.text} a las {trigger_time}"

    def _fuzzy_find(
        self, query: str, reminders: list[Reminder], threshold: float = 0.4
    ) -> Reminder | None:
        """Find best matching reminder: substring first, then SequenceMatcher.

        Args:
            query: Search text from user.
            reminders: List of active reminders to search.
            threshold: Minimum SequenceMatcher ratio to accept.

        Returns:
            Best matching Reminder or None if no match above threshold.
        """
        query_lower = query.lower()

        # Substring match first (exact containment)
        for r in reminders:
            if query_lower in r.text.lower():
                return r

        # Fallback to SequenceMatcher
        best_match: Reminder | None = None
        best_ratio = 0.0

        for r in reminders:
            ratio = SequenceMatcher(None, query_lower, r.text.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = r

        if best_ratio >= threshold:
            return best_match

        return None
