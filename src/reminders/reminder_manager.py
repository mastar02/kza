"""ReminderManager — CRUD, fuzzy cancellation, and voice formatting."""
import logging
from datetime import datetime
from difflib import SequenceMatcher

from src.reminders.reminder_store import ReminderStore, Reminder, ReminderScope
from src.core.sqlite_work import AuthorityGuard
from src.home_assistant.control_policy import ActionOrigin, HomeActor
from src.home_assistant.home_actions import HomeActionService, request_from_service

logger = logging.getLogger(__name__)


class ReminderManager:
    """High-level reminder operations with fuzzy search and voice output."""

    def __init__(self, store: ReminderStore, config: dict | None = None, *, home_actions: HomeActionService | None = None):
        self._store = store
        self._config = config or {}
        self.home_actions = home_actions if home_actions is not None else HomeActionService(None)
        self._tts_prefix = self._config.get("tts_prefix", "Recuerda")
        self._default_time = self._config.get("default_time", "09:00")

    async def create(
        self,
        user_id: str,
        text: str,
        trigger_at: float,
        recurrence: str | None = None,
        ha_actions: list[dict] | None = None,
        *, guard: AuthorityGuard | None = None, scope: ReminderScope = ReminderScope.PRIVATE,
    ) -> Reminder:
        """Create a new reminder via the store."""
        scope = ReminderScope(scope)
        for action in ha_actions or []:
            if type(action) is not dict or action.keys() - {"domain", "service", "entity_id", "data"}:
                raise ValueError("invalid_reminder_action")
            request = request_from_service(
                action.get("domain", ""), action.get("service", ""), action.get("entity_id", ""), action.get("data"),
                request_id="admission", origin=ActionOrigin.REMINDER,
            )
            await self.home_actions.validate_selection(request, actor=HomeActor.household())
        reminder = await self._store.create(
            user_id=user_id,
            text=text,
            trigger_at=trigger_at,
            recurrence=recurrence,
            ha_actions=ha_actions,
            guard=guard,
            scope=scope,
        )
        logger.info(
            "Created reminder '%s' for user %s at %s",
            text,
            user_id,
            datetime.fromtimestamp(trigger_at).strftime("%H:%M"),
        )
        return reminder

    async def cancel_by_id(self, reminder_id: str, *, guard: AuthorityGuard | None = None,
                           shared_only: bool = False) -> bool:
        """Cancel only within the original authority and selected native scope."""
        return await self._store.cancel(reminder_id, guard=guard, shared_only=shared_only)

    async def cancel_by_text(self, user_id: str | None, search_text: str, *,
                             shared_only: bool = False,
                             guard: AuthorityGuard | None = None) -> bool:
        """Apply scope before fuzzy matching and recheck it at cancellation."""
        active = await self._store.get_active_for_user(user_id, shared_only=shared_only, guard=guard)
        if not active:
            return False
        match = self._fuzzy_find(search_text, active)
        if match is None:
            return False
        return await self._store.cancel(match.id, shared_only=shared_only, guard=guard)

    async def get_active(self, user_id: str | None, *, shared_only: bool = False,
                         guard: AuthorityGuard | None = None) -> list[Reminder]:
        """Project eligible active jobs without deriving authority from a profile."""
        return await self._store.get_active_for_user(user_id, shared_only=shared_only, guard=guard)

    async def get_today(self, user_id: str | None, *, shared_only: bool = False,
                        guard: AuthorityGuard | None = None) -> list[Reminder]:
        """Return eligible active jobs before the end of the current day."""
        now = datetime.now()
        cutoff = now.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp()
        active = await self._store.get_active_for_user(user_id, shared_only=shared_only, guard=guard)
        return [reminder for reminder in active if reminder.trigger_at <= cutoff]

    def format_for_voice(self, reminder: Reminder) -> str:
        """Format only explicitly shared material for unauthenticated household speech."""
        if reminder.scope is not ReminderScope.SHARED:
            raise PermissionError("personal_reminder_voice_session_required")
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
