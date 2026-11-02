"""Business logic for list operations with fuzzy matching and auto-creation."""
from dataclasses import dataclass
import logging
from difflib import SequenceMatcher

from src.lists.list_store import ListStore, UserList, ListItem, OwnerType
from src.lists.ha_sync import HASyncManager
from src.core.task_outcome import DispatchState, EffectState, PersistenceState, TaskOutcome, TaskStatus
from src.core.sqlite_work import AuthorityGuard

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ListWriteResult:
    """Keep the committed local object separate from optional replication."""

    item: ListItem | None
    outcome: TaskOutcome


class ListManager:
    """High-level list operations with fuzzy matching and auto-creation.

    Wraps ListStore to provide user-facing list management including
    automatic default list creation, fuzzy item removal, and shared
    list support.
    """

    def __init__(
        self,
        store: ListStore,
        ha_client=None,
        config: dict | None = None,
    ):
        self._store = store
        self._ha_client = ha_client
        self._config = config or {}
        self._default_list_name: str = self._config.get(
            "default_list_name", "compras"
        )
        self._ha_sync: HASyncManager | None = (
            HASyncManager(ha_client, self._config.get("ha_entity_prefix", "todo.kza"))
            if ha_client and self._config.get("ha_sync_enabled", False)
            else None
        )

    async def create_list(
        self, user_id: str, list_name: str, shared: bool = False, *, guard: AuthorityGuard | None = None,
    ) -> UserList:
        """Create a new list for a user or as shared.

        Args:
            user_id: Owner of the list.
            list_name: Name for the new list.
            shared: If True, the list is visible to all users.

        Returns:
            The newly created UserList.
        """
        owner_type = OwnerType.SHARED if shared else OwnerType.USER
        lst = await self._store.create_list(
            name=list_name, owner_type=owner_type, owner_id=user_id, guard=guard,
        )
        logger.info(
            "Created list '%s' for user=%s (shared=%s)", list_name, user_id, shared
        )
        return lst

    async def delete_list(self, user_id: str, list_name: str) -> bool:
        """Delete a list by name.

        Args:
            user_id: User requesting the deletion.
            list_name: Name of the list to delete.

        Returns:
            True if the list was found and deleted, False otherwise.
        """
        lst = await self._store.find_list_by_name(list_name, user_id)
        if lst is None:
            logger.warning(
                "Delete failed: list '%s' not found for user=%s", list_name, user_id
            )
            return False
        if await self._store.delete_list(lst.id) is not True:
            return False
        logger.info("Deleted list '%s' for user=%s", list_name, user_id)
        return True

    async def add_item(self, user_id: str, item_text: str, list_name: str | None = None) -> ListItem:
        """Add locally; remote replication cannot erase a committed write."""
        result = await self.add_item_with_outcome(user_id, item_text, list_name)
        return result.item

    async def add_item_with_outcome(self, user_id: str, item_text: str,
                                    list_name: str | None = None) -> ListWriteResult:
        """Retain the committed item even when optional HA replication fails."""
        lst = await self._resolve_list(user_id, list_name)
        item = await self._store.add_item(list_id=lst.id, text=item_text, added_by=user_id)
        replication = await self._replicate("sync_add_item", lst.name, item_text)
        return ListWriteResult(item, TaskOutcome(TaskStatus.COMPLETED,
            persistence=PersistenceState.COMMITTED, effect=EffectState.OBSERVED,
            observation_scope="software_state", replication=replication))

    async def _replicate(self, method: str, *args) -> DispatchState:
        if self._ha_sync is None:
            return DispatchState.NOT_REQUIRED
        try:
            accepted = await getattr(self._ha_sync, method)(*args)
            return DispatchState.ACCEPTED if accepted is True else DispatchState.UNCERTAIN
        except Exception:
            logger.warning("List committed locally; HA replication remains uncertain")
            return DispatchState.UNCERTAIN

    async def remove_item(self, user_id: str, item_text: str, list_name: str | None = None) -> bool:
        """Remove a matched item; missing targets never count as a mutation."""
        return (await self.remove_item_with_outcome(user_id, item_text, list_name)).outcome.task_completed

    async def remove_item_with_outcome(self, user_id: str, item_text: str,
                                      list_name: str | None = None) -> ListWriteResult:
        """Separate local deletion from optional HA replication."""
        lst = await self._store.find_list_by_name(list_name or self._default_list_name, user_id)
        items = await self._store.get_items(lst.id) if lst is not None else []
        match = self._fuzzy_find(item_text, items)
        if match is None:
            return ListWriteResult(None, TaskOutcome(TaskStatus.NEEDS_INPUT,
                persistence=PersistenceState.NOT_WRITTEN, effect=EffectState.NOT_OBSERVED))
        removed = await self._store.remove_item(match.id)
        if removed is not True:
            return ListWriteResult(None, TaskOutcome(TaskStatus.NEEDS_INPUT,
                persistence=PersistenceState.NOT_WRITTEN, effect=EffectState.NOT_OBSERVED))
        replication = await self._replicate("sync_remove_item", lst.name, match.text)
        return ListWriteResult(match, TaskOutcome(TaskStatus.COMPLETED,
            persistence=PersistenceState.COMMITTED, effect=EffectState.OBSERVED,
            observation_scope="software_state", replication=replication))

    async def get_items(
        self, user_id: str, list_name: str | None = None
    ) -> list[ListItem]:
        """Get all items in a list.

        Args:
            user_id: User requesting items.
            list_name: Target list name. Uses default if None.

        Returns:
            List of ListItem objects.
        """
        lst = await self._store.find_list_by_name(
            list_name or self._default_list_name, user_id
        )
        if lst is None:
            return []
        return await self._store.get_items(lst.id)

    async def clear_list(self, user_id: str, list_name: str | None = None) -> bool:
        """Remove items only when the requested list exists and commit succeeds."""
        lst = await self._store.find_list_by_name(list_name or self._default_list_name, user_id)
        if lst is None or await self._store.clear_list(lst.id) is not True:
            return False
        logger.info("Cleared list '%s' for user=%s", lst.name, user_id)
        return True

    async def get_items_by_list_id(self, list_id: str) -> list[ListItem]:
        """Get items for a specific list by ID."""
        return await self._store.get_items(list_id)

    async def add_item_to_list_id(
        self, list_id: str, item_text: str, user_id: str | None = None, *, guard: AuthorityGuard | None = None,
    ) -> ListItem:
        """Add item to a specific list by ID."""
        return await self._store.add_item(list_id, item_text, added_by=user_id, guard=guard)

    async def remove_item_by_id(self, item_id: str, *, list_id: str | None = None,
                                guard: AuthorityGuard | None = None) -> bool:
        """Return whether the identified item was actually removed."""
        return await self._store.remove_item(item_id, list_id=list_id, guard=guard)

    async def get_all_lists(self, user_id: str) -> list[UserList]:
        """Get all lists accessible to a user.

        Args:
            user_id: User requesting their lists.

        Returns:
            List of UserList objects (owned + shared).
        """
        return await self._store.get_lists_for_user(user_id)

    async def _resolve_list(
        self, user_id: str, list_name: str | None
    ) -> UserList:
        """Find a list by name or auto-create the default list.

        Args:
            user_id: User context for the lookup.
            list_name: List name to find, or None for default.

        Returns:
            The resolved UserList.
        """
        name = list_name or self._default_list_name
        lst = await self._store.find_list_by_name(name, user_id)
        if lst is not None:
            return lst
        logger.info(
            "Auto-creating list '%s' for user=%s", name, user_id
        )
        return await self._store.create_list(
            name=name, owner_type=OwnerType.USER, owner_id=user_id
        )

    @staticmethod
    def _fuzzy_find(
        query: str,
        items: list[ListItem],
        threshold: float = 0.5,
    ) -> ListItem | None:
        """Find the best matching item using substring then SequenceMatcher.

        Tries exact substring match first (case-insensitive). If no
        substring match is found, falls back to SequenceMatcher with
        the given threshold.

        Args:
            query: Text to search for.
            items: List of ListItem to search through.
            threshold: Minimum similarity ratio for SequenceMatcher.

        Returns:
            The best matching ListItem or None if no match found.
        """
        query_lower = query.lower()

        # Substring match first
        for item in items:
            if query_lower in item.text.lower():
                return item

        # Fallback to SequenceMatcher
        best_match: ListItem | None = None
        best_ratio = threshold
        for item in items:
            ratio = SequenceMatcher(
                None, query_lower, item.text.lower()
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = item

        return best_match
