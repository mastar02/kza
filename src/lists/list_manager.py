"""Business logic for list operations with fuzzy matching and auto-creation."""
import logging
from difflib import SequenceMatcher

from src.lists.list_store import ListStore, UserList, ListItem
from src.lists.ha_sync import HASyncManager

logger = logging.getLogger(__name__)


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
        self, user_id: str, list_name: str, shared: bool = False
    ) -> UserList:
        """Create a new list for a user or as shared.

        Args:
            user_id: Owner of the list.
            list_name: Name for the new list.
            shared: If True, the list is visible to all users.

        Returns:
            The newly created UserList.
        """
        owner_type = "shared" if shared else "user"
        lst = await self._store.create_list(
            name=list_name, owner_type=owner_type, owner_id=user_id
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
        await self._store.delete_list(lst.id)
        logger.info("Deleted list '%s' for user=%s", list_name, user_id)
        return True

    async def add_item(
        self,
        user_id: str,
        item_text: str,
        list_name: str | None = None,
    ) -> ListItem:
        """Add an item to a list, auto-creating the default list if needed.

        Args:
            user_id: User adding the item.
            item_text: Text of the item to add.
            list_name: Target list name. Uses default if None.

        Returns:
            The newly created ListItem.
        """
        lst = await self._resolve_list(user_id, list_name)
        item = await self._store.add_item(
            list_id=lst.id, text=item_text, added_by=user_id
        )
        logger.info(
            "Added item '%s' to list '%s' (user=%s)", item_text, lst.name, user_id
        )
        if self._ha_sync:
            await self._ha_sync.sync_add_item(lst.name, item_text)
        return item

    async def remove_item(
        self,
        user_id: str,
        item_text: str,
        list_name: str | None = None,
    ) -> bool:
        """Remove an item using fuzzy matching.

        Tries substring match first, then falls back to SequenceMatcher.

        Args:
            user_id: User requesting removal.
            item_text: Query text to match against item texts.
            list_name: Target list name. Uses default if None.

        Returns:
            True if an item was found and removed, False otherwise.
        """
        lst = await self._resolve_list(user_id, list_name)
        items = await self._store.get_items(lst.id)
        match = self._fuzzy_find(item_text, items)
        if match is None:
            logger.info(
                "No fuzzy match for '%s' in list '%s'", item_text, lst.name
            )
            return False
        matched_text = match.text
        await self._store.remove_item(match.id)
        logger.info(
            "Removed item '%s' (matched '%s') from list '%s'",
            matched_text,
            item_text,
            lst.name,
        )
        if self._ha_sync:
            await self._ha_sync.sync_remove_item(lst.name, matched_text)
        return True

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

    async def clear_list(
        self, user_id: str, list_name: str | None = None
    ) -> None:
        """Remove all items from a list.

        Args:
            user_id: User requesting the clear.
            list_name: Target list name. Uses default if None.
        """
        lst = await self._store.find_list_by_name(
            list_name or self._default_list_name, user_id
        )
        if lst is None:
            return
        await self._store.clear_list(lst.id)
        logger.info("Cleared list '%s' for user=%s", lst.name, user_id)

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
            name=name, owner_type="user", owner_id=user_id
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
