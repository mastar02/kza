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
        self, user_id: str | None, list_name: str, shared: bool = False, *,
        guard: AuthorityGuard | None = None, shared_only: bool = False,
    ) -> UserList:
        """Create an explicitly shared or trusted personal list."""
        if shared_only and not shared:
            raise ValueError("shared_list_required")
        owner_type = OwnerType.SHARED if shared else OwnerType.USER
        return await self._store.create_list(
            name=list_name, owner_type=owner_type, owner_id=None if shared_only else user_id,
            guard=guard, shared_only=shared_only)

    async def delete_list(self, user_id: str | None, list_name: str, *,
                          shared_only: bool = False, guard: AuthorityGuard | None = None) -> bool:
        """Delete only a list selected and rechecked within the requested scope."""
        lst = await self._store.find_list_by_name(list_name, user_id,
            shared_only=shared_only, guard=guard)
        if lst is None:
            return False
        return await self._store.delete_list(lst.id, shared_only=shared_only, guard=guard)

    async def add_item(self, user_id: str | None, item_text: str, list_name: str | None = None, *,
                       shared_only: bool = False, guard: AuthorityGuard | None = None) -> ListItem | None:
        """Add locally only after an eligible list and committed item are selected."""
        result = await self.add_item_with_outcome(user_id, item_text, list_name,
            shared_only=shared_only, guard=guard)
        return result.item

    async def add_item_with_outcome(self, user_id: str | None, item_text: str,
                                    list_name: str | None = None, *, shared_only: bool = False,
                                    guard: AuthorityGuard | None = None) -> ListWriteResult:
        """Keep scope rejection separate from a committed local write and replication."""
        lst = await self._resolve_list(user_id, list_name, shared_only=shared_only, guard=guard)
        if lst is None:
            return ListWriteResult(None, TaskOutcome(TaskStatus.NEEDS_INPUT,
                persistence=PersistenceState.NOT_WRITTEN, effect=EffectState.NOT_OBSERVED))
        item = await self._store.add_item(list_id=lst.id, text=item_text,
            added_by=None if shared_only else user_id, shared_only=shared_only, guard=guard)
        if item is None:
            return ListWriteResult(None, TaskOutcome(TaskStatus.NEEDS_INPUT,
                persistence=PersistenceState.NOT_WRITTEN, effect=EffectState.NOT_OBSERVED))
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

    async def remove_item(self, user_id: str | None, item_text: str, list_name: str | None = None, *,
                          shared_only: bool = False, guard: AuthorityGuard | None = None) -> bool:
        """Remove a matching eligible item only after a committed native deletion."""
        return (await self.remove_item_with_outcome(user_id, item_text, list_name,
            shared_only=shared_only, guard=guard)).outcome.task_completed

    async def remove_item_with_outcome(self, user_id: str | None, item_text: str,
                                      list_name: str | None = None, *, shared_only: bool = False,
                                      guard: AuthorityGuard | None = None) -> ListWriteResult:
        """Apply scope before fuzzy matching and again before native deletion."""
        lst = await self._store.find_list_by_name(list_name or self._default_list_name, user_id,
            shared_only=shared_only, guard=guard)
        items = await self._store.get_items(lst.id, shared_only=shared_only, guard=guard) if lst is not None else []
        match = self._fuzzy_find(item_text, items)
        if match is None:
            return ListWriteResult(None, TaskOutcome(TaskStatus.NEEDS_INPUT,
                persistence=PersistenceState.NOT_WRITTEN, effect=EffectState.NOT_OBSERVED))
        removed = await self._store.remove_item(match.id, list_id=lst.id,
            shared_only=shared_only, guard=guard)
        if removed is not True:
            return ListWriteResult(None, TaskOutcome(TaskStatus.NEEDS_INPUT,
                persistence=PersistenceState.NOT_WRITTEN, effect=EffectState.NOT_OBSERVED))
        replication = await self._replicate("sync_remove_item", lst.name, match.text)
        return ListWriteResult(match, TaskOutcome(TaskStatus.COMPLETED,
            persistence=PersistenceState.COMMITTED, effect=EffectState.OBSERVED,
            observation_scope="software_state", replication=replication))

    async def get_list(self, user_id: str | None, list_name: str | None = None, *,
                       shared_only: bool = False, guard: AuthorityGuard | None = None) -> UserList | None:
        """Resolve an existing eligible list without implicitly creating one."""
        return await self._store.find_list_by_name(list_name or self._default_list_name, user_id,
            shared_only=shared_only, guard=guard)

    async def get_items(self, user_id: str | None, list_name: str | None = None, *,
                        shared_only: bool = False, guard: AuthorityGuard | None = None) -> list[ListItem]:
        """Read only items whose parent is eligible at each native projection."""
        lst = await self._store.find_list_by_name(list_name or self._default_list_name, user_id,
            shared_only=shared_only, guard=guard)
        if lst is None:
            return []
        return await self._store.get_items(lst.id, shared_only=shared_only, guard=guard)

    async def clear_list(self, user_id: str | None, list_name: str | None = None, *,
                         shared_only: bool = False, guard: AuthorityGuard | None = None) -> bool:
        """Clear only an eligible list after its current scope is rechecked."""
        lst = await self._store.find_list_by_name(list_name or self._default_list_name, user_id,
            shared_only=shared_only, guard=guard)
        if lst is None:
            return False
        return await self._store.clear_list(lst.id, shared_only=shared_only, guard=guard)

    async def get_items_by_list_id(self, list_id: str, *, shared_only: bool = False,
                                   guard: AuthorityGuard | None = None) -> list[ListItem]:
        """Read a list by ID with an original authority and optional shared scope."""
        return await self._store.get_items(list_id, shared_only=shared_only, guard=guard)

    async def add_item_to_list_id(
        self, list_id: str, item_text: str, user_id: str | None = None, *,
        guard: AuthorityGuard | None = None, shared_only: bool = False,
    ) -> ListItem | None:
        """Check the parent and add an item by ID under original authority."""
        return await self._store.add_item(list_id, item_text,
            added_by=None if shared_only else user_id, guard=guard, shared_only=shared_only)

    async def remove_item_by_id(self, item_id: str, *, list_id: str | None = None,
                                guard: AuthorityGuard | None = None,
                                shared_only: bool = False) -> bool:
        """Delete an identified item only from an eligible selected list."""
        return await self._store.remove_item(item_id, list_id=list_id,
            guard=guard, shared_only=shared_only)

    async def get_all_lists(self, user_id: str | None, *, shared_only: bool = False,
                            guard: AuthorityGuard | None = None) -> list[UserList]:
        """Project eligible list names without selecting private homonyms first."""
        return await self._store.get_lists_for_user(user_id, shared_only=shared_only, guard=guard)

    async def _resolve_list(self, user_id: str | None, list_name: str | None, *,
                            shared_only: bool = False,
                            guard: AuthorityGuard | None = None) -> UserList | None:
        """Resolve an eligible list; household voice cannot create personal defaults."""
        name = list_name or self._default_list_name
        lst = await self._store.find_list_by_name(name, user_id, shared_only=shared_only, guard=guard)
        if lst is not None or shared_only:
            return lst
        return await self._store.create_list(name=name, owner_type=OwnerType.USER, owner_id=user_id,
            guard=guard)

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
