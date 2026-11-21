"""Business logic for list operations with fuzzy matching and auto-creation."""
import logging
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.core.sqlite_work import AuthorityGuard
from src.core.task_outcome import DispatchState, EffectState, PersistenceState, TaskOutcome, TaskStatus
from src.home_assistant.ha_client import HomeAssistantClient
from src.lists.ha_sync import (
    HASyncManager, SyncReport, SyncStatus, _is_shared_target, _normalize, _select_shared_list,
)
from src.lists.list_store import ListItem, ListStore, OwnerType, UserList

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ListWriteResult:
    """Keep the committed local object separate from optional replication."""

    item: ListItem | None
    outcome: TaskOutcome
    sync_report: SyncReport | None = None


class ListManager:
    """High-level list operations with fuzzy matching and auto-creation.

    Wraps ListStore to provide user-facing list management including
    automatic default list creation, fuzzy item removal, and shared
    list support.
    """

    def __init__(
        self,
        store: ListStore,
        ha_client: HomeAssistantClient | None = None,
        config: dict | None = None,
    ) -> None:
        self._store = store
        config = config or {}
        self._default_list_name: str = config.get("default_list_name", "compras")
        self._ha_sync_enabled = bool(config.get("ha_sync_enabled", False))
        self._ha_sync = (
            HASyncManager(
                ha_client, store=store, list_name=self._default_list_name,
                entity_id=config.get("ha_todo_entity"),
            )
            if self._ha_sync_enabled
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
        item = await self._add_local_item(
            lst.id, item_text, user_id, shared_only=shared_only, guard=guard, target=lst,
        )
        if item is None:
            return ListWriteResult(None, TaskOutcome(TaskStatus.NEEDS_INPUT,
                persistence=PersistenceState.NOT_WRITTEN, effect=EffectState.NOT_OBSERVED))
        report = await self._replicate(lst, guard)
        return ListWriteResult(item, TaskOutcome(TaskStatus.COMPLETED,
            persistence=PersistenceState.COMMITTED, effect=EffectState.OBSERVED,
            observation_scope="software_state", replication=self._dispatch_state(report)), report)

    async def _replicate(self, lst: UserList, guard: AuthorityGuard | None) -> SyncReport | None:
        if self._ha_sync is None or not _is_shared_target(lst, self._default_list_name):
            return None
        try:
            return await self._ha_sync.sync(lst, guard=guard)
        except Exception:
            logger.warning("List committed locally; HA reconciliation remains incomplete")
            return SyncReport(SyncStatus.INCOMPLETE)

    @staticmethod
    def _dispatch_state(report: SyncReport | None) -> DispatchState:
        if report is None:
            return DispatchState.NOT_REQUIRED
        return DispatchState.ACCEPTED if report.status is SyncStatus.COMPLETE else DispatchState.UNCERTAIN

    async def sync_with_ha(self, *, guard: AuthorityGuard | None = None) -> SyncReport:
        """Reconcile the configured shared list, including changes originating in HA."""
        if self._ha_sync is None:
            return SyncReport(SyncStatus.DISABLED)
        return await self._ha_sync.sync(guard=guard)

    async def _sync_target(self, list_id: str | None, guard: AuthorityGuard | None) -> UserList | None:
        if not self._ha_sync_enabled or list_id is None:
            return None
        selected = await _select_shared_list(self._store, self._default_list_name, guard)
        return selected if selected is not None and selected.id == list_id else None

    async def _add_local_item(
        self, list_id: str, text: str, user_id: str | None, *,
        shared_only: bool, guard: AuthorityGuard | None, target: UserList | None,
        owner: str | None = None,
    ) -> ListItem | None:
        added_by = None if shared_only else user_id
        if (target is None or not self._ha_sync_enabled or self._ha_sync is None
                or not _is_shared_target(target, self._default_list_name)):
            return await self._store.add_item(
                list_id, text, added_by=added_by, guard=guard, shared_only=shared_only, owner=owner,
            )

        return await self._ha_sync.add_local(
            target, text, added_by=added_by, shared_only=shared_only, guard=guard,
        )

    async def remove_item(self, user_id: str | None, item_text: str, list_name: str | None = None, *,
                          shared_only: bool = False, guard: AuthorityGuard | None = None) -> bool:
        """Remove a matching eligible item only after a committed native deletion."""
        return (await self.remove_item_with_outcome(user_id, item_text, list_name,
            shared_only=shared_only, guard=guard)).outcome.task_completed

    async def remove_item_with_outcome(self, user_id: str | None, item_text: str,
                                      list_name: str | None = None, *, shared_only: bool = False,
                                      guard: AuthorityGuard | None = None) -> ListWriteResult:
        """Apply scope before fuzzy matching and again before native deletion."""
        lst = await self.get_list(user_id, list_name, shared_only=shared_only, guard=guard)
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
        if self._ha_sync_enabled and self._ha_sync is not None:
            self._ha_sync.note_removals(lst, (match,))
        report = await self._replicate(lst, guard)
        return ListWriteResult(match, TaskOutcome(TaskStatus.COMPLETED,
            persistence=PersistenceState.COMMITTED, effect=EffectState.OBSERVED,
            observation_scope="software_state", replication=self._dispatch_state(report)), report)

    async def get_list(self, user_id: str | None, list_name: str | None = None, *,
                       shared_only: bool = False, guard: AuthorityGuard | None = None) -> UserList | None:
        """Resolve an existing eligible list without implicitly creating one."""
        name = list_name or self._default_list_name
        lst = await self._store.find_list_by_name(name, user_id, shared_only=shared_only, guard=guard)
        if lst is not None or not self._ha_sync_enabled or _normalize(name) != _normalize(self._default_list_name):
            return lst
        if not shared_only and user_id is not None:
            visible = await self._store.get_lists_for_user(user_id, guard=guard)
            for personal in visible:
                if personal.owner_type is OwnerType.USER and _normalize(personal.name) == _normalize(name):
                    return personal
        return await _select_shared_list(self._store, self._default_list_name, guard)

    async def get_items(self, user_id: str | None, list_name: str | None = None, *,
                        shared_only: bool = False, guard: AuthorityGuard | None = None) -> list[ListItem]:
        """Read only items whose parent is eligible at each native projection."""
        lst = await self.get_list(user_id, list_name, shared_only=shared_only, guard=guard)
        if lst is None:
            return []
        return await self._store.get_items(lst.id, shared_only=shared_only, guard=guard)

    async def clear_list(self, user_id: str | None, list_name: str | None = None, *,
                         shared_only: bool = False, guard: AuthorityGuard | None = None) -> bool:
        """Clear only an eligible list after its current scope is rechecked."""
        lst = await self.get_list(user_id, list_name, shared_only=shared_only, guard=guard)
        if lst is None:
            return False
        target = self._ha_sync_enabled and _is_shared_target(lst, self._default_list_name)
        items = await self._store.get_items(lst.id, shared_only=True, guard=guard) if target else []
        cleared = await self._store.clear_list(lst.id, shared_only=shared_only, guard=guard)
        if cleared and target:
            self._ha_sync.note_removals(lst, items)
            await self._replicate(lst, guard)
        return cleared

    async def get_items_by_list_id(self, list_id: str, *, shared_only: bool = False,
                                   guard: AuthorityGuard | None = None) -> list[ListItem]:
        """Read a list by ID with an original authority and optional shared scope."""
        return await self._store.get_items(list_id, shared_only=shared_only, guard=guard)

    async def get_items_in_scope(self, list_id: str, *, shared_only: bool = False, owner: str | None = None,
                                 guard: AuthorityGuard | None = None) -> list[ListItem] | None:
        """Read a list by ID in a session scope; None when the list is not in that scope.

        Args:
            list_id: List identifier.
            shared_only: Only shared lists are in scope.
            owner: Person whose own lists are in scope besides the shared ones.
            guard: Authority guard of the caller.
        """
        return await self._store.get_items_in_scope(list_id, shared_only=shared_only, owner=owner, guard=guard)

    async def add_item_to_list_id(
        self, list_id: str, item_text: str, user_id: str | None = None, *,
        guard: AuthorityGuard | None = None, shared_only: bool = False, owner: str | None = None,
    ) -> ListItem | None:
        """Check the parent and add an item by ID under original authority."""
        target = await self._sync_target(list_id, guard)
        item = await self._add_local_item(
            list_id, item_text, user_id, shared_only=shared_only, guard=guard, owner=owner, target=target,
        )
        if item is not None and target is not None:
            await self._replicate(target, guard)
        return item

    async def remove_item_by_id(self, item_id: str, *, list_id: str | None = None,
                                guard: AuthorityGuard | None = None,
                                shared_only: bool = False, owner: str | None = None) -> bool:
        """Delete an identified item only from an eligible selected list."""
        target = await self._sync_target(list_id, guard)
        items = await self._store.get_items(target.id, shared_only=True, guard=guard) if target else []
        removed = await self._store.remove_item(item_id, list_id=list_id,
            guard=guard, shared_only=shared_only, owner=owner)
        if removed and target is not None:
            self._ha_sync.note_removals(target, (item for item in items if item.id == item_id))
            await self._replicate(target, guard)
        return removed

    async def get_all_lists(self, user_id: str | None, *, shared_only: bool = False,
                            guard: AuthorityGuard | None = None) -> list[UserList]:
        """Project eligible list names without selecting private homonyms first."""
        return await self._store.get_lists_for_user(user_id, shared_only=shared_only, guard=guard)

    async def _resolve_list(self, user_id: str | None, list_name: str | None, *,
                            shared_only: bool = False,
                            guard: AuthorityGuard | None = None) -> UserList | None:
        """Resolve an eligible list; household voice cannot create personal defaults."""
        lst = await self.get_list(user_id, list_name, shared_only=shared_only, guard=guard)
        if lst is not None or shared_only:
            return lst
        return await self._store.create_list(
            name=list_name or self._default_list_name, owner_type=OwnerType.USER, owner_id=user_id, guard=guard,
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
