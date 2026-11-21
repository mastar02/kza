"""Shared-list reconciliation with Home Assistant's todo platform."""

import asyncio
import logging
import math
import re
import sqlite3
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum

from src.core.sqlite_work import AuthorityGuard
from src.home_assistant.ha_client import HomeAssistantClient, ServiceCallStatus
from src.lists.list_store import ListItem, ListStore, OwnerType, UserList, _generate_id

logger = logging.getLogger(__name__)
_TODO_ENTITY = re.compile(r"todo\.[a-z0-9_]+")


class TodoStatus(StrEnum):
    PENDING = "needs_action"
    COMPLETED = "completed"
    REMOVED = "removed"


class SyncStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    AUTHORITY_REVOKED = "authority_revoked"
    OUT_OF_SCOPE = "out_of_scope"
    INVALID_CONFIGURATION = "invalid_configuration"
    DISABLED = "disabled"


class ConflictWinner(StrEnum):
    LOCAL = "local"
    HA = "ha"


class ConflictReason(StrEnum):
    TIMESTAMP = "timestamp"
    HA_FALLBACK = "ha_fallback"
    EXPLICIT_LOCAL_ADD = "explicit_local_add"


@dataclass(frozen=True)
class SyncConflict:
    """Outcome of concurrent or initially divergent changes to one item."""

    text: str
    local_status: TodoStatus
    remote_status: TodoStatus
    winner: ConflictWinner
    reason: ConflictReason


@dataclass
class SyncReport:
    """Counts of confirmed effects and conflicts for a reconciliation attempt."""

    status: SyncStatus
    added_local: int = 0
    added_remote: int = 0
    completed_resolved: int = 0
    removed_resolved: int = 0
    conflicts: list[SyncConflict] = field(default_factory=list)


@dataclass(frozen=True)
class _ItemState:
    text: str = ""
    status: TodoStatus = TodoStatus.REMOVED
    changed_at: float | None = None
    ref: str | None = None


_ABSENT = _ItemState()


@dataclass(frozen=True)
class _Resolution:
    local: _ItemState
    remote: _ItemState
    winner: ConflictWinner
    conflict: SyncConflict | None

    @property
    def desired(self) -> _ItemState:
        return self.local if self.winner is ConflictWinner.LOCAL else self.remote


def _normalize(text: str) -> str:
    return text.strip().casefold()


def _is_shared_target(lst: UserList, name: str) -> bool:
    return lst.owner_type is OwnerType.SHARED and _normalize(lst.name) == _normalize(name)


async def _select_shared_list(
    store: ListStore, name: str, guard: AuthorityGuard | None,
) -> UserList | None:
    lists = await store.get_lists_for_user(None, shared_only=True, guard=guard)
    matches = [lst for lst in lists if _is_shared_target(lst, name)]
    return matches[0] if len(matches) == 1 else None


def _timestamp(value: object) -> float | None:
    if type(value) in (int, float):
        try:
            return float(value) if math.isfinite(value) else None
        except OverflowError:
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed.timestamp() if parsed.utcoffset() is not None else None
        except (ValueError, OverflowError, OSError):
            pass
    return None


def _local_state(item: ListItem, added_at: float | None = None) -> _ItemState:
    pending_since = added_at if added_at is not None else item.created_at
    return _ItemState(
        item.text, TodoStatus.COMPLETED if item.completed else TodoStatus.PENDING,
        item.completed_at if item.completed else pending_since, item.id,
    )


def _representative(states: Iterable[_ItemState]) -> _ItemState:
    def order(state: _ItemState) -> tuple:
        timestamp = state.changed_at if state.changed_at is not None else float("-inf")
        return (-timestamp, state.status is not TodoStatus.PENDING, state.ref or "", state.text)

    return min(states, key=order, default=_ABSENT)


def _changed(current: _ItemState, previous: _ItemState) -> bool:
    return (
        current.status is not previous.status
        or (current.ref is not None and previous.ref is not None and current.ref != previous.ref)
        or (current.changed_at is not None and previous.changed_at is not None
            and current.changed_at > previous.changed_at)
        or (current.status is TodoStatus.REMOVED and current.changed_at is not None
            and previous is _ABSENT)
    )


def _resolve(
    local: _ItemState, remote: _ItemState, prior_local: _ItemState, prior_remote: _ItemState,
    *, explicit_add: bool,
) -> _Resolution:
    local_changed = explicit_add or _changed(local, prior_local)
    if local.status is remote.status or not local_changed:
        return _Resolution(local, remote, ConflictWinner.HA, None)

    comparable = local.changed_at is not None and remote.changed_at is not None
    newer_remote = comparable and remote.changed_at > local.changed_at
    remote_changed = _changed(remote, prior_remote) or (explicit_add and newer_remote)
    if not remote_changed:
        return _Resolution(local, remote, ConflictWinner.LOCAL, None)

    if comparable:
        winner = ConflictWinner.LOCAL if local.changed_at > remote.changed_at else ConflictWinner.HA
        reason = ConflictReason.TIMESTAMP
    elif explicit_add:
        winner, reason = ConflictWinner.LOCAL, ConflictReason.EXPLICIT_LOCAL_ADD
    else:
        winner, reason = ConflictWinner.HA, ConflictReason.HA_FALLBACK
    return _Resolution(local, remote, winner, SyncConflict(
        local.text or remote.text, local.status, remote.status, winner, reason,
    ))


def _check_authority(guard: AuthorityGuard | None) -> None:
    if guard is not None:
        try:
            allowed = guard()
        except Exception:
            raise PermissionError("list_sync_authority_unavailable") from None
        if allowed is not True:
            raise PermissionError("list_sync_authority_denied")


class _IncompleteSync(Exception):
    """The selected list or its remote state could not be confirmed."""


class HASyncManager:
    """Reconcile one configured shared list with its explicit HA todo entity.

    The reconciliation snapshot and unpropagated add intents live in process
    memory. After restart, KZA uploads an item again if HA deleted it while
    KZA was down and only its absence is observable. If a reopen never reached
    HA and HA still reports an untimed completion, the first sync restores
    that completion locally. List storage keeps the schema pinned by
    SCHEMA_CONTRACT_EPOCH.

    The legacy one-way methods remain available to callers that explicitly
    supply an entity prefix without a store. ListManager uses reconciliation.
    """

    def __init__(
        self,
        ha_client: HomeAssistantClient | None,
        entity_prefix: str | None = None,
        *,
        store: ListStore | None = None,
        list_name: str = "compras",
        entity_id: str | None = None,
    ) -> None:
        self._ha = ha_client
        self._prefix = entity_prefix if store is None else None
        self._store = store
        self._list_name = _normalize(list_name)
        self._entity = entity_id if isinstance(entity_id, str) and _TODO_ENTITY.fullmatch(entity_id) else None
        self._lock = asyncio.Lock()
        self._local_add_lock = asyncio.Lock()
        self._list_id: str | None = None
        self._last_local: dict[str, _ItemState] = {}
        self._last_remote: dict[str, _ItemState] = {}
        self._outgoing: dict[str, set[_ItemState]] = defaultdict(set)
        self._deletions: dict[tuple[str, str], _ItemState] = {}
        self._local_add_at: dict[str, dict[str, float]] = defaultdict(dict)

    def targets(self, lst: UserList) -> bool:
        """Check whether a local list is eligible for the configured sync."""
        return _is_shared_target(lst, self._list_name)

    async def add_local(
        self, lst: UserList, text: str, *, added_by: str | None,
        shared_only: bool, guard: AuthorityGuard | None,
    ) -> ListItem | None:
        """Deduplicate pending items and reopen a completed match under the guard."""
        key = _normalize(text)

        def match_or_reopen(db: sqlite3.Connection) -> tuple[bool, ListItem | None, float | None]:
            parent = db.execute(
                "SELECT 1 FROM lists WHERE id = ? AND owner_type = 'shared' AND name = ?",
                (lst.id, lst.name),
            ).fetchone()
            if parent is None:
                return False, None, None

            rows = db.execute(
                "SELECT * FROM list_items WHERE list_id = ? ORDER BY completed, created_at, id",
                (lst.id,),
            )
            match = next((row for row in rows if _normalize(row["text"]) == key), None)
            if match is None:
                return True, None, None
            item = self._store._row_to_item(match)
            if not item.completed:
                return True, item, time.time()

            db.execute(
                "UPDATE list_items SET completed = 0, completed_at = NULL WHERE id = ? AND list_id = ?",
                (item.id, lst.id),
            )
            return True, replace(item, completed=False, completed_at=None), time.time()

        def publish(result: tuple[bool, ListItem | None, float | None]) -> None:
            _, item, added_at = result
            if item is not None and added_at is not None and self._entity is not None:
                self._local_add_at[lst.id][item.id] = added_at

        async with self._local_add_lock:
            eligible, item, _ = await self._store._work.write(
                match_or_reopen, guard=guard, publish=publish,
            )
            if item is not None or not eligible:
                return item
            return await self._store.add_item(
                lst.id, text, added_by=added_by, guard=guard, shared_only=shared_only,
            )

    def note_removals(self, lst: UserList, items: Iterable[ListItem]) -> None:
        """Remember mutation times for committed local removals."""
        if not self.targets(lst) or self._entity is None:
            return
        now = time.time()
        for item in items:
            if key := _normalize(item.text):
                self._deletions[lst.id, key] = _ItemState(item.text, changed_at=now)

    def _select_snapshot(self, lst: UserList) -> None:
        if self._list_id != lst.id:
            if self._list_id is not None:
                self._local_add_at.pop(self._list_id, None)
            self._list_id = lst.id
            self._last_local.clear()
            self._last_remote.clear()
            self._outgoing.clear()

    async def sync(
        self, lst: UserList | None = None, *, guard: AuthorityGuard | None = None,
    ) -> SyncReport:
        """Reconcile under the caller's authority and report confirmed effects."""
        if lst is not None and not self.targets(lst):
            return SyncReport(SyncStatus.OUT_OF_SCOPE)
        if self._store is None:
            return SyncReport(SyncStatus.DISABLED)
        if self._entity is None or not self._list_name:
            return SyncReport(SyncStatus.INVALID_CONFIGURATION)
        if self._ha is None:
            return SyncReport(SyncStatus.INCOMPLETE)

        report = SyncReport(SyncStatus.INCOMPLETE)
        async with self._lock:
            try:
                _check_authority(guard)
                selected = await _select_shared_list(self._store, self._list_name, guard)
                if selected is None or (lst is not None and selected.id != lst.id):
                    return SyncReport(SyncStatus.OUT_OF_SCOPE)
                self._select_snapshot(selected)
                await self._reconcile(selected, report, guard)
                report.status = SyncStatus.COMPLETE
            except PermissionError:
                report.status = SyncStatus.AUTHORITY_REVOKED
            except _IncompleteSync:
                logger.debug("Shared list reconciliation incomplete")
        return report

    async def _reconcile(self, lst: UserList, report: SyncReport, guard: AuthorityGuard | None) -> None:
        remote = await self._read_remote(guard)
        items = await self._store.get_items_in_scope(lst.id, shared_only=True, guard=guard)
        if items is None:
            raise _IncompleteSync
        add_times = self._local_add_at.get(lst.id, {}).copy()
        local = defaultdict(list)
        for item in items:
            if key := _normalize(item.text):
                local[key].append(_local_state(item, add_times.get(item.id)))

        deletions = {key: state for (list_id, key), state in self._deletions.items() if lst.id == list_id}
        desired = {}
        for key in sorted(local.keys() | remote.keys() | self._last_local.keys() | deletions.keys()):
            local_state = _representative(local[key]) if key in local else deletions.get(key, _ABSENT)
            states = remote.get(key, ())
            active = [state for state in states if state.status is not TodoStatus.REMOVED]
            remote_state = _representative(active or states)
            prior_remote = self._last_remote.get(key, _ABSENT)
            if any(state.status is remote_state.status and (state.ref is None or state.ref == remote_state.ref)
                   for state in self._outgoing.get(key, ())):
                prior_remote = remote_state
            explicit_add = local_state.status is TodoStatus.PENDING and local_state.ref in add_times
            resolution = _resolve(
                local_state, remote_state, self._last_local.get(key, _ABSENT), prior_remote,
                explicit_add=explicit_add,
            )
            desired[key] = resolution

        remote_snapshot = {}
        for key, resolution in desired.items():
            if resolution.remote.status is resolution.desired.status:
                remote_snapshot[key] = resolution.remote
                continue
            remote_snapshot[key] = await self._apply_remote(
                key, resolution.desired, remote.get(key, ()), report, guard,
            )

        async with self._local_add_lock:
            if self._local_add_at.get(lst.id, {}) != add_times:
                raise _IncompleteSync
            local_snapshot, report.added_local = await self._apply_local(lst, items, local, desired, guard)
            # Consume intents only once both sides have been reconciled.
            current_adds = self._local_add_at.get(lst.id, {})
            for item_id, added_at in add_times.items():
                if current_adds.get(item_id) == added_at:
                    del current_adds[item_id]
        self._last_local = {_normalize(item.text): _local_state(item) for item in local_snapshot
                            if _normalize(item.text)}
        self._last_remote = {key: state for key, state in remote_snapshot.items() if state is not _ABSENT}
        self._outgoing.clear()
        for key, deletion in deletions.items():
            if self._deletions.get((lst.id, key)) is deletion:
                self._deletions.pop((lst.id, key))

        for resolution in desired.values():
            before = resolution.remote if resolution.winner is ConflictWinner.LOCAL else resolution.local
            after = resolution.desired
            if before.status is not after.status:
                if after.status is TodoStatus.REMOVED:
                    report.removed_resolved += 1
                elif TodoStatus.COMPLETED in (before.status, after.status):
                    report.completed_resolved += 1
            if resolution.conflict is not None:
                report.conflicts.append(resolution.conflict)

    async def _read_remote(self, guard: AuthorityGuard | None) -> dict[str, list[_ItemState]]:
        # HA supplies summary, uid and status. Adapters may also supply
        # updated_at and an explicit removed item carrying its deletion time.
        response = await self._call_ha(
            lambda: self._ha.call_service_with_response(
                "todo", "get_items", self._entity,
                data={"status": [TodoStatus.PENDING.value, TodoStatus.COMPLETED.value]},
            ),
            guard,
        )
        if not isinstance(response, dict):
            raise _IncompleteSync
        service_response = response.get("service_response", response)
        payload = service_response.get(self._entity) if isinstance(service_response, dict) else None
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise _IncompleteSync

        result = defaultdict(list)
        refs = set()
        for raw in payload["items"]:
            if not isinstance(raw, dict):
                raise _IncompleteSync
            text = raw.get("summary")
            if not isinstance(text, str) or not (key := _normalize(text)):
                raise _IncompleteSync
            try:
                status = TodoStatus(raw.get("status"))
            except ValueError:
                raise _IncompleteSync from None
            ref = raw.get("uid")
            if ref is not None and (not isinstance(ref, str) or not ref.strip() or ref in refs):
                raise _IncompleteSync
            if ref is not None:
                refs.add(ref)
            result[key].append(_ItemState(text, status, _timestamp(raw.get("updated_at")), ref))
        return result

    async def _apply_remote(
        self, key: str, desired: _ItemState, states: Iterable[_ItemState],
        report: SyncReport, guard: AuthorityGuard | None,
    ) -> _ItemState:
        active = [state for state in states if state.status is not TodoStatus.REMOVED]
        if desired.status is TodoStatus.REMOVED:
            for state in active:
                await self._write("remove_item", {"item": state.ref or state.text}, key, _ABSENT, guard)
            return _ABSENT

        if not active:
            pending = _ItemState(desired.text.strip(), TodoStatus.PENDING)
            await self._write("add_item", {"item": pending.text}, key, pending, guard)
            report.added_remote += 1
            active.append(pending)

        updated = []
        for state in active:
            if state.status is not desired.status:
                state = _ItemState(state.text, desired.status, ref=state.ref)
                await self._write(
                    "update_item", {"item": state.ref or state.text, "status": state.status.value},
                    key, state, guard,
                )
            updated.append(state)
        return _representative(updated)

    async def _apply_local(
        self, lst: UserList, items: list[ListItem], local: dict[str, list[_ItemState]],
        desired: dict[str, _Resolution], guard: AuthorityGuard | None,
    ) -> tuple[list[ListItem], int]:
        # Use the store's existing SQLite owner so reopen, deduplication and
        # imports commit together. Check scope and the full read set at commit.
        def apply(db: sqlite3.Connection) -> tuple[list[ListItem], int]:
            def current_items() -> list[ListItem]:
                rows = db.execute("SELECT * FROM list_items WHERE list_id = ?", (lst.id,))
                return [self._store._row_to_item(row) for row in rows]

            parent = db.execute(
                "SELECT 1 FROM lists WHERE id = ? AND owner_type = 'shared' AND name = ?",
                (lst.id, lst.name),
            ).fetchone()
            original = {item.id: item for item in items}
            if parent is None or {item.id: item for item in current_items()} != original:
                raise _IncompleteSync

            added = 0
            now = time.time()
            for key, resolution in desired.items():
                keep = resolution.local.ref
                target = resolution.desired
                if target.status is TodoStatus.REMOVED:
                    keep = None
                for duplicate in local.get(key, ()):
                    if duplicate.ref != keep:
                        db.execute("DELETE FROM list_items WHERE id = ? AND list_id = ?", (duplicate.ref, lst.id))
                if target.status is TodoStatus.REMOVED or target.status is resolution.local.status:
                    continue
                complete = target.status is TodoStatus.COMPLETED
                completed_at = (target.changed_at if target.changed_at is not None else now) if complete else None
                if keep is None:
                    db.execute(
                        "INSERT INTO list_items (id, list_id, text, completed, created_at, completed_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (_generate_id(), lst.id, target.text.strip(), complete, now, completed_at),
                    )
                    added += 1
                else:
                    db.execute(
                        "UPDATE list_items SET completed = ?, completed_at = ? WHERE id = ? AND list_id = ?",
                        (complete, completed_at, keep, lst.id),
                    )
            return current_items(), added

        return await self._store._work.write(apply, guard=guard)

    async def _write(
        self, action: str, data: dict, key: str, expected: _ItemState, guard: AuthorityGuard | None,
    ) -> None:
        def dispatch() -> None:
            # An uncertain dispatch may have applied. Recognize its observed
            # effect on retry, including the pending step of a completed add.
            self._outgoing[key].add(expected)

        outcome = await self._call_ha(
            lambda: self._ha.call_service_once(
                "todo", action, self._entity, data, guard=guard, on_dispatch=dispatch,
            ),
            guard,
        )
        if outcome.status in (ServiceCallStatus.REJECTED, ServiceCallStatus.NOT_SENT):
            self._outgoing[key].discard(expected)
        if outcome.status is not ServiceCallStatus.ACCEPTED:
            raise _IncompleteSync

    @staticmethod
    async def _call_ha[T](invoke: Callable[[], Awaitable[T]], guard: AuthorityGuard | None) -> T:
        _check_authority(guard)
        try:
            result = await invoke()
        except Exception:
            _check_authority(guard)
            raise _IncompleteSync from None
        _check_authority(guard)
        return result

    def _entity_id(self, list_name: str) -> str:
        """Generate the entity ID for an explicit legacy prefix."""
        safe_name = list_name.lower().replace(" ", "_").replace("-", "_")
        return f"{self._prefix}_{safe_name}"

    async def _legacy_write(
        self, action: str, list_name: str, data: dict, guard: AuthorityGuard | None,
    ) -> bool:
        if self._ha is None or self._prefix is None:
            return False
        entity = self._entity_id(list_name)
        try:
            args = {} if guard is None else {"guard": guard}
            outcome = await self._call_ha(
                lambda: self._ha.call_service_once("todo", action, entity, data, **args), guard,
            )
            return outcome.status is ServiceCallStatus.ACCEPTED
        except (_IncompleteSync, PermissionError):
            return False

    async def sync_add_item(
        self, list_name: str, item_text: str, *, guard: AuthorityGuard | None = None,
    ) -> bool:
        """Send an add through the legacy explicit-prefix adapter."""
        return await self._legacy_write("add_item", list_name, {"item": item_text}, guard)

    async def sync_remove_item(
        self, list_name: str, item_text: str, *, guard: AuthorityGuard | None = None,
    ) -> bool:
        """Send a removal through the legacy explicit-prefix adapter."""
        return await self._legacy_write("remove_item", list_name, {"item": item_text}, guard)

    async def sync_complete_item(
        self, list_name: str, item_text: str, *, guard: AuthorityGuard | None = None,
    ) -> bool:
        """Send a completion through the legacy explicit-prefix adapter."""
        return await self._legacy_write(
            "update_item", list_name, {"item": item_text, "status": "completed"}, guard,
        )
