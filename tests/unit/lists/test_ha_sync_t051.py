"""T051 reconciliation against real SQLite and a loopback HA REST service."""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from src.core.task_outcome import DispatchState, PersistenceState, TaskStatus
from src.home_assistant.ha_client import HomeAssistantClient
from src.lists.ha_sync import ConflictReason, ConflictWinner, SyncStatus, TodoStatus
from src.lists.list_manager import ListManager
from src.lists.list_store import ListStore, OwnerType

ENTITY = "todo.shopping_list"
CONFIG = {"default_list_name": "compras", "ha_sync_enabled": True, "ha_todo_entity": ENTITY}


@dataclass
class TodoServer:
    """Model the REST boundary and retain every call for scope assertions."""

    items: dict[str, dict] = field(default_factory=dict)
    calls: list[tuple[str, dict]] = field(default_factory=list)
    http_error: int | None = None
    no_entity: bool = False
    malformed: bool = False
    fail_action: str | None = None
    pause_action: str | None = None
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    client: HomeAssistantClient | None = None
    next_uid: int = 0

    @property
    def writes(self) -> list[tuple[str, dict]]:
        return [(action, data) for action, data in self.calls if action != "get_items"]

    def put(self, summary: str, status: TodoStatus = TodoStatus.PENDING, **extra) -> str:
        self.next_uid += 1
        uid = str(self.next_uid)
        self.items[uid] = {"uid": uid, "summary": summary, "status": status.value, **extra}
        return uid

    async def handle(self, request: web.Request) -> web.Response:
        action = request.match_info["action"]
        data = await request.json()
        self.calls.append((action, data))
        assert data["entity_id"] == ENTITY
        if self.pause_action == action:
            self.entered.set()
            await self.release.wait()
        if self.http_error is not None:
            return web.Response(status=self.http_error)
        if self.fail_action == action:
            return web.Response(status=503)
        if action == "get_items":
            assert request.query["return_response"] == "true"
            assert data["status"] == [TodoStatus.PENDING.value, TodoStatus.COMPLETED.value]
            if self.no_entity:
                return web.json_response({"service_response": {}, "changed_states": []})
            items = [{"summary": "broken"}] if self.malformed else list(self.items.values())
            return web.json_response({
                "service_response": {ENTITY: {"items": items}}, "changed_states": [],
            })

        if action == "add_item":
            self.put(data["item"])
        else:
            uid = data["item"]
            if uid not in self.items:
                uid = next(key for key, item in self.items.items() if item["summary"] == uid)
            if action == "update_item":
                self.items[uid]["status"] = data["status"]
            elif action == "remove_item":
                del self.items[uid]
            else:
                raise AssertionError(action)
        return web.json_response([])


@pytest_asyncio.fixture
async def ha():
    fake = TodoServer()
    app = web.Application()
    app.router.add_post("/api/services/todo/{action}", fake.handle)
    async with TestServer(app) as server:
        fake.client = HomeAssistantClient(str(server.make_url("/")), "test-token", timeout=1)
        try:
            yield fake
        finally:
            fake.release.set()
            await fake.client.close()


@pytest_asyncio.fixture
async def store(tmp_path):
    result = ListStore(str(tmp_path / "lists.db"))
    await result.initialize()
    try:
        yield result
    finally:
        await result.close()


@pytest_asyncio.fixture
async def shared(store):
    return await store.create_list("  CoMPrAs ", OwnerType.SHARED, owner_id=None)


@pytest.mark.asyncio
async def test_personal_and_other_shared_lists_never_call_ha(store, ha):
    manager = ListManager(store, ha.client, CONFIG)
    personal = await manager.create_list("member", "compras")
    other_shared = await manager.create_list("member", "tareas", shared=True)
    ha.put("remote-only")
    private = await manager.add_item_with_outcome("member", "private item")
    assert private.outcome.replication is DispatchState.NOT_REQUIRED
    assert private.sync_report is None
    assert await manager.get_items("member", shared_only=True) == []
    assert await manager.get_items_in_scope(personal.id, shared_only=True) is None
    assert await manager.add_item_to_list_id(personal.id, "visitor", shared_only=True) is None
    assert not await manager.remove_item_by_id(private.item.id, list_id=personal.id, shared_only=True)
    assert await manager.get_all_lists("member", shared_only=True) == [other_shared]
    assert (await manager.remove_item_with_outcome("member", "private")).outcome.replication is DispatchState.NOT_REQUIRED
    extra = await manager.add_item_to_list_id(personal.id, "personal", "member", owner="member")
    assert await manager.remove_item_by_id(extra.id, list_id=personal.id, owner="member")
    assert await manager.clear_list("member")
    assert await manager.add_item("member", "shared elsewhere", "tareas")
    assert (await manager.sync_with_ha()).status is SyncStatus.OUT_OF_SCOPE
    assert ha.calls == []


@pytest.mark.asyncio
async def test_normalized_default_does_not_route_owned_personal_item_to_shared(store, shared, ha):
    manager = ListManager(store, ha.client, CONFIG)
    personal = await manager.create_list("member", " ComprAs")
    result = await manager.add_item_with_outcome("member", "private")
    assert result.item.list_id == personal.id
    assert result.outcome.replication is DispatchState.NOT_REQUIRED
    assert await manager.get_items("member", shared_only=True) == []
    assert await store.get_items(shared.id) == []
    assert ha.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override", "status"),
    [
        ({"ha_todo_entity": None}, SyncStatus.INVALID_CONFIGURATION),
        ({"ha_todo_entity": ""}, SyncStatus.INVALID_CONFIGURATION),
        ({"ha_todo_entity": "light.kitchen"}, SyncStatus.INVALID_CONFIGURATION),
        ({"ha_todo_entity": "todo.shopping_list, todo.personal"}, SyncStatus.INVALID_CONFIGURATION),
        ({"ha_sync_enabled": False}, SyncStatus.DISABLED),
    ],
)
async def test_invalid_entity_and_disabled_never_call_ha(store, ha, override, status):
    target = await store.create_list("compras", OwnerType.SHARED, owner_id=None)
    manager = ListManager(store, ha.client, CONFIG | override)
    assert await manager.add_item("member", "local")
    assert [item.text for item in await store.get_items(target.id)] == ["local"]
    assert (await manager.sync_with_ha()).status is status
    assert ha.calls == []


@pytest.mark.asyncio
async def test_bidirectional_casefold_deduplication_and_idempotence(store, shared, ha):
    manager = ListManager(store, ha.client, CONFIG)
    await store.add_item(shared.id, "  Leche ")
    await store.add_item(shared.id, "leche")
    await store.add_item(shared.id, "Pan")
    ha.put("lECHe")
    ha.put("  HUEVOS ")
    ha.put("huevos")

    report = await manager.sync_with_ha()
    assert report.status is SyncStatus.COMPLETE
    assert (report.added_local, report.added_remote) == (1, 1)
    assert not report.conflicts
    assert [data["item"] for action, data in ha.writes if action == "add_item"] == ["Pan"]
    items = await store.get_items(shared.id)
    assert sorted(item.text.strip().casefold() for item in items) == ["huevos", "leche", "pan"]

    write_count = len(ha.writes)
    second = await manager.sync_with_ha()
    assert (second.status, second.added_local, second.added_remote) == (SyncStatus.COMPLETE, 0, 0)
    assert (second.completed_resolved, second.removed_resolved, second.conflicts) == (0, 0, [])
    assert len(ha.writes) == write_count

    by_name = await manager.add_item("member", "   pAn  ")
    assert by_name.id == next(item.id for item in items if item.text == "Pan")
    by_id = await manager.add_item_to_list_id(shared.id, "PAN", "member", owner="member")
    assert by_id.id == by_name.id
    assert len(ha.writes) == write_count


@pytest.mark.asyncio
@pytest.mark.parametrize("history", ["initial", "known", "concurrent_timestamp"])
async def test_readding_completed_item_reopens_locally_and_in_ha(store, shared, ha, history):
    manager = ListManager(store, ha.client, CONFIG)
    item = await store.add_item(shared.id, "leche")
    await store.complete_item(item.id)
    metadata = {"updated_at": item.created_at - 60} if history == "concurrent_timestamp" else {}
    uid = ha.put("LECHE", TodoStatus.COMPLETED, **metadata)

    if history != "initial":
        assert (await manager.sync_with_ha()).status is SyncStatus.COMPLETE
    if history == "concurrent_timestamp":
        ha.items[uid]["updated_at"] = datetime.now(UTC).timestamp()

    result = await manager.add_item_with_outcome("member", " Leche ")
    assert result.item.id == item.id and result.item.created_at == item.created_at
    assert not result.item.completed and result.item.completed_at is None
    assert await store.get_items(shared.id) == [result.item]
    assert result.sync_report.status is SyncStatus.COMPLETE
    assert result.sync_report.completed_resolved == 1
    if history == "concurrent_timestamp":
        conflict, = result.sync_report.conflicts
        assert (conflict.winner, conflict.reason) == (ConflictWinner.LOCAL, ConflictReason.TIMESTAMP)
    assert ha.items[uid]["status"] == TodoStatus.PENDING
    assert ha.writes == [("update_item", {"entity_id": ENTITY, "item": uid, "status": TodoStatus.PENDING})]

    second = await manager.sync_with_ha()
    assert (second.status, second.completed_resolved, second.added_remote) == (SyncStatus.COMPLETE, 0, 0)
    assert len(ha.writes) == 1
    assert await store.get_items(shared.id) == [result.item]


@pytest.mark.asyncio
async def test_readding_completed_item_reopens_without_ha_entity(store, shared, ha):
    manager = ListManager(store, ha.client, {"ha_sync_enabled": True})
    item = await store.add_item(shared.id, "leche")
    await store.complete_item(item.id)

    result = await manager.add_item_with_outcome("member", "Leche")
    assert result.item.id == item.id and result.item.created_at == item.created_at
    assert not result.item.completed and result.item.completed_at is None
    assert await store.get_items(shared.id) == [result.item]
    assert result.sync_report.status is SyncStatus.INVALID_CONFIGURATION
    assert ha.calls == []


async def _tick_in_ha_then_readd(store, shared, ha, *, with_snapshot):
    manager = ListManager(store, ha.client, CONFIG)
    item = await manager.add_item(None, "leche", shared_only=True)
    uid, = ha.items
    assert ha.items[uid]["status"] == TodoStatus.PENDING
    ha.items[uid]["status"] = TodoStatus.COMPLETED
    ha.calls.clear()
    if not with_snapshot:
        manager = ListManager(store, ha.client, CONFIG)

    result = await manager.add_item_with_outcome(None, "Leche", shared_only=True)
    assert result.outcome.task_completed
    assert result.item.id == item.id and not result.item.completed
    assert await store.get_items(shared.id) == [result.item]
    assert ha.items[uid]["status"] == TodoStatus.PENDING
    assert ha.writes == [("update_item", {"entity_id": ENTITY, "item": uid, "status": TodoStatus.PENDING})]
    conflict, = result.sync_report.conflicts
    assert (conflict.winner, conflict.reason) == (ConflictWinner.LOCAL, ConflictReason.EXPLICIT_LOCAL_ADD)

    ha.calls.clear()
    assert (await manager.sync_with_ha()).status is SyncStatus.COMPLETE
    assert ha.writes == []

    # Once the explicit add converges, a new completion is a remote change.
    ha.items[uid]["status"] = TodoStatus.COMPLETED
    assert (await manager.sync_with_ha()).status is SyncStatus.COMPLETE
    assert (await store.get_items(shared.id))[0].completed
    assert ha.writes == []


@pytest.mark.asyncio
async def test_ha_tick_without_intermediate_sync_then_readd_with_snapshot(store, shared, ha):
    await _tick_in_ha_then_readd(store, shared, ha, with_snapshot=True)


@pytest.mark.asyncio
async def test_ha_tick_without_intermediate_sync_then_readd_without_snapshot(store, shared, ha):
    await _tick_in_ha_then_readd(store, shared, ha, with_snapshot=False)


@pytest.mark.asyncio
async def test_ha_clear_completed_without_sync_then_readd(store, shared, ha):
    manager = ListManager(store, ha.client, CONFIG)
    uid = ha.put("leche", TodoStatus.COMPLETED)
    assert (await manager.sync_with_ha()).status is SyncStatus.COMPLETE
    completed, = await store.get_items(shared.id)
    assert completed.completed
    del ha.items[uid]
    ha.calls.clear()

    result = await manager.add_item_with_outcome(None, "Leche", shared_only=True)
    assert result.outcome.task_completed
    assert result.item.id == completed.id and not result.item.completed
    assert await store.get_items(shared.id) == [result.item]
    assert [item["status"] for item in ha.items.values()] == [TodoStatus.PENDING]
    assert ha.writes == [("add_item", {"entity_id": ENTITY, "item": "leche"})]
    conflict, = result.sync_report.conflicts
    assert (conflict.winner, conflict.reason) == (ConflictWinner.LOCAL, ConflictReason.EXPLICIT_LOCAL_ADD)

    ha.calls.clear()
    assert (await manager.sync_with_ha()).status is SyncStatus.COMPLETE
    assert ha.writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("remote_status", [TodoStatus.COMPLETED, TodoStatus.REMOVED])
async def test_add_outcome_reflects_a_newer_remote_mutation(store, shared, ha, remote_status):
    manager = ListManager(store, ha.client, CONFIG)
    initial = await manager.add_item(None, "leche", shared_only=True)
    remote, = ha.items.values()
    remote["status"] = remote_status.value
    remote["updated_at"] = datetime.now(UTC).timestamp() + 60

    result = await manager.add_item_with_outcome(None, "Leche", shared_only=True)
    assert not result.outcome.task_completed
    assert result.outcome.status is TaskStatus.UNCERTAIN
    assert result.outcome.persistence is PersistenceState.UNCERTAIN
    conflict, = result.sync_report.conflicts
    assert (conflict.winner, conflict.reason) == (ConflictWinner.HA, ConflictReason.TIMESTAMP)
    items = await store.get_items(shared.id)
    if remote_status is TodoStatus.COMPLETED:
        assert result.item.id == initial.id and result.item.completed
        assert items == [result.item]
    else:
        assert result.item is None and items == []


@pytest.mark.asyncio
async def test_add_by_id_returns_only_a_pending_post_sync_item(store, shared, ha):
    manager = ListManager(store, ha.client, CONFIG)
    item = await manager.add_item_to_list_id(shared.id, "leche", shared_only=True)
    remote, = ha.items.values()
    remote["status"] = TodoStatus.COMPLETED.value
    remote["updated_at"] = datetime.now(UTC).timestamp() + 60

    assert await manager.add_item_to_list_id(shared.id, "Leche", shared_only=True) is None
    effective, = await store.get_items(shared.id)
    assert effective.id == item.id and effective.completed


@pytest.mark.asyncio
async def test_revoked_authority_rolls_back_local_reopen(store, shared, ha, monkeypatch):
    manager = ListManager(store, ha.client, CONFIG)
    item = await store.add_item(shared.id, "leche")
    completed = await store.complete_item(item.id)
    allowed = True
    row_to_item = store._row_to_item

    def revoke_at_match(row):
        nonlocal allowed
        allowed = False
        return row_to_item(row)

    with monkeypatch.context() as patch:
        patch.setattr(store, "_row_to_item", revoke_at_match)
        with pytest.raises(PermissionError):
            await manager.add_item_with_outcome("member", "Leche", guard=lambda: allowed)

    assert await store.get_items(shared.id) == [completed]
    assert ha.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remote_time", "winner", "reason"),
    [
        (None, ConflictWinner.HA, ConflictReason.HA_FALLBACK),
        ("older", ConflictWinner.LOCAL, ConflictReason.TIMESTAMP),
        ("newer", ConflictWinner.HA, ConflictReason.TIMESTAMP),
        ("tie", ConflictWinner.HA, ConflictReason.TIMESTAMP),
    ],
)
async def test_concurrent_completion_and_removal(store, shared, ha, remote_time, winner, reason):
    manager = ListManager(store, ha.client, CONFIG)
    uid = ha.put("Pan")
    assert (await manager.sync_with_ha()).added_local == 1
    pending, = await store.get_items(shared.id)

    # Edits accumulate independently during an outage.
    ha.http_error = 503
    completed = await store.complete_item(pending.id)
    assert (await manager.sync_with_ha()).status is SyncStatus.INCOMPLETE
    ha.http_error = None
    del ha.items[uid]
    if remote_time:
        offset = {"older": -60, "newer": 60, "tie": 0}[remote_time]
        timestamp = completed.completed_at + offset
        if remote_time != "tie":
            timestamp = datetime.fromtimestamp(timestamp, UTC).isoformat()
        ha.put("Pan", TodoStatus.REMOVED, updated_at=timestamp)

    report = await manager.sync_with_ha()
    assert report.status is SyncStatus.COMPLETE
    conflict, = report.conflicts
    assert (conflict.local_status, conflict.remote_status) == (TodoStatus.COMPLETED, TodoStatus.REMOVED)
    assert (conflict.winner, conflict.reason) == (winner, reason)

    if winner is ConflictWinner.LOCAL:
        assert (report.completed_resolved, report.removed_resolved) == (1, 0)
        assert (await store.get_items(shared.id))[0].completed
        assert any(item["status"] == TodoStatus.COMPLETED for item in ha.items.values())
    else:
        assert (report.completed_resolved, report.removed_resolved) == (0, 1)
        assert await store.get_items(shared.id) == []
        assert ha.writes == []


@pytest.mark.asyncio
async def test_reopening_and_nonconflicting_offline_deletion(store, shared, ha):
    manager = ListManager(store, ha.client, CONFIG)
    uid = ha.put("pan", TodoStatus.COMPLETED)
    assert (await manager.sync_with_ha()).completed_resolved == 1
    completed, = await store.get_items(shared.id)
    ha.items[uid]["status"] = TodoStatus.PENDING
    reopened = await manager.sync_with_ha()
    assert (reopened.completed_resolved, reopened.conflicts) == (1, [])
    pending, = await store.get_items(shared.id)
    assert pending.id == completed.id
    assert not pending.completed and pending.completed_at is None

    ha.http_error = 503
    removed = await manager.remove_item_with_outcome("member", "pan")
    assert removed.item.id == pending.id
    assert removed.sync_report.status is SyncStatus.INCOMPLETE
    ha.http_error = None
    report = await manager.sync_with_ha()
    assert (report.removed_resolved, report.conflicts) == (1, [])
    assert not ha.items
    assert await store.get_items(shared.id) == []


@pytest.mark.asyncio
async def test_local_deletion_conflicts_with_remote_completion_without_timestamp(store, shared, ha):
    manager = ListManager(store, ha.client, CONFIG)
    uid = ha.put("pan")
    assert (await manager.sync_with_ha()).status is SyncStatus.COMPLETE
    ha.items[uid]["status"] = TodoStatus.COMPLETED
    result = await manager.remove_item_with_outcome("member", "pan")
    conflict, = result.sync_report.conflicts
    assert (conflict.local_status, conflict.remote_status, conflict.winner) == (
        TodoStatus.REMOVED, TodoStatus.COMPLETED, ConflictWinner.HA,
    )
    assert (await store.get_items(shared.id))[0].completed
    assert ha.writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["503", "unavailable", "malformed", "timeout", "disconnect"])
async def test_failed_read_keeps_local_data_and_retry_recovers(store, shared, ha, failure):
    manager = ListManager(store, ha.client, CONFIG)
    uid = ha.put("already synced")
    assert (await manager.sync_with_ha()).status is SyncStatus.COMPLETE
    before = await store.get_items(shared.id)
    original_client = ha.client
    if failure == "503":
        ha.http_error = 503
    elif failure == "unavailable":
        ha.no_entity = True
    elif failure == "malformed":
        ha.malformed = True
    elif failure == "timeout":
        ha.pause_action = "get_items"

        class ShortReadClient(HomeAssistantClient):
            async def call_service_with_response(self, domain, service, entity_id, **kwargs):
                return await super().call_service_with_response(
                    domain, service, entity_id, timeout=0.05, **kwargs,
                )

        ha.client = ShortReadClient(original_client.url, "test-token")
        manager = ListManager(store, ha.client, CONFIG)
        ha.pause_action = None
        assert (await manager.sync_with_ha()).status is SyncStatus.COMPLETE
        ha.pause_action = "get_items"
    else:
        offline = TestServer(web.Application())
        await offline.start_server()
        # Keep the server socket allocated until the transport for the test is chosen.
        ha.client = HomeAssistantClient(str(offline.make_url("/")), "test-token")
        manager = ListManager(store, ha.client, CONFIG)
        await offline.close()

    try:
        report = await manager.sync_with_ha()
        assert report.status is SyncStatus.INCOMPLETE
        assert await store.get_items(shared.id) == before
        assert ha.writes == []
        assert uid in ha.items
    finally:
        if ha.client is not original_client:
            await ha.client.close()
            ha.client = original_client

    ha.http_error = None
    ha.no_entity = ha.malformed = False
    ha.pause_action = None
    ha.release.set()
    if failure in {"timeout", "disconnect"}:
        manager = ListManager(store, original_client, CONFIG)
    assert (await manager.sync_with_ha()).status is SyncStatus.COMPLETE
    assert await store.get_items(shared.id) == before


@pytest.mark.asyncio
async def test_partial_completed_add_is_retried_without_duplicate_or_local_reopen(store, shared, ha):
    manager = ListManager(store, ha.client, CONFIG)
    item = await store.add_item(shared.id, "pan")
    completed = await store.complete_item(item.id)
    ha.put("new remote item")
    ha.fail_action = "update_item"
    report = await manager.sync_with_ha()
    assert (report.status, report.added_local, report.added_remote) == (SyncStatus.INCOMPLETE, 0, 1)
    assert await store.get_items(shared.id) == [completed]

    ha.fail_action = None
    retried = await manager.sync_with_ha()
    assert (retried.status, retried.added_local, retried.added_remote) == (SyncStatus.COMPLETE, 1, 0)
    assert [action for action, _ in ha.writes] == ["add_item", "update_item", "update_item"]
    assert next(item for item in await store.get_items(shared.id) if item.text == "pan") == completed


@pytest.mark.asyncio
async def test_local_write_during_remote_dispatch_is_preserved(store, shared, ha):
    manager = ListManager(store, ha.client, CONFIG)
    await store.add_item(shared.id, "pan")
    ha.put("huevos")
    ha.pause_action = "add_item"

    task = asyncio.create_task(manager.sync_with_ha())
    try:
        await asyncio.wait_for(ha.entered.wait(), timeout=1)
        await store.add_item(shared.id, "leche")
    finally:
        ha.release.set()
        report = await task
    assert (report.status, report.added_local, report.added_remote) == (SyncStatus.INCOMPLETE, 0, 1)
    assert {item.text for item in await store.get_items(shared.id)} == {"pan", "leche"}

    retried = await manager.sync_with_ha()
    assert (retried.status, retried.added_local, retried.added_remote) == (SyncStatus.COMPLETE, 1, 1)
    assert {item.text for item in await store.get_items(shared.id)} == {"pan", "leche", "huevos"}
    assert [data["item"] for _, data in ha.writes] == ["pan", "leche"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["get_items", "add_item"])
async def test_authority_revoked_during_await_stops_further_calls_and_local_updates(store, shared, ha, action):
    manager = ListManager(store, ha.client, CONFIG)
    local = await store.add_item(shared.id, "pan")
    ha.put("huevos")
    ha.pause_action = action
    allowed = True

    def authority() -> bool:
        return allowed

    task = asyncio.create_task(manager.sync_with_ha(guard=authority))
    try:
        await asyncio.wait_for(ha.entered.wait(), timeout=1)
        allowed = False
    finally:
        ha.release.set()
        report = await task
    assert report.status is SyncStatus.AUTHORITY_REVOKED
    assert await store.get_items(shared.id) == [local]
    assert [name for name, _ in ha.calls] == (["get_items", "add_item"] if action == "add_item" else ["get_items"])
    ha.calls.clear()

    denied = await manager.sync_with_ha(guard=authority)
    assert denied.status is SyncStatus.AUTHORITY_REVOKED
    assert ha.calls == []
