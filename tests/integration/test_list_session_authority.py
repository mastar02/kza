"""Real dashboard leases guard every native list transaction and its lifecycle."""

import asyncio
from contextlib import AsyncExitStack
import sqlite3
import threading
from types import SimpleNamespace

import aiosqlite
from httpx import ASGITransport, AsyncClient
import pytest

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority
from src.lists.list_manager import ListManager
from src.lists.list_store import ListStore, OwnerType
from tests.mocks.memory_graph import Clock

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def list_graph(tmp_path, dashboard_auth):
    clock = Clock()
    async with AsyncExitStack() as stack:
        authority = DashboardAuthority(tmp_path / 'auth.db', clock=clock, session_ttl_s=60)
        stack.push_async_callback(authority.aclose)
        await authority.initialize()
        path = tmp_path / 'lists.db'
        store = ListStore(str(path))
        stack.push_async_callback(store.close)
        await store.initialize()
        api = DashboardAPI(authority=authority, list_manager=ListManager(store))
        stack.push_async_callback(api.stop)
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://testserver') as client:
            token = dashboard_auth['Authorization'].split(' ', 1)[1]
            assert (await client.post('/api/admin/auth/login', json={'token': token})).status_code == 200
            yield SimpleNamespace(authority=authority, store=store, client=client, path=path, clock=clock, shutdown=stack)


async def seed(graph):
    target = await graph.store.create_list('Synthetic original list', OwnerType.USER, 'u1')
    item = await graph.store.add_item(target.id, 'Synthetic original item', 'u1')
    return target, item


async def mutate(graph, operation, target, item):
    if operation == 'create':
        return await graph.client.post('/api/lists', json={'name': 'Withdrawn synthetic list', 'user_id': 'u1'})
    if operation == 'add':
        return await graph.client.post(f'/api/lists/{target.id}/items', json={'text': 'Withdrawn synthetic item'})
    return await graph.client.delete(f'/api/lists/{target.id}/items/{item.id}')


@pytest.mark.parametrize('operation', ['create', 'add', 'remove'])
@pytest.mark.parametrize('cancel_observer', [False, True])
async def test_session_retired_after_provisional_list_sql_rolls_back_only_its_write(
    list_graph, monkeypatch, dashboard_auth, operation, cancel_observer,
):
    graph = list_graph
    target, item = await seed(graph)
    entered, release = threading.Event(), threading.Event()
    original = graph.store._work.write
    held = False

    async def gated_write(function, *, guard=None):
        def provisional(database):
            nonlocal held
            result = function(database)
            if not held:
                held = True
                entered.set()
                assert release.wait(5)
            return result
        return await original(provisional, guard=guard)

    monkeypatch.setattr(graph.store._work, 'write', gated_write)
    pending = asyncio.create_task(mutate(graph, operation, target, item))
    independent = reader = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if cancel_observer:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
        independent = asyncio.create_task(graph.client.post(f'/api/lists/{target.id}/items',
            json={'text': 'Independent authorized item'}, headers=dashboard_auth))
        reader = asyncio.create_task(graph.store.get_items(target.id))
    finally:
        release.set()
    if not cancel_observer:
        assert (await pending).status_code == 403
    assert (await independent).status_code == 200
    observed = await reader
    assert item.id in {row.id for row in observed}
    assert 'Withdrawn synthetic item' not in {row.text for row in observed}
    assert {row.text for row in await graph.store.get_items(target.id)} == {
        item.text, 'Independent authorized item'}
    assert [row.name for row in await graph.store.get_lists_for_user('u1')] == [target.name]


@pytest.mark.parametrize('operation', ['create', 'add', 'remove', 'complete', 'clear', 'delete'])
async def test_every_list_writer_rechecks_guard_before_native_commit(list_graph, monkeypatch, operation):
    graph = list_graph
    target, item = await seed(graph)
    current = threading.Event()
    current.set()
    original = graph.store._work.write

    async def retire_before_commit(function, *, guard=None):
        def provisional(database):
            result = function(database)
            current.clear()
            return result
        return await original(provisional, guard=guard)

    monkeypatch.setattr(graph.store._work, 'write', retire_before_commit)
    with pytest.raises(PermissionError):
        if operation == 'create':
            await graph.store.create_list('Withdrawn list', OwnerType.USER, 'u1', guard=current.is_set)
        elif operation == 'add':
            await graph.store.add_item(target.id, 'Withdrawn item', guard=current.is_set)
        elif operation == 'remove':
            await graph.store.remove_item(item.id, list_id=target.id, guard=current.is_set)
        elif operation == 'complete':
            await graph.store.complete_item(item.id, list_id=target.id, guard=current.is_set)
        elif operation == 'clear':
            await graph.store.clear_list(target.id, guard=current.is_set)
        else:
            await graph.store.delete_list(target.id, guard=current.is_set)
    assert [row.name for row in await graph.store.get_lists_for_user('u1')] == [target.name]
    assert await graph.store.get_items(target.id) == [item]


async def test_http_item_deletion_cannot_substitute_another_list_id(list_graph):
    graph = list_graph
    target, item = await seed(graph)
    other = await graph.store.create_list('Synthetic other list', OwnerType.USER, 'u1')
    response = await graph.client.delete(f'/api/lists/{other.id}/items/{item.id}')
    assert response.status_code == 404
    assert await graph.store.get_items(target.id) == [item]


async def test_committed_list_write_and_connection_close_remain_owned_after_cancellation(list_graph):
    graph = list_graph
    entered, release = asyncio.Event(), asyncio.Event()
    original = graph.store._work._work._worker

    async def held_worker(function, **kwargs):
        result = await original(function, **kwargs)
        if kwargs.get('write') and kwargs.get('guard') is not None:
            entered.set()
            await release.wait()
        return result

    graph.store._work._work._worker = held_worker
    pending = asyncio.create_task(graph.client.post('/api/lists',
        json={'name': 'Committed synthetic list', 'user_id': 'u1'}))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
        database = graph.store._db
        closing = asyncio.create_task(graph.store.close())
        await asyncio.sleep(0)
        assert not closing.done() and graph.store._db is database
        with pytest.raises(RuntimeError, match='list_store_closing'):
            await graph.store.initialize()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not graph.store._closing.done()
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
    await graph.store.close()
    assert graph.store._db is None
    await graph.store.initialize()
    replacement = graph.store._db
    assert replacement is not database
    assert [row.name for row in await graph.store.get_lists_for_user('u1')] == ['Committed synthetic list']
    await asyncio.sleep(0)
    assert graph.store._db is replacement
    with sqlite3.connect(graph.path) as external:
        assert external.execute('SELECT name FROM lists').fetchone()[0] == 'Committed synthetic list'


async def test_close_waiting_for_initial_connection_cannot_lose_its_retained_owner(
    list_graph, tmp_path, monkeypatch,
):
    store = ListStore(str(tmp_path / 'starting.db'))
    list_graph.shutdown.push_async_callback(store.close)
    entered, release = asyncio.Event(), asyncio.Event()
    original = aiosqlite.connect

    async def held_connect(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(aiosqlite, 'connect', held_connect)
    starting = asyncio.create_task(store.initialize())
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        closing = asyncio.create_task(store.close())
        await asyncio.sleep(0)
        retained = store._closing
        assert retained is not None and not retained.done()
        release.set()
        with pytest.raises(RuntimeError, match='list_store_closing'):
            await starting
        await closing
        assert store._closing is retained and retained.done()
        assert store._db is None and store._work is None
        await store.initialize()
        assert await store.get_lists_for_user('u1') == []
    finally:
        release.set()
        await asyncio.gather(*(t for t in (starting, closing) if t is not None), return_exceptions=True)


async def test_concurrent_initialization_cannot_replace_an_open_connection(list_graph, tmp_path, monkeypatch):
    store = ListStore(str(tmp_path / 'concurrent-start.db'))
    list_graph.shutdown.push_async_callback(store.close)
    entered, release = asyncio.Event(), asyncio.Event()
    original = aiosqlite.connect
    calls = []

    async def held_connect(*args, **kwargs):
        calls.append(True)
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(aiosqlite, 'connect', held_connect)
    starting = asyncio.create_task(store.initialize())
    duplicate = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        duplicate = asyncio.create_task(store.initialize())
        await asyncio.sleep(0)
        release.set()
        await starting
        database = store._db
        with pytest.raises(RuntimeError, match='list_store_already_initialized'):
            await duplicate
        assert calls == [True] and store._db is database
        assert await store.get_lists_for_user('u1') == []
    finally:
        release.set()
        await asyncio.gather(*(t for t in (starting, duplicate) if t is not None), return_exceptions=True)
