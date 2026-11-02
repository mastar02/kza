"""Original HTTP leases govern native reminders; approved commit owns its wake."""

import asyncio
from contextlib import AsyncExitStack
import logging
import sqlite3
import threading
import time
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient
import pytest

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_store import ReminderActionStatus, ReminderStore
from tests.mocks.memory_graph import Clock

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def reminder_graph(tmp_path, dashboard_auth):
    clock, wake = Clock(), asyncio.Event()
    state = SimpleNamespace(published=0)
    def on_created():
        state.published += 1
        wake.set()
    async with AsyncExitStack() as stack:
        authority = DashboardAuthority(tmp_path / 'auth.db', clock=clock, session_ttl_s=60)
        stack.push_async_callback(authority.aclose)
        await authority.initialize()
        path = tmp_path / 'reminders.db'
        store = ReminderStore(str(path), on_created=on_created)
        stack.push_async_callback(store.close)
        await store.initialize()
        api = DashboardAPI(authority=authority, reminder_manager=ReminderManager(store))
        stack.push_async_callback(api.stop)
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://testserver') as client:
            token = dashboard_auth['Authorization'].split(' ', 1)[1]
            assert (await client.post('/api/admin/auth/login', json={'token': token})).status_code == 200
            yield SimpleNamespace(authority=authority, store=store, client=client, path=path,
                                  clock=clock, wake=wake, state=state)


@pytest.mark.parametrize('operation', ['create', 'cancel'])
@pytest.mark.parametrize('cancel_observer', [False, True])
async def test_retired_reminder_session_rolls_back_only_its_provisional_sql(
    reminder_graph, monkeypatch, dashboard_auth, operation, cancel_observer,
):
    graph = reminder_graph
    original_reminder = await graph.store.create('u1', 'Original synthetic reminder', time.time() + 600)
    graph.wake.clear()
    graph.state.published = 0
    entered, release = threading.Event(), threading.Event()
    original = graph.store._work.write
    held = False

    async def gated_write(function, *, guard=None, publish=None):
        def provisional(database):
            nonlocal held
            result = function(database)
            if not held:
                held = True
                entered.set()
                assert release.wait(5)
            return result
        return await original(provisional, guard=guard, publish=publish)

    monkeypatch.setattr(graph.store._work, 'write', gated_write)
    if operation == 'create':
        request = graph.client.post('/api/reminders', json={'user_id': 'u1',
            'text': 'Withdrawn synthetic reminder', 'trigger_at': time.time() + 600})
    else:
        request = graph.client.delete(f'/api/reminders/{original_reminder.id}')
    pending = asyncio.create_task(request)
    independent = reader = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if cancel_observer:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
        independent = asyncio.create_task(graph.client.post('/api/reminders', headers=dashboard_auth,
            json={'user_id': 'u1', 'text': 'Independent authorized reminder', 'trigger_at': time.time() + 600}))
        reader = asyncio.create_task(graph.store.get_active_for_user('u1'))
    finally:
        release.set()
    if not cancel_observer:
        assert (await pending).status_code == 403
    assert (await independent).status_code == 200
    observed = await reader
    assert original_reminder.id in {row.id for row in observed}
    assert 'Withdrawn synthetic reminder' not in {row.text for row in observed}
    assert {row.text for row in await graph.store.get_active_for_user('u1')} == {
        'Original synthetic reminder', 'Independent authorized reminder'}
    assert graph.state.published == 1 and graph.wake.is_set()


@pytest.mark.parametrize('operation', ['create', 'cancel', 'fired', 'trigger', 'cleanup', 'claim', 'finish'])
async def test_every_reminder_writer_rechecks_original_guard_before_commit(reminder_graph, monkeypatch, operation):
    graph = reminder_graph
    reminder = await graph.store.create('u1', 'Original synthetic reminder', 1)
    if operation == 'cleanup':
        await graph.store.mark_fired(reminder.id)
    action = {'entity_id': 'light.synthetic', 'operation': 'turn_on'}
    await graph.store.claim_home_action('existing-occurrence', action)
    before = await graph.store.get_by_id(reminder.id)
    receipt = await graph.store.get_home_action_receipt('existing-occurrence')
    published = graph.state.published
    current = threading.Event()
    current.set()
    original = graph.store._work.write

    async def retire_before_commit(function, *, guard=None, publish=None):
        def provisional(database):
            result = function(database)
            current.clear()
            return result
        return await original(provisional, guard=guard, publish=publish)

    monkeypatch.setattr(graph.store._work, 'write', retire_before_commit)
    guard = current.is_set
    with pytest.raises(PermissionError):
        if operation == 'create':
            await graph.store.create('u1', 'Withdrawn reminder', 2, guard=guard)
        elif operation == 'cancel':
            await graph.store.cancel(reminder.id, guard=guard)
        elif operation == 'fired':
            await graph.store.mark_fired(reminder.id, guard=guard)
        elif operation == 'trigger':
            await graph.store.update_trigger(reminder.id, 2, guard=guard)
        elif operation == 'cleanup':
            await graph.store.cleanup_old(0, guard=guard)
        elif operation == 'claim':
            await graph.store.claim_home_action('withdrawn-occurrence', action, guard=guard)
        else:
            await graph.store.finish_home_action('existing-occurrence', ReminderActionStatus.UNCERTAIN,
                                                'synthetic', guard=guard)
    assert await graph.store.get_by_id(reminder.id) == before
    assert await graph.store.get_home_action_receipt('existing-occurrence') == receipt
    assert await graph.store.get_home_action_receipt('withdrawn-occurrence') is None
    assert graph.state.published == published
    assert await graph.store._work.read(lambda db: db.execute('SELECT COUNT(*) FROM reminders').fetchone()[0]) == 1


async def test_committed_reminder_wake_remains_owned_after_logout_cancel_and_close(reminder_graph, monkeypatch):
    graph = reminder_graph
    entered, release = asyncio.Event(), asyncio.Event()
    original = graph.store._db._execute
    held = False

    async def after_native_commit(function, *args, **kwargs):
        nonlocal held
        result = await original(function, *args, **kwargs)
        if not held:
            held = True
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(graph.store._db, '_execute', after_native_commit)
    pending = asyncio.create_task(graph.client.post('/api/reminders', json={
        'user_id': 'u1', 'text': 'Committed synthetic reminder', 'trigger_at': time.time() + 600}))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert not graph.wake.is_set()
        with sqlite3.connect(graph.path) as external:
            assert external.execute('SELECT text FROM reminders').fetchone()[0] == 'Committed synthetic reminder'
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
        database = graph.store._db
        closing = asyncio.create_task(graph.store.close())
        await asyncio.sleep(0)
        assert not closing.done() and graph.store._db is database
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not graph.store._closing.done()
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
    await graph.store.close()
    assert graph.wake.is_set() and graph.state.published == 1
    assert graph.store._db is None
    await graph.store.initialize()
    assert len(await graph.store.get_active_for_user('u1')) == 1
    await asyncio.sleep(0)
    assert graph.store._db is not None


async def test_wake_failure_preserves_committed_reminder_and_logs_only_fixed_diagnostic(
    reminder_graph, monkeypatch, caplog,
):
    graph = reminder_graph
    private = 'Synthetic private notifier exception.'
    def failed():
        raise RuntimeError(private)
    monkeypatch.setattr(graph.store, '_on_created', failed)
    caplog.set_level(logging.ERROR, logger='src.reminders.reminder_store')
    reminder = await graph.store.create('u1', 'Synthetic committed text', time.time() + 600)
    assert await graph.store.get_by_id(reminder.id) == reminder
    assert 'reminder_scheduler_wake_failed' in caplog.text and private not in caplog.text
