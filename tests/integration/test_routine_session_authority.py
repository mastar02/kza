"""Real factory leases govern routine commits and cancellation-safe publication."""

import asyncio
from contextvars import ContextVar
from datetime import datetime, timezone
import sqlite3
import threading
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient
import pytest

from src.routines.models import ClaimResult, RoutineConflict, RunStatus, parse_routine_spec
from tests.integration.test_dashboard_sessions import leaves as application_leaves, session_graph as application_graph

pytestmark = pytest.mark.asyncio
leaves = application_leaves
session_graph = application_graph


def definition(name='Synthetic original routine', *, enabled=True, now=None):
    schedule = None if now is None else {'at': now.strftime('%H:%M'), 'weekdays': [now.weekday()]}
    return {'name': name, 'description': '', 'enabled': enabled, 'timezone': 'America/Montevideo',
            'schedule': schedule, 'conditions': [], 'actions': [{'entity_id': 'light.living', 'operation': 'turn_on'}]}


@pytest.fixture
async def routine_graph(session_graph, dashboard_auth):
    scheduler = session_graph.routine_scheduler
    assert scheduler is not None
    async with AsyncClient(transport=ASGITransport(app=session_graph.dashboard.app),
                           base_url='http://testserver') as client:
        token = dashboard_auth['Authorization'].split(' ', 1)[1]
        assert (await client.post('/api/admin/auth/login', json={'token': token})).status_code == 200
        yield SimpleNamespace(graph=session_graph, client=client, scheduler=scheduler, store=scheduler.store)


async def mutate(graph, operation, routine):
    route = f'/api/routines/{routine.routine_id}'
    if operation == 'create':
        return await graph.client.post('/api/routines', json=definition('Withdrawn synthetic routine'))
    if operation == 'update':
        return await graph.client.put(route, json={**definition('Withdrawn synthetic routine'),
                                                  'expected_revision': routine.revision})
    if operation == 'delete':
        return await graph.client.delete(route, params={'expected_revision': routine.revision})
    if operation == 'execute':
        return await graph.client.post(route + '/execute', json={'request_id': 'withdrawn'})
    return await graph.client.post(route + '/' + operation, json={'expected_revision': routine.revision})


@pytest.mark.parametrize('operation', ['create', 'update', 'delete', 'execute', 'enable', 'disable'])
@pytest.mark.parametrize('cancel_observer', [False, True])
async def test_retired_session_rolls_back_provisional_routine_mutation_without_disturbing_another_writer(
    routine_graph, dashboard_auth, monkeypatch, operation, cancel_observer,
):
    graph = routine_graph
    routine = await graph.scheduler.register_routine(parse_routine_spec(definition(enabled=operation != 'enable')),
                                                     actor='synthetic-seed')
    entered, release = threading.Event(), threading.Event()
    original = graph.store._work.write
    held = False

    async def gated_write(function, *, guard=None, publish=None):
        def provisional(database):
            nonlocal held
            result = function(database)
            if guard is not None and not held:
                held = True
                entered.set()
                assert release.wait(5)
            return result
        return await original(provisional, guard=guard, publish=publish)

    monkeypatch.setattr(graph.store._work, 'write', gated_write)
    pending = asyncio.create_task(mutate(graph, operation, routine))
    independent = reader = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if cancel_observer:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
        independent = asyncio.create_task(graph.client.post('/api/routines', headers=dashboard_auth,
            json=definition('Independent authorized routine')))
        reader = asyncio.create_task(graph.store.get(routine.routine_id))
    finally:
        release.set()
    if not cancel_observer:
        assert (await pending).status_code == 403
    assert (await independent).status_code == 200
    assert await reader == routine
    assert await graph.store.get(routine.routine_id) == routine
    assert {row.spec.name for row in await graph.store.list_all()} == {
        routine.spec.name, 'Independent authorized routine'}
    assert await graph.store._work.read(lambda db: db.execute('SELECT COUNT(*) FROM routine_runs').fetchone()[0]) == 0
    assert not graph.scheduler._workers
    assert (await graph.graph.ha_client.get_entity_state('light.living'))['state'] == 'off'


async def prepare_claim(graph, source, monkeypatch):
    now = datetime.now(timezone.utc)
    local_now = now.astimezone(graph.scheduler._zone)
    payload = definition(now=local_now if source == 'scheduled' else None)
    routine = await graph.scheduler.register_routine(parse_routine_spec(payload), actor='synthetic-seed')
    if source == 'scheduled':
        graph.scheduler._boot_at = now.replace(second=0, microsecond=0)
    entered, release = asyncio.Event(), asyncio.Event()
    original = graph.store._db._execute
    held = False

    async def after_native_commit(function, *args, **kwargs):
        nonlocal held
        result = await original(function, *args, **kwargs)
        if isinstance(result, ClaimResult) and not held:
            held = True
            entered.set()
            await release.wait()
        return result

    monkeypatch.setattr(graph.store._db, '_execute', after_native_commit)
    return routine, now, entered, release


def submit_claim(graph, routine, now, source):
    if source == 'scheduled':
        return asyncio.create_task(graph.scheduler._check_time_triggers(now))
    return asyncio.create_task(graph.client.post(f'/api/routines/{routine.routine_id}/execute',
                                               json={'request_id': 'approved'}))


async def terminal_run(graph, run_id):
    async with asyncio.timeout(3):
        while True:
            run = await graph.store.get_run(run_id)
            if run is not None and run.status is not RunStatus.RUNNING and run_id not in graph.scheduler._workers:
                return run
            await asyncio.sleep(.001)


@pytest.mark.parametrize('source', ['manual', 'scheduled'])
async def test_committed_claim_publishes_once_after_observer_cancel_and_logout(
    routine_graph, dashboard_auth, monkeypatch, source,
):
    graph = routine_graph
    routine, now, entered, release = await prepare_claim(graph, source, monkeypatch)
    marker = ContextVar('synthetic_private_routine_caller', default=None)
    execution_contexts, callback_contexts, calls = [], [], []
    original_execute = graph.scheduler.executor.execute_run
    original_callback = graph.scheduler._observe_worker
    original_call = graph.graph.ha_client.call_service_once

    async def execute(run):
        execution_contexts.append(marker.get())
        return await original_execute(run)

    def callback(run_id, task):
        callback_contexts.append(marker.get())
        return original_callback(run_id, task)

    async def call(*args, **kwargs):
        calls.append(True)
        return await original_call(*args, **kwargs)

    monkeypatch.setattr(graph.scheduler.executor, 'execute_run', execute)
    monkeypatch.setattr(graph.scheduler, '_observe_worker', callback)
    monkeypatch.setattr(graph.graph.ha_client, 'call_service_once', call)
    token = marker.set('Synthetic private caller context')
    pending = submit_claim(graph, routine, now, source)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert not graph.scheduler._workers
        with sqlite3.connect(graph.store.db_path) as external:
            run_id, status = external.execute('SELECT run_id, status FROM routine_runs').fetchone()
        assert status == 'running'
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
        release.set()
        run = await terminal_run(graph, run_id)
        assert run.status is RunStatus.SUCCEEDED
        if source == 'scheduled':
            await graph.scheduler._check_time_triggers(now)
        else:
            rejoined = await graph.client.post(f'/api/routines/{routine.routine_id}/execute',
                headers=dashboard_auth, json={'request_id': 'approved'})
            assert rejoined.status_code == 202 and rejoined.json()['run_id'] == run_id
        assert calls == [True]
        assert execution_contexts == [None] and callback_contexts == [None]
    finally:
        marker.reset(token)
        release.set()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize('source', ['manual', 'scheduled'])
async def test_stop_drains_late_claim_publication_before_workers_and_survives_observer_cancel(
    routine_graph, monkeypatch, source,
):
    graph = routine_graph
    routine, now, entered, release = await prepare_claim(graph, source, monkeypatch)
    original_call = graph.graph.ha_client.call_service_once
    calls = []
    async def call(*args, **kwargs):
        calls.append(True)
        return await original_call(*args, **kwargs)
    monkeypatch.setattr(graph.graph.ha_client, 'call_service_once', call)
    pending = submit_claim(graph, routine, now, source)
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        database = graph.store._db
        stopping = asyncio.create_task(graph.scheduler.stop())
        await asyncio.sleep(0)
        assert not stopping.done() and graph.store._db is database
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert graph.scheduler._stop_task is not None and not graph.scheduler._stop_task.done()
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
    await asyncio.wait_for(graph.scheduler.stop(), 3)
    assert graph.store._db is None and not graph.scheduler._workers and calls == []
    with sqlite3.connect(graph.store.db_path) as external:
        assert external.execute('SELECT status FROM routine_runs').fetchone()[0] == 'interrupted'
        assert external.execute('SELECT status FROM routine_run_actions').fetchone()[0] == 'not_executed'
    await graph.scheduler.start()
    assert graph.store._db is not None
    await asyncio.sleep(0)
    assert calls == []


@pytest.mark.parametrize('restart', [False, True])
async def test_stop_requested_during_startup_recovery_cannot_be_overwritten(routine_graph, monkeypatch, restart):
    graph = routine_graph
    if restart:
        await graph.scheduler.stop()
    previous_stop = graph.scheduler._stop_task
    entered, release = asyncio.Event(), asyncio.Event()
    original = graph.store.recover_interrupted

    async def held_recovery():
        result = await original()
        entered.set()
        await release.wait()
        return result

    monkeypatch.setattr(graph.store, 'recover_interrupted', held_recovery)
    starting = asyncio.create_task(graph.scheduler.start())
    stopping = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        stopping = asyncio.create_task(graph.scheduler.stop())
        await asyncio.sleep(0)
        assert graph.scheduler._stop_task is not previous_stop
        release.set()
        with pytest.raises(RoutineConflict, match='scheduler_stopping'):
            await starting
        await asyncio.wait_for(stopping, 3)
        assert graph.scheduler._stopping and not graph.scheduler._running
        assert graph.store._db is None and graph.scheduler._scheduler_task is None
    finally:
        release.set()
        await asyncio.gather(*(t for t in (starting, stopping) if t is not None), return_exceptions=True)
    await graph.scheduler.start()
    assert graph.scheduler._running and not graph.scheduler._stopping
    assert graph.store._db is not None
