"""Authenticated review authority at real SQLite transaction boundaries."""

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
import threading
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
import pytest

from src.dashboard.api import DashboardAPI
from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import DashboardAuthority, DashboardPrincipal
from src.memory.fact_ledger import FactState, MemoryReviewer
from src.memory.services import build_memory_services
from tests.mocks.memory_graph import Clock, MemoryClient


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def review_graph(tmp_path, dashboard_auth):
    clock = Clock()
    async with AsyncExitStack() as stack:
        authority = DashboardAuthority(tmp_path / 'auth.db', clock=clock, session_ttl_s=60)
        stack.push_async_callback(authority.aclose)
        await authority.initialize()
        services = await build_memory_services({'memory': {'privacy_db_path': str(tmp_path / 'privacy.db'),
                                               'preferences_path': str(tmp_path / 'preferences.json')}},
            shutdown=stack, chroma_client=MemoryClient(tmp_path / 'index'), clock=clock)
        api = DashboardAPI(authority=authority, memory_manager=services.manager)
        stack.push_async_callback(api.stop)
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://testserver') as client:
            token = dashboard_auth['Authorization'].split(' ', 1)[1]
            assert (await client.post('/api/admin/auth/login', json={'token': token})).status_code == 200
            yield SimpleNamespace(service=services.manager.long_term, manager=services.manager,
                                  authority=authority, client=client, clock=clock)


async def proposal(graph, *, confirmed=False, content='Synthetic private preference.'):
    source = graph.manager.source_for_capture(str(uuid4()), graph.clock.now, 'synthetic-room')
    value = await graph.service.acall(graph.service.propose, content, (source,))
    if confirmed:
        value = await graph.service.acall(graph.service.confirm, value['id'], MemoryReviewer('synthetic-seed'))
    return value, source


async def mutate(client, operation, identity):
    if operation == 'delete':
        return await client.delete('/api/memory/facts/' + identity)
    return await client.post('/api/memory/proposals/' + identity + '/' + operation)


async def assert_fact_unchanged(graph, before):
    fact = await graph.service.acall(graph.service.ledger.get, before['id'], review=True)
    assert fact is not None
    assert fact.state.value == before['state'] and fact.revision == before['revision']
    assert fact.content == before['content']


async def test_review_waiting_for_native_ledger_lock_cannot_outlive_session(review_graph, monkeypatch):
    graph = review_graph
    before, _ = await proposal(graph)
    ledger = graph.service.ledger
    original_lock, original_review = ledger._lock, ledger._reviewer
    entered, release, local = threading.Event(), threading.Event(), threading.local()
    def reviewed(actor):
        original_review(actor)
        if actor._current is not None:
            local.review = True
    class ObservedLock:
        def __enter__(self):
            if getattr(local, 'review', False) and not getattr(local, 'waited', False):
                local.waited = True
                entered.set()
                assert release.wait(5)
            return original_lock.__enter__()
        def __exit__(self, *args):
            return original_lock.__exit__(*args)
    monkeypatch.setattr(ledger, '_reviewer', reviewed)
    monkeypatch.setattr(ledger, '_lock', ObservedLock())
    pending = asyncio.create_task(mutate(graph.client, 'confirm', before['id']))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
    finally:
        release.set()
    assert (await pending).status_code == 403
    await assert_fact_unchanged(graph, before)


@pytest.mark.parametrize('operation', ['confirm', 'reject', 'delete'])
@pytest.mark.parametrize('cancel_observer', [False, True])
async def test_review_revoked_after_provisional_sql_rolls_back_only_its_mutation(
    review_graph, monkeypatch, dashboard_auth, operation, cancel_observer,
):
    graph = review_graph
    before, _ = await proposal(graph, confirmed=operation == 'delete')
    independent, _ = await proposal(graph, content='Independent synthetic fact.')
    entered, release = threading.Event(), threading.Event()
    original = graph.service.ledger._queue
    def queued(db, identity, revision, action):
        original(db, identity, revision, action)
        if identity == before['id']:
            entered.set()
            assert release.wait(5)
    monkeypatch.setattr(graph.service.ledger, '_queue', queued)
    pending = asyncio.create_task(mutate(graph.client, operation, before['id']))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if cancel_observer:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
        other = asyncio.create_task(graph.client.post('/api/memory/proposals/' + independent['id'] + '/confirm',
                                                     headers=dashboard_auth))
        await asyncio.sleep(0)
    finally:
        release.set()
    if cancel_observer:
        await asyncio.gather(*tuple(graph.service._operations), return_exceptions=True)
    else:
        assert (await pending).status_code == 403
    assert (await other).status_code == 200
    await assert_fact_unchanged(graph, before)
    confirmed = await graph.service.acall(graph.service.ledger.get, independent['id'])
    assert confirmed.state is FactState.CONFIRMED
    if operation == 'delete':
        assert await graph.service.acall(graph.service._collection.count) == 2


async def test_conversation_erasure_rechecks_authority_after_provisional_publication(review_graph, monkeypatch):
    graph = review_graph
    before, source = await proposal(graph, confirmed=True)
    ledger = graph.service.ledger
    lease = graph.authority.resolve(None, graph.client.cookies.get(COOKIE_NAME))
    actor = MemoryReviewer.from_dashboard(lease.principal,
                                         lambda: lease.current(frozenset({'memory:review'})))
    entered, release = threading.Event(), threading.Event()
    original = ledger._publish_revocation_admission
    def published():
        original()
        if ledger._pending_conversations:
            entered.set()
            assert release.wait(5)
    monkeypatch.setattr(ledger, '_publish_revocation_admission', published)
    pending = asyncio.create_task(graph.service.acall(graph.service.delete_conversation, source.conversation_id, actor))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
    finally:
        release.set()
    with pytest.raises(PermissionError):
        await pending
    await assert_fact_unchanged(graph, before)
    assert not ledger.conversation_committed_revoked(source.conversation_id)
    assert not ledger._pending_conversations and not ledger._pending_fact_ids
    def persisted_operations():
        with ledger._transaction() as db:
            return db.execute('SELECT count(*) FROM conversation_erasures').fetchone()[0]
    assert await graph.service.acall(persisted_operations) == 0


@pytest.mark.parametrize('cancel_observer', [False, True])
async def test_committed_review_reconciles_index_after_session_retires(
    review_graph, monkeypatch, dashboard_auth, cancel_observer,
):
    graph = review_graph
    before, _ = await proposal(graph)
    entered, release = threading.Event(), threading.Event()
    original = graph.service._notify
    def notify():
        entered.set()
        assert release.wait(5)
        original()
    monkeypatch.setattr(graph.service, '_notify', notify)
    pending = asyncio.create_task(mutate(graph.client, 'confirm', before['id']))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if cancel_observer:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
    finally:
        release.set()
    if cancel_observer:
        await asyncio.gather(*tuple(graph.service._operations), return_exceptions=True)
    else:
        await pending
    fact = await graph.service.acall(graph.service.ledger.get, before['id'])
    assert fact.state is FactState.CONFIRMED and fact.confirmed_by == 'dashboard-admin'
    assert await graph.service.acall(graph.service._collection.count) == 1
    visible = await graph.client.get('/api/memory/facts', headers=dashboard_auth)
    assert visible.status_code == 200
    assert visible.json()['facts'][0]['id'] == before['id']


async def test_retired_actor_is_denied_before_sdk_initialization(review_graph, monkeypatch):
    graph = review_graph
    before, _ = await proposal(graph)
    lease = graph.authority.resolve(None, graph.client.cookies.get(COOKIE_NAME))
    actor = MemoryReviewer.from_dashboard(lease.principal, lease.current)
    assert await graph.authority.logout(lease)
    calls, original = [], graph.service.initialize
    def initialize():
        calls.append(True)
        original()
    monkeypatch.setattr(graph.service, 'initialize', initialize)
    with pytest.raises(PermissionError):
        await graph.service.acall(graph.service.confirm, before['id'], actor)
    assert calls == []


async def test_a_read_permission_and_operator_label_cannot_mint_memory_review():
    principal = DashboardPrincipal('dashboard-admin', frozenset({'memory:read'}))
    with pytest.raises(PermissionError):
        MemoryReviewer.from_dashboard(principal, lambda: True)
