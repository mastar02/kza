"""Household and evaluation sessions cannot read or review memory through the real routes.

Facts have no owner: only an operator session holds ``memory:read`` and
``memory:review``. A person bound to a household credential, or a voice label,
widens nothing. Denials come from the route permission model before any
handler runs, so nothing is read or written.
"""

from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
import pytest

from src.core.account_metadata import ActiveAccounts
from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample
from src.memory.fact_ledger import FactState, MemoryReviewer
from src.memory.services import build_memory_services
from tests.mocks.memory_graph import Clock, MemoryClient

pytestmark = pytest.mark.asyncio
PRIVATE = 'Synthetic private memory that must never reach a denied body.'
PERSON = 'synthetic-person'


@pytest.fixture
async def memory_graph(tmp_path, dashboard_auth):
    """The real dashboard routes, authority and memory services (the application graph runs with memory off)."""
    clock = Clock()
    async with AsyncExitStack() as stack:
        authority = DashboardAuthority(tmp_path / 'auth.db', clock=clock, session_ttl_s=600,
                                       account_provider=lambda: ActiveAccounts(True, frozenset({PERSON})))
        stack.push_async_callback(authority.aclose)
        await authority.initialize()
        services = await build_memory_services({'memory': {'privacy_db_path': str(tmp_path / 'privacy.db'),
                                               'preferences_path': str(tmp_path / 'preferences.json')}},
            shutdown=stack, chroma_client=MemoryClient(tmp_path / 'index'), clock=clock)
        api = DashboardAPI(authority=authority, memory_manager=services.manager)
        stack.push_async_callback(api.stop)
        yield SimpleNamespace(manager=services.manager, memory=services.manager.long_term, authority=authority, api=api, clock=clock,
                              root=authority.resolve(dashboard_auth['Authorization']))


def source(graph):
    return graph.manager.source_for_capture(str(uuid4()), graph.clock.now, 'synthetic-room')


async def seed(graph):
    memory = graph.memory
    proposal = await memory.acall(memory.propose, PRIVATE + ' proposal', (source(graph),))
    other = await memory.acall(memory.propose, PRIVATE + ' second proposal', (source(graph),))
    confirmed = await memory.acall(memory.confirm, other['id'], MemoryReviewer('synthetic-seed'))
    automatic_id = await memory.astore_fact(PRIVATE + ' automatic', 'fact', sources=(source(graph),))
    return {'proposal': proposal['id'], 'confirmed': confirmed['id'], 'automatic': automatic_id}


async def ledger_state(graph, ids):
    rows = {}
    for name, identity in ids.items():
        fact = await graph.memory.acall(graph.memory.ledger.get, identity, review=True)
        rows[name] = None if fact is None else (fact.state.value, fact.revision, fact.content, fact.confirmed_by)
    return rows


@asynccontextmanager
async def login(graph, role, **extra):
    now = graph.clock.now
    samples = (EvaluationSample('vf_' + 'e' * 64, frozenset({'f' * 64}), frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 60),) \
        if role is DashboardRole.EVALUATOR else ()
    grant = await graph.authority.issue_credential(graph.root, subject=f'synthetic-memory-{role.value}',
                                                   role=role, expires_at=now + 120, samples=samples, **extra)
    async with AsyncClient(transport=ASGITransport(app=graph.api.app), base_url='http://testserver') as client:
        response = await client.post('/api/admin/auth/login', json={'token': grant.token})
        assert response.status_code == 200
        yield client


async def every_route(client, ids):
    return [
        await client.get('/api/memory'),
        await client.get('/api/memory/facts'),
        await client.get('/api/memory/proposals'),
        await client.post(f"/api/memory/proposals/{ids['proposal']}/confirm"),
        await client.post(f"/api/memory/proposals/{ids['proposal']}/reject"),
        await client.delete(f"/api/memory/facts/{ids['confirmed']}"),
        await client.delete(f"/api/memory/facts/{ids['automatic']}"),
    ]


@pytest.mark.parametrize('role,personal', [(DashboardRole.HOME, False), (DashboardRole.HOME, True), (DashboardRole.EVALUATOR, False)])
async def test_household_and_evaluator_sessions_cannot_read_or_review_memory(memory_graph, role, personal):
    graph = memory_graph
    ids = await seed(graph)
    before = await ledger_state(graph, ids)
    # Index work (index_jobs, policy_index_deletions), tombstones and state counts, not only fact rows (N7).
    status_before = await graph.memory.acall(graph.memory.ledger.status)
    # A person bound to a household credential (personal scope) grants nothing extra.
    extra = {'user_id': PERSON} if personal else {}
    async with login(graph, role, **extra) as client:
        identity = (await client.get('/api/admin/auth/whoami')).json()
        assert not {'memory:read', 'memory:review'} & set(identity['permissions'])
        assert bool(identity['personal_scope']) is personal
        denied = await every_route(client, ids)
        assert [(response.status_code, response.json().get('detail')) for response in denied] == [(403, 'dashboard_permission_denied')] * len(denied)
        assert not any('Synthetic private memory' in response.text for response in denied)
    assert await ledger_state(graph, ids) == before
    assert await graph.memory.acall(graph.memory.ledger.status) == status_before
    # Positive control: an operator session on the same graph reads and reviews.
    async with login(graph, DashboardRole.OPERATOR) as client:
        snapshot = await client.get('/api/memory')
        assert snapshot.status_code == 200
        assert {item['id'] for item in snapshot.json()['proposals']} == {ids['proposal']}
        assert {ids['confirmed'], ids['automatic']} <= {item['id'] for item in snapshot.json()['facts']}
        confirmed = await client.post(f"/api/memory/proposals/{ids['proposal']}/confirm")
        assert confirmed.status_code == 200 and confirmed.json()['fact']['confirmed_by'] == 'synthetic-memory-operator'
        first = await client.delete(f"/api/memory/facts/{ids['automatic']}")
        again = await client.delete(f"/api/memory/facts/{ids['automatic']}")
        assert (first.json()['deleted'], again.json()['deleted']) == (True, False)
        gone = await client.post(f"/api/memory/proposals/{ids['automatic']}/confirm")
        assert (gone.status_code, gone.json()['detail']) == (404, 'memory_proposal_unavailable')


async def test_snapshot_cap_hides_a_confirmed_memory_behind_100_newer_automatic_facts(memory_graph):
    """Pins the contract the Memoria view's cap rule depends on (U3 F1)."""
    graph = memory_graph
    memory = graph.memory
    old = await memory.acall(memory.propose, 'Synthetic old confirmed memory.', (source(graph),))
    await memory.acall(memory.confirm, old['id'], MemoryReviewer('synthetic-seed'))
    for index in range(100):
        graph.clock.now += 1
        await memory.astore_fact(f'Synthetic newer automatic fact {index}.', 'fact', sources=(source(graph),))
    async with login(graph, DashboardRole.OPERATOR) as client:
        snapshot = (await client.get('/api/memory')).json()
    assert len(snapshot['facts']) == 100
    assert old['id'] not in {item['id'] for item in snapshot['facts'] + snapshot['proposals']}
    stored = await memory.acall(memory.ledger.get, old['id'])
    assert stored is not None and stored.state is FactState.CONFIRMED and stored.expires_at is None
