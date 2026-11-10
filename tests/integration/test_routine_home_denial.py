"""Household and evaluation sessions cannot change or run routines on the real graph.

HOME holds ``routines:execute`` and ``home:control`` but no route accepts
``routines:execute``; this pins that the permission grants nothing today, and
that a person or voice binding on the credential does not widen it.
"""

from contextlib import asynccontextmanager
import time

from httpx import ASGITransport, AsyncClient
import pytest

from src.dashboard.authority import DashboardRole, EvaluationModality, EvaluationSample
from src.routines.models import parse_routine_spec
from src.users.user_manager import PermissionLevel
from tests.integration.test_dashboard_sessions import leaves as application_leaves, session_graph as application_graph

pytestmark = pytest.mark.asyncio
leaves = application_leaves
session_graph = application_graph


def definition(name: str = 'Synthetic household routine') -> dict:
    return {'name': name, 'description': '', 'enabled': True, 'timezone': 'America/Montevideo',
            'schedule': None, 'conditions': [], 'actions': [{'entity_id': 'light.living', 'operation': 'turn_on'}]}


async def snapshot(store) -> list[tuple]:
    rows = []
    for routine in await store.list_all():
        latest = await store.get_latest_run(routine.routine_id)
        rows.append((routine.routine_id, routine.revision, routine.spec.name, routine.spec.enabled, latest.run_id if latest else None))
    return sorted(rows)


@asynccontextmanager
async def login(graph, dashboard_auth, role, **extra):
    authority = graph.dashboard_authority
    now = time.time()
    samples = (EvaluationSample('vf_' + 'c' * 64, frozenset({'d' * 64}), frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 60),) \
        if role is DashboardRole.EVALUATOR else ()
    grant = await authority.issue_credential(authority.resolve(dashboard_auth['Authorization']), subject=f'synthetic-{role.value}',
                                             role=role, expires_at=now + 120, samples=samples, **extra)
    async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
        response = await client.post('/api/admin/auth/login', json={'token': grant.token})
        assert response.status_code == 200
        yield client


@pytest.mark.parametrize('personal', [False, True])
async def test_household_session_reads_routines_but_cannot_create_change_preview_or_run_them(session_graph, dashboard_auth, personal):
    graph = session_graph
    scheduler = graph.routine_scheduler
    routine = await scheduler.register_routine(parse_routine_spec(definition()), actor='synthetic-seed')
    before = await snapshot(scheduler.store)
    extra = {}
    if personal:
        # An administrator-level person bound to a household credential grants nothing extra.
        account, _ = await graph.voice_registry.acall(graph.pipeline.command_processor.user_manager.add_user,
                                                      'Persona sintética', PermissionLevel.ADMIN)
        extra = {'user_id': account.user_id}
    async with login(graph, dashboard_auth, DashboardRole.HOME, **extra) as client:
        identity = (await client.get('/api/admin/auth/whoami')).json()
        assert {'routines:execute', 'home:control'} <= set(identity['permissions'])
        assert 'routines:manage' not in identity['permissions']
        assert bool(identity['personal_scope']) is personal
        route = f'/api/routines/{routine.routine_id}'
        listed = await client.get('/api/routines')
        assert listed.status_code == 200 and [item['routine_id'] for item in listed.json()] == [routine.routine_id]
        assert (await client.get(route)).status_code == 200
        denied = [
            await client.post('/api/routines', json=definition('Household attempt')),
            await client.post('/api/routines/preview', json=definition('Household preview')),
            await client.put(route, json={**definition('Household change'), 'expected_revision': routine.revision}),
            await client.delete(route, params={'expected_revision': routine.revision}),
            await client.post(route + '/disable', json={'expected_revision': routine.revision}),
            await client.post(route + '/enable', json={'expected_revision': routine.revision}),
            await client.post(route + '/execute', json={'request_id': 'household-attempt'}),
            await client.get(route + '/runs/unknown-run'),
            await client.get(route + '/attempts', params={'request_id': 'household-attempt'}),
        ]
        # The route permission model denies, not some other 403 layer.
        assert [(response.status_code, response.json().get('detail')) for response in denied] == [(403, 'dashboard_permission_denied')] * len(denied)
    assert await snapshot(scheduler.store) == before
    assert await scheduler.store.get_latest_run(routine.routine_id) is None
    # Positive control: the same graph accepts an operator's run, so the fixture does not deny everything.
    async with login(graph, dashboard_auth, DashboardRole.OPERATOR) as client:
        accepted = await client.post(f'/api/routines/{routine.routine_id}/execute', json={'request_id': 'operator-control'})
        assert accepted.status_code == 202 and accepted.json()['routine_id'] == routine.routine_id


async def test_evaluator_session_cannot_read_routines(session_graph, dashboard_auth):
    graph = session_graph
    routine = await graph.routine_scheduler.register_routine(parse_routine_spec(definition()), actor='synthetic-seed')
    async with login(graph, dashboard_auth, DashboardRole.EVALUATOR) as client:
        for response in (await client.get('/api/routines'), await client.get(f'/api/routines/{routine.routine_id}'),
                         await client.post(f'/api/routines/{routine.routine_id}/execute', json={'request_id': 'evaluator'})):
            assert (response.status_code, response.json().get('detail')) == (403, 'dashboard_permission_denied')
    assert await graph.routine_scheduler.store.get_latest_run(routine.routine_id) is None
