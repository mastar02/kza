"""Routine permission and receipt visibility on the real dashboard graph."""

import asyncio
from contextlib import asynccontextmanager
import json
import time

from httpx import ASGITransport, AsyncClient
import pytest
from starlette.requests import Request

from src.dashboard.auth import COOKIE_NAME, _route_permissions
from src.dashboard.authority import DashboardPrincipal, DashboardRole, EvaluationModality, EvaluationSample
from src.routines.models import RoutineBrake, RunStatus, parse_routine_spec
from src.routines.routine_store import HOUSEHOLD_VOICE
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


async def wait_for_worker(scheduler, run_id: str) -> None:
    worker = scheduler._workers.get(run_id)
    if worker is not None:
        await asyncio.wait_for(asyncio.shield(worker), 3)


@asynccontextmanager
async def login(graph, dashboard_auth, role, *, subject=None, **extra):
    authority = graph.dashboard_authority
    now = time.time()
    samples = (EvaluationSample('vf_' + 'c' * 64, frozenset({'d' * 64}), frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 60),) \
        if role is DashboardRole.EVALUATOR else ()
    grant = await authority.issue_credential(authority.resolve(dashboard_auth['Authorization']), subject=subject or f'synthetic-{role.value}',
                                             role=role, expires_at=now + 120, samples=samples, **extra)
    async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
        response = await client.post('/api/admin/auth/login', json={'token': grant.token})
        assert response.status_code == 200
        yield client


@pytest.mark.parametrize('personal', [False, True])
async def test_household_can_execute_and_read_own_receipt_but_cannot_manage(session_graph, dashboard_auth, personal):
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
        ]
        assert [(response.status_code, response.json().get('detail')) for response in denied] == [(403, 'dashboard_permission_denied')] * len(denied)
        assert await snapshot(scheduler.store) == before

        attempt = {'request_id': '..'}
        first, replay = await asyncio.gather(*(client.post(route + '/execute', json=attempt) for _ in range(2)))
        assert first.status_code == replay.status_code == 202
        run = first.json()
        assert run['routine_id'] == routine.routine_id and run['run_id'] == replay.json()['run_id']
        assert run['actor'] == identity['subject'] and run['source'] == 'manual'
        await wait_for_worker(scheduler, run['run_id'])
        run_route = route + '/runs/' + run['run_id']
        terminal = await client.get(run_route)
        assert terminal.status_code == 200 and terminal.json()['status'] == 'succeeded'
        assert terminal.json()['actions'][0]['status'] == 'confirmed'
        assert (await graph.ha_client.get_entity_state('light.living'))['state'] == 'on'
        consulted = await client.get(route + '/attempts', params=attempt)
        assert consulted.status_code == 200 and consulted.json() == terminal.json()
        assert (await client.get(route)).json()['latest_run'] == terminal.json()
        assert (await client.get('/api/routines')).json()[0]['latest_run'] == terminal.json()

        async with login(graph, dashboard_auth, DashboardRole.HOME, subject='synthetic-other-home') as other:
            for response in (await other.get(run_route), await other.get(route + '/runs/unknown-run'),
                             await other.get(route + '/attempts', params=attempt)):
                assert (response.status_code, response.json()) == (404, {'detail': 'routine_not_found'})
            assert (await other.get(route)).json()['latest_run'] is None
            assert (await other.get('/api/routines')).json()[0]['latest_run'] is None
            other_claim = await other.post(route + '/execute', json=attempt)
            assert other_claim.status_code == 202
            other_run = other_claim.json()
            assert other_run['run_id'] != run['run_id'] and other_run['actor'] == 'synthetic-other-home'
            await wait_for_worker(scheduler, other_run['run_id'])
        assert (await client.get(route)).json()['latest_run'] is None
        assert (await client.get('/api/routines')).json()[0]['latest_run'] is None
        assert (await client.get(route + '/attempts', params=attempt)).json() == terminal.json()

        async with login(graph, dashboard_auth, DashboardRole.OPERATOR) as operator:
            assert (await operator.get(run_route)).json() == terminal.json()
            assert (await operator.get(route + '/attempts', params=attempt)).status_code == 404
            disabled = await operator.post(route + '/disable', json={'expected_revision': routine.revision})
            assert disabled.status_code == 200 and disabled.json()['latest_run']['run_id'] == other_run['run_id']
        refused = await client.post(route + '/execute', json={'request_id': 'fresh-disabled'})
        assert (refused.status_code, refused.json()) == (409, {'detail': 'routine_disabled'})
        after_disable = await client.post(route + '/execute', json=attempt)
        assert after_disable.status_code == 202 and after_disable.json() == terminal.json()
        assert (await client.get(route + '/attempts', params={'request_id': 'fresh-disabled'})).status_code == 404


@asynccontextmanager
async def routine_socket(app, client):
    incoming, outgoing = asyncio.Queue(), asyncio.Queue()
    cookie = f'{COOKIE_NAME}={client.cookies.get(COOKIE_NAME)}'.encode()
    scope = {'type': 'websocket', 'path': '/ws', 'raw_path': b'/ws', 'root_path': '',
             'scheme': 'ws', 'query_string': b'', 'server': ('testserver', 80),
             'headers': [(b'cookie', cookie)], 'subprotocols': [], 'state': {}}
    await incoming.put({'type': 'websocket.connect'})
    task = asyncio.create_task(app(scope, incoming.get, outgoing.put))
    try:
        assert (await asyncio.wait_for(outgoing.get(), 3))['type'] == 'websocket.accept'
        yield outgoing
    finally:
        await incoming.put({'type': 'websocket.disconnect', 'code': 1000})
        await asyncio.wait_for(task, 3)


async def test_latest_run_is_projected_for_each_rest_and_ws_reader(session_graph, dashboard_auth):
    graph = session_graph
    scheduler = graph.routine_scheduler
    home_subject = 'synthetic-home'
    # Subscribe the foreign reader first so redaction cannot mutate the shared payload.
    async with (
        login(graph, dashboard_auth, DashboardRole.HOME, subject='synthetic-other-home') as other,
        login(graph, dashboard_auth, DashboardRole.HOME, subject=home_subject) as home,
        login(graph, dashboard_auth, DashboardRole.OPERATOR) as operator,
        routine_socket(graph.dashboard.app, other) as other_ws,
        routine_socket(graph.dashboard.app, home) as home_ws,
        routine_socket(graph.dashboard.app, operator) as operator_ws,
    ):
        for source in ('manual', 'scheduled', HOUSEHOLD_VOICE):
            payload = definition(f'Synthetic {source} routine')
            created = await operator.post('/api/routines', json=payload)
            assert created.status_code == 200
            routine = created.json()
            for socket in (other_ws, home_ws, operator_ws):
                frame = json.loads((await asyncio.wait_for(socket.get(), 3))['text'])
                assert frame == {'type': 'routine_created', 'routine': routine}

            routine_id = routine['routine_id']
            # Give each historical source the same actor to test source and ownership independently.
            key = json.dumps([source, 'synthetic-history'])
            claim = await scheduler.store.claim_run(routine_id, key, actor=home_subject, source=source)
            run = await scheduler.store.finish_run(claim.run.run_id, RunStatus.SKIPPED)
            brake = await scheduler.brake_by_name(payload['name'], RoutineBrake.STOP,
                request_id=f'{source}-brake', guard=lambda: True)
            brake_json = {'receipt_id': brake.receipt_id, 'source': HOUSEHOLD_VOICE,
                          'operation': 'stop', 'status': 'nothing_to_stop'}
            route = f'/api/routines/{routine_id}'
            updated = await operator.put(route, json={**payload, 'expected_revision': routine['revision']})
            assert updated.status_code == 200
            for client, socket, visible in ((other, other_ws, False), (home, home_ws, source == 'manual'),
                                            (operator, operator_ws, True)):
                expected_run = updated.json()['latest_run'] if visible else None
                if visible:
                    assert expected_run['run_id'] == run.run_id
                listed = await client.get('/api/routines')
                listed_routine = next(row for row in listed.json() if row['routine_id'] == routine_id)
                detail = await client.get(route)
                assert detail.status_code == listed.status_code == 200
                frame = json.loads((await asyncio.wait_for(socket.get(), 3))['text'])
                assert frame['type'] == 'routine_updated'
                for projected in (listed_routine, detail.json(), frame['routine']):
                    assert projected['routine_id'] == routine_id
                    assert projected['latest_run'] == expected_run
                    assert projected['latest_voice_brake'] == brake_json
                direct = await client.get(route + '/runs/' + run.run_id)
                if visible:
                    assert direct.status_code == 200 and direct.json() == expected_run
                else:
                    assert (direct.status_code, direct.json()) == (404, {'detail': 'routine_not_found'})


@pytest.mark.parametrize('grants, receipts, execute', [
    ({'home:control'}, False, False),
    ({'routines:execute'}, True, False),
    ({'routines:manage'}, True, False),
    ({'home:control', 'routines:execute'}, True, True),
    ({'home:control', 'routines:manage'}, True, True),
])
async def test_routine_admission_requires_one_routine_grant_and_control_for_execute(grants, receipts, execute):
    principal = DashboardPrincipal('synthetic', frozenset(grants))
    for method, tail, permitted in (('GET', '/attempts', receipts), ('GET', '/runs/run', receipts),
                                    ('POST', '/execute', execute)):
        request = Request({'type': 'http', 'method': method, 'path': '/api/routines/routine' + tail,
                           'headers': [], 'query_string': b''})
        assert (_route_permissions(request, principal) <= principal.permissions) is permitted


async def test_retired_household_session_cannot_claim_manual_run(session_graph, dashboard_auth, monkeypatch):
    graph = session_graph
    scheduler = graph.routine_scheduler
    routine = await scheduler.register_routine(parse_routine_spec(definition()), actor='synthetic-seed')
    entered, release = asyncio.Event(), asyncio.Event()
    execute_by_id = scheduler.execute_by_id

    async def held_execution(*args, **kwargs):
        entered.set()
        await release.wait()
        return await execute_by_id(*args, **kwargs)

    monkeypatch.setattr(scheduler, 'execute_by_id', held_execution)
    async with login(graph, dashboard_auth, DashboardRole.HOME) as client:
        pending = asyncio.create_task(client.post(f'/api/routines/{routine.routine_id}/execute',
                                                  json={'request_id': 'retired-home'}))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert (await client.post('/api/admin/auth/logout')).json()['session_revoked']
        finally:
            release.set()
            response = await pending
        assert response.status_code == 403
    assert await scheduler.store.get_latest_run(routine.routine_id) is None
    assert (await graph.ha_client.get_entity_state('light.living'))['state'] == 'off'


async def test_evaluator_session_cannot_read_routines(session_graph, dashboard_auth):
    graph = session_graph
    routine = await graph.routine_scheduler.register_routine(parse_routine_spec(definition()), actor='synthetic-seed')
    route = f'/api/routines/{routine.routine_id}'
    async with login(graph, dashboard_auth, DashboardRole.EVALUATOR) as client:
        for response in (await client.get('/api/routines'), await client.get(route), await client.get(route + '/runs/run'),
                         await client.get(route + '/attempts', params={'request_id': 'evaluator'}),
                         await client.post(route + '/execute', json={'request_id': 'evaluator'})):
            assert (response.status_code, response.json().get('detail')) == (403, 'dashboard_permission_denied')
    assert await graph.routine_scheduler.store.get_latest_run(routine.routine_id) is None
