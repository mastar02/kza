"""Actual factory sessions and explicit standalone bearer compatibility."""

from contextlib import AsyncExitStack
import time

from httpx import ASGITransport, AsyncClient
import pytest

from src.dashboard import system_monitor
from src.dashboard.api import DashboardAPI
from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import (
    DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample,
)
from src.main import build_application
from tests.integration.test_capability_application import configuration, leaves as capability_leaves
from tests.mocks.task_transport import TaskTransport


pytestmark = pytest.mark.asyncio


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    return capability_leaves.__wrapped__(monkeypatch, tmp_path)


@pytest.fixture
async def session_graph(leaves, tmp_path, dashboard_auth):
    async with TaskTransport().serve() as url:
        config, _ = configuration(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            yield graph


async def test_factory_cookie_is_opaque_and_works_through_owned_native_observations(
    session_graph, dashboard_auth, monkeypatch,
):
    graph = session_graph
    token = dashboard_auth['Authorization'].split(' ', 1)[1]
    monkeypatch.setattr(system_monitor, 'gpu_snapshot', lambda: [])
    monkeypatch.setattr(system_monitor, 'services_snapshot', lambda: [])
    async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
        response = await client.post('/api/admin/auth/login', json={'token': token})
        assert response.status_code == 200 and response.json() == {'ok': True}
        cookie = client.cookies.get(COOKIE_NAME)
        assert cookie.startswith('kzas_') and cookie != token
        assert 'HttpOnly' in response.headers['set-cookie']
        assert 'SameSite=strict' in response.headers['set-cookie']
        identity = await client.get('/api/admin/auth/whoami')
        assert identity.status_code == 200 and identity.json()['role'] == 'operator'
        assert 'admin:manage' in identity.json()['permissions']
        for path in ('/api/system/gpus', '/api/system/services'):
            observed = await client.get(path)
            assert observed.status_code == 200 and observed.json() == []
            assert observed.headers['X-KZA-Source'] == 'real'
        conversations = await client.get('/api/conversations')
        assert conversations.status_code == 200 and conversations.json() == []
        assert conversations.headers['X-KZA-Observation-State'] == 'unavailable'
        assert (await client.get('/api/voices')).status_code == 428
        assert (await client.get('/api/admin/auth/whoami', headers={'Authorization': 'Bearer invalid'})).status_code == 401


@pytest.mark.parametrize('cookie', [False, True])
async def test_factory_administrative_permission_values_reach_real_home_control(
    session_graph, dashboard_auth, cookie,
):
    graph = session_graph
    async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
        headers = dashboard_auth
        if cookie:
            token = dashboard_auth['Authorization'].split(' ', 1)[1]
            assert (await client.post('/api/admin/auth/login', json={'token': token})).status_code == 200
            headers = {}
        response = await client.post('/api/home/actions', headers=headers,
            json={'entity_id': 'light.living', 'operation': 'turn_on', 'request_id': 'test'})
        assert response.status_code == 200
        state = await graph.ha_client.get_entity_state('light.living')
        assert state['state'] == 'on'


@pytest.mark.parametrize('role', [DashboardRole.HOME, DashboardRole.EVALUATOR])
async def test_role_and_permissions_come_from_server_grant_not_login_fields(session_graph, dashboard_auth, role):
    graph = session_graph
    now = time.time()
    scope = (EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}),
                              frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 60),) if role is DashboardRole.EVALUATOR else ()
    grant = await graph.dashboard_authority.issue_credential(
        graph.dashboard_authority.resolve(dashboard_auth['Authorization']), subject='synthetic-member', role=role,
        expires_at=now + 120, samples=scope)
    async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
        response = await client.post('/api/admin/auth/login', json={
            'token': grant.token, 'role': 'operator', 'permissions': ['admin:manage'], 'user_id': 'untrusted-person',
        })
        assert response.status_code == 200
        identity = (await client.get('/api/admin/auth/whoami')).json()
        assert identity['role'] == role.value and not identity['personal_scope']
        assert 'admin:manage' not in identity['permissions']
        for path in ('/api/users', '/api/conversations', '/api/voices', '/api/memory/facts',
                     '/api/system/gpus', '/api/home/microphones', '/api/lists?user_id=another-person'):
            assert (await client.get(path)).status_code == 403
        assert (await client.post('/api/llm/endpoints/synthetic/clear-cooldown')).status_code == 403
        catalog = await client.get('/api/home/catalog')
        assert catalog.status_code == (200 if role is DashboardRole.HOME else 403)
        logout = await client.post('/api/admin/auth/logout')
        assert logout.status_code == 200 and logout.json()['session_revoked'] is True


async def test_logout_replay_is_denied_while_other_session_and_bootstrap_bearer_remain_valid(
    session_graph, dashboard_auth,
):
    app = session_graph.dashboard.app
    token = dashboard_auth['Authorization'].split(' ', 1)[1]
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver') as one, \
               AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver') as two:
        assert (await one.post('/api/admin/auth/login', json={'token': token})).status_code == 200
        assert (await two.post('/api/admin/auth/login', json={'token': token})).status_code == 200
        first = one.cookies.get(COOKIE_NAME)
        assert (await one.post('/api/admin/auth/logout')).json()['session_revoked'] is True
        assert (await one.get('/api/admin/auth/whoami')).status_code == 401
        replay = await one.get('/api/admin/auth/whoami', headers={'Cookie': f'{COOKIE_NAME}={first}'})
        assert replay.status_code == 401
        assert (await two.get('/api/admin/auth/whoami')).status_code == 200
        assert (await one.get('/api/admin/auth/whoami', headers=dashboard_auth)).status_code == 200


async def test_http_expiration_is_checked_by_server_while_browser_cookie_is_still_present(
    session_graph, dashboard_auth, monkeypatch,
):
    authority = session_graph.dashboard_authority
    clock = [time.time() + 1]
    monkeypatch.setattr(authority, '_clock', lambda: clock[0])
    grant = await authority.issue_credential(authority.resolve(dashboard_auth['Authorization']),
        subject='synthetic-short-session', role=DashboardRole.HOME, expires_at=clock[0] + 2)
    async with AsyncClient(transport=ASGITransport(app=session_graph.dashboard.app), base_url='http://testserver') as client:
        assert (await client.post('/api/admin/auth/login', json={'token': grant.token})).status_code == 200
        assert client.cookies.get(COOKIE_NAME)
        assert (await client.get('/api/admin/auth/whoami')).status_code == 200
        clock[0] += 2
        assert client.cookies.get(COOKIE_NAME)
        assert (await client.get('/api/admin/auth/whoami')).status_code == 401


async def test_role_reduction_keeps_bearer_valid_but_invalidates_old_cookie(session_graph, dashboard_auth):
    authority = session_graph.dashboard_authority
    root = authority.resolve(dashboard_auth['Authorization'])
    grant = await authority.issue_credential(root, subject='synthetic-reduced', role=DashboardRole.OPERATOR,
                                            expires_at=time.time() + 120)
    async with AsyncClient(transport=ASGITransport(app=session_graph.dashboard.app), base_url='http://testserver') as client:
        assert (await client.post('/api/admin/auth/login', json={'token': grant.token})).status_code == 200
        assert await authority.revise_credential(root, grant.identity, role=DashboardRole.HOME, expires_at=time.time() + 90)
        assert (await client.get('/api/admin/auth/whoami')).status_code == 401
        identity = await client.get('/api/admin/auth/whoami', headers={'Authorization': 'Bearer ' + grant.token})
        assert identity.status_code == 200 and identity.json()['role'] == 'home'
        assert (await client.get('/api/users', headers={'Authorization': 'Bearer ' + grant.token})).status_code == 403


async def test_standalone_accepts_only_admin_bearer_and_emits_no_session_or_database(tmp_path, dashboard_auth, monkeypatch):
    monkeypatch.chdir(tmp_path)
    api = DashboardAPI()
    try:
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://testserver') as client:
            assert (await client.get('/api/admin/auth/whoami', headers=dashboard_auth)).status_code == 200
            token = dashboard_auth['Authorization'].split(' ', 1)[1]
            assert (await client.get('/api/admin/auth/whoami', headers={'Cookie': f'{COOKIE_NAME}={token}'})).status_code == 401
            assert (await client.get('/api/admin/auth/whoami', headers={'Cookie': f'{COOKIE_NAME}=kzas_untrusted'})).status_code == 401
            response = await client.post('/api/admin/auth/login', json={'token': token})
            assert response.status_code == 503 and 'set-cookie' not in response.headers
            logout = await client.post('/api/admin/auth/logout', headers=dashboard_auth)
            assert logout.status_code == 200 and logout.json()['session_revoked'] is False
        assert not list(tmp_path.rglob('*.db'))
    finally:
        await api.stop()


@pytest.mark.parametrize('state', ['uninitialized', 'closed', 'failed', 'invalid-type'])
async def test_injected_unavailable_owner_never_falls_back_to_valid_administrative_bearer(
    tmp_path, dashboard_auth, state,
):
    authority = DashboardAuthority(tmp_path / 'auth.db')
    if state == 'closed':
        await authority.aclose()
    elif state == 'failed':
        authority._failed = True
    api = DashboardAPI(authority=object() if state == 'invalid-type' else authority)
    try:
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://testserver') as client:
            assert (await client.get('/api/admin/auth/whoami', headers=dashboard_auth)).status_code == 503
            token = dashboard_auth['Authorization'].split(' ', 1)[1]
            assert (await client.post('/api/admin/auth/login', json={'token': token})).status_code == 503
        assert not authority.path.exists()
    finally:
        await api.stop()
        await authority.aclose()


async def test_a_cookie_from_another_application_cannot_authenticate_this_authority(
    session_graph, dashboard_auth, tmp_path,
):
    authority = DashboardAuthority(tmp_path / 'other-app' / 'auth.db')
    await authority.initialize()
    other = DashboardAPI(authority=authority)
    token = dashboard_auth['Authorization'].split(' ', 1)[1]
    try:
        async with AsyncClient(transport=ASGITransport(app=session_graph.dashboard.app), base_url='http://testserver') as client:
            assert (await client.post('/api/admin/auth/login', json={'token': token})).status_code == 200
            cookie = client.cookies.get(COOKIE_NAME)
        async with AsyncClient(transport=ASGITransport(app=other.app), base_url='http://testserver') as client:
            assert (await client.get('/api/admin/auth/whoami', headers={'Cookie': f'{COOKIE_NAME}={cookie}'})).status_code == 401
    finally:
        await other.stop()
        await authority.aclose()
