"""Original web authority across real HA waits; loopback effects only."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
import time

from httpx import ASGITransport, AsyncClient
import pytest

from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import DashboardRole
from src.home_assistant.ha_client import HomeAssistantClient, ServiceCallStatus
from src.main import build_application
from tests.integration.test_capability_application import leaves as capability_leaves
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_task_outcomes import selection
from tests.mocks.application_graph import offline_config
from tests.mocks.task_transport import TaskTransport


pytestmark = pytest.mark.asyncio


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    return capability_leaves.__wrapped__(monkeypatch, tmp_path)


@asynccontextmanager
async def web_graph(tmp_path, leaves, dashboard_auth, *, group=False, hold=False):
    transport = TaskTransport(hold=hold)
    async with transport.serve() as url:
        config = offline_config(tmp_path, url)
        config['home_assistant']['control_selection_path'] = selection(tmp_path, group=group)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
                token = dashboard_auth['Authorization'].split(' ', 1)[1]
                assert (await client.post('/api/admin/auth/login', json={'token': token})).status_code == 200
                yield graph, client, transport
        assert graph.ha_client._session.closed


def intent(identity='authority-test', entity='light.living'):
    return {'entity_id': entity, 'operation': 'turn_on', 'request_id': identity}


async def receipt(client, dashboard_auth, identity='authority-test'):
    response = await client.get('/api/home/actions', params={'request_id': identity}, headers=dashboard_auth)
    assert response.status_code == 200
    result = response.json()
    assert result['request_id'] == identity
    return result


@pytest.mark.parametrize('legacy', [False, True])
async def test_minimum_home_role_controls_selected_target_through_guarded_http_adapter(
    leaves, tmp_path, dashboard_auth, legacy,
):
    async with web_graph(tmp_path, leaves, dashboard_auth) as (graph, client, transport):
        authority = graph.dashboard_authority
        grant = await authority.issue_credential(authority.resolve(dashboard_auth['Authorization']),
            subject='synthetic-home', role=DashboardRole.HOME, expires_at=time.time() + 120)
        assert (await client.post('/api/admin/auth/login', json={'token': grant.token})).status_code == 200
        if legacy:
            result = await client.post('/api/entities/light.living/call', params={'service': 'turn_on'},
                                       json={'request_id': 'home-legacy'})
        else:
            result = await client.post('/api/home/actions', json=intent())
        assert result.status_code == 200 and result.json()['dispatch'] == 'accepted'
        assert result.json()['effect'] == 'observed'
        assert len(transport.sends) == 1


@pytest.mark.parametrize('cancel_first', [False, True])
async def test_revocation_while_waiting_for_entity_lane_preserves_no_send_and_original_id(
    leaves, tmp_path, dashboard_auth, cancel_first,
):
    async with web_graph(tmp_path, leaves, dashboard_auth) as (graph, client, transport):
        expected_sends = []
        async with graph.home_actions._entity_lanes(('light.living',)):
            pending = asyncio.create_task(client.post('/api/home/actions', json=intent()))
            await until(lambda: graph.home_actions._lanes['light.living'][1] == 2)
            if cancel_first:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            assert (await client.post('/api/admin/auth/logout')).json()['session_revoked']
            rejoined = asyncio.create_task(client.post('/api/home/actions', json=intent(), headers=dashboard_auth))
            await asyncio.sleep(0)
            assert not rejoined.done()
            if cancel_first:
                independent = await client.post('/api/home/actions',
                    json=intent('independent', 'light.second'), headers=dashboard_auth)
                assert independent.status_code == 200 and independent.json()['dispatch'] == 'accepted'
                assert (await receipt(client, dashboard_auth, 'independent'))['entity_id'] == 'light.second'
                expected_sends = ['light.second']
            assert [s[0] for s in transport.sends] == expected_sends
        if not cancel_first:
            await pending
        assert (await rejoined).status_code == 200
        result = await receipt(client, dashboard_auth)
        assert result['dispatch'] == 'not_sent' and result['reason'] == 'home_control_denied'
        assert result['effect'] == 'not_observed'
        assert [s[0] for s in transport.sends] == expected_sends
        repeated = await client.post('/api/home/actions', json=intent(), headers=dashboard_auth)
        assert repeated.json() == result
        assert [s[0] for s in transport.sends] == expected_sends


@pytest.mark.parametrize('entity_state', ['off', 'unavailable'])
@pytest.mark.parametrize('withdrawal', ['logout', 'expiry', 'role-revision'])
async def test_original_authority_is_rechecked_after_real_ha_preflight(
    leaves, tmp_path, dashboard_auth, monkeypatch, withdrawal, entity_state,
):
    async with web_graph(tmp_path, leaves, dashboard_auth) as (graph, client, transport):
        transport.states['light.living']['state'] = entity_state
        authority = graph.dashboard_authority
        grant = None
        headers = dashboard_auth
        if withdrawal == 'role-revision':
            grant = await authority.issue_credential(authority.resolve(dashboard_auth['Authorization']),
                subject='synthetic-revised', role=DashboardRole.OPERATOR, expires_at=time.time() + 120)
            assert (await client.post('/api/admin/auth/login', json={'token': grant.token})).status_code == 200
            headers = {'Authorization': 'Bearer ' + grant.token}
        cookie = client.cookies.get(COOKIE_NAME)
        deadline = authority.resolve(None, cookie).principal.expires_at
        entered, release = asyncio.Event(), asyncio.Event()
        context = ContextVar('synthetic_sensitive_home_request', default=None)
        seen, original = [], graph.ha_client.get_entity_state_fresh
        async def held_read(*args, **kwargs):
            value = await original(*args, **kwargs)
            seen.append(context.get())
            entered.set()
            await release.wait()
            return value
        monkeypatch.setattr(graph.ha_client, 'get_entity_state_fresh', held_read)
        marker = context.set('synthetic-private-observer')
        pending = asyncio.create_task(client.post('/api/home/actions', json=intent()))
        context.reset(marker)
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if withdrawal == 'logout':
                assert (await client.post('/api/admin/auth/logout')).json()['session_revoked']
            elif withdrawal == 'expiry':
                monkeypatch.setattr(authority, '_clock', lambda: deadline)
            else:
                assert await authority.revise_credential(authority.resolve(dashboard_auth['Authorization']),
                    grant.identity, role=DashboardRole.HOME, expires_at=time.time() + 90)
        finally:
            release.set()
        await pending
        result = await receipt(client, headers)
        assert result['dispatch'] == 'not_sent' and result['effect'] == 'not_observed'
        assert result['reason'] == 'home_control_denied' and transport.sends == []
        assert seen == [None]


async def test_transport_session_lock_cannot_outlive_original_cookie_authority(
    leaves, tmp_path, dashboard_auth, monkeypatch,
):
    async with web_graph(tmp_path, leaves, dashboard_auth) as (graph, client, transport):
        client_ha = graph.ha_client
        entered, release, session_wait = asyncio.Event(), asyncio.Event(), asyncio.Event()
        dispatches, in_send = [], [False]
        send, ensure = client_ha.call_service_once, client_ha._ensure_session
        async def observed_ensure():
            if in_send[0]:
                session_wait.set()
            return await ensure()
        async def held_send(*args, **kwargs):
            entered.set()
            await release.wait()
            in_send[0] = True
            original_dispatch = kwargs['on_dispatch']
            def dispatched():
                dispatches.append(True)
                original_dispatch()
            kwargs['on_dispatch'] = dispatched
            return await send(*args, **kwargs)
        monkeypatch.setattr(client_ha, '_ensure_session', observed_ensure)
        monkeypatch.setattr(client_ha, 'call_service_once', held_send)
        pending = asyncio.create_task(client.post('/api/home/actions', json=intent()))
        locked = False
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await client_ha._session_lock.acquire()
            locked = True
            release.set()
            await asyncio.wait_for(session_wait.wait(), 3)
            assert (await client.post('/api/admin/auth/logout')).json()['session_revoked']
        finally:
            release.set()
            if locked:
                client_ha._session_lock.release()
        await pending
        result = await receipt(client, dashboard_auth)
        assert result['dispatch'] == 'not_sent' and result['reason'] == 'home_control_denied'
        assert transport.sends == [] and dispatches == []


async def test_withdrawal_between_compound_leaves_keeps_first_effect_and_stops_later_sends(
    leaves, tmp_path, dashboard_auth, monkeypatch,
):
    async with web_graph(tmp_path, leaves, dashboard_auth, group=True) as (graph, client, transport):
        entered, release = asyncio.Event(), asyncio.Event()
        original = graph.ha_client.confirm_entity_state
        async def confirmation(entity_id, *args, **kwargs):
            result = await original(entity_id, *args, **kwargs)
            if entity_id == 'light.living':
                entered.set()
                await release.wait()
            return result
        monkeypatch.setattr(graph.ha_client, 'confirm_entity_state', confirmation)
        pending = asyncio.create_task(client.post('/api/home/actions', json=intent(entity='light.group')))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert [s[0] for s in transport.sends] == ['light.living']
            assert (await client.post('/api/admin/auth/logout')).json()['session_revoked']
        finally:
            release.set()
        await pending
        result = await receipt(client, dashboard_auth)
        assert result['reason'] == 'partial_effects'
        assert [leaf['dispatch'] for leaf in result['effects']] == ['accepted', 'not_sent', 'not_sent']
        assert [leaf['effect'] for leaf in result['effects']] == ['observed', 'not_observed', 'not_observed']
        assert [s[0] for s in transport.sends] == ['light.living']


async def test_post_already_started_is_not_replayed_or_relabelled_unsent_after_logout(
    leaves, tmp_path, dashboard_auth,
):
    async with web_graph(tmp_path, leaves, dashboard_auth, hold=True) as (graph, client, transport):
        pending = asyncio.create_task(client.post('/api/home/actions', json=intent()))
        try:
            await asyncio.wait_for(transport.entered.wait(), 3)
            assert len(transport.sends) == 1
            assert (await client.post('/api/admin/auth/logout')).json()['session_revoked']
        finally:
            transport.release.set()
        await pending
        result = await receipt(client, dashboard_auth)
        assert result['dispatch'] == 'accepted' and result['effect'] == 'observed'
        repeated = await client.post('/api/home/actions', json=intent(), headers=dashboard_auth)
        assert repeated.json() == result and len(transport.sends) == 1


@pytest.mark.parametrize('allowed', [True, False, None, 1])
async def test_real_transport_checks_guard_after_session_wait_and_reports_actual_dispatch(allowed):
    transport = TaskTransport()
    async with transport.serve() as url:
        client = HomeAssistantClient(url, 'synthetic-ha-token')
        state, dispatches = [True], []
        pending, locked = None, False
        try:
            await client._session_lock.acquire()
            locked = True
            try:
                pending = asyncio.create_task(client.call_service_once('light', 'turn_on', 'light.living',
                    guard=lambda: state[0], on_dispatch=lambda: dispatches.append(True)))
                await asyncio.sleep(0)
                assert not pending.done() and not transport.sends
                state[0] = allowed
            finally:
                client._session_lock.release()
                locked = False
            result = await pending
            if allowed is True:
                assert result.status is ServiceCallStatus.ACCEPTED
                assert dispatches == [True] and len(transport.sends) == 1
            else:
                assert result.status is ServiceCallStatus.NOT_SENT
                assert result.reason == 'service_authority_denied'
                assert dispatches == [] and transport.sends == []
        finally:
            if locked:
                client._session_lock.release()
            if pending is not None:
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await client.close()


@pytest.mark.parametrize('failure,reason', [
    ('guard', 'service_authority_unavailable'), ('session', 'ha_session_unavailable'),
    ('dispatch', 'service_dispatch_unavailable'),
])
async def test_local_transport_failure_before_post_never_becomes_a_remote_attempt(failure, reason, monkeypatch):
    transport = TaskTransport()
    async with transport.serve() as url:
        client = HomeAssistantClient(url, 'synthetic-ha-token')
        dispatches = []
        if failure == 'session':
            async def unavailable():
                raise OSError('synthetic-session-failure')
            monkeypatch.setattr(client, '_ensure_session', unavailable)
        def guard():
            if failure == 'guard':
                raise RuntimeError('synthetic-private-failure')
            return True
        def dispatched():
            if failure == 'dispatch':
                raise RuntimeError('synthetic-dispatch-failure')
            dispatches.append(True)
        try:
            result = await client.call_service_once('light', 'turn_on', 'light.living',
                                                    guard=guard, on_dispatch=dispatched)
            assert result.status is ServiceCallStatus.NOT_SENT and result.reason == reason
            assert transport.sends == [] and dispatches == []
            assert 'synthetic-private-failure' not in repr(result)
        finally:
            await client.close()
