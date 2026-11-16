"""T034-U2: household web projection of shared reminder notices (read, dismiss, status counts)."""

from contextlib import AsyncExitStack
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient
import pytest
from starlette.requests import Request

from src.core.voice_delivery import DeliveryState, GenerationState, ResponseResult
from src.dashboard.api import DashboardAPI
from src.dashboard.auth import _route_permissions
from src.dashboard.authority import DashboardAuthority, DashboardRole, _permissions
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_scheduler import NoticeDismissal, ReminderScheduler
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
from tests.mocks.home_control import selected_actions
from tests.mocks.responses import voice_output

pytestmark = pytest.mark.asyncio
ACTION = {'domain': 'light', 'service': 'turn_on', 'entity_id': 'light.patio'}


def cut(spoken: int = 0, started: bool = True) -> ResponseResult:
    return ResponseResult('x', GenerationState.CANCELLED, DeliveryState.CANCELLED, spoken_segments=spoken,
                          output_started=started)


@pytest.fixture
async def notices_graph(tmp_path, dashboard_auth):
    async with AsyncExitStack() as stack:
        authority = DashboardAuthority(tmp_path / 'auth.db', session_ttl_s=600)
        stack.push_async_callback(authority.aclose)
        await authority.initialize()
        store = ReminderStore(str(tmp_path / 'reminders.db'))
        stack.push_async_callback(store.close)
        await store.initialize()
        ha, tts = AsyncMock(), voice_output()
        scheduler = ReminderScheduler(store, tts, None, ha, {'output_zone': 'zone_living'},
                                      home_actions=selected_actions(ha, ('light.patio',)))
        api = DashboardAPI(authority=authority, reminder_manager=ReminderManager(store), reminder_scheduler=scheduler)
        stack.push_async_callback(api.stop)
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://testserver') as client:
            root = authority.resolve(dashboard_auth['Authorization'])
            home = await authority.issue_credential(root, subject='synthetic-home', role=DashboardRole.HOME,
                                                    expires_at=time.time() + 300)
            assert (await client.post('/api/admin/auth/login', json={'token': home.token})).status_code == 200
            yield SimpleNamespace(store=store, scheduler=scheduler, client=client, ha=ha, tts=tts, app=api.app)


async def notice_rows(store):
    return await store._work.read(lambda db: [(row[0], row[1], json.loads(row[2])['attempt']) for row in db.execute(
        "SELECT status, reason, body FROM reminder_home_actions WHERE request_id LIKE 'notice:%' "
        "ORDER BY rowid").fetchall()])


def request(method: str, path: str) -> Request:
    return Request({'type': 'http', 'method': method, 'path': path, 'query_string': b'', 'headers': []})


async def test_notice_routes_need_household_task_permissions():
    """The home role holds tasks:read/tasks:write; an evaluator holds neither."""
    assert _route_permissions(request('GET', '/api/reminders/notices')) == {'tasks:read'}
    assert _route_permissions(request('POST', '/api/reminders/notices/dismiss')) == {'tasks:write'}
    assert {'tasks:read', 'tasks:write'} <= _permissions(DashboardRole.HOME, None)
    assert not {'tasks:read', 'tasks:write'} & _permissions(DashboardRole.EVALUATOR, None)
    # The status diagnostics stay operator-only.
    assert _route_permissions(request('GET', '/api/reminders/status')) == {'diagnostics:read'}
    assert 'diagnostics:read' not in _permissions(DashboardRole.HOME, None)


async def test_home_role_sees_shared_notices_only_with_source_and_freshness(notices_graph):
    graph = notices_graph
    now = time.time()
    private = await graph.store.create('u1', 'PRIVADO_NOTICE_417', now - 30)
    pending = await graph.store.create('u1', 'Sacar la basura', now - 20, scope=ReminderScope.SHARED)
    graph.tts.aspeak.return_value = cut()
    heard = await graph.store.create('u1', 'Regar las plantas', now - 10, ha_actions=[ACTION],
                                     scope=ReminderScope.SHARED)
    assert await graph.scheduler._fire_reminder(heard) is False
    graph.tts.aspeak.return_value = ResponseResult('x', GenerationState.COMPLETE, DeliveryState.FAILED)
    assert await graph.scheduler._fire_reminder(pending) is False
    response = await graph.client.get('/api/reminders/notices')
    assert response.status_code == 200
    assert response.headers['Cache-Control'] == 'no-store'
    assert response.headers['X-KZA-Source-Id'] == 'reminder-store'
    assert response.headers['X-KZA-Observation-State'] == 'fresh' and 'X-KZA-Observed-At' in response.headers
    data = response.json()
    assert data['source'] == 'reminder-store' and data['observed_at']
    assert 'PRIVADO_NOTICE_417' not in response.text and private.id not in response.text
    by_id = {entry['reminder_id']: entry for entry in data['notices']}
    assert set(by_id) == {pending.id, heard.id}
    assert by_id[pending.id] | {'claimed_at': None} == {
        'reminder_id': pending.id, 'trigger_at': pending.trigger_at, 'text': 'Sacar la basura', 'status': 'pending',
        'reason': None, 'label': 'no entregado', 'pending': True, 'attempts': 1, 'failed_attempts': 1,
        'claimed_at': None}
    assert (by_id[heard.id]['status'], by_id[heard.id]['reason'], by_id[heard.id]['label'],
            by_id[heard.id]['pending']) == ('uncertain', 'delivery_cancelled', 'cortado', False)
    assert data['counts'] == {'pending': 1, 'pending_not_delivered': 1, 'confirmed': 0, 'uncertain': 1}
    graph.ha.call_service_once.assert_not_called()


async def test_dismiss_from_the_web_confirms_advances_without_actions_and_never_speaks(notices_graph):
    graph = notices_graph
    reminder = await graph.store.create('u1', 'Regar', time.time() - 5, ha_actions=[ACTION],
                                        scope=ReminderScope.SHARED)
    private = await graph.store.create('u1', 'PRIVADO', time.time() - 5)
    response = await graph.client.post('/api/reminders/notices/dismiss',
                                       json={'reminder_id': reminder.id, 'trigger_at': reminder.trigger_at})
    assert response.status_code == 200 and response.json()['status'] == 'dismissed'
    assert await notice_rows(graph.store) == [('confirmed', 'dismissed', 1)]
    assert (await graph.store.get_by_id(reminder.id)).state is ReminderState.FIRED
    listed = (await graph.client.get('/api/reminders/notices')).json()['notices']
    assert [(e['status'], e['label'], e['pending']) for e in listed] == [('confirmed', 'descartado', False)]
    await graph.scheduler._deliver_due(time.time())
    graph.tts.aspeak.assert_not_awaited()
    graph.ha.call_service_once.assert_not_called()
    # A second dismiss of the finished occurrence is an early retention purge.
    again = await graph.client.post('/api/reminders/notices/dismiss',
                                    json={'reminder_id': reminder.id, 'trigger_at': reminder.trigger_at})
    assert again.status_code == 200 and again.json()['status'] == 'purged'
    assert await notice_rows(graph.store) == []
    hidden = await graph.client.post('/api/reminders/notices/dismiss',
                                     json={'reminder_id': private.id, 'trigger_at': private.trigger_at})
    assert hidden.status_code == 404
    assert (await graph.store.get_by_id(private.id)).state is ReminderState.ACTIVE
    invalid = await graph.client.post('/api/reminders/notices/dismiss',
                                      json={'reminder_id': reminder.id, 'trigger_at': 1., 'extra': 1})
    assert invalid.status_code == 422


@pytest.mark.parametrize('recurrence', [None, 'daily'])
async def test_dismissing_an_occurrence_that_is_not_due_yet_changes_nothing(notices_graph, recurrence):
    """Review round 1 R4: a future shared occurrence cannot be dismissed (or advanced) before it is due."""
    graph = notices_graph
    future = await graph.store.create('u1', 'Regar', time.time() + 3600, ha_actions=[ACTION],
                                      recurrence=recurrence, scope=ReminderScope.SHARED)
    response = await graph.client.post('/api/reminders/notices/dismiss',
                                       json={'reminder_id': future.id, 'trigger_at': future.trigger_at})
    assert response.status_code == 404 and response.json()['detail'] == 'reminder_notice_not_found'
    assert await graph.scheduler.dismiss_notice(future.id, future.trigger_at) is NoticeDismissal.NOT_FOUND
    assert await notice_rows(graph.store) == []
    current = await graph.store.get_by_id(future.id)
    assert (current.state, current.trigger_at, current.fire_count) == (ReminderState.ACTIVE, future.trigger_at, 0)
    graph.tts.aspeak.assert_not_awaited()
    graph.ha.call_service_once.assert_not_called()


@pytest.mark.parametrize('method, path', [('GET', '/api/reminders/notices'),
                                          ('POST', '/api/reminders/notices/dismiss')])
async def test_notice_routes_reject_unauthenticated_and_foreign_origin(notices_graph, dashboard_auth, method, path):
    graph = notices_graph
    reminder = await graph.store.create('u1', 'Regar', time.time() - 5, scope=ReminderScope.SHARED)
    body = {'reminder_id': reminder.id, 'trigger_at': reminder.trigger_at} if method == 'POST' else None
    async with AsyncClient(transport=ASGITransport(app=graph.app), base_url='http://testserver') as anonymous:
        assert (await anonymous.request(method, path, json=body)).status_code == 401
        foreign = await anonymous.request(method, path, json=body,
                                          headers={**dashboard_auth, 'Origin': 'https://other.invalid'})
        assert foreign.status_code == 403
    assert await notice_rows(graph.store) == []
    assert (await graph.store.get_by_id(reminder.id)).state is ReminderState.ACTIVE


async def test_status_reports_durable_notice_counts(notices_graph, dashboard_auth):
    graph = notices_graph
    graph.tts.aspeak.return_value = cut(spoken=1)
    heard = await graph.store.create('u1', 'Uno', time.time() - 5, scope=ReminderScope.SHARED)
    await graph.scheduler._fire_reminder(heard)
    graph.tts.aspeak.return_value = ResponseResult('x', GenerationState.COMPLETE, DeliveryState.COMPLETE)
    done = await graph.store.create('u1', 'Dos', time.time() - 5, scope=ReminderScope.SHARED)
    assert await graph.scheduler._fire_reminder(done) is True
    await graph.store.create('u1', 'Tres', time.time() - 5, scope=ReminderScope.SHARED)
    denied = await graph.client.get('/api/reminders/status')
    assert denied.status_code == 403  # home role: diagnostics are operator-only
    status = (await graph.client.get('/api/reminders/status', headers=dashboard_auth)).json()
    assert status['notices'] == {'pending': 1, 'pending_not_delivered': 0, 'confirmed': 1, 'uncertain': 1}
    assert 'delivery_failures' not in status and status['output_zone_configured'] is True


@pytest.mark.parametrize('result, status, reason, label', [
    (ResponseResult('x', GenerationState.COMPLETE, DeliveryState.COMPLETE), 'confirmed', None, 'entregado'),
    (cut(), 'uncertain', 'delivery_cancelled', 'cortado'),
    (ResponseResult('x', GenerationState.COMPLETE, DeliveryState.FAILED, spoken_segments=1), 'uncertain',
     'delivery_partial', 'incompleto'),
    (ResponseResult('x', GenerationState.COMPLETE, DeliveryState.TIMED_OUT, 'response_delivery_timeout',
                    timeout_stage=None), 'uncertain', 'delivery_partial', 'incompleto'),
    (RuntimeError('device lost'), 'uncertain', 'delivery_unverified', 'incierto'),
])
async def test_finished_notices_are_labelled_by_reason_for_display_only(notices_graph, result, status, reason, label):
    """Q3: a possibly heard notice is shown on the web (cortado, incompleto, incierto), never repeated by voice."""
    graph = notices_graph
    if isinstance(result, Exception):
        graph.tts.aspeak.side_effect = result
    else:
        graph.tts.aspeak.return_value = result
    reminder = await graph.store.create('u1', 'Regar', time.time() - 5, scope=ReminderScope.SHARED)
    await graph.scheduler._fire_reminder(reminder)
    [entry] = (await graph.client.get('/api/reminders/notices')).json()['notices']
    assert (entry['status'], entry['reason'], entry['label'], entry['pending']) == (status, reason, label, False)
    assert entry['text'] == 'Regar' and entry['attempts'] == 1
