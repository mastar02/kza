"""T035-U1: web list and reminder routes take authority and scope from the session.

A household session bound to a person sees shared tasks and that person's own;
a household session without a person (also a visitor, user answer Q17) and an
operator (user answer Q16, 'Solo tareas compartidas') see shared tasks only. The
client never names an owner. Every matrix entry is tagged for C02 (layer
``web``) and asserts every effect counter, the task status and the persistence.
"""

import asyncio
from contextlib import AsyncExitStack
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient
import pytest
from starlette.requests import Request

from src.core.account_metadata import ActiveAccounts
from src.dashboard.api import DashboardAPI
from src.dashboard.auth import _route_permissions
from src.dashboard.authority import DashboardRole, DashboardAuthority, EvaluationModality, EvaluationSample
from src.dashboard.home_catalog import HomeRoom
from src.lists.list_manager import ListManager
from src.lists.list_store import ListStore, OwnerType
from src.reminders.reminder_manager import ReminderManager
from src.reminders.reminder_scheduler import ReminderScheduler
from src.reminders.reminder_store import ReminderScope, ReminderState, ReminderStore
from tests.mocks.c02_matrix import C02Entry, effects
from tests.mocks.home_control import selected_actions
from tests.mocks.memory_graph import Clock
from tests.mocks.responses import voice_output

pytestmark = pytest.mark.asyncio

ACTION = {'domain': 'light', 'service': 'turn_on', 'entity_id': 'light.patio', 'data': {'brightness_pct': 40}}
DAY = 86400.0
SESSIONS = ('operator', 'home', 'visitor', 'ana', 'evaluator', 'none')


class FakeCatalog:
    """Home catalog with one room holding the patio light."""

    def __init__(self) -> None:
        self.reads = 0

    async def snapshot(self):
        self.reads += 1
        return SimpleNamespace(rooms=(HomeRoom('patio', 'Patio', ('light.patio',)),))


@pytest.fixture
async def tasks_graph(tmp_path, dashboard_auth):
    clock = Clock()
    async with AsyncExitStack() as stack:
        authority = DashboardAuthority(tmp_path / 'auth.db', clock=clock, session_ttl_s=600,
                                       account_provider=lambda: ActiveAccounts(True, frozenset({'ana', 'beto'})))
        stack.push_async_callback(authority.aclose)
        await authority.initialize()
        lists = ListStore(str(tmp_path / 'lists.db'))
        stack.push_async_callback(lists.close)
        await lists.initialize()
        reminders = ReminderStore(str(tmp_path / 'reminders.db'))
        stack.push_async_callback(reminders.close)
        await reminders.initialize()
        ha, tts, scheduler_now = AsyncMock(), voice_output(), [time.time()]
        scheduler = ReminderScheduler(reminders, tts, None, ha, {'output_zone': 'zone_living'},
                                      home_actions=selected_actions(ha, ('light.patio',)),
                                      clock=lambda: scheduler_now[0])
        catalog = FakeCatalog()
        api = DashboardAPI(authority=authority, list_manager=ListManager(lists), home_catalog=catalog,
                           reminder_manager=ReminderManager(reminders), reminder_scheduler=scheduler)
        stack.push_async_callback(api.stop)
        root = authority.resolve(dashboard_auth['Authorization'])

        async def bearer(**grant) -> dict:
            issued = await authority.issue_credential(root, **grant)
            return {'Authorization': f'Bearer {issued.token}'}

        evaluation = EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}),
                                      frozenset({EvaluationModality.AUDIO_PREVIEW}), clock.now + 600)
        headers = {
            'operator': dashboard_auth,
            'home': await bearer(subject='synthetic-home', role=DashboardRole.HOME, expires_at=clock.now + DAY),
            # Q17: household credential without a person, issued for about 7 days.
            'visitor': await bearer(subject='synthetic-visitor', role=DashboardRole.HOME,
                                    expires_at=clock.now + 7 * DAY),
            'ana': await bearer(subject='synthetic-ana', role=DashboardRole.HOME, user_id='ana',
                                expires_at=clock.now + DAY),
            'beto': await bearer(subject='synthetic-beto', role=DashboardRole.HOME, user_id='beto',
                                 expires_at=clock.now + DAY),
            'evaluator': await bearer(subject='synthetic-evaluator', role=DashboardRole.EVALUATOR,
                                      expires_at=clock.now + 600, samples=(evaluation,)),
            'none': {},
        }
        now = time.time()
        seeded = SimpleNamespace(
            S=await lists.create_list('compras', OwnerType.SHARED, None),
            A=await lists.create_list('LISTA_ANA', OwnerType.USER, 'ana'),
            B=await lists.create_list('LISTA_BETO', OwnerType.USER, 'beto'),
        )
        seeded.s1 = await lists.add_item(seeded.S.id, 'leche')
        seeded.a1 = await lists.add_item(seeded.A.id, 'ITEM_ANA', 'ana')
        seeded.b1 = await lists.add_item(seeded.B.id, 'ITEM_BETO', 'beto')
        seeded.RS = await reminders.create('household', 'Regar el patio', now + 3600, ha_actions=[ACTION],
                                           scope=ReminderScope.SHARED)
        seeded.RT = await reminders.create('household', 'Sacar la basura', now + 7200, scope=ReminderScope.SHARED)
        seeded.RA = await reminders.create('ana', 'RECORDATORIO_ANA', now + 3600)
        seeded.RB = await reminders.create('beto', 'RECORDATORIO_BETO', now + 3600)
        seeded.RF = await reminders.create('household', 'Ya avisado', now - 60, scope=ReminderScope.SHARED)
        assert await reminders.mark_fired(seeded.RF.id)
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://testserver') as client:
            yield SimpleNamespace(client=client, headers=headers, lists=lists, reminders=reminders, ha=ha, tts=tts,
                                  scheduler=scheduler, authority=authority, clock=clock, root=root, seeded=seeded,
                                  catalog=catalog, api=api, scheduler_now=scheduler_now)


async def snapshot(graph) -> dict:
    """Every row a task route may change."""
    lists = await graph.lists._work.read(lambda db: (
        {tuple(r) for r in db.execute('SELECT id, name, owner_type, owner_id FROM lists').fetchall()},
        {tuple(r) for r in db.execute('SELECT id, list_id, text, completed, added_by FROM list_items').fetchall()}))
    reminders = await graph.reminders._work.read(lambda db: {
        r[0]: (r[1], r[2]) for r in db.execute('SELECT id, state, user_id FROM reminders').fetchall()})
    return {'lists': lists[0], 'items': lists[1], 'reminders': reminders}


def measured(graph, before: dict, after: dict) -> dict:
    """Every C02 counter of one web request; the counters with no leaf in this graph stay zero."""
    return effects(
        list_mutations=len(before['lists'] ^ after['lists']) + len(before['items'] ^ after['items']),
        reminder_rows_delta=len(after['reminders']) - len(before['reminders']),
        reminder_transitions=sum(before['reminders'][k][0] != v[0] for k, v in after['reminders'].items()
                                 if k in before['reminders']),
        ha_service_calls=graph.ha.call_service_once.await_count,
        spoken_utterances=graph.tts.aspeak.await_count,
    )


def _actor(session: str) -> str:
    return 'web_person' if session in ('ana', 'beto') else 'web_home'


def _web(id_: str, session: str, method: str, path: str, body: dict | None, status: int, detail: str | None,
         family: str, effect_class: str, *, writes: dict | None = None, task_status: str | None = 'failed',
         persistence: str | None = None, visible: tuple[str, ...] = (), hidden: tuple[str, ...] = (),
         note: str = '') -> pytest.param:
    """One tagged web entry; ``task_status`` None is a gated rejection before any handler (no_turn)."""
    if persistence is None and task_status is not None:
        persistence = 'not_written' if method != 'GET' else 'not_required'
    if task_status == 'completed':
        persistence = 'committed' if method != 'GET' else 'not_required'
    entry = C02Entry(
        id_, f"{method} {path}" + (f" {body}" if body is not None else ''), 'web', family, effect_class,
        'no_turn' if task_status is None else task_status, None, task_status, persistence, 'none',
        effects=effects(**(writes or {})), actor=_actor(session),
        conditions={'session': session, 'http_status': status, 'detail': detail,
                    'visible': visible, 'hidden': hidden}, unit='T035-U1', note=note)
    return pytest.param(entry, method, path, body, id=id_)


SHARED_LIST = {'name': 'ferreteria', 'shared': True}
PRIVATE_LIST = {'name': 'PRIVADA_NUEVA'}
SHARED_REMINDER = {'text': 'Cerrar la puerta', 'trigger_at': 4_000_000_000.0, 'scope': 'shared'}
PRIVATE_REMINDER = {'text': 'PRIVADO_NUEVO', 'trigger_at': 4_000_000_000.0}
PRIVATE_TEXTS = ('LISTA_ANA', 'LISTA_BETO', 'RECORDATORIO_ANA', 'RECORDATORIO_BETO')

MATRIX = (
    # Q16: an operator holds shared tasks only.
    _web('C02-LST-0701', 'operator', 'GET', '/api/lists', None, 200, None, 'list', 'privacy_disclosure',
         task_status='completed', visible=('compras',), hidden=PRIVATE_TEXTS),
    _web('C02-LST-0702', 'operator', 'GET', '/api/lists?scope=personal', None, 403, 'personal_session_required',
         'list', 'privacy_disclosure'),
    _web('C02-LST-0703', 'operator', 'POST', '/api/lists', PRIVATE_LIST, 403, 'personal_session_required',
         'list', 'unauthorized_effect'),
    _web('C02-LST-0704', 'operator', 'POST', '/api/lists', SHARED_LIST, 200, None, 'list', 'false_confirmation',
         task_status='completed', writes={'list_mutations': 1}),
    _web('C02-LST-0705', 'operator', 'GET', '/api/lists/{A}/items', None, 404, 'list_not_found', 'list',
         'privacy_disclosure', hidden=('ITEM_ANA',)),
    _web('C02-LST-0706', 'operator', 'POST', '/api/lists/{A}/items', {'text': 'intruso'}, 404, 'list_not_found',
         'list', 'unauthorized_effect'),
    _web('C02-LST-0707', 'operator', 'DELETE', '/api/lists/{A}/items/{a1}', None, 404, 'List item not found',
         'list', 'unauthorized_effect'),
    _web('C02-LST-0708', 'operator', 'GET', '/api/lists/{S}/items', None, 200, None, 'list', 'privacy_disclosure',
         task_status='completed', visible=('leche',)),
    _web('C02-LST-0709', 'operator', 'GET', '/api/lists?user_id=ana', None, 422, 'user_id_not_accepted', 'list',
         'privacy_disclosure', hidden=PRIVATE_TEXTS),
    _web('C02-LST-0710', 'operator', 'POST', '/api/lists', {'name': 'x', 'user_id': 'ana'}, 422, None, 'list',
         'unauthorized_effect', task_status=None, note='pydantic extra_forbidden'),
    _web('C02-REM-0701', 'operator', 'GET', '/api/reminders', None, 200, None, 'reminder', 'privacy_disclosure',
         task_status='completed', visible=('Regar el patio', 'Sacar la basura'), hidden=PRIVATE_TEXTS),
    _web('C02-REM-0702', 'operator', 'GET', '/api/reminders?scope=personal', None, 403,
         'personal_session_required', 'reminder', 'privacy_disclosure'),
    _web('C02-REM-0703', 'operator', 'POST', '/api/reminders', PRIVATE_REMINDER, 403, 'personal_session_required',
         'reminder', 'unauthorized_effect'),
    _web('C02-REM-0704', 'operator', 'POST', '/api/reminders', SHARED_REMINDER, 200, None, 'reminder',
         'false_confirmation', task_status='completed', writes={'reminder_rows_delta': 1}),
    _web('C02-REM-0705', 'operator', 'DELETE', '/api/reminders/{RA}', None, 404, 'Reminder not found', 'reminder',
         'unauthorized_effect'),
    _web('C02-REM-0706', 'operator', 'DELETE', '/api/reminders/{RT}', None, 200, None, 'reminder',
         'false_confirmation', task_status='completed', writes={'reminder_transitions': 1}),
    # A household session without a person: shared tasks only.
    _web('C02-LST-0711', 'home', 'GET', '/api/lists', None, 200, None, 'list', 'privacy_disclosure',
         task_status='completed', visible=('compras',), hidden=PRIVATE_TEXTS),
    _web('C02-LST-0712', 'home', 'POST', '/api/lists', PRIVATE_LIST, 403, 'personal_session_required', 'list',
         'unauthorized_effect'),
    _web('C02-LST-0713', 'home', 'POST', '/api/lists/{S}/items', {'text': 'pan'}, 200, None, 'list',
         'false_confirmation', task_status='completed', writes={'list_mutations': 1}),
    _web('C02-LST-0714', 'home', 'DELETE', '/api/lists/{B}/items/{b1}', None, 404, 'List item not found', 'list',
         'unauthorized_effect'),
    _web('C02-REM-0707', 'home', 'GET', '/api/reminders?scope=personal', None, 403, 'personal_session_required',
         'reminder', 'privacy_disclosure'),
    _web('C02-REM-0708', 'home', 'POST', '/api/reminders', PRIVATE_REMINDER, 403, 'personal_session_required',
         'reminder', 'unauthorized_effect'),
    _web('C02-REM-0709', 'home', 'POST', '/api/reminders', {**SHARED_REMINDER, 'user_id': 'ana'}, 422, None,
         'reminder', 'unauthorized_effect', task_status=None, note='pydantic extra_forbidden'),
    _web('C02-REM-0710', 'home', 'DELETE', '/api/reminders/{RB}', None, 404, 'Reminder not found', 'reminder',
         'unauthorized_effect'),
    _web('C02-REM-0711', 'home', 'DELETE', '/api/reminders/{RF}', None, 409, 'reminder_not_active', 'reminder',
         'false_confirmation'),
    # Q17: a visitor is a household credential without a person.
    _web('C02-REM-0712', 'visitor', 'GET', '/api/reminders', None, 200, None, 'reminder', 'privacy_disclosure',
         task_status='completed', visible=('Regar el patio',), hidden=PRIVATE_TEXTS),
    _web('C02-REM-0713', 'visitor', 'POST', '/api/reminders', PRIVATE_REMINDER, 403, 'personal_session_required',
         'reminder', 'unauthorized_effect'),
    # A household session bound to a person: shared tasks and that person's own.
    _web('C02-LST-0715', 'ana', 'GET', '/api/lists', None, 200, None, 'list', 'privacy_disclosure',
         task_status='completed', visible=('compras', 'LISTA_ANA'), hidden=('LISTA_BETO',)),
    _web('C02-LST-0716', 'ana', 'GET', '/api/lists/{B}/items', None, 404, 'list_not_found', 'list',
         'privacy_disclosure', hidden=('ITEM_BETO',)),
    _web('C02-LST-0717', 'ana', 'POST', '/api/lists/{B}/items', {'text': 'intruso'}, 404, 'list_not_found', 'list',
         'unauthorized_effect'),
    _web('C02-LST-0718', 'ana', 'DELETE', '/api/lists/{A}/items/{b1}', None, 404, 'List item not found', 'list',
         'unauthorized_effect', note='item of another person, under her own list id'),
    _web('C02-LST-0719', 'ana', 'POST', '/api/lists', PRIVATE_LIST, 200, None, 'list', 'false_confirmation',
         task_status='completed', writes={'list_mutations': 1}),
    _web('C02-LST-0720', 'ana', 'GET', '/api/lists?user_id=beto', None, 422, 'user_id_not_accepted', 'list',
         'privacy_disclosure', hidden=PRIVATE_TEXTS),
    _web('C02-REM-0714', 'ana', 'GET', '/api/reminders', None, 200, None, 'reminder', 'privacy_disclosure',
         task_status='completed', visible=('Regar el patio', 'RECORDATORIO_ANA'), hidden=('RECORDATORIO_BETO',)),
    _web('C02-REM-0715', 'ana', 'DELETE', '/api/reminders/{RB}', None, 404, 'Reminder not found', 'reminder',
         'unauthorized_effect'),
    _web('C02-REM-0716', 'ana', 'POST', '/api/reminders', {**PRIVATE_REMINDER, 'user_id': 'beto'}, 422, None,
         'reminder', 'unauthorized_effect', task_status=None, note='pydantic extra_forbidden'),
    _web('C02-REM-0717', 'ana', 'DELETE', '/api/reminders/{RA}', None, 200, None, 'reminder', 'false_confirmation',
         task_status='completed', writes={'reminder_transitions': 1}),
    # Neither an evaluator nor an anonymous client reaches a task handler.
    _web('C02-LST-0721', 'evaluator', 'GET', '/api/lists', None, 403, 'dashboard_permission_denied', 'list',
         'privacy_disclosure', task_status=None, hidden=PRIVATE_TEXTS + ('compras',)),
    _web('C02-REM-0718', 'evaluator', 'POST', '/api/reminders', SHARED_REMINDER, 403,
         'dashboard_permission_denied', 'reminder', 'unauthorized_effect', task_status=None),
    _web('C02-REM-0719', 'none', 'DELETE', '/api/reminders/{RT}', None, 401, 'auth required', 'reminder',
         'unauthorized_effect', task_status=None),
    # T035-U1-RV1: a time or recurrence the scheduler cannot hold is refused before any row exists.
    _web('C02-REM-0720', 'visitor', 'POST', '/api/reminders', {**SHARED_REMINDER, 'trigger_at': 1e300}, 422, None,
         'reminder', 'unauthorized_effect', task_status=None, note='pydantic less_than'),
    _web('C02-REM-0721', 'visitor', 'POST', '/api/reminders',
         '{"text": "Cerrar la puerta", "trigger_at": Infinity, "scope": "shared"}', 422, None, 'reminder',
         'unauthorized_effect', task_status=None, note='pydantic finite_number'),
    _web('C02-REM-0722', 'visitor', 'POST', '/api/reminders',
         '{"text": "Cerrar la puerta", "trigger_at": NaN, "scope": "shared"}', 422, None, 'reminder',
         'unauthorized_effect', task_status=None, note='pydantic finite_number'),
    _web('C02-REM-0723', 'home', 'POST', '/api/reminders', {**SHARED_REMINDER, 'trigger_at': -1e300}, 422, None,
         'reminder', 'unauthorized_effect', task_status=None, note='pydantic greater_than'),
    _web('C02-REM-0724', 'home', 'POST', '/api/reminders', {**SHARED_REMINDER, 'recurrence': 'weekly:9'}, 422,
         None, 'reminder', 'unauthorized_effect', task_status=None, note='pydantic string_pattern_mismatch'),
    _web('C02-REM-0725', 'ana', 'POST', '/api/reminders', {**PRIVATE_REMINDER, 'text': 'x' * 501}, 422, None,
         'reminder', 'unauthorized_effect', task_status=None, note='pydantic string_too_long'),
    # T035-U1-RV2: a write route refuses a client owner in the query string too.
    _web('C02-LST-0722', 'ana', 'POST', '/api/lists?user_id=beto', PRIVATE_LIST, 422, 'user_id_not_accepted',
         'list', 'unauthorized_effect'),
    _web('C02-LST-0723', 'ana', 'POST', '/api/lists/{A}/items?user_id=beto', {'text': 'pan'}, 422,
         'user_id_not_accepted', 'list', 'unauthorized_effect'),
    _web('C02-LST-0724', 'ana', 'DELETE', '/api/lists/{A}/items/{a1}?owner=beto', None, 422,
         'unexpected_query_parameter', 'list', 'unauthorized_effect'),
    _web('C02-REM-0726', 'home', 'POST', '/api/reminders?user_id=ana', SHARED_REMINDER, 422, 'user_id_not_accepted',
         'reminder', 'unauthorized_effect'),
    _web('C02-REM-0727', 'ana', 'DELETE', '/api/reminders/{RA}?user_id=ana', None, 422, 'user_id_not_accepted',
         'reminder', 'unauthorized_effect'),
)


@pytest.mark.parametrize('entry, method, path, body', MATRIX)
async def test_task_route_role_scope_matrix(tasks_graph, entry, method, path, body):
    graph = tasks_graph
    seeded = graph.seeded
    target = path.format(**{key: getattr(seeded, key).id for key in ('S', 'A', 'B', 's1', 'a1', 'b1', 'RS', 'RT',
                                                                      'RA', 'RB', 'RF')})
    before = await snapshot(graph)
    headers = graph.headers[entry.conditions['session']]
    if isinstance(body, str):  # a raw JSON body (Infinity, NaN) that httpx would not encode
        response = await graph.client.request(method, target, content=body,
                                              headers={**headers, 'Content-Type': 'application/json'})
    else:
        response = await graph.client.request(method, target, json=body, headers=headers)
    after = await snapshot(graph)
    assert response.status_code == entry.conditions['http_status'], response.text
    payload = response.json()
    if entry.conditions['detail'] is not None:
        assert payload['detail'] == entry.conditions['detail']
    assert measured(graph, before, after) == entry.effects
    if entry.task_status is None:
        assert 'outcome' not in payload  # refused before any task handler ran
    elif isinstance(payload, dict) and 'outcome' in payload:
        assert payload['outcome']['status'] == entry.task_status
        assert payload['outcome']['persistence'] == entry.persistence
    else:
        assert entry.task_status == 'completed' and method == 'GET' and response.status_code == 200
    for text in entry.conditions['visible']:
        assert text in response.text
    for text in entry.conditions['hidden']:
        assert text not in response.text


async def test_matrix_is_tagged_for_c02():
    entries = [param.values[0] for param in MATRIX]
    assert len({entry.id for entry in entries}) == len(entries)
    assert {entry.layer for entry in entries} == {'web'} and {entry.unit for entry in entries} == {'T035-U1'}
    assert {entry.actor for entry in entries} == {'web_home', 'web_person'}
    assert {entry.conditions['session'] for entry in entries} == set(SESSIONS)


async def test_task_routes_need_task_permissions_and_status_stays_a_diagnosis():
    def permissions(method, path):
        return set(_route_permissions(Request({'type': 'http', 'method': method, 'path': path,
                                               'query_string': b'', 'headers': []})))
    for path in ('/api/lists', '/api/lists/x/items', '/api/reminders'):
        assert permissions('GET', path) == {'tasks:read'}
    for method, path in (('POST', '/api/lists'), ('POST', '/api/lists/x/items'), ('DELETE', '/api/lists/x/items/y'),
                         ('POST', '/api/reminders'), ('DELETE', '/api/reminders/x')):
        assert permissions(method, path) == {'tasks:write'}
    assert permissions('GET', '/api/reminders/status') == {'diagnostics:read'}


async def test_personal_reads_return_only_the_session_persons_own(tasks_graph):
    graph = tasks_graph
    seeded = graph.seeded
    ana = graph.headers['ana']
    lists = (await graph.client.get('/api/lists?scope=personal', headers=ana)).json()
    assert [(row['id'], row['owner_type'], row['owner_id']) for row in lists] == [(seeded.A.id, 'user', 'ana')]
    shared = (await graph.client.get('/api/lists?scope=shared', headers=ana)).json()
    assert [(row['id'], row['owner_id']) for row in shared] == [(seeded.S.id, None)]
    reminders = (await graph.client.get('/api/reminders?scope=personal', headers=ana)).json()
    assert [row['id'] for row in reminders] == [seeded.RA.id]
    items = (await graph.client.get(f'/api/lists/{seeded.A.id}/items', headers=ana)).json()
    assert [row['text'] for row in items] == ['ITEM_ANA']
    beto = (await graph.client.get('/api/reminders', headers=graph.headers['beto'])).json()
    assert {row['id'] for row in beto} == {seeded.RS.id, seeded.RT.id, seeded.RB.id}


async def test_created_objects_take_their_owner_from_the_session(tasks_graph):
    graph = tasks_graph
    ana = graph.headers['ana']
    personal = (await graph.client.post('/api/lists', json={'name': 'mia'}, headers=ana)).json()
    shared = (await graph.client.post('/api/lists', json={'name': 'casa', 'shared': True}, headers=ana)).json()
    assert (personal['owner_type'], personal['owner_id']) == ('user', 'ana')
    # A shared object never records the person whose session created it.
    assert (shared['owner_type'], shared['owner_id']) == ('shared', None)
    assert [(row.owner_type, row.owner_id) for row in await graph.lists.get_lists_for_user('ana')
            if row.id == shared['id']] == [(OwnerType.SHARED, None)]
    item = (await graph.client.post(f"/api/lists/{shared['id']}/items", json={'text': 'pan'}, headers=ana)).json()
    assert item['outcome']['persistence'] == 'committed'
    private = (await graph.client.post('/api/reminders', json=PRIVATE_REMINDER, headers=ana)).json()
    household = (await graph.client.post('/api/reminders', json=SHARED_REMINDER, headers=ana)).json()
    assert (await graph.reminders.get_by_id(private['id'])).user_id == 'ana'
    assert (await graph.reminders.get_by_id(household['id'])).user_id == 'household'
    beto = await graph.client.get('/api/reminders', headers=graph.headers['beto'])
    assert 'PRIVADO_NUEVO' not in beto.text and household['id'] in beto.text


async def test_shared_reminder_read_carries_its_actions_with_room_and_no_service_data(tasks_graph):
    graph = tasks_graph
    seeded = graph.seeded
    response = await graph.client.get('/api/reminders', headers=graph.headers['home'])
    assert response.status_code == 200 and response.headers['Cache-Control'] == 'no-store'
    by_id = {row['id']: row for row in response.json()}
    assert by_id[seeded.RS.id]['ha_actions'] == [
        {'domain': 'light', 'service': 'turn_on', 'entity_id': 'light.patio', 'room': 'Patio'}]
    assert by_id[seeded.RT.id]['ha_actions'] is None
    assert 'brightness_pct' not in response.text and 'data' not in by_id[seeded.RS.id]['ha_actions'][0]
    assert graph.catalog.reads == 1


async def test_reminder_read_survives_an_unreadable_catalog(tasks_graph, monkeypatch):
    graph = tasks_graph
    monkeypatch.setattr(graph.catalog, 'snapshot', AsyncMock(side_effect=RuntimeError('ha down')))
    response = await graph.client.get('/api/reminders', headers=graph.headers['home'])
    assert response.status_code == 200
    by_id = {row['id']: row for row in response.json()}
    assert by_id[graph.seeded.RS.id]['ha_actions'][0]['room'] is None


@pytest.mark.parametrize('deleted', [True, False])
async def test_household_deletes_a_shared_reminder_with_actions_and_they_never_run(tasks_graph, deleted):
    graph = tasks_graph
    reminder = graph.seeded.RS
    if deleted:
        response = await graph.client.delete(f'/api/reminders/{reminder.id}', headers=graph.headers['home'])
        assert response.status_code == 200 and response.json()['outcome']['persistence'] == 'committed'
        assert (await graph.reminders.get_by_id(reminder.id)).state is ReminderState.CANCELLED
    graph.scheduler_now[0] = reminder.trigger_at + 1
    await graph.scheduler._deliver_due(reminder.trigger_at + 1)
    # The control run (not deleted) shows the same scheduler would have run the action.
    assert graph.ha.call_service_once.await_count == (0 if deleted else 1)


@pytest.mark.parametrize('session', ['home', 'visitor'])
async def test_visitor_and_household_credentials_are_accepted_until_their_own_expiry(tasks_graph, session):
    """Q17: the visitor credential works at day 6 and is refused after its ~7 day expiry."""
    graph = tasks_graph
    headers = graph.headers[session]
    graph.clock.now += 6 * DAY
    response = await graph.client.get('/api/reminders', headers=headers)
    if session == 'home':
        assert response.status_code == 401  # issued for one day
        return
    assert response.status_code == 200 and 'RECORDATORIO_ANA' not in response.text
    assert (await graph.client.post('/api/reminders', json=PRIVATE_REMINDER, headers=headers)).json()[
        'detail'] == 'personal_session_required'
    graph.clock.now += DAY + 1
    assert (await graph.client.get('/api/reminders', headers=headers)).status_code == 401


@pytest.mark.parametrize('withdrawal', ['revoke', 'retire_person', 'expire', 'revise_without_person'])
@pytest.mark.parametrize('operation', ['create_list', 'add_item', 'create_reminder', 'cancel_reminder'])
async def test_authority_withdrawn_before_commit_writes_nothing(tasks_graph, monkeypatch, withdrawal, operation):
    graph = tasks_graph
    seeded = graph.seeded
    issued = await graph.authority.issue_credential(graph.root, subject='synthetic-ana-2', role=DashboardRole.HOME,
                                                    user_id='ana', expires_at=graph.clock.now + DAY)
    headers = {'Authorization': f'Bearer {issued.token}'}
    store = graph.reminders if operation.endswith('reminder') else graph.lists
    entered, release = threading.Event(), threading.Event()
    original = store._work.write

    async def held(function, *, guard=None, **kwargs):
        def provisional(database):
            result = function(database)
            entered.set()
            assert release.wait(5)
            return result
        return await original(provisional, guard=guard, **kwargs)

    monkeypatch.setattr(store._work, 'write', held)
    request = {
        'create_list': ('POST', '/api/lists', PRIVATE_LIST),
        'add_item': ('POST', f'/api/lists/{seeded.A.id}/items', {'text': 'RETIRADO'}),
        'create_reminder': ('POST', '/api/reminders', PRIVATE_REMINDER),
        'cancel_reminder': ('DELETE', f'/api/reminders/{seeded.RA.id}', None),
    }[operation]
    before = await snapshot(graph)
    pending = asyncio.create_task(graph.client.request(request[0], request[1], json=request[2], headers=headers))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if withdrawal == 'revoke':
            assert await graph.authority.revoke_credential(graph.root, issued.identity)
        elif withdrawal == 'retire_person':
            assert await graph.authority.retire_user(graph.root, 'ana')
        elif withdrawal == 'expire':
            graph.clock.now += DAY + 1
        else:
            assert await graph.authority.revise_credential(graph.root, issued.identity, role=DashboardRole.HOME,
                                                           expires_at=graph.clock.now + DAY)
    finally:
        release.set()
    response = await pending
    # A withdrawn session receives only the fixed delivery denial, never a receipt or content.
    assert response.status_code == 403 and response.json() == {'detail': 'dashboard_session_retired'}
    monkeypatch.setattr(store._work, 'write', original)
    assert measured(graph, before, await snapshot(graph)) == effects()


async def test_task_routes_without_their_manager_are_unavailable_with_an_outcome(dashboard_auth):
    api = DashboardAPI()
    async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://testserver',
                           headers=dashboard_auth) as client:
        for method, path, body in (('GET', '/api/lists', None), ('POST', '/api/lists', SHARED_LIST),
                                   ('GET', '/api/reminders', None), ('DELETE', '/api/reminders/x', None)):
            response = await client.request(method, path, json=body)
            assert response.status_code == 503
            assert response.json()['outcome']['handled'] is False
            assert response.json()['outcome']['persistence'] == ('not_required' if method == 'GET'
                                                                 else 'not_written')
    await api.stop()


@pytest.mark.parametrize('recurrence', ['daily', 'weekdays', 'weekly:1', 'weekly:7', 'monthly:1', 'monthly:31'])
async def test_every_recurrence_the_scheduler_walks_is_accepted_and_reads_back(tasks_graph, recurrence):
    """T035-U1-RV1: the recurrence bound refuses only what the scheduler cannot walk."""
    graph = tasks_graph
    headers = graph.headers['visitor']
    created = await graph.client.post('/api/reminders', headers=headers,
                                      json={**SHARED_REMINDER, 'recurrence': recurrence})
    assert created.status_code == 200 and created.json()['outcome']['persistence'] == 'committed'
    listed = await graph.client.get('/api/reminders', headers=headers)
    assert listed.status_code == 200
    assert {row['id']: row['recurrence'] for row in listed.json()}[created.json()['id']] == recurrence
