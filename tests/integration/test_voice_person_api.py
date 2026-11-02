"""Explicit person metadata scopes and original-reader HTTP delivery."""

import asyncio
from contextlib import AsyncExitStack
import sqlite3
import threading

from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio
from starlette.requests import Request

from src.dashboard.api import DashboardAPI
from src.dashboard.auth import COOKIE_NAME
from src.dashboard.auth import _route_permissions, voice_person_scope_requested
from src.dashboard.authority import DashboardRole
from src.dashboard.live_erasure import build_live_services
from src.dashboard.live_retention_types import LiveSubscriptionMode
from tests.integration.test_voice_person_links import people_graph as _native_people_graph
from tests.mocks.rest_browser import ready_browser

people_graph = _native_people_graph


@pytest_asyncio.fixture
async def people_web(people_graph):
    g = people_graph
    async with AsyncExitStack() as stack:
        g.bus = await build_live_services({}, privacy=g.privacy, owner=g.owner, shutdown=stack)
        g.api = DashboardAPI(authority=g.authority, event_bus=g.bus, voice_registry=g.voices,
            user_manager=g.manager, observability_use_mocks=False)
        stack.push_async_callback(g.api.stop)
        g.hold_profiles = False
        g.entered, g.release = asyncio.Event(), asyncio.Event()
        @g.api.app.middleware('http')
        async def buffered(request,call_next):
            response = await call_next(request)
            if g.hold_profiles and request.method=='GET' and request.url.path=='/api/voices/profiles':
                g.entered.set()
                await g.release.wait()
            return response
        async with AsyncClient(transport=ASGITransport(app=g.api.app), base_url='http://testserver') as client:
            assert (await client.post('/api/admin/auth/login', json={'token':'synthetic-person-root'})).status_code == 200
            g.client = client
            g.reader = g.authority.resolve(None,client.cookies[COOKIE_NAME]).reader_binding(frozenset({'voices:read','voices:review'}))
            async with ready_browser(g.bus, version=3, reader=g.reader, mode=LiveSubscriptionMode.VOICES_CONTROL_ONLY) as peer:
                g.peer = peer
                try:
                    yield g
                finally:
                    g.release.set()


def path(g):
    return '/api/voices/profiles/'+g.profile['id']+'/person'


def body(g, user_id, revision=0):
    return {'user_id':user_id, 'expected_profile_revision':g.profile['revision'], 'expected_link_revision':revision}


async def profiles(g, *, person=False):
    return await g.client.get('/api/voices/profiles'+('?include_person=true' if person else ''),
        headers={'X-KZA-Live-Connection':g.peer.connection})


async def test_link_api_preserves_ordinary_shape_and_acoustic_profile(people_web):
    g = people_web
    response = await g.client.put(path(g), json=body(g,g.a))
    assert response.status_code == 200 and response.json() == {'ok':True,'profile_id':g.profile['id'],'link_revision':1}
    ordinary = (await profiles(g)).json()['snapshot']['rows'][0]
    assert set(ordinary) == {'id','name','revision','confirmed_at','confirmed_by','expires_at'}
    enriched = await profiles(g,person=True)
    assert enriched.status_code == 200
    row = enriched.json()['snapshot']['rows'][0]
    assert set(row) == set(ordinary) | {'person_link'}
    assert row['person_link'] == {'user_id':g.a,'revision':1}
    assert {key:value for key,value in row.items() if key!='person_link'} == ordinary
    assert (await g.client.put(path(g),json=body(g,None,1))).json()['link_revision'] == 2
    assert (await g.client.put(path(g),json=body(g,g.b,0))).status_code == 409
    assert (await g.voices.acall(g.voices.ledger.snapshot))['profiles'] == [g.profile]


@pytest.mark.parametrize('payload', [
    {'expected_profile_revision':1,'expected_link_revision':0},
    {'user_id':None,'expected_profile_revision':True,'expected_link_revision':0},
    {'user_id':None,'expected_profile_revision':1,'expected_link_revision':-1},
    {'user_id':None,'expected_profile_revision':1,'expected_link_revision':0,'role':'operator'},
    {'user_id':None,'expected_profile_revision':1,'expected_link_revision':0,'consent':True},
])
async def test_closed_link_body_never_selects_roles_or_consent(people_web,payload):
    g = people_web
    assert (await g.client.put(path(g),json=payload)).status_code == 422
    with sqlite3.connect(g.database) as db:
        assert db.execute('SELECT count(*) FROM voice_profile_people').fetchone()[0] == 0


async def test_home_identity_cannot_project_or_write_person_metadata(people_web,monkeypatch):
    g = people_web
    grant = await g.authority.issue_credential(g.root,subject='synthetic-home-person',role=DashboardRole.HOME,
        expires_at=g.clock.now+60)
    calls = []
    original = g.voices.acall
    async def observed(function,*args):
        calls.append(function.__name__)
        return await original(function,*args)
    monkeypatch.setattr(g.voices,'acall',observed)
    async with AsyncClient(transport=ASGITransport(app=g.api.app),base_url='http://testserver') as client:
        assert (await client.post('/api/admin/auth/login',json={'token':grant.token})).status_code == 200
        assert (await client.get('/api/voices/profiles?include_person=true')).status_code == 403
        assert (await client.put(path(g),json=body(g,g.a))).status_code == 403
    assert calls == []


async def test_enriched_read_requires_same_private_connection_and_original_reader(people_web):
    g = people_web
    assert (await g.client.get('/api/voices/profiles?include_person=true')).status_code == 428
    async with AsyncClient(transport=ASGITransport(app=g.api.app),base_url='http://testserver') as other:
        assert (await other.post('/api/admin/auth/login',json={'token':'synthetic-person-root'})).status_code == 200
        assert (await other.get('/api/voices/profiles?include_person=true',
            headers={'X-KZA-Live-Connection':g.peer.connection})).status_code == 428


async def test_retiring_target_does_not_fabricate_a_confirmed_unlink(people_web):
    g = people_web
    assert (await g.client.put(path(g),json=body(g,g.a))).status_code == 200
    await g.authority.retire_user(g.root,g.a)
    response = await profiles(g,person=True)
    assert response.status_code == 404 and 'person_link' not in response.text
    assert (await profiles(g)).status_code == 200
    assert (await g.client.put(path(g),json=body(g,None,1))).status_code == 200
    assert (await profiles(g,person=True)).json()['snapshot']['rows'][0]['person_link'] == {'user_id':None,'revision':2}


@pytest.mark.parametrize('change',['target','replace','unlink'])
async def test_delivered_person_view_retires_its_profile_channel_until_ack(people_web,change):
    g = people_web
    await g.client.put(path(g),json=body(g,g.a))
    batch = (await profiles(g,person=True)).json()['snapshot']
    state = g.api.app.state.rest_delivery_owner.connection(g.peer.connection,reader=g.reader,
        minimum_protocol=3,channel='voices:profiles')
    assert state.snapshots[batch['event_id']].available(g.privacy.ledger,g.clock.now)
    if change == 'target':
        await g.authority.retire_user(g.root,g.a)
    else:
        assert (await g.client.put(path(g),json=body(g,g.b if change=='replace' else None,1))).status_code == 200
    assert not state.snapshots[batch['event_id']].available(g.privacy.ledger,g.clock.now)
    async with asyncio.timeout(3):
        frame = await g.peer.outgoing.get()
    assert frame['type'] == 'rest_retire' and frame['channel'] == 'voices:profiles'
    assert frame['payload']['event_ids'] == [batch['event_id']]
    assert await g.bus._retention.receipts.pending(g.sources[0].conversation_id) == 1
    await g.peer.message({'type':'live_ack','receipts':[batch['receipt']]})
    async with asyncio.timeout(3):
        while await g.bus._retention.receipts.pending(g.sources[0].conversation_id):
            await asyncio.sleep(.001)
    assert (await g.voices.acall(g.voices.ledger.snapshot))['profiles'] == [g.profile]


async def test_account_delete_retires_then_clears_links_before_json(people_web):
    g = people_web
    await g.client.put(path(g),json=body(g,g.a))
    grant = await g.authority.issue_credential(g.root,subject='synthetic-personal-a',role=DashboardRole.HOME,
        user_id=g.a,expires_at=g.clock.now+60)
    personal = await g.authority.login(grant.token)
    response = await g.client.delete('/api/admin/users/'+g.a)
    assert response.status_code == 200
    assert not personal.lease.current() and g.authority.resolve(grant.token) is None
    assert g.manager.get_user(g.a) is None and g.links.user_links_cleared(frozenset({g.a}))
    assert (await profiles(g,person=True)).json()['snapshot']['rows'][0]['person_link'] == {'user_id':None,'revision':2}
    assert (await g.voices.acall(g.voices.ledger.snapshot))['profiles'] == [g.profile]


async def test_managed_catalog_adds_its_own_pure_link_proof(people_web):
    g = people_web
    await g.client.put(path(g),json=body(g,g.a))
    managed = await g.voices.load_accounts(str(g.manager.data_path),retirement_proof=g.authority.users_retired)
    await g.authority.retire_user(g.root,g.a)
    with pytest.raises(PermissionError,match='voice_links'):
        await asyncio.to_thread(managed.remove_user,g.a)
    assert managed.get_user(g.a) is not None
    await g.voices.acall(g.links.clear_user_links,frozenset({g.a}),g.authority.users_retired)
    assert (await asyncio.to_thread(managed.remove_user,g.a))[0]


@pytest.mark.parametrize('stage',['read','headers','buffer'])
@pytest.mark.parametrize('change',['target','replace','unlink'])
async def test_held_enriched_response_drops_retired_relationship_and_cursor(people_web,monkeypatch,stage,change):
    g = people_web
    assert (await g.client.put(path(g),json=body(g,g.a))).status_code==200
    original = g.voices.acall
    async def held(function,*args):
        value = await original(function,*args)
        if stage=='read' and function.__name__=='project':
            g.entered.set()
            await g.release.wait()
        return value
    monkeypatch.setattr(g.voices,'acall',held)
    g.hold_profiles = stage=='buffer'
    messages = []
    async def application(scope,receive,send):
        async def actual(message):
            if stage=='headers' and message['type']=='http.response.start':
                g.entered.set()
                await g.release.wait()
            messages.append(message.copy())
            await send(message)
        await g.api.app(scope,receive,actual)
    async with AsyncClient(transport=ASGITransport(app=application),base_url='http://testserver',
            cookies=g.client.cookies) as client:
        pending = asyncio.create_task(client.get('/api/voices/profiles?include_person=true',
            headers={'X-KZA-Live-Connection':g.peer.connection}))
        try:
            await asyncio.wait_for(g.entered.wait(),3)
            if change=='target':
                await g.authority.retire_user(g.root,g.a)
            else:
                assert (await g.client.put(path(g),json=body(g,g.b if change=='replace' else None,1))).status_code==200
            g.release.set()
            response = await pending
            assert response.status_code in {200,503}
            assert g.a not in response.text and g.profile['name'] not in response.text
            if response.status_code==200:
                assert response.json().get('snapshot') is None
                assert response.json().get('next_cursor') is None
        finally:
            g.release.set()
            await asyncio.gather(pending,return_exceptions=True)
    assert g.a.encode() not in b''.join(message.get('body',b'') for message in messages)


@pytest.mark.parametrize('stage',['cleanup_before','cleanup_after','catalog_replace'])
async def test_partial_account_retirement_retries_without_restoring_link_or_grant(people_web,monkeypatch,stage):
    g = people_web
    await g.client.put(path(g),json=body(g,g.a))
    original = g.links.clear_user_links
    def clear(*args):
        if stage=='cleanup_before':
            raise OSError('synthetic cleanup unavailable before commit')
        original(*args)
        raise OSError('synthetic cleanup result unavailable after commit')
    before = g.manager.data_path.read_bytes(),g.manager._version
    with monkeypatch.context() as changes:
        if stage.startswith('cleanup'):
            changes.setattr(g.links,'clear_user_links',clear)
        else:
            def replace(*args):
                raise OSError('synthetic catalog replace unavailable')
            changes.setattr('src.users.user_manager.os.replace',replace)
        response = await g.client.delete('/api/admin/users/'+g.a)
    assert response.status_code==503 and response.json()['detail']['credentials_retired'] is True
    assert g.authority.users_retired(frozenset({g.a}))
    assert g.manager.get_user(g.a) is not None
    assert (g.manager.data_path.read_bytes(),g.manager._version)==before
    with sqlite3.connect(g.database) as db:
        link = db.execute('SELECT user_id,revision FROM voice_profile_people').fetchone()
    assert link==((g.a,1) if stage=='cleanup_before' else (None,2))
    assert (await g.client.delete('/api/admin/users/'+g.a)).status_code==200
    assert g.manager.get_user(g.a) is None
    with sqlite3.connect(g.database) as db:
        assert db.execute('SELECT user_id,revision FROM voice_profile_people').fetchone()==(None,2)


async def test_cancelled_observer_and_closing_admin_owner_still_drain_link_cleanup(people_web,monkeypatch):
    g = people_web
    await g.client.put(path(g),json=body(g,g.a))
    original = g.links.clear_user_links
    entered, release = threading.Event(),threading.Event()
    calls = []
    def clear(*args):
        calls.append(True)
        entered.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(g.links,'clear_user_links',clear)
    request = asyncio.create_task(g.client.delete('/api/admin/users/'+g.a))
    closing = None
    try:
        assert await asyncio.to_thread(entered.wait,3)
        assert g.authority.users_retired(frozenset({g.a}))
        assert g.manager.get_user(g.a) is not None
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        closing = g.api.admin_work.begin_close()
        await asyncio.sleep(0)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.gather(request,return_exceptions=True)
        if closing is not None:
            await closing
    assert calls==[True] and g.manager.get_user(g.a) is None
    assert g.links.user_links_cleared(frozenset({g.a}))


@pytest.mark.parametrize('query,person',[
    ('include_person=f',False),('include_person=n',False),('include_person=OFF',False),
    ('include_person=true&include_person=f',False),('include_person=f&include_person=true',True),
])
async def test_person_scope_and_query_boolean_select_the_same_final_parameter(people_web,query,person):
    g = people_web
    request = Request({'type':'http','method':'GET','path':'/api/voices/profiles',
        'query_string':query.encode(),'headers':[]})
    assert voice_person_scope_requested(request) is person
    response = await g.client.get('/api/voices/profiles?'+query,
        headers={'X-KZA-Live-Connection':g.peer.connection})
    assert response.status_code==200
    assert ('person_link' in response.json()['snapshot']['rows'][0]) is person


def test_metadata_write_needs_review_and_administration_independently_of_grant_management():
    request = Request({'type':'http','method':'PUT','path':'/api/voices/profiles/00000000-0000-4000-8000-000000000000/person',
        'query_string':b'','headers':[]})
    assert _route_permissions(request)==frozenset({'voices:review','admin:manage'})


async def test_retired_requester_cleans_durable_links_but_cannot_borrow_replacement_login_for_json(people_web,monkeypatch):
    g = people_web
    await g.client.put(path(g),json=body(g,g.a))
    original = g.links.clear_user_links
    entered, release = threading.Event(),threading.Event()
    def clear(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(g.links,'clear_user_links',clear)
    before = g.manager.data_path.read_bytes(),g.manager._version
    pending = asyncio.create_task(g.client.delete('/api/admin/users/'+g.a))
    try:
        assert await asyncio.to_thread(entered.wait,3)
        assert g.authority.users_retired(frozenset({g.a}))
        assert (await g.client.post('/api/admin/auth/logout')).status_code==200
        assert (await g.client.post('/api/admin/auth/login',json={'token':'synthetic-person-root'})).status_code==200
        replacement = g.client.cookies[COOKIE_NAME]
    finally:
        release.set()
    response = await pending
    assert response.status_code==403 and response.json()=={'detail':'dashboard_session_retired'}
    assert g.client.cookies[COOKIE_NAME]==replacement
    assert (g.manager.data_path.read_bytes(),g.manager._version)==before
    assert g.manager.get_user(g.a) is not None
    with sqlite3.connect(g.database) as db:
        assert db.execute('SELECT user_id,revision FROM voice_profile_people').fetchone()==(None,2)
    assert (await g.client.get('/api/admin/auth/whoami')).status_code==200
    assert (await g.client.delete('/api/admin/users/'+g.a)).status_code==200
    assert g.manager.get_user(g.a) is None
