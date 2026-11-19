"""Voice review bodies and permanent labels require distinct actual page receipts."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import re
import sqlite3
import threading
import time
from uuid import uuid4
from weakref import ref

import httpx
import pytest

from src.dashboard import authority as authority_module
from src.dashboard import voice_api as voice_api_module
from src.dashboard.api import DashboardAPI
from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import DashboardRole, EvaluationModality, EvaluationSample
from src.dashboard.live_retention_types import VOICE_CHANNELS, LiveSubscriptionMode, RevisionStamp
from src.memory import fact_ledger as fact_ledger_module
from src.memory.fact_ledger import FactLedger, MemoryOrigin, MemoryReviewer, MemorySource
from src.users import voice_ledger as voice_ledger_module
from src.users.voice_projection import VoiceProjection
from src.users.voice_ledger import VoiceLedger
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_live_stream_receipts import flush
from tests.integration.test_voice_capture_order import CAPTURE_ROW, capture
from tests.integration.test_voice_capture_review import observe
from tests.mocks.ambient_graph import Clock
from tests.mocks.rest_browser import ready_browser
from tests.mocks.voice_registry import ACTOR,seed,web_graph
from tests.mocks.browser_authority import fixture_token

pytestmark = pytest.mark.asyncio


async def test_same_credential_and_subject_cannot_exchange_individual_login_connections(tmp_path,dashboard_auth,monkeypatch):
    async with web_graph(tmp_path) as (_,_,voices,_,bus):
        authority = bus.test_authority
        credential = await authority.issue_credential(authority.resolve(fixture_token()),subject='same-synthetic-subject',
                    role=DashboardRole.OPERATOR,expires_at=time.time()+120)
        api = DashboardAPI(voice_registry=voices,event_bus=bus,authority=authority)
        calls = []
        project = voices.ledger.project

        def observed(*args):
            calls.append(True)
            return project(*args)

        monkeypatch.setattr(voices.ledger,'project',observed)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as first, \
                    httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as second:
                for client in (first,second):
                    assert (await client.post('/api/admin/auth/login',json={'token':credential.token})).status_code==200
                first_reader = authority.resolve(None,first.cookies[COOKIE_NAME]).reader_binding()
                second_reader = authority.resolve(None,second.cookies[COOKIE_NAME]).reader_binding()
                assert first_reader.namespace!=second_reader.namespace
                page_id = str(uuid4())
                async with ready_browser(bus,version=3,reader=first_reader,client_id=page_id) as a, \
                        ready_browser(bus,version=3,reader=second_reader,client_id=page_id) as b:
                    for client,connection in ((first,b.connection),(second,a.connection)):
                        assert (await client.get('/api/voices/profiles',headers={'X-KZA-Live-Connection':connection})).status_code==428
                        assert (await client.get('/api/admin/auth/whoami')).status_code==200
                    assert (await first.get('/api/voices/profiles',headers={**dashboard_auth,'X-KZA-Live-Connection':a.connection})).status_code==428
                    assert calls==[]
                    assert (await first.get('/api/voices/profiles',headers={'X-KZA-Live-Connection':a.connection})).status_code==200
                    assert calls==[True]
        finally:
            await api.stop()


async def step(privacy,operation):
    def read():
        with privacy.ledger._transaction() as db:
            return dict(db.execute("SELECT s.* FROM erasure_steps s JOIN erasure_targets t ON t.id=s.target_id WHERE s.operation_id=? AND t.participant='streams' AND t.version=4",(operation['id'],)).fetchone())
    return await privacy.acall(read)


@pytest.mark.parametrize('path',['/api/voices','/api/voices/profiles','/api/voices/captures/unknown/review'])
async def test_voice_reads_require_current_auth_and_confirmed_version_three_before_native_read(tmp_path,dashboard_auth,monkeypatch,path):
    async with web_graph(tmp_path) as (_,_,voices,_,bus):
        calls = []
        monkeypatch.setattr(voices.ledger,'project',lambda *args:calls.append(args))
        app = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority).app
        async with ready_browser(bus) as old:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
                assert (await client.get(path)).status_code==401
                assert (await client.get(path,headers=dashboard_auth)).status_code==428
                assert (await client.get(path,headers={**dashboard_auth,'X-KZA-Live-Connection':old.connection})).status_code==428
        assert not calls


async def test_review_ticket_precedes_outer_body_and_early_ack_cannot_retire_actual_send(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (privacy,owner,voices,clock,bus):
        async with ready_browser(bus,version=3) as peer:
            source,identity = await observe(voices,clock,text='Synthetic retained listening transcript')
            api = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority)
            @api.app.middleware('http')
            async def buffered(request,call_next):
                return await call_next(request)
            entered,release = asyncio.Event(),asyncio.Event()
            documents = []
            receipts = bus._retention.receipts
            async def actual(scope,receive,send):
                async def transmit(message):
                    if message['type']=='http.response.body' and b'Synthetic retained listening transcript' in message.get('body',b''):
                        documents.append(json.loads(message['body']))
                        assert await receipts.pending(source.conversation_id)==1
                        entered.set()
                        await release.wait()
                    await send(message)
                await api.app(scope,receive,transmit)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=actual),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                observing = asyncio.create_task(client.get('/api/voices/captures/'+identity+'/review'))
                try:
                    await until(entered.is_set)
                    ticket = documents[0]['snapshot']['receipt']
                    await peer.message({'type':'live_ack','receipts':[dict(ticket,nonce=str(uuid4()))]})
                    await asyncio.sleep(.01)
                    assert await receipts.pending(source.conversation_id)==1
                    await peer.message({'type':'live_ack','receipts':[ticket]})
                    async with asyncio.timeout(3):
                        while await receipts.pending(source.conversation_id):
                            await asyncio.sleep(.001)
                    operation = await owner.request(source.conversation_id,MemoryReviewer('synthetic-admin'))
                    await flush(owner)
                    assert (await step(privacy,operation))['state']!='complete'
                    release.set()
                    assert (await observing).status_code==200
                    await until(lambda:bus.resource_status()['rest_operations']==0)
                    clock.now+=2
                    await flush(owner)
                    assert (await step(privacy,operation))['state']=='complete'
                    assert (await privacy.acall(privacy.ledger.erasures.snapshot,operation['id']))['state']!='complete'
                finally:
                    release.set()
                    await observing


async def test_confirmed_name_survives_raw_ttl_but_cannot_authorize_a_new_raw_ticket(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        sources,identities = await seed(voices,clock,ttl=10)
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic permanent voice',str(uuid4()),ACTOR)
        clock.now+=10
        await voices.acall(voices.ledger.prune)
        async with ready_browser(bus,version=3) as peer:
            app = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                response = await client.get('/api/voices/profiles')
                assert response.status_code==200
                batch = response.json()['snapshot']
                assert batch['rows']==[profile] and batch['expires_at']==clock.now+300
                assert batch['channel']=='voices:profiles'
                assert await bus._retention.receipts.pending(sources[0].conversation_id)==1
                assert (await client.get('/api/voices/captures/'+identities[0]+'/review')).status_code==404
                receipts = bus._retention.receipts
                assert await receipts.issue(peer.connection,str(uuid4()),clock.now+300,tuple(sources),reader=peer.reader) is None
                assert await receipts.issue_profiles(peer.connection,str(uuid4()),clock.now+300,tuple(sources),((profile['id'],2),),reader=peer.reader) is None


@pytest.mark.parametrize('mode',[mode for mode in LiveSubscriptionMode if mode.permits_channel('voices:profiles')])
async def test_deleted_profile_retires_only_its_metadata_channel(tmp_path,dashboard_auth,mode):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        _,identities = await seed(voices,clock)
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic profile',str(uuid4()),ACTOR)
        async with ready_browser(bus,version=3,mode=mode) as peer:
            app = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                profiles = (await client.get('/api/voices/profiles')).json()['snapshot']
                captures = (await client.get('/api/voices')).json()['snapshot']
                assert (await client.request('DELETE','/api/voices/profiles/'+profile['id'],json={'expected_revision':1})).json()['deleted']
                # Captures without a profile association have no mutable label
                # to invalidate; control order must not hide a spurious retire.
                state = next(state for state in bus._retention._states.values() if state.browser.connection==peer.connection)
                assert state.snapshots[captures['event_id']].available(voices.privacy.ledger,clock.now)
                async with asyncio.timeout(3):
                    control = await peer.outgoing.get()
                assert control['type']=='rest_retire' and control['channel']=='voices:profiles'
                assert control['payload']['event_ids']==[profiles['event_id']]
                assert captures['event_id'] not in control['payload']['event_ids']


async def test_source_delete_routes_profile_capture_and_audio_controls_separately(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,owner,voices,clock,bus):
        samples = [await observe(voices,clock) for _ in range(3)]
        await voices.acall(voices.ledger.confirm,tuple(item[1] for item in samples),'Synthetic mixed channels',str(uuid4()),ACTOR)
        source,identity = samples[0]
        async with ready_browser(bus,version=3) as peer:
            app = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                batches = [(await client.get(path)).json()['snapshot'] for path in (
                    '/api/voices/profiles','/api/voices','/api/voices/captures/'+identity+'/review')]
                await owner.request(source.conversation_id,MemoryReviewer('synthetic-admin'))
                controls = []
                async with asyncio.timeout(3):
                    while len(controls)<3:
                        controls.append(await peer.outgoing.get())
                assert {control['channel'] for control in controls}=={batch['channel'] for batch in batches}
                for control in controls:
                    assert control['payload']['event_ids']==[next(batch['event_id'] for batch in batches if batch['channel']==control['channel'])]


async def test_profile_deleted_behind_framework_buffer_never_reaches_outer_private_body(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        _,identities = await seed(voices,clock)
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic buffered private name',str(uuid4()),ACTOR)
        async with ready_browser(bus,version=3) as peer:
            entered,release = asyncio.Event(),asyncio.Event()
            api = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority)
            @api.app.middleware('http')
            async def buffered(request,call_next):
                response = await call_next(request)
                entered.set()
                await release.wait()
                return response
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                observing = asyncio.create_task(client.get('/api/voices/profiles'))
                try:
                    await until(entered.is_set)
                    other = VoiceLedger(voices.privacy.ledger)
                    await voices.acall(other.delete_profile,profile['id'],1,ACTOR)
                    release.set()
                    response = await observing
                    assert 'Synthetic buffered private name' not in response.text
                    assert response.json()['snapshot'] is None
                finally:
                    release.set()
                    await observing


async def test_profile_metadata_ticket_cannot_launder_an_audio_payload_past_capture_ttl(tmp_path,dashboard_auth,monkeypatch):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        sources,identities = await seed(voices,clock,ttl=10)
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic profile',str(uuid4()),ACTOR)
        clock.now+=10
        def invalid(*args):
            return VoiceProjection([{**profile,'audio_base64':'Synthetic forbidden audio'}],tuple(sources),clock.now,clock.now+300,
                'voices:profiles',profile_refs=((profile['id'],1),))
        monkeypatch.setattr(voices.ledger,'project',invalid)
        async with ready_browser(bus,version=3) as peer:
            app = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                response = await client.get('/api/voices/profiles')
                assert response.status_code==503 and 'Synthetic forbidden audio' not in response.text
                assert await bus._retention.receipts.pending(sources[0].conversation_id)==0


async def test_cancelled_voice_http_observer_keeps_actual_unknown_native_read_in_local_receipt(tmp_path,dashboard_auth,monkeypatch):
    async with web_graph(tmp_path) as (privacy,owner,voices,clock,bus):
        await seed(voices,clock)
        entered,release = threading.Event(),threading.Event()
        project = voices.ledger.project
        def held(*args):
            entered.set()
            assert release.wait(3)
            return project(*args)
        monkeypatch.setattr(voices.ledger,'project',held)
        async with ready_browser(bus,version=3) as peer:
            app = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                observing = asyncio.create_task(client.get('/api/voices'))
                try:
                    await until(entered.is_set)
                    observing.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observing
                    operation = await owner.request(str(uuid4()),MemoryReviewer('synthetic-admin'))
                    await flush(owner)
                    assert (await step(privacy,operation))['pending_count']>=1
                    release.set()
                    await until(lambda:bus.resource_status()['rest_operations']==0)
                    clock.now+=2
                    await flush(owner)
                    assert (await step(privacy,operation))['state']=='complete'
                finally:
                    release.set()
                    await asyncio.gather(observing,return_exceptions=True)


async def test_profile_pagination_preserves_complete_dependency_budget_without_dropping_lineage(tmp_path):
    from dataclasses import replace
    from src.memory.fact_ledger import FactDependency
    from tests.mocks.memory_sources import memory_source
    from tests.mocks.voice_registry import VECTOR
    async with web_graph(tmp_path) as (privacy,_,voices,clock,_):
        def facts():
            result = []
            for index in range(260):
                fact = privacy.ledger.store('Synthetic dependency '+str(index),'fact',.9,(memory_source(now=clock.now),))
                result.append(FactDependency(fact.id,fact.revision,fact.expires_at))
            return result
        dependencies = await privacy.acall(facts)
        async def captures(parents):
            ids = []
            for offset in range(3):
                source = replace(memory_source(now=clock.now),fact_dependencies=tuple(parents[offset::3]))
                ids.append(await voices.acall(voices.ledger.put,'synthetic-budget-model',VECTOR,1.,(source,)))
            return tuple(ids)
        profiles = []
        for index in range(2):
            ids = await captures(dependencies[index*130:(index+1)*130])
            profiles.append(await voices.acall(voices.ledger.confirm,ids,'Synthetic bounded profile',str(uuid4()),ACTOR))
        first = await voices.acall(voices.ledger.project,'profiles',None,ACTOR)
        second = await voices.acall(voices.ledger.project,'profiles',first.next_cursor,ACTOR)
        assert len(first.rows)==len(second.rows)==1
        assert {row['id'] for batch in (first,second) for row in batch.rows}=={profile['id'] for profile in profiles}
        assert all(sum(len(source.fact_dependencies) for source in batch.sources)==130 for batch in (first,second))
        excessive = await captures(dependencies[:258])
        with pytest.raises(ValueError,match='voice_profile_source_capacity'):
            await voices.acall(voices.ledger.confirm,excessive,'Synthetic excessive profile',str(uuid4()),ACTOR)


async def test_suggestion_ticket_includes_the_permanent_profile_sources_without_extending_raw_capture(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,owner,voices,clock,bus):
        samples = [await observe(voices,clock,ttl=10) for _ in range(3)]
        profile = await voices.acall(voices.ledger.confirm,tuple(identity for _,identity in samples),'Synthetic source-linked suggestion',str(uuid4()),ACTOR)
        clock.now+=10
        await voices.acall(voices.ledger.prune)
        current,identity = await observe(voices,clock,ttl=60)
        async with ready_browser(bus,version=3) as peer:
            app = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                response = await client.get('/api/voices')
                assert response.status_code==200
                batch = response.json()['snapshot']
                assert batch['rows'][0]['id']==identity and batch['rows'][0]['suggested_profile']==profile['id']
                assert batch['expires_at']==current.expires_at
                parent = samples[0][0]
                assert await bus._retention.receipts.pending(parent.conversation_id)==1
                await owner.request(parent.conversation_id,MemoryReviewer('synthetic-admin'))
                async with asyncio.timeout(3):
                    control = await peer.outgoing.get()
                assert control['channel']=='voices:captures' and control['payload']['event_ids']==[batch['event_id']]


async def test_receipt_attributes_transitive_fact_conversations_before_parent_erasure(tmp_path,dashboard_auth):
    from dataclasses import replace
    from src.memory.fact_ledger import FactDependency
    from tests.mocks.memory_sources import memory_source
    from tests.mocks.voice_registry import VECTOR
    async with web_graph(tmp_path) as (privacy,owner,voices,clock,bus):
        origin = memory_source(now=clock.now)
        first = await privacy.acall(privacy.ledger.store,'Synthetic ancestor fact','fact',.9,(origin,))
        middle = replace(memory_source(now=clock.now),fact_dependencies=(FactDependency(first.id,first.revision,first.expires_at),))
        second = await privacy.acall(privacy.ledger.store,'Synthetic derived fact','fact',.9,(middle,))
        current = replace(memory_source(now=clock.now),fact_dependencies=(FactDependency(second.id,second.revision,second.expires_at),))
        await voices.acall(voices.ledger.put,'synthetic-parent-model',VECTOR,1.,(current,))
        async with ready_browser(bus,version=3) as peer:
            app = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                batch = (await client.get('/api/voices')).json()['snapshot']
                assert await bus._retention.receipts.pending(origin.conversation_id)==1
                operation = await owner.request(origin.conversation_id,MemoryReviewer('synthetic-admin'))
                await flush(owner)
                assert (await step(privacy,operation))['state']!='complete'
                await peer.message({'type':'live_ack','receipts':[batch['receipt']]})
                async with asyncio.timeout(3):
                    while await bus._retention.receipts.pending(origin.conversation_id):
                        await asyncio.sleep(.001)


async def test_partial_restoration_of_source_index_cannot_lose_a_pending_page_obligation(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (privacy,_,voices,clock,bus):
        sources,_ = await seed(voices,clock)
        async with ready_browser(bus,version=3) as peer:
            app = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                batch = (await client.get('/api/voices')).json()['snapshot']
                def damage():
                    with privacy.ledger._transaction() as db:
                        db.execute('DELETE FROM live_delivery_sources WHERE ticket=? AND conversation_id=?',
                            (batch['receipt']['ticket'],sources[0].conversation_id))
                await privacy.acall(damage)
                assert await bus._retention.receipts.pending(sources[0].conversation_id)==1
                assert await bus._retention.receipts.pending(str(uuid4()))==1


async def test_capture_metadata_cannot_declare_a_suggestion_without_its_reviewed_parent_proof(tmp_path,dashboard_auth,monkeypatch):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        sources,identities = await seed(voices,clock)
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic parent proof',str(uuid4()),ACTOR)
        row = dict((await voices.acall(voices.ledger.snapshot))['fingerprints'][0],
            suggested_profile=profile['id'],similarity=1.,association='suggested')
        def invalid(*args):
            return VoiceProjection([row],(sources[0],),clock.now,sources[0].expires_at,'voices:captures')
        monkeypatch.setattr(voices.ledger,'project',invalid)
        async with ready_browser(bus,version=3) as peer:
            app = DashboardAPI(voice_registry=voices,event_bus=bus, authority=bus.test_authority).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                response = await client.get('/api/voices')
                assert response.status_code==503 and 'suggested_profile' not in response.text


# --- T044-U1: filtered query route, admin search and the QB transcript read ---------------------

QUERY = '/api/voices/captures/query'
TEXT_FIELDS = {'id','observed_at','expires_at','transcript','room_id','duration_s','transcript_scope',
    'transcript_metadata','variant'}


def query_body(**values):
    """The closed body: every key on every request, text null for filters (R3-5)."""
    return {'room':None,'window':None,'group_state':None,'association':None,'listenable_only':False,
        'text':None,'cursor':None,**values}


@asynccontextmanager
async def voice_api(voices,bus):
    api = DashboardAPI(voice_registry=voices,event_bus=bus,authority=bus.test_authority)
    try:
        yield api
    finally:
        await api.stop()


@asynccontextmanager
async def cookie_client(api,authority,token):
    """A browser-like client holding only the kza session cookie."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as client:
        assert (await client.post('/api/admin/auth/login',json={'token':token})).status_code==200
        client.lease_reader = authority.resolve(None,client.cookies[COOKIE_NAME]).reader_binding()
        yield client


async def credential(authority,dashboard_auth,role,subject,**changes):
    now = time.time()
    if role is DashboardRole.EVALUATOR:
        changes['samples'] = (EvaluationSample('vf_'+'a'*64,frozenset({'b'*64}),
            frozenset({EvaluationModality.AUDIO_PREVIEW}),now+60),)
    return await authority.issue_credential(authority.resolve(dashboard_auth['Authorization']),subject=subject,
        role=role,expires_at=now+120,**changes)


async def test_query_route_permission_matrix_session_rule_and_no_separate_search_path(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus), voice_api(voices,bus) as api:
        await capture(voices,clock,text='Busqueda sintetica de la sala')
        authority = bus.test_authority
        operator = await credential(authority,dashboard_auth,DashboardRole.OPERATOR,'synthetic-operator')
        home = await credential(authority,dashboard_auth,DashboardRole.HOME,'synthetic-home')
        evaluator = await credential(authority,dashboard_auth,DashboardRole.EVALUATOR,'synthetic-evaluator')
        search = query_body(text='busqueda')

        async def ask(client,headers,body):
            response = await client.post(QUERY,json=body,headers=headers)
            return response.status_code,response.json().get('detail')

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as bearer, \
                cookie_client(api,authority,operator.token) as cookie, \
                cookie_client(api,authority,fixture_token()) as shared:
            session_bearer = {'Authorization':'Bearer '+cookie.cookies[COOKIE_NAME]}
            async with ready_browser(bus,version=3) as legacy_peer, \
                    ready_browser(bus,version=3,reader=authority.resolve('Bearer '+operator.token).reader_binding()) as operator_peer, \
                    ready_browser(bus,version=3,reader=cookie.lease_reader) as cookie_peer, \
                    ready_browser(bus,version=3,reader=shared.lease_reader) as shared_peer:
                carriers = [
                    (bearer,{**dashboard_auth,'X-KZA-Live-Connection':legacy_peer.connection}),
                    (bearer,{'Authorization':'Bearer '+operator.token,'X-KZA-Live-Connection':operator_peer.connection}),
                    (bearer,{**session_bearer,'X-KZA-Live-Connection':cookie_peer.connection}),
                ]
                for client,headers in carriers:
                    assert (await ask(client,headers,query_body()))[0]==200
                    assert await ask(client,headers,search)==(403,'voice_search_requires_session')
                for client,peer in ((cookie,cookie_peer),(shared,shared_peer)):
                    headers = {'X-KZA-Live-Connection':peer.connection}
                    assert (await ask(client,headers,query_body()))[0]==200
                    response = await client.post(QUERY,json=search,headers=headers)
                    assert response.status_code==200 and response.json()['snapshot']['matching_count']==1
                for denied in (home,evaluator):
                    for body in (query_body(),search):
                        assert await ask(bearer,{'Authorization':'Bearer '+denied.token},body)==(403,'dashboard_permission_denied')
                    async with cookie_client(api,authority,denied.token) as other:
                        assert (await ask(other,{},search))==(403,'dashboard_permission_denied')
                assert (await bearer.post('/api/voices/captures/search',json=search,headers=dashboard_auth)).status_code in {404,405}


async def test_reviewer_without_admin_filters_but_cannot_search(tmp_path,dashboard_auth,monkeypatch):
    """F4: no shipped role is a reviewer without admin:manage; the principal is built by hand."""
    async with web_graph(tmp_path) as (_,_,voices,clock,bus), voice_api(voices,bus) as api:
        await capture(voices,clock,text='Texto sintetico')
        authority = bus.test_authority
        reviewer = await credential(authority,dashboard_auth,DashboardRole.HOME,'synthetic-reviewer')
        original = authority_module._permissions
        monkeypatch.setattr(authority_module,'_permissions',lambda role,user_id:
            frozenset({'voices:read','voices:review'}) if role is DashboardRole.HOME else original(role,user_id))
        async with cookie_client(api,authority,reviewer.token) as client:
            assert 'admin:manage' not in authority.resolve(None,client.cookies[COOKIE_NAME]).principal.permissions
            async with ready_browser(bus,version=3,reader=client.lease_reader) as peer:
                headers = {'X-KZA-Live-Connection':peer.connection}
                response = await client.post(QUERY,json=query_body(),headers=headers)
                assert response.status_code==200 and response.json()['snapshot']['matching_count']==1
                response = await client.post(QUERY,json=query_body(text='texto'),headers=headers)
                assert (response.status_code,response.json()['detail'])==(403,'voice_search_denied')


async def test_search_reader_and_guard_carry_admin_manage_and_filters_do_not(tmp_path,dashboard_auth,monkeypatch):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus), voice_api(voices,bus) as api:
        await capture(voices,clock)
        asked = []
        for name in ('session_reader','session_guard'):
            original = getattr(voice_api_module,name)
            def record(request,permissions,*args,original=original,name=name):
                asked.append((name,permissions))
                return original(request,permissions,*args)
            monkeypatch.setattr(voice_api_module,name,record)
        async with cookie_client(api,bus.test_authority,fixture_token()) as client, \
                ready_browser(bus,version=3,reader=client.lease_reader) as peer:
            headers = {'X-KZA-Live-Connection':peer.connection}
            assert (await client.post(QUERY,json=query_body(),headers=headers)).status_code==200
            assert asked==[('session_guard',frozenset({'voices:read','voices:review'})),
                ('session_reader',frozenset({'voices:read','voices:review'}))]
            asked.clear()
            assert (await client.post(QUERY,json=query_body(text='texto'),headers=headers)).status_code==200
            assert asked==[('session_guard',frozenset({'voices:read','voices:review','admin:manage'})),
                ('session_reader',frozenset({'voices:read','voices:review','admin:manage'}))]


async def test_admin_authority_revoked_during_a_search_stops_the_response(tmp_path,dashboard_auth,monkeypatch):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus), voice_api(voices,bus) as api:
        for _ in range(3):
            await capture(voices,clock,text='Texto sintetico revocado')
        authority = bus.test_authority
        operator = await credential(authority,dashboard_auth,DashboardRole.OPERATOR,'synthetic-operator')
        original = authority_module._permissions
        scan = voices.ledger._query_scan
        def revoking(*args):
            monkeypatch.setattr(authority_module,'_permissions',lambda role,user_id:
                original(role,user_id)-{'admin:manage'})
            return scan(*args)
        async with cookie_client(api,authority,operator.token) as client, \
                ready_browser(bus,version=3,reader=client.lease_reader) as peer:
            monkeypatch.setattr(voices.ledger,'_query_scan',revoking)
            response = await client.post(QUERY,json=query_body(text='revocado'),headers={'X-KZA-Live-Connection':peer.connection})
            assert response.status_code==403 and 'revocado' not in response.text


@pytest.mark.parametrize('change',[
    {'missing':'text'},{'missing':'cursor'},{'missing':'room'},{'extra':('limit',10)},
    {'raw':b'{"room":null,"room":null,"window":null,"group_state":null,"association":null,"listenable_only":false,"text":null,"cursor":null}'},
    {'raw':b'not json Sala sintetica'},{'raw':b'[]'},{'raw':b'\xff\xfe'},
    {'raw':b'{"room":null,"window":{"since":NaN,"until":1},"group_state":null,"association":null,"listenable_only":false,"text":null,"cursor":null}'},
    {'set':{'listenable_only':'true'}},{'set':{'listenable_only':1}},{'set':{'group_state':'verified'}},
    {'set':{'association':'manual'}},{'set':{'room':{'kind':'room'}}},{'set':{'room':{'kind':'unrecorded','id':'Sala sintetica'}}},
    {'set':{'room':{'kind':'room','id':'Sala\x00sintetica'}}},{'set':{'room':{'kind':'room','id':'x'*129}}},
    {'set':{'room':{'kind':'patio','id':'Sala sintetica'}}},
    {'window':(10,10)},{'set':{'window':{'since':-1.0,'until':10.0}}},{'window':(-172800-301,0)},{'window':(0,301)},{'window':(True,1)},
    {'set':{'text':'a'}},{'set':{'text':'   '}},{'set':{'text':'á'}},{'set':{'text':'uno dos tres cuatro cinco seis siete ocho nueve'}},
    {'set':{'text':'x'*65}},{'set':{'text':7}},
    {'set':{'cursor':'vc1_'+'0'*16+'_'+'a'*64}},{'set':{'cursor':'vc2_'+'z'*114}},
    {'oversize':True},{'url':'?text=Sala+sintetica'},{'url':'?room=Sala+sintetica'},
])
async def test_invalid_query_bodies_get_one_fixed_422_without_echo(tmp_path,dashboard_auth,change):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus), voice_api(voices,bus) as api:
        calls = []
        voices.ledger.query_captures = lambda *args:calls.append(args)
        body = query_body(text='Sala sintetica')
        if 'missing' in change:
            del body[change['missing']]
        if 'extra' in change:
            body[change['extra'][0]] = change['extra'][1]
        body.update(change.get('set',{}))
        if 'window' in change:
            since,until = change['window']
            body['window'] = {'since':clock.now+since if type(since) is not bool else since,'until':clock.now+until}
        content = change.get('raw',json.dumps(body).encode())
        if change.get('oversize'):
            content = json.dumps(query_body(room={'kind':'room','id':'Sala sintetica'},text='x '*2100)).encode()
        async with cookie_client(api,bus.test_authority,fixture_token()) as client, \
                ready_browser(bus,version=3,reader=client.lease_reader) as peer:
            response = await client.post(QUERY+change.get('url',''),content=content,
                headers={'X-KZA-Live-Connection':peer.connection,'Content-Type':'application/json'})
        assert (response.status_code,response.json())==(422,{'detail':'invalid_voice_query'})
        assert response.headers['cache-control']=='no-store'
        assert 'Sala' not in response.text and 'sintetica' not in response.text and calls==[]


async def test_cursor_of_another_filter_query_or_process_is_a_409(tmp_path,dashboard_auth,monkeypatch):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus), voice_api(voices,bus) as api:
        for _ in range(55):
            await capture(voices,clock,room='Sala sintetica')
        sala = query_body(room={'kind':'room','id':'Sala sintetica'})
        async with cookie_client(api,bus.test_authority,fixture_token()) as client, \
                ready_browser(bus,version=3,reader=client.lease_reader) as peer:
            headers = {'X-KZA-Live-Connection':peer.connection}
            async def ask(body):
                response = await client.post(QUERY,json=body,headers=headers)
                await peer.message({'type':'live_ack','receipts':[response.json()['snapshot']['receipt']]}) \
                    if response.status_code==200 and response.json()['snapshot'] else None
                return response
            first = (await ask(sala)).json()['snapshot']
            cursor = first['next_cursor']
            assert cursor.startswith('vc2_') and 'Sala' not in cursor
            assert len((await ask({**sala,'cursor':cursor})).json()['snapshot']['rows'])==5
            for other in (query_body(cursor=cursor),{**sala,'listenable_only':True,'cursor':cursor},
                    {**sala,'text':'texto','cursor':cursor}):
                response = await ask(other)
                assert (response.status_code,response.json()['detail'])==(409,'voice_cursor_filter_mismatch')
            monkeypatch.setattr(voice_ledger_module,'_QUERY_CURSOR_KEY',b'\x02'*32)
            response = await ask({**sala,'cursor':cursor})
            assert (response.status_code,response.json()['detail'])==(409,'voice_cursor_filter_mismatch')


async def test_query_page_envelope_count_room_options_and_metadata_only_rows(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus), voice_api(voices,bus) as api:
        for room in ('Sala sintetica','Cocina sintetica',None):
            await capture(voices,clock,room=room,text='Texto privado de prueba')
        await capture(voices,clock,review=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as client, \
                ready_browser(bus,version=3) as peer:
            headers = {**dashboard_auth,'X-KZA-Live-Connection':peer.connection}
            document = (await client.post(QUERY,json=query_body(room={'kind':'room','id':'Sala sintetica'}),headers=headers)).json()
            snapshot = document['snapshot']
            assert snapshot['channel']=='voices:captures' and len(snapshot['rows'])==1
            assert snapshot['matching_count']==1 and snapshot['scan_complete'] is True
            assert snapshot['count_observed_at']==snapshot['observed_at'] and snapshot['count_expires_at']==snapshot['rows'][0]['expires_at']
            assert snapshot['room_options']==['Cocina sintetica','Sala sintetica']
            assert snapshot['room_options_expires_at']==snapshot['rows'][0]['expires_at']
            assert snapshot['rows'][0]['room_id']=='Sala sintetica' and document['empty_page'] is None
            assert 'Texto privado' not in json.dumps(document)
            assert all(set(row)=={*CAPTURE_ROW} for row in snapshot['rows'])
            # An empty page has no ticket: only a complete zero count leaves, never room options.
            empty = (await client.post(QUERY,json=query_body(room={'kind':'room','id':'Patio sintetico'}),headers=headers)).json()
            assert empty['snapshot'] is None and empty['next_cursor'] is None
            assert empty['empty_page']=={'matching_count':0,'scan_complete':True,'count_observed_at':clock.now}
            assert 'Sala' not in json.dumps(empty)
            # GET /api/voices keeps its parameters and gains room_id on every row (QA).
            listed = (await client.get('/api/voices',headers=headers)).json()['snapshot']
            assert set(listed)=={'event_id','expires_at','observed_at','receipt','channel','next_cursor','rows'}
            assert sorted(str(row['room_id']) for row in listed['rows'])==['Cocina sintetica','None','None','Sala sintetica']


async def test_an_empty_page_never_carries_a_non_zero_count(tmp_path,dashboard_auth):
    """An empty page has no ticket, so a count of revocable off-page rows never leaves with it."""
    async with web_graph(tmp_path) as (_,_,voices,clock,bus), voice_api(voices,bus) as api:
        for _ in range(3):
            await capture(voices,clock,room='Sala sintetica')
        # The digest covers only the filters: a valid cursor positioned after every row of the window.
        filters = voice_ledger_module.VoiceCaptureFilter(None,None,None,None,None,None,False)
        cursor = voice_ledger_module.capture_query_cursor(0.0,'vf_'+'0'*64,filters.digest())
        direct = await voices.acall(voices.ledger.query_captures,filters,cursor,ACTOR)
        assert direct.rows==[] and direct.summary.matching_count==3 and direct.summary.scan_complete
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as client, \
                ready_browser(bus,version=3) as peer:
            headers = {**dashboard_auth,'X-KZA-Live-Connection':peer.connection}
            response = await client.post(QUERY,json=query_body(cursor=cursor),headers=headers)
            document = response.json()
            assert response.status_code==200 and document['snapshot'] is None and document['next_cursor'] is None
            assert document['empty_page']=={'matching_count':None,'scan_complete':True,'count_observed_at':clock.now}
            assert 'Sala' not in response.text


async def test_off_page_erasure_retires_the_filtered_response_and_its_count(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,owner,voices,clock,bus), voice_api(voices,bus) as api:
        made = [await capture(voices,clock) for _ in range(60)]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as client, \
                ready_browser(bus,version=3) as peer:
            headers = {**dashboard_auth,'X-KZA-Live-Connection':peer.connection}
            snapshot = (await client.post(QUERY,json=query_body(listenable_only=True),headers=headers)).json()['snapshot']
            assert snapshot['matching_count']==60 and len(snapshot['rows'])==50
            off_page = made[0]
            assert off_page[1] not in {row['id'] for row in snapshot['rows']}
            await owner.request(off_page[0].conversation_id,MemoryReviewer('synthetic-admin'))
            async with asyncio.timeout(3):
                control = await peer.outgoing.get()
            assert control['type']=='rest_retire' and control['channel']=='voices:captures'
            assert control['payload']['event_ids']==[snapshot['event_id']]
            again = (await client.post(QUERY,json=query_body(listenable_only=True),headers=headers)).json()['snapshot']
            assert again['matching_count']==59


async def test_filter_and_search_requests_share_one_request_line_and_leave_no_record(tmp_path,dashboard_auth,caplog):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus), voice_api(voices,bus) as api:
        await capture(voices,clock,room='Sala sintetica',text='Registro secreto sintetico')
        lines = []
        @api.app.middleware('http')
        async def line(request,call_next):
            response = await call_next(request)
            lines.append((request.scope['method'],request.scope['path'],request.scope['query_string'],response.status_code))
            return response
        def listing():
            return sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob('*') if not path.name.endswith('-journal'))
        before = listing()
        caplog.set_level(logging.DEBUG)
        async with cookie_client(api,bus.test_authority,fixture_token()) as client, \
                ready_browser(bus,version=3,reader=client.lease_reader) as peer:
            headers = {'X-KZA-Live-Connection':peer.connection}
            window = {'since':clock.now-60,'until':clock.now+1}
            filtered = await client.post(QUERY,json=query_body(room={'kind':'room','id':'Sala sintetica'},window=window),headers=headers)
            searched = await client.post(QUERY,json=query_body(text='registro secreto'),headers=headers)
            refused = await client.post(QUERY,json=query_body(text='r',room={'kind':'room','id':'Sala sintetica'}),headers=headers)
        assert filtered.status_code==searched.status_code==200 and refused.status_code==422
        assert searched.json()['snapshot']['matching_count']==1
        assert lines[-3][:3]==lines[-2][:3]==('POST',QUERY,b'') and lines[-3][3]==lines[-2][3]
        assert listing()==before
        logged = '\n'.join(record.getMessage()+repr(record.args) for record in caplog.records)
        for secret in ('Sala sintetica','registro','secreto','Registro'):
            assert secret not in logged and secret not in refused.text
        # QD: the query path writes no line at all (no principal, query or result count).
        assert not [record for record in caplog.records if record.name.startswith(('src.dashboard.voice','src.users.voice'))]


async def test_transcript_read_is_the_reviews_text_variant_without_audio(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (privacy,owner,voices,clock,bus), voice_api(voices,bus) as api:
        source,identity = await capture(voices,clock,room='Sala sintetica',text='Transcripción sintética legible')
        statements = []
        privacy.ledger._db.set_trace_callback(statements.append)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as client, \
                ready_browser(bus,version=3) as peer:
            headers = {**dashboard_auth,'X-KZA-Live-Connection':peer.connection}
            response = await client.get('/api/voices/captures/'+identity+'/transcript',headers=headers)
            privacy.ledger._db.set_trace_callback(None)
            snapshot = response.json()['snapshot']
            assert response.status_code==200 and snapshot['channel']=='voices:review' and len(snapshot['rows'])==1
            row = snapshot['rows'][0]
            assert set(row)==TEXT_FIELDS and row['variant']=='text'
            assert (row['transcript'],row['room_id'],row['id'])==('Transcripción sintética legible','Sala sintetica',identity)
            assert row['expires_at']==source.expires_at and 'audio' not in response.text
            reads = [statement for statement in statements if 'audio_wav' in statement]
            assert reads and all(re.sub(r'(typeof|length)\(r\.audio_wav\)','',statement).count('audio_wav')==0
                for statement in reads)
            for path in ('/api/voices/captures/vf_bad/transcript','/api/voices/captures/'+identity+'/transcript?x=1',
                    '/api/voices/captures/'+identity.upper().replace('VF_','vf_')+'/transcript'):
                response = await client.get(path,headers=headers)
                assert (response.status_code,response.json())==(422,{'detail':'invalid_voice_capture_id'})
            # An erasure after delivery retires the delivered text on voices:review.
            await owner.request(source.conversation_id,MemoryReviewer('synthetic-admin'))
            async with asyncio.timeout(3):
                control = await peer.outgoing.get()
            assert control['channel']=='voices:review' and control['payload']['event_ids']==[snapshot['event_id']]
            response = await client.get('/api/voices/captures/'+identity+'/transcript',headers=headers)
            assert response.status_code==404 and 'Transcripción' not in response.text


@pytest.mark.parametrize('damage',[
    "UPDATE voice_capture_reviews SET audio_wav=CAST(audio_wav AS TEXT)",
    "UPDATE voice_capture_reviews SET audio_wav=substr(audio_wav,1,25643)",
    "UPDATE voice_capture_reviews SET clip_duration_s=7.0",
    "UPDATE voice_capture_reviews SET clip_duration_s=0.5",
    'expired',
])
async def test_review_and_transcript_share_the_sql_admission(tmp_path,dashboard_auth,damage):
    async with web_graph(tmp_path) as (privacy,_,voices,clock,bus), voice_api(voices,bus) as api:
        _,identity = await capture(voices,clock,text='Texto con admisión',ttl=30)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as client, \
                ready_browser(bus,version=3) as peer:
            headers = {**dashboard_auth,'X-KZA-Live-Connection':peer.connection}
            for kind in ('review','transcript'):
                assert (await client.get('/api/voices/captures/'+identity+'/'+kind,headers=headers)).status_code==200
            if damage=='expired':
                clock.now += 31
            else:
                def corrupt():
                    with privacy.ledger._transaction() as db:
                        db.execute(damage)
                await privacy.acall(corrupt)
            for kind in ('review','transcript'):
                response = await client.get('/api/voices/captures/'+identity+'/'+kind,headers=headers)
                assert response.status_code==404 and 'admisión' not in response.text


async def test_transcript_read_requires_review_permission(tmp_path,dashboard_auth,monkeypatch):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus), voice_api(voices,bus) as api:
        _,identity = await capture(voices,clock)
        authority = bus.test_authority
        path = '/api/voices/captures/'+identity+'/transcript'
        home = await credential(authority,dashboard_auth,DashboardRole.HOME,'synthetic-home')
        evaluator = await credential(authority,dashboard_auth,DashboardRole.EVALUATOR,'synthetic-evaluator')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as client:
            for denied in (home,evaluator):
                assert (await client.get(path,headers={'Authorization':'Bearer '+denied.token})).status_code==403
            original = authority_module._permissions
            monkeypatch.setattr(authority_module,'_permissions',lambda role,user_id:
                frozenset({'voices:read'}) if role is DashboardRole.HOME else original(role,user_id))
            response = await client.get(path,headers={'Authorization':'Bearer '+home.token})
            assert (response.status_code,response.json()['detail'])==(403,'voice_review_denied')


@pytest.mark.parametrize('variant,row',[
    ('text',{'id':'vf_'+'a'*64,'observed_at':0,'expires_at':1,'transcript':'t','room_id':None,'duration_s':1.,
        'transcript_scope':'full_segment','transcript_metadata':None,'variant':'text','audio_base64':'AAAA'}),
    ('audio',{'id':'vf_'+'a'*64,'observed_at':0,'expires_at':1,'transcript':'t','room_id':None,'duration_s':1.,
        'transcript_scope':'full_segment','transcript_metadata':None,'variant':'text'}),
])
async def test_review_delivery_checks_each_variants_closed_field_set(tmp_path,dashboard_auth,monkeypatch,variant,row):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus), voice_api(voices,bus) as api:
        source,identity = await capture(voices,clock)
        forged = dict(row,id=identity,observed_at=clock.now,expires_at=source.expires_at)
        monkeypatch.setattr(voices.ledger,'project',lambda *args,**kwargs:VoiceProjection([forged],(source,),clock.now,
            source.expires_at,'voices:review'))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as client, \
                ready_browser(bus,version=3) as peer:
            response = await client.get('/api/voices/captures/'+identity+('/transcript' if variant=='text' else '/review'),
                headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection})
            assert response.status_code==503 and 'AAAA' not in response.text
    assert VOICE_CHANNELS==frozenset({'voices:profiles','voices:captures','voices:review'})


async def test_every_revocation_assignment_site_advances_the_generation(tmp_path,monkeypatch):
    clock = Clock(2_000_000_000.)
    ledger = FactLedger(str(tmp_path/'facts.db'),clock=clock)
    generation = ledger.revocations
    def moved(action):
        stamp = RevisionStamp(ref(generation),generation.projection_revision)
        action()
        return not stamp.current()
    reviewer = MemoryReviewer('synthetic-admin')
    source = MemorySource(str(uuid4()),str(uuid4()),clock.now-1,clock.now+3600,MemoryOrigin.VOICE)
    try:
        assert moved(ledger.initialize)  # Start and success of initialize.
        def read():
            with ledger._transaction() as db:
                db.execute('SELECT 1')
        assert not moved(read)  # A transaction that revokes nothing keeps every view.
        fact = ledger.store('Synthetic fact','fact',.9,(source,))
        assert moved(lambda:ledger.delete_fact(fact.id,reviewer))  # Receipts and pending fact ids.
        seen = []
        observer = Observer(seen,generation)
        ledger.add_revocation_observer(observer.pending)
        assert moved(lambda:ledger.delete_conversation(source.conversation_id,reviewer))
        assert seen and seen[0][1]>seen[0][0]  # Pending conversations advanced before commit.
        def failed():
            with ledger._transaction() as db:
                db.execute('INSERT INTO conversation_tombstones VALUES (?,?,?)',(str(uuid4()),clock.now,'synthetic'))
                ledger._authority_version += 1
                raise RuntimeError('synthetic rollback')
        before = generation.projection_revision
        with pytest.raises(RuntimeError):
            failed()
        assert generation.projection_revision==before  # The rollback re-read the unchanged committed state.
        stamp = RevisionStamp(ref(generation),generation.projection_revision)
        assert moved(ledger.close)
        assert not stamp.current()
        monkeypatch.setattr(fact_ledger_module,'initialize_fact_owner_schema',
            lambda *args,**kwargs:(_ for _ in ()).throw(sqlite3.DatabaseError('synthetic failure')))
        before = generation.projection_revision
        with pytest.raises(sqlite3.DatabaseError):
            ledger.initialize()
        assert generation.projection_revision==before+2 and not ledger.authority_available  # Start and failure.
    finally:
        ledger.close()


class Observer:
    """Record the generation seen when the pending revocation is published, before commit."""

    def __init__(self,seen,generation):
        self.seen,self.generation = seen,generation
        self.start = generation.projection_revision

    def pending(self,conversations):
        self.seen.append((self.start,self.generation.projection_revision))


async def test_a_revocation_state_change_alone_retires_query_pages_but_not_the_plain_list(tmp_path,dashboard_auth):
    """The fence is the fact ledger's revocation generation, not the group or profile revision."""
    async with web_graph(tmp_path) as (privacy,_,voices,clock,bus), voice_api(voices,bus) as api:
        await capture(voices,clock)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver') as client, \
                ready_browser(bus,version=3) as peer:
            headers = {**dashboard_auth,'X-KZA-Live-Connection':peer.connection}
            listed = (await client.get('/api/voices',headers=headers)).json()['snapshot']
            page = (await client.post(QUERY,json=query_body(),headers=headers)).json()['snapshot']
            fences = voices.ledger.groups.projection_revision,voices.ledger.projection_revision
            generation = privacy.ledger.revocations.projection_revision
            other = MemorySource(str(uuid4()),str(uuid4()),clock.now-1,clock.now+3600,MemoryOrigin.VOICE)
            await privacy.acall(privacy.ledger.store,'Synthetic unrelated fact','fact',.9,(other,))
            assert privacy.ledger.revocations.projection_revision>generation
            assert (voices.ledger.groups.projection_revision,voices.ledger.projection_revision)==fences
            async with asyncio.timeout(3):
                control = await peer.outgoing.get()
            assert control['channel']=='voices:captures' and control['payload']['event_ids']==[page['event_id']]
            state = next(state for state in bus._retention._states.values() if state.browser.connection==peer.connection)
            assert state.snapshots[listed['event_id']].available(privacy.ledger,clock.now)
