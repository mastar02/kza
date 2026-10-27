"""Voice review bodies and permanent labels require distinct actual page receipts."""

import asyncio
import json
import threading
from uuid import uuid4

import httpx
import pytest

from src.dashboard.api import DashboardAPI
from src.memory.fact_ledger import MemoryReviewer
from src.users.voice_projection import VoiceProjection
from src.users.voice_ledger import VoiceLedger
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_live_stream_receipts import flush
from tests.integration.test_voice_capture_review import observe
from tests.mocks.rest_browser import ready_browser
from tests.mocks.voice_registry import ACTOR,seed,web_graph

pytestmark = pytest.mark.asyncio


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
        app = DashboardAPI(voice_registry=voices,event_bus=bus).app
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
            api = DashboardAPI(voice_registry=voices,event_bus=bus)
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
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
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
                assert await receipts.issue(peer.connection,str(uuid4()),clock.now+300,tuple(sources)) is None
                assert await receipts.issue_profiles(peer.connection,str(uuid4()),clock.now+300,tuple(sources),((profile['id'],2),)) is None


async def test_deleted_profile_retires_only_its_metadata_channel(tmp_path,dashboard_auth):
    async with web_graph(tmp_path) as (_,_,voices,clock,bus):
        _,identities = await seed(voices,clock)
        profile = await voices.acall(voices.ledger.confirm,identities,'Synthetic profile',str(uuid4()),ACTOR)
        async with ready_browser(bus,version=3) as peer:
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
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
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
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
            api = DashboardAPI(voice_registry=voices,event_bus=bus)
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
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
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
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
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
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
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
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
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
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
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
            app = DashboardAPI(voice_registry=voices,event_bus=bus).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                response = await client.get('/api/voices')
                assert response.status_code==503 and 'suggested_profile' not in response.text
