"""Real evaluation cookies, source-bound receipts and actual HTTP ownership."""

import asyncio
import base64
from contextlib import AsyncExitStack, asynccontextmanager
import json
import threading
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from starlette.responses import StreamingResponse

from src.dashboard.api import DashboardAPI
from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import DashboardRole, EvaluationModality
from src.dashboard.live_erasure import build_live_services
from src.dashboard.live_retention_types import LiveSubscriptionMode as Mode
from src.dashboard.rest_delivery import RestOperation
from src.memory.fact_ledger import MemoryReviewer, MemorySource
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_live_stream_receipts import flush
from tests.integration import test_voice_evaluation_reader as native_fixtures
from tests.integration.test_voice_evaluation_reader import sql
from tests.integration.test_voice_review_delivery import step
from tests.mocks.rest_browser import ready_browser

pytestmark = pytest.mark.asyncio
evaluation_graph = native_fixtures.evaluation_graph
PERMISSIONS = frozenset({'evaluation:read'})


@pytest_asyncio.fixture
async def evaluation_http(evaluation_graph):
    g = evaluation_graph
    async with AsyncExitStack() as closing:
        g.bus = await build_live_services({},privacy=g.privacy,owner=g.owner,shutdown=closing)
        g.bus.test_authority = g.authority
        g.bus.test_reader = g.root.reader_binding()
        g.api = DashboardAPI(voice_registry=g.voices,event_bus=g.bus,authority=g.authority)
        closing.push_async_callback(g.api.stop)
        yield g


@asynccontextmanager
async def client_for(g,issued,*,connection=None,app=None):
    headers = {} if connection is None else {'X-KZA-Live-Connection':connection}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=g.api.app if app is None else app),
            base_url='http://testserver',cookies={COOKIE_NAME:issued.session.token},headers=headers) as client:
        yield client


def sample_path(g,modality):
    return '/api/evaluation/samples/'+g.identity+'?modality='+modality.value


@pytest.mark.parametrize('modality',list(EvaluationModality))
async def test_exact_evaluation_response_has_one_closed_selected_payload(evaluation_http,modality):
    g = evaluation_http
    issued = await g.issue(modality,modes=frozenset(EvaluationModality))
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    async with ready_browser(g.bus,version=3,reader=reader,mode=Mode.EVALUATION_CONTROL_ONLY) as peer:
        assert peer.ready_frame['channels']==['evaluation:sample']
        async with client_for(g,issued,connection=peer.connection) as client:
            response = await client.get(sample_path(g,modality))
        assert response.status_code==200 and response.headers['cache-control']=='no-store'
        envelope = response.json()
        assert set(envelope)=={'connection_id','snapshot','channel'}
        snapshot = envelope['snapshot']
        assert set(snapshot)=={'event_id','expires_at','observed_at','receipt','channel','rows'}
        assert envelope['channel']==snapshot['channel']=='evaluation:sample'
        assert snapshot['expires_at']==issued.sample.expires_at and len(snapshot['rows'])==1
        row = snapshot['rows'][0]
        assert row['id']==g.identity and row['modality']==modality.value
        if modality is EvaluationModality.AUDIO_PREVIEW:
            assert set(row)=={'id','modality','audio_mime','audio_base64','clip_duration_s','audio_metadata'}
            assert base64.b64decode(row['audio_base64'])==g.review.audio_wav
            assert row['audio_metadata'] is None
        else:
            assert set(row)=={'id','modality','transcript_scope','transcript','transcript_metadata'}
            assert row['transcript']==g.review.transcript
            assert row['transcript_metadata'] is None
        for source in g.sources:
            assert await g.bus._retention.receipts.pending(source.conversation_id)==1


@pytest.mark.parametrize('mode',[Mode.TECHNICAL,Mode.VOICES_CONTROL_ONLY])
async def test_other_server_channel_cannot_start_native_evaluation(evaluation_http,monkeypatch,mode):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    called = []
    original = g.voices.evaluation.read
    def observed(actor):
        called.append(True)
        return original(actor)
    monkeypatch.setattr(g.voices.evaluation,'read',observed)
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    async with ready_browser(g.bus,version=3,reader=reader,mode=mode) as peer:
        async with client_for(g,issued,connection=peer.connection) as client:
            response = await client.get(sample_path(g,EvaluationModality.TRANSCRIPT_FULL))
        assert response.status_code==428 and called==[]


async def test_missing_duplicate_and_foreign_login_connections_never_read(evaluation_http,monkeypatch):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    other = await g.authority.login(issued.grant.token)
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    other_reader = other.lease.reader_binding(PERMISSIONS)
    assert reader.namespace!=other_reader.namespace
    calls = []
    original = g.voices.evaluation.read
    def observed(actor):
        calls.append(True)
        return original(actor)
    monkeypatch.setattr(g.voices.evaluation,'read',observed)
    async with ready_browser(g.bus,version=3,reader=reader,mode=Mode.EVALUATION_CONTROL_ONLY) as own, \
            ready_browser(g.bus,version=3,reader=other_reader,mode=Mode.EVALUATION_CONTROL_ONLY) as foreign:
        async with client_for(g,issued) as client:
            path = sample_path(g,EvaluationModality.TRANSCRIPT_FULL)
            assert (await client.get(path)).status_code==428
            assert (await client.get(path,headers={'X-KZA-Live-Connection':foreign.connection})).status_code==428
            duplicate = [('X-KZA-Live-Connection',own.connection),('X-KZA-Live-Connection',foreign.connection)]
            assert (await client.get(path,headers=duplicate)).status_code==428
            assert calls==[]
            assert (await client.get(path,headers={'X-KZA-Live-Connection':own.connection})).status_code==200
            assert calls==[True]


@pytest.mark.parametrize('suffix',['&person_id=other','&sources=other','&expires_at=9999999999',
    '&cursor=other','&modality=audio_preview'])
async def test_extra_query_cannot_expand_evaluation_scope(evaluation_http,monkeypatch,suffix):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    def unexpected(_):
        raise AssertionError('invalid query reached native reader')
    monkeypatch.setattr(g.voices.evaluation,'read',unexpected)
    async with client_for(g,issued) as client:
        response = await client.get(sample_path(g,EvaluationModality.TRANSCRIPT_FULL)+suffix)
    assert response.status_code==422 and response.headers['cache-control']=='no-store'


@pytest.mark.parametrize('modality',list(EvaluationModality))
async def test_actual_native_sources_cannot_be_replaced_before_ticket(evaluation_http,monkeypatch,modality):
    g = evaluation_http
    issued = await g.issue(modality)
    rows = await sql(g,'SELECT sources_json FROM voice_fingerprints WHERE id=?',(g.other,))
    other_sources = tuple(MemorySource.from_dict(value) for value in json.loads(rows[0]['sources_json']))
    original = g.voices.evaluation.read
    def wrong_sources(actor):
        projection = original(actor)
        projection.sources = other_sources
        # Instance shadowing must not bypass the class's validation either.
        projection.__post_init__ = lambda:None
        return projection
    monkeypatch.setattr(g.voices.evaluation,'read',wrong_sources)
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    async with ready_browser(g.bus,version=3,reader=reader,mode=Mode.EVALUATION_CONTROL_ONLY) as peer:
        async with client_for(g,issued,connection=peer.connection) as client:
            response = await client.get(sample_path(g,modality))
        assert response.status_code==503
        for source in g.sources:
            assert await g.bus._retention.receipts.pending(source.conversation_id)==0
        assert 'snapshot' not in response.json() or response.json()['snapshot'] is None


@pytest.mark.parametrize('change',['extra_row','extra_field','fingerprint','modality','other_actor','fence'])
async def test_mutated_native_dto_never_creates_a_ticket(evaluation_http,monkeypatch,change):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    other = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    original = g.voices.evaluation.read
    def altered(actor):
        projection = original(actor)
        if change=='extra_row':
            projection.rows.append(dict(projection.rows[0]))
        elif change=='extra_field':
            projection.rows[0]['person_link'] = {'user_id':'unapproved'}
        elif change=='fingerprint':
            projection.rows[0]['id'] = g.other
        elif change=='modality':
            projection.rows[0]['modality'] = EvaluationModality.AUDIO_PREVIEW.value
        elif change=='other_actor':
            projection.actor = other.actor
        else:
            projection.revision = type(projection.revision)(())
        projection.__post_init__ = lambda:None
        return projection
    monkeypatch.setattr(g.voices.evaluation,'read',altered)
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    async with ready_browser(g.bus,version=3,reader=reader,mode=Mode.EVALUATION_CONTROL_ONLY) as peer:
        async with client_for(g,issued,connection=peer.connection) as client:
            response = await client.get(sample_path(g,EvaluationModality.TRANSCRIPT_FULL))
        assert response.status_code==503
        for source in g.sources:
            assert await g.bus._retention.receipts.pending(source.conversation_id)==0
        assert g.review.transcript not in response.text


@pytest.mark.parametrize('role',[DashboardRole.HOME,DashboardRole.OPERATOR])
async def test_non_evaluator_role_never_reaches_native_reader(evaluation_http,monkeypatch,role):
    g = evaluation_http
    credential = await g.authority.issue_credential(g.root,subject='synthetic-other-role',role=role,
        expires_at=g.clock.now+90)
    session = await g.authority.login(credential.token)
    def unexpected(_):
        raise AssertionError('non evaluator entered native reader')
    monkeypatch.setattr(g.voices.evaluation,'read',unexpected)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=g.api.app),base_url='http://testserver',
            cookies={COOKIE_NAME:session.token}) as client:
        response = await client.get(sample_path(g,EvaluationModality.TRANSCRIPT_FULL))
    assert response.status_code==403 and response.headers['cache-control']=='no-store'


async def test_evaluation_grant_does_not_open_voice_person_operator_or_other_sample_data(evaluation_http,monkeypatch):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.AUDIO_PREVIEW)
    calls = []
    original = g.voices.evaluation.read
    def observed(actor):
        calls.append(True)
        return original(actor)
    monkeypatch.setattr(g.voices.evaluation,'read',observed)
    async with client_for(g,issued) as client:
        for path in ('/api/voices','/api/voices/profiles','/api/voices/captures/'+g.identity+'/review',
                '/api/admin/users','/api/memory','/api/home/catalog','/api/conversations',
                sample_path(g,EvaluationModality.TRANSCRIPT_FULL),
                '/api/evaluation/samples/'+g.other+'?modality=audio_preview'):
            assert (await client.get(path)).status_code==403
        assert (await client.post('/api/voices/profiles',json={})).status_code==403
        assert (await client.post('/api/evaluation/samples/'+g.identity,json={})).status_code in {403,405}
    assert calls==[]


async def test_sample_expiry_while_headers_wait_retires_body_with_session_still_current(evaluation_http):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL,sample_ttl=3.25)
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    entered,release = asyncio.Event(),asyncio.Event()
    documents = []
    async def actual(scope,receive,send):
        async def held(message):
            if message['type']=='http.response.start' and message['status']==200:
                entered.set()
                await release.wait()
            elif message['type']=='http.response.body' and message.get('body'):
                documents.append(json.loads(message['body']))
            await send(message)
        await g.api.app(scope,receive,held)
    async with ready_browser(g.bus,version=3,reader=reader,mode=Mode.EVALUATION_CONTROL_ONLY) as peer:
        async with client_for(g,issued,connection=peer.connection,app=actual) as client:
            pending = asyncio.create_task(client.get(sample_path(g,EvaluationModality.TRANSCRIPT_FULL)))
            try:
                await until(entered.is_set)
                g.clock.now = issued.sample.expires_at
                assert issued.session.lease.current()
            finally:
                release.set()
                response = await pending
        assert response.status_code==200 and response.json().get('snapshot') is None
        assert g.review.transcript not in json.dumps(documents)


@pytest.mark.parametrize('change',['none','sample','source'])
async def test_completed_inner_response_keeps_original_scope_behind_framework_buffer(evaluation_http,change):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL,sample_ttl=3.25)
    entered,release = asyncio.Event(),asyncio.Event()
    @g.api.app.middleware('http')
    async def buffered(request,call_next):
        response = await call_next(request)
        body = b''.join([part async for part in response.body_iterator])
        entered.set()
        await release.wait()
        async def chunks():
            yield body
        return StreamingResponse(chunks(),status_code=response.status_code,headers=dict(response.headers))
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    async with ready_browser(g.bus,version=3,reader=reader,mode=Mode.EVALUATION_CONTROL_ONLY) as peer:
        async with client_for(g,issued,connection=peer.connection) as client:
            pending = asyncio.create_task(client.get(sample_path(g,EvaluationModality.TRANSCRIPT_FULL)))
            try:
                await until(entered.is_set)
                if change=='sample':
                    g.clock.now = issued.sample.expires_at
                elif change=='source':
                    await g.owner.request(g.sources[0].conversation_id,MemoryReviewer('synthetic-admin'))
                    await flush(g.owner)
                assert issued.session.lease.current()
            finally:
                release.set()
                response = await pending
        assert response.status_code==200
        if change!='none':
            assert response.json().get('snapshot') is None and g.review.transcript not in response.text
        else:
            assert response.json()['snapshot']['rows'][0]['transcript']==g.review.transcript


async def test_sample_expiry_after_send_admission_is_checked_by_actual_transmit(evaluation_http):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL,sample_ttl=3.25)
    projection = await g.voices.acall(g.voices.evaluation.read,issued.actor)
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    async with ready_browser(g.bus,version=3,reader=reader,mode=Mode.EVALUATION_CONTROL_ONLY) as peer:
        actor,digests,deadline = issued.actor,projection.source_digests,projection.expires_at
        operation = RestOperation(sources=projection.sources,expires_at=deadline,connection=peer.connection,
            reader=reader,guard=lambda:actor.current(digests,deadline),revision=projection.revision,
            private_envelope=True,channel='evaluation:sample')
        sent = []
        def expire():
            g.clock.now = issued.sample.expires_at
        async def application(scope,receive,send):
            asyncio.get_running_loop().call_soon(expire)
            await send({'type':'http.response.body','body':json.dumps({'snapshot':projection.rows}).encode()})
        async def receive():
            return {'type':'http.disconnect'}
        async def actual(message):
            sent.append(message)
        await g.bus._retention.rest._scope(operation,application,{'type':'http'},receive,actual)
        assert issued.session.lease.current()
        assert sent==[{'type':'http.response.body','body':b'{"snapshot":null}'}]


async def test_early_ack_cannot_hide_actual_outer_evaluation_body(evaluation_http):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    entered,release = asyncio.Event(),asyncio.Event()
    documents = []
    receipts = g.bus._retention.receipts
    source = g.sources[0]
    async def actual(scope,receive,send):
        async def held(message):
            if message['type']=='http.response.body' and g.review.transcript.encode() in message.get('body',b''):
                documents.append(json.loads(message['body']))
                assert await receipts.pending(source.conversation_id)==1
                entered.set()
                await release.wait()
            await send(message)
        await g.api.app(scope,receive,held)
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    async with ready_browser(g.bus,version=3,reader=reader,mode=Mode.EVALUATION_CONTROL_ONLY) as peer:
        async with client_for(g,issued,connection=peer.connection,app=actual) as client:
            pending = asyncio.create_task(client.get(sample_path(g,EvaluationModality.TRANSCRIPT_FULL)))
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
                erase = await g.owner.request(source.conversation_id,MemoryReviewer('synthetic-admin'))
                await flush(g.owner)
                assert (await step(g.privacy,erase))['state']!='complete'
                release.set()
                assert (await pending).status_code==200
                await until(lambda:g.bus.resource_status()['rest_operations']==0)
                g.clock.now+=2
                await flush(g.owner)
                assert (await step(g.privacy,erase))['state']=='complete'
            finally:
                release.set()
                await asyncio.gather(pending,return_exceptions=True)


async def test_cancelled_observer_keeps_actual_native_read_in_erasure_inventory(evaluation_http,monkeypatch):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.AUDIO_PREVIEW)
    entered,release = threading.Event(),threading.Event()
    original = g.voices.evaluation.read
    def held(actor):
        entered.set()
        assert release.wait(5)
        return original(actor)
    monkeypatch.setattr(g.voices.evaluation,'read',held)
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    async with ready_browser(g.bus,version=3,reader=reader,mode=Mode.EVALUATION_CONTROL_ONLY) as peer:
        async with client_for(g,issued,connection=peer.connection) as client:
            pending = asyncio.create_task(client.get(sample_path(g,EvaluationModality.AUDIO_PREVIEW)))
            try:
                await until(entered.is_set)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                operation = await g.owner.request(str(uuid4()),MemoryReviewer('synthetic-admin'))
                await flush(g.owner)
                assert (await step(g.privacy,operation))['pending_count']>=1
                release.set()
                await until(lambda:g.bus.resource_status()['rest_operations']==0)
                g.clock.now+=2
                await flush(g.owner)
                assert (await step(g.privacy,operation))['state']=='complete'
            finally:
                release.set()
                await asyncio.gather(pending,return_exceptions=True)
