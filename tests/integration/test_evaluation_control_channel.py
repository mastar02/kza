"""Exclusive evaluation control transport and durable per-login clear domains."""

import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import DashboardRole, EvaluationModality
from src.dashboard.live_event_bus import LiveEvent, LiveEventType
from src.dashboard.live_retention_types import LiveSubscriptionMode as Mode
from tests.integration import test_voice_evaluation_delivery as evaluation_fixtures
from tests.integration.test_voice_evaluation_delivery import client_for, PERMISSIONS, sample_path
from tests.integration.test_voice_evaluation_reader import sql
from tests.mocks.memory_sources import memory_source
from tests.mocks.rest_browser import ready_browser

pytestmark = pytest.mark.asyncio
evaluation_graph = evaluation_fixtures.evaluation_graph
evaluation_http = evaluation_fixtures.evaluation_http


@asynccontextmanager
async def evaluation_socket(app,authorization=None,*,cookie=None):
    incoming,outgoing = asyncio.Queue(),asyncio.Queue()
    headers = []
    if authorization is not None:
        headers.append((b'authorization',authorization.encode()))
    if cookie is not None:
        headers.append((b'cookie',(COOKIE_NAME+'='+cookie).encode()))
    scope = {'type':'websocket','path':'/ws/evaluation','raw_path':b'/ws/evaluation','root_path':'',
        'scheme':'ws','query_string':b'','server':('localhost',80),'client':('127.0.0.1',12345),
        'headers':headers,'subprotocols':[],'state':{}}
    await incoming.put({'type':'websocket.connect'})
    observing = asyncio.create_task(app(scope,incoming.get,outgoing.put))
    try:
        yield incoming,outgoing
    finally:
        await incoming.put({'type':'websocket.disconnect','code':1000})
        await asyncio.wait_for(observing,3)


@pytest.mark.parametrize('version',[1,2,3])
async def test_actual_evaluation_socket_requires_protocol_three_before_reset(evaluation_http,version):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    async with evaluation_socket(g.api.app,'Bearer '+issued.grant.token) as (incoming,outgoing):
        assert (await asyncio.wait_for(outgoing.get(),3))['type']=='websocket.accept'
        await incoming.put({'type':'websocket.receive','text':json.dumps({
            'type':'live_hello','version':version,'client_id':str(uuid4())})})
        message = await asyncio.wait_for(outgoing.get(),3)
        if version<3:
            assert message['type']=='websocket.close'
        else:
            reset = json.loads(message['text'])
            assert reset['type']=='live_reset'
            await incoming.put({'type':'websocket.receive','text':json.dumps({
                'type':'live_reset_ack',**reset['payload']})})
            ready = json.loads((await asyncio.wait_for(outgoing.get(),3))['text'])
            assert set(ready)=={'type','connection_id','channels','ts'}
            assert ready['type']=='rest_ready' and ready['channels']==['evaluation:sample']
    if version<3:
        rows = await sql(g,'SELECT (SELECT COUNT(*) FROM live_connections),'
            '(SELECT COUNT(*) FROM live_resets),(SELECT COUNT(*) FROM live_deliveries)')
        assert tuple(rows[0])==(0,0,0)


async def test_actual_cookie_socket_and_http_share_only_the_original_login(evaluation_http):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    async with evaluation_socket(g.api.app,cookie=issued.session.token) as (incoming,outgoing):
        assert (await asyncio.wait_for(outgoing.get(),3))['type']=='websocket.accept'
        await incoming.put({'type':'websocket.receive','text':json.dumps({
            'type':'live_hello','version':3,'client_id':str(uuid4())})})
        reset = json.loads((await asyncio.wait_for(outgoing.get(),3))['text'])
        await incoming.put({'type':'websocket.receive','text':json.dumps({'type':'live_reset_ack',**reset['payload']})})
        ready = json.loads((await asyncio.wait_for(outgoing.get(),3))['text'])
        assert ready['channels']==['evaluation:sample']
        async with client_for(g,issued,connection=ready['connection_id']) as client:
            response = await client.get(sample_path(g,EvaluationModality.TRANSCRIPT_FULL))
            assert response.status_code==200
            ticket = response.json()['snapshot']['receipt']
            assert response.json()['snapshot']['rows'][0]['transcript']==g.review.transcript
            other = await g.authority.login(issued.grant.token)
            client.cookies.set(COOKIE_NAME,other.token)
            assert (await client.get(sample_path(g,EvaluationModality.TRANSCRIPT_FULL))).status_code==428
        await incoming.put({'type':'websocket.receive','text':json.dumps({'type':'live_ack','receipts':[ticket]})})
        async with asyncio.timeout(3):
            for source in g.sources:
                while await g.bus._retention.receipts.pending(source.conversation_id):
                    await asyncio.sleep(.001)


@pytest.mark.parametrize('role',[DashboardRole.HOME,DashboardRole.OPERATOR])
async def test_actual_socket_denies_other_roles_before_subscribe_or_accept(evaluation_http,role):
    g = evaluation_http
    credential = await g.authority.issue_credential(g.root,subject='synthetic-denied-socket',role=role,
        expires_at=g.clock.now+90)
    async with evaluation_socket(g.api.app,'Bearer '+credential.token) as (_,outgoing):
        message = await asyncio.wait_for(outgoing.get(),3)
        assert message['type']=='websocket.close' and message['code']==1008
    assert not g.bus._retention._states


async def test_evaluation_subscription_never_projects_general_events_or_enqueues_payload(evaluation_http,monkeypatch):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    subscription,queue = await g.bus.subscribe(reader=reader,mode=Mode.EVALUATION_CONTROL_ONLY)
    calls = []
    def forbidden(*args):
        calls.append(True)
        raise AssertionError('evaluation control projected general payload')
    monkeypatch.setattr(g.bus._retention,'admit',forbidden)
    try:
        assert subscription not in g.bus._subscribers
        for kind in LiveEventType:
            await g.bus.publish(LiveEvent(kind,payload={'stt':'synthetic-unapproved-broadcast'}))
        assert queue.empty() and calls==[]
        assert (await sql(g,'SELECT COUNT(*) FROM live_deliveries'))[0][0]==0
    finally:
        await g.bus.unsubscribe(subscription)


async def test_evaluation_mode_denies_every_voice_channel_and_general_wire_frame(evaluation_http):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    async with ready_browser(g.bus,version=3,reader=reader,mode=Mode.EVALUATION_CONTROL_ONLY) as peer:
        state = g.bus._retention.rest.connection(peer.connection,reader=reader,
            channel='evaluation:sample',minimum_protocol=3)
        assert state is not None
        for channel in ('conversations','voices:review','voices:captures','voices:profiles'):
            with pytest.raises(PermissionError,match='live_channel_denied'):
                await state.browser.before_frame(SimpleNamespace(channel=channel))
        sent = []
        async def send(frame):
            sent.append(frame)
        for frame in (
                {'type':'turn','payload':{'stt':'unapproved'},'ts':''},
                {'type':'rest_ready','connection_id':peer.connection,'channels':['voices:review'],'ts':''},
                {'type':'rest_ready','connection_id':peer.connection,'channels':['evaluation:sample'],'ts':'','extra':True}):
            with pytest.raises(PermissionError,match='live_channel_denied'):
                await g.bus._retention._send(send,frame,state)
        assert sent==[]
        assert (await sql(g,'SELECT COUNT(*) FROM live_deliveries'))[0][0]==0


async def test_reset_cannot_adopt_another_mode_or_cookie_debt_for_same_page(evaluation_http):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    reader = issued.session.lease.reader_binding(PERMISSIONS)
    modes = (Mode.TECHNICAL,Mode.VOICES_CONTROL_ONLY,Mode.EVALUATION_CONTROL_ONLY)
    sources = tuple(memory_source(now=g.clock.now,ttl=60) for _ in modes)
    store,page = g.bus._retention.receipts,str(uuid4())
    for mode,source in zip(modes,sources):
        connection = str(uuid4())
        challenge = await store.hello(page,connection,reader=reader,mode=mode)
        assert await store.reset_ack(connection,challenge.id,challenge.nonce,reader=reader)
        assert await store.issue(connection,str(uuid4()),source.expires_at,(source,),reader=reader)
        await store.disconnect(connection,reader=reader)
    other = await g.authority.login(issued.grant.token)
    other_reader = other.lease.reader_binding(PERMISSIONS)
    assert other_reader.namespace!=reader.namespace
    connection = str(uuid4())
    challenge = await store.hello(page,connection,reader=other_reader,mode=Mode.EVALUATION_CONTROL_ONLY)
    assert await store.reset_ack(connection,challenge.id,challenge.nonce,reader=other_reader)
    await store.disconnect(connection,reader=other_reader)
    for source in sources:
        assert await store.pending(source.conversation_id)==1
    for index,mode in enumerate(modes):
        connection = str(uuid4())
        challenge = await store.hello(page,connection,reader=reader,mode=mode)
        assert await store.reset_ack(connection,challenge.id,challenge.nonce,reader=reader)
        await store.disconnect(connection,reader=reader)
        for position,source in enumerate(sources):
            assert await store.pending(source.conversation_id)==int(position>index)
