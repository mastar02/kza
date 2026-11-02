"""Voice-only connections share original-reader proofs, never technical debt."""

import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from src.dashboard.api import DashboardAPI
from src.dashboard.live_event_bus import LiveEvent, LiveEventType
from src.dashboard.live_retention_types import LiveSubscriptionMode as Mode, VOICE_CHANNELS
from tests.integration.test_live_receipts import graph as receipt_graph, identity, READER
from tests.integration.test_live_stream_receipts import graph
from tests.integration.test_rest_receipts import events_for
from tests.mocks.memory_sources import memory_source
from tests.mocks.rest_browser import ready_browser
from tests.mocks.voice_registry import web_graph

pytestmark = pytest.mark.asyncio
PRIVATE = 'Synthetic technical content excluded from voices'


async def test_control_subscription_never_projects_or_enqueues_general_publications(tmp_path,monkeypatch):
    async with graph(tmp_path) as (privacy,_,bus,_,_):
        subscription,queue = await bus.subscribe(reader=bus.test_reader,mode=Mode.VOICES_CONTROL_ONLY)
        calls = []
        def forbidden(*args):
            calls.append(True)
            raise AssertionError('voice-only audience caused general projection')
        monkeypatch.setattr(bus._retention,'admit',forbidden)
        try:
            assert subscription not in bus._subscribers
            assert bus.subscriber_count==1
            for kind in LiveEventType:
                await bus.publish(LiveEvent(kind,payload={'stt':PRIVATE}))
            assert queue.empty() and calls==[]
            def count():
                with privacy.ledger._transaction() as db:
                    return db.execute('SELECT COUNT(*) FROM live_deliveries').fetchone()[0]
            assert await privacy.acall(count)==0
        finally:
            await bus.unsubscribe(subscription)


async def test_technical_peer_receives_frames_while_voice_peer_has_no_general_queue_or_tickets(tmp_path):
    async with graph(tmp_path) as (privacy,_,bus,_,clock):
        page = str(uuid4())
        async with ready_browser(bus,version=3,mode=Mode.VOICES_CONTROL_ONLY,client_id=page) as voice, \
                ready_browser(bus,version=3,client_id=page) as technical:
            assert voice.ready_frame['channels']==sorted(VOICE_CHANNELS)
            assert 'channels' not in technical.ready_frame
            source = memory_source(now=clock.now)
            await bus.publish(LiveEvent(LiveEventType.TURN,payload={'stt':PRIVATE},sources=(source,)))
            frame = await asyncio.wait_for(technical.outgoing.get(),3)
            assert frame['payload']['stt']==PRIVATE and 'receipt' in frame
            assert voice.outgoing.empty()
            state = bus._retention.rest.connection(voice.connection,reader=voice.reader,channel='voices:profiles',minimum_protocol=3)
            assert state.queue.empty() and not state.browser._tickets
            def count():
                with privacy.ledger._transaction() as db:
                    return db.execute('SELECT COUNT(*) FROM live_deliveries WHERE connection_id=?',(voice.connection,)).fetchone()[0]
            assert await privacy.acall(count)==0


async def test_voice_uuid_cannot_admit_technical_native_reader(tmp_path,dashboard_auth,monkeypatch):
    async with web_graph(tmp_path) as (privacy,_,voices,_,bus):
        async with events_for(tmp_path,privacy) as events, ready_browser(bus,version=3,mode=Mode.VOICES_CONTROL_ONLY) as voice:
            reads = []
            def forbidden(*args,**kwargs):
                reads.append(True)
                raise AssertionError('voice UUID admitted technical events')
            monkeypatch.setattr(events,'get_events',forbidden)
            api = DashboardAPI(authority=bus.test_authority,event_bus=bus,event_logger=events,
                               voice_registry=voices,observability_use_mocks=False)
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://testserver',
                        headers={**dashboard_auth,'X-KZA-Live-Connection':voice.connection}) as client:
                    response = await client.get('/api/conversations')
                    assert response.status_code==428 and reads==[]
                    assert (await client.get('/api/voices/profiles')).status_code==200
                    assert (await client.get('/api/admin/auth/whoami')).status_code==200
            finally:
                await api.stop()


async def test_control_mode_rejects_foreign_channel_before_ticket_or_frame(tmp_path):
    async with graph(tmp_path) as (privacy,_,bus,_,_):
        async with ready_browser(bus,version=3,mode=Mode.VOICES_CONTROL_ONLY) as voice:
            state = bus._retention.rest.connection(voice.connection,reader=voice.reader,channel='voices:review',minimum_protocol=3)
            with pytest.raises(PermissionError,match='live_channel_denied'):
                await state.browser.before_frame(SimpleNamespace(channel='conversations'))
            sent = []
            async def send(frame):
                sent.append(frame)
            with pytest.raises(PermissionError,match='live_channel_denied'):
                await bus._retention._send(send,{'type':'turn','payload':{'stt':PRIVATE},'ts':''},state)
            assert sent==[]
            def count():
                with privacy.ledger._transaction() as db:
                    return db.execute('SELECT COUNT(*) FROM live_deliveries').fetchone()[0]
            assert await privacy.acall(count)==0


async def test_scope_reset_covers_only_its_server_domain_for_same_login_and_page(tmp_path):
    async with receipt_graph(tmp_path) as (_,store,clock):
        page,technical,voice = identity(),identity(),identity()
        challenge = await store.hello(page,technical,reader=READER,mode=Mode.TECHNICAL)
        assert await store.reset_ack(technical,challenge.id,challenge.nonce,reader=READER)
        first = memory_source(now=clock.now)
        await store.issue(technical,identity(),first.expires_at,(first,),reader=READER)
        await store.disconnect(technical,reader=READER)
        challenge = await store.hello(page,voice,reader=READER,mode=Mode.VOICES_CONTROL_ONLY)
        assert await store.reset_ack(voice,challenge.id,challenge.nonce,reader=READER)
        assert await store.pending(first.conversation_id)==1
        second = memory_source(now=clock.now)
        await store.issue(voice,identity(),second.expires_at,(second,),reader=READER)
        await store.disconnect(voice,reader=READER)
        technical,voice = identity(),identity()
        challenge = await store.hello(page,technical,reader=READER,mode=Mode.TECHNICAL)
        assert await store.reset_ack(technical,challenge.id,challenge.nonce,reader=READER)
        assert await store.pending(first.conversation_id)==0
        assert await store.pending(second.conversation_id)==1
        challenge = await store.hello(page,voice,reader=READER,mode=Mode.VOICES_CONTROL_ONLY)
        assert await store.reset_ack(voice,challenge.id,challenge.nonce,reader=READER)
        assert await store.pending(second.conversation_id)==0


async def test_unprefixed_legacy_page_is_never_adopted_and_wire_cannot_choose_prefix(tmp_path):
    async with receipt_graph(tmp_path) as (privacy,store,clock):
        page,connection = identity(),identity()
        challenge = await store.hello(page,connection,reader=READER)
        assert await store.reset_ack(connection,challenge.id,challenge.nonce,reader=READER)
        source = memory_source(now=clock.now)
        await store.issue(connection,identity(),source.expires_at,(source,),reader=READER)
        await store.disconnect(connection,reader=READER)
        def restore_old_page():
            with privacy.ledger._transaction() as db:
                db.execute('UPDATE live_deliveries SET client_id=?',(page,))
        await privacy.acall(restore_old_page)
        for mode in Mode:
            next_connection = identity()
            challenge = await store.hello(page,next_connection,reader=READER,mode=mode)
            assert await store.reset_ack(next_connection,challenge.id,challenge.nonce,reader=READER)
        assert await store.pending(source.conversation_id)==1
        with pytest.raises(RuntimeError,match='live_receipt_failed'):
            await store.hello('technical:'+page,identity(),reader=READER)


@asynccontextmanager
async def socket_frames(app,authorization):
    incoming,outgoing = asyncio.Queue(),asyncio.Queue()
    scope = {'type':'websocket','path':'/ws/voices','raw_path':b'/ws/voices','root_path':'',
        'scheme':'ws','query_string':b'','server':('localhost',80),'client':('127.0.0.1',12345),
        'headers':[(b'authorization',authorization.encode())],'subprotocols':[],'state':{}}
    await incoming.put({'type':'websocket.connect'})
    observing = asyncio.create_task(app(scope,incoming.get,outgoing.put))
    try:
        yield incoming,outgoing
    finally:
        await incoming.put({'type':'websocket.disconnect','code':1000})
        await asyncio.wait_for(observing,3)


@pytest.mark.parametrize('version',[1,2,3])
async def test_actual_voice_route_requires_version_three_before_connection_or_reset(tmp_path,dashboard_auth,version):
    async with web_graph(tmp_path) as (privacy,_,voices,_,bus):
        api = DashboardAPI(authority=bus.test_authority,event_bus=bus,voice_registry=voices)
        try:
            async with socket_frames(api.app,dashboard_auth['Authorization']) as (incoming,outgoing):
                assert (await asyncio.wait_for(outgoing.get(),3))['type']=='websocket.accept'
                await incoming.put({'type':'websocket.receive','text':json.dumps({'type':'live_hello','version':version,'client_id':str(uuid4())})})
                message = await asyncio.wait_for(outgoing.get(),3)
                if version<3:
                    assert message['type']=='websocket.close'
                else:
                    frame = json.loads(message['text'])
                    assert frame['type']=='live_reset'
                    await incoming.put({'type':'websocket.receive','text':json.dumps({'type':'live_reset_ack',**frame['payload']})})
                    frame = json.loads((await asyncio.wait_for(outgoing.get(),3))['text'])
                    assert frame['type']=='rest_ready' and frame['channels']==sorted(VOICE_CHANNELS)
            if version<3:
                def counts():
                    with privacy.ledger._transaction() as db:
                        return tuple(db.execute('SELECT (SELECT COUNT(*) FROM live_connections),(SELECT COUNT(*) FROM live_resets),(SELECT COUNT(*) FROM live_deliveries)').fetchone())
                assert await privacy.acall(counts)==(0,0,0)
        finally:
            await api.stop()


@pytest.mark.parametrize('role',['home','evaluator'])
async def test_voice_route_denies_nonreviewers_before_subscribe(tmp_path,dashboard_auth,role):
    import time
    from src.dashboard.authority import DashboardRole,EvaluationModality,EvaluationSample
    from tests.mocks.browser_authority import fixture_token
    async with web_graph(tmp_path) as (_,_,voices,_,bus):
        authority = bus.test_authority
        samples = (EvaluationSample('vf_'+'a'*64,frozenset({'b'*64}),frozenset({EvaluationModality.AUDIO_PREVIEW}),time.time()+100),) if role=='evaluator' else ()
        credential = await authority.issue_credential(authority.resolve(fixture_token()),subject='synthetic-nonreviewer',
            role=DashboardRole(role),expires_at=time.time()+120,samples=samples)
        api = DashboardAPI(authority=authority,event_bus=bus,voice_registry=voices)
        try:
            async with socket_frames(api.app,'Bearer '+credential.token) as (_,outgoing):
                frame = await asyncio.wait_for(outgoing.get(),3)
                assert frame['type']=='websocket.close' and frame['code']==1008
                assert bus.subscriber_count==0 and not bus._retention._states
        finally:
            await api.stop()
