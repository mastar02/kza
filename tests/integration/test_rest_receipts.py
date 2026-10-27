"""Actual REST reads, middleware buffers and send callbacks join browser receipts."""

import asyncio
from contextlib import asynccontextmanager
import json
import threading

import httpx
import pytest

from src.analytics.event_logger import EventLogger
from src.core.logging import LogContext,get_context
from src.core.owned_work import WorkCloseTimeout
from src.dashboard.api import DashboardAPI
from tests.integration.test_live_stream_receipts import graph,flush
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.memory_sources import memory_source
from tests.mocks.rest_browser import ready_browser

pytestmark = pytest.mark.asyncio
PRIVATE = 'Synthetic private REST receipt turn'


@asynccontextmanager
async def events_for(tmp_path, privacy):
    events = EventLogger(str(tmp_path/'events.db'),source_policy=privacy.ledger)
    await events.acall(events.initialize)
    try:
        yield events
    finally:
        await events.aclose()


async def rest_step(privacy, operation):
    def read():
        with privacy.ledger._transaction() as db:
            return dict(db.execute("""SELECT s.* FROM erasure_steps s JOIN erasure_targets t ON t.id=s.target_id
                WHERE s.operation_id=? AND t.participant='streams' AND t.version=3""",(operation['id'],)).fetchone())
    return await privacy.acall(read)


async def test_rest_requires_auth_and_a_confirmed_version_two_connection_before_read(tmp_path,dashboard_auth,monkeypatch):
    async with graph(tmp_path) as (privacy,_,bus,_,_):
        async with events_for(tmp_path,privacy) as events:
            reads = []
            monkeypatch.setattr(events,'get_events',lambda **kw:reads.append(True))
            app = DashboardAPI(event_logger=events,event_bus=bus,observability_use_mocks=False).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
                assert (await client.get('/api/conversations')).status_code==401
                assert (await client.get('/api/conversations',headers=dashboard_auth)).status_code==428
                response = await client.get('/api/conversations',headers={**dashboard_auth,'X-KZA-Live-Connection':'unknown'})
                assert response.status_code==428 and not reads


async def test_ticket_precedes_actual_outer_body_and_ack_cannot_hide_buffered_send(tmp_path,dashboard_auth):
    from src.memory.fact_ledger import MemoryReviewer
    async with graph(tmp_path) as (privacy,owner,bus,store,clock):
        async with events_for(tmp_path,privacy) as events, ready_browser(bus) as peer:
            source = memory_source(now=clock.now)
            await events.alog('light.synthetic','turn_on',trigger_phrase=PRIVATE,sources=(source,))
            api = DashboardAPI(event_logger=events,event_bus=bus,observability_use_mocks=False)
            # This BaseHTTPMiddleware sits between the private Response and the
            # actual outer send. Its buffered copy must remain in the journal.
            @api.app.middleware('http')
            async def buffered(request,call_next):
                return await call_next(request)
            entered, release = asyncio.Event(),asyncio.Event()
            documents = []
            async def gated(scope,receive,send):
                async def actual(message):
                    assert get_context()=={}
                    if message['type']=='http.response.body' and PRIVATE.encode() in message.get('body',b''):
                        documents.append(json.loads(message['body']))
                        assert await store.pending(source.conversation_id)==1
                        entered.set()
                        await release.wait()
                    await send(message)
                await api.app(scope,receive,actual)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gated),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                with LogContext(private='Synthetic caller context'):
                    observing = asyncio.create_task(client.get('/api/conversations'))
                try:
                    await until(entered.is_set)
                    receipt = documents[0]['snapshot']['receipt']
                    await peer.message({'type':'live_ack','receipts':[receipt]})
                    async with asyncio.timeout(3):
                        while await store.pending(source.conversation_id):
                            await asyncio.sleep(.001)
                    operation = await owner.request(source.conversation_id,MemoryReviewer('synthetic-admin'))
                    await flush(owner)
                    assert (await rest_step(privacy,operation))['state']=='pending'
                    assert bus.resource_status()['rest_operations']>0
                    release.set()
                    assert (await observing).status_code==200
                    await until(lambda:bus.resource_status()['rest_operations']==0)
                    clock.now+=2
                    await flush(owner)
                    assert (await rest_step(privacy,operation))['state']=='complete'
                    assert (await privacy.acall(privacy.ledger.erasures.snapshot,operation['id']))['state']!='complete'
                finally:
                    release.set()
                    await observing


async def test_cancelled_observer_keeps_unknown_native_read_in_streams_claim(tmp_path,dashboard_auth,monkeypatch):
    from src.memory.fact_ledger import MemoryReviewer
    async with graph(tmp_path) as (privacy,owner,bus,_,clock):
        async with events_for(tmp_path,privacy) as events, ready_browser(bus) as peer:
            source = memory_source(now=clock.now)
            await events.alog('light.synthetic','turn_on',trigger_phrase=PRIVATE,sources=(source,))
            entered,release = threading.Event(),threading.Event()
            get = events.get_events
            def blocked(**kwargs):
                assert get_context()=={}
                entered.set()
                assert release.wait(3)
                return get(**kwargs)
            monkeypatch.setattr(events,'get_events',blocked)
            app = DashboardAPI(event_logger=events,event_bus=bus,observability_use_mocks=False).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                with LogContext(private='Synthetic cancelled caller context'):
                    observing = asyncio.create_task(client.get('/api/conversations'))
                try:
                    await until(entered.is_set)
                    observing.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observing
                    operation = await owner.request(source.conversation_id,MemoryReviewer('synthetic-admin'))
                    await flush(owner)
                    assert (await rest_step(privacy,operation))['state']=='pending'
                    assert bus.resource_status()['rest_operations']==1
                finally:
                    release.set()
                    await until(lambda:bus.resource_status()['rest_operations']==0)
                clock.now+=2
                await flush(owner)
                assert (await rest_step(privacy,operation))['state']=='complete'


@pytest.mark.parametrize('change',['expire','revoke','session'])
async def test_source_or_session_change_while_headers_wait_never_starts_private_body(tmp_path,dashboard_auth,monkeypatch,change):
    from src.memory.fact_ledger import MemoryReviewer
    async with graph(tmp_path) as (privacy,_,bus,store,clock):
        async with events_for(tmp_path,privacy) as events, ready_browser(bus) as peer:
            source = memory_source(now=clock.now,ttl=60)
            await events.alog('light.synthetic','turn_on',trigger_phrase=PRIVATE,sources=(source,))
            api = DashboardAPI(event_logger=events,event_bus=bus,observability_use_mocks=False)
            entered,release = asyncio.Event(),asyncio.Event()
            async def gated(scope,receive,send):
                async def actual(message):
                    if message['type']=='http.response.start':
                        entered.set()
                        await release.wait()
                    await send(message)
                await api.app(scope,receive,actual)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gated),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                observing = asyncio.create_task(client.get('/api/conversations'))
                try:
                    await until(entered.is_set)
                    assert await store.pending(source.conversation_id)==1
                    if change=='expire':
                        clock.now+=60
                    elif change=='revoke':
                        await privacy.acall(privacy.ledger.delete_conversation,source.conversation_id,MemoryReviewer('synthetic-admin'))
                    else:
                        monkeypatch.setenv('KZA_DASHBOARD_TOKEN','synthetic-rotated-token')
                    release.set()
                    response = await observing
                    assert PRIVATE not in response.text and response.json()['snapshot'] is None
                    assert await store.pending(source.conversation_id)==1
                finally:
                    release.set()
                    await observing


async def test_receipt_metadata_contains_no_transcript_and_disconnect_does_not_retire(tmp_path,dashboard_auth):
    async with graph(tmp_path) as (privacy,_,bus,store,clock):
        async with events_for(tmp_path,privacy) as events:
            source = memory_source(now=clock.now)
            await events.alog('light.synthetic','turn_on',trigger_phrase=PRIVATE,sources=(source,))
            app = DashboardAPI(event_logger=events,event_bus=bus,observability_use_mocks=False).app
            async with ready_browser(bus) as peer:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                        headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                    response = await client.get('/api/conversations')
                    assert response.json()['snapshot']['rows'][0]['stt']==PRIVATE
                    assert response.headers['cache-control']=='no-store'
            assert await store.pending(source.conversation_id)==1
            def metadata():
                with privacy.ledger._transaction() as db:
                    return str(db.execute('SELECT * FROM live_deliveries').fetchall())
            assert PRIVATE not in await privacy.acall(metadata)


async def test_scope_capacity_and_close_timeout_keep_actual_native_operations(tmp_path,dashboard_auth,monkeypatch):
    async with graph(tmp_path) as (privacy,_,bus,_,_):
        async with events_for(tmp_path,privacy) as events, ready_browser(bus) as peer:
            entered,release = [],threading.Event()
            get = events.get_events
            def blocked(**kwargs):
                entered.append(True)
                assert release.wait(3)
                return get(**kwargs)
            monkeypatch.setattr(events,'get_events',blocked)
            app = DashboardAPI(event_logger=events,event_bus=bus,observability_use_mocks=False).app
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver',
                    headers={**dashboard_auth,'X-KZA-Live-Connection':peer.connection}) as client:
                observers = [asyncio.create_task(client.get('/api/conversations')) for _ in range(8)]
                try:
                    await until(lambda:len(entered)==2 and bus.resource_status()['rest_operations']==8)
                    assert (await client.get('/api/conversations')).status_code==503
                    assert len(entered)==2 and events.resource_status()['operations']==8
                    for observing in observers:
                        observing.cancel()
                    await asyncio.gather(*observers,return_exceptions=True)
                    with pytest.raises(WorkCloseTimeout):
                        await bus.aclose()
                    assert bus.resource_status()['rest_operations']==8
                    assert not bus._retention._closing.done()
                finally:
                    release.set()
                    await asyncio.gather(*observers,return_exceptions=True)
                    await bus.aclose()
                assert bus.resource_status()['rest_operations']==0 and events.resource_status()['operations']==0
