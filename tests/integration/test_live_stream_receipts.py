"""Browser obligations survive disconnect and join exact STREAMS cleanup claims."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import json
from uuid import uuid4

import pytest

from src.dashboard.live_erasure import build_live_services
from src.dashboard.live_event_bus import LiveEvent, LiveEventType
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.ambient_graph import Clock
from tests.mocks.memory_sources import memory_source

pytestmark=pytest.mark.asyncio
ACTOR=MemoryReviewer('synthetic-admin')


@asynccontextmanager
async def graph(tmp_path,timeout=1):
    async with AsyncExitStack() as stack:
        clock=Clock(2_000_000_000.)
        privacy=PrivacyServices(str(tmp_path/'privacy.db'),clock=clock,close_timeout_s=timeout)
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        owner=ErasureCoordinator(privacy,close_timeout_s=timeout)
        stack.push_async_callback(owner.aclose)
        bus=await build_live_services({'dashboard':{'live_event_queue_size':2}},privacy=privacy,owner=owner,shutdown=stack)
        yield privacy,owner,bus,bus._retention.receipts,clock


@asynccontextmanager
async def browser(bus,client=None,gate=None,handshake=True):
    sub,_=await bus.subscribe()
    incoming,outgoing=asyncio.Queue(),asyncio.Queue()
    async def send(frame):
        await outgoing.put(frame)
        if gate is not None and frame['type']=='turn':
            await gate.wait()
    observing=asyncio.create_task(bus.browser_stream(sub,send,receiver=incoming.get))
    async def message(doc):
        await incoming.put({'type':'websocket.receive','text':json.dumps(doc)})
    client=client or str(uuid4())
    try:
        if handshake:
            await message({'type':'live_hello','version':1,'client_id':client})
            async with asyncio.timeout(3):
                reset=await outgoing.get()
            assert reset['type']=='live_reset'
            await message({'type':'live_reset_ack',**reset['payload']})
            await until(lambda:bus._retention._states[sub].browser.ready)
        yield message,outgoing,client
    finally:
        if gate is not None:
            gate.set()
        await incoming.put({'type':'websocket.disconnect','code':1000})
        async with asyncio.timeout(3):
            await observing


async def flush(owner):
    await owner.reconcile()
    await asyncio.gather(*(active.task for active in owner._active.values()))
    await owner.reconcile()


async def local_step(privacy,operation):
    def read():
        with privacy.ledger._transaction() as db:
            return dict(db.execute("""SELECT s.* FROM erasure_steps s JOIN erasure_targets t ON t.id=s.target_id
                WHERE s.operation_id=? AND t.participant='streams' AND t.version=1""",(operation['id'],)).fetchone())
    return await privacy.acall(read)


async def publish(bus,source):
    await bus.publish(LiveEvent(LiveEventType.TURN,payload={'stt':'Synthetic private receipt turn'},sources=(source,)))


async def test_private_delivery_waits_for_clear_handshake_and_durable_ticket(tmp_path):
    async with graph(tmp_path) as (_,_,bus,store,clock):
        async with browser(bus,handshake=False) as (message,outgoing,client):
            source=memory_source(now=clock.now)
            await publish(bus,source)
            await asyncio.sleep(.01)
            assert outgoing.empty() and await store.pending(source.conversation_id)==0
            await message({'type':'live_hello','version':1,'client_id':client})
            async with asyncio.timeout(3):
                reset=await outgoing.get()
            assert reset['type']=='live_reset'
            assert outgoing.empty()
            await message({'type':'live_reset_ack',**reset['payload']})
            async with asyncio.timeout(3):
                frame=await outgoing.get()
            assert frame['type']=='turn' and frame['receipt']['ticket']
            assert await store.pending(source.conversation_id)==1


async def test_local_step_stays_pending_until_browser_ack_and_inventory_blocks_global_completion(tmp_path):
    async with graph(tmp_path) as (privacy,owner,bus,store,clock):
        async with browser(bus) as (message,outgoing,_):
            source=memory_source(now=clock.now)
            await publish(bus,source)
            async with asyncio.timeout(3):
                frame=await outgoing.get()
            operation=await owner.request(source.conversation_id,ACTOR)
            await flush(owner)
            assert (await local_step(privacy,operation))['state']=='pending'
            async with asyncio.timeout(3):
                retired=await outgoing.get()
            assert retired['payload']['receipts']==[{'event_id':frame['event_id'],**frame['receipt']}]
            await message({'type':'live_ack','receipts':[frame['receipt']]})
            async with asyncio.timeout(3):
                while await store.pending(source.conversation_id):
                    await asyncio.sleep(.001)
            clock.now+=2
            await flush(owner)
            assert (await local_step(privacy,operation))['state']=='complete'
            assert (await privacy.acall(privacy.ledger.erasures.snapshot,operation['id']))['state']!='complete'


async def test_peer_ack_does_not_hide_a_still_running_actual_send_callback(tmp_path):
    async with graph(tmp_path) as (privacy,owner,bus,store,clock):
        gate=asyncio.Event()
        async with browser(bus,gate=gate) as (message,outgoing,_):
            source=memory_source(now=clock.now)
            await publish(bus,source)
            async with asyncio.timeout(3):
                frame=await outgoing.get()
            # The synthetic peer has cleared, but the ASGI send still owns bytes.
            await message({'type':'live_ack','receipts':[frame['receipt']]})
            async with asyncio.timeout(3):
                while await store.pending(source.conversation_id):
                    await asyncio.sleep(.001)
            operation=await owner.request(source.conversation_id,ACTOR)
            await flush(owner)
            assert (await local_step(privacy,operation))['pending_count']>0
            gate.set()
            await until(lambda:bus.pending_conversation(source.conversation_id)==0)
            clock.now+=2
            await flush(owner)
            assert (await local_step(privacy,operation))['state']=='complete'


async def test_unacknowledged_client_survives_restart_and_only_its_new_clear_settles_it(tmp_path):
    async with graph(tmp_path) as (privacy,owner,bus,store,clock):
        async with browser(bus) as (_,outgoing,client):
            source=memory_source(now=clock.now)
            await publish(bus,source)
            async with asyncio.timeout(3):
                await outgoing.get()
        operation=await owner.request(source.conversation_id,ACTOR)
        await flush(owner)
        assert (await local_step(privacy,operation))['state']=='pending'
    async with graph(tmp_path) as (privacy,owner,bus,store,clock):
        assert await store.pending(source.conversation_id)==1
        async with browser(bus,client=client):
            assert await store.pending(source.conversation_id)==0
            await flush(owner)
            assert (await local_step(privacy,operation))['state']=='complete'


async def test_unattributed_restored_ticket_blocks_receipt_without_invented_lineage(tmp_path):
    async with graph(tmp_path) as (privacy,owner,bus,store,clock):
        async with browser(bus) as (message,outgoing,_):
            source=memory_source(now=clock.now)
            await publish(bus,source)
            async with asyncio.timeout(3):
                frame=await outgoing.get()
            def corrupt():
                with privacy.ledger._transaction() as db:
                    db.execute('DELETE FROM live_delivery_sources')
            await privacy.acall(corrupt)
            unrelated=memory_source(now=clock.now)
            operation=await owner.request(unrelated.conversation_id,ACTOR)
            await flush(owner)
            assert (await local_step(privacy,operation))['state']=='pending'
            await message({'type':'live_ack','receipts':[frame['receipt']]})
            async with asyncio.timeout(3):
                while await store.pending(unrelated.conversation_id):
                    await asyncio.sleep(.001)
            clock.now+=2
            await flush(owner)
            assert (await local_step(privacy,operation))['state']=='complete'


async def test_idle_page_cannot_hold_handshake_slot_after_its_fixed_deadline(tmp_path):
    async with graph(tmp_path) as (_,_,bus,_,_):
        sub,_=await bus.subscribe()
        incoming=asyncio.Queue()
        frames=[]
        async def send(frame):
            frames.append(frame)
        task=asyncio.create_task(bus.browser_stream(sub,send,receiver=incoming.get))
        await until(lambda:bus._retention._states[sub].browser is not None)
        bus._retention._states[sub].browser._deadline-=10
        bus._retention.prune()
        async with asyncio.timeout(3):
            await task
        assert not frames and bus.subscriber_count==0


async def test_cancelled_ack_observer_and_close_retain_actual_sqlite_commit_and_lease(tmp_path,monkeypatch):
    import threading
    from src.core.owned_work import WorkCloseTimeout
    async with graph(tmp_path,timeout=.03) as (privacy,_,bus,store,clock):
        async with browser(bus) as (message,outgoing,_):
            source=memory_source(now=clock.now)
            await publish(bus,source)
            async with asyncio.timeout(3):
                frame=await outgoing.get()
            entered,release=threading.Event(),threading.Event()
            ack=store._ack
            def held(db,*args):
                entered.set()
                assert release.wait(3)
                return ack(db,*args)
            monkeypatch.setattr(store,'_ack',held)
            await message({'type':'live_ack','receipts':[frame['receipt']]})
            try:
                await until(entered.is_set)
                with pytest.raises(WorkCloseTimeout):
                    await privacy.aclose()
                assert privacy.resource_status()['writer_lease']
                assert store.status()['native_operations']==1
            finally:
                release.set()
                await privacy.begin_close()
            assert not privacy.resource_status()['writer_lease']


async def test_restored_ticket_reopens_previously_completed_local_step(tmp_path):
    async with graph(tmp_path) as (privacy,owner,bus,store,clock):
        async with browser(bus) as (message,outgoing,_):
            source=memory_source(now=clock.now)
            await publish(bus,source)
            async with asyncio.timeout(3):
                frame=await outgoing.get()
            def snapshot():
                with privacy.ledger._transaction() as db:
                    return dict(db.execute('SELECT * FROM live_deliveries').fetchone())
            ticket=await privacy.acall(snapshot)
            operation=await owner.request(source.conversation_id,ACTOR)
            await message({'type':'live_ack','receipts':[frame['receipt']]})
            async with asyncio.timeout(3):
                while await store.pending(source.conversation_id):
                    await asyncio.sleep(.001)
            await flush(owner)
            assert (await local_step(privacy,operation))['state']=='complete'
    # Simulate restoring an older metadata table while keeping the newer journal.
    async with graph(tmp_path) as (privacy,owner,bus,store,clock):
        def restore():
            with privacy.ledger._transaction() as db:
                db.execute('INSERT INTO live_deliveries VALUES (?,?,?,?,?,?,?,?)',tuple(ticket.values()))
                db.execute('INSERT INTO live_delivery_sources VALUES (?,?)',(ticket['ticket'],source.conversation_id))
        await privacy.acall(restore)
        assert (await local_step(privacy,operation))['state']=='pending'
        await flush(owner)
        assert (await local_step(privacy,operation))['state']=='pending'
        async with browser(bus,client=ticket['client_id']):
            clock.now+=2
            await flush(owner)
            assert (await local_step(privacy,operation))['state']=='complete'


async def test_disconnect_during_native_ticket_issue_never_starts_late_private_send(tmp_path,monkeypatch):
    import threading
    async with graph(tmp_path) as (_,_,bus,store,clock):
        sub,_=await bus.subscribe()
        incoming,outgoing=asyncio.Queue(),asyncio.Queue()
        async def send(frame):
            await outgoing.put(frame)
        observing=asyncio.create_task(bus.browser_stream(sub,send,receiver=incoming.get))
        async def message(doc):
            await incoming.put({'type':'websocket.receive','text':json.dumps(doc)})
        await message({'type':'live_hello','version':1,'client_id':str(uuid4())})
        async with asyncio.timeout(3):
            reset=await outgoing.get()
        await message({'type':'live_reset_ack',**reset['payload']})
        await until(lambda:bus._retention._states[sub].browser.ready)
        entered,release=threading.Event(),threading.Event()
        issue=store._issue
        def held(db,*args):
            entered.set()
            assert release.wait(3)
            return issue(db,*args)
        monkeypatch.setattr(store,'_issue',held)
        source=memory_source(now=clock.now)
        await publish(bus,source)
        try:
            await until(entered.is_set)
            await incoming.put({'type':'websocket.disconnect','code':1000})
            await until(lambda:bus._retention._states[sub].closed)
        finally:
            release.set()
            async with asyncio.timeout(3):
                await observing
        assert outgoing.empty() and await store.pending(source.conversation_id)==1
