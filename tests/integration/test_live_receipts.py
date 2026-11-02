"""Durable live obligations are retired by exact client capabilities only."""

from contextlib import AsyncExitStack, asynccontextmanager
import asyncio
from contextlib import closing
import sqlite3
import threading
from uuid import uuid4

import pytest

from src.dashboard.live_receipts import LiveReceiptStore
from src.core.reader_binding import ReaderBinding
from src.memory.privacy_services import PrivacyServices
from tests.mocks.memory_sources import memory_source
from tests.mocks.ambient_graph import Clock

pytestmark=pytest.mark.asyncio
READER = ReaderBinding('a'*64, lambda: True)


def identity():
    return str(uuid4())


@asynccontextmanager
async def graph(tmp_path, **kwargs):
    async with AsyncExitStack() as stack:
        clock=Clock(2_000_000_000.)
        privacy=PrivacyServices(str(tmp_path/'privacy.db'),clock=clock)
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        store=LiveReceiptStore(privacy.ledger,**kwargs)
        privacy.add_borrower(store.begin_close)
        stack.push_async_callback(lambda:store.begin_close())
        yield privacy,store,clock


async def connected(store, client=None):
    client,connection=client or identity(),identity()
    reset=await store.hello(client,connection, reader=READER)
    assert await store.reset_ack(connection,reset.id,reset.nonce, reader=READER)
    return client,connection


async def test_receipt_is_content_free_and_wrong_nonce_or_other_connection_cannot_retire(tmp_path):
    async with graph(tmp_path) as (privacy,store,clock):
        _,first=await connected(store)
        _,second=await connected(store)
        source=memory_source(now=clock.now)
        ticket=await store.issue(first,identity(),source.expires_at,(source,), reader=READER)
        assert await store.pending(source.conversation_id)==1
        await store.ack(second,((ticket.ticket,ticket.nonce),), reader=READER)
        await store.ack(first,((ticket.ticket,identity()),), reader=READER)
        assert await store.pending(source.conversation_id)==1
        await store.ack(first,((ticket.ticket,ticket.nonce),), reader=READER)
        assert await store.pending(source.conversation_id)==0
        assert 'Synthetic' not in repr(ticket)
        with privacy.ledger._transaction() as db:
            assert db.execute('SELECT COUNT(*) FROM live_delivery_sources').fetchone()[0]==0


async def test_disconnect_expiry_and_restart_never_infer_browser_erasure(tmp_path):
    async with graph(tmp_path) as (_,store,clock):
        client,connection=await connected(store)
        source=memory_source(now=clock.now,ttl=1)
        ticket=await store.issue(connection,identity(),source.expires_at,(source,), reader=READER)
        await store.disconnect(connection, reader=READER)
        clock.now+=2
        assert await store.pending(source.conversation_id)==1
    async with graph(tmp_path) as (_,store,_):
        assert await store.pending(source.conversation_id)==1
        new_connection=identity()
        reset=await store.hello(client,new_connection, reader=READER)
        await store.ack(new_connection,((ticket.ticket,ticket.nonce),), reader=READER)
        assert await store.pending(source.conversation_id)==1
        assert not await store.reset_ack(new_connection,reset.id,identity(), reader=READER)
        assert await store.pending(source.conversation_id)==1
        assert await store.reset_ack(new_connection,reset.id,reset.nonce, reader=READER)
        assert await store.pending(source.conversation_id)==0


async def test_reset_ack_covers_only_its_client_and_captured_cutoff(tmp_path):
    async with graph(tmp_path) as (_,store,clock):
        _,connection=await connected(store)
        _,other=await connected(store)
        source,later,unrelated=(memory_source(now=clock.now) for _ in range(3))
        await store.issue(connection,identity(),source.expires_at,(source,), reader=READER)
        reset=await store.reset(connection, reader=READER)
        await store.issue(connection,identity(),later.expires_at,(later,), reader=READER)
        await store.issue(other,identity(),unrelated.expires_at,(unrelated,), reader=READER)
        assert not await store.reset_ack(other,reset.id,reset.nonce, reader=READER)
        assert await store.reset_ack(connection,reset.id,reset.nonce, reader=READER)
        assert await store.pending(source.conversation_id)==0
        assert await store.pending(later.conversation_id)==1
        assert await store.pending(unrelated.conversation_id)==1


async def test_duplicate_page_cannot_evict_live_connection_or_take_its_reset(tmp_path):
    async with graph(tmp_path) as (_,store,_):
        client,connection=await connected(store)
        with pytest.raises(RuntimeError,match='live_receipt_failed'):
            await store.hello(client,identity(), reader=READER)
        reset=await store.reset(connection, reader=READER)
        assert await store.reset_ack(connection,reset.id,reset.nonce, reader=READER)


async def test_global_and_client_caps_fail_closed_without_deleting_unacked_rows(tmp_path):
    async with graph(tmp_path,capacity=3,client_capacity=2) as (_,store,clock):
        _,connection=await connected(store)
        _,other=await connected(store)
        source=memory_source(now=clock.now)
        assert await store.issue(connection,identity(),source.expires_at,(source,), reader=READER)
        assert await store.issue(connection,identity(),source.expires_at,(source,), reader=READER)
        assert await store.issue(connection,identity(),source.expires_at,(source,), reader=READER) is None
        assert await store.issue(other,identity(),source.expires_at,(source,), reader=READER)
        assert await store.issue(other,identity(),source.expires_at,(source,), reader=READER) is None
        assert await store.pending(source.conversation_id)==3


async def test_concurrent_page_handshakes_preserve_all_connection_leases(tmp_path):
    import asyncio
    async with graph(tmp_path) as (_,store,_):
        pages=await asyncio.gather(*(connected(store) for _ in range(16)))
        for _,connection in pages:
            reset=await store.reset(connection, reader=READER)
            assert await store.reset_ack(connection,reset.id,reset.nonce, reader=READER)


async def test_second_receipt_owner_cannot_recover_over_the_live_writer(tmp_path):
    async with graph(tmp_path) as (privacy,store,_):
        _,connection=await connected(store)
        with pytest.raises(RuntimeError,match='live_receipt_owner_active'):
            LiveReceiptStore(privacy.ledger)
        reset=await store.reset(connection, reader=READER)
        assert await store.reset_ack(connection,reset.id,reset.nonce, reader=READER)


async def test_failed_disconnect_metadata_retries_without_evicting_an_active_peer(tmp_path,monkeypatch):
    async with graph(tmp_path) as (_,store,_):
        client,connection=await connected(store)
        disconnect=store._disconnect
        def failed(db,namespace,identity):
            disconnect(db,namespace,identity)
            raise OSError('synthetic_disconnect_failure')
        monkeypatch.setattr(store,'_disconnect',failed)
        with pytest.raises(RuntimeError,match='live_receipt_failed'):
            await store.disconnect(connection, reader=READER)
        monkeypatch.setattr(store,'_disconnect',disconnect)
        reset=await store.hello(client,identity(), reader=READER)
        assert reset.id


async def test_same_page_in_another_original_login_cannot_cover_earlier_tickets(tmp_path):
    first_reader = ReaderBinding('1'*64, lambda: True)
    second_reader = ReaderBinding('2'*64, lambda: True)
    async with graph(tmp_path) as (privacy,store,clock):
        client,first,second = identity(),identity(),identity()
        challenge = await store.hello(client,first,reader=first_reader)
        assert await store.reset_ack(first,challenge.id,challenge.nonce,reader=first_reader)
        source = memory_source(now=clock.now)
        ticket = await store.issue(first,identity(),source.expires_at,(source,),reader=first_reader)
        other = await store.hello(client,second,reader=second_reader)
        assert await store.reset_ack(second,other.id,other.nonce,reader=second_reader)
        assert await store.pending(source.conversation_id)==1
        await store.ack(second,((ticket.ticket,ticket.nonce),),reader=second_reader)
        assert await store.pending(source.conversation_id)==1
        with pytest.raises(RuntimeError,match='live_receipt_failed'):
            await store.reset(first,reader=second_reader)
        await store.disconnect(first,reader=second_reader)
        with privacy.ledger._transaction() as db:
            assert db.execute('SELECT COUNT(*) FROM live_connections WHERE connection_id=?',(first,)).fetchone()[0]==1
        await store.ack(first,((ticket.ticket,ticket.nonce),),reader=first_reader)
        assert await store.pending(source.conversation_id)==0


@pytest.mark.parametrize('operation', ['hello','issue','ack','reset_ack'])
async def test_original_reader_retired_at_native_precommit_cannot_change_obligations(tmp_path,monkeypatch,operation):
    active = [True]
    reader = ReaderBinding('3'*64, lambda:active[0])
    async with graph(tmp_path) as (privacy,store,clock):
        client,connection = identity(),identity()
        challenge = await store.hello(client,connection,reader=reader)
        assert await store.reset_ack(connection,challenge.id,challenge.nonce,reader=reader)
        source = memory_source(now=clock.now)
        ticket = await store.issue(connection,identity(),source.expires_at,(source,),reader=reader)
        challenge = await store.reset(connection,reader=reader)
        native = getattr(store,'_'+operation)

        def retiring(db,*args):
            result = native(db,*args)
            active[0] = False
            return result

        monkeypatch.setattr(store,'_'+operation,retiring)
        with pytest.raises(RuntimeError,match='live_receipt_failed'):
            if operation=='hello':
                await store.hello(identity(),identity(),reader=reader)
            elif operation=='issue':
                await store.issue(connection,identity(),source.expires_at,(source,),reader=reader)
            elif operation=='ack':
                await store.ack(connection,((ticket.ticket,ticket.nonce),),reader=reader)
            else:
                await store.reset_ack(connection,challenge.id,challenge.nonce,reader=reader)
        assert await store.pending(source.conversation_id)==1
        with privacy.ledger._transaction() as db:
            assert db.execute('SELECT COUNT(*) FROM live_connections').fetchone()[0]==1
            assert db.execute('SELECT COUNT(*) FROM live_deliveries').fetchone()[0]==1
        await store.disconnect(connection,reader=reader)
        assert await store.pending(source.conversation_id)==1


async def test_retirement_waiting_for_native_lock_launches_no_receipt_mutation(tmp_path,monkeypatch):
    active = [True]
    reader = ReaderBinding('4'*64, lambda:active[0])
    async with graph(tmp_path) as (privacy,store,clock):
        client,connection = identity(),identity()
        challenge = await store.hello(client,connection,reader=reader)
        assert await store.reset_ack(connection,challenge.id,challenge.nonce,reader=reader)
        source = memory_source(now=clock.now)
        entered,release,attempted = threading.Event(),threading.Event(),threading.Event()
        invoke = store._invoke
        calls = []

        def holding():
            with privacy.ledger._lock:
                entered.set()
                assert release.wait(5)

        def observed(*args):
            attempted.set()
            return invoke(*args)

        def forbidden(*args):
            calls.append(True)
            raise AssertionError('retired reader entered receipt mutation')

        monkeypatch.setattr(store,'_invoke',observed)
        monkeypatch.setattr(store,'_issue',forbidden)
        blocker = asyncio.create_task(asyncio.to_thread(holding))
        pending = None
        try:
            assert await asyncio.to_thread(entered.wait,3)
            pending = asyncio.create_task(store.issue(connection,identity(),source.expires_at,(source,),reader=reader))
            assert await asyncio.to_thread(attempted.wait,3)
            active[0] = False
        finally:
            release.set()
            await blocker
        with pytest.raises(RuntimeError,match='live_receipt_failed'):
            await pending
        assert calls==[]
        assert await store.pending(source.conversation_id)==0


async def test_legacy_unowned_tickets_survive_migration_and_cannot_be_claimed_by_page_id(tmp_path):
    async with graph(tmp_path) as (_,store,clock):
        client,connection = await connected(store)
        source = memory_source(now=clock.now)
        await store.issue(connection,identity(),source.expires_at,(source,),reader=READER)
    with closing(sqlite3.connect(tmp_path/'privacy.db')) as db,db:
        db.execute('ALTER TABLE live_deliveries DROP COLUMN owner_namespace')
        db.execute('DROP TABLE live_connections')
        db.execute('DROP TABLE live_resets')
        db.execute('CREATE TABLE live_connections(connection_id TEXT PRIMARY KEY,client_id TEXT NOT NULL UNIQUE,runtime_id TEXT NOT NULL)')
        db.execute('CREATE TABLE live_resets(id TEXT PRIMARY KEY,client_id TEXT NOT NULL,connection_id TEXT NOT NULL UNIQUE,nonce TEXT NOT NULL,cutoff INTEGER NOT NULL)')
    async with graph(tmp_path) as (privacy,store,_):
        assert await store.pending(source.conversation_id)==1
        next_connection = identity()
        challenge = await store.hello(client,next_connection,reader=READER)
        assert await store.reset_ack(next_connection,challenge.id,challenge.nonce,reader=READER)
        assert await store.pending(source.conversation_id)==1
        with privacy.ledger._transaction() as db:
            assert db.execute('SELECT owner_namespace FROM live_deliveries').fetchone()[0]=='legacy-unbound'


@pytest.mark.parametrize('operation',['ack','reset_ack'])
async def test_retirement_during_ledger_exit_reads_rolls_back_provisional_ack(tmp_path,monkeypatch,operation):
    active = [True]
    reader = ReaderBinding('5'*64,lambda:active[0])
    async with graph(tmp_path) as (privacy,store,clock):
        client,connection = identity(),identity()
        challenge = await store.hello(client,connection,reader=reader)
        assert await store.reset_ack(connection,challenge.id,challenge.nonce,reader=reader)
        source = memory_source(now=clock.now)
        ticket = await store.issue(connection,identity(),source.expires_at,(source,),reader=reader)
        challenge = await store.reset(connection,reader=reader)
        ledger = privacy.ledger
        original = ledger._read_fact_receipts
        counts = []

        def retiring(db):
            counts.append(db.execute('SELECT COUNT(*) FROM live_deliveries').fetchone()[0])
            active[0] = False
            return original(db)

        monkeypatch.setattr(ledger,'_read_fact_receipts',retiring)
        ledger._pending_fact_ids = frozenset({identity()})
        with pytest.raises(RuntimeError,match='live_receipt_failed'):
            if operation=='ack':
                await store.ack(connection,((ticket.ticket,ticket.nonce),),reader=reader)
            else:
                await store.reset_ack(connection,challenge.id,challenge.nonce,reader=reader)
        assert counts[0]==0 and counts[-1]==1
        assert await store.pending(source.conversation_id)==1
