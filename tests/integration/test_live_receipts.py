"""Durable live obligations are retired by exact client capabilities only."""

from contextlib import AsyncExitStack, asynccontextmanager
from uuid import uuid4

import pytest

from src.dashboard.live_receipts import LiveReceiptStore
from src.memory.privacy_services import PrivacyServices
from tests.mocks.memory_sources import memory_source
from tests.mocks.ambient_graph import Clock

pytestmark=pytest.mark.asyncio


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
    reset=await store.hello(client,connection)
    assert await store.reset_ack(connection,reset.id,reset.nonce)
    return client,connection


async def test_receipt_is_content_free_and_wrong_nonce_or_other_connection_cannot_retire(tmp_path):
    async with graph(tmp_path) as (privacy,store,clock):
        _,first=await connected(store)
        _,second=await connected(store)
        source=memory_source(now=clock.now)
        ticket=await store.issue(first,identity(),source.expires_at,(source,))
        assert await store.pending(source.conversation_id)==1
        await store.ack(second,((ticket.ticket,ticket.nonce),))
        await store.ack(first,((ticket.ticket,identity()),))
        assert await store.pending(source.conversation_id)==1
        await store.ack(first,((ticket.ticket,ticket.nonce),))
        assert await store.pending(source.conversation_id)==0
        assert 'Synthetic' not in repr(ticket)
        with privacy.ledger._transaction() as db:
            assert db.execute('SELECT COUNT(*) FROM live_delivery_sources').fetchone()[0]==0


async def test_disconnect_expiry_and_restart_never_infer_browser_erasure(tmp_path):
    async with graph(tmp_path) as (_,store,clock):
        client,connection=await connected(store)
        source=memory_source(now=clock.now,ttl=1)
        ticket=await store.issue(connection,identity(),source.expires_at,(source,))
        await store.disconnect(connection)
        clock.now+=2
        assert await store.pending(source.conversation_id)==1
    async with graph(tmp_path) as (_,store,_):
        assert await store.pending(source.conversation_id)==1
        new_connection=identity()
        reset=await store.hello(client,new_connection)
        await store.ack(new_connection,((ticket.ticket,ticket.nonce),))
        assert await store.pending(source.conversation_id)==1
        assert not await store.reset_ack(new_connection,reset.id,identity())
        assert await store.pending(source.conversation_id)==1
        assert await store.reset_ack(new_connection,reset.id,reset.nonce)
        assert await store.pending(source.conversation_id)==0


async def test_reset_ack_covers_only_its_client_and_captured_cutoff(tmp_path):
    async with graph(tmp_path) as (_,store,clock):
        _,connection=await connected(store)
        _,other=await connected(store)
        source,later,unrelated=(memory_source(now=clock.now) for _ in range(3))
        await store.issue(connection,identity(),source.expires_at,(source,))
        reset=await store.reset(connection)
        await store.issue(connection,identity(),later.expires_at,(later,))
        await store.issue(other,identity(),unrelated.expires_at,(unrelated,))
        assert not await store.reset_ack(other,reset.id,reset.nonce)
        assert await store.reset_ack(connection,reset.id,reset.nonce)
        assert await store.pending(source.conversation_id)==0
        assert await store.pending(later.conversation_id)==1
        assert await store.pending(unrelated.conversation_id)==1


async def test_duplicate_page_cannot_evict_live_connection_or_take_its_reset(tmp_path):
    async with graph(tmp_path) as (_,store,_):
        client,connection=await connected(store)
        with pytest.raises(RuntimeError,match='live_receipt_failed'):
            await store.hello(client,identity())
        reset=await store.reset(connection)
        assert await store.reset_ack(connection,reset.id,reset.nonce)


async def test_global_and_client_caps_fail_closed_without_deleting_unacked_rows(tmp_path):
    async with graph(tmp_path,capacity=3,client_capacity=2) as (_,store,clock):
        _,connection=await connected(store)
        _,other=await connected(store)
        source=memory_source(now=clock.now)
        assert await store.issue(connection,identity(),source.expires_at,(source,))
        assert await store.issue(connection,identity(),source.expires_at,(source,))
        assert await store.issue(connection,identity(),source.expires_at,(source,)) is None
        assert await store.issue(other,identity(),source.expires_at,(source,))
        assert await store.issue(other,identity(),source.expires_at,(source,)) is None
        assert await store.pending(source.conversation_id)==3


async def test_concurrent_page_handshakes_preserve_all_connection_leases(tmp_path):
    import asyncio
    async with graph(tmp_path) as (_,store,_):
        pages=await asyncio.gather(*(connected(store) for _ in range(16)))
        for _,connection in pages:
            reset=await store.reset(connection)
            assert await store.reset_ack(connection,reset.id,reset.nonce)


async def test_second_receipt_owner_cannot_recover_over_the_live_writer(tmp_path):
    async with graph(tmp_path) as (privacy,store,_):
        _,connection=await connected(store)
        with pytest.raises(RuntimeError,match='live_receipt_owner_active'):
            LiveReceiptStore(privacy.ledger)
        reset=await store.reset(connection)
        assert await store.reset_ack(connection,reset.id,reset.nonce)


async def test_failed_disconnect_metadata_retries_without_evicting_an_active_peer(tmp_path,monkeypatch):
    async with graph(tmp_path) as (_,store,_):
        client,connection=await connected(store)
        disconnect=store._disconnect
        def failed(db,identity):
            disconnect(db,identity)
            raise OSError('synthetic_disconnect_failure')
        monkeypatch.setattr(store,'_disconnect',failed)
        with pytest.raises(RuntimeError,match='live_receipt_failed'):
            await store.disconnect(connection)
        monkeypatch.setattr(store,'_disconnect',disconnect)
        reset=await store.hello(client,identity())
        assert reset.id
