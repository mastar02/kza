"""Real vector cleanup without semantic recall, across target receipts and restarts."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from unittest.mock import MagicMock

import pytest

from src.core.owned_work import WorkCloseTimeout
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant
from src.memory.fact_ledger import FactLedger, MemoryReviewer
from src.memory.index_deletions import IndexCleanupError
from src.memory.index_erasure import PolicyIndexEraser
from src.memory.privacy_services import PrivacyServices, build_privacy_services
from src.memory.services import build_memory_services
from tests.integration.test_transient_privacy import until
from tests.mocks.memory_graph import Clock, MemoryClient
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')


@asynccontextmanager
async def graph(root, *, clock=None, automatic_days=30):
    clock = clock or Clock()
    config = {'memory': {'enabled':False, 'privacy_db_path':str(root/'privacy.db'),
                         'chroma_path':str(root/'index'), 'automatic_retention_days':automatic_days}}
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        descriptor = {'path':str(root/'index'), 'collection':'user_memories_policy_v1'}
        target = await privacy.acall(privacy.ledger.erasures.register_target, ErasureParticipant.SEMANTIC_INDEX, 1, descriptor)
        client = MemoryClient(root/'index')
        policy = client.get_or_create_collection(descriptor['collection'])
        legacy = client.get_or_create_collection('user_memories')
        driver = PolicyIndexEraser(privacy, descriptor)
        owner.register(ErasureParticipant.SEMANTIC_INDEX, driver.clean, versions=frozenset({1}), begin_close=driver.begin_close)
        from types import SimpleNamespace
        g = SimpleNamespace(privacy=privacy, owner=owner, driver=driver, cleanup=driver.maintenance,
                            clock=clock, config=config, target=target, policy=policy, legacy=legacy, descriptor=descriptor)
        try:
            yield g
        finally:
            await asyncio.wait_for(asyncio.shield(owner.begin_close()), 3)


async def seed(g, content='synthetic policy fact', *, permanent=False):
    source = memory_source(now=g.clock())
    fact = await g.privacy.acall(g.privacy.ledger.store, content, 'fact', .8, (source,), proposed=permanent)
    if permanent:
        fact = await g.privacy.acall(g.privacy.ledger.confirm, fact.id, ACTOR)
    g.policy.upsert(ids=[fact.id], documents=[fact.content])
    await g.privacy.acall(g.privacy.ledger.acknowledge, fact.id, fact.revision)
    return fact


async def settle(cleanup):
    async with asyncio.timeout(3):
        while True:
            await cleanup.reconcile()
            tasks = [task for _, task in cleanup._active.values()]
            if not tasks:
                return
            await asyncio.gather(*(asyncio.shield(task) for task in tasks))


async def test_individual_deletion_cleans_real_policy_index_and_preserves_live_confirmed_and_legacy(tmp_path):
    async with graph(tmp_path) as g:
        erased = await seed(g)
        live = await seed(g, 'synthetic live fact')
        confirmed = await seed(g, 'synthetic confirmed fact', permanent=True)
        g.legacy.upsert(ids=['legacy'], documents=['synthetic unsourced legacy'])
        await g.privacy.acall(g.privacy.ledger.delete_fact, erased.id, ACTOR)
        await settle(g.cleanup)
        assert g.policy.get(ids=[erased.id])['ids']==[]
        assert set(g.policy.get()['ids'])=={live.id,confirmed.id}
        assert g.legacy.get()['documents']==['synthetic unsourced legacy']
        status = await g.privacy.acall(g.privacy.ledger.status)
        assert status['pending_index']==status['pending_index_cleanup']==0
        assert not (await g.privacy.acall(g.privacy.ledger.erasures.due))


async def test_automatic_expiry_removes_vector_without_conversation_tombstone_or_recall(tmp_path):
    async with graph(tmp_path) as g:
        erased = await seed(g)
        confirmed = await seed(g, 'synthetic permanent fact', permanent=True)
        g.clock.now += 30*86400
        await settle(g.cleanup)
        assert g.policy.get(ids=[erased.id])['ids']==[]
        assert g.policy.get(ids=[confirmed.id])['ids']==[confirmed.id]
        assert (await g.privacy.acall(g.privacy.ledger.status))['tombstones']==0
        with sqlite3.connect(g.privacy.ledger.path) as db:
            assert db.execute('SELECT content FROM facts WHERE id=?', (erased.id,)).fetchone()[0] is None


async def test_deletion_wakes_sleeping_cleanup_without_manual_reconcile(tmp_path):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        g.cleanup.start()
        await settle(g.cleanup)
        await g.privacy.acall(g.privacy.ledger.delete_fact, fact.id, ACTOR)
        await until(lambda:not g.policy.get(ids=[fact.id])['ids'])
        assert not g.config['memory']['enabled']


async def test_new_earlier_expiry_wakes_cleanup_without_enabling_recall(tmp_path):
    async with graph(tmp_path, clock=time.time, automatic_days=.2/86400) as g:
        g.cleanup.start()
        await settle(g.cleanup)
        fact = await seed(g)
        assert g.policy.get(ids=[fact.id])['ids']==[fact.id]
        await until(lambda:not g.policy.get(ids=[fact.id])['ids'])
        assert await g.privacy.acall(g.privacy.ledger.get, fact.id) is None


async def test_all_historical_targets_get_independent_receipts(tmp_path):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        other = MemoryClient(tmp_path/'other-index').get_or_create_collection(g.descriptor['collection'])
        other.upsert(ids=[fact.id], documents=[fact.content])
        await g.privacy.acall(g.privacy.ledger.erasures.register_target, ErasureParticipant.SEMANTIC_INDEX, 1,
                             {'path':str(tmp_path/'other-index'),'collection':g.descriptor['collection']})
        await g.privacy.acall(g.privacy.ledger.delete_fact, fact.id, ACTOR)
        await settle(g.cleanup)
        assert g.policy.count()==other.count()==0
        targets = await g.privacy.acall(g.privacy.ledger.index_deletions.targets)
        assert len(targets)==2 and all(target['pending']==0 for target in targets)


async def test_later_registered_target_backfills_after_global_job_was_acknowledged(tmp_path):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_fact, fact.id, ACTOR)
        await settle(g.cleanup)
        assert (await g.privacy.acall(g.privacy.ledger.status))['pending_index']==0
        restored = MemoryClient(tmp_path/'discovered-index').get_or_create_collection(g.descriptor['collection'])
        restored.upsert(ids=[fact.id], documents=[fact.content])
        await g.privacy.acall(g.privacy.ledger.erasures.register_target, ErasureParticipant.SEMANTIC_INDEX, 1,
                             {'path':str(tmp_path/'discovered-index'),'collection':g.descriptor['collection']})
        assert (await g.privacy.acall(g.privacy.ledger.status))['pending_index_cleanup']==1
        await settle(g.cleanup)
        assert restored.count()==0


async def test_failed_target_remains_durable_while_another_target_cleans(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        other_descriptor = {'path':str(tmp_path/'other-index'),'collection':g.descriptor['collection']}
        other = MemoryClient(tmp_path/'other-index').get_or_create_collection(g.descriptor['collection'])
        other.upsert(ids=[fact.id], documents=[fact.content])
        await g.privacy.acall(g.privacy.ledger.erasures.register_target, ErasureParticipant.SEMANTIC_INDEX, 1, other_descriptor)
        delete = g.driver._delete_rows
        def failed(descriptor, ids):
            if descriptor==g.descriptor:
                raise OSError('synthetic private error text')
            delete(descriptor, ids)
        monkeypatch.setattr(g.driver,'_delete_rows',failed)
        await g.privacy.acall(g.privacy.ledger.delete_fact, fact.id, ACTOR)
        await settle(g.cleanup)
        assert g.policy.count()==1 and other.count()==0
        targets = await g.privacy.acall(g.privacy.ledger.index_deletions.targets)
        assert sum(target['pending'] for target in targets)==1
        assert any(target['error']==IndexCleanupError.FAILED for target in targets)
        assert 'private error text' not in repr(g.cleanup.status())
        monkeypatch.setattr(g.driver,'_delete_rows',delete)
        g.clock.now += 2
        await settle(g.cleanup)
        assert g.policy.count()==0


async def test_ack_failure_retries_receipt_without_repeating_native_delete(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_fact, fact.id, ACTOR)
        finish = g.privacy.ledger.index_deletions.finish
        delete = g.driver._delete_rows
        calls = []
        def observed(*args):
            calls.append(args)
            return delete(*args)
        def failed(*args):
            raise OSError('synthetic unavailable receipt store')
        monkeypatch.setattr(g.driver,'_delete_rows',observed)
        monkeypatch.setattr(g.privacy.ledger.index_deletions,'finish',failed)
        await g.cleanup.reconcile()
        await until(lambda:all(task.done() for _,task in g.cleanup._active.values()))
        with pytest.raises(OSError):
            await g.cleanup.reconcile()
        assert len(calls)==1 and g.policy.count()==0 and len(g.cleanup._active)==1
        assert next(iter(g.cleanup._active.values()))[1].exception() is None
        monkeypatch.setattr(g.privacy.ledger.index_deletions,'finish',finish)
        await settle(g.cleanup)
        assert len(calls)==1


async def test_fact_and_destination_cleanup_roll_back_together(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        g.cleanup.start()
        await settle(g.cleanup)
        enqueue = g.privacy.ledger.index_deletions.enqueue
        def failed(*args):
            enqueue(*args)
            raise OSError('synthetic transaction failure')
        monkeypatch.setattr(g.privacy.ledger.index_deletions,'enqueue',failed)
        with pytest.raises(OSError):
            await g.privacy.acall(g.privacy.ledger.delete_fact, fact.id, ACTOR)
        assert await g.privacy.acall(g.privacy.ledger.get, fact.id) is not None
        assert await g.privacy.acall(g.privacy.ledger.index_deletions.targets)==[]
        await settle(g.cleanup)
        assert g.policy.count()==1


async def test_leased_restart_rechecks_completed_receipts_against_restored_index(tmp_path):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_fact, fact.id, ACTOR)
        await settle(g.cleanup)
        assert g.policy.count()==0
    # Restore only the old physical projection while retaining durable deletion.
    restored = MemoryClient(tmp_path/'index').get_or_create_collection(g.descriptor['collection'])
    restored.upsert(ids=[fact.id], documents=[fact.content])
    async with graph(tmp_path) as fresh:
        assert (await fresh.privacy.acall(fresh.privacy.ledger.status))['pending_index_cleanup']==1
        await settle(fresh.cleanup)
        assert restored.count()==0


async def test_repeated_initialize_does_not_recover_live_claim_and_old_token_cannot_ack_restart(tmp_path):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_fact, fact.id, ACTOR)
        old = await g.privacy.acall(g.privacy.ledger.index_deletions.claim,g.target)
        await g.privacy.acall(g.privacy.initialize)
        assert await g.privacy.acall(g.privacy.ledger.index_deletions.claim,g.target) is None
        standalone = FactLedger(g.privacy.ledger.path,clock=g.clock)
        try:
            await asyncio.to_thread(standalone.initialize)
            assert await g.privacy.acall(g.privacy.ledger.index_deletions.claim,g.target) is None
        finally:
            await asyncio.to_thread(standalone.close)
    async with graph(tmp_path) as g:
        current = await g.privacy.acall(g.privacy.ledger.index_deletions.claim,g.target)
        assert current.items[0].token != old.items[0].token
        await g.privacy.acall(g.privacy.ledger.index_deletions.finish,old)
        targets = await g.privacy.acall(g.privacy.ledger.index_deletions.targets)
        assert targets[0]['running']==1 and g.policy.count()==1
        await g.driver.clean_deletions(current)
        assert g.policy.count()==0


async def test_stale_revision_receipt_cannot_clear_current_claim(tmp_path):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        batch = await g.privacy.acall(g.privacy.ledger.index_deletions.claim,g.target)
        stale = replace(batch,items=(replace(batch.items[0],revision=batch.items[0].revision-1),))
        await g.privacy.acall(g.privacy.ledger.index_deletions.finish,stale)
        assert (await g.privacy.acall(g.privacy.ledger.index_deletions.targets))[0]['running']==1
        await g.driver.clean_deletions(batch)


@pytest.mark.parametrize('change',['version','descriptor'])
async def test_claim_binds_target_version_and_descriptor_before_native_deletion(tmp_path, change):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        batch = await g.privacy.acall(g.privacy.ledger.index_deletions.claim,g.target)
        forged = replace(batch,version=2) if change=='version' else replace(batch,descriptor={**batch.descriptor,'path':str(tmp_path/'unregistered')})
        with pytest.raises(ValueError,match='invalid_index_cleanup_claim'):
            await g.driver.clean_deletions(forged)
        with pytest.raises(ValueError,match='invalid_index_cleanup_claim'):
            await g.privacy.acall(g.privacy.ledger.index_deletions.finish,forged)
        assert g.policy.count()==1
        assert (await g.privacy.acall(g.privacy.ledger.index_deletions.targets))[0]['running']==1
        await g.driver.clean_deletions(batch)


async def test_native_cleanup_claim_survives_observer_cancel_and_close_timeout(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        entered,release = threading.Event(),threading.Event()
        delete = g.driver._delete_rows
        def blocked(*args):
            entered.set()
            release.wait(3)
            delete(*args)
        monkeypatch.setattr(g.driver,'_delete_rows',blocked)
        await g.cleanup.reconcile()
        competitor = PrivacyServices(g.privacy.ledger.path)
        try:
            await until(entered.is_set)
            observing = asyncio.create_task(g.cleanup.reconcile())
            await asyncio.sleep(0)
            observing.cancel()
            await asyncio.gather(observing,return_exceptions=True)
            assert len(g.cleanup._active)==1
            g.privacy._close_timeout_s=.01
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
            with pytest.raises(RuntimeError,match='writer_already_active'):
                await competitor.acall(competitor.initialize)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.shield(g.owner.begin_close()),3)
            await asyncio.wait_for(asyncio.shield(g.privacy.begin_close()),3)
            await competitor.aclose()
        assert g.policy.count()==0


async def test_missing_policy_collection_is_not_created_and_legacy_is_untouched(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        g.legacy.upsert(ids=['legacy'],documents=['synthetic unsourced legacy'])
        g.policy.collection._client.delete_collection(name=g.descriptor['collection'])
        from chromadb.api.client import Client
        def forbidden(*args,**kwargs):
            raise AssertionError('cleanup created a collection')
        monkeypatch.setattr(Client,'get_or_create_collection',forbidden)
        await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        await settle(g.cleanup)
        assert g.legacy.get()['documents']==['synthetic unsourced legacy']
        assert (await g.privacy.acall(g.privacy.ledger.status))['pending_index_cleanup']==0


async def test_unsupported_target_remains_pending_without_erasing_its_data(tmp_path):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        unknown = MemoryClient(tmp_path/'future-index').get_or_create_collection(g.descriptor['collection'])
        unknown.upsert(ids=[fact.id],documents=[fact.content])
        await g.privacy.acall(g.privacy.ledger.erasures.register_target,ErasureParticipant.SEMANTIC_INDEX,2,
                             {'path':str(tmp_path/'future-index'),'collection':g.descriptor['collection']})
        await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        await settle(g.cleanup)
        assert g.policy.count()==0 and unknown.count()==1
        targets = await g.privacy.acall(g.privacy.ledger.index_deletions.targets)
        assert any(row['error']==IndexCleanupError.UNSUPPORTED for row in targets)


async def test_cleanup_does_not_reinitialize_persisted_embedding_function(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        from tests.mocks.memory_graph import SyntheticEmbedding
        def forbidden(*args, **kwargs):
            raise AssertionError('cleanup loaded an embedding function')
        monkeypatch.setattr(SyntheticEmbedding,'build_from_config',forbidden)
        await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        await settle(g.cleanup)
        assert g.policy.count()==0


async def test_blocked_historical_target_does_not_hold_current_target_barrier(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        other_descriptor = {'path':str(tmp_path/'other-index'),'collection':g.descriptor['collection']}
        other = MemoryClient(tmp_path/'other-index').get_or_create_collection(g.descriptor['collection'])
        other.upsert(ids=[fact.id],documents=[fact.content])
        await g.privacy.acall(g.privacy.ledger.erasures.register_target,ErasureParticipant.SEMANTIC_INDEX,1,other_descriptor)
        await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        entered,release = threading.Event(),threading.Event()
        delete = g.driver._delete_rows
        def blocked(descriptor,ids):
            if descriptor==other_descriptor:
                entered.set()
                release.wait(3)
            delete(descriptor,ids)
        monkeypatch.setattr(g.driver,'_delete_rows',blocked)
        g.cleanup.start()
        try:
            await until(entered.is_set)
            await until(lambda:g.policy.count()==0)
            assert other.count()==1
            assert (await g.privacy.acall(g.privacy.ledger.status))['pending_index_cleanup']==1
        finally:
            release.set()
        await settle(g.cleanup)
        assert other.count()==0


async def test_old_upsert_finishes_before_individual_cleanup_receipt(tmp_path):
    clock = Clock()
    config = {'memory': {'privacy_db_path':str(tmp_path/'privacy.db'),'chroma_path':str(tmp_path/'index')}}
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config,shutdown=stack,clock=clock)
        services = await build_memory_services(config,shutdown=stack,privacy=privacy,chroma_client=MemoryClient(tmp_path/'index'))
        services.maintenance.cancel()
        await asyncio.gather(services.maintenance,return_exceptions=True)
        memory = services.manager.long_term
        descriptor = memory.index_descriptor
        driver = PolicyIndexEraser(privacy,descriptor,memory=memory)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        owner.register(ErasureParticipant.SEMANTIC_INDEX,driver.clean,versions=frozenset({1}),begin_close=driver.begin_close)
        source = memory_source(now=clock())
        fact = await privacy.acall(privacy.ledger.store,'synthetic old upsert','fact',.8,(source,))
        entered,release = threading.Event(),threading.Event()
        def blocked():
            entered.set()
            release.wait(3)
        memory._collection.before_upsert = blocked
        writing = asyncio.create_task(memory.areconcile())
        try:
            await until(entered.is_set)
            await privacy.acall(privacy.ledger.delete_fact,fact.id,ACTOR)
            await driver.maintenance.reconcile()
            assert driver.maintenance._active
            assert (await privacy.acall(privacy.ledger.index_deletions.targets))[0]['running']==1
        finally:
            release.set()
        await writing
        await settle(driver.maintenance)
        assert memory._collection.get(ids=[fact.id])['ids']==[]
        assert (await privacy.acall(privacy.ledger.status))['pending_index_cleanup']==0


async def test_cleanup_removes_quarantined_policy_projection_but_keeps_quarantined_content(tmp_path):
    async with graph(tmp_path) as g:
        fact = await seed(g)
    with sqlite3.connect(tmp_path/'privacy.db') as database:
        database.execute('ALTER TABLE fact_sources DROP COLUMN dependencies')
    async with graph(tmp_path) as g:
        await settle(g.cleanup)
        assert g.policy.count()==0
        with sqlite3.connect(g.privacy.ledger.path) as database:
            assert database.execute('SELECT state,content FROM facts WHERE id=?',(fact.id,)).fetchone()==('quarantined',fact.content)
        assert await g.privacy.acall(g.privacy.ledger.get,fact.id) is None


async def test_claim_handoff_is_retained_when_scheduler_observer_is_cancelled(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        entered,release = threading.Event(),threading.Event()
        claim = g.privacy.ledger.index_deletions.claim
        def blocked(*args):
            batch = claim(*args)
            entered.set()
            release.wait(3)
            return batch
        monkeypatch.setattr(g.privacy.ledger.index_deletions,'claim',blocked)
        observing = asyncio.create_task(g.cleanup.reconcile())
        try:
            await until(entered.is_set)
            observing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observing
            assert not g.cleanup._cycle.done()
        finally:
            release.set()
        await settle(g.cleanup)
        assert g.policy.count()==0


async def test_index_cleanup_is_bounded_to_32_ids_per_native_batch(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        facts = [await seed(g,f'synthetic fact {i}') for i in range(65)]
        for fact in facts:
            await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        sizes = []
        delete = g.driver._delete_rows
        def observed(descriptor,ids):
            sizes.append(len(ids))
            return delete(descriptor,ids)
        monkeypatch.setattr(g.driver,'_delete_rows',observed)
        await settle(g.cleanup)
        assert sizes==[32,32,1] and g.policy.count()==0


async def test_saturated_default_executor_does_not_starve_cleanup(tmp_path):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        loop = asyncio.get_running_loop()
        previous = loop._default_executor
        pool = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(pool)
        entered,release = threading.Event(),threading.Event()
        def blocked():
            entered.set()
            release.wait(3)
        occupying = asyncio.create_task(asyncio.to_thread(blocked))
        try:
            await until(entered.is_set)
            await asyncio.wait_for(settle(g.cleanup),1)
            assert g.policy.count()==0 and not occupying.done()
        finally:
            release.set()
            await occupying
            loop._default_executor=previous
            pool.shutdown(wait=True)


async def test_runtime_cleanup_of_absent_index_never_imports_models_or_creates_directory(tmp_path):
    script = textwrap.dedent('''
        import asyncio, importlib.abc, sys
        from contextlib import AsyncExitStack
        from pathlib import Path
        from uuid import uuid4
        class NoModels(importlib.abc.MetaPathFinder):
            def find_spec(self,fullname,path=None,target=None):
                if fullname.split('.')[0] in {'torch','chromadb','sentence_transformers'}:
                    raise AssertionError('cleanup imported a model or vector runtime: '+fullname)
        sys.meta_path.insert(0,NoModels())
        from src.memory.privacy_services import build_privacy_services
        from src.memory.index_erasure import build_erasure_services
        from src.memory.fact_ledger import MemorySource, MemoryOrigin, MemoryReviewer
        from src.memory.erasure_journal import ErasureParticipant
        async def check():
            root = Path(sys.argv[1])
            config = {'memory': {'enabled':False,'privacy_db_path':str(root/'privacy.db'),'chroma_path':str(root/'absent-index')}}
            async with AsyncExitStack() as stack:
                privacy = await build_privacy_services(config,shutdown=stack,clock=lambda:2000000000.0)
                source = MemorySource(str(uuid4()),str(uuid4()),2000000000.0,2000000010.0,MemoryOrigin.VOICE)
                fact = await privacy.acall(privacy.ledger.store,'synthetic fact','fact',.8,(source,))
                await privacy.acall(privacy.ledger.delete_fact,fact.id,MemoryReviewer('synthetic-admin'))
                owner = await build_erasure_services(config,privacy=privacy,shutdown=stack)
                cleanup = owner._drivers[ErasureParticipant.SEMANTIC_INDEX].callback.__self__.maintenance
                for _ in range(3):
                    await cleanup.reconcile()
                    await asyncio.gather(*(asyncio.shield(task) for _,task in cleanup._active.values()))
                assert (await privacy.acall(privacy.ledger.status))['pending_index_cleanup']==0
                assert not (root/'absent-index').exists()
        asyncio.run(check())
    ''')
    result = await asyncio.to_thread(subprocess.run,[sys.executable,'-c',script,str(tmp_path)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr


async def test_writer_uses_registered_canonical_location_if_configured_symlink_changes(tmp_path, monkeypatch):
    import chromadb
    import src.memory.services as module
    first,second = tmp_path/'first',tmp_path/'second'
    first.mkdir()
    second.mkdir()
    alias = tmp_path/'alias'
    alias.symlink_to(first,target_is_directory=True)
    config = {'memory':{'privacy_db_path':str(tmp_path/'privacy.db'),'chroma_path':str(alias)}}
    register = module.register_policy_index_target
    async def changed(*args):
        descriptor = await register(*args)
        alias.unlink()
        alias.symlink_to(second,target_is_directory=True)
        return descriptor
    client = MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(module,'register_policy_index_target',changed)
    monkeypatch.setattr(chromadb,'PersistentClient',factory)
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config,shutdown=stack)
        services = await build_memory_services(config,shutdown=stack,privacy=privacy)
        assert factory.call_args.kwargs['path']==str(first)
        assert services.manager.long_term.index_descriptor['path']==str(first)
        assert Path(services.manager.long_term.index_descriptor['path']).is_absolute()
        driver = PolicyIndexEraser(privacy,{'path':str(second),'collection':'user_memories_policy_v1'},memory=services.manager.long_term)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        owner.register(ErasureParticipant.SEMANTIC_INDEX,driver.clean,versions=frozenset({1}),begin_close=driver.begin_close)
        assert driver._lock_for(services.manager.long_term.index_descriptor) is services.manager.long_term._index_lock


async def test_unknown_fact_authority_keeps_cleanup_pending_instead_of_claiming_erasure(tmp_path):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        def unknown():
            with g.privacy.ledger._transaction() as database:
                database.execute("UPDATE facts SET state='future_unknown_state' WHERE id=?",(fact.id,))
        await g.privacy.acall(unknown)
        await settle(g.cleanup)
        assert g.policy.count()==1
        target = (await g.privacy.acall(g.privacy.ledger.index_deletions.targets))[0]
        assert target['pending']==1 and target['error']==IndexCleanupError.FAILED


async def test_conversation_receipt_cannot_hide_unknown_index_cleanup_authority(tmp_path):
    async with graph(tmp_path) as g:
        fact = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_fact,fact.id,ACTOR)
        def unknown():
            with g.privacy.ledger._transaction() as database:
                database.execute("UPDATE facts SET state='future_unknown_state' WHERE id=?",(fact.id,))
        await g.privacy.acall(unknown)
        receipt = await g.owner.request(fact.sources[0].conversation_id,ACTOR)
        await g.owner.reconcile()
        await asyncio.gather(*(active.task for active in g.owner._active.values()))
        await g.owner.reconcile()
        snapshot = await g.privacy.acall(g.privacy.ledger.erasures.snapshot,receipt['id'])
        step = next(step for step in snapshot['steps'] if step['participant']=='semantic_index')
        assert step['state']=='pending' and step['pending_count']==1
        assert g.policy.count()==1
