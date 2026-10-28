"""Atomic erasure intent, durable claims and real policy-only index cleanup."""

import asyncio
from collections import Counter
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sqlite3
import threading
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.core.owned_work import WorkCloseTimeout
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureError, ErasureParticipant as Participant, ErasureResult, ErasureState
from src.memory.fact_ledger import FactLedger, MemoryReviewer
from src.memory.index_erasure import PolicyIndexEraser, build_erasure_services
from src.memory.privacy_services import PrivacyServices, build_privacy_services
from src.memory.services import build_memory_services
from tests.integration.test_transient_privacy import until
from tests.mocks.memory_graph import Clock, MemoryClient
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')


@asynccontextmanager
async def graph(tmp_path, *, close_timeout_s=1):
    clock = Clock()
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock, close_timeout_s=close_timeout_s)
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        owner = ErasureCoordinator(privacy, close_timeout_s=close_timeout_s)
        stack.push_async_callback(owner.aclose)
        yield privacy, owner, clock, stack


async def target(privacy, participant=Participant.CONTEXTS, *, version=1, descriptor=None):
    return await privacy.acall(privacy.ledger.erasures.register_target, participant, version,
                              descriptor or {'synthetic': 'no private content'})


async def snapshot(privacy, receipt):
    return await privacy.acall(privacy.ledger.erasures.snapshot, receipt['id'])


async def flush(owner):
    """Finish immediately completing synthetic callbacks, without advancing retry time."""
    await owner.reconcile()
    await asyncio.gather(*(a.task for a in owner._active.values()))
    await owner.reconcile()


async def test_revocation_scrub_and_journal_are_atomic_and_idempotent(tmp_path, monkeypatch):
    async with graph(tmp_path) as (privacy, owner, clock, _):
        ledger = privacy.ledger
        source = memory_source(now=clock.now)
        fact = await privacy.acall(ledger.store, 'synthetic private fact', 'fact', .8, (source,))
        original = ledger.erasures.ensure_operation
        def fail(db, *args):
            original(db, *args)
            raise OSError('synthetic failure after inserting intent')
        monkeypatch.setattr(ledger.erasures, 'ensure_operation', fail)
        with pytest.raises(OSError):
            await owner.request(source.conversation_id, ACTOR)
        assert privacy.sources_available((source,)) and await privacy.acall(ledger.get, fact.id)
        with sqlite3.connect(ledger.path) as db:
            assert db.execute('SELECT COUNT(*) FROM conversation_erasures').fetchone()[0] == 0
            assert db.execute('SELECT COUNT(*) FROM conversation_tombstones').fetchone()[0] == 0
        monkeypatch.setattr(ledger.erasures, 'ensure_operation', original)
        receipt = await owner.request(source.conversation_id, ACTOR)
        assert not privacy.sources_available((source,))
        assert await privacy.acall(ledger.get, fact.id) is None
        clock.now += 10
        again = await owner.request(source.conversation_id, MemoryReviewer('second-admin'))
        assert again == receipt
        assert receipt['requested_by'] == ACTOR.subject and receipt['requested_at'] == clock.now-10
        assert receipt['revoked'] and receipt['state'] == 'blocked'
        assert len(receipt['steps']) == len(Participant)
        assert all(s['last_error'] == 'participant_unregistered' for s in receipt['steps'])
        with sqlite3.connect(ledger.path) as db:
            assert db.execute('SELECT content FROM facts WHERE id=?', (fact.id,)).fetchone()[0] is None


async def test_failed_scrub_rolls_back_intent_and_keeps_sources_live(tmp_path, monkeypatch):
    async with graph(tmp_path) as (privacy, owner, clock, _):
        source = memory_source(now=clock.now)
        await privacy.acall(privacy.ledger.store, 'synthetic fact', 'fact', .8, (source,))
        def fail(*args):
            raise OSError('synthetic scrub failure')
        monkeypatch.setattr(privacy.ledger, '_delete', fail)
        with pytest.raises(OSError):
            await owner.request(source.conversation_id, ACTOR)
        assert privacy.sources_available((source,))
        with sqlite3.connect(privacy.ledger.path) as db:
            assert db.execute('SELECT COUNT(*) FROM conversation_erasures').fetchone()[0] == 0


async def test_unauthenticated_request_never_writes_tombstone_or_intent(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, _):
        with pytest.raises(PermissionError):
            await owner.request(str(uuid4()), None)
        assert (await privacy.acall(privacy.ledger.status))['tombstones'] == 0


async def test_missing_driver_never_completes_and_receipts_exclude_storage_details(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, _):
        await target(privacy, descriptor={'path': '/synthetic/private-location'})
        receipt = await owner.request(str(uuid4()), ACTOR)
        await owner.reconcile()
        status = await snapshot(privacy, receipt)
        context = next(s for s in status['steps'] if s['participant'] == 'contexts')
        assert context['state'] == 'pending' and context['last_error'] == 'participant_unavailable'
        assert status['state'] == 'blocked'
        encoded = json.dumps(status)
        for private in ('private-location', 'descriptor', 'claim_token', 'cursor', 'target_id'):
            assert private not in encoded


async def test_complete_requires_every_participant_and_new_targets_reopen_old_operations(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, _):
        seen = []
        async def clean(step):
            seen.append((step.participant, step.target_id))
            return ErasureResult(True)
        for participant in Participant:
            owner.register(participant, clean, versions=frozenset({1}))
            await target(privacy, participant)
        receipt = await owner.request(str(uuid4()), ACTOR)
        for _ in range(4):
            await flush(owner)
        status = await snapshot(privacy, receipt)
        assert status['state'] == 'complete' and len(seen) == len(Participant)
        await target(privacy, Participant.CONTEXTS, descriptor={'path': '/synthetic/changed-location'})
        assert (await snapshot(privacy, receipt))['state'] == 'pending'
        await flush(owner)
        assert (await snapshot(privacy, receipt))['state'] == 'complete'
        assert len(seen) == len(Participant)+1
        # Re-registering the exact target does not revive completed work.
        await target(privacy, Participant.CONTEXTS, descriptor={'path': '/synthetic/changed-location'})
        await flush(owner)
        assert len(seen) == len(Participant)+1


async def test_unsupported_historical_version_stays_pending(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, _):
        called = []
        async def clean(step):
            called.append(step)
            return ErasureResult(True)
        owner.register(Participant.CONTEXTS, clean, versions=frozenset({2}))
        await target(privacy, version=1)
        await target(privacy, version=2)
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner)
        steps = [s for s in (await snapshot(privacy, receipt))['steps'] if s['participant'] == 'contexts']
        assert {s['state'] for s in steps} == {'complete', 'pending'}
        assert [step.version for step in called] == [2]


@pytest.mark.parametrize('invalid', [False, True])
async def test_failure_or_invalid_result_is_content_free_and_retries_after_backoff(tmp_path, invalid):
    async with graph(tmp_path) as (privacy, owner, clock, _):
        calls = []
        async def clean(step):
            calls.append(step)
            if len(calls) == 1:
                if invalid:
                    return {'done': True, 'secret': 'synthetic transcript'}
                raise OSError('synthetic transcript must never enter a receipt')
            return ErasureResult(True)
        await target(privacy)
        owner.register(Participant.CONTEXTS, clean, versions=frozenset({1}))
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner) if invalid else await owner.reconcile()
        if not invalid:
            await until(lambda: all(a.task.done() for a in owner._active.values()))
            await owner.reconcile()
        status = await snapshot(privacy, receipt)
        step = next(s for s in status['steps'] if s['participant'] == 'contexts')
        assert step['last_error'] == ('invalid_cleanup_result' if invalid else 'cleanup_failed')
        assert 'synthetic transcript' not in json.dumps(status)
        await owner.reconcile()
        assert len(calls) == 1
        clock.now += 2
        await flush(owner)
        assert len(calls) == 2
        assert next(s for s in (await snapshot(privacy, receipt))['steps'] if s['participant'] == 'contexts')['state'] == 'complete'


async def test_ack_failure_keeps_completed_callback_without_duplicate_cleanup(tmp_path, monkeypatch):
    async with graph(tmp_path) as (privacy, owner, _, _):
        calls = []
        async def clean(step):
            calls.append(step)
            return ErasureResult(True)
        await target(privacy)
        owner.register(Participant.CONTEXTS, clean, versions=frozenset({1}))
        await owner.request(str(uuid4()), ACTOR)
        await owner.reconcile()
        await until(lambda: all(a.task.done() for a in owner._active.values()))
        finish = privacy.ledger.erasures.finish
        def fail(*args):
            raise OSError('synthetic unavailable receipt store')
        monkeypatch.setattr(privacy.ledger.erasures, 'finish', fail)
        with pytest.raises(OSError):
            await owner.reconcile()
        assert len(calls) == 1 and owner.resource_status()['active'] == 1
        monkeypatch.setattr(privacy.ledger.erasures, 'finish', finish)
        await owner.reconcile()
        assert len(calls) == 1 and owner.resource_status()['active'] == 0


async def test_cancel_during_claim_handoff_and_repeated_observers_cannot_orphan_or_duplicate(tmp_path, monkeypatch):
    async with graph(tmp_path) as (privacy, owner, _, _):
        entered, release = threading.Event(), threading.Event()
        cleanup_entered, cleanup_release = asyncio.Event(), asyncio.Event()
        claim = privacy.ledger.erasures.claim
        calls = []
        def blocked(row):
            step = claim(row)
            entered.set()
            release.wait(3)
            return step
        async def clean(step):
            calls.append(step)
            cleanup_entered.set()
            await cleanup_release.wait()
            return ErasureResult(True)
        await target(privacy)
        owner.register(Participant.CONTEXTS, clean, versions=frozenset({1}))
        receipt = await owner.request(str(uuid4()), ACTOR)
        monkeypatch.setattr(privacy.ledger.erasures, 'claim', blocked)
        observing = asyncio.create_task(owner.reconcile())
        try:
            await until(entered.is_set)
            observing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observing
            assert owner.resource_status()['cycle_running']
            release.set()
            await cleanup_entered.wait()
            for _ in range(5):
                await owner.reconcile()
            assert len(calls) == 1
            assert next(s for s in (await snapshot(privacy, receipt))['steps'] if s['participant'] == 'contexts')['state'] == 'running'
        finally:
            release.set()
            cleanup_release.set()
        await flush(owner)
        assert len(calls) == 1


async def test_close_timeout_keeps_callback_and_shared_writer_lease_until_actual_exit(tmp_path):
    async with graph(tmp_path, close_timeout_s=.01) as (privacy, owner, _, _):
        entered, release = asyncio.Event(), asyncio.Event()
        async def clean(step):
            entered.set()
            await release.wait()
            return ErasureResult(True)
        await target(privacy)
        owner.register(Participant.CONTEXTS, clean, versions=frozenset({1}))
        await owner.request(str(uuid4()), ACTOR)
        await owner.reconcile()
        await entered.wait()
        competitor = PrivacyServices(privacy.ledger.path)
        try:
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert owner.resource_status()['active'] == 1 and privacy.resource_status()['writer_lease']
            with pytest.raises(RuntimeError, match='writer_already_active'):
                await competitor.acall(competitor.initialize)
        finally:
            release.set()
            await owner.aclose()
            await privacy.aclose()
            await competitor.aclose()
        assert not privacy.resource_status()['writer_lease']


async def test_only_leased_restart_recovers_running_claim_and_rejects_stale_ack(tmp_path):
    clock = Clock()
    path = str(tmp_path/'privacy.db')
    privacy = PrivacyServices(path, clock=clock)
    await privacy.acall(privacy.initialize)
    await target(privacy)
    receipt = await privacy.acall(privacy.ledger.erasures.request, str(uuid4()), ACTOR)
    row = (await privacy.acall(privacy.ledger.erasures.due))[0]
    old_claim = await privacy.acall(privacy.ledger.erasures.claim, row)
    standalone = FactLedger(path, clock=clock)
    try:
        await asyncio.to_thread(standalone.initialize)
        assert (await privacy.acall(privacy.ledger.erasures.due)) == []
        await privacy.acall(privacy.initialize)
        assert (await privacy.acall(privacy.ledger.erasures.due)) == []
    finally:
        await asyncio.to_thread(standalone.close)
        await privacy.aclose()
    reopened = PrivacyServices(path, clock=clock)
    try:
        await reopened.acall(reopened.initialize)
        assert (await snapshot(reopened, receipt))['steps'][2]['participant'] == 'contexts'
        recovered = (await reopened.acall(reopened.ledger.erasures.due))[0]
        new_claim = await reopened.acall(reopened.ledger.erasures.claim, recovered)
        assert old_claim.claim_token != new_claim.claim_token
        assert not await reopened.acall(reopened.ledger.erasures.finish, old_claim, ErasureResult(True))
        assert await reopened.acall(reopened.ledger.erasures.finish, new_claim, ErasureResult(True))
    finally:
        await reopened.aclose()


async def test_prejournal_tombstone_preserves_actor_and_time_when_recovered(tmp_path):
    clock = Clock()
    path = str(tmp_path/'privacy.db')
    ledger = FactLedger(path, clock=clock)
    conversation_id = str(uuid4())
    ledger.initialize()
    with ledger._transaction() as db:
        db.execute('INSERT INTO conversation_tombstones VALUES (?,?,?)', (conversation_id, clock.now-100, 'original-admin'))
    ledger.close()
    privacy = PrivacyServices(path, clock=clock)
    try:
        await privacy.acall(privacy.initialize)
        receipt = await privacy.acall(privacy.ledger.erasures.request, conversation_id, ACTOR)
        assert receipt['requested_by'] == 'original-admin' and receipt['requested_at'] == clock.now-100
        assert receipt['state'] == 'blocked'
    finally:
        await privacy.aclose()


async def test_disabled_recall_cleans_real_policy_index_and_preserves_legacy_and_live_facts(tmp_path, monkeypatch):
    config = {'memory': {'enabled': False, 'privacy_db_path': str(tmp_path/'privacy.db'),
                         'chroma_path': str(tmp_path/'chroma')}}
    client = MemoryClient(tmp_path/'chroma')
    legacy = client.get_or_create_collection('user_memories')
    policy = client.get_or_create_collection('user_memories_policy_v1')
    legacy.upsert(ids=['legacy'], documents=['unsourced legacy stays quarantined'])
    async with AsyncExitStack() as stack:
        clock = Clock()
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        source = memory_source(now=clock.now)
        fact = await privacy.acall(privacy.ledger.store, 'erased synthetic fact', 'fact', .8, (source,))
        survivor = await privacy.acall(privacy.ledger.store, 'live synthetic fact', 'fact', .8, (memory_source(now=clock.now),))
        policy.upsert(ids=[fact.id, survivor.id], documents=[fact.content, survivor.content])
        # Cleanup-only access must never construct an embedding function or a
        # collection. Existing fake vectors were created before installing it.
        from tests.mocks.memory_graph import SyntheticEmbedding
        def forbid(*args, **kwargs):
            raise AssertionError('cleanup attempted embedding initialization')
        monkeypatch.setattr(SyntheticEmbedding, 'build_from_config', forbid)
        semantic_client = MagicMock()
        assert await build_memory_services(config, shutdown=stack, privacy=privacy, chroma_client=semantic_client) is None
        owner = await build_erasure_services(config, privacy=privacy, shutdown=stack)
        receipt = await owner.request(source.conversation_id, ACTOR)
        for _ in range(4):
            await flush(owner)
        assert policy.get(ids=[fact.id])['ids'] == []
        assert policy.get(ids=[survivor.id])['ids'] == [survivor.id]
        assert legacy.get(ids=['legacy'])['documents'] == ['unsourced legacy stays quarantined']
        semantic_client.get_or_create_collection.assert_not_called()
        status = await snapshot(privacy, receipt)
        index = next(s for s in status['steps'] if s['participant'] == 'semantic_index')
        assert index['state'] == 'complete' and status['state'] == 'blocked'
        assert (await privacy.acall(privacy.ledger.status))['pending_index'] == 1  # live upsert remains


async def test_absent_index_never_creates_files_or_opens_a_client(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, _):
        descriptor = {'path': str(tmp_path/'absent'), 'collection': 'user_memories_policy_v1'}
        factory = MagicMock(side_effect=AssertionError('absent target opened'))
        driver = PolicyIndexEraser(privacy, descriptor, client_factory=factory)
        owner.register(Participant.SEMANTIC_INDEX, driver.clean, versions=frozenset({1}), begin_close=driver.begin_close)
        await target(privacy, Participant.SEMANTIC_INDEX, descriptor=descriptor)
        receipt = await owner.request(str(uuid4()), ACTOR)
        await flush(owner)
        assert not Path(descriptor['path']).exists()
        factory.assert_not_called()
        assert next(s for s in (await snapshot(privacy, receipt))['steps'] if s['participant'] == 'semantic_index')['state'] == 'complete'


async def test_historical_index_locations_are_retained_and_cleaned_after_configuration_change(tmp_path):
    clock = Clock()
    source = memory_source(now=clock.now)
    config = {'memory': {'enabled': False, 'privacy_db_path': str(tmp_path/'privacy.db'), 'chroma_path': str(tmp_path/'old')}}
    old_client = MemoryClient(tmp_path/'old')
    old = old_client.get_or_create_collection('user_memories_policy_v1')
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        fact = await privacy.acall(privacy.ledger.store, 'synthetic erased fact', 'fact', .8, (source,))
        old.upsert(ids=[fact.id], documents=[fact.content])
        # Persist old configuration before a request, then close with no cleanup.
        await target(privacy, Participant.SEMANTIC_INDEX, descriptor={'path': str(tmp_path/'old'), 'collection': 'user_memories_policy_v1'})
        receipt = await privacy.acall(privacy.ledger.erasures.request, source.conversation_id, ACTOR)
    config['memory']['chroma_path'] = str(tmp_path/'new')
    new_client = MemoryClient(tmp_path/'new')
    new = new_client.get_or_create_collection('user_memories_policy_v1')
    new.upsert(ids=[fact.id], documents=['copied stale fact'])
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        owner = await build_erasure_services(config, privacy=privacy, shutdown=stack)
        for _ in range(4):
            await flush(owner)
        assert old.get(ids=[fact.id])['ids'] == new.get(ids=[fact.id])['ids'] == []
        steps = [s for s in (await snapshot(privacy, receipt))['steps'] if s['participant'] == 'semantic_index']
        assert len(steps) == 2 and all(s['state'] == 'complete' for s in steps)


async def test_stale_upsert_cannot_arrive_after_index_erasure_receipt(tmp_path):
    async with AsyncExitStack() as stack:
        config = {'memory': {'privacy_db_path': str(tmp_path/'privacy.db'), 'chroma_path': str(tmp_path/'chroma')}}
        clock = Clock()
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        services = await build_memory_services(config, shutdown=stack, privacy=privacy, chroma_client=MemoryClient(tmp_path/'chroma'))
        # Stop the automatic reconciler so the test owns the precise old-upsert race.
        services.maintenance.cancel()
        await asyncio.gather(services.maintenance, return_exceptions=True)
        memory = services.manager.long_term
        entered, release = threading.Event(), threading.Event()
        def block():
            entered.set()
            release.wait(3)
        memory._collection.before_upsert = block
        source = memory_source(now=clock.now)
        writing = asyncio.create_task(memory.astore_fact('synthetic fact in old upsert', 'fact', sources=(source,)))
        try:
            await until(entered.is_set)
            owner = await build_erasure_services(config, privacy=privacy, shutdown=stack, memory=memory)
            receipt = await owner.request(source.conversation_id, ACTOR)
            await owner.reconcile()
            await until(lambda: owner.resource_status()['active'] == 1)
            status = await snapshot(privacy, receipt)
            assert next(s for s in status['steps'] if s['participant'] == 'semantic_index')['state'] == 'running'
            assert not writing.done()
        finally:
            release.set()
        fact_id = await writing
        await flush(owner)
        assert memory._collection.get(ids=[fact_id])['ids'] == []
        assert next(s for s in (await snapshot(privacy, receipt))['steps'] if s['participant'] == 'semantic_index')['state'] == 'complete'


async def test_cursor_receipts_cannot_ack_another_version_or_claim(tmp_path):
    async with graph(tmp_path) as (privacy, owner, _, _):
        await target(privacy)
        await owner.request(str(uuid4()), ACTOR)
        row = (await privacy.acall(privacy.ledger.erasures.due))[0]
        step = await privacy.acall(privacy.ledger.erasures.claim, row)
        assert not await privacy.acall(privacy.ledger.erasures.finish, replace(step, version=2), ErasureResult(True))
        assert not await privacy.acall(privacy.ledger.erasures.finish, replace(step, claim_token=str(uuid4())), ErasureResult(True))
        assert await privacy.acall(privacy.ledger.erasures.finish, step, ErasureResult(False, cursor='fact_synthetic', pending_count=1))
        status = await privacy.acall(privacy.ledger.erasures.snapshot, step.operation_id)
        context = next(s for s in status['steps'] if s['participant'] == 'contexts')
        assert context['state'] == ErasureState.PENDING and context['pending_count'] == 1
        assert context['last_error'] == ErasureError.PENDING


async def test_real_index_batch_cursor_survives_restart_and_rechecks_old_receipts(tmp_path, monkeypatch):
    clock = Clock()
    config = {'memory': {'enabled': False, 'privacy_db_path': str(tmp_path/'privacy.db'), 'chroma_path': str(tmp_path/'chroma')}}
    descriptor = {'path': str(tmp_path/'chroma'), 'collection': 'user_memories_policy_v1'}
    client = MemoryClient(tmp_path/'chroma')
    policy = client.get_or_create_collection(descriptor['collection'])
    from chromadb.api.models.Collection import Collection
    delete = Collection.delete
    batches = []
    def observed_delete(self, *args, **kwargs):
        batches.append(tuple(kwargs['ids']))
        return delete(self, *args, **kwargs)
    monkeypatch.setattr(Collection, 'delete', observed_delete)
    async with graph(tmp_path) as (privacy, owner, _, _):
        source = memory_source(now=clock.now)
        facts = [await privacy.acall(privacy.ledger.store, f'synthetic fact {i}', 'fact', .8, (source,)) for i in range(33)]
        policy.upsert(ids=[f.id for f in facts], documents=[f.content for f in facts])
        driver = PolicyIndexEraser(privacy, descriptor)
        owner.register(Participant.SEMANTIC_INDEX, driver.clean, versions=frozenset({1}), begin_close=driver.begin_close)
        await target(privacy, Participant.SEMANTIC_INDEX, descriptor=descriptor)
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert [len(batch) for batch in batches] == [32]
        assert policy.count() == 1
        assert next(s for s in (await snapshot(privacy, receipt))['steps'] if s['participant'] == 'semantic_index')['state'] == 'pending'
    clock.now += 1
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        owner = await build_erasure_services(config, privacy=privacy, shutdown=stack)
        await flush(owner)
        assert policy.count() == 0
        maintenance = owner._drivers[Participant.SEMANTIC_INDEX].callback.__self__.maintenance
        await maintenance.reconcile()
        await asyncio.gather(*(task for _, task in maintenance._active.values()))
        await maintenance.reconcile()
        counts = Counter(fact_id for batch in batches for fact_id in batch)
        assert set(counts) == {fact.id for fact in facts}
        # Startup must recheck physical storage; a restored old index cannot
        # hide behind a receipt from the previous leased generation.
        assert sum(counts.values()) == 32+33
        assert all(counts[fact_id]==2 for fact_id in batches[0])
        clock.now += 1
        await flush(owner)
        assert counts == Counter(fact_id for batch in batches for fact_id in batch)
        assert next(s for s in (await snapshot(privacy, receipt))['steps'] if s['participant'] == 'semantic_index')['state'] == 'complete'


async def test_failed_native_index_cleanup_stays_pending_and_retries_without_legacy_edits(tmp_path, monkeypatch):
    async with graph(tmp_path) as (privacy, owner, clock, _):
        descriptor = {'path': str(tmp_path/'chroma'), 'collection': 'user_memories_policy_v1'}
        client = MemoryClient(tmp_path/'chroma')
        policy = client.get_or_create_collection(descriptor['collection'])
        source = memory_source(now=clock.now)
        fact = await privacy.acall(privacy.ledger.store, 'synthetic fact', 'fact', .8, (source,))
        policy.upsert(ids=[fact.id], documents=[fact.content])
        from chromadb.api.models.Collection import Collection
        delete = Collection.delete
        def failed_delete(*args, **kwargs):
            raise OSError('synthetic private index path must not enter receipts')
        monkeypatch.setattr(Collection, 'delete', failed_delete)
        driver = PolicyIndexEraser(privacy, descriptor)
        owner.register(Participant.SEMANTIC_INDEX, driver.clean, versions=frozenset({1}), begin_close=driver.begin_close)
        await target(privacy, Participant.SEMANTIC_INDEX, descriptor=descriptor)
        receipt = await owner.request(source.conversation_id, ACTOR)
        await owner.reconcile()
        await until(lambda: all(a.task.done() for a in owner._active.values()))
        await owner.reconcile()
        status = await snapshot(privacy, receipt)
        index = next(s for s in status['steps'] if s['participant'] == 'semantic_index')
        assert index['state'] == 'pending' and index['last_error'] == 'cleanup_failed'
        assert 'private index path' not in json.dumps(status)
        assert policy.count() == 1 and (await privacy.acall(privacy.ledger.status))['pending_index'] == 1
        monkeypatch.setattr(Collection, 'delete', delete)
        clock.now += 2
        await flush(owner)
        assert policy.count() == 0
        assert next(s for s in (await snapshot(privacy, receipt))['steps'] if s['participant'] == 'semantic_index')['state'] == 'complete'


async def test_native_index_delete_keeps_writer_lease_through_close_timeout_and_cancellation(tmp_path, monkeypatch):
    async with graph(tmp_path, close_timeout_s=.01) as (privacy, owner, clock, _):
        descriptor = {'path': str(tmp_path/'chroma'), 'collection': 'user_memories_policy_v1'}
        client = MemoryClient(tmp_path/'chroma')
        policy = client.get_or_create_collection(descriptor['collection'])
        source = memory_source(now=clock.now)
        fact = await privacy.acall(privacy.ledger.store, 'synthetic fact', 'fact', .8, (source,))
        policy.upsert(ids=[fact.id], documents=[fact.content])
        from chromadb.api.models.Collection import Collection
        delete = Collection.delete
        entered, release = threading.Event(), threading.Event()
        def blocked(self, *args, **kwargs):
            entered.set()
            release.wait(3)
            return delete(self, *args, **kwargs)
        monkeypatch.setattr(Collection, 'delete', blocked)
        driver = PolicyIndexEraser(privacy, descriptor)
        owner.register(Participant.SEMANTIC_INDEX, driver.clean, versions=frozenset({1}), begin_close=driver.begin_close)
        await target(privacy, Participant.SEMANTIC_INDEX, descriptor=descriptor)
        await owner.request(source.conversation_id, ACTOR)
        await owner.reconcile()
        closing = None
        competitor = PrivacyServices(privacy.ledger.path)
        try:
            await until(entered.is_set)
            with pytest.raises(WorkCloseTimeout):
                await privacy.aclose()
            assert driver._work.status()['operations'] == 1 and privacy.resource_status()['writer_lease']
            closing = asyncio.create_task(owner.aclose())
            await asyncio.sleep(0)
            closing.cancel()
            closing.cancel()
            with pytest.raises(WorkCloseTimeout):
                await closing
            with pytest.raises(RuntimeError, match='writer_already_active'):
                await competitor.acall(competitor.initialize)
        finally:
            release.set()
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            # The deliberately tiny observation deadline remains unchanged.
            # Releasing the native gate does not imply disk cleanup has exited;
            # observe the actual retained drains before asserting final release.
            await asyncio.wait_for(asyncio.shield(owner.begin_close()), 3)
            await asyncio.wait_for(asyncio.shield(privacy.begin_close()), 3)
            await competitor.aclose()
        assert policy.count() == 0 and driver._work.status()['operations'] == 0


async def test_semantic_factory_registers_target_before_first_index_access_even_if_initialization_fails(tmp_path):
    config = {'memory': {'privacy_db_path': str(tmp_path/'privacy.db'), 'chroma_path': str(tmp_path/'chroma')}}
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack)
        observed = []
        def first_index_access(**kwargs):
            with sqlite3.connect(privacy.ledger.path) as db:
                observed.extend(json.loads(row[0]) for row in db.execute('SELECT descriptor FROM erasure_targets'))
            raise OSError('synthetic crash during index initialization')
        client = MagicMock()
        client.get_or_create_collection.side_effect = first_index_access
        with pytest.raises(RuntimeError, match='^memory_services_initialization_failed$') as failure:
            await build_memory_services(config, shutdown=stack, privacy=privacy, chroma_client=client)
        assert failure.value.__context__ is None
        client.close.assert_not_called()  # The factory borrowed this client.
        assert observed == [{'path': str(tmp_path/'chroma'), 'collection': 'user_memories_policy_v1'}]


@pytest.mark.parametrize('scenario', ['exception', 'invalid_result', 'extra_fields'])
async def test_failed_ack_retains_only_sanitized_outcome_instead_of_original_driver_exception(tmp_path, monkeypatch, scenario):
    async with graph(tmp_path) as (privacy, owner, _, _):
        @dataclass(frozen=True)
        class RichResult(ErasureResult):
            private: str = 'synthetic private transcript in extra field'
        async def clean(step):
            if scenario == 'invalid_result':
                return {'done': True, 'private': 'synthetic private transcript in invalid result'}
            if scenario == 'extra_fields':
                return RichResult(True)
            raise OSError('synthetic private transcript in driver exception')
        await target(privacy)
        owner.register(Participant.CONTEXTS, clean, versions=frozenset({1}))
        await owner.request(str(uuid4()), ACTOR)
        await owner.reconcile()
        await until(lambda: all(a.task.done() for a in owner._active.values()))
        finish = privacy.ledger.erasures.finish
        def unavailable(*args):
            raise OSError('synthetic unavailable journal')
        monkeypatch.setattr(privacy.ledger.erasures, 'finish', unavailable)
        try:
            with pytest.raises(OSError):
                await owner.reconcile()
            retained = next(iter(owner._active.values())).task
            assert retained.exception() is None
            assert 'private transcript' not in repr(retained.result())
        finally:
            monkeypatch.setattr(privacy.ledger.erasures, 'finish', finish)


async def test_owned_cleanup_tasks_cannot_retain_arbitrary_request_context(tmp_path):
    from weakref import ref
    from src.core.logging import LogContext, get_context
    class Payload:
        pass
    async with graph(tmp_path) as (privacy,owner,_,_):
        entered,release=asyncio.Event(),asyncio.Event()
        seen=[]
        async def held(step):
            seen.append(get_context())
            entered.set()
            await release.wait()
            return ErasureResult(True,pending_count=0)
        await target(privacy,Participant.STREAMS)
        owner.register(Participant.STREAMS,held,versions=frozenset({1}))
        await owner.request(memory_source().conversation_id,ACTOR)
        payload=Payload()
        reference=ref(payload)
        try:
            with LogContext(private=payload):
                await owner.reconcile()
            payload=None
            await until(entered.is_set)
            assert seen==[{}]
            assert reference() is None
        finally:
            release.set()
            await flush(owner)


async def test_cancelled_cycle_observer_does_not_leave_private_context_in_done_callback(tmp_path,monkeypatch):
    from contextvars import Context
    from weakref import ref
    from src.core.logging import LogContext
    async with graph(tmp_path) as (privacy,owner,_,_):
        await target(privacy)
        async def clean(step):
            return ErasureResult(True)
        owner.register(Participant.CONTEXTS,clean,versions=frozenset({1}))
        await owner.request(memory_source().conversation_id,ACTOR)
        entered,release=threading.Event(),threading.Event()
        claim=owner.journal.claim
        def held(row):
            entered.set()
            assert release.wait(3)
            return claim(row)
        monkeypatch.setattr(owner.journal,'claim',held)
        class Payload:
            pass
        holder=[Payload()]
        reference=ref(holder[0])
        async def observe():
            with LogContext(private=holder.pop()):
                await owner.reconcile()
        observer=asyncio.create_task(observe(),context=Context())
        try:
            await until(entered.is_set)
            observer.cancel()
            try:
                await observer
            except asyncio.CancelledError:
                pass
            observer=None
            assert reference() is None
        finally:
            release.set()
            await owner._cycle
            await flush(owner)
