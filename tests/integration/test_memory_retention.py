"""Authoritative retention/review against real temporary Chroma and SQLite."""

import asyncio
import threading
from uuid import uuid4

import pytest

from src.dashboard.auth import DashboardPrincipal
from src.memory.fact_ledger import MemoryOrigin, MemoryReviewer, MemorySource
from src.memory.long_term import LongTermMemory, MemoryCloseTimeout
from tests.mocks.memory_graph import Clock, MemoryClient

pytestmark = pytest.mark.asyncio


def source(clock, *, conversation_id=None):
    return MemorySource(str(uuid4()), conversation_id or str(uuid4()), clock.now, clock.now+48*3600, MemoryOrigin.AMBIENT)


def reviewer():
    return MemoryReviewer.from_dashboard(DashboardPrincipal('test-admin', frozenset({'memory:review'})))


def graph(tmp_path, *, clock=None, **options):
    clock = clock or Clock()
    client = MemoryClient(tmp_path / 'chroma')
    memory = LongTermMemory(client, ledger_path=str(tmp_path / 'privacy.db'), clock=clock, **options)
    return memory, client, clock


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.001)


async def test_automatic_facts_expire_at_30_days_without_access_renewal_or_restart_revival(tmp_path):
    memory, client, clock = graph(tmp_path)
    origin = source(clock)
    fact_id = await memory.astore_fact('Le gusta cocinar.', 'preference', sources=(origin,))
    before = memory.ledger.get(fact_id)
    clock.now += 29*86400
    assert (await memory.asearch_relevant('cocinar'))[0]['id'] == fact_id
    assert memory.ledger.get(fact_id).expires_at == before.expires_at
    await memory.aclose()
    clock.now += 86400
    reopened = LongTermMemory(client, ledger_path=str(tmp_path / 'privacy.db'), clock=clock)
    try:
        assert await reopened.asearch_relevant('cocinar') == []
        assert await reopened.acall(reopened.get_by_category, 'preference') == []
        await reopened.areconcile()
        assert reopened._collection.count() == 0
        assert reopened.ledger.status()['states'] == {'deleted': 1}
    finally:
        await reopened.aclose()


async def test_legacy_collection_remains_unchanged_and_is_not_queried_for_context(tmp_path):
    memory, client, _ = graph(tmp_path)
    legacy = client.get_or_create_collection('user_memories')
    legacy.upsert(ids=['legacy'], documents=['Un registro sin fuentes.'], metadatas=[{'origin': 'ambient'}])
    try:
        assert await memory.asearch_relevant('registro') == []
        assert legacy.count() == 1
        assert legacy.get(ids=['legacy'])['documents'] == ['Un registro sin fuentes.']
        assert (await memory.acall(memory.get_stats))['legacy_policy'] == 'quarantined'
        assert client.requested == ['user_memories', 'user_memories_policy_v1']
    finally:
        await memory.aclose()


async def test_proposal_is_excluded_until_manual_review_and_confirmation_does_not_expire(tmp_path):
    memory, _, clock = graph(tmp_path)
    origin = source(clock)
    try:
        proposal = await memory.acall(memory.propose, 'Recordar este dato.', (origin,))
        assert await memory.asearch_relevant('dato') == []
        with pytest.raises(PermissionError):
            await memory.acall(memory.confirm, proposal['id'], None)
        confirmed = await memory.acall(memory.confirm, proposal['id'], reviewer())
        assert confirmed['confirmed_by'] == 'test-admin' and confirmed['expires_at'] is None
        original_confirmation = confirmed['confirmed_at']
        clock.now += 90*86400
        assert (await memory.asearch_relevant('dato'))[0]['state'] == 'confirmed'
        again = await memory.acall(memory.confirm, proposal['id'], reviewer())
        assert again['confirmed_at'] == original_confirmation
        assert await memory.acall(memory.delete_fact, proposal['id'], reviewer())
        assert await memory.asearch_relevant('dato') == []
    finally:
        await memory.aclose()


async def test_expired_proposal_cannot_be_confirmed_or_recreated_by_replay(tmp_path):
    memory, _, clock = graph(tmp_path)
    origin = source(clock)
    try:
        proposal = await memory.acall(memory.propose, 'Propuesta vencida.', (origin,))
        clock.now += 48*3600
        with pytest.raises(LookupError):
            await memory.acall(memory.confirm, proposal['id'], reviewer())
        with pytest.raises(PermissionError):
            await memory.acall(memory.propose, 'Propuesta vencida.', (origin,))
        assert await memory.acall(memory.proposals) == []
    finally:
        await memory.aclose()


async def test_deleted_conversation_blocks_pending_and_replayed_derivation_after_restart(tmp_path):
    memory, client, clock = graph(tmp_path)
    origin = source(clock)
    await memory.astore_fact('Un hecho derivado.', 'fact', sources=(origin,))
    assert await memory.acall(memory.delete_conversation, origin.conversation_id, reviewer()) == 1
    await memory.aclose()
    reopened = LongTermMemory(client, ledger_path=str(tmp_path / 'privacy.db'), clock=clock)
    try:
        with pytest.raises(PermissionError):
            await reopened.astore_fact('Un hecho tardío.', 'fact', sources=(origin,))
        assert await reopened.asearch_relevant('hecho') == []
    finally:
        await reopened.aclose()


async def test_expiry_scrubs_catalog_and_keeps_failed_vector_cleanup_durable(tmp_path):
    memory, client, clock = graph(tmp_path)
    fact_id = await memory.astore_fact('Contenido temporal.', 'fact', sources=(source(clock),))
    memory._collection.delete_error = OSError('synthetic index unavailable')
    clock.now += 30*86400
    status = await memory.areconcile()
    assert status['pending_index'] == 1 and memory.ledger.get(fact_id) is None
    assert memory._collection.count() == 1
    assert await memory.asearch_relevant('temporal') == []
    await memory.aclose()
    memory, _, _ = graph(tmp_path, clock=clock)
    try:
        await memory.areconcile()
        assert memory._collection.count() == 0 and memory.ledger.status()['pending_index'] == 0
    finally:
        await memory.aclose()


async def test_slow_old_upsert_cannot_acknowledge_newer_conversation_deletion(tmp_path):
    memory, _, clock = graph(tmp_path)
    await memory.acall(memory.initialize)
    entered, release = threading.Event(), threading.Event()

    def blocked_upsert():
        entered.set()
        assert release.wait(3)

    memory._collection.before_upsert = blocked_upsert
    origin = source(clock)
    saving = asyncio.create_task(memory.astore_fact('Un resultado tardío.', 'fact', sources=(origin,)))
    try:
        await until(entered.is_set)
        assert await memory.acall(memory.delete_conversation, origin.conversation_id, reviewer()) == 1
        assert await memory.asearch_relevant('tardío') == []
        release.set()
        await saving
        assert memory.ledger.status()['pending_index'] == 1
        assert await memory.asearch_relevant('tardío') == []
        await memory.areconcile()
        assert memory._collection.count() == 0 and memory.ledger.status()['pending_index'] == 0
    finally:
        release.set()
        await asyncio.gather(saving, return_exceptions=True)
        await memory.aclose()


async def test_replaying_one_derivation_never_duplicates_or_renews_its_fact(tmp_path):
    memory, _, clock = graph(tmp_path)
    origin = source(clock)
    try:
        first = await memory.astore_fact('Un hecho.', 'fact', sources=(origin,))
        initial = memory.ledger.get(first)
        clock.now += 3600
        repeated = await memory.astore_fact('Un hecho.', 'fact', sources=(origin,))
        assert repeated == first and memory.ledger.get(first).expires_at == initial.expires_at
        assert memory._collection.count() == 1
        assert await memory.acall(memory.delete_fact, first, reviewer())
        with pytest.raises(PermissionError):
            await memory.astore_fact('Un hecho.', 'fact', sources=(origin,))
    finally:
        await memory.aclose()


@pytest.mark.parametrize('metadata', [{'confirmed': True}, {'expires_at': 9e9}, {'confirmed_by': 'voice'}, {'source': 'guess'}])
async def test_metadata_cannot_mint_confirmation_or_retention(tmp_path, metadata):
    memory, _, clock = graph(tmp_path)
    try:
        with pytest.raises(ValueError):
            await memory.astore_fact('Un hecho.', 'fact', sources=(source(clock),), metadata=metadata)
    finally:
        await memory.aclose()


async def test_cancelled_native_work_remains_owned_and_close_reports_it(tmp_path):
    memory, _, clock = graph(tmp_path, shutdown_timeout_s=.05)
    await memory.acall(memory.initialize)
    entered, release = threading.Event(), threading.Event()

    def blocked_upsert():
        entered.set()
        assert release.wait(3)

    memory._collection.before_upsert = blocked_upsert
    task = asyncio.create_task(memory.astore_fact('Un hecho.', 'fact', sources=(source(clock),)))
    try:
        await until(entered.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert memory.resource_status()['operations'] == 1
        with pytest.raises(MemoryCloseTimeout):
            await memory.aclose()
        assert memory.resource_status()['operations'] == 1
        release.set()
        await until(lambda: memory.resource_status()['operations'] == 0)
        await memory.aclose()
    finally:
        release.set()
        await memory.aclose()


async def test_only_one_policy_writer_can_use_a_persistent_ledger(tmp_path):
    memory, client, clock = graph(tmp_path)
    await memory.acall(memory.initialize)
    duplicate = LongTermMemory(client, ledger_path=str(tmp_path/'privacy.db'), clock=clock)
    try:
        with pytest.raises(RuntimeError, match='memory_writer_already_active'):
            await duplicate.acall(duplicate.initialize)
        assert (tmp_path/'privacy.db').stat().st_mode & 0o777 == 0o600
    finally:
        await duplicate.aclose()
        await memory.aclose()


async def test_worker_admission_stays_bounded_until_actual_native_exit(tmp_path):
    memory, _, _ = graph(tmp_path)
    release = threading.Event()
    tasks = [asyncio.create_task(memory.acall(release.wait, 3)) for _ in range(8)]
    try:
        await until(lambda: memory.resource_status()['operations'] == 8)
        with pytest.raises(RuntimeError, match='memory_operation_capacity'):
            await memory.acall(lambda: None)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert memory.resource_status()['operations'] == 8
    finally:
        release.set()
        await memory.aclose()
