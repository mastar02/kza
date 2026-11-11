"""A prompt read survives facts the ledger admits before the index holds them."""

import threading
import time

import pytest

from src.memory.fact_ledger import MemoryReviewer
from src.memory.memory_manager import MemoryManager
from tests.mocks.memory_graph import MemoryClient
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio


def _lag_reconcile_barrier(manager):
    """Store while another reconcile pass owns the writer barrier (the full-suite race)."""
    memory = manager.long_term
    assert memory._index_lock.acquire(blocking=False)
    try:
        return memory.store_fact('PENDING_FACT_457', 'fact', .9, sources=(memory_source(now=time.time()),))
    finally:
        memory._index_lock.release()


def _lag_upsert_error(manager):
    """Store while the index rejects the upsert, leaving a retried job behind."""
    memory = manager.long_term
    index = memory._collection
    index.upsert_error = RuntimeError('synthetic_index_unavailable')
    try:
        return memory.store_fact('PENDING_FACT_457', 'fact', .9, sources=(memory_source(now=time.time()),))
    finally:
        index.upsert_error = None


@pytest.mark.parametrize('lag', [_lag_reconcile_barrier, _lag_upsert_error], ids=['reconcile_barrier', 'upsert_error'])
async def test_unindexed_eligible_fact_does_not_fail_the_prompt_read(tmp_path, lag):
    client = MemoryClient(tmp_path / 'chroma')
    manager = MemoryManager(client, preferences_path=str(tmp_path / 'preferences.json'),
                            ledger_path=str(tmp_path / 'privacy.db'))
    memory = manager.long_term
    try:
        await manager.ainitialize()
        indexed = await memory.astore_fact('INDEXED_FACT_312', 'fact', .9, sources=(memory_source(now=time.time()),))
        pending = await memory.acall(lag, manager)
        # The ledger admits the new fact; the index does not hold it yet.
        assert pending in memory.ledger.eligible_ids(.6, scope='all')
        assert memory._collection.get(ids=[pending])['ids'] == []
        assert memory.ledger.status()['pending_index'] == 1

        assert [row['id'] for row in await memory.asearch_relevant('cielo', min_confidence=.6, scope='all')] == [indexed]
        prompt = await manager.aprompt_context('cielo')
        assert 'INDEXED_FACT_312' in prompt.text and 'PENDING_FACT_457' not in prompt.text

        # Once maintenance indexes it, the same read selects it.
        await memory.areconcile()
        assert memory.ledger.status()['pending_index'] == 0
        prompt = await manager.aprompt_context('cielo')
        assert 'INDEXED_FACT_312' in prompt.text and 'PENDING_FACT_457' in prompt.text
    finally:
        await memory.aclose()
        client.close()


async def test_held_index_barrier_fails_the_search_fast_within_its_bound(tmp_path, monkeypatch):
    import src.memory.long_term as long_term_module

    monkeypatch.setattr(long_term_module, 'SEARCH_BARRIER_TIMEOUT_S', .05)
    client = MemoryClient(tmp_path / 'chroma')
    manager = MemoryManager(client, preferences_path=str(tmp_path / 'preferences.json'),
                            ledger_path=str(tmp_path / 'privacy.db'))
    memory = manager.long_term
    try:
        await manager.ainitialize()
        indexed = await memory.astore_fact('INDEXED_FACT_312', 'fact', .9, sources=(memory_source(now=time.time()),))
        # A hung index writer (reconcile or erasure batch) owns the barrier.
        assert memory._index_lock.acquire(blocking=False)
        try:
            started = time.monotonic()
            with pytest.raises(RuntimeError, match='memory_search_busy'):
                await memory.asearch_relevant('cielo', min_confidence=.6, scope='all')
            with pytest.raises(RuntimeError, match='memory_prompt_unavailable'):
                await manager.aprompt_context('cielo')
            # Two bounded waits, nowhere near the default bound.
            assert time.monotonic() - started < long_term_module.SEARCH_BARRIER_TIMEOUT_S * 2 + 1
            assert not memory._query_lock.locked()
        finally:
            memory._index_lock.release()
        assert [row['id'] for row in await memory.asearch_relevant('cielo', min_confidence=.6, scope='all')] == [indexed]
    finally:
        await memory.aclose()
        client.close()


async def test_index_delete_cannot_land_between_the_presence_read_and_the_query(tmp_path):
    client = MemoryClient(tmp_path / 'chroma')
    manager = MemoryManager(client, preferences_path=str(tmp_path / 'preferences.json'),
                            ledger_path=str(tmp_path / 'privacy.db'))
    memory = manager.long_term
    try:
        await manager.ainitialize()
        kept = await memory.astore_fact('KEPT_FACT_204', 'fact', .9, sources=(memory_source(now=time.time()),))
        doomed = await memory.astore_fact('DOOMED_FACT_861', 'fact', .9, sources=(memory_source(now=time.time()),))
        index = memory._collection
        real_query = index.query
        deletions = []

        def query(**kwargs):
            # The search has already read which IDs the index holds. Delete one
            # of them now on another thread, as a review deletion does: ledger
            # delete, then its reconcile pass removes the index row unless the
            # search still owns the writer barrier.
            assert doomed in kwargs['ids']
            worker = threading.Thread(target=lambda: deletions.append(
                memory.delete_fact(doomed, MemoryReviewer('fixture-owner'))))
            worker.start()
            worker.join()
            return real_query(**kwargs)

        index.query = query
        try:
            results = await memory.asearch_relevant('cielo', min_confidence=.6, scope='all')
        finally:
            index.query = real_query
        assert deletions == [True]
        assert [row['id'] for row in results] == [kept]
        # The barrier only deferred the index cleanup; the next pass removes it.
        assert index.get(ids=[doomed])['ids'] == [doomed]
        await memory.areconcile()
        assert index.get(ids=[doomed])['ids'] == []
        assert memory.ledger.status()['pending_index'] == 0
    finally:
        await memory.aclose()
        client.close()
