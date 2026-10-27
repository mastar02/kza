"""Private lineage and actual local diagnostic cleanup behind durable receipts."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import sqlite3
import threading

import pytest

from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.obs.diagnostic_erasure import build_diagnostic_services
from tests.integration.test_diagnostic_metric_privacy import request, receiver
from tests.integration.test_diagnostic_retention import rows
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.ambient_graph import Clock
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')


@asynccontextmanager
async def graph(tmp_path, *, config=None, timeout=1, history=0):
    config = config or {'observability': {'metrics_path': str(tmp_path/'metrics.jsonl')},
        'analytics': {'enabled': False}, 'memory': {'enabled': False}}
    async with AsyncExitStack() as stack:
        clock = Clock(2_000_000_000.)
        privacy = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock, close_timeout_s=timeout)
        stack.push_async_callback(privacy.aclose)
        await privacy.acall(privacy.initialize)
        from src.memory.erasure_journal import ErasureParticipant
        from src.obs.diagnostic_erasure import diagnostic_descriptor
        for index in range(history):
            descriptor = diagnostic_descriptor(str(tmp_path/f'history-{index}.jsonl'))
            await privacy.acall(privacy.ledger.erasures.register_target,
                ErasureParticipant.DIAGNOSTICS, 2, descriptor)
        owner = ErasureCoordinator(privacy, close_timeout_s=timeout)
        stack.push_async_callback(owner.aclose)
        driver = await build_diagnostic_services(config, privacy=privacy, owner=owner)
        yield privacy, owner, driver, clock, config


async def flush(owner):
    await owner.reconcile()
    await asyncio.gather(*(active.task for active in owner._active.values()))
    await owner.reconcile()


async def numeric_steps(privacy, receipt):
    def select():
        with privacy.ledger._transaction() as db:
            return [dict(row) for row in db.execute("""SELECT s.* FROM erasure_steps s
                JOIN erasure_targets t ON s.target_id=t.id WHERE s.operation_id=?
                AND t.participant='diagnostics' AND t.version=2""", (receipt['id'],))]
    return await privacy.acall(select)


async def test_private_lineage_never_enters_file_or_tcp_and_mixed_sources_erase_together(tmp_path):
    async with receiver() as (port, received):
        config = {'observability': {'metrics_path': str(tmp_path/'metrics.jsonl'),
            'logstash_host': '127.0.0.1', 'logstash_port': port}}
        async with graph(tmp_path, config=config) as (privacy, owner, driver, clock, _):
            first, second, other = (memory_source(now=clock.now) for _ in range(3))
            emitter = driver.emitter
            request(emitter, sources=(first, second))
            request(emitter, sources=(other,))
            await emitter.flush()
            await until(lambda: len(received) == 2)
            assert len(rows(emitter)) == 2
            assert all(source.conversation_id not in ''.join(received)+emitter.path.read_text()
                for source in (first, second, other))
            receipt = await owner.request(second.conversation_id, ACTOR)
            await flush(owner)
            assert len(rows(emitter)) == 1
            assert (await numeric_steps(privacy, receipt))[0]['state'] == 'complete'
            snapshot = await privacy.acall(privacy.ledger.erasures.snapshot, receipt['id'])
            assert snapshot['state'] != 'complete'
            with sqlite3.connect(emitter._store.catalog) as db:
                assert db.execute('SELECT conversation_id FROM lineage').fetchall() == [(other.conversation_id,)]


async def test_queued_writer_is_not_acknowledged_until_actual_exit_and_cannot_refill(tmp_path, monkeypatch):
    async with graph(tmp_path) as (privacy, owner, driver, clock, _):
        emitter = driver.emitter
        source = memory_source(now=clock.now)
        entered, release = threading.Event(), threading.Event()
        write = emitter._write
        def held(record):
            entered.set()
            assert release.wait(3)
            write(record)
        monkeypatch.setattr(emitter, '_write', held)
        request(emitter, sources=(source,))
        try:
            await until(entered.is_set)
            receipt = await owner.request(source.conversation_id, ACTOR)
            await flush(owner)
            step = (await numeric_steps(privacy, receipt))[0]
            assert step['state'] == 'pending' and step['pending_count'] == 1
            assert not rows(emitter)
        finally:
            release.set()
            await emitter.flush()
        clock.now += 2
        await flush(owner)
        assert not rows(emitter)
        assert (await numeric_steps(privacy, receipt))[0]['state'] == 'complete'


async def test_unattributed_previous_numeric_rows_block_local_completion(tmp_path):
    from src.obs.metrics_emitter import MetricsEmitter
    clock = Clock(2_000_000_000.)
    old = MetricsEmitter(str(tmp_path/'metrics.jsonl'), clock=clock)
    request(old)
    await old.flush()
    await old.aclose()
    # A real v1 catalog had no source relation. Migration must not invent it.
    with sqlite3.connect(old._store.catalog) as db:
        db.execute('DROP TABLE lineage')
        db.execute('PRAGMA user_version=1')
    async with graph(tmp_path) as (privacy, owner, driver, clock, _):
        source = memory_source(now=clock.now)
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        step = (await numeric_steps(privacy, receipt))[0]
        assert step['state'] == 'pending' and step['pending_count'] == 1
        assert len(rows(driver.emitter)) == 1


async def test_historical_path_is_cleaned_with_metrics_and_semantic_memory_disabled(tmp_path):
    async with graph(tmp_path) as (_, _, driver, clock, config):
        source = memory_source(now=clock.now)
        request(driver.emitter, sources=(source,))
        await driver.emitter.flush()
        previous_path = driver.emitter.path
    config['observability'] = {'enabled': False, 'metrics_path': str(tmp_path/'next.jsonl')}
    async with graph(tmp_path, config=config) as (privacy, owner, driver, _, _):
        assert driver.emitter is None
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert previous_path.read_text() == ''
        assert all(step['state'] == 'complete' for step in await numeric_steps(privacy, receipt))


async def test_restored_diagnostic_catalog_reopens_previously_completed_receipt(tmp_path):
    async with graph(tmp_path) as (privacy, owner, driver, clock, config):
        emitter = driver.emitter
        source = memory_source(now=clock.now)
        request(emitter, sources=(source,))
        await emitter.flush()
        catalog = emitter._store.catalog
        def backup():
            with emitter._store._lock:
                with sqlite3.connect(catalog) as db, sqlite3.connect(tmp_path/'backup.db') as copy:
                    db.backup(copy)
            return (tmp_path/'backup.db').read_bytes()
        snapshot = await asyncio.to_thread(backup)
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert (await numeric_steps(privacy, receipt))[0]['state'] == 'complete'
        assert not rows(emitter)
    catalog.write_bytes(snapshot)
    async with graph(tmp_path, config=config) as (privacy, owner, driver, _, _):
        assert (await numeric_steps(privacy, receipt))[0]['state'] == 'pending'
        assert not rows(driver.emitter)
        await flush(owner)
        assert (await numeric_steps(privacy, receipt))[0]['state'] == 'complete'


async def test_actual_router_and_native_wake_carry_their_sources(tmp_path):
    from types import SimpleNamespace
    from src.memory.producer_context import producer_context
    from src.wakeword.whisper_wake import WhisperWakeDetector
    from tests.integration.test_prompt_provenance import request_router
    from tests.mocks.responses import response_handler_mock
    async with graph(tmp_path) as (privacy, owner, driver, clock, _):
        source = memory_source(now=clock.now)
        router = request_router(None, response_handler_mock())
        router.metrics_emitter = driver.emitter
        token = router._memory_source.set(source)
        try:
            router._log_latency({'latency_ms': 20., 'timings': {'stt': 12.},
                'intent':'domotics', 'path':'fast', 'success':True})
        finally:
            router._memory_source.reset(token)
        detector = WhisperWakeDetector(SimpleNamespace(), wake_words=['synthetic'], metrics_emitter=driver.emitter)
        with producer_context('synthetic-room', (source,)):
            await asyncio.to_thread(detector._emit_wake, False, None, 'rejected', '', 80., 20.)
        await driver.emitter.flush()
        assert {row['event_type'] for row in rows(driver.emitter)} == {'request', 'wake'}
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert not rows(driver.emitter)
        assert (await numeric_steps(privacy, receipt))[0]['state'] == 'complete'


async def test_revocation_during_tcp_connect_waits_for_actual_exit_without_sending(tmp_path, monkeypatch):
    import src.obs.metrics_emitter as module
    config = {'observability': {'metrics_path': str(tmp_path/'metrics.jsonl'), 'logstash_host':'synthetic.invalid'}}
    entered, release = threading.Event(), threading.Event()
    sent = []
    class Connection:
        def settimeout(self, value):
            pass
        def sendall(self, payload):
            sent.append(payload)
        def close(self):
            pass
    def connect(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return Connection()
    monkeypatch.setattr(module.socket, 'create_connection', connect)
    async with graph(tmp_path, config=config) as (privacy, owner, driver, clock, _):
        source = memory_source(now=clock.now)
        request(driver.emitter, sources=(source,))
        try:
            await until(entered.is_set)
            receipt = await owner.request(source.conversation_id, ACTOR)
            await flush(owner)
            assert not rows(driver.emitter)
            assert (await numeric_steps(privacy, receipt))[0]['pending_count'] == 1
        finally:
            release.set()
            await driver.emitter.flush()
        clock.now += 2
        await flush(owner)
        assert not sent
        assert (await numeric_steps(privacy, receipt))[0]['state'] == 'complete'


async def test_unrelated_blocked_writer_does_not_delay_matching_receipt(tmp_path, monkeypatch):
    async with graph(tmp_path) as (privacy, owner, driver, clock, _):
        source, other = (memory_source(now=clock.now) for _ in range(2))
        emitter = driver.emitter
        request(emitter, sources=(source,))
        await emitter.flush()
        write = emitter._write
        entered, release = threading.Event(), threading.Event()
        def held(record):
            entered.set()
            assert release.wait(3)
            write(record)
        monkeypatch.setattr(emitter, '_write', held)
        request(emitter, sources=(other,))
        try:
            await until(entered.is_set)
            receipt = await owner.request(source.conversation_id, ACTOR)
            await flush(owner)
            assert (await numeric_steps(privacy, receipt))[0]['state'] == 'complete'
            assert emitter.pending_conversation(other.conversation_id) == 1
        finally:
            release.set()
            await emitter.flush()
        assert len(rows(emitter)) == 1


async def test_pending_revocation_does_not_irreversibly_delete_before_failed_commit(tmp_path):
    async with graph(tmp_path) as (privacy, _, driver, clock, _):
        source = memory_source(now=clock.now)
        emitter = driver.emitter
        request(emitter, sources=(source,))
        await emitter.flush()
        connection = privacy.ledger._db
        entered, release = threading.Event(), threading.Event()
        class FailedCommit:
            def __getattr__(self, name):
                return getattr(connection, name)
            def commit(self):
                entered.set()
                assert release.wait(3)
                raise OSError('synthetic_commit_failure')
        privacy.ledger._db = FailedCommit()
        deleting = asyncio.create_task(privacy.acall(privacy.ledger.delete_conversation, source.conversation_id, ACTOR))
        try:
            await until(entered.is_set)
            assert privacy.ledger.conversation_revoked(source.conversation_id)
            assert not privacy.ledger.conversation_committed_revoked(source.conversation_id)
            await emitter.aprune()
            assert len(rows(emitter)) == 1
            request(emitter, sources=(source,))
            await emitter.flush()
            assert len(rows(emitter)) == 1
        finally:
            release.set()
            with pytest.raises(OSError, match='synthetic_commit_failure'):
                await deleting
            privacy.ledger._db = connection
        assert not privacy.ledger.conversation_revoked(source.conversation_id)
        await emitter.aprune()
        assert len(rows(emitter)) == 1


async def test_native_reservation_before_post_is_included_in_erasure_barrier(tmp_path, monkeypatch):
    async with graph(tmp_path) as (privacy, owner, driver, clock, _):
        emitter = driver.emitter
        source = memory_source(now=clock.now)
        entered, release = threading.Event(), threading.Event()
        record = emitter._store.record
        def held(*args):
            result = record(*args)
            entered.set()
            assert release.wait(3)
            return result
        monkeypatch.setattr(emitter._store, 'record', held)
        posting = asyncio.create_task(asyncio.to_thread(request, emitter, sources=(source,)))
        try:
            await until(entered.is_set)
            receipt = await owner.request(source.conversation_id, ACTOR)
            await flush(owner)
            assert (await numeric_steps(privacy, receipt))[0]['pending_count'] == 1
        finally:
            release.set()
            await posting
            await emitter.flush()
        clock.now += 2
        await flush(owner)
        assert not rows(emitter)
        assert (await numeric_steps(privacy, receipt))[0]['state'] == 'complete'


async def test_orphan_quarantine_from_partial_restore_is_not_acknowledged_as_empty(tmp_path):
    from uuid import uuid4
    unverified = tmp_path/('metrics_policy_v1.jsonl.quarantine-'+str(uuid4()))
    unverified.write_text('Synthetic unknown legacy content\n')
    async with graph(tmp_path) as (privacy, owner, driver, clock, _):
        source = memory_source(now=clock.now)
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert (await numeric_steps(privacy, receipt))[0]['pending_count'] == 1
        assert unverified.read_text() == 'Synthetic unknown legacy content\n'
        assert not rows(driver.emitter)


async def test_cancelled_observer_and_close_timeout_retain_native_cleanup_and_privacy_lease(tmp_path, monkeypatch):
    from src.core.owned_work import WorkCloseTimeout
    async with graph(tmp_path, timeout=.03) as (privacy, owner, driver, clock, _):
        emitter = driver.emitter
        source = memory_source(now=clock.now)
        request(emitter, sources=(source,))
        await emitter.flush()
        erase = emitter._store.erase
        entered, release = threading.Event(), threading.Event()
        def held(conversation):
            entered.set()
            assert release.wait(3)
            return erase(conversation)
        monkeypatch.setattr(emitter._store, 'erase', held)
        receipt = await owner.request(source.conversation_id, ACTOR)
        await owner.reconcile()
        cleanup = next(active.task for active in owner._active.values()
            if active.step.participant.value == 'diagnostics')
        try:
            await until(entered.is_set)
            observer = asyncio.ensure_future(asyncio.shield(cleanup))
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            with pytest.raises(WorkCloseTimeout):
                await owner.aclose()
            assert not cleanup.done()
            assert privacy.resource_status()['writer_lease']
            assert emitter.resource_status()['retention']['writer_lease']
            contender = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock)
            try:
                with pytest.raises(RuntimeError, match='privacy_writer_already_active'):
                    await contender.acall(contender.initialize)
            finally:
                await contender.aclose()
        finally:
            release.set()
            await owner.begin_close()
        assert not rows(emitter)
        assert (await numeric_steps(privacy, receipt))[0]['state'] == 'complete'


@pytest.mark.parametrize('corruption', ['payload_matching', 'payload_unrelated', 'lineage'])
async def test_unverified_restored_catalog_is_not_republished_and_receipts_follow_physical_repair(tmp_path, corruption):
    async with graph(tmp_path) as (_, _, driver, clock, config):
        source = memory_source(now=clock.now)
        request(driver.emitter, sources=(source,))
        await driver.emitter.flush()
        path, catalog = driver.emitter.path, driver.emitter._store.catalog
    with sqlite3.connect(catalog) as db:
        if corruption.startswith('payload'):
            db.execute('UPDATE metrics SET payload=?', ('{"event_type":"request","diagnostic_schema":1,"text":"Synthetic private restore"}',))
        else:
            db.execute('UPDATE lineage SET conversation_id=?', ('Synthetic private restore',))
    async with graph(tmp_path, config=config) as (privacy, owner, driver, _, _):
        assert 'Synthetic private restore' not in path.read_text()
        erased = source if corruption == 'payload_matching' else memory_source(now=clock.now)
        receipt = await owner.request(erased.conversation_id, ACTOR)
        await flush(owner)
        if corruption == 'payload_matching':
            # A startup failure may still be the selected maintenance observer.
            # Its fixed retry must be allowed before asserting physical repair.
            clock.now += 2
            await flush(owner)
        assert ((await numeric_steps(privacy, receipt))[0]['state'] == 'complete') == (corruption == 'payload_matching')


async def test_failed_physical_publication_cannot_acknowledge_catalog_only_cleanup(tmp_path, monkeypatch):
    async with graph(tmp_path) as (privacy, owner, driver, clock, _):
        source = memory_source(now=clock.now)
        emitter = driver.emitter
        request(emitter, sources=(source,))
        await emitter.flush()
        publish = emitter._store._publish
        def failed(db):
            raise OSError('synthetic_projection_failure')
        monkeypatch.setattr(emitter._store, '_publish', failed)
        receipt = await owner.request(source.conversation_id, ACTOR)
        await flush(owner)
        assert (await numeric_steps(privacy, receipt))[0]['state'] == 'pending'
        assert len(rows(emitter)) == 1
        monkeypatch.setattr(emitter._store, '_publish', publish)
        clock.now += 2
        await flush(owner)
        assert not rows(emitter)
        assert (await numeric_steps(privacy, receipt))[0]['state'] == 'complete'


async def test_many_historical_destinations_share_one_actual_native_worker(tmp_path, monkeypatch):
    config = {'observability': {'enabled':False, 'metrics_path':str(tmp_path/'metrics.jsonl')}}
    async with graph(tmp_path, config=config, history=4) as (_, owner, driver, clock, _):
        assert driver.emitter is None and len(driver._history) == 5
        entered, release = threading.Event(), threading.Event()
        native = driver._invoke_history
        active = peak = calls = 0
        def held(*args):
            nonlocal active, peak, calls
            active += 1
            peak = max(peak, active)
            calls += 1
            entered.set()
            try:
                assert release.wait(3)
                return native(*args)
            finally:
                active -= 1
        monkeypatch.setattr(driver, '_invoke_history', held)
        try:
            await until(entered.is_set)
            source = memory_source(now=clock.now)
            await owner.request(source.conversation_id, ACTOR)
            await owner.reconcile()
            await until(lambda: len(driver._operations) > 1)
            assert peak == 1 and driver._disk.status()['operations'] == 1
        finally:
            release.set()
            await owner.begin_close()
        assert peak == 1 and active == 0 and calls > 1
