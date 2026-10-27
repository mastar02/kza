"""Numeric diagnostics expire without new input or deadline renewal on restart."""

from datetime import datetime, timezone
import asyncio
import json
from pathlib import Path
import sqlite3
import threading

import pytest

from src.obs.metrics_emitter import MetricsEmitter
from src.core.owned_work import WorkCloseTimeout
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_diagnostic_metric_privacy import request, receiver
from tests.mocks.ambient_graph import Clock

pytestmark = pytest.mark.asyncio
DAY = 86400


def rows(emitter):
    result = []
    for path in (emitter.path, emitter.path.with_suffix(emitter.path.suffix+'.1')):
        if path.exists():
            result.extend(json.loads(line) for line in path.read_text().splitlines())
    return result


async def test_active_and_rotated_files_expire_without_another_metric(tmp_path):
    clock = Clock(2_000_000_000.)
    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'), clock=clock,
        retention_days=1, rotate_bytes=400)
    try:
        await emitter.start()
        for _ in range(6):
            request(emitter)
            await emitter.flush()
        assert rows(emitter)
        assert all(row['recorded_at'] == clock.now and row['expires_at'] == clock.now+DAY for row in rows(emitter))
        clock.now += DAY
        await until(lambda: not rows(emitter) and emitter.resource_status()['retention']['rows'] == 0
                    and emitter.resource_status()['retention']['ready'])
        assert emitter.resource_status()['retention']['rows'] == 0
    finally:
        await emitter.aclose()


async def test_queued_metric_deadline_precedes_native_writer_and_tcp(tmp_path, monkeypatch):
    import threading
    clock = Clock(2_000_000_000.)
    async with receiver() as (port, received):
        emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'), clock=clock,
            retention_days=1, logstash_host='127.0.0.1', logstash_port=port)
        entered, release = threading.Event(), threading.Event()
        write = emitter._write

        def held(record):
            entered.set()
            assert release.wait(3)
            write(record)

        monkeypatch.setattr(emitter, '_write', held)
        try:
            request(emitter)
            await until(entered.is_set)
            clock.now += 2*DAY
        finally:
            release.set()
            await emitter.flush()
        try:
            assert not rows(emitter) and not received
        finally:
            await emitter.aclose()


async def test_shorter_policy_survives_failed_projection_and_later_widening(tmp_path, monkeypatch):
    clock = Clock(2_000_000_000.)
    path = str(tmp_path/'metrics.jsonl')
    first = MetricsEmitter(path, clock=clock, retention_days=90)
    request(first)
    await first.flush()
    born = rows(first)[0]['recorded_at']
    await first.aclose()
    clock.now += 2*DAY
    short = MetricsEmitter(path, clock=clock, retention_days=1)

    def failed_publish(*args):
        raise OSError('synthetic_projection_failure')

    monkeypatch.setattr(short._store, '_publish', failed_publish)
    try:
        with pytest.raises(RuntimeError, match='diagnostic_retention_failed'):
            await short.aprune()
        assert not short.resource_status()['retention']['ready']
    finally:
        await short.aclose()
    # Even a backwards wall clock cannot reimport a deleted cache row.
    clock.now = born+DAY/2
    reopened = MetricsEmitter(path, clock=clock, retention_days=90)
    try:
        await reopened.start()
        assert not rows(reopened)
        clock.now += 2*DAY
        request(reopened)
        await reopened.flush()
        assert rows(reopened)[0]['expires_at'] == clock.now+90*DAY
    finally:
        await reopened.aclose()


async def test_previous_numeric_schema_uses_original_timestamp_not_import_time(tmp_path):
    clock = Clock(2_000_000_000.)
    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'), clock=clock, retention_days=1)
    old = {'event_type': 'request', 'diagnostic_schema': 1, 'service': 'kza-voice',
        'success': True, '@timestamp': datetime.fromtimestamp(clock.now-2*DAY, timezone.utc).isoformat()}
    emitter.path.write_text(json.dumps(old)+'\n')
    try:
        await emitter.start()
        assert not rows(emitter)
    finally:
        await emitter.aclose()


async def test_unverified_policy_file_is_preserved_in_quarantine_and_never_sent(tmp_path):
    clock = Clock(2_000_000_000.)
    async with receiver() as (port, received):
        emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'), clock=clock,
            logstash_host='127.0.0.1', logstash_port=port)
        original = json.dumps({'event_type': 'request', 'diagnostic_schema': 1,
            'text': 'Synthetic private unverified log'})+'\n'
        emitter.path.write_text(original)
        try:
            await emitter.start()
            assert not rows(emitter) and not received
            quarantines = list(tmp_path.glob('metrics_policy_v1.jsonl.quarantine-*'))
            assert len(quarantines) == 1 and quarantines[0].read_text() == original
            assert emitter.resource_status()['retention']['quarantines'] == 1
        finally:
            await emitter.aclose()


async def test_second_writer_cannot_reinterpret_or_republish_the_live_catalog(tmp_path):
    clock = Clock(2_000_000_000.)
    path = str(tmp_path/'metrics.jsonl')
    first = MetricsEmitter(path, clock=clock)
    second = MetricsEmitter(path, clock=clock)
    try:
        await first.start()
        with pytest.raises(RuntimeError, match='diagnostic_writer_busy'):
            await second.aprune()
    finally:
        await second.aclose()
        await first.aclose()


async def test_expired_record_is_not_sent_after_connect_finishes(tmp_path, monkeypatch):
    clock = Clock(2_000_000_000.)
    entered, release = threading.Event(), threading.Event()
    sent = []

    class Connection:
        def settimeout(self, timeout):
            pass

        def sendall(self, line):
            sent.append(line)

        def close(self):
            pass

    def connect(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return Connection()

    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'), clock=clock,
        retention_days=1, logstash_host='synthetic-only')
    monkeypatch.setattr('src.obs.metrics_emitter.socket.create_connection', connect)
    try:
        request(emitter)
        await until(entered.is_set)
        clock.now += DAY
    finally:
        release.set()
        await emitter.flush()
        await emitter.aclose()
    assert not sent


@pytest.mark.parametrize('observer', ['waiting', 'cancelled'])
async def test_blocked_prune_retains_writer_lease_through_close_timeout(tmp_path, monkeypatch, observer):
    clock = Clock(2_000_000_000.)
    path = str(tmp_path/'metrics.jsonl')
    emitter = MetricsEmitter(path, clock=clock, retention_days=1, close_timeout_s=.03)
    request(emitter)
    await emitter.flush()
    # Finish startup maintenance before installing the gate for this exact sweep.
    await emitter.aprune()
    entered, release = threading.Event(), threading.Event()
    publish = emitter._store._publish

    def held_publish(db):
        entered.set()
        assert release.wait(3)
        return publish(db)

    monkeypatch.setattr(emitter._store, '_publish', held_publish)
    pruning = asyncio.create_task(emitter.aprune())
    second = MetricsEmitter(path, clock=clock)
    try:
        await until(entered.is_set)
        if observer == 'cancelled':
            pruning.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pruning
        with pytest.raises(WorkCloseTimeout):
            await emitter.aclose()
        with pytest.raises(RuntimeError, match='diagnostic_writer_busy'):
            await second.aprune()
        assert not emitter.begin_close().done()
    finally:
        release.set()
        await asyncio.gather(pruning, return_exceptions=True)
        await emitter.begin_close()
        await second.aclose()
    third = MetricsEmitter(path, clock=clock)
    try:
        await third.start()
        assert third.resource_status()['retention']['ready']
    finally:
        await third.aclose()


async def test_failed_projection_stage_is_retired_on_retry_without_restart(tmp_path, monkeypatch):
    import src.obs.diagnostic_store as module
    clock = Clock(2_000_000_000.)
    path = str(tmp_path/'metrics.jsonl')
    first = MetricsEmitter(path, clock=clock, retention_days=90)
    request(first)
    await first.flush()
    await first.aclose()
    clock.now += DAY/2
    short = MetricsEmitter(path, clock=clock, retention_days=1)
    unlink = Path.unlink

    def fail_replace(*args):
        raise OSError('synthetic_replace_failure')

    def fail_stage_unlink(target, *args, **kwargs):
        if target.name.startswith('.diag-'):
            raise OSError('synthetic_stage_unlink_failure')
        return unlink(target, *args, **kwargs)

    try:
        with monkeypatch.context() as faults:
            faults.setattr(module.os, 'replace', fail_replace)
            faults.setattr(Path, 'unlink', fail_stage_unlink)
            with pytest.raises(RuntimeError, match='diagnostic_retention_failed'):
                await short.aprune()
        assert list(tmp_path.glob('.diag-*.tmp'))
        clock.now += 2*DAY
        await short.aprune()
        assert not rows(short)
        assert not list(tmp_path.glob('.diag-*.tmp'))
        assert short.resource_status()['retention']['ready']
    finally:
        await short.aclose()


async def test_valid_previous_numeric_rows_keep_birth_deadline_and_identity_without_tcp_replay(tmp_path):
    clock = Clock(2_000_000_000.)
    born = clock.now-DAY/2
    async with receiver() as (port, received):
        path = str(tmp_path/'metrics.jsonl')
        emitter = MetricsEmitter(path, clock=clock, retention_days=1,
            logstash_host='127.0.0.1', logstash_port=port)
        emitter.path.write_text(json.dumps({'event_type': 'wake', 'diagnostic_schema': 1,
            'matched': False, 'service': 'kza-voice',
            '@timestamp': datetime.fromtimestamp(born, timezone.utc).isoformat()})+'\n')
        await emitter.start()
        original = rows(emitter)[0]
        assert original['recorded_at'] == born and original['expires_at'] == born+DAY
        await emitter.aclose()
        clock.now += DAY/4
        wider = MetricsEmitter(path, clock=clock, retention_days=90,
            logstash_host='127.0.0.1', logstash_port=port)
        try:
            await wider.start()
            assert rows(wider)[0] == original
            assert not received
        finally:
            await wider.aclose()


async def test_deadline_is_rechecked_after_native_encoding_wait(tmp_path, monkeypatch):
    clock = Clock(2_000_000_000.)
    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'), clock=clock, retention_days=1)
    entered, release = threading.Event(), threading.Event()
    encode = emitter._store._line

    def held_encode(*args):
        entered.set()
        assert release.wait(3)
        return encode(*args)

    monkeypatch.setattr(emitter._store, '_line', held_encode)
    try:
        request(emitter)
        await until(entered.is_set)
        clock.now += 2*DAY
    finally:
        release.set()
        await emitter.flush()
    try:
        assert not rows(emitter)
        assert emitter.resource_status()['retention']['rows'] == 0
    finally:
        await emitter.aclose()


async def test_unrecognized_catalog_is_not_reinterpreted_or_modified(tmp_path):
    clock = Clock(2_000_000_000.)
    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'), clock=clock)
    with sqlite3.connect(emitter._store.catalog) as db:
        db.execute('CREATE TABLE unrelated(value TEXT)')
        db.execute('INSERT INTO unrelated VALUES (?)', ('Synthetic unverified data',))
    original = emitter._store.catalog.read_bytes()
    try:
        with pytest.raises(RuntimeError, match='unsupported_diagnostic_catalog'):
            await emitter.aprune()
        assert emitter._store.catalog.read_bytes() == original
        assert not emitter.path.exists()
    finally:
        await emitter.aclose()


async def test_retained_error_does_not_keep_native_unverified_payload_or_message(tmp_path, monkeypatch):
    from weakref import ref
    import traceback

    class Payload:
        pass

    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'), clock=Clock(2_000_000_000.))
    emitter.path.write_text('{"event_type":"request","text":"Synthetic private log"}\n')
    references = []

    def failed_decode(*args):
        payload = Payload()
        references.append(ref(payload))
        raise OSError('Synthetic private error', payload)

    monkeypatch.setattr(emitter._store, '_decode', failed_decode)
    try:
        with pytest.raises(RuntimeError, match='diagnostic_retention_failed') as retained:
            await emitter.aprune()
        assert references and references[0]() is None
        assert retained.value.__context__ is None
        formatted = ''.join(traceback.TracebackException.from_exception(retained.value).format())
        assert 'Synthetic private error' not in formatted
    finally:
        await emitter.aclose()


async def test_startup_timeout_does_not_orphan_initial_cleanup_or_block_forever(tmp_path, monkeypatch):
    emitter = MetricsEmitter(str(tmp_path/'metrics.jsonl'),
        clock=Clock(2_000_000_000.), close_timeout_s=.03)
    entered, release = threading.Event(), threading.Event()
    publish = emitter._store._publish

    def held(db):
        entered.set()
        assert release.wait(3)
        publish(db)

    monkeypatch.setattr(emitter._store, '_publish', held)
    starting = asyncio.create_task(emitter.start())
    try:
        await until(entered.is_set)
        with pytest.raises(WorkCloseTimeout, match='diagnostic_initialization_pending'):
            await starting
        assert not emitter._sweep.done()
        assert emitter.resource_status()['retention']['writer_lease']
    finally:
        release.set()
        await asyncio.gather(starting, return_exceptions=True)
        await emitter.aprune()
        await emitter.aclose()
