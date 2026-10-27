"""Actual collector queue, file intents, expiry and native close under one authority."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import threading
from unittest.mock import AsyncMock
from uuid import uuid4
from weakref import ref

import numpy as np
import pytest

from src.ambient.textual_wake import TextualWakeDetector
from src.core.owned_work import WorkCloseTimeout
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.pipeline.capture_room import CaptureRoom
from src.wakeword.clip_store import build_wake_clip_store, clip_sources
from tests.mocks.wake_clips import graph, until

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')


async def seed(g, *, source=None):
    source = source or g.source()
    assert g.writer.submit('synthetic-room', .8, np.ones(1600, dtype=np.float32)*.2, sources=(source,))
    await g.idle()
    return source


def gate_encoder(g, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    encode = g.writer._encode
    def blocked(job):
        entered.set()
        assert release.wait(4)
        encode(job)
    monkeypatch.setattr(g.writer, '_encode', blocked)
    return entered, release


async def test_unsourced_or_expired_inputs_cannot_create_files(tmp_path):
    async with graph(tmp_path) as g:
        assert not g.writer.submit('synthetic-room', .5, np.ones(160))
        old = replace(g.source(), captured_at=g.clock.now-2, expires_at=g.clock.now-1)
        assert not g.writer.submit_missed('synthetic-room', -1., np.ones(160), sources=(old,))
        assert not await g.rows() and not Path(g.target.root).exists()


@pytest.mark.parametrize('bucket', ['accepted', 'rejected', 'missed'])
async def test_idle_background_ttl_removes_wav_and_all_metadata_in_every_bucket(tmp_path, bucket):
    async with graph(tmp_path, background=True) as g:
        source = g.source(ttl=2)
        if bucket == 'missed':
            assert g.writer.submit_missed('synthetic-room', -1., np.ones(160), sources=(source,))
        else:
            assert g.writer.submit('synthetic-room', .5, np.ones(160), accepted=bucket=='accepted', sources=(source,))
        await g.idle()
        assert len(g.files()) == 1
        g.clock.now += 2
        await until(lambda: not g.files())
        assert not await g.rows()


async def test_revocation_removes_existing_clip_without_another_submission(tmp_path):
    async with graph(tmp_path, background=True) as g:
        source = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        await until(lambda: not g.files())
        assert not await g.rows()
        assert not g.writer.submit('synthetic-room', .9, np.ones(160), sources=(source,))


async def test_mixed_sources_keep_oldest_deadline_and_original_ids(tmp_path):
    async with graph(tmp_path) as g:
        first, second = g.source(ttl=2), g.source(ttl=20)
        assert g.writer.submit('synthetic-room', .5, np.ones(160), sources=(first,second))
        await g.idle()
        row, = await g.rows()
        assert clip_sources(row['sources_json']) == (first,second)
        assert row['expires_at'] == first.expires_at
        g.clock.now += 2
        await g.store.reconcile()
        assert not g.files() and not await g.rows()
        assert g.privacy.sources_available((second,))


async def test_expired_queued_wave_is_freed_while_native_encoder_remains_owned(tmp_path, monkeypatch):
    async with graph(tmp_path, background=True) as g:
        entered, release = gate_encoder(g, monkeypatch)
        source = g.source(ttl=1)
        try:
            assert g.writer.submit('synthetic-room', .5, np.ones(160), sources=(source,))
            await until(entered.is_set)
            assert g.writer.submit('synthetic-room', .6, np.ones(160), sources=(source,))
            queued = ref(g.writer._queue[0].audio)
            native = ref(g.writer._busy.audio)
            g.clock.now += 1
            await until(lambda: not g.writer.resource_status()['queued'])
            assert queued() is None and native() is not None
            assert g.writer.resource_status()['busy']
        finally:
            release.set()
            await g.idle()
        assert native() is None
        assert not g.files() and not await g.rows()


async def test_native_encoder_exit_keeps_writer_lease_after_close_timeout(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        entered, release = gate_encoder(g, monkeypatch)
        try:
            assert g.writer.submit('synthetic-room', .5, np.ones(160), sources=(g.source(),))
            await until(entered.is_set)
            for close in (g.writer.aclose, g.store.aclose, g.privacy.aclose):
                with pytest.raises(WorkCloseTimeout):
                    await close()
            assert g.privacy.resource_status()['writer_lease']
            assert g.writer.resource_status()['native_running']
        finally:
            release.set()
            await g.privacy.begin_close()
        assert not g.privacy.resource_status()['writer_lease']
        assert not g.writer.resource_status()['native_running'] and not g.files()


async def test_revocation_before_publish_cannot_repopulate_after_cleanup(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        entered, release = threading.Event(), threading.Event()
        publish = g.store.publish
        def blocked(*args):
            entered.set()
            assert release.wait(4)
            return publish(*args)
        monkeypatch.setattr(g.store, 'publish', blocked)
        source = g.source()
        try:
            assert g.writer.submit('synthetic-room', .5, np.ones(160), sources=(source,))
            await until(entered.is_set)
            assert len(list((Path(g.target.root)/'policy_v1').glob('*.part'))) == 1
            await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
            await g.store.reconcile()
            assert len(await g.rows()) == 1  # Locator is kept while its real writer is inside native work.
        finally:
            release.set()
            await g.idle()
        assert not g.files() and not await g.rows()
        assert g.writer.resource_status()['written'] == 0


async def test_cleanup_observer_cancel_does_not_cancel_actual_unlink_or_lease(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        source = await seed(g)
        await g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR)
        entered, release = threading.Event(), threading.Event()
        unlink = g.store._unlink
        def blocked(*args):
            entered.set()
            assert release.wait(4)
            return unlink(*args)
        monkeypatch.setattr(g.store, '_unlink', blocked)
        observing = asyncio.create_task(g.store.reconcile())
        try:
            await until(entered.is_set)
            observing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observing
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease'] and g.store.resource_status()['operations']
        finally:
            release.set()
            await g.privacy.begin_close()
        assert not g.files()
        with sqlite3.connect(tmp_path/'privacy.db') as db:
            assert db.execute('SELECT COUNT(*) FROM wake_clips_v1').fetchone()[0] == 0


async def test_failed_unlink_retains_locator_then_retries_without_private_logs(tmp_path, monkeypatch, caplog):
    async with graph(tmp_path) as g:
        source = await seed(g)
        g.clock.now = source.expires_at
        unlink = g.store._unlink
        def fail(*args):
            raise OSError('synthetic private path and payload')
        monkeypatch.setattr(g.store, '_unlink', fail)
        await g.store.reconcile()
        assert len(g.files()) == len(await g.rows()) == 1
        assert g.store.resource_status()['state'] == 'unavailable'
        assert 'synthetic private path and payload' not in caplog.text
        monkeypatch.setattr(g.store, '_unlink', unlink)
        await g.store.reconcile()
        assert not g.files() and not await g.rows()


async def test_replaced_root_cannot_redirect_encoder_or_cleanup(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        source = await seed(g)
        root, moved, outside = Path(g.target.root), tmp_path/'moved', tmp_path/'outside'
        outside.mkdir()
        sentinel = outside/'synthetic-sentinel'
        sentinel.write_bytes(b'synthetic')
        root.rename(moved)
        root.symlink_to(outside, target_is_directory=True)
        try:
            assert g.writer.submit('synthetic-room', .6, np.ones(160), sources=(g.source(),))
            await g.idle()
            g.clock.now = source.expires_at
            await g.store.reconcile()
            assert g.store.resource_status()['state'] == 'unavailable'
            assert sentinel.read_bytes() == b'synthetic'
            assert list(outside.iterdir()) == [sentinel]
        finally:
            root.unlink()
            moved.rename(root)
        await g.store.reconcile()
        assert not g.files() and not await g.rows()


async def test_cleanup_unlinks_own_symlink_without_following_its_target(tmp_path):
    async with graph(tmp_path) as g:
        source = await seed(g)
        clip, = g.files()
        outside = tmp_path/'synthetic-sentinel.wav'
        outside.write_bytes(b'synthetic untouched')
        clip.unlink()
        clip.symlink_to(outside)
        g.clock.now = source.expires_at
        await g.store.reconcile()
        assert outside.read_bytes() == b'synthetic untouched'
        assert not clip.is_symlink() and not await g.rows()


async def test_disabled_feature_restart_retires_registered_historical_root(tmp_path):
    async with graph(tmp_path) as g:
        source = await seed(g)
        clock = g.clock
        root = g.target.root
    clock.now = source.expires_at
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock)
        await privacy.acall(privacy.initialize)
        stack.push_async_callback(privacy.aclose)
        store = await build_wake_clip_store({'rooms': {'wake_word': {'capture_clips': {'enabled': False, 'dir': str(tmp_path/'new-disabled-root')}}}}, privacy=privacy, shutdown=stack)
        await until(lambda: not list((Path(root)/'policy_v1').glob('*.wav')))
        assert not (tmp_path/'new-disabled-root').exists()
        assert store.resource_status()['writers'] == 0


@pytest.mark.parametrize('encoded', ['[]', '{}', 'not JSON', '[{}]'])
async def test_unknown_source_metadata_blocks_cleanup_and_preserves_locator(tmp_path, encoded):
    async with graph(tmp_path) as g:
        await seed(g)
        def corrupt():
            with g.privacy.ledger._transaction() as db:
                db.execute('UPDATE wake_clips_v1 SET sources_json=?', (encoded,))
        await g.store._work.run(corrupt)
        await g.store.reconcile()
        assert g.store.resource_status()['state'] == 'unavailable'
        assert len(g.files()) == len(await g.rows()) == 1


@pytest.mark.parametrize('sequence', [0,-1])
async def test_nonpositive_sequence_cannot_be_silently_skipped(tmp_path, sequence):
    async with graph(tmp_path) as g:
        await seed(g)
        def corrupt():
            with g.privacy.ledger._transaction() as db:
                db.execute('UPDATE wake_clips_v1 SET sequence=?', (sequence,))
        await g.store._work.run(corrupt)
        await g.store.reconcile()
        assert g.store.resource_status()['state'] == 'unavailable'
        assert len(g.files()) == 1


async def test_each_physical_sweep_scans_at_most_128_metadata_rows(tmp_path):
    async with graph(tmp_path) as g:
        source = await seed(g)
        def clone():
            with g.privacy.ledger._transaction() as db:
                row = dict(db.execute('SELECT * FROM wake_clips_v1').fetchone())
                columns = [name for name in row if name != 'sequence']
                for _ in range(129):
                    row['id'] = str(uuid4())
                    db.execute(f"INSERT INTO wake_clips_v1 ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", tuple(row[name] for name in columns))
        await g.store._work.run(clone)
        g.clock.now = source.expires_at
        await g.store.reconcile()
        assert len(await g.rows()) == 2 and g.store.resource_status()['state'] == 'pending'
        await g.store.reconcile()
        assert not await g.rows() and not g.files()


async def test_real_native_capture_actor_passes_its_source_without_minting_another(tmp_path):
    async with graph(tmp_path) as g:
        def process(audio, frames, timestamp, status):
            assert g.writer.submit('synthetic-room', .8, audio[:,0])
        room = CaptureRoom('synthetic-room', process=process, poll=lambda: None, clear=lambda: None,
                           listening=lambda: False, privacy=g.privacy)
        g.privacy.add_borrower(room.begin_close)
        room.push(g.clock.now, np.ones((1280,1), dtype=np.float32), 1280)
        await room.step()
        source = room.source
        await g.idle()
        row, = await g.rows()
        assert clip_sources(row['sources_json']) == (source,)
        assert row['expires_at'] == source.expires_at


async def test_textual_wake_clip_inherits_the_exact_ambient_source(tmp_path):
    async with graph(tmp_path) as g:
        source = g.source(ttl=5)
        detector = TextualWakeDetector(dispatch_fn=AsyncMock(return_value={}),
            last_acoustic_command_ts_fn=lambda _: 0.,
            on_missed_fn=lambda room, audio: g.writer.submit_missed(room, -1., audio))
        assert await detector.maybe_dispatch('synthetic-room', 'Nexa prendé la luz', 'human_direct',
                                             None, np.ones(1600), vad_prob=.9, memory_source=source)
        await g.idle()
        row, = await g.rows()
        assert row['bucket'] == 'missed' and clip_sources(row['sources_json']) == (source,)
        assert row['expires_at'] == source.expires_at


async def test_local_wake_cleanup_does_not_claim_all_collectors_erased(tmp_path):
    async with graph(tmp_path) as g:
        source = await seed(g)
        owner = ErasureCoordinator(g.privacy)
        receipt = await owner.request(source.conversation_id, ACTOR)
        await g.store.reconcile()
        await owner.reconcile()
        value = await g.privacy.acall(g.privacy.ledger.erasures.snapshot, receipt['id'])
        assert not g.files() and not await g.rows()
        assert value['state'] == 'blocked'
        assert all(r['state'] != 'complete' for r in value['steps'] if r['participant'] == 'collectors')
        assert g.store.resource_status()['scope'] == 'managed_wake_clips_only'


async def test_encode_errors_do_not_retain_audio_or_copy_payload_to_logs(tmp_path, monkeypatch, caplog):
    async with graph(tmp_path) as g:
        waves = []
        def fail(job):
            waves.append(ref(job.audio))
            raise RuntimeError('synthetic private transcript and path')
        monkeypatch.setattr(g.writer, '_encode', fail)
        await seed(g)
        assert all(w() is None for w in waves)
        assert not g.files() and not await g.rows()
        assert 'synthetic private transcript and path' not in caplog.text
        status = json.dumps(g.writer.resource_status()) + json.dumps(g.store.resource_status())
        assert 'synthetic-room' not in status and str(tmp_path) not in status


async def test_observed_native_expiry_cannot_be_undone_by_clock_rollback(tmp_path, monkeypatch):
    async with graph(tmp_path, background=True) as g:
        entered, release = gate_encoder(g, monkeypatch)
        source = g.source(ttl=1)
        try:
            assert g.writer.submit('synthetic-room', .5, np.ones(160), sources=(source,))
            await until(entered.is_set)
            g.clock.now = source.expires_at
            await until(lambda: g.writer._busy.cancelled.is_set())
            g.clock.now = source.captured_at
        finally:
            release.set()
            await g.idle()
        assert not g.files() and not await g.rows()
        assert not g.writer.resource_status()['written']


async def test_file_commit_admission_precedes_tombstone_and_later_physical_cleanup(tmp_path, monkeypatch):
    import src.wakeword.clip_store as module
    async with graph(tmp_path) as g:
        entered, release = threading.Event(), threading.Event()
        rename = module.os.replace
        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(4)
            return rename(*args, **kwargs)
        monkeypatch.setattr(module.os, 'replace', blocked)
        source = g.source()
        deleting = None
        try:
            assert g.writer.submit('synthetic-room', .5, np.ones(160), sources=(source,))
            await until(entered.is_set)
            deleting = asyncio.create_task(g.privacy.acall(g.privacy.ledger.delete_conversation, source.conversation_id, ACTOR))
            await asyncio.sleep(.02)
            assert not deleting.done()
        finally:
            release.set()
            await g.idle()
            if deleting is not None:
                await deleting
        assert g.writer.resource_status()['written'] == 1
        await g.store.reconcile()
        assert not g.files() and not await g.rows()


async def test_private_storage_and_writer_join_do_not_need_the_default_executor(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)
    asyncio.get_running_loop().set_default_executor(pool)
    entered, release = threading.Event(), threading.Event()
    def block():
        entered.set()
        assert release.wait(4)
    blocked = asyncio.create_task(asyncio.to_thread(block))
    try:
        await until(entered.is_set)
        async with graph(tmp_path) as g:
            try:
                await seed(g)
                await g.store.reconcile()
                await g.writer.aclose()
                await g.store.aclose()
                assert not blocked.done() and not g.writer.resource_status()['native_running']
                assert all(not thread.is_alive() for thread in g.store._executor._threads)
            finally:
                release.set()  # Authority's final connection/lease shutdown may use the general pool.
    finally:
        release.set()
        await blocked


async def test_orphaned_part_intent_is_reconciled_after_leased_restart(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        discard = g.store.discard
        def encode_failure(job):
            g.writer.__class__._encode(g.writer, job)
            raise OSError('synthetic interrupted encode')
        def cannot_unlink(*args):
            raise OSError('synthetic interrupted cleanup')
        monkeypatch.setattr(g.writer, '_encode', encode_failure)
        monkeypatch.setattr(g.store, 'discard', cannot_unlink)
        await seed(g)
        row, = await g.rows()
        assert not row['published']
        assert len(list((Path(g.target.root)/'policy_v1').glob('*.part'))) == 1
        monkeypatch.setattr(g.store, 'discard', discard)
        clock = g.clock
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock)
        await privacy.acall(privacy.initialize)
        stack.push_async_callback(privacy.aclose)
        await build_wake_clip_store({'rooms': {'wake_word': {'capture_clips': {'enabled': False, 'dir': str(tmp_path/'captured')}}}},
                                   privacy=privacy, shutdown=stack)
        await until(lambda: not list((tmp_path/'captured'/'policy_v1').glob('*.part')))


async def test_blocked_diagnostic_sink_cannot_keep_the_failed_waveform(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.wakeword.wake_clip_writer as module
    async with graph(tmp_path) as g:
        waves = []
        entered, release = threading.Event(), threading.Event()
        def fail(job):
            waves.append(ref(job.audio))
            raise RuntimeError('synthetic private exception')
        def log(message, error_type):
            assert error_type == 'RuntimeError' and 'synthetic private' not in message
            entered.set()
            assert release.wait(4)
        monkeypatch.setattr(g.writer, '_encode', fail)
        monkeypatch.setattr(module, 'logger', SimpleNamespace(warning=log))
        try:
            assert g.writer.submit('synthetic-room', .5, np.ones(160), sources=(g.source(),))
            await until(entered.is_set)
            assert waves and all(wave() is None for wave in waves)
            assert not g.writer.resource_status()['busy']
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
        finally:
            release.set()
            await g.privacy.begin_close()
