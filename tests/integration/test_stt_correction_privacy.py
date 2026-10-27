"""Real STT correction writes, revocable views, expiry and retained native scopes."""

import asyncio
from contextlib import AsyncExitStack
import json
import sqlite3
import threading
from unittest.mock import patch
from weakref import ref

import numpy as np
import pytest

from src.core.owned_work import WorkCloseTimeout
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices
from src.memory.producer_context import producer_context
from src.training.nightly_trainer import NightlyConfig, NightlyTrainer
from src.training.stt_correction_collector import STTCorrection, build_stt_correction_collector
from src.wakeword.clip_store import build_wake_clip_store, clip_sources
from tests.mocks.stt_corrections import graph
from tests.mocks.wake_clips import until

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')


def block_encoder(g, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    entered.calls = 0
    encode = g.collector._encode
    def blocked(job):
        entered.calls += 1
        entered.set()
        assert release.wait(5)
        encode(job)
    monkeypatch.setattr(g.collector, '_encode', blocked)
    return entered, release


def files(g):
    return list((g.collector.data_dir/'policy_v1').glob('*'))


async def test_both_original_and_feedback_sources_are_required(tmp_path):
    async with graph(tmp_path) as g:
        original, feedback = g.sources()
        args = dict(audio=np.ones(1600), original_text='synthetic original', corrected_text='synthetic correction')
        assert await g.collector.add_correction(**args) is None
        assert await g.collector.add_correction(**args, sources=(original,)) is None
        assert await g.collector.add_correction(**args, correction_sources=(feedback,)) is None
        assert not files(g) and not await g.rows()
        with producer_context('synthetic-room', (feedback,)):
            assert await g.collector.add_correction(**args, sources=(original,))
        row, = await g.rows()
        assert clip_sources(row['sources_json']) == (original, feedback)


async def test_feedback_deadline_is_not_reset_when_persisting_a_correction(tmp_path):
    async with graph(tmp_path) as g:
        original, feedback = g.sources(original_ttl=20, feedback_ttl=2)
        g.clock.now += 1
        assert await g.add(sources=(original, feedback))
        row, = await g.rows()
        assert row['captured_at'] == min(original.captured_at, feedback.captured_at)
        assert row['expires_at'] == feedback.expires_at


@pytest.mark.parametrize('which', ['original', 'feedback'])
async def test_deleting_either_conversation_clears_wav_text_and_existing_view(tmp_path, which):
    async with graph(tmp_path, background=True) as g:
        original, feedback = g.sources()
        assert await g.add(sources=(original, feedback))
        view, = await g.collector.get_training_pairs()
        erased = original if which == 'original' else feedback
        await g.privacy.acall(g.privacy.ledger.delete_conversation, erased.conversation_id, ACTOR)
        assert not view.available and view.original_text == view.corrected_text == view.audio_path == ''
        assert not await g.collector.get_training_pairs()
        await until(lambda: not files(g))
        assert not await g.rows()


async def test_reads_enforce_deadline_without_maintenance_or_another_capture(tmp_path):
    async with graph(tmp_path) as g:
        sources = g.sources(original_ttl=2)
        assert await g.add(sources=sources)
        view, = await g.collector.get_training_pairs()
        assert view.corrected_text
        g.clock.now = sources[0].expires_at
        assert view.corrected_text == view.original_text == view.audio_path == ''
        assert view.user_id is None and not view.available
        assert not await g.collector.get_training_pairs()
        assert await g.collector.get_corrections_count() == 0
        assert files(g)  # Read fencing does not falsely claim physical cleanup.
        await g.store.reconcile()
        assert not files(g) and not await g.rows()
        g.clock.now = sources[0].captured_at
        assert not view.available and view.corrected_text == ''


async def test_background_expiry_clears_unused_views_and_cascades_metadata(tmp_path):
    async with graph(tmp_path, background=True) as g:
        sources = g.sources(original_ttl=1)
        assert await g.add(sources=sources)
        view, = await g.collector.get_training_pairs()
        g.clock.now = sources[0].expires_at
        await until(view._retired.is_set)
        assert view.__dict__['corrected_text'] == ''
        await until(lambda: not files(g))
        assert not await g.rows()


async def test_native_write_survives_cancelled_observer_and_retains_authority_lease(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        entered, release = block_encoder(g, monkeypatch)
        adding = asyncio.create_task(g.add())
        try:
            await until(entered.is_set)
            row, = await g.rows()
            assert not row['published'] and row['original_text']
            adding.cancel()
            with pytest.raises(asyncio.CancelledError):
                await adding
            for close in (g.collector.aclose, g.privacy.aclose):
                with pytest.raises(WorkCloseTimeout):
                    await close()
            assert g.privacy.resource_status()['writer_lease']
            assert g.collector.resource_status()['active_writes'] == 1
        finally:
            release.set()
            await g.privacy.begin_close()
        assert not g.privacy.resource_status()['writer_lease'] and not files(g)
        with sqlite3.connect(tmp_path/'privacy.db') as db:
            assert db.execute('SELECT COUNT(*) FROM stt_corrections_v1').fetchone()[0] == 0


async def test_expired_executor_queue_releases_audio_before_blocked_encoders_exit(tmp_path, monkeypatch):
    async with graph(tmp_path, background=True) as g:
        entered, release = block_encoder(g, monkeypatch)
        sources = g.sources(original_ttl=1)
        adding = [asyncio.create_task(g.add(sources=sources)) for _ in range(8)]
        try:
            await until(lambda: len(g.collector._jobs) == 8 and entered.calls == 2)
            # Both encoders occupy the private pool. Wait until physical
            # maintenance is actually queued there before advancing source TTL.
            await until(lambda: g.store._sweep is not None and not g.store._sweep.done()
                        and g.store._work.status()['operations'] > 0)
            assert await g.add() is None
            queued = [job for job in g.collector._jobs.values() if job.id not in g.collector._active]
            waves = [ref(job.audio) for job in queued]
            assert len(queued) == 6
            g.clock.now = sources[0].expires_at
            await until(lambda: all(job.audio is None for job in queued))
            assert all(wave() is None for wave in waves)
            assert all(job.original_text == job.corrected_text == '' for job in queued)
            assert g.collector.resource_status()['active_writes'] == 2
            g.clock.now = sources[0].captured_at
        finally:
            release.set()
            await asyncio.gather(*adding)
        assert not files(g) and not await g.rows()


async def test_copy_is_independent_of_the_callers_audio_buffer(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        import wave
        entered, release = block_encoder(g, monkeypatch)
        audio = np.ones(1600,dtype=np.float32)*.5
        adding = asyncio.create_task(g.add(audio=audio))
        try:
            await until(entered.is_set)
            audio[:] = 0
        finally:
            release.set()
            identifier = await adding
        with wave.open(str(g.collector.data_dir/'policy_v1'/(identifier+'.wav')), 'rb') as wav:
            assert np.frombuffer(wav.readframes(1600), dtype=np.int16).max() > 16000


async def test_revocation_after_encoding_before_publish_cannot_refill_erased_text(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        entered, release = threading.Event(), threading.Event()
        publish = g.store.publish
        def blocked(*args):
            entered.set()
            assert release.wait(5)
            return publish(*args)
        monkeypatch.setattr(g.store, 'publish', blocked)
        sources = g.sources()
        adding = asyncio.create_task(g.add(sources=sources))
        try:
            await until(entered.is_set)
            await g.privacy.acall(g.privacy.ledger.delete_conversation, sources[1].conversation_id, ACTOR)
            await g.store.reconcile()
            assert len(await g.rows()) == 1  # Actual native writer still owns its locator.
        finally:
            release.set()
            assert await adding is None
        assert not files(g) and not await g.rows()


async def test_failed_unlink_keeps_text_locator_for_retry_and_blocks_reads(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        sources = g.sources()
        assert await g.add(sources=sources)
        await g.privacy.acall(g.privacy.ledger.delete_conversation, sources[0].conversation_id, ACTOR)
        unlink = g.store._unlink
        def fail(*args):
            raise OSError('synthetic private path')
        monkeypatch.setattr(g.store, '_unlink', fail)
        await g.store.reconcile()
        assert files(g) and len(await g.rows()) == 1
        assert not await g.collector.get_training_pairs()
        assert g.store.resource_status()['state'] == 'unavailable'
        monkeypatch.setattr(g.store, '_unlink', unlink)
        await g.store.reconcile()
        assert not await g.rows() and not files(g)


async def test_legacy_json_and_wav_are_quarantined_from_reads_and_cleanup(tmp_path):
    async with graph(tmp_path) as g:
        legacy = g.collector.data_dir/'synthetic_legacy'
        legacy.mkdir(parents=True)
        (legacy/'audio.wav').write_bytes(b'synthetic legacy')
        (legacy/'metadata.json').write_text(json.dumps({'original_text':'synthetic private legacy',
            'corrected_text':'synthetic legacy correction', 'audio_path':str(legacy/'audio.wav')}))
        assert not await g.collector.get_training_pairs()
        assert await g.collector.get_corrections_count() == 0
        await g.collector.cleanup_old()
        assert (legacy/'audio.wav').read_bytes() == b'synthetic legacy'
        assert (legacy/'metadata.json').exists()


async def test_used_mark_never_extends_deadline_or_accepts_foreign_paths(tmp_path):
    async with graph(tmp_path) as g:
        sources = g.sources(original_ttl=3)
        assert await g.add(sources=sources)
        view, = await g.collector.get_training_pairs()
        sentinel = tmp_path/'metadata.json'
        sentinel.write_text('synthetic untouched')
        fake = STTCorrection(str(tmp_path/'audio.wav'),'synthetic original','synthetic corrected','synthetic timestamp', id=view.id)
        await g.collector.mark_used([fake])
        assert await g.collector.get_corrections_count() == 1
        assert sentinel.read_text() == 'synthetic untouched'
        await g.collector.mark_used([view])
        row, = await g.rows()
        assert row['used'] and row['expires_at'] == sources[0].expires_at
        g.clock.now = sources[0].expires_at
        await g.store.reconcile()
        assert not files(g) and not await g.rows()


async def test_shortened_retention_also_shortens_previously_returned_views(tmp_path):
    async with graph(tmp_path) as g:
        sources = g.sources()
        assert await g.add(sources=sources)
        view, = await g.collector.get_training_pairs()
        await g.collector.cleanup_old(days=1/86400)
        assert view.expires_at == sources[0].captured_at+1
        g.clock.now += 1
        assert view.corrected_text == ''
        await g.store.reconcile()
        assert not files(g)
        with pytest.raises(ValueError):
            await g.collector.cleanup_old(days=90)


async def test_rotating_a_managed_file_invalidates_a_view_even_with_live_sources(tmp_path):
    async with graph(tmp_path) as g:
        assert await g.add()
        view, = await g.collector.get_training_pairs()
        assert g.privacy.sources_available(view.sources)
        await g.store._work.run(g.store.rotate, g.collector._target, 'stt_correction', 0)
        assert not view.available and view.original_text == view.corrected_text == ''
        assert not await g.rows()


async def test_disabled_feature_restart_still_retires_old_root_and_cascades_text(tmp_path):
    async with graph(tmp_path) as g:
        sources = g.sources(original_ttl=2)
        assert await g.add(sources=sources)
        clock = g.clock
    clock.now = sources[0].expires_at
    config = {'rooms': {'wake_word': {'capture_clips': {'enabled':False, 'dir':str(tmp_path/'unused_wake')}}},
              'training': {'stt_corrections': {'enabled':False, 'data_dir':str(tmp_path/'unused_corrections')}}}
    async with AsyncExitStack() as stack:
        privacy = PrivacyServices(str(tmp_path/'privacy.db'), clock=clock)
        await privacy.acall(privacy.initialize)
        stack.push_async_callback(privacy.aclose)
        store = await build_wake_clip_store(config, privacy=privacy, shutdown=stack)
        assert await build_stt_correction_collector(config, store=store) is None
        await until(lambda: not list((tmp_path/'corrections'/'policy_v1').glob('*')))
        def count():
            with privacy.ledger._transaction() as db:
                return db.execute('SELECT COUNT(*) FROM stt_corrections_v1').fetchone()[0]
        assert await store._work.run(count) == 0
        assert not (tmp_path/'unused_corrections').exists()


async def test_native_read_error_is_sanitized_and_does_not_log_private_payload(tmp_path, monkeypatch, caplog):
    async with graph(tmp_path) as g:
        def fail(*args):
            raise OSError('synthetic private transcription and path')
        monkeypatch.setattr(g.collector, '_read', fail)
        with pytest.raises(RuntimeError, match='^stt_correction_operation_failed$') as caught:
            await g.collector.get_training_pairs()
        assert caught.value.__context__ is None
        assert 'synthetic private transcription and path' not in caplog.text


async def test_held_views_are_scrubbed_on_collector_close(tmp_path):
    async with graph(tmp_path) as g:
        assert await g.add(user_id='synthetic-user')
        view, = await g.collector.get_training_pairs()
        await g.collector.aclose()
        assert not view.available and view.corrected_text == '' and view.user_id is None
        assert view.__dict__['original_text'] == ''
        assert files(g)  # Valid persisted data keeps its original expiry, independent of this reader.


async def test_real_private_collector_cannot_enter_legacy_untracked_training_export(tmp_path):
    async with graph(tmp_path) as g:
        assert await g.add()
        trainer = NightlyTrainer(config=NightlyConfig(data_dir=str(tmp_path/'unsafe_exports')),
                                 stt_correction_collector=g.collector)
        with patch('asyncio.create_subprocess_exec') as spawning:
            with pytest.raises(RuntimeError, match='private_stt_training_session_unavailable'):
                await trainer._run_whisper_finetune()
        spawning.assert_not_called()
        assert not list((tmp_path/'unsafe_exports').rglob('*.jsonl'))


async def test_cancelled_reader_keeps_native_ownership_then_releases_raw_views(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        assert await g.add()
        entered, release = threading.Event(), threading.Event()
        read = g.collector._read
        references = []
        def blocked(*args):
            views = read(*args)
            references.extend(ref(view) for view in views)
            entered.set()
            try:
                assert release.wait(5)
                return views
            finally:
                views = None
        monkeypatch.setattr(g.collector, '_read', blocked)
        reading = asyncio.create_task(g.collector.get_training_pairs())
        try:
            await until(entered.is_set)
            reading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reading
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
            assert all(view() is not None and view().corrected_text == '' for view in references)
        finally:
            release.set()
            await g.privacy.begin_close()
        await until(lambda: all(view() is None for view in references))


async def test_blocked_diagnostic_after_failure_does_not_retain_job_payload(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import src.training.stt_correction_collector as module
    async with graph(tmp_path) as g:
        entered, release = threading.Event(), threading.Event()
        jobs, waves = [], []
        def fail(job):
            jobs.append(job)
            waves.append(ref(job.audio))
            raise OSError('synthetic private payload')
        def log(message, error_type):
            assert 'synthetic private payload' not in message and error_type == 'OSError'
            entered.set()
            assert release.wait(5)
        monkeypatch.setattr(g.collector, '_encode', fail)
        monkeypatch.setattr(module, 'logger', SimpleNamespace(warning=log))
        adding = asyncio.create_task(g.add())
        try:
            await until(entered.is_set)
            assert all(job.audio is None and job.original_text == job.corrected_text == '' for job in jobs)
            assert all(wave() is None for wave in waves)
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
        finally:
            release.set()
            await adding
            await g.privacy.begin_close()


async def test_incomplete_wav_keeps_text_intent_until_physical_cleanup_can_retry(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        encode = g.collector._encode
        discard = g.store.discard
        def interrupted(job):
            encode(job)
            raise OSError('synthetic interrupted encode')
        def unavailable(*args):
            raise OSError('synthetic cleanup unavailable')
        monkeypatch.setattr(g.collector, '_encode', interrupted)
        monkeypatch.setattr(g.store, 'discard', unavailable)
        assert await g.add() is None
        row, = await g.rows()
        assert not row['published'] and row['original_text']
        assert len(files(g)) == 1 and files(g)[0].suffix == '.part'
        assert not await g.collector.get_training_pairs()
        monkeypatch.setattr(g.store, 'discard', discard)
        await g.store.reconcile()
        assert not files(g) and not await g.rows()


async def test_outstanding_read_views_have_bounded_admission(tmp_path):
    async with graph(tmp_path) as g:
        for _ in range(3):
            assert await g.add()
        retained = []
        for _ in range(100):
            try:
                retained.extend(await g.collector.get_training_pairs())
            except RuntimeError as error:
                assert str(error) == 'stt_correction_operation_failed'
                break
        else:
            raise AssertionError('expected bounded admission of retained views')
        status = g.collector.resource_status()
        assert 0 < status['read_views'] <= status['view_capacity'] == 256
        await g.collector.aclose()
        assert all(view.corrected_text == '' for view in retained)


async def test_retired_views_release_read_capacity_for_fresh_captures(tmp_path):
    async with graph(tmp_path) as g:
        sources = g.sources(original_ttl=1)
        assert await g.add(sources=sources)
        retained = []
        for _ in range(100):
            try:
                retained.extend(await g.collector.get_training_pairs())
            except RuntimeError:
                break
        assert retained
        g.clock.now = sources[0].expires_at
        assert await g.add()
        fresh, = await g.collector.get_training_pairs()
        assert fresh.available and fresh.corrected_text == 'synthetic corrected'
        assert all(view.corrected_text == '' for view in retained)
        assert g.collector.resource_status()['read_views'] == 1


async def test_shortened_durable_deadline_fences_a_still_running_encoder(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        peer = await build_stt_correction_collector({'training': {'stt_corrections': {
            'enabled': True, 'data_dir': str(g.collector.data_dir)}}}, store=g.store)
        entered, release = block_encoder(g, monkeypatch)
        adding = asyncio.create_task(g.add())
        try:
            await until(entered.is_set)
            await peer.cleanup_old(days=1/86400)
            row, = await g.rows()
            assert row['expires_at'] == g.clock.now + 1
            g.clock.now += 1
        finally:
            release.set()
            identifier = await adding
        assert identifier is None
        assert not files(g) and not await g.rows()


async def test_shortened_deadline_is_rechecked_after_actual_native_rename(tmp_path, monkeypatch):
    import src.wakeword.clip_store as module
    async with graph(tmp_path) as g:
        peer = await build_stt_correction_collector({'training': {'stt_corrections': {
            'enabled': True, 'data_dir': str(g.collector.data_dir)}}}, store=g.store)
        encoding, release_encode = block_encoder(g, monkeypatch)
        renaming, release_rename = threading.Event(), threading.Event()
        rename = module.os.replace
        def blocked(*args, **kwargs):
            renaming.set()
            assert release_rename.wait(5)
            return rename(*args, **kwargs)
        monkeypatch.setattr(module.os, 'replace', blocked)
        adding = asyncio.create_task(g.add())
        try:
            await until(encoding.is_set)
            await peer.cleanup_old(days=1/86400)
            release_encode.set()
            await until(renaming.is_set)
            g.clock.now += 1
        finally:
            release_encode.set()
            release_rename.set()
            identifier = await adding
        assert identifier is None
        assert not files(g) and not await g.rows()


async def test_peer_shortening_fences_already_held_views_before_file_sweep(tmp_path):
    async with graph(tmp_path) as g:
        assert await g.add()
        view, = await g.collector.get_training_pairs()
        peer = await build_stt_correction_collector({'training': {'stt_corrections': {
            'enabled': True, 'data_dir': str(g.collector.data_dir)}}}, store=g.store)
        await peer.cleanup_old(days=1/86400)
        assert files(g)
        g.clock.now += 1
        assert not view.available
        assert view.corrected_text == view.original_text == ''
        await g.store.reconcile()
        assert not files(g) and not await g.rows()
