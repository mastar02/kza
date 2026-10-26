"""Capture erasure against real SQLite/FLAC, native gates and restarted owners."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.ambient.audio_archive import AudioArchiver
from src.ambient.erasure import build_capture_erasure
from src.ambient.store import AmbientStore
from src.ambient.transcriber import AmbientTranscriber
from src.ambient.types import AmbientUtterance, RawSegment
from src.core.owned_work import WorkCloseTimeout
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.fact_ledger import MemoryReviewer
from src.memory.privacy_services import PrivacyServices, build_privacy_services
from tests.integration.test_transient_privacy import until
from tests.mocks.memory_graph import Clock
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio
ACTOR = MemoryReviewer('synthetic-admin')
AUDIO = np.full(1600, .1, dtype=np.float32)


def configuration(root, *, location='current'):
    return {'memory': {'enabled': False, 'privacy_db_path': str(root/'privacy.db')},
            'ambient': {'enabled': False, 'db_path': str(root/location/'ambient.db'),
                        'keep_audio': {'enabled': False, 'dir': str(root/location/'audio')}}}


@asynccontextmanager
async def graph(root, *, location='current'):
    clock = Clock()
    config = configuration(root, location=location)
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        owner = ErasureCoordinator(privacy, close_timeout_s=.01)
        stack.push_async_callback(owner.aclose)
        driver = await build_capture_erasure(config, privacy=privacy, owner=owner)
        store = AmbientStore(config['ambient']['db_path'], audio_dir=config['ambient']['keep_audio']['dir'],
                             source_policy=privacy.ledger, clock=clock)
        stack.push_async_callback(store.close)
        archive = AudioArchiver(str(store.audio_dir), enabled=True, min_free_bytes=0,
            privacy=privacy, store=store, discard_audio=store.discard_audio)
        await driver.attach(SimpleNamespace(store=store, transcriber=SimpleNamespace(_archiver=archive)))
        await store.init()
        yield SimpleNamespace(privacy=privacy, owner=owner, driver=driver, store=store, archive=archive,
                              clock=clock, config=config, stack=stack)


async def capture(g):
    source = memory_source(now=g.clock.now)
    utt = AmbientUtterance('synthetic-room', source.captured_at, source.captured_at+1,
        'synthetic private text', capture_id=source.capture_id, conversation_id=source.conversation_id,
        expires_at=source.expires_at)
    return source, await g.store.add(utt)


async def flush(owner):
    await owner.reconcile()
    await asyncio.gather(*(a.task for a in owner._active.values()))
    await owner.reconcile()


async def status(g, receipt):
    snapshot = await g.privacy.acall(g.privacy.ledger.erasures.snapshot, receipt['id'])
    return [s for s in snapshot['steps'] if s['participant']=='captures']


def counts(path):
    with sqlite3.connect(path) as db:
        return {table: db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
                for table in ('utterances', 'capture_conversations', 'audio_write_intents', 'pending_audio_deletions')}


async def test_locator_is_durable_before_encoder_and_new_files_use_capture_uuid(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        source, row = await capture(g)
        write = g.archive._write_sync
        observed = []
        def encode(path, mono):
            observed.append(counts(g.store.db_path)['audio_write_intents'])
            assert not path.exists()
            return write(path, mono)
        monkeypatch.setattr(g.archive, '_write_sync', encode)
        path = await g.archive.write('synthetic-room', row, AUDIO, source=source)
        assert observed == [1]
        assert Path(path).name == source.capture_id+'.flac'
        assert Path(path).parent.name == 'policy_v1'
        assert Path(path).stat().st_mode & 0o777 == 0o600
        assert Path(path).parent.stat().st_mode & 0o777 == 0o700
        assert counts(g.store.db_path)['audio_write_intents'] == 1
        assert await g.store.set_audio_path(row, path)
        assert counts(g.store.db_path)['audio_write_intents'] == 0
        assert Path(g.store.db_path).stat().st_mode & 0o777 == 0o600


async def test_revoked_capture_erases_rows_intents_clips_and_scope_but_preserves_other_conversation(tmp_path):
    async with graph(tmp_path) as g:
        source, row = await capture(g)
        survivor, survivor_row = await capture(g)
        path = await g.archive.write('synthetic-room', row, AUDIO, source=source)
        other = await g.archive.write('synthetic-room', survivor_row, AUDIO, source=survivor)
        await g.store.set_audio_path(survivor_row, other)
        receipt = await g.owner.request(source.conversation_id, ACTOR)
        await flush(g.owner)
        assert (await status(g, receipt))[0]['state'] == 'complete'
        assert not Path(path).exists() and Path(other).exists()
        assert counts(g.store.db_path) == {'utterances':1,'capture_conversations':1,'audio_write_intents':0,'pending_audio_deletions':0}
        assert await g.store.read_audio(survivor_row)
        assert (await g.privacy.acall(g.privacy.ledger.erasures.snapshot, receipt['id']))['state'] == 'blocked'


async def test_live_conversation_cannot_be_physically_erased_without_revocation(tmp_path):
    async with graph(tmp_path) as g:
        source, _ = await capture(g)
        with pytest.raises(PermissionError):
            await g.store.erase_conversation(source.conversation_id)
        assert counts(g.store.db_path)['utterances'] == 1


async def test_revocation_waits_native_encoder_even_when_ttl_purge_runs_first(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        source, row = await capture(g)
        entered, release = threading.Event(), threading.Event()
        write = g.archive._write_sync
        def blocked(path, mono):
            entered.set()
            release.wait(3)
            return write(path, mono)
        monkeypatch.setattr(g.archive, '_write_sync', blocked)
        writing = asyncio.create_task(g.archive.write('synthetic-room', row, AUDIO, source=source))
        try:
            await until(entered.is_set)
            receipt = await g.owner.request(source.conversation_id, ACTOR)
            await g.owner.reconcile()
            await g.store.purge_expired()
            assert (await status(g, receipt))[0]['state'] == 'running'
            assert counts(g.store.db_path)['audio_write_intents'] == 1
            assert counts(g.store.db_path)['pending_audio_deletions'] == 1
            with pytest.raises(RuntimeError, match='encoders_still_running'):
                await g.store.erase_conversation(source.conversation_id)
        finally:
            release.set()
        assert await writing is None
        await flush(g.owner)
        assert (await status(g, receipt))[0]['state'] == 'complete'
        assert list(Path(g.store.audio_dir).rglob('*.flac')) == []
        assert counts(g.store.db_path) == dict.fromkeys(counts(g.store.db_path), 0)


async def test_unattached_written_clip_is_found_after_owner_restart(tmp_path):
    async with graph(tmp_path) as g:
        source, row = await capture(g)
        path = await g.archive.write('synthetic-room', row, AUDIO, source=source)
        assert counts(g.store.db_path)['audio_write_intents'] == 1
    async with graph(tmp_path) as g:
        receipt = await g.owner.request(source.conversation_id, ACTOR)
        await flush(g.owner)
        assert not Path(path).exists()
        assert (await status(g, receipt))[0]['state'] == 'complete'
        assert counts(g.store.db_path)['audio_write_intents'] == 0


async def test_unlink_failure_stays_pending_through_restart_and_keeps_no_text(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        source, row = await capture(g)
        path = await g.archive.write('synthetic-room', row, AUDIO, source=source)
        await g.store.set_audio_path(row, path)
        unlink = Path.unlink
        def denied(candidate, *args, **kwargs):
            if str(candidate) == path:
                raise PermissionError('synthetic private locator')
            return unlink(candidate, *args, **kwargs)
        monkeypatch.setattr(Path, 'unlink', denied)
        receipt = await g.owner.request(source.conversation_id, ACTOR)
        await flush(g.owner)
        assert (await status(g, receipt))[0]['state'] == 'pending'
        assert (await status(g, receipt))[0]['pending_count'] == 1
        assert counts(g.store.db_path)['utterances'] == 0
        assert counts(g.store.db_path)['pending_audio_deletions'] == 1
    monkeypatch.setattr(Path, 'unlink', unlink)
    async with graph(tmp_path) as g:
        g.clock.now += 2
        await flush(g.owner)
        assert not Path(path).exists() and (await status(g, receipt))[0]['state'] == 'complete'


async def test_historical_disabled_location_remains_required_after_configuration_change(tmp_path):
    async with graph(tmp_path, location='old') as g:
        source, row = await capture(g)
        path = await g.archive.write('synthetic-room', row, AUDIO, source=source)
        receipt = await g.owner.request(source.conversation_id, ACTOR)
        old_database = g.store.db_path
    async with graph(tmp_path, location='new') as g:
        survivor, row = await capture(g)
        other = await g.archive.write('synthetic-room', row, AUDIO, source=survivor)
        await flush(g.owner)
        assert not Path(path).exists() and Path(other).exists()
        assert counts(old_database)['utterances'] == 0
        assert len(await status(g, receipt)) == 2
        assert all(step['state']=='complete' for step in await status(g, receipt))


@pytest.mark.parametrize('clip_present', [False, True])
async def test_missing_database_does_not_create_it_or_claim_orphaned_clips_erased(tmp_path, clip_present):
    clock = Clock()
    config = configuration(tmp_path)
    if clip_present:
        clip = tmp_path/'current'/'audio'/'unknown.flac'
        clip.parent.mkdir(parents=True)
        clip.write_bytes(b'synthetic orphan')
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        await build_capture_erasure(config, privacy=privacy, owner=owner)
        receipt = await owner.request(memory_source(now=clock.now).conversation_id, ACTOR)
        await flush(owner)
        step = (await status(SimpleNamespace(privacy=privacy), receipt))[0]
        assert step['state'] == ('pending' if clip_present else 'complete')
        assert not Path(config['ambient']['db_path']).exists()
        if clip_present:
            assert clip.read_bytes() == b'synthetic orphan'


async def test_archive_traversal_permission_error_is_unknown_not_empty(tmp_path, monkeypatch):
    import src.ambient.erasure as module
    config = configuration(tmp_path)
    Path(config['ambient']['keep_audio']['dir']).mkdir(parents=True)
    scan = module.os.scandir
    def denied(path):
        if str(path) == config['ambient']['keep_audio']['dir']:
            raise PermissionError('synthetic inaccessible archive')
        return scan(path)
    monkeypatch.setattr(module.os, 'scandir', denied)
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        await build_capture_erasure(config, privacy=privacy, owner=owner)
        receipt = await owner.request(memory_source().conversation_id, ACTOR)
        await flush(owner)
        step = (await status(SimpleNamespace(privacy=privacy), receipt))[0]
        assert step['state'] == 'pending' and step['last_error'] == 'cleanup_failed'


@pytest.mark.parametrize('change', ['expire', 'revoke'])
async def test_expired_or_revoked_source_cannot_start_a_late_archive(tmp_path, monkeypatch, change):
    async with graph(tmp_path) as g:
        source, row = await capture(g)
        encode = AsyncMock()
        monkeypatch.setattr(g.archive, '_write_sync', encode)
        assert await g.archive.write('synthetic-room', row, AUDIO) is None
        if change == 'expire':
            g.clock.now = source.expires_at
        else:
            receipt = await g.owner.request(source.conversation_id, ACTOR)
            await flush(g.owner)
        assert await g.archive.write('synthetic-room', row, AUDIO, source=source) is None
        encode.assert_not_called()
        if change == 'revoke':
            assert (await status(g, receipt))[0]['state'] == 'complete'


async def test_repeated_cancellation_and_close_timeout_do_not_release_live_encoder_lease(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        g.privacy._close_timeout_s = .01
        source, row = await capture(g)
        entered, release = threading.Event(), threading.Event()
        write = g.archive._write_sync
        def blocked(path, mono):
            entered.set()
            release.wait(3)
            return write(path, mono)
        monkeypatch.setattr(g.archive, '_write_sync', blocked)
        writing = asyncio.create_task(g.archive.write('synthetic-room', row, AUDIO, source=source))
        competitor = PrivacyServices(g.privacy.ledger.path)
        try:
            await until(entered.is_set)
            writing.cancel()
            writing.cancel()
            await asyncio.sleep(0)
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease'] and not writing.done()
            with pytest.raises(RuntimeError, match='writer_already_active'):
                await competitor.acall(competitor.initialize)
        finally:
            release.set()
            await asyncio.gather(writing, return_exceptions=True)
            await asyncio.wait_for(asyncio.shield(g.owner.begin_close()), 3)
            await asyncio.wait_for(asyncio.shield(g.privacy.begin_close()), 3)
            await competitor.aclose()
        assert list(Path(g.store.audio_dir).rglob('*.flac')) == []


async def test_textual_wake_finishing_after_erasure_cannot_create_a_new_clip(tmp_path):
    async with graph(tmp_path) as g:
        ready, release = asyncio.Event(), asyncio.Event()
        async def wake(*args, memory_source=None, **kwargs):
            receipt = await g.owner.request(memory_source.conversation_id, ACTOR)
            await flush(g.owner)
            ready.set()
            await release.wait()
            return receipt
        stt = SimpleNamespace(transcribe=AsyncMock(return_value=SimpleNamespace(text='synthetic private text', avg_logprob=-.1, no_speech_prob=.01)),
                              asr_mono=lambda audio: audio)
        tr = AmbientTranscriber(None, None, stt, SimpleNamespace(tag=AsyncMock(return_value=('unknown', .0))),
            SimpleNamespace(estimate=lambda audio: None), SimpleNamespace(classify=lambda **kwargs:'live'), g.store,
            ['synthetic-room'], privacy=g.privacy, archiver=g.archive)
        tr.attach_textual_wake(SimpleNamespace(maybe_dispatch=wake))
        pending = asyncio.create_task(tr._handle_segment('synthetic-room', RawSegment(g.clock.now, g.clock.now+1, AUDIO, vad_prob=.9)))
        try:
            await asyncio.wait_for(ready.wait(), 3)
        finally:
            release.set()
            await pending
        assert g.archive.stats['written'] == 0
        assert counts(g.store.db_path)['utterances'] == 0
        assert list(Path(g.store.audio_dir).rglob('*.flac')) == []


async def test_identical_rowids_in_different_databases_do_not_overwrite_or_delete_other_clips(tmp_path):
    async with graph(tmp_path) as g:
        source, first_row = await capture(g)
        first_path = await g.archive.write('synthetic-room', first_row, AUDIO, source=source)
        second_store = AmbientStore(str(tmp_path/'other.db'), audio_dir=str(g.store.audio_dir),
                                   source_policy=g.privacy.ledger, clock=g.clock)
        g.stack.push_async_callback(second_store.close)
        other_archive = AudioArchiver(str(second_store.audio_dir), enabled=True, min_free_bytes=0,
            privacy=g.privacy, store=second_store, discard_audio=second_store.discard_audio)
        await g.driver.attach(SimpleNamespace(store=second_store, transcriber=SimpleNamespace(_archiver=other_archive)))
        await second_store.init()
        second = memory_source(now=g.clock.now)
        row = await second_store.add(AmbientUtterance('synthetic-room', second.captured_at, second.captured_at+1,
            'synthetic other capture', capture_id=second.capture_id, conversation_id=second.conversation_id, expires_at=second.expires_at))
        second_path = await other_archive.write('synthetic-room', row, AUDIO, source=second)
        await second_store.set_audio_path(row, second_path)
        assert first_row == row and first_path != second_path
        receipt = await g.owner.request(source.conversation_id, ACTOR)
        await flush(g.owner)
        assert not Path(first_path).exists() and Path(second_path).exists()
        assert await second_store.read_audio(row)
        assert all(step['state']=='complete' for step in await status(g, receipt))


async def test_unattached_expired_intent_is_purged_after_restart_without_a_conversation_request(tmp_path):
    async with graph(tmp_path) as g:
        source, row = await capture(g)
        path = await g.archive.write('synthetic-room', row, AUDIO, source=source)
    async with graph(tmp_path) as g:
        g.clock.now = source.expires_at
        await g.store.purge_expired()
        assert not Path(path).exists()
        assert counts(g.store.db_path) == dict.fromkeys(counts(g.store.db_path), 0)


async def test_same_database_keeps_old_archive_root_after_directory_changes(tmp_path):
    async with graph(tmp_path) as g:
        source, row = await capture(g)
        old_clip = await g.archive.write('synthetic-room', row, AUDIO, source=source)
        await g.store.set_audio_path(row, old_clip)
        second_store = AmbientStore(g.store.db_path, audio_dir=str(tmp_path/'new-audio'),
                                   source_policy=g.privacy.ledger, clock=g.clock)
        g.stack.push_async_callback(second_store.close)
        archive = AudioArchiver(str(second_store.audio_dir), enabled=True, min_free_bytes=0,
            privacy=g.privacy, store=second_store, discard_audio=second_store.discard_audio)
        await g.driver.attach(SimpleNamespace(store=second_store, transcriber=SimpleNamespace(_archiver=archive)))
        await second_store.init()
        new_source = memory_source(now=g.clock.now, conversation_id=source.conversation_id)
        new_row = await second_store.add(AmbientUtterance('synthetic-room', new_source.captured_at, new_source.captured_at+1,
            'synthetic new capture', capture_id=new_source.capture_id, conversation_id=new_source.conversation_id, expires_at=new_source.expires_at))
        new_clip = await archive.write('synthetic-room', new_row, AUDIO, source=new_source)
        await second_store.set_audio_path(new_row, new_clip)
        receipt = await g.owner.request(source.conversation_id, ACTOR)
        await flush(g.owner)
        assert not Path(old_clip).exists() and not Path(new_clip).exists()
        assert len(await status(g, receipt)) == 2
        assert all(step['state']=='complete' for step in await status(g, receipt))


@pytest.mark.parametrize('db_path', [':memory:', 'file:untracked?mode=memory'])
async def test_runtime_capture_targets_cannot_disable_durable_locator_storage(tmp_path, db_path):
    config = configuration(tmp_path)
    config['ambient']['db_path'] = db_path
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        with pytest.raises(ValueError, match='persistent_file'):
            await build_capture_erasure(config, privacy=privacy, owner=owner)


async def test_unregistered_archive_path_stays_pending_without_deleting_foreign_file(tmp_path):
    async with graph(tmp_path) as g:
        source, row = await capture(g)
        foreign = tmp_path/'outside.flac'
        foreign.write_bytes(b'synthetic unowned clip')
        await g.store.set_audio_path(row, str(foreign))
        receipt = await g.owner.request(source.conversation_id, ACTOR)
        await flush(g.owner)
        step = (await status(g, receipt))[0]
        assert step['state'] == 'pending' and step['pending_count'] == 1
        assert foreign.read_bytes() == b'synthetic unowned clip'
        assert counts(g.store.db_path)['utterances'] == 0


async def test_corrupt_historical_database_stays_pending_without_replacement(tmp_path):
    config = configuration(tmp_path)
    database = Path(config['ambient']['db_path'])
    database.parent.mkdir(parents=True)
    database.write_bytes(b'synthetic corrupt database')
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        await build_capture_erasure(config, privacy=privacy, owner=owner)
        receipt = await owner.request(memory_source().conversation_id, ACTOR)
        await flush(owner)
        step = (await status(SimpleNamespace(privacy=privacy), receipt))[0]
        assert step['state'] == 'pending' and step['last_error'] == 'cleanup_failed'
        assert database.read_bytes() == b'synthetic corrupt database'


async def test_cancelled_attachment_discards_written_file_through_durable_cleanup(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        source, row = await capture(g)
        entered = asyncio.Event()
        async def blocked(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(g.store, 'set_audio_path', blocked)
        tr = AmbientTranscriber(None, None, None, None, None, None, g.store, [],
                                archiver=g.archive, privacy=g.privacy)
        pending = asyncio.create_task(tr._archive_audio('synthetic-room', row, AUDIO, source=source))
        await asyncio.wait_for(entered.wait(), 3)
        assert counts(g.store.db_path)['audio_write_intents'] == 1
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert list(Path(g.store.audio_dir).rglob('*.flac')) == []
        assert counts(g.store.db_path)['audio_write_intents'] == 0


async def test_encoder_respects_earlier_store_deadline_before_attachment(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        g.store.retention_hours = 1/3600  # Earlier than the raw source's 48 h.
        source, row = await capture(g)
        entered, release = threading.Event(), threading.Event()
        write = g.archive._write_sync
        def blocked(path, mono):
            entered.set()
            release.wait(3)
            return write(path, mono)
        monkeypatch.setattr(g.archive, '_write_sync', blocked)
        writing = asyncio.create_task(g.archive.write('synthetic-room', row, AUDIO, source=source))
        try:
            await until(entered.is_set)
            g.clock.now += 2
        finally:
            release.set()
        assert await writing is None
        assert list(Path(g.store.audio_dir).rglob('*.flac')) == []
