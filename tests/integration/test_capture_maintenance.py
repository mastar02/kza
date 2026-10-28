"""Capture deadlines continue across disabled features and historical stores."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.ambient.audio_archive import AudioArchiver
from src.ambient.erasure import build_capture_erasure
from src.ambient.store import AmbientStore
from src.ambient.types import AmbientUtterance
from src.core.owned_work import WorkCloseTimeout
from src.memory.erasure_coordinator import ErasureCoordinator
from src.memory.erasure_journal import ErasureParticipant
from src.memory.privacy_services import PrivacyServices, build_privacy_services
from tests.integration.test_transient_privacy import until
from tests.mocks.memory_graph import Clock
from tests.mocks.memory_sources import memory_source

pytestmark = pytest.mark.asyncio
AUDIO = np.zeros(1600, dtype=np.float32)


@asynccontextmanager
async def graph(root, *, clock=None):
    clock = clock or Clock()
    config = {'memory': {'enabled': False, 'privacy_db_path': str(root/'privacy.db')},
              'ambient': {'enabled': False, 'db_path': str(root/'capture.db'),
                          'keep_audio': {'enabled': False, 'dir': str(root/'audio')}}}
    async with AsyncExitStack() as stack:
        privacy = await build_privacy_services(config, shutdown=stack, clock=clock)
        owner = ErasureCoordinator(privacy)
        stack.push_async_callback(owner.aclose)
        g = SimpleNamespace(config=config, privacy=privacy, owner=owner, clock=clock, stores=[], driver=None)
        async def make_store(*, database=None, audio=None, retention_hours=48):
            store = AmbientStore(str(database or root/'capture.db'), audio_dir=str(audio or root/'audio'),
                source_policy=privacy.ledger, clock=clock, retention_hours=retention_hours)
            archive = AudioArchiver(str(store.audio_dir), enabled=True, min_free_bytes=0,
                privacy=privacy, store=store, discard_audio=store.discard_audio)
            stack.push_async_callback(store.close)
            if g.driver is not None:
                await g.driver.attach(SimpleNamespace(store=store, transcriber=SimpleNamespace(_archiver=archive)))
            await store.init()
            g.stores.append((store, archive))
            return store, archive
        async def start():
            g.driver = await build_capture_erasure(config, privacy=privacy, owner=owner)
            for store, archive in g.stores:
                await g.driver.attach(SimpleNamespace(store=store, transcriber=SimpleNamespace(_archiver=archive)))
            return g.driver.retention
        g.make_store, g.start = make_store, start
        try:
            yield g
        finally:
            await asyncio.wait_for(asyncio.shield(owner.begin_close()), 3)


async def add_capture(g, store, *, source=None):
    source = source or memory_source(now=g.clock())
    row = await store.add(AmbientUtterance('synthetic-room', source.captured_at, source.captured_at+1,
        'synthetic private text', capture_id=source.capture_id, conversation_id=source.conversation_id, expires_at=source.expires_at))
    return source, row


def rows(database):
    with sqlite3.connect(database) as connection:
        return connection.execute('SELECT COUNT(*) FROM utterances').fetchone()[0]


async def settle(retention):
    async with asyncio.timeout(3):
        while True:
            await retention.reconcile()
            tasks = [task for _, task in retention._active.values()]
            if not tasks:
                return
            await asyncio.gather(*(asyncio.shield(task) for task in tasks))


async def test_disabled_features_purge_expired_text_and_unattached_clip_at_startup(tmp_path):
    async with graph(tmp_path) as g:
        store, archive = await g.make_store()
        source, row = await add_capture(g, store)
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        g.clock.now += 48*3600
        retention = await g.start()
        await settle(retention)
        assert rows(store.db_path) == 0 and not Path(clip).exists()
        assert (await store.audio_cleanup_status())['pending'] == 0
        assert all(target['state'] != 'unavailable' for target in retention.status()['targets'])
        assert not g.config['ambient']['enabled'] and not g.config['memory']['enabled']


@pytest.mark.parametrize('already_due', [False, True])
async def test_new_earlier_deadline_wakes_sleeping_maintenance_without_transcriber(tmp_path, monkeypatch, already_due):
    async with graph(tmp_path, clock=time.time) as g:
        store, archive = await g.make_store(retention_hours=.1/3600)
        retention = await g.start()
        await settle(retention)
        source, row = await add_capture(g, store)
        deadline = min(source.expires_at, source.captured_at + .1)
        encoded = []
        write = archive._write_sync
        def observed(*args, **kwargs):
            encoded.append(True)
            return write(*args, **kwargs)
        monkeypatch.setattr(archive, '_write_sync', observed)
        if already_due:
            async with asyncio.timeout(2):
                while time.time() < deadline:
                    await asyncio.sleep(.001)
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        if clip is None:
            assert time.time() >= deadline and archive.stats['written'] == 0
        else:
            assert archive.stats['written'] == 1
        if already_due:
            assert clip is None and not encoded
        assert archive.stats['failed'] == 0
        async with asyncio.timeout(2):
            while rows(store.db_path) or clip is not None and Path(clip).exists():
                await asyncio.sleep(.01)


async def test_deadlines_survive_restart_and_reads_do_not_renew_them(tmp_path):
    async with graph(tmp_path) as g:
        store, archive = await g.make_store()
        source, row = await add_capture(g, store, source=memory_source(now=g.clock(), ttl=30))
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        await store.set_audio_path(row, clip)
        await store.read_audio(row)
        await g.start()
    async with graph(tmp_path, clock=Clock(source.expires_at)) as g:
        retention = await g.start()
        await settle(retention)
        assert rows(g.config['ambient']['db_path']) == 0 and not Path(clip).exists()


async def test_historical_roots_in_same_database_purge_without_erase_request(tmp_path):
    async with graph(tmp_path) as g:
        old, old_archive = await g.make_store(audio=tmp_path/'old-audio')
        source, row = await add_capture(g, old)
        old_clip = await old_archive.write('synthetic-room', row, AUDIO, source=source)
        await old.set_audio_path(row, old_clip)
        retention = await g.start()
        new, new_archive = await g.make_store(audio=tmp_path/'new-audio')
        second, second_row = await add_capture(g, new)
        new_clip = await new_archive.write('synthetic-room', second_row, AUDIO, source=second)
        await new.set_audio_path(second_row, new_clip)
        g.clock.now += 48*3600
        retention.notify(old.db_path)
        await settle(retention)
        assert not Path(old_clip).exists() and not Path(new_clip).exists()
        assert rows(old.db_path) == 0


async def test_one_corrupt_target_does_not_prevent_other_registered_store_cleanup(tmp_path):
    async with graph(tmp_path) as g:
        store, archive = await g.make_store()
        source, row = await add_capture(g, store)
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        corrupt = tmp_path/'corrupt.db'
        corrupt.write_bytes(b'synthetic corrupt database')
        await g.privacy.acall(g.privacy.ledger.erasures.register_target, ErasureParticipant.CAPTURES, 1,
                             {'db_path':str(corrupt), 'audio_dir':str(tmp_path/'other-audio')})
        g.clock.now += 48*3600
        retention = await g.start()
        await settle(retention)
        assert rows(store.db_path) == 0 and not Path(clip).exists()
        assert any(target['state']=='unavailable' for target in retention.status()['targets'])
        assert corrupt.read_bytes() == b'synthetic corrupt database'


async def test_unlink_failure_is_durable_and_retried_after_restart(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        store, archive = await g.make_store()
        source, row = await add_capture(g, store)
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        unlink = Path.unlink
        def denied(path, *args, **kwargs):
            if str(path)==clip:
                raise PermissionError('synthetic private locator')
            return unlink(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'unlink', denied)
        g.clock.now += 48*3600
        retention = await g.start()
        await settle(retention)
        assert rows(store.db_path) == 0 and Path(clip).exists()
        assert (await store.audio_cleanup_status())['pending'] == 1
        assert any(target['pending_files']==1 for target in retention.status()['targets'])
    monkeypatch.setattr(Path, 'unlink', unlink)
    async with graph(tmp_path, clock=Clock(source.expires_at)) as g:
        retention = await g.start()
        await settle(retention)
        assert not Path(clip).exists()


async def test_shared_file_live_in_another_database_is_protected_until_its_deadline(tmp_path):
    async with graph(tmp_path) as g:
        first, _ = await g.make_store()
        old_source, old_row = await add_capture(g, first)
        shared = tmp_path/'audio'/'shared.flac'
        shared.parent.mkdir(parents=True)
        shared.write_bytes(b'synthetic shared clip')
        await first.set_audio_path(old_row, str(shared))
        retention = await g.start()
        g.clock.now += 24*3600
        peer, _ = await g.make_store(database=tmp_path/'peer.db')
        peer_source, peer_row = await add_capture(g, peer)
        await peer.set_audio_path(peer_row, str(shared))
        g.clock.now = old_source.expires_at
        retention.notify(first.db_path)
        await settle(retention)
        assert rows(first.db_path) == 0 and rows(peer.db_path) == 1
        assert shared.exists() and (await first.audio_cleanup_status())['pending'] == 1
        g.clock.now = peer_source.expires_at
        retention.notify(peer.db_path)
        retention.notify(first.db_path)
        await settle(retention)
        assert not shared.exists() and rows(peer.db_path) == 0


async def test_unknown_shared_peer_prevents_unlink_without_exposing_private_errors(tmp_path):
    async with graph(tmp_path) as g:
        store, archive = await g.make_store()
        source, row = await add_capture(g, store)
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        corrupt = tmp_path/'private-peer.db'
        corrupt.write_bytes(b'synthetic invalid peer')
        await g.privacy.acall(g.privacy.ledger.erasures.register_target, ErasureParticipant.CAPTURES, 1,
                             {'db_path':str(corrupt), 'audio_dir':str(store.audio_dir)})
        g.clock.now += 48*3600
        retention = await g.start()
        await settle(retention)
        assert Path(clip).exists() and (await store.audio_cleanup_status())['pending'] == 1
        encoded = json.dumps(retention.status())
        assert 'private-peer' not in encoded and 'synthetic' not in encoded and 'db_path' not in encoded


async def test_legacy_orphan_cleanup_is_durable_and_respects_shared_live_rowids(tmp_path):
    async with graph(tmp_path) as g:
        store, _ = await g.make_store()
        retention = await g.start()
        peer, _ = await g.make_store(database=tmp_path/'peer.db')
        _, row = await add_capture(g, peer)
        orphan = tmp_path/'audio'/'999.flac'
        protected = tmp_path/'audio'/f'{row}.flac'
        orphan.parent.mkdir(parents=True)
        for path in (orphan, protected):
            path.write_bytes(b'synthetic old clip')
            os.utime(path, (g.clock()-48*3600-1,)*2)
        retention._orphan_due.clear()
        retention.notify(store.db_path)
        await settle(retention)
        assert not orphan.exists() and protected.exists()


async def test_missing_store_is_not_created_and_unknown_orphan_is_not_assumed_expired(tmp_path):
    async with graph(tmp_path) as g:
        orphan = tmp_path/'audio'/'unknown.flac'
        orphan.parent.mkdir(parents=True)
        orphan.write_bytes(b'synthetic unknown')
        retention = await g.start()
        await settle(retention)
        assert not Path(g.config['ambient']['db_path']).exists()
        assert orphan.exists()
        assert retention.status()['targets'][0]['state']=='unavailable'


async def test_empty_disabled_store_does_not_create_database_or_archive(tmp_path):
    async with graph(tmp_path) as g:
        retention = await g.start()
        await settle(retention)
        assert not Path(g.config['ambient']['db_path']).exists()
        assert not Path(g.config['ambient']['keep_audio']['dir']).exists()
        assert retention.status()['targets'][0]['state']=='current'


async def test_saturated_default_executor_does_not_starve_retention(tmp_path):
    async with graph(tmp_path) as g:
        store, archive = await g.make_store()
        source, row = await add_capture(g, store)
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        g.clock.now += 48*3600
        loop = asyncio.get_running_loop()
        previous = loop._default_executor
        pool = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(pool)
        entered, release = threading.Event(), threading.Event()
        def blocked():
            entered.set()
            release.wait(3)
        occupying = asyncio.create_task(asyncio.to_thread(blocked))
        try:
            await until(entered.is_set)
            retention = await g.start()
            await asyncio.wait_for(settle(retention), 1)
            assert not Path(clip).exists() and not occupying.done()
        finally:
            release.set()
            await occupying
            loop._default_executor = previous
            pool.shutdown(wait=True)


async def test_blocked_unlink_keeps_lease_and_other_target_can_still_finish(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        first, archive = await g.make_store()
        source, row = await add_capture(g, first)
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        entered, release = threading.Event(), threading.Event()
        unlink = Path.unlink
        def blocked(path, *args, **kwargs):
            if str(path)==clip:
                entered.set()
                release.wait(3)
            return unlink(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'unlink', blocked)
        retention = await g.start()
        second, second_archive = await g.make_store(database=tmp_path/'second.db', audio=tmp_path/'second-audio')
        other_source, other_row = await add_capture(g, second)
        other_clip = await second_archive.write('synthetic-room', other_row, AUDIO, source=other_source)
        g.clock.now += 48*3600
        retention.notify(first.db_path)
        retention.notify(second.db_path)
        competitor = PrivacyServices(g.privacy.ledger.path)
        try:
            await until(entered.is_set)
            async with asyncio.timeout(2):
                while Path(other_clip).exists():
                    await asyncio.sleep(.01)
            g.privacy._close_timeout_s = .01
            with pytest.raises(WorkCloseTimeout):
                await g.privacy.aclose()
            assert g.privacy.resource_status()['writer_lease']
            with pytest.raises(RuntimeError, match='writer_already_active'):
                await competitor.acall(competitor.initialize)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.shield(g.owner.begin_close()), 3)
            await asyncio.wait_for(asyncio.shield(g.privacy.begin_close()), 3)
            await competitor.aclose()
        assert not Path(clip).exists() and retention.status()['disk']['operations']==0


async def test_late_harvest_does_not_push_an_already_due_sweep_into_the_future(tmp_path):
    async with graph(tmp_path) as g:
        store, archive = await g.make_store()
        source, row = await add_capture(g, store)
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        retention = await g.start()
        # Observe actual sweep exit before the scheduler harvests its receipt.
        await retention.reconcile()
        await asyncio.gather(*(asyncio.shield(task) for _, task in retention._active.values()))
        g.clock.now = source.expires_at
        await settle(retention)
        assert rows(store.db_path) == 0 and not Path(clip).exists()


async def test_concurrent_schema_initialization_is_atomic_across_store_connections(tmp_path):
    for attempt in range(5):
        database = str(tmp_path/f'concurrent-{attempt}.db')
        stores = [AmbientStore(database) for _ in range(3)]
        try:
            assert all(await asyncio.gather(*(store.init() for store in stores)))
            with sqlite3.connect(database) as connection:
                columns = [row[1] for row in connection.execute('PRAGMA table_info(pending_audio_deletions)')]
                assert columns.count('conversation_id') == 1
        finally:
            await asyncio.gather(*(store.close() for store in stores))


async def test_cancelled_scheduler_observer_does_not_cancel_actual_sweep(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        store, archive = await g.make_store()
        source, row = await add_capture(g, store)
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        entered, release = threading.Event(), threading.Event()
        unlink = Path.unlink
        def blocked(path, *args, **kwargs):
            if str(path)==clip:
                entered.set()
                release.wait(3)
            return unlink(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'unlink', blocked)
        g.clock.now = source.expires_at
        retention = await g.start()
        try:
            await until(entered.is_set)
            # Cancelling only a scheduler observer cannot reach actual jobs.
            observing = asyncio.create_task(retention.reconcile())
            await asyncio.sleep(0)
            observing.cancel()
            await asyncio.gather(observing, return_exceptions=True)
            assert retention.status()['disk']['operations'] > 0
            assert any(not task.done() for _, task in retention._active.values())
        finally:
            release.set()
        await settle(retention)
        assert not Path(clip).exists()


async def test_archive_close_waits_reservation_and_discard_not_only_native_encoder(tmp_path, monkeypatch):
    async with graph(tmp_path) as g:
        store, archive = await g.make_store()
        source, row = await add_capture(g, store)
        ready, release = asyncio.Event(), asyncio.Event()
        reserve = store.reserve_audio_path
        async def blocked(*args):
            reservation = await reserve(*args)
            ready.set()
            await release.wait()
            return reservation
        monkeypatch.setattr(store, 'reserve_audio_path', blocked)
        writing = asyncio.create_task(archive.write('synthetic-room', row, AUDIO, source=source))
        await ready.wait()
        closing = archive.begin_close()
        try:
            await asyncio.sleep(.01)
            assert not closing.done() and not writing.done() and not archive._writes
        finally:
            release.set()
        assert await writing is None
        await closing
        assert not list(store.audio_dir.rglob('*.flac'))


async def test_unsupported_target_stays_visible_and_prevents_unknown_file_deletion(tmp_path):
    async with graph(tmp_path) as g:
        store, archive = await g.make_store()
        source, row = await add_capture(g, store)
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        await g.privacy.acall(g.privacy.ledger.erasures.register_target, ErasureParticipant.CAPTURES, 2,
                             {'future_private_layout':'synthetic unknown'})
        g.clock.now = source.expires_at
        retention = await g.start()
        await settle(retention)
        assert rows(store.db_path)==0 and Path(clip).exists()
        assert retention.status()['registry_errors'][0]['error']=='unsupported_capture_retention_target'
        assert (await store.audio_cleanup_status())['pending']==1


async def test_disabled_maintenance_honors_a_shorter_configured_limit(tmp_path):
    async with graph(tmp_path) as g:
        store, archive = await g.make_store()
        source, row = await add_capture(g, store)
        clip = await archive.write('synthetic-room', row, AUDIO, source=source)
        g.config['ambient']['retention_hours'] = 1
        g.clock.now += 3600
        retention = await g.start()
        await settle(retention)
        assert not Path(clip).exists() and rows(store.db_path)==0


async def test_disabled_runtime_retention_never_imports_semantic_or_audio_models(tmp_path):
    script = textwrap.dedent('''
        import asyncio, importlib.abc, sys
        from contextlib import AsyncExitStack
        from pathlib import Path
        class NoModels(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'torch','chromadb','sentence_transformers','faster_whisper','onnxruntime'}:
                    raise AssertionError('retention attempted a model import: '+fullname)
        sys.meta_path.insert(0, NoModels())
        from src.memory.privacy_services import build_privacy_services
        from src.memory.erasure_coordinator import ErasureCoordinator
        from src.ambient.erasure import build_capture_erasure
        async def check():
            root = Path(sys.argv[1])
            config = {'memory': {'enabled': False, 'privacy_db_path': str(root/'privacy.db')},
                      'ambient': {'enabled': False, 'db_path': str(root/'absent.db'),
                                  'keep_audio': {'enabled': False,'dir':str(root/'absent-audio')}}}
            async with AsyncExitStack() as stack:
                privacy = await build_privacy_services(config, shutdown=stack)
                owner = ErasureCoordinator(privacy)
                stack.push_async_callback(owner.aclose)
                driver = await build_capture_erasure(config, privacy=privacy, owner=owner)
                for _ in range(3):
                    await driver.retention.reconcile()
                    await asyncio.gather(*(asyncio.shield(task) for _, task in driver.retention._active.values()))
                assert not (root/'absent.db').exists()
                assert not (root/'absent-audio').exists()
            assert privacy.ledger._db is None
        asyncio.run(check())
    ''')
    result = await asyncio.to_thread(subprocess.run, [sys.executable, '-c', script, str(tmp_path)],
                                     capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout+result.stderr
