"""Capture privacy through SQLite, archived bytes, cleanup and restart."""

import asyncio
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.ambient.audio_archive import AudioArchiver
from src.ambient.store import AmbientStore
from src.ambient.transcriber import AmbientPath, AmbientTranscriber
from src.ambient.types import AmbientUtterance


@dataclass
class Clock:
    now: float = 2_000_000_000.0

    def __call__(self):
        return self.now


def utterance(clock, **changes):
    return AmbientUtterance(**({'room_id': 'test_room', 't0': clock.now, 't1': clock.now + 1,
        'text': 'hola que tal todo bien', 'lang_ok': True, 'vad_prob': .2} | changes))


@pytest.mark.asyncio
async def test_expired_text_is_hidden_from_all_readers_without_purge(tmp_path):
    clock = Clock()
    store = AmbientStore(str(tmp_path / 'capture.db'), clock=clock)
    await store.init()
    try:
        await store.add(utterance(clock))
        assert len(await store.utterances_between('test_room', 0, clock.now + 2)) == 1
        assert len(await store.undistilled_live()) == 1
        assert (await store.spanish_keep_rate())['n'] == 1
        clock.now += 48 * 3600
        assert await store.utterances_between('test_room', 0, clock.now) == []
        assert await store.undistilled_live() == []
        assert (await store.spanish_keep_rate()) == {
            'n': 0, 'ok': 0, 'not_ok': 0, 'unclassified': 0, 'rate': None, 'by_room': {}}
        assert (await store.spanish_keep_rate(since_s=0))['n'] == 0
        # This proves read-time enforcement without relying on a cleanup job.
        assert (await (await store._db.execute('SELECT COUNT(*) FROM utterances')).fetchone())[0] == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_language_flag_preserves_far_field_text_while_distillation_filters_vad(tmp_path):
    from src.ambient.language_quality import make_quality_fn
    clock = Clock()
    store = AmbientStore(str(tmp_path / 'capture.db'), clock=clock)
    await store.init()
    try:
        lang, probability, keep = make_quality_fn(lambda text: ('es', .9))('hola que tal todo bien', .2)
        await store.add(utterance(clock, lang=lang, lang_prob=probability, lang_ok=keep))
        rows = await store.utterances_between('test_room', clock.now - 1, clock.now + 2)
        assert rows[0]['text'] == 'hola que tal todo bien' and rows[0]['lang_ok'] == 1
        assert await store.undistilled_live(min_vad_prob=.45, spanish_only=True) == []
        assert len(await store.undistilled_live(min_vad_prob=0, spanish_only=True)) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_audio_and_text_expire_after_restart_and_cleanup_removes_bytes(tmp_path):
    clock = Clock()
    db = str(tmp_path / 'capture.db')
    store = AmbientStore(db, clock=clock, audio_dir=str(tmp_path))
    await store.init()
    capture = utterance(clock)
    row_id = await store.add(capture)
    audio = tmp_path / 'clip.flac'
    audio.write_bytes(b'synthetic clip')
    assert await store.set_audio_path(row_id, str(audio))
    assert await store.read_audio(row_id) == b'synthetic clip'
    await store.close()
    clock.now += 48 * 3600
    reopened = AmbientStore(db, clock=clock, audio_dir=str(tmp_path))
    await reopened.init()
    try:
        assert await reopened.read_audio(row_id) is None
        assert await reopened.utterances_between('test_room', 0, clock.now) == []
        assert await reopened.purge_expired() == 1
        assert not audio.exists()
        assert (await reopened.audio_cleanup_status())['pending'] == 0
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_original_expiry_cannot_be_extended_by_config_change_or_future_capture_time(tmp_path):
    clock = Clock()
    db = str(tmp_path / 'capture.db')
    store = AmbientStore(db, retention_hours=1, clock=clock)
    await store.init()
    await store.add(utterance(clock, t0=clock.now + 365 * 86400, t1=clock.now + 365 * 86400 + 1))
    await store.close()
    clock.now += 3600
    reopened = AmbientStore(db, retention_hours=48, clock=clock)
    await reopened.init()
    try:
        assert await reopened.undistilled_live() == []
        assert await reopened.purge_expired() == 1
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_failed_unlink_remains_durable_and_retries_after_restart_even_with_fresh_mtime(tmp_path, monkeypatch):
    clock = Clock()
    db = str(tmp_path / 'capture.db')
    store = AmbientStore(db, clock=clock, audio_dir=str(tmp_path))
    await store.init()
    row_id = await store.add(utterance(clock))
    audio = tmp_path / 'clip.flac'
    audio.write_bytes(b'synthetic clip')
    await store.set_audio_path(row_id, str(audio))
    unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path == audio:
            raise PermissionError('synthetic unavailable filesystem')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', denied)
    clock.now += 48 * 3600
    assert await store.purge_expired() == 1
    assert audio.exists()
    assert (await store.audio_cleanup_status())['pending'] == 1
    await store.close()
    monkeypatch.setattr(Path, 'unlink', unlink)
    reopened = AmbientStore(db, clock=clock, audio_dir=str(tmp_path))
    await reopened.init()
    try:
        assert await reopened.purge_expired() == 0
        assert not audio.exists()
        assert (await reopened.audio_cleanup_status())['pending'] == 0
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_late_audio_pointer_cannot_resurrect_an_expired_capture(tmp_path):
    clock = Clock()
    store = AmbientStore(str(tmp_path / 'capture.db'), clock=clock, audio_dir=str(tmp_path))
    await store.init()
    try:
        row_id = await store.add(utterance(clock))
        clock.now += 48 * 3600
        assert await store.purge_expired() == 1
        late = tmp_path / 'late.flac'
        late.write_bytes(b'late native write')
        assert await store.set_audio_path(row_id, str(late)) is False
        assert not late.exists()
        assert await store.read_audio(row_id) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_owning_archive_write_cancellation_prevents_an_abandoned_late_clip(tmp_path, monkeypatch):
    clock = Clock()
    store = AmbientStore(str(tmp_path / 'capture.db'), clock=clock, audio_dir=str(tmp_path))
    await store.init()
    entered, release = threading.Event(), threading.Event()
    archiver = AudioArchiver(str(tmp_path), enabled=True, discard_audio=store.discard_audio)

    def delayed_write(path, mono):
        entered.set()
        assert release.wait(2)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'late native write')
        return True

    monkeypatch.setattr(archiver, '_write_sync', delayed_write)
    task = asyncio.create_task(archiver.write('test_room', 1, np.ones(1600, dtype=np.float32)))
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(.001)
        task.cancel()
        await asyncio.sleep(.01)
        task.cancel()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not list(tmp_path.rglob('*.flac'))
        assert (await store.audio_cleanup_status())['pending'] == 0
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()


@pytest.mark.parametrize('hours', [True, 0, -1, 49, math.nan, math.inf, '48'])
def test_raw_retention_must_be_finite_and_no_longer_than_48_hours(tmp_path, hours):
    with pytest.raises(ValueError):
        AmbientStore(str(tmp_path / 'capture.db'), retention_hours=hours)


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(.001)


@pytest.mark.asyncio
async def test_cancelled_expiry_transaction_keeps_atomic_cleanup_intent(tmp_path, monkeypatch):
    clock = Clock()
    db = str(tmp_path / 'capture.db')
    store = AmbientStore(db, clock=clock, audio_dir=str(tmp_path))
    await store.init()
    audio = tmp_path / 'clip.flac'
    audio.write_bytes(b'synthetic clip')
    row_id = await store.add(utterance(clock))
    await store.set_audio_path(row_id, str(audio))
    clock.now += 48 * 3600
    entered, release = asyncio.Event(), asyncio.Event()
    queue = store._queue_audio_deletion

    async def paused_queue(path, conversation_id=None):
        entered.set()
        await release.wait()
        await queue(path, conversation_id)

    monkeypatch.setattr(store, '_queue_audio_deletion', paused_queue)
    purge = asyncio.create_task(store.purge_expired())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        purge.cancel()
        await asyncio.sleep(.01)
        assert not purge.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await purge
        assert await store.undistilled_live() == []
        assert (await store.audio_cleanup_status())['pending'] == 1
    finally:
        release.set()
        await asyncio.gather(purge, return_exceptions=True)
        await store.close()
    reopened = AmbientStore(db, clock=clock, audio_dir=str(tmp_path))
    await reopened.init()
    try:
        assert await reopened.purge_expired() == 0
        assert not audio.exists()
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_cancelled_unlink_is_awaited_and_acknowledged_without_native_work_left(tmp_path, monkeypatch):
    clock = Clock()
    store = AmbientStore(str(tmp_path / 'capture.db'), clock=clock, audio_dir=str(tmp_path))
    await store.init()
    audio = tmp_path / 'clip.flac'
    audio.write_bytes(b'synthetic clip')
    row_id = await store.add(utterance(clock))
    await store.set_audio_path(row_id, str(audio))
    entered, release = threading.Event(), threading.Event()
    unlink = Path.unlink

    def delayed_unlink(path, *args, **kwargs):
        if path == audio:
            entered.set()
            assert release.wait(2)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', delayed_unlink)
    clock.now += 48 * 3600
    purge = asyncio.create_task(store.purge_expired())
    try:
        await until(entered.is_set)
        purge.cancel()
        await asyncio.sleep(.01)
        purge.cancel()
        assert not purge.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await purge
        assert not audio.exists() and (await store.audio_cleanup_status())['pending'] == 0
    finally:
        release.set()
        await asyncio.gather(purge, return_exceptions=True)
        await store.close()


@pytest.mark.asyncio
async def test_clip_expiring_during_disk_read_is_not_returned(tmp_path, monkeypatch):
    clock = Clock()
    store = AmbientStore(str(tmp_path / 'capture.db'), clock=clock, audio_dir=str(tmp_path))
    await store.init()
    audio = tmp_path / 'clip.flac'
    audio.write_bytes(b'synthetic clip')
    row_id = await store.add(utterance(clock))
    await store.set_audio_path(row_id, str(audio))
    read = Path.read_bytes
    entered, release = threading.Event(), threading.Event()

    def delayed_read(path):
        entered.set()
        assert release.wait(2)
        return read(path)

    monkeypatch.setattr(Path, 'read_bytes', delayed_read)
    reading = asyncio.create_task(store.read_audio(row_id))
    try:
        await until(entered.is_set)
        clock.now += 48 * 3600
        release.set()
        assert await reading is None
    finally:
        release.set()
        await asyncio.gather(reading, return_exceptions=True)
        await store.close()


@pytest.mark.asyncio
async def test_partial_write_failure_discards_bytes_via_durable_cleanup(tmp_path, monkeypatch):
    clock = Clock()
    store = AmbientStore(str(tmp_path / 'capture.db'), clock=clock, audio_dir=str(tmp_path))
    await store.init()
    archiver = AudioArchiver(str(tmp_path), enabled=True, discard_audio=store.discard_audio)

    def partial_write(path, mono):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'incomplete native write')
        raise OSError('synthetic I/O failure after first bytes')

    monkeypatch.setattr(archiver, '_write_sync', partial_write)
    try:
        assert await archiver.write('test_room', 1, np.ones(1600, dtype=np.float32)) is None
        assert not list(tmp_path.rglob('*.flac'))
        assert (await store.audio_cleanup_status())['pending'] == 0
        assert archiver.stats['failed'] == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_configured_archive_boundary_cannot_read_or_delete_an_outside_file(tmp_path):
    clock = Clock()
    archive = tmp_path / 'archive'
    archive.mkdir()
    outside = tmp_path / 'outside.flac'
    outside.write_bytes(b'outside archive')
    store = AmbientStore(str(tmp_path / 'capture.db'), clock=clock, audio_dir=str(archive))
    await store.init()
    try:
        row_id = await store.add(utterance(clock))
        await store.set_audio_path(row_id, str(outside))
        assert await store.read_audio(row_id) is None
        clock.now += 48 * 3600
        assert await store.purge_expired() == 1
        assert outside.read_bytes() == b'outside archive'
        assert (await store.audio_cleanup_status())['pending'] == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_new_capture_ids_are_stable_and_do_not_depend_on_sqlite_row_id(tmp_path):
    clock = Clock()
    db = str(tmp_path / 'capture.db')
    store = AmbientStore(db, clock=clock)
    await store.init()
    capture = utterance(clock)
    await store.add(capture)
    await store.close()
    reopened = AmbientStore(db, clock=clock)
    await reopened.init()
    fresh = AmbientStore(str(tmp_path / 'different.db'), clock=clock)
    await fresh.init()
    try:
        rows = await reopened.undistilled_live()
        assert rows[0]['capture_id'] == capture.capture_id
        await fresh.add(utterance(clock))
        other = (await fresh.undistilled_live())[0]
        assert rows[0]['id'] == other['id'] == 1
        assert rows[0]['capture_id'] != other['capture_id']
    finally:
        await fresh.close()
        await reopened.close()


@pytest.mark.asyncio
async def test_transcriber_late_archive_is_removed_after_source_expires(tmp_path, monkeypatch):
    clock = Clock()
    store = AmbientStore(str(tmp_path / 'capture.db'), clock=clock, audio_dir=str(tmp_path))
    await store.init()
    row_id = await store.add(utterance(clock))
    archiver = AudioArchiver(str(tmp_path), enabled=True, discard_audio=store.discard_audio)
    transcriber = AmbientTranscriber(None, None, None, None, None, None, store, [], archiver=archiver)
    entered, release = threading.Event(), threading.Event()
    write = archiver._write_sync

    def late_write(path, mono):
        entered.set()
        assert release.wait(2)
        return write(path, mono)

    monkeypatch.setattr(archiver, '_write_sync', late_write)
    task = asyncio.create_task(transcriber._archive_audio('test_room', row_id, np.ones(1600, dtype=np.float32)))
    try:
        await until(entered.is_set)
        clock.now += 48 * 3600
        assert await store.purge_expired() == 1
        release.set()
        await task
        assert not list(tmp_path.rglob('*.flac'))
        assert await store.read_audio(row_id) is None
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()


@pytest.mark.asyncio
async def test_purge_worker_wakes_for_new_earlier_deadline_and_runs_at_startup(tmp_path):
    import time
    store = AmbientStore(str(tmp_path / 'capture.db'), retention_hours=.0002, audio_dir=str(tmp_path))
    await store.init()
    expired = tmp_path / 'expired.flac'
    expired.write_bytes(b'expired clip')
    await store.add(AmbientUtterance('test_room', time.time()-100, time.time()-99, audio_path=str(expired)))
    await store.close()
    transcriber = AmbientTranscriber(None, None, None, None, None, None, store, [])
    path = AmbientPath(None, transcriber, store, None)
    await path.start()
    try:
        await until(lambda: not expired.exists())
        # Worker had no live rows and would otherwise sleep for an hour.
        now = time.time()
        row_id = await store.add(AmbientUtterance('test_room', now, now+.1))
        clip = tmp_path / 'new.flac'
        clip.write_bytes(b'new live clip')
        assert await store.set_audio_path(row_id, str(clip))
        await until(lambda: not clip.exists())
        assert await store.read_audio(row_id) is None
    finally:
        await path.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('changes', [{'t0': math.nan}, {'t1': math.inf}, {'t0': True}, {'t1': 0}])
async def test_invalid_capture_timestamps_do_not_create_immortal_rows(tmp_path, changes):
    clock = Clock()
    store = AmbientStore(str(tmp_path / 'capture.db'), clock=clock)
    await store.init()
    try:
        with pytest.raises(ValueError):
            await store.add(utterance(clock, **changes))
        assert await store.undistilled_live() == []
    finally:
        await store.close()
