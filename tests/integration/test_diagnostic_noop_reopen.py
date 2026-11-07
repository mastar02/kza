"""Real local historical owners; no synthetic admission or compiled ancestry.

The clock and OS fault observers are the only synthetic seams. Catalogs and
canonical projections come from original record/append/prune operations; the
historical callback, retained worker, lease and close paths stay original.
"""
from __future__ import annotations

from contextlib import asynccontextmanager, closing, contextmanager, nullcontext
import asyncio
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import sqlite3
import stat
import threading

import pytest

from src.memory.privacy_services import PrivacyServices
from src.obs.diagnostic_erasure import DiagnosticServices
from src.obs.diagnostic_store import DiagnosticStore
from src.obs.metrics_emitter import project_diagnostic
from tests.mocks.ambient_graph import Clock

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def _services(tmp_path: Path):
    clock = Clock(2_000_000_000.)
    privacy = PrivacyServices(str(tmp_path / 'privacy.db'), clock=clock)
    driver = DiagnosticServices(privacy, retention_days=1, rotate_bytes=400)
    try:
        await privacy.acall(privacy.initialize)
        yield driver, clock
    finally:
        try:
            await asyncio.wait_for(asyncio.shield(driver.begin_close()), 5)
            assert driver._disk.status()['operations'] == 0
        finally:
            await privacy.aclose()


def _store(path: Path, driver: DiagnosticServices, clock: Clock,
           *, rotate_bytes: int = 400) -> DiagnosticStore:
    return DiagnosticStore(path, clock=clock, retention_days=1,
        rotate_bytes=rotate_bytes, project=project_diagnostic,
        policy=driver.privacy.ledger)


def _seed(path: Path, driver: DiagnosticServices, clock: Clock,
          shape: str = 'rotated') -> None:
    count = {'empty': 0, 'populated': 1, 'rotated': 8, 'large': 400}[shape]
    limit = 10**7 if shape == 'large' else 400
    with closing(_store(path, driver, clock, rotate_bytes=limit)) as store:
        store.prune()
        # Equal timestamps, canonical UUID order, and only public safe fields.
        records = [store.record({'event_type': 'error', 'success': bool(i % 2)})
                   for i in range(count)]
        assert all(record is not None for record in records)
        for record in sorted(records, key=lambda item: item.identifier):
            assert store.append(record) is not None
        store.prune()
    active, rotated = _projections(path)
    if shape == 'rotated':
        assert active and rotated
    if shape == 'large':
        assert len(active) > 65536 and not rotated


def _catalog(path: Path) -> Path:
    return path.with_suffix(path.suffix + '.sqlite3')


def _projections(path: Path) -> tuple[bytes, bytes]:
    return path.read_bytes(), path.with_suffix(path.suffix + '.1').read_bytes()


def _rows(path: Path) -> dict:
    with closing(sqlite3.connect(_catalog(path).resolve().as_uri() + '?mode=ro',
                                 uri=True)) as db:
        return {
            'version': db.execute('PRAGMA user_version').fetchone()[0],
            'state': db.execute('SELECT rowid,key,value FROM state ORDER BY key').fetchall(),
            'metrics': db.execute('SELECT * FROM metrics ORDER BY id').fetchall(),
            'lineage': db.execute('SELECT * FROM lineage ORDER BY record_id,conversation_id').fetchall(),
            'staged': db.execute('SELECT * FROM staged').fetchall(),
        }


def _snapshot(path: Path) -> dict:
    blob = _catalog(path).read_bytes()
    return {'catalog_sha256': hashlib.sha256(blob).hexdigest(),
        'header': blob[:100], 'rows': _rows(path), 'projections': _projections(path)}


def _assert_clean(path: Path) -> None:
    rows = _rows(path)
    state = {key: value for _rowid, key, value in rows['state']}
    assert rows['version'] == 2 and rows['staged'] == []
    assert state['dirty'] == 0 and state['active_bytes'] == path.stat().st_size
    assert not list(path.parent.glob('.diag-*'))
    assert not any(Path(str(_catalog(path)) + suffix).exists()
                   for suffix in ('-journal', '-wal', '-shm'))


def _assert_lease_released(path: Path) -> None:
    fd = os.open(str(_catalog(path)) + '.lock', os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _observe_close(monkeypatch: pytest.MonkeyPatch) -> list:
    original = DiagnosticStore.close
    observed = []

    def close(store: DiagnosticStore) -> None:
        before = store.status()
        lease = os.fstat(store._lease) if store._lease is not None else None
        try:
            return original(store)
        finally:
            observed.append((store, before, lease, store.status()))

    monkeypatch.setattr(DiagnosticStore, 'close', close)
    return observed


@contextmanager
def _io(monkeypatch: pytest.MonkeyPatch, *, before_open=None, before_read=None):
    """Observe only the original comparison descriptors; preserve OS calls."""
    original_open, original_read, original_close = os.open, os.read, os.close
    live, opened, reads = {}, [], []

    def open_(path, flags, *args, **kwargs):
        comparison = bool(flags & os.O_NONBLOCK) and (
            bool(flags & os.O_DIRECTORY) or kwargs.get('dir_fd') is not None)
        if comparison and before_open is not None:
            before_open(path, flags, args, kwargs)
        fd = original_open(path, flags, *args, **kwargs)
        if comparison:
            info = os.fstat(fd)
            entry = (fd, str(path), info.st_dev, info.st_ino, info.st_mode)
            assert fd not in live
            live[fd] = entry
            opened.append(entry)
        return fd

    def read(fd: int, size: int) -> bytes:
        if fd in live:
            reads.append((live[fd], size))
            assert stat.S_ISREG(os.fstat(fd).st_mode), 'nonregular descriptor was read'
            if before_read is not None:
                before_read(fd, size, live[fd])
        return original_read(fd, size)

    def close(fd: int) -> None:
        original_close(fd)
        live.pop(fd, None)

    with monkeypatch.context() as patch:
        patch.setattr(os, 'open', open_)
        patch.setattr(os, 'read', read)
        patch.setattr(os, 'close', close)
        try:
            yield opened, reads
        finally:
            assert live == {}, 'comparison descriptors survived the original return'


async def _history(driver: DiagnosticServices, path: Path) -> int:
    return await asyncio.wait_for(driver._run_history(str(path)), 3)


@contextmanager
def _fifo_rescue(path: Path):
    """Release a broken blocking reader, then fail rather than hang teardown."""
    original_open, original_close = os.open, os.close
    finished, rescued = threading.Event(), threading.Event()
    failures = []

    def rescue():
        if not finished.wait(1):
            rescued.set()
            try:
                fd = original_open(path, os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW)
                try:
                    finished.wait(3)
                finally:
                    original_close(fd)
            except BaseException as error:
                failures.append(error)

    emergency = threading.Thread(target=rescue, daemon=True)
    emergency.start()
    try:
        yield
    finally:
        finished.set()
        emergency.join(3)
        assert not emergency.is_alive() and not failures
        assert not rescued.is_set(), 'emergency FIFO unblock was required'


@pytest.mark.parametrize('shape', ['empty', 'populated', 'rotated', 'large'])
async def test_original_history_reopen_preserves_catalog_bytes_and_rowids(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock, shape)
        before = _snapshot(path)
        closed = _observe_close(monkeypatch)
        with _io(monkeypatch) as (opened, reads):
            for _ in range(3):
                assert await _history(driver, path) == 0
                assert _snapshot(path) == before
                _assert_lease_released(path)
        assert len(closed) == 3 and len({id(row[0]) for row in closed}) == 3
        assert all(old['ready'] and old['writer_lease'] and lease is not None
                   and not new['writer_lease'] and store._closed
                   for store, old, lease, new in closed)
        assert len(opened) == 9 and reads
        assert all(size <= 65536 for _entry, size in reads)
        _assert_clean(path)


@pytest.mark.parametrize('routing', ['stable_alias', 'relative'])
async def test_original_history_accepts_stable_legacy_parent_routes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, routing: str) -> None:
    async with _services(tmp_path) as (driver, clock):
        actual = tmp_path / 'physical' / 'metrics.jsonl'
        await driver._disk.run(_seed, actual, driver, clock)
        if routing == 'stable_alias':
            alias = tmp_path / 'alias'
            alias.symlink_to(actual.parent, target_is_directory=True)
            routed = alias / actual.name
        else:
            monkeypatch.chdir(tmp_path)
            routed = Path('physical') / actual.name
        before = _snapshot(actual)
        assert await _history(driver, routed) == 0
        assert _snapshot(actual) == before
        _assert_lease_released(actual)


@pytest.mark.parametrize('partition', [0, 1])
@pytest.mark.parametrize('fault', ['missing', 'truncated', 'same_length', 'trailing'])
async def test_clean_bit_does_not_authorize_damaged_projection(
        tmp_path: Path, partition: int, fault: str) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock)
        expected, rows = _projections(path), _rows(path)
        target = path if partition == 0 else path.with_suffix(path.suffix + '.1')
        if fault == 'missing':
            target.unlink()
        elif fault == 'truncated':
            target.write_bytes(expected[partition][:-1])
        elif fault == 'same_length':
            target.write_bytes(b'!' + expected[partition][1:])
        else:
            target.write_bytes(expected[partition] + b'\n')
        assert await _history(driver, path) == 0
        assert _projections(path) == expected
        after = _rows(path)
        assert after['metrics'] == rows['metrics'] and after['lineage'] == rows['lineage']
        _assert_clean(path)
        repaired = _snapshot(path)
        assert await _history(driver, path) == 0
        assert _snapshot(path) == repaired


async def test_exact_cache_still_repairs_persisted_active_bytes(tmp_path: Path) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock)
        expected, rows = _projections(path), _rows(path)
        with closing(sqlite3.connect(_catalog(path))) as db:
            db.execute("UPDATE state SET value=value+1 WHERE key='active_bytes'")
            db.commit()
        assert await _history(driver, path) == 0
        assert _projections(path) == expected and _rows(path)['metrics'] == rows['metrics']
        _assert_clean(path)


async def test_failed_comparison_rechecks_corrupted_cache_on_same_instance(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock)
        before = _snapshot(path)
        store = _store(path, driver, clock)
        injected = []

        def fail_read(fd, size, entry):
            if not injected:
                injected.append(entry)
                raise OSError(errno.EIO, 'injected original projection read failure')

        try:
            with _io(monkeypatch, before_read=fail_read):
                with pytest.raises(OSError, match='injected original projection read failure'):
                    await driver._disk.run(store.prune)
            assert injected and not store.status()['ready']
            assert store.status()['writer_lease'] and _snapshot(path) == before
            path.write_bytes(b'!' + before['projections'][0][1:])
            await driver._disk.run(store.prune)
            assert store.status()['ready'] and _projections(path) == before['projections']
            _assert_clean(path)
            repaired = _snapshot(path)
            await driver._disk.run(store.prune)
            assert _snapshot(path) == repaired
        finally:
            await driver._disk.run(store.close)
        _assert_lease_released(path)


async def test_cache_mismatch_keeps_durable_dirty_through_original_replace_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock)
        expected, rows = _projections(path), _rows(path)
        path.write_bytes(b'!' + expected[0][1:])
        original = os.replace
        replaced = []

        def replace(source, destination, *args, **kwargs):
            if Path(destination) == path.with_suffix(path.suffix + '.1'):
                raise OSError(errno.EIO, 'injected second original projection replace')
            result = original(source, destination, *args, **kwargs)
            replaced.append(str(destination))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(os, 'replace', replace)
            with pytest.raises(RuntimeError, match='^diagnostic_retention_failed$'):
                await _history(driver, path)
        assert str(path) in replaced
        assert {key: value for _rowid, key, value in _rows(path)['state']}['dirty'] == 1
        assert _rows(path)['metrics'] == rows['metrics']
        _assert_lease_released(path)
        assert await _history(driver, path) == 0
        assert _projections(path) == expected
        _assert_clean(path)


@pytest.mark.parametrize('kind', ['fifo', 'symlink', 'hardlink', 'directory'])
async def test_unsafe_clean_projection_is_not_read_or_republished(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock)
        before = _snapshot(path)
        target = path.with_suffix(path.suffix + '.1')
        canary = tmp_path / 'external-canary'
        canary.write_bytes(before['projections'][1])
        target.unlink()
        if kind == 'fifo':
            os.mkfifo(target)
        elif kind == 'symlink':
            target.symlink_to(canary)
        elif kind == 'hardlink':
            os.link(canary, target)
        else:
            target.mkdir()
        closed = _observe_close(monkeypatch)
        with _fifo_rescue(target) if kind == 'fifo' else nullcontext():
            with _io(monkeypatch) as (opened, _reads):
                with pytest.raises(RuntimeError, match='^diagnostic_retention_failed$'):
                    await _history(driver, path)
        assert not any(entry[1] == target.name for entry in opened)
        assert canary.read_bytes() == before['projections'][1]
        assert path.read_bytes() == before['projections'][0]
        assert hashlib.sha256(_catalog(path).read_bytes()).hexdigest() == before['catalog_sha256']
        assert len(closed) == 1 and not closed[0][1]['ready']
        assert closed[0][0]._closed and not closed[0][3]['writer_lease']
        _assert_lease_released(path)


async def test_fifo_replacement_at_original_leaf_open_never_blocks_or_reads(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock)
        before = _snapshot(path)
        injected = threading.Event()

        def substitute(name, flags, args, kwargs):
            if kwargs.get('dir_fd') is not None and name == path.name and not injected.is_set():
                path.unlink()
                os.mkfifo(path)
                injected.set()

        with _fifo_rescue(path):
            with _io(monkeypatch, before_open=substitute) as (opened, reads):
                with pytest.raises(RuntimeError, match='^diagnostic_retention_failed$'):
                    await _history(driver, path)
        assert injected.is_set() and stat.S_ISFIFO(path.lstat().st_mode)
        assert any(stat.S_ISFIFO(entry[4]) for entry in opened)
        assert not any(stat.S_ISFIFO(entry[0][4]) for entry in reads)
        assert hashlib.sha256(_catalog(path).read_bytes()).hexdigest() == before['catalog_sha256']
        assert path.with_suffix(path.suffix + '.1').read_bytes() == before['projections'][1]
        _assert_lease_released(path)


@pytest.mark.parametrize('routing', ['physical_replace', 'alias_redirect'])
async def test_parent_drift_cannot_publish_through_another_root(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, routing: str) -> None:
    async with _services(tmp_path) as (driver, clock):
        physical = tmp_path / 'physical'
        actual = physical / 'metrics.jsonl'
        await driver._disk.run(_seed, actual, driver, clock)
        before = _snapshot(actual)
        destination = tmp_path / 'other'
        destination.mkdir()
        canaries = {name: b'foreign destination must remain unchanged\n' for name in
                    (actual.name, actual.name + '.1', _catalog(actual).name)}
        for name, payload in canaries.items():
            (destination / name).write_bytes(payload)
        routed = actual
        alias = tmp_path / 'alias'
        retired = tmp_path / 'retired'
        if routing == 'alias_redirect':
            alias.symlink_to(physical, target_is_directory=True)
            routed = alias / actual.name
        injected = []

        def drift(name, flags, args, kwargs):
            if kwargs.get('dir_fd') is not None and not injected:
                injected.append(True)
                if routing == 'alias_redirect':
                    alias.unlink()
                    alias.symlink_to(destination, target_is_directory=True)
                else:
                    physical.rename(retired)
                    destination.rename(physical)

        with _io(monkeypatch, before_open=drift):
            with pytest.raises(RuntimeError, match='^diagnostic_retention_failed$'):
                await _history(driver, routed)
        assert injected
        source = actual if routing == 'alias_redirect' else retired / actual.name
        foreign = destination if routing == 'alias_redirect' else physical
        assert _snapshot(source) == before
        assert {name: (foreign / name).read_bytes() for name in canaries} == canaries
        assert not list(foreign.glob('.diag-*'))
        _assert_lease_released(source)


@pytest.mark.parametrize('mutation', ['replace_same_bytes', 'rewrite_during_read'])
async def test_leaf_identity_and_in_place_drift_reject_even_equal_final_bytes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock)
        before = _snapshot(path)
        injected = []

        def drift(fd, size, entry):
            if entry[1] == path.name and not injected:
                injected.append(True)
                if mutation == 'replace_same_bytes':
                    path.unlink()
                    path.write_bytes(before['projections'][0])
                else:
                    original = path.stat()
                    path.write_bytes(before['projections'][0])
                    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))

        with _io(monkeypatch, before_read=drift):
            with pytest.raises(RuntimeError, match='^diagnostic_retention_failed$'):
                await _history(driver, path)
        assert injected and _snapshot(path) == before
        _assert_lease_released(path)


async def test_dirty_matching_cache_still_uses_original_publication(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock)
        expected = _projections(path)
        with closing(sqlite3.connect(_catalog(path))) as db:
            db.execute("UPDATE state SET value=1 WHERE key='dirty'")
            db.commit()
        original = os.replace
        targets = []

        def replace(source, target, *args, **kwargs):
            result = original(source, target, *args, **kwargs)
            targets.append(Path(target))
            return result

        monkeypatch.setattr(os, 'replace', replace)
        assert await _history(driver, path) == 0
        assert targets == [path, path.with_suffix(path.suffix + '.1')]
        assert _projections(path) == expected
        _assert_clean(path)


async def test_real_version_one_catalog_migrates_once_then_reopens_without_writes(
        tmp_path: Path) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock)
        expected, rows = _projections(path), _rows(path)
        with closing(sqlite3.connect(_catalog(path))) as db:
            db.execute('DROP INDEX diagnostic_conversation')
            db.execute('DROP TABLE lineage')
            db.execute('PRAGMA user_version=1')
            db.commit()
        assert await _history(driver, path) == 0
        assert _projections(path) == expected and _rows(path)['metrics'] == rows['metrics']
        _assert_clean(path)
        migrated = _snapshot(path)
        assert await _history(driver, path) == 0
        assert _snapshot(path) == migrated


async def test_matching_invalid_catalog_payload_cannot_become_ready(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock, 'populated')
        # Explicit invalid input: preserve the already projected safe bytes but
        # alter the stored schema type. A projection-only comparison would pass.
        with closing(sqlite3.connect(_catalog(path))) as db:
            db.execute("UPDATE metrics SET payload=replace(payload, '\"diagnostic_schema\": 1', '\"diagnostic_schema\": true')")
            db.commit()
            payload = db.execute('SELECT payload FROM metrics').fetchone()[0]
        assert 'true' in payload
        before = _snapshot(path)
        closed = _observe_close(monkeypatch)
        with pytest.raises(RuntimeError, match='^diagnostic_retention_failed$'):
            await _history(driver, path)
        assert len(closed) == 1 and not closed[0][1]['ready']
        assert _snapshot(path) == before
        _assert_lease_released(path)


async def test_cancelled_history_observers_retain_real_comparison_and_writer_lease(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _services(tmp_path) as (driver, clock):
        path = tmp_path / 'diagnostics' / 'metrics.jsonl'
        await driver._disk.run(_seed, path, driver, clock)
        before = _snapshot(path)
        entered, release = threading.Event(), threading.Event()
        held = []
        closed = _observe_close(monkeypatch)

        def hold(fd, size, entry):
            if entry[1] == path.name and not held:
                held.append((fd, os.fstat(fd)))
                entered.set()
                assert release.wait(3), 'comparison worker was not released'

        async def observe_close(task):
            await asyncio.shield(task)

        observer = close_observer = None
        with _io(monkeypatch, before_read=hold):
            try:
                observer = asyncio.create_task(driver._run_history(str(path)))
                async with asyncio.timeout(1):
                    while not entered.is_set():
                        await asyncio.sleep(.001)
                retained, = tuple(driver._operations)
                assert not retained.done() and driver._disk.status()['operations'] == 1
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                closing_task = driver.begin_close()
                close_observer = asyncio.create_task(observe_close(closing_task))
                await asyncio.sleep(0)
                close_observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await close_observer
                assert driver._closing is closing_task and not closing_task.done()
                assert not retained.done() and not closed
                assert driver._disk.status()['operations'] == 1
                fd, original = held[0]
                actual = os.fstat(fd)
                assert (actual.st_dev, actual.st_ino) == (original.st_dev, original.st_ino)
                # This is a second real store. Its original nonblocking lease
                # rejection occurs before it can open a SQLite connection.
                with closing(_store(path, driver, clock)) as contender:
                    with pytest.raises(RuntimeError, match='^diagnostic_writer_busy$'):
                        contender.prune()
                assert not retained.done() and not closing_task.done()
                release.set()
                await asyncio.wait_for(asyncio.shield(closing_task), 5)
                assert retained.done() and retained.result() == 0
            finally:
                release.set()
                if observer is not None:
                    await asyncio.wait_for(asyncio.gather(observer, return_exceptions=True), 5)
                if close_observer is not None:
                    await asyncio.wait_for(asyncio.gather(close_observer, return_exceptions=True), 5)
                await asyncio.wait_for(asyncio.shield(driver.begin_close()), 5)
        assert not driver._operations
        assert driver._disk.status() == {'closed': True, 'operations': 0, 'capacity': 1}
        assert len(closed) == 2
        historical = [row for row in closed if row[2] is not None]
        assert len(historical) == 1 and historical[0][1]['ready']
        assert historical[0][0]._closed and not historical[0][3]['writer_lease']
        assert _snapshot(path) == before
        _assert_lease_released(path)
