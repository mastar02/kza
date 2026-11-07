"""Real catalog leases/probes; managed classification here issues no authority."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from src.core import runtime_admission
from src.obs.diagnostic_erasure import DiagnosticServices
from src.obs.diagnostic_store import DiagnosticStore
from src.obs.metrics_emitter import DiagnosticFailure, project_diagnostic


async def _until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


def _services():
    # Empty real catalogs need no policy decision; these tests do not certify
    # controller ancestry or fabricate a receipt/verified RuntimeAdmission.
    ledger = SimpleNamespace(clock=time.time, committed_revoked_conversations=lambda: ())
    return DiagnosticServices(SimpleNamespace(ledger=ledger))


@pytest.fixture
def managed_stage(monkeypatch):
    stage = runtime_admission._stage()
    monkeypatch.setattr(runtime_admission, "_verified_runs", {("gate-unit", "unit-binding")})
    monkeypatch.setattr(runtime_admission, "_installed_runs", {"gate-unit"})
    monkeypatch.delenv(runtime_admission.TICKET_ENV, raising=False)
    return stage


def _seed(path):
    store = DiagnosticStore(path, clock=time.time, retention_days=90,
                            rotate_bytes=1024 * 1024, project=project_diagnostic)
    try:
        store.prune()
    finally:
        store.close()


@pytest.mark.parametrize("first", ["history", "probe"])
@pytest.mark.asyncio
async def test_real_historical_lease_and_original_probe_serialize_in_both_orders(
        tmp_path, monkeypatch, managed_stage, first):
    stage = managed_stage
    path = tmp_path / "state/data/metrics.jsonl"
    _seed(path)
    services = _services()
    original_mutex = stage._ADMISSION_VALIDATION_LOCK
    original_flock, original_close = fcntl.flock, DiagnosticStore.close
    history_attempted, probe_attempted = threading.Event(), threading.Event()
    history_acquired, probe_acquired = threading.Event(), threading.Event()
    close_entered, release_close, release_probe = threading.Event(), threading.Event(), threading.Event()
    stores, errors, locks = [], [], []

    class ObservedMutex:
        def __enter__(self):
            event = probe_attempted if threading.current_thread().name == "gate-probe" else history_attempted
            event.set()
            original_mutex.acquire()
            return self

        def __exit__(self, *args):
            original_mutex.release()

    def observe_flock(fd, operation):
        result = original_flock(fd, operation)
        if operation == fcntl.LOCK_EX | fcntl.LOCK_NB:
            actor = "probe" if threading.current_thread().name == "gate-probe" else "history"
            info = os.fstat(fd)
            locks.append((actor, info.st_dev, info.st_ino))
            (probe_acquired if actor == "probe" else history_acquired).set()
            if actor == "probe" and first == "probe":
                assert release_probe.wait(3), "original probe was not released"
        return result

    def hold_original_close(store):
        if store.path == path:
            stores.append((store, store._lease))
            close_entered.set()
            try:
                assert release_close.wait(3), "original catalog close was not released"
            finally:
                original_close(store)
            return None
        return original_close(store)

    def probe():
        try:
            stage._assert_writers_closed(tmp_path / "state")
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(stage, "_ADMISSION_VALIDATION_LOCK", ObservedMutex())
    monkeypatch.setattr(stage.fcntl, "flock", observe_flock)
    monkeypatch.setattr(DiagnosticStore, "close", hold_original_close)
    thread = threading.Thread(target=probe, name="gate-probe", daemon=True)
    observer = None
    try:
        if first == "history":
            observer = asyncio.create_task(services._run_history(str(path)))
            await _until(close_entered.is_set)
            assert history_acquired.is_set() and stores[0][0]._lease is not None
            thread.start()
            await _until(probe_attempted.is_set)
            assert not probe_acquired.is_set() and thread.is_alive()
        else:
            thread.start()
            await _until(probe_acquired.is_set)
            observer = asyncio.create_task(services._run_history(str(path)))
            await _until(history_attempted.is_set)
            assert not history_acquired.is_set() and not close_entered.is_set()
            assert not observer.done()
            release_probe.set()
            await _until(close_entered.is_set)
        # Reaching here on the event loop while the native owner waits proves
        # that acquiring the shared gate did not block the event-loop thread.
        assert services._disk.status()["operations"] == 1
        os.fstat(stores[0][1])
        release_close.set()
        assert await asyncio.wait_for(asyncio.shield(observer), 3) == 0
    finally:
        release_probe.set()
        release_close.set()
        if observer is not None:
            await asyncio.wait_for(asyncio.gather(observer, return_exceptions=True), 5)
        await asyncio.wait_for(asyncio.shield(services.begin_close()), 5)
        if thread.ident is not None:
            await asyncio.to_thread(thread.join, 5)
    assert not thread.is_alive() and not errors
    assert [row[0] for row in locks] == (["history", "probe"] if first == "history" else ["probe", "history"])
    assert locks[0][1:] == locks[1][1:]
    assert len(stores) == 1 and stores[0][0]._closed and stores[0][0]._lease is None
    with pytest.raises(OSError):
        os.fstat(stores[0][1])
    assert services._disk.status() == {"closed": True, "operations": 0, "capacity": 1}
    assert not services._operations


@pytest.mark.asyncio
async def test_cancelled_observers_keep_real_history_gate_and_close_owner(
        tmp_path, monkeypatch, managed_stage):
    stage = managed_stage
    path = tmp_path / "state/data/metrics.jsonl"
    _seed(path)
    services = _services()
    entered, release, attempted = threading.Event(), threading.Event(), threading.Event()
    original_close, original_mutex = DiagnosticStore.close, stage._ADMISSION_VALIDATION_LOCK
    stores, errors = [], []

    class ObservedMutex:
        def __enter__(self):
            if threading.current_thread().name == "cancel-probe":
                attempted.set()
            original_mutex.acquire()
            return self

        def __exit__(self, *args):
            original_mutex.release()

    def hold_close(store):
        if store.path == path:
            stores.append((store, store._lease))
            entered.set()
            try:
                assert release.wait(3), "retained native close did not get released"
            finally:
                original_close(store)
            return None
        return original_close(store)

    def probe():
        try:
            stage._assert_writers_closed(tmp_path / "state")
        except BaseException as error:
            errors.append(error)

    async def observe_close(task):
        await asyncio.shield(task)

    monkeypatch.setattr(stage, "_ADMISSION_VALIDATION_LOCK", ObservedMutex())
    monkeypatch.setattr(DiagnosticStore, "close", hold_close)
    observer = asyncio.create_task(services._run_history(str(path)))
    thread = threading.Thread(target=probe, name="cancel-probe", daemon=True)
    closing = close_observer = None
    try:
        await _until(entered.is_set)
        retained = next(iter(services._operations))
        assert not retained.done()
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        closing = services.begin_close()
        close_observer = asyncio.create_task(observe_close(closing))
        await asyncio.sleep(0)
        close_observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_observer
        thread.start()
        await _until(attempted.is_set)
        assert not retained.done() and not closing.done() and thread.is_alive()
        assert services._closing is closing and stores[0][0]._lease == stores[0][1]
        assert services._disk.status()["operations"] == 1
        os.fstat(stores[0][1])
        release.set()
        await asyncio.wait_for(asyncio.shield(closing), 5)
        assert retained.done() and retained.result() == 0
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(observer, return_exceptions=True), 5)
        await asyncio.wait_for(asyncio.shield(services.begin_close()), 5)
        if thread.ident is not None:
            await asyncio.to_thread(thread.join, 5)
    assert not thread.is_alive() and not errors
    assert len(stores) == 1 and stores[0][0]._closed and stores[0][0]._lease is None
    with pytest.raises(OSError):
        os.fstat(stores[0][1])
    assert not services._operations
    assert services._disk.status() == {"closed": True, "operations": 0, "capacity": 1}


def test_external_real_writer_still_rejects_history_and_original_probe(tmp_path, managed_stage):
    stage = managed_stage
    path = tmp_path / "state/data/metrics.jsonl"
    _seed(path)
    lock_path = Path(str(path.with_suffix(path.suffix + ".sqlite3")) + ".lock")
    child = "\n".join([
        "import fcntl,json,os,sys",
        "fd=os.open(sys.argv[1],os.O_RDWR|os.O_NOFOLLOW)",
        "try:",
        " fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)",
        " info=os.fstat(fd)",
        " os.write(1,(json.dumps([os.getpid(),info.st_dev,info.st_ino])+'\\n').encode())",
        " os.read(0,1)",
        "finally: os.close(fd)",
    ])
    process = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", child, str(lock_path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    services = _services()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(3), "external writer failed to acquire real lease"
        identity = json.loads(os.read(process.stdout.fileno(), 4096))
        info = lock_path.stat()
        assert identity == [process.pid, info.st_dev, info.st_ino] and process.poll() is None
        assert services._invoke_history(str(path), None) is DiagnosticFailure.FAILED
        with pytest.raises(ValueError, match="^runtime_admission_writer_active$"):
            stage._assert_writers_closed(tmp_path / "state")
        assert process.poll() is None
        process.stdin.write(b"x")
        process.stdin.flush()
        process.stdin.close()
        assert process.wait(5) == 0
        assert services._invoke_history(str(path), None) == 0
        stage._assert_writers_closed(tmp_path / "state")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(5)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


@pytest.mark.parametrize("signal", ["marker", "ticket", "installed"])
def test_gate_classification_is_sticky_scheduling_only(tmp_path, monkeypatch, signal):
    stage = runtime_admission._stage()
    monkeypatch.setattr(runtime_admission, "_verified_runs", set())
    monkeypatch.setattr(runtime_admission, "_installed_runs", set())
    monkeypatch.delenv(runtime_admission.TICKET_ENV, raising=False)
    path = tmp_path / "state/data/metrics.jsonl"
    path.parent.mkdir(parents=True)
    if signal == "marker":
        # Marker bytes are deliberately not an admission: selection only
        # serializes with probes and does not call a validator or grant access.
        (path.parent.parent / runtime_admission.MARKER).write_text("{}\n")
    elif signal == "ticket":
        monkeypatch.setenv(runtime_admission.TICKET_ENV, "not-an-authority-ticket")
    else:
        runtime_admission._verified_runs.add(("installed", "binding"))
        runtime_admission._installed_runs.add("installed")
        assert not (path.parent.parent / runtime_admission.MARKER).exists()
        assert runtime_admission.TICKET_ENV not in os.environ
    assert runtime_admission.diagnostic_history_validation_gate(path) is stage._ADMISSION_VALIDATION_LOCK


@pytest.mark.asyncio
async def test_unmanaged_original_history_does_not_load_admission_companion(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_admission, "_verified_runs", set())
    monkeypatch.setattr(runtime_admission, "_installed_runs", set())
    monkeypatch.delenv(runtime_admission.TICKET_ENV, raising=False)

    def unexpected():
        raise AssertionError("unmanaged operation loaded admission companion")

    monkeypatch.setattr(runtime_admission, "_stage", unexpected)
    services = _services()
    path = tmp_path / "metrics.jsonl"
    try:
        assert await services._run_history(str(path)) == 0
    finally:
        await asyncio.wait_for(asyncio.shield(services.begin_close()), 5)
    assert path.is_file() and path.with_suffix(path.suffix + ".sqlite3").is_file()
    assert not services._operations
    assert services._disk.status() == {"closed": True, "operations": 0, "capacity": 1}
