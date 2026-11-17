"""The rehearsal manager lock of tests/mocks/real_systemd.py, without a user manager (review AWR-R1).

The lock is an ``flock`` on ``$XDG_RUNTIME_DIR/kza-rehearsal-manager.lock``;
here ``XDG_RUNTIME_DIR`` is a private temporary directory, so these tests
never touch the account's real lock or its user manager.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time

import pytest

from tests.mocks import real_systemd


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert real_systemd._HELD == []
    yield tmp_path
    for descriptor in list(real_systemd._HELD):
        real_systemd._release(descriptor)


def _taken_elsewhere(path) -> bool:
    """Whether another open file description (another process, in effect) is refused the lock now."""
    descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)
    return False


def _holder(lock: int, lock_wait: float = 0.0, reports: list[str] | None = None) -> real_systemd.UserSystemd:
    """A UserSystemd holding ``lock`` whose rehearsal setup never ran (no unit, no systemctl)."""
    holder = real_systemd.UserSystemd.__new__(real_systemd.UserSystemd)
    holder._lock, holder.lock_wait, holder.lock_wait_reports, holder.cleaned = lock, lock_wait, reports or [], None
    return holder


def test_a_retried_cleanup_never_releases_a_later_holders_lock(runtime, monkeypatch):
    """A cleanup() whose work raised already released the lock; its retry must not close the descriptor
    number the kernel has since handed to a later holder (AWR-R1)."""
    def failing_cleanup(self):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(real_systemd.UserSystemd, "_cleanup", failing_cleanup)
    first = _holder(*real_systemd._acquire())
    stale = first._lock
    with pytest.raises(RuntimeError, match="cleanup failed"):
        first.cleanup()
    assert real_systemd._HELD == [] and first._lock is None
    later = _holder(*real_systemd._acquire())
    assert later._lock == stale  # the kernel hands out the lowest free number: the case AWR-R1 is about
    with pytest.raises(RuntimeError, match="cleanup failed"):
        first.cleanup()
    assert real_systemd._HELD == [later._lock]
    assert _taken_elsewhere(runtime / real_systemd.LOCK_NAME)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        later.cleanup()
    assert real_systemd._HELD == [] and not _taken_elsewhere(runtime / real_systemd.LOCK_NAME)


def test_a_failed_setup_releases_the_lock_once(runtime, monkeypatch):
    def failing_setup(self, workdir):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(real_systemd.UserSystemd, "_setup", failing_setup)
    with pytest.raises(RuntimeError, match="setup failed"):
        real_systemd.UserSystemd(runtime / "work")
    assert real_systemd._HELD == [] and not _taken_elsewhere(runtime / real_systemd.LOCK_NAME)


def test_a_long_wait_names_the_holder_and_the_holder_is_recorded(runtime, monkeypatch, caplog):
    """AWR-R4: the holder writes its pid, cwd and test into the lock file; a waiter reports it every
    LOCK_WAIT_REPORT_SECONDS (shortened here) and still gets the lock once it is free."""
    monkeypatch.setattr(real_systemd, "LOCK_WAIT_REPORT_SECONDS", 0.2)
    path = runtime / real_systemd.LOCK_NAME
    other = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)  # a holder outside _HELD
    fcntl.flock(other, fcntl.LOCK_EX)
    os.pwrite(other, json.dumps({"pid": 4242, "cwd": "/elsewhere", "test": "other::test"}).encode(), 0)
    acquired: list[tuple[int, float]] = []
    waiter = threading.Thread(target=lambda: acquired.append(real_systemd._acquire()))
    with caplog.at_level(logging.WARNING, logger=real_systemd.__name__):
        waiter.start()
        deadline = time.monotonic() + 30
        while not any("other::test" in record.getMessage() for record in caplog.records):
            assert time.monotonic() < deadline and waiter.is_alive(), caplog.text
            time.sleep(0.05)
        assert not acquired
        os.close(other)
        waiter.join(30)
    assert not waiter.is_alive() and real_systemd._HELD == [acquired[0][0]]
    report = next(record.getMessage() for record in caplog.records if "other::test" in record.getMessage())
    assert '"pid": 4242' in report and "/elsewhere" in report, report
    assert report in acquired[0][2], acquired  # the same report goes to the sweep or cleanup evidence
    holder = json.loads(path.read_text())
    assert holder["pid"] == os.getpid() and holder["cwd"] == os.getcwd(), holder
    assert "test_a_long_wait_names_the_holder" in holder["test"], holder
    real_systemd._release(acquired[0][0])
    assert path.read_text() == "" and not _taken_elsewhere(path)
