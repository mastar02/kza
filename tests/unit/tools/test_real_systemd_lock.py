"""The rehearsal manager lock of tests/mocks/real_systemd.py, without a user manager (review AWR-R1).

The lock is an ``flock`` on ``$XDG_RUNTIME_DIR/kza-rehearsal-manager.lock``;
here ``XDG_RUNTIME_DIR`` is a private temporary directory, so these tests
never touch the account's real lock or its user manager.
"""
from __future__ import annotations

import fcntl
import os

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


def _holder(lock: int, lock_wait: float = 0.0) -> real_systemd.UserSystemd:
    """A UserSystemd holding ``lock`` whose rehearsal setup never ran (no unit, no systemctl)."""
    holder = real_systemd.UserSystemd.__new__(real_systemd.UserSystemd)
    holder._lock, holder.lock_wait, holder.cleaned = lock, lock_wait, None
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
