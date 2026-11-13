"""Original raw tempfile descriptor ownership in the unmanaged account publisher."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.users import user_manager as catalog
from src.users.user_manager import PermissionLevel, UserManager


@pytest.fixture
def descriptors(monkeypatch):
    allocated, closes = [], []
    original_mkstemp, original_close = catalog.tempfile.mkstemp, catalog.os.close

    def mkstemp(*args, **kwargs):
        allocated.append(original_mkstemp(*args, **kwargs))
        return allocated[-1]

    def close(descriptor):
        if allocated and descriptor == allocated[-1][0]:
            closes.append(descriptor)
        return original_close(descriptor)

    monkeypatch.setattr(catalog.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(catalog.os, "close", close)
    return allocated, closes


def _fail_fdopen(monkeypatch, allocated, before=None):
    """Fail the first transfer stage (FileIO construction) for the allocated fd."""
    original = catalog.FileIO

    def file_io(descriptor, *args, **kwargs):
        if not allocated or descriptor != allocated[-1][0]:
            return original(descriptor, *args, **kwargs)
        if before is not None:
            before(allocated[-1][1])
        raise OSError("synthetic fdopen failure")

    monkeypatch.setattr(catalog, "FileIO", file_io)


class _CountingFileIO(catalog.FileIO):
    """Real FileIO that records each close of the account descriptor."""

    closes: list = []

    def close(self):
        if not self.closed:
            type(self).closes.append(self.fileno())
        return super().close()


def _assert_closed(descriptor):
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_fdopen_failure_closes_raw_descriptor_once_and_removes_own_temp(tmp_path, monkeypatch, descriptors):
    allocated, closes = descriptors
    users = UserManager(str(tmp_path / "users.json"))
    _fail_fdopen(monkeypatch, allocated)
    with pytest.raises(OSError, match="synthetic fdopen failure"):
        users.add_user("Synthetic", PermissionLevel.GUEST)
    descriptor, temporary = allocated[-1]
    assert closes == [descriptor]
    _assert_closed(descriptor)
    assert not Path(temporary).exists() and not users.data_path.exists()
    assert users._users == {} and users._version == 0


def test_fdopen_failure_with_substituted_canary_preserves_foreign_file(tmp_path, monkeypatch, descriptors):
    allocated, closes = descriptors
    users = UserManager(str(tmp_path / "users.json"))

    def substitute(temporary):
        canary = tmp_path / "canary"
        canary.write_bytes(b"canary")
        os.replace(canary, temporary)

    _fail_fdopen(monkeypatch, allocated, substitute)
    with pytest.raises(OSError, match="synthetic fdopen failure"):
        users.add_user("Synthetic", PermissionLevel.GUEST)
    descriptor, temporary = allocated[-1]
    assert closes == [descriptor]
    _assert_closed(descriptor)
    assert Path(temporary).read_bytes() == b"canary" and not users.data_path.exists()


def test_handler_fstat_failure_still_closes_once_and_keeps_unproven_name(tmp_path, monkeypatch, descriptors):
    allocated, closes = descriptors
    users = UserManager(str(tmp_path / "users.json"))
    original_fstat = catalog.os.fstat

    def fstat(descriptor):
        if allocated and descriptor == allocated[-1][0]:
            raise OSError("synthetic fstat failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(catalog.os, "fstat", fstat)
    _fail_fdopen(monkeypatch, allocated)
    with pytest.raises(OSError, match="synthetic fdopen failure"):
        users.add_user("Synthetic", PermissionLevel.GUEST)
    descriptor, temporary = allocated[-1]
    assert closes == [descriptor]
    monkeypatch.setattr(catalog.os, "fstat", original_fstat)
    _assert_closed(descriptor)
    assert Path(temporary).exists() and not users.data_path.exists()


def test_publication_destination_is_the_allocated_tempfile_inode(tmp_path, monkeypatch, descriptors):
    allocated, closes = descriptors
    users = UserManager(str(tmp_path / "users.json"))
    identities = []
    original = catalog.FileIO

    def fdopen(descriptor, *args, **kwargs):
        if allocated and descriptor == allocated[-1][0]:
            info = os.fstat(descriptor)
            identities.append((info.st_dev, info.st_ino))
        return original(descriptor, *args, **kwargs)

    monkeypatch.setattr(catalog, "FileIO", fdopen)
    for name in ("Synthetic A", "Synthetic B"):
        user, _ = users.add_user(name, PermissionLevel.GUEST)
        assert user is not None
        info = os.lstat(users.data_path)
        assert (info.st_dev, info.st_ino) == identities[-1]
        _assert_closed(allocated[-1][0])
    assert closes == [] and not list(tmp_path.glob(".accounts-*"))


def test_wrapper_failure_after_fileio_closes_only_through_owning_file(tmp_path, monkeypatch, descriptors):
    allocated, closes = descriptors
    users = UserManager(str(tmp_path / "users.json"))
    _CountingFileIO.closes = []
    monkeypatch.setattr(catalog, "FileIO", _CountingFileIO)

    def wrapper(*args, **kwargs):
        raise MemoryError("synthetic wrapper failure")

    monkeypatch.setattr(catalog, "TextIOWrapper", wrapper)
    with pytest.raises(MemoryError, match="synthetic wrapper failure"):
        users.add_user("Synthetic", PermissionLevel.GUEST)
    descriptor, temporary = allocated[-1]
    # Owned by FileIO: exactly one close through it, never os.close of the number.
    assert _CountingFileIO.closes == [descriptor] and closes == []
    _assert_closed(descriptor)
    assert not Path(temporary).exists() and not users.data_path.exists()
    assert users._users == {} and users._version == 0


def test_buffered_wrapper_failure_also_closes_only_through_owning_file(tmp_path, monkeypatch, descriptors):
    allocated, closes = descriptors
    users = UserManager(str(tmp_path / "users.json"))
    _CountingFileIO.closes = []
    monkeypatch.setattr(catalog, "FileIO", _CountingFileIO)
    monkeypatch.setattr(catalog, "BufferedWriter", lambda raw: (_ for _ in ()).throw(MemoryError("buffer")))
    with pytest.raises(MemoryError, match="buffer"):
        users.add_user("Synthetic", PermissionLevel.GUEST)
    descriptor, temporary = allocated[-1]
    assert _CountingFileIO.closes == [descriptor] and closes == []
    _assert_closed(descriptor)
    assert not Path(temporary).exists()


def test_two_stage_transfer_writes_bytes_identical_to_original_fdopen(tmp_path, monkeypatch, descriptors):
    monkeypatch.setattr(catalog.time, "time", lambda: 1234.5)
    users = UserManager(str(tmp_path / "users.json"))
    user, _ = users.add_user("Señora Ñandú", PermissionLevel.ADULT)
    assert user is not None
    published = users.data_path.read_bytes()
    data = json.loads(published)
    descriptor, reference = catalog.tempfile.mkstemp(dir=tmp_path)
    with os.fdopen(descriptor, "w") as stream:  # The original publisher call.
        json.dump(data, stream, indent=2)
    assert Path(reference).read_bytes() == published


@pytest.mark.parametrize("reuse", [False, True], ids=["fresh_inode", "reused_inode_number"])
def test_cleanup_never_deletes_a_substitute_with_the_freed_own_inode_number(
    tmp_path, monkeypatch, descriptors, reuse,
):
    """ext4 hands a freed inode number to the next allocation at once.

    The substitute is written at the own name after the own temporary is
    unlinked. With reuse=True its lstat reports the own (st_dev, st_ino) on
    every platform, as ext4 does, so only the pinned own inode tells them apart.
    """
    allocated, _closes = descriptors
    users = UserManager(str(tmp_path / "users.json"))
    owned, calls, real_lstat = {}, [], os.lstat

    def lstat(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        own = owned.get(os.fspath(path))
        if own is None or (info.st_dev, info.st_ino) == own:
            return info
        fields = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
        fields.update(st_dev=own[0], st_ino=own[1])
        return SimpleNamespace(**fields)

    def before_save():
        calls.append(bool(allocated))
        if allocated:  # After the own temporary was written and closed.
            temporary = allocated[-1][1]
            info = real_lstat(temporary)
            owned[temporary] = (info.st_dev, info.st_ino)
            Path(temporary).unlink()
            Path(temporary).write_bytes(b"canary")
            return False
        return None

    if reuse:
        monkeypatch.setattr(catalog.os, "lstat", lstat)
    with pytest.raises(PermissionError, match="user_catalog_authority_retired"):
        users.add_user("Synthetic", PermissionLevel.GUEST, before_save=before_save)
    descriptor, temporary = allocated[-1]
    _assert_closed(descriptor)
    assert calls == [False, False, True] and Path(temporary).read_bytes() == b"canary"
    assert not users.data_path.exists() and users._users == {}


def test_a_failed_pin_closes_the_raw_descriptor_once_and_keeps_the_unproven_name(tmp_path, monkeypatch, descriptors):
    allocated, closes = descriptors
    users = UserManager(str(tmp_path / "users.json"))
    real_dup, injected = os.dup, []

    def dup(descriptor):
        if allocated and descriptor == allocated[-1][0] and not injected:
            injected.append(descriptor)
            raise OSError("synthetic dup failure")
        return real_dup(descriptor)

    monkeypatch.setattr(catalog.os, "dup", dup)
    with pytest.raises(OSError, match="synthetic dup failure"):
        users.add_user("Synthetic", PermissionLevel.GUEST)
    descriptor, temporary = allocated[-1]
    assert injected == [descriptor] and closes == [descriptor]
    _assert_closed(descriptor)
    assert Path(temporary).exists() and not users.data_path.exists() and users._users == {}
