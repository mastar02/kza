"""Explicit private modes under the managed root, whatever the umask (server umask 0002)."""
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

from scripts import adoption
from scripts import stage_runtime as stage


@pytest.fixture(params=[0o002, 0o022, 0o277], ids=["umask_002", "umask_022", "umask_277"])
def umask(request):
    previous = os.umask(request.param)
    try:
        yield request.param
    finally:
        os.umask(previous)


def mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def test_private_directories_creates_every_missing_level_exactly_0700(tmp_path, umask):
    target = tmp_path / "a" / "b" / "c"
    stage.private_directories(target)
    assert [mode(path) for path in (tmp_path / "a", tmp_path / "a" / "b", target)] == [0o700] * 3
    stage.private_directories(target)  # exist_ok
    assert mode(target) == 0o700


def test_private_directories_keeps_an_existing_older_level_and_refuses_a_file(tmp_path, umask):
    older = tmp_path / "admission"
    older.mkdir()
    os.chmod(older, 0o775)
    stage.private_directories(older / "manual-components" / "heads")
    assert mode(older) == 0o775
    assert mode(older / "manual-components") == mode(older / "manual-components" / "heads") == 0o700
    (tmp_path / "file").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        stage.private_directories(tmp_path / "file")


def test_open_private_file_creates_0600_and_keeps_an_existing_files_mode(tmp_path, umask):
    lock = tmp_path / ".stage.lock"
    os.close(stage.open_private_file(lock, os.O_WRONLY | os.O_APPEND))
    assert mode(lock) == 0o600
    os.chmod(lock, 0o664)
    os.close(stage.open_private_file(lock, os.O_WRONLY | os.O_APPEND))
    assert mode(lock) == 0o664
    link = tmp_path / "link"
    link.symlink_to(lock)
    with pytest.raises(OSError):
        stage.open_private_file(link, os.O_WRONLY | os.O_APPEND)


def test_write_private_file_is_exactly_0600_and_never_overwrites(tmp_path, umask):
    path = tmp_path / "release.json"
    stage.write_private_file(path, b"data")
    assert path.read_bytes() == b"data" and mode(path) == 0o600
    with pytest.raises(FileExistsError):
        stage.write_private_file(path, b"other")
    assert path.read_bytes() == b"data"


def test_locked_root_creates_the_stage_lock_0600(tmp_path, umask):
    root = tmp_path / "releases"
    with stage._locked_root(root):
        pass
    assert mode(root) == 0o700 and mode(root / ".stage.lock") == 0o600


@pytest.mark.parametrize("copy_umask", [0o002, 0o022])
def test_the_private_state_copy_is_0700_and_0600_whatever_the_source(tmp_path, copy_umask):
    """shutil.copytree itself needs an owner-writable umask, so 0o277 is not a case here."""
    source = tmp_path / "source"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "x.db").write_bytes(b"x")
    os.chmod(source / "sub" / "x.db", 0o664)
    os.chmod(source / "sub", 0o775)
    previous = os.umask(copy_umask)
    try:
        stage._copy_private_tree(source, tmp_path / "copy")
    finally:
        os.umask(previous)
    copy = tmp_path / "copy"
    assert [mode(copy), mode(copy / "sub"), mode(copy / "sub" / "x.db")] == [0o700, 0o700, 0o600]


def test_a_level_another_writer_creates_first_is_kept_as_it_is(tmp_path, monkeypatch, umask):
    """The mkdir race: a level that appears between the check and mkdir is accepted unchanged."""
    real_mkdir, raced = os.mkdir, []

    def mkdir(path, mode=0o777, *args, **kwargs):
        if not raced:
            raced.append(path)
            real_mkdir(path, 0o700)
            os.chmod(path, 0o775)  # An older version's level, created just before ours.
            raise FileExistsError(path)
        return real_mkdir(path, mode, *args, **kwargs)
    monkeypatch.setattr(stage.os, "mkdir", mkdir)
    stage.private_directories(tmp_path / "admission" / "heads")
    assert raced == [tmp_path / "admission"]
    assert mode(tmp_path / "admission") == 0o775 and mode(tmp_path / "admission" / "heads") == 0o700


def test_the_schema_only_child_runs_with_a_private_umask(tmp_path, monkeypatch):
    """Whatever the controller's umask, the child creates private entries in the prepared copy."""
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(returncode=0, stdout='{"phase": "schema", "schema_only": true}\n', stderr="")
    monkeypatch.setattr(adoption.subprocess, "run", run)
    state = tmp_path / "copy" / "state"
    assert adoption.run_child(sys.executable, tmp_path / "runtime", state, "schema")["schema_only"] is True
    assert [call["umask"] for call in calls] == [0o077] and calls[0]["cwd"] == os.fspath(state.parent)
