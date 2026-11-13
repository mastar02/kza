"""The prepared runtime is read-only; slot removal uses one helper (U3b-0e 2b, F2/N10)."""

import errno
import os
from pathlib import Path
import stat

import pytest

from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    output = tmp_path_factory.mktemp("archive") / "runtime.tar.gz"
    build_runtime(Path(__file__).resolve().parents[3], output)
    return output


@pytest.fixture
def slot(archive, tmp_path):
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text("schema: t\n")
    (source / ".env").write_text("A=1\n")
    root = tmp_path / "root"
    stage.prepare_slot(archive, root, "s", source)
    return root.resolve() / "slots" / "s"


def _modes(runtime: Path) -> dict:
    result = {}
    for directory, names, files in os.walk(runtime):
        for name in names + files:
            path = Path(directory) / name
            result[path.relative_to(runtime).as_posix()] = path.lstat().st_mode
    return result


def test_runtime_is_read_only_after_prepare(slot):
    runtime = slot / "runtime"
    assert stat.S_IMODE(runtime.lstat().st_mode) == 0o555
    modes = _modes(runtime)
    assert {"config", "data", ".env", "release.json", "src/main.pyc"} <= modes.keys()
    for relative, mode in modes.items():
        if relative in {"config", "data", ".env"}:
            assert stat.S_ISLNK(mode), relative
        elif stat.S_ISDIR(mode):
            assert stat.S_IMODE(mode) == 0o555, relative
        else:
            assert stat.S_ISREG(mode) and stat.S_IMODE(mode) == 0o444, relative
    # The state stays private and writable through the runtime links.
    assert stat.S_IMODE((slot / "state").lstat().st_mode) == 0o700
    (runtime / "data" / "written.db").write_bytes(b"x")
    assert (slot / "state" / "data" / "written.db").read_bytes() == b"x"


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses mode bits")
@pytest.mark.parametrize("write", [
    lambda runtime: (runtime / "models").mkdir(),                   # a cwd-relative ./models default
    lambda runtime: (runtime / "pretrained_models").mkdir(),        # speechbrain's own default
    lambda runtime: (runtime / "src" / "core" / "new.pyc").write_bytes(b""),
    lambda runtime: (runtime / "src" / "main.pyc").write_bytes(b"x"),
    lambda runtime: (runtime / "release.json").unlink(),
])
def test_a_write_into_the_runtime_fails_with_eacces(slot, write):
    runtime = slot / "runtime"
    before = _modes(runtime)
    with pytest.raises(PermissionError) as error:
        write(runtime)
    assert error.value.errno == errno.EACCES
    assert _modes(runtime) == before


def test_activation_observes_the_read_only_tree(slot):
    root = slot.parents[1]
    assert stage.activate_slot(root, "s") is None
    observation = stage._artifact_tree(slot / "runtime")
    assert observation["entries"]["."]["stat"][2] & 0o777 == 0o555
    assert stage._artifact_tree(slot / "runtime", observation) == observation


def test_make_writable_then_remove_removes_a_prepared_slot(slot):
    stage.make_writable_then_remove(slot)
    assert not slot.exists() and slot.parent.is_dir()


def test_make_writable_then_remove_never_follows_links(tmp_path):
    outside = tmp_path / "outside"
    (outside / "sub").mkdir(parents=True)
    (outside / "sub" / "keep.txt").write_text("keep")
    os.chmod(outside / "sub", 0o555)
    os.chmod(outside, 0o555)
    tree = tmp_path / "tree"
    (tree / "a" / "b").mkdir(parents=True)
    (tree / "a" / "b" / "f").write_text("x")
    (tree / "a" / "link").symlink_to(outside)
    (tree / "a" / "file-link").symlink_to(outside / "sub" / "keep.txt")
    stage.make_runtime_read_only(tree)
    try:
        stage.make_writable_then_remove(tree)
        assert not tree.exists()
        assert (outside / "sub" / "keep.txt").read_text() == "keep"
        assert stat.S_IMODE(outside.stat().st_mode) == 0o555
        assert stat.S_IMODE((outside / "sub").stat().st_mode) == 0o555
    finally:
        os.chmod(outside, 0o755)
        os.chmod(outside / "sub", 0o755)


def test_make_writable_then_remove_a_link_or_file(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    stage.make_writable_then_remove(link)
    assert not link.is_symlink() and target.is_dir()
    with pytest.raises(FileNotFoundError):
        stage.make_writable_then_remove(tmp_path / "missing")


def test_make_writable_then_remove_refuses_a_mount_crossing(tmp_path, monkeypatch):
    tree = tmp_path / "tree"
    (tree / "mounted" / "x").mkdir(parents=True)
    stage.make_runtime_read_only(tree)
    mounted = (tree / "mounted").lstat().st_ino
    original = os.fstat

    def fstat(descriptor):
        info = original(descriptor)
        if info.st_ino != mounted:
            return info
        values = list(info)
        values[stat.ST_DEV] = info.st_dev + 1
        return os.stat_result(values)
    monkeypatch.setattr(stage.os, "fstat", fstat)
    with pytest.raises(ValueError, match="make_writable_mount_crossing"):
        stage.make_writable_then_remove(tree)
    monkeypatch.undo()
    assert stat.S_IMODE(tree.lstat().st_mode) == 0o555  # nothing changed before the refusal
    stage.make_writable_then_remove(tree)


@pytest.mark.parametrize("case", ["mounted_root", "nested_ismount", "table_bind", "table_bind_root",
                                  "parent_device"])
def test_make_writable_then_remove_refuses_bind_mounts_before_any_change(tmp_path, monkeypatch, case):
    """Review F7: a same-device bind mount is seen through ismount or the mount table."""
    tree = tmp_path / "tree"
    (tree / "bound" / "x").mkdir(parents=True)
    stage.make_runtime_read_only(tree)
    real = os.path.realpath(tree)
    if case == "mounted_root":
        monkeypatch.setattr(stage.os.path, "ismount", lambda path: os.fspath(path) == os.fspath(tree))
    elif case == "nested_ismount":
        monkeypatch.setattr(stage.os.path, "ismount", lambda path: os.fspath(path).endswith("/bound"))
    elif case == "table_bind":
        monkeypatch.setattr(stage, "_mount_points", lambda: ["/", real + "/bound"])
    elif case == "table_bind_root":
        monkeypatch.setattr(stage, "_mount_points", lambda: [real])
    else:
        original = Path.lstat

        def lstat(self):
            info = original(self)
            if self == tmp_path:
                values = list(info)
                values[stat.ST_DEV] = info.st_dev + 1
                return os.stat_result(values)
            return info
        monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="make_writable_mount_crossing"):
        stage.make_writable_then_remove(tree)
    monkeypatch.undo()
    assert stat.S_IMODE(tree.lstat().st_mode) == 0o555
    assert stat.S_IMODE((tree / "bound").lstat().st_mode) == 0o555


def test_a_sibling_mount_with_a_shared_prefix_is_not_a_crossing(tmp_path, monkeypatch):
    tree = tmp_path / "tree"
    (tree / "x").mkdir(parents=True)
    monkeypatch.setattr(stage, "_mount_points", lambda: [os.path.realpath(tree) + "-other", "/"])
    stage.make_writable_then_remove(tree)
    assert not tree.exists()


def test_mount_points_unescape_the_table(monkeypatch, tmp_path):
    table = tmp_path / "mountinfo"
    table.write_text("36 35 98:0 /mnt1 /mnt\\040point rw,noatime master:1 - ext3 /dev/root rw\n"
                     "37 35 98:0 / /a\\011b\\134c rw - ext4 /dev/x rw\nshort line\n")
    real_open = open
    monkeypatch.setattr("builtins.open", lambda path, *a, **k: real_open(
        table if path == "/proc/self/mountinfo" else path, *a, **k))
    assert stage._mount_points() == ["/mnt point", "/a\tb\\c"]


def test_make_runtime_read_only_refuses_special_files(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    os.mkfifo(tree / "fifo")
    with pytest.raises(ValueError, match="runtime_read_only_special_file"):
        stage.make_runtime_read_only(tree)
