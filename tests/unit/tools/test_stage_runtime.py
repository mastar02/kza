"""Offline promotion must select compatible code and data as one slot."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import os
import subprocess
import sys
import tarfile

import pytest

from scripts.build_runtime import build_runtime
from scripts.stage_runtime import activate_slot, prepare_slot


@pytest.fixture
def archive(tmp_path):
    """Use the real builder and a committed application, without starting it."""
    output = tmp_path / "runtime.tar.gz"
    build_runtime(Path(__file__).resolve().parents[3], output)
    return output


@pytest.fixture
def state(tmp_path):
    source = tmp_path / "offline-backup"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text("schema: old\n")
    (source / ".env").write_text("TEST_ONLY=private\n")
    with sqlite3.connect(source / "data" / "state.db") as db:
        db.execute("CREATE TABLE memories (value TEXT)")
        db.execute("INSERT INTO memories VALUES ('before migration')")
    return source


def test_promote_and_rollback_restore_runtime_and_sqlite(archive, state, tmp_path):
    root = tmp_path / "staging"
    old_archive = tmp_path / "old.tar.gz"
    build_runtime(Path(__file__).resolve().parents[3], old_archive,
                  "0fd3d166f1b17726f8ef998b320f37a42ff11d81")
    original = prepare_slot(old_archive, root, "old", state)
    assert activate_slot(root, "old") is None
    old_runtime = (root / "current").resolve()

    # A real copy of stopped state becomes a candidate's independent state.
    candidate = prepare_slot(archive, root, "candidate", old_runtime.parent / "state")
    assert original["git_commit"] != candidate["git_commit"]
    assert original["files"]["src/main.pyc"] != candidate["files"]["src/main.pyc"]
    new_runtime = root / "slots" / "candidate" / "runtime"
    with sqlite3.connect(new_runtime / "data" / "state.db") as db:
        db.execute("ALTER TABLE memories ADD COLUMN provenance TEXT")
        db.execute("UPDATE memories SET value = 'after migration'")
    (new_runtime / "config" / "settings.yaml").write_text("schema: new\n")
    assert activate_slot(root, "candidate") == "slots/old/runtime"
    assert (root / "current" / "config" / "settings.yaml").read_text() == "schema: new\n"

    assert activate_slot(root, "old") == "slots/candidate/runtime"
    assert (root / "current").resolve() == old_runtime
    with sqlite3.connect(root / "current" / "data" / "state.db") as db:
        assert db.execute("SELECT * FROM memories").fetchall() == [("before migration",)]
        assert len(db.execute("PRAGMA table_info(memories)").fetchall()) == 1
    assert (root / "current" / "config" / "settings.yaml").read_text() == "schema: old\n"
    assert json.loads((old_runtime / "release.json").read_text()) == original
    assert not list(old_runtime.rglob("*.py"))
    assert (old_runtime.parent.stat().st_mode & 0o777) == 0o700
    assert (old_runtime.parent / "state" / ".env").stat().st_mode & 0o777 == 0o600


def test_compiled_main_and_frontend_outside_checkout(archive, state, tmp_path):
    root = tmp_path / "staging"
    prepare_slot(archive, root, "candidate", state)
    activate_slot(root, "candidate")
    environment = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-c", "\n".join([
            "from pathlib import Path",
            "import os",
            "import src.main",
            "from fastapi.testclient import TestClient",
            "from src.dashboard.api import DashboardAPI",
            "assert Path(src.main.__file__).suffix == '.pyc'",
            "assert Path(src.main.__file__).resolve().is_relative_to(Path.cwd().resolve())",
            "from src.core.egress import EgressDenied",
            "from src.llm.reasoner import HttpReasoner",
            "reasoner = HttpReasoner(base_url='http://127.0.0.1:8200/v1')",
            "try:",
            "    reasoner.generate('offline smoke')",
            "except EgressDenied:",
            "    pass",
            "else:",
            "    raise AssertionError('Cloud gateway must require consent')",
            "client = TestClient(DashboardAPI(home_enabled=True).app)",
            "for path in ('/', '/home/home.css', '/home/home.js', '/home/request-id.js', '/home/routines.js', '/obs/src/session.js'):",
            "    assert client.get(path).status_code == 200, path",
            "os.environ.pop('KZA_DASHBOARD_TOKEN', None)",
            "assert client.get('/api/health').status_code == 503",
            "os.environ['KZA_DASHBOARD_TOKEN'] = 'compiled-smoke-session'",
            "assert client.get('/api/health').status_code == 401",
            "assert client.get('/api/health', headers={'Authorization': 'Bearer compiled-smoke-session'}).status_code == 200",
        ])],
        cwd=root / "current", env=environment, check=True,
        capture_output=True, text=True, timeout=60,
    )


def test_bad_candidate_does_not_replace_active_slot(archive, state, tmp_path):
    root = tmp_path / "staging"
    prepare_slot(archive, root, "old", state)
    activate_slot(root, "old")
    prepare_slot(archive, root, "broken", state)
    (root / "slots" / "broken" / "runtime" / "src" / "main.pyc").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        activate_slot(root, "broken")
    assert (root / "current").readlink() == Path("slots/old/runtime")


def test_extra_sources_or_redirected_state_cannot_be_activated(archive, state, tmp_path):
    root = tmp_path / "staging"
    prepare_slot(archive, root, "candidate", state)
    runtime = root / "slots" / "candidate" / "runtime"
    (runtime / "src" / "main.py").write_text("raise RuntimeError('unexpected source')")
    with pytest.raises(ValueError, match="Unexpected"):
        activate_slot(root, "candidate")
    (runtime / "src" / "main.py").unlink()
    (runtime / "data").unlink()
    (runtime / "data").symlink_to(state / "data")
    with pytest.raises(ValueError, match="same slot"):
        activate_slot(root, "candidate")
    assert not (root / "current").exists()


def test_refuses_overwrite_and_linked_state(archive, state, tmp_path):
    root = tmp_path / "staging"
    prepare_slot(archive, root, "old", state)
    with pytest.raises(FileExistsError):
        prepare_slot(archive, root, "old", state)
    (state / "data" / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="links"):
        prepare_slot(archive, root, "bad", state)
    assert not (root / "slots" / "bad").exists()


@pytest.mark.parametrize("name,kind", [
    ("../escape", "file"), ("/escape", "file"), ("src/evil.py", "file"),
    ("src/link", "link"), ("src/main.pyc", "duplicate"), ("data/private", "file"),
    ("src//alias", "file"),
])
def test_rejects_unsafe_archives(name, kind, state, tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(name)
        member.size = 1
        if kind == "link":
            member.type = tarfile.SYMTYPE
            member.linkname = "../../escape"
        bundle.addfile(member, io.BytesIO(b"x"))
        if kind == "duplicate":
            bundle.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        prepare_slot(archive, tmp_path / "root", "bad", state)
    assert not (tmp_path / "escape").exists()


def test_refuses_manifest_checksum_or_python_mismatch(state, tmp_path):
    archive = tmp_path / "bad-manifest.tar.gz"
    files = {"src/main.pyc": b"main", "src/__init__.pyc": b"init"}
    manifest = {
        "format": "kza-cpython-bytecode-v1", "git_commit": "a" * 40,
        "python": "3.13", "python_magic": importlib.util.MAGIC_NUMBER.hex(),
        "external_paths": ["config", "data", "models", "logs"],
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    files["src/main.pyc"] = b"altered"
    files["release.json"] = json.dumps(manifest).encode()
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="manifest"):
        prepare_slot(archive, tmp_path / "root", "bad", state)
