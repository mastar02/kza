"""Tests del manifest incremental del code-index."""

import json

from src.code_index.manifest import IndexManifest, git_blob_hash


def test_git_blob_hash_matches_git():
    # `echo "hello" | git hash-object --stdin` → ce01362...
    assert git_blob_hash(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_diff_detects_added_changed_deleted(tmp_path):
    m = IndexManifest(tmp_path / "manifest.json")
    m.files = {
        "src/a.py": {"hash": "h1", "card_done": True},
        "src/b.py": {"hash": "h2", "card_done": True},
    }
    diff = m.diff({"src/a.py": "h1-nuevo", "src/c.py": "h3"})
    assert diff.changed == ["src/a.py"]
    assert diff.added == ["src/c.py"]
    assert diff.deleted == ["src/b.py"]
    assert not diff.empty


def test_diff_retries_pending_cards():
    m = IndexManifest.__new__(IndexManifest)
    m.files = {"src/a.py": {"hash": "h1", "card_done": False}}
    m.head_sha = None
    diff = m.diff({"src/a.py": "h1"})  # hash igual pero card pendiente
    assert diff.changed == ["src/a.py"]


def test_diff_empty_when_no_changes(tmp_path):
    m = IndexManifest(tmp_path / "manifest.json")
    m.files = {"src/a.py": {"hash": "h1", "card_done": True}}
    assert m.diff({"src/a.py": "h1"}).empty


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "sub" / "manifest.json"
    m = IndexManifest(path)
    m.files = {"src/a.py": {"hash": "h1", "card_done": True}}
    m.head_sha = "abc123"
    m.save()

    m2 = IndexManifest(path)
    m2.load()
    assert m2.files == m.files
    assert m2.head_sha == "abc123"


def test_load_missing_file_is_noop(tmp_path):
    m = IndexManifest(tmp_path / "nope.json")
    m.load()
    assert m.files == {}
    assert m.head_sha is None


def test_update_and_remove_persist_immediately(tmp_path):
    path = tmp_path / "manifest.json"
    m = IndexManifest(path)
    m.update_file("src/a.py", "h1", card_done=False)
    on_disk = json.loads(path.read_text())
    assert on_disk["files"]["src/a.py"] == {"hash": "h1", "card_done": False}

    m.remove_file("src/a.py")
    on_disk = json.loads(path.read_text())
    assert on_disk["files"] == {}
