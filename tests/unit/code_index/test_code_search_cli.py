"""Tests de la lógica de formato/drift del CLI code_search."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "code_search",
    Path(__file__).resolve().parents[3] / "tools" / "code_search.py",
)
code_search = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(code_search)


def _result(path="src/a.py", blob_hash="X"):
    return {
        "path": path,
        "name": "fa",
        "kind": "function",
        "start_line": 1,
        "end_line": 2,
        "blob_hash": blob_hash,
        "score": 0.9,
        "snippet": "def fa():\n    return 1",
        "card": "## Propósito\nCard de a.",
    }


def test_blob_hash_matches_git():
    assert (
        code_search.git_blob_hash(b"hello\n")
        == "ce013625030ba8dba906f756967f9e9ca394464a"
    )


def test_format_fresh_result_no_stale(tmp_path):
    f = tmp_path / "src" / "a.py"
    f.parent.mkdir()
    f.write_text("def fa():\n    return 1\n")
    res = _result(blob_hash=code_search.git_blob_hash(f.read_bytes()))

    out = code_search.format_result(res, tmp_path, seen_cards=set())

    assert "STALE" not in out
    assert "src/a.py:1-2" in out
    assert "## Propósito" in out


def test_format_stale_when_local_differs(tmp_path):
    f = tmp_path / "src" / "a.py"
    f.parent.mkdir()
    f.write_text("def fa():\n    return 999\n")
    res = _result(blob_hash="hash-viejo-del-indice")

    out = code_search.format_result(res, tmp_path, seen_cards=set())

    assert "⚠ STALE" in out


def test_format_missing_local_file(tmp_path):
    out = code_search.format_result(_result(), tmp_path, seen_cards=set())
    assert "⚠ LOCAL MISSING" in out


def test_card_printed_once_per_path(tmp_path):
    seen: set = set()
    out1 = code_search.format_result(_result(), tmp_path, seen_cards=seen)
    out2 = code_search.format_result(_result(), tmp_path, seen_cards=seen)
    assert "## Propósito" in out1
    assert "## Propósito" not in out2
