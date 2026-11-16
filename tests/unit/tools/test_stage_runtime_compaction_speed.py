"""SC-0b copy speed: the representation changes keep every compaction check exact.

O1: the content-addressed authority memo (scoped, every byte still hashed).
O9: one parse per distinct sources text in the deadline projection.
O4: D5 compares sorted per-row SHA-256 digests of the canonical row JSON.
P9: the free-space rule covers the measured peak of the current code.
"""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from scripts import compaction
from scripts import stage_runtime as stage
from src.memory import owner_compaction as oc
from src.memory.fact_ledger import FactLedger, MemoryReviewer
from tests.mocks.memory_sources import memory_source

PRIVACY = "data/privacy.db"
NOW = 2_000_000_000.0


def _state(tmp_path, deletions=2):
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    (state / "config").mkdir()
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    try:
        for index in range(deletions):
            fact = ledger.store("Synthetic fact %d" % index, "fact", .9, (memory_source(now=NOW - 1),))
            ledger.delete_fact(fact.id, MemoryReviewer("synthetic-reviewer"))
        ledger.store("Kept synthetic fact", "fact", .9, (memory_source(now=NOW - 1),))
    finally:
        ledger.close()
    with closing(sqlite3.connect(state / PRIVACY)) as db, db:
        db.execute("INSERT INTO voice_group_schema VALUES (1)")
    (state / "config" / "settings.yaml").write_text("synthetic: 1\n")
    return state.resolve()


def _counting(monkeypatch):
    calls = []
    original = stage._capture_state_authority

    def capture(state):
        calls.append(state)
        return original(state)
    monkeypatch.setattr(stage, "_capture_state_authority", capture)
    return calls


# --- O1 -----------------------------------------------------------------------------------------------

def test_memo_hit_equals_a_fresh_capture_and_is_a_private_copy(tmp_path, monkeypatch):
    state = _state(tmp_path)
    fresh = stage.capture_state_authority(state)
    calls = _counting(monkeypatch)
    with stage.authority_memo():
        first = stage.capture_state_authority(state)
        first["projection"]["tables"].clear()  # a caller mutating its result never reaches the memo
        second = stage.capture_state_authority(state)
        third = stage.capture_state_authority(state)
    assert len(calls) == 1
    assert second == third == fresh
    assert second is not third


def test_memo_is_off_outside_its_scope_and_cleared_on_exit(tmp_path, monkeypatch):
    state = _state(tmp_path)
    calls = _counting(monkeypatch)
    stage.capture_state_authority(state)
    stage.capture_state_authority(state)
    assert len(calls) == 2
    with stage.authority_memo():
        with stage.authority_memo():
            stage.capture_state_authority(state)
        stage.capture_state_authority(state)  # the inner exit kept the outer scope's memo
    assert len(calls) == 3 and stage._AUTHORITY_MEMO == {}
    with stage.authority_memo():
        stage.capture_state_authority(state)
    assert len(calls) == 4


def _flip_one_byte(path: Path) -> None:
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0x01
    path.write_bytes(bytes(raw))


def _same_size_same_mtime(path: Path, payload: bytes) -> None:
    info = path.stat()
    assert len(payload) == info.st_size
    path.write_bytes(payload)
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))


@pytest.mark.parametrize("change", ["byte", "same_size_mtime", "added_file", "removed_file", "protected_row"])
def test_any_change_misses_and_recaptures_exactly(tmp_path, monkeypatch, change):
    state = _state(tmp_path)
    settings = state / "config" / "settings.yaml"
    calls = _counting(monkeypatch)
    with stage.authority_memo():
        before = stage.capture_state_authority(state)
        if change == "byte":
            _flip_one_byte(settings)
        elif change == "same_size_mtime":
            _same_size_same_mtime(settings, b"synthetic: 2\n")
        elif change == "added_file":
            (state / "config" / "extra.json").write_text("{}")
        elif change == "removed_file":
            settings.unlink()
        else:
            with closing(sqlite3.connect(state / PRIVACY)) as db, db:
                db.execute("UPDATE facts SET content='Changed synthetic fact' WHERE state='deleted'")
        after = stage.capture_state_authority(state)
    assert len(calls) == 2
    monkeypatch.undo()
    assert after == stage.capture_state_authority(state)
    assert after != before


def test_a_symlink_or_special_file_gives_no_key_and_the_full_capture_refuses(tmp_path):
    state = _state(tmp_path)
    with stage.authority_memo():
        stage.capture_state_authority(state)
        (state / "config" / "link.yaml").symlink_to(state / "config" / "settings.yaml")
        assert stage._authority_tree_key(state) is None
        with pytest.raises(ValueError, match="runtime_admission_state_symlink"):
            stage.capture_state_authority(state)
        (state / "config" / "link.yaml").unlink()
        os.mkfifo(state / "config" / "pipe")
        assert stage._authority_tree_key(state) is None
        assert stage.capture_state_authority(state) == stage._capture_state_authority(state)


def test_a_capture_racing_a_change_is_not_stored(tmp_path, monkeypatch):
    state = _state(tmp_path)
    original = stage._capture_state_authority
    settings = state / "config" / "settings.yaml"

    def racing(path):
        result = original(path)
        settings.write_text("synthetic: 3\n")  # the tree moves while the capture runs
        return result
    monkeypatch.setattr(stage, "_capture_state_authority", racing)
    with stage.authority_memo():
        stage.capture_state_authority(state)
        assert stage._AUTHORITY_MEMO == {}


def test_a_different_contract_never_reuses_a_result(tmp_path, monkeypatch):
    state = _state(tmp_path)
    calls = _counting(monkeypatch)
    with stage.authority_memo():
        stage.capture_state_authority(state)
        monkeypatch.setattr(stage, "_PROTECTED_TABLES", stage._PROTECTED_TABLES | {"voice_group_schema"})
        widened = stage.capture_state_authority(state)
    assert len(calls) == 2
    assert "voice_group_schema" in widened["projection"]["tables"][PRIVACY]


def test_compaction_commands_and_worker_run_inside_the_memo_scope():
    import inspect
    for function in (compaction.compact_plan, compaction.compact_start, compaction._continue,
                     oc.run_materialization):
        assert "authority_memo()" in inspect.getsource(function)


# --- O9 -----------------------------------------------------------------------------------------------

def _reference_projection(db, table, identity, sources, deadline):
    hashes = []
    for row in db.execute("SELECT " + ",".join((identity, sources, deadline)) + " FROM " + table):
        ceiling = stage._source_deadline(row[sources])
        if row[deadline] is not None and row[deadline] < ceiling:
            hashes.append(stage._digest([row[identity], row[sources], row[deadline]]))
    return {"columns": [identity, sources, deadline], "row_hashes": sorted(hashes)}


def _deadline_db(rows):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE t (id INTEGER, sources_json TEXT, expires_at REAL)")
    db.executemany("INSERT INTO t VALUES (?,?,?)", rows)
    return db


def test_deadline_projection_parses_each_distinct_text_once_with_the_same_result(monkeypatch):
    shared = json.dumps([{"expires_at": 50.0}, {"expires_at": 40}])
    rows = [(index, shared if index % 3 else json.dumps([{"expires_at": float(index)}]),
             None if index % 7 == 0 else float(index % 60)) for index in range(300)]
    with closing(_deadline_db(rows)) as db:
        expected = _reference_projection(db, "t", "id", "sources_json", "expires_at")
        parsed = []
        original = stage._source_deadline
        monkeypatch.setattr(stage, "_source_deadline", lambda text: parsed.append(text) or original(text))
        assert stage._deadline_projection(db, "t", "id", "sources_json", "expires_at") == expected
    assert len(parsed) == len(set(parsed)) == len({row[1] for row in rows})
    assert expected["row_hashes"]


@pytest.mark.parametrize("bad", ["[]", "{}", "not json", '[{"expires_at": "x"}]', None])
def test_a_malformed_sources_text_still_refuses_after_cached_duplicates(bad):
    good = json.dumps([{"expires_at": 10.0}])
    with closing(_deadline_db([(1, good, 1.0), (2, good, 2.0), (3, bad, 3.0), (4, good, 4.0)])) as db:
        with pytest.raises((ValueError, TypeError)):
            stage._deadline_projection(db, "t", "id", "sources_json", "expires_at")


# --- O4 -----------------------------------------------------------------------------------------------

def _k(tmp_path, rows, columns="id INTEGER, label TEXT, score REAL, payload BLOB"):
    path = tmp_path / ("k-%d.db" % len(list(tmp_path.glob("k-*.db"))))
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE t (" + columns + ")")
        db.executemany("INSERT INTO t VALUES (" + ",".join("?" * len(rows[0])) + ")", rows) if rows else None
    return path


def _blob_plan(tmp_path, payloads):
    root = tmp_path / "snapshot"
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    refs = {}
    for payload in payloads:
        digest = hashlib.sha256(payload).hexdigest()
        (root / "blobs" / digest).write_bytes(payload)
        refs[payload] = {"$blob": {"path": "blobs/" + digest, "sha256": digest, "size": len(payload)}}
    return SimpleNamespace(latest=SimpleNamespace(snapshot_root=root)), refs


def _compare(tmp_path, k_rows, snapshot_rows, plan=None, columns="id INTEGER, label TEXT, score REAL, payload BLOB"):
    with closing(sqlite3.connect(_k(tmp_path, k_rows, columns))) as db:
        names, actual = oc._destination_digests(db, "t")
    wanted = oc._expected_digests(plan or SimpleNamespace(latest=None), snapshot_rows, names)
    return wanted is not None and actual == wanted


def test_d5_digests_accept_the_same_multiset_in_any_order_with_blobs(tmp_path):
    plan, refs = _blob_plan(tmp_path, [b"\x00\x01", b"voice"])
    k_rows = [(1, "a", 0.5, b"\x00\x01"), (2, "b", None, b"voice"), (2, "b", None, b"voice"), (3, None, 1.0, None)]
    snapshot = [{"payload": None, "score": 1.0, "label": None, "id": 3},
                {"id": 2, "label": "b", "score": None, "payload": refs[b"voice"]},
                {"id": 1, "label": "a", "score": 0.5, "payload": refs[b"\x00\x01"]},
                {"id": 2, "label": "b", "score": None, "payload": refs[b"voice"]}]
    assert _compare(tmp_path, k_rows, snapshot, plan)


@pytest.mark.parametrize("mutation", ["value", "int_for_real", "multiplicity", "missing_row", "extra_column",
                                      "missing_column", "swapped_columns", "blob_bytes"])
def test_d5_digests_refuse_any_difference(tmp_path, mutation):
    plan, refs = _blob_plan(tmp_path, [b"\x00\x01", b"\x00\x02"])
    k_rows = [(1, "a", 2.0, b"\x00\x01"), (2, "b", 3.0, None)]
    snapshot = [{"id": 1, "label": "a", "score": 2.0, "payload": refs[b"\x00\x01"]},
                {"id": 2, "label": "b", "score": 3.0, "payload": None}]
    columns = "id INTEGER, label TEXT, score REAL, payload BLOB"
    if mutation == "value":
        snapshot[1]["label"] = "c"
    elif mutation == "int_for_real":
        snapshot[0]["score"] = 2
    elif mutation == "multiplicity":
        snapshot.append(dict(snapshot[1]))
        k_rows.append(k_rows[0])
    elif mutation == "missing_row":
        snapshot.pop()
    elif mutation == "extra_column":
        snapshot[0]["extra"] = None
        snapshot[1]["extra"] = None
    elif mutation == "missing_column":
        k_rows = [row + (None,) for row in k_rows]
        columns += ", extra TEXT"
    elif mutation == "swapped_columns":
        k_rows = [(1, "2.0", "a", b"\x00\x01"), (2, "3.0", "b", None)]
        columns = "id INTEGER, score TEXT, label TEXT, payload BLOB"
        snapshot = [{"id": 1, "label": "2.0", "score": "a", "payload": refs[b"\x00\x01"]},
                    {"id": 2, "label": "3.0", "score": "b", "payload": None}]
    else:
        snapshot[0]["payload"] = refs[b"\x00\x02"]
    assert not _compare(tmp_path, k_rows, snapshot, plan, columns)


def test_d5_digest_of_an_empty_table_matches_only_an_empty_snapshot(tmp_path):
    assert _compare(tmp_path, [], [])
    assert not _compare(tmp_path, [(1, "a", 1.0, None)], [])
