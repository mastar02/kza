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


@pytest.mark.parametrize("mutation", ["value", "int_for_real", "multiplicity", "duplicate_in_k_only",
                                      "missing_row", "extra_column",
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
    elif mutation == "duplicate_in_k_only":
        k_rows.append(k_rows[1])
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


# --- verify_snapshot: one canonical serialization, the same three facts ---------------------------------

def _image(tmp_path):
    from src.memory import state_snapshot as snapshots
    root = tmp_path / "image"
    (root / "state").mkdir(parents=True)
    data = root / "state" / "ordinary.json"
    data.write_text('{"ordinary":"original"}')
    binding = {"state_root": str(tmp_path / "original")}
    identity = hashlib.sha256(snapshots._json(binding)).hexdigest()
    manifest = {"schema_version": 1, "source_binding_id": identity, "source_binding": binding, "score": 1.0,
                "files": {"state/ordinary.json": {"sha256": snapshots._sha(data), "size": data.stat().st_size}}}
    (root / "manifest.json").write_bytes(snapshots._json(manifest) + b"\n")
    return snapshots, snapshots.ClosedStateSnapshot(root, identity, binding, manifest)


@pytest.mark.parametrize("mutation", ["non_canonical_file", "no_newline", "equal_but_retyped_dto", "dto_value",
                                      "file_replaced_with_other_canonical", "file_and_dto_changed_together",
                                      "not_a_dict"])
def test_verify_snapshot_refuses_every_departure_from_the_pinned_canonical_text(tmp_path, mutation):
    snapshots, image = _image(tmp_path)
    path = image.snapshot_root / "manifest.json"
    returned = snapshots.verify_snapshot(image)
    assert returned == image.manifest and returned is not image.manifest
    canonical = snapshots._json(image.manifest)
    if mutation == "non_canonical_file":
        path.write_bytes(json.dumps(image.manifest, sort_keys=True).encode() + b"\n")  # same value, other text
    elif mutation == "no_newline":
        path.write_bytes(canonical)
    elif mutation == "equal_but_retyped_dto":
        image.manifest["score"] = 1  # == 1.0 for a deep compare, not for the pinned text
    elif mutation == "dto_value":
        image.manifest["files"]["state/ordinary.json"]["size"] += 1
    elif mutation == "file_replaced_with_other_canonical":
        path.write_bytes(snapshots._json(dict(image.manifest, score=2.0)) + b"\n")
    elif mutation == "file_and_dto_changed_together":
        image.manifest["score"] = 2.0  # only the hash captured with the DTO tells
        path.write_bytes(snapshots._json(image.manifest) + b"\n")
    else:
        path.write_bytes(b"[]\n")
    with pytest.raises(ValueError, match="state_snapshot_manifest_changed"):
        snapshots.verify_snapshot(image)


# --- P9 -----------------------------------------------------------------------------------------------

S, M, L, R = 154_116_457, 377_000_000, 93_335_552, 17_000_000  # SC-0b: the measured synthetic state
PEAK_GROWTH = 861_000_000  # SC-0b: filesystem growth at the peak (VACUUM of K's events.db), temp on the root
SERVER_PEAKS = (942_010_368, 823_345_152)  # SC-0b after the change: free-space drop every 10 ms (shared disk)


def test_free_space_need_is_exactly_the_terms_of_the_measured_peak():
    margin = compaction.FREE_MARGIN
    assert margin == 256 << 20 == oc.FREE_MARGIN
    base = compaction.free_space_need(0, 0, 0, 0, temp_on_root=True)
    assert base == {"root": margin, "temp": 0}
    for args, root in (((1, 0, 0, 0), 2), ((0, 1, 0, 0), 1), ((0, 0, 1, 0), 2), ((0, 0, 0, 1), 1)):
        assert compaction.free_space_need(*args, temp_on_root=True)["root"] - margin == root
    assert compaction.free_space_need(0, 0, 1, 0, temp_on_root=False) == {"root": margin + 1, "temp": margin + 1}


def test_free_space_need_covers_the_measured_peak_with_the_real_manifest_and_with_the_estimate():
    margin = compaction.FREE_MARGIN
    real = compaction.free_space_need(S, M, L, R, temp_on_root=True)["root"]
    assert real - margin >= PEAK_GROWTH
    estimated = compaction.free_space_need(S, compaction.MANIFEST_ESTIMATE_FACTOR * S, L, R, temp_on_root=True)
    assert estimated["root"] >= real
    assert 2 * S + (256 << 20) < PEAK_GROWTH  # the old rule fell short of the measured peak
    assert all(real >= peak for peak in SERVER_PEAKS)


def test_worker_need_is_k_plus_the_vacuum_of_its_largest_database():
    manifest = {"files": {"state/data/events.db": {"size": L, "kind": "sqlite_backup"},
                          "state/data/ambient.db": {"size": 60_375_040, "kind": "sqlite_backup"},
                          "state/config/settings.yaml": {"size": 405_865, "kind": "file"},
                          "history/x.json": {"size": 10**9, "kind": "history_archive"}}}
    snapshot = SimpleNamespace(manifest=manifest)
    state = L + 60_375_040 + 405_865
    assert oc.free_space_need(snapshot, temp_on_root=True) == {"root": state + 2 * L + oc.FREE_MARGIN, "temp": 0}
    assert oc.free_space_need(snapshot, temp_on_root=False) == {"root": state + L + oc.FREE_MARGIN,
                                                                "temp": L + oc.FREE_MARGIN}


def test_worker_refuses_before_k_when_the_real_need_does_not_fit(tmp_path, monkeypatch):
    destination = tmp_path / "slots" / "k" / "state"
    destination.mkdir(parents=True)
    snapshot = SimpleNamespace(manifest={"files": {"state/data/events.db": {"size": 1000, "kind": "sqlite_backup"}}})
    need = oc.free_space_need(snapshot, temp_on_root=True)["root"]
    monkeypatch.setenv("SQLITE_TMPDIR", str(tmp_path))
    monkeypatch.setattr(oc.shutil, "disk_usage", lambda path: SimpleNamespace(free=need))
    oc._check_free_space(snapshot, destination)
    monkeypatch.setattr(oc.shutil, "disk_usage", lambda path: SimpleNamespace(free=need - 1))
    with pytest.raises(oc.CompactionCloneError, match="runtime_admission_compaction_capacity"):
        oc._check_free_space(snapshot, destination)


def test_sqlite_temp_directory_follows_sqlites_own_search_order(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    assert compaction.sqlite_temp_directory({"SQLITE_TMPDIR": str(first), "TMPDIR": str(second)}) == first
    assert compaction.sqlite_temp_directory({"SQLITE_TMPDIR": str(tmp_path / "missing"),
                                             "TMPDIR": str(second)}) == second
    assert compaction.sqlite_temp_directory({}) in {Path("/var/tmp"), Path("/usr/tmp"), Path("/tmp"), None}


def _capacity_root(tmp_path):
    state = _state(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "code.pyc").write_bytes(b"\0" * 4096)
    (runtime / "link").symlink_to(state)  # never followed
    root = tmp_path / "root"
    root.mkdir()
    binding = {"state_root": str(state), "runtime_root": str(runtime), "authority": {}}
    return root, binding, state


def test_capacity_refuses_below_the_need_and_reports_it_content_free(tmp_path, monkeypatch):
    root, binding, state = _capacity_root(tmp_path)
    monkeypatch.setenv("SQLITE_TMPDIR", str(tmp_path))
    largest = max((path.stat().st_size for path in state.rglob("*.db")), default=0)
    need = compaction.free_space_need(compaction._state_bytes(state),
                                      compaction.MANIFEST_ESTIMATE_FACTOR * compaction._state_bytes(state),
                                      largest, 4096, temp_on_root=True)["root"]
    monkeypatch.setattr(compaction.shutil, "disk_usage", lambda path: SimpleNamespace(free=need))
    report = compaction.capacity(root, binding, PRIVACY)
    assert report["need_bytes"] == need and report["temp_need_bytes"] == 0 and report["temp_free_bytes"] is None
    monkeypatch.setattr(compaction.shutil, "disk_usage", lambda path: SimpleNamespace(free=need - 1))
    with pytest.raises(compaction.CompactionRefusal, match="runtime_admission_compaction_capacity") as error:
        compaction.capacity(root, binding, PRIVACY)
    assert set(error.value.report) >= {"need_bytes", "free_bytes", "state_bytes", "temp_need_bytes"}


def test_capacity_checks_sqlites_temp_filesystem_when_it_is_another_one(tmp_path, monkeypatch):
    root, binding, state = _capacity_root(tmp_path)
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setenv("SQLITE_TMPDIR", str(temp))
    real_device = compaction._device
    monkeypatch.setattr(compaction, "_device", lambda path: -1 if Path(path) == temp else real_device(path))
    frees = {}
    monkeypatch.setattr(compaction.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=frees.get(Path(path), 10**12)))
    report = compaction.capacity(root, binding, PRIVACY)
    assert report["temp_need_bytes"] == compaction._largest_sqlite(state) + compaction.FREE_MARGIN
    frees[temp] = report["temp_need_bytes"] - 1
    with pytest.raises(compaction.CompactionRefusal, match="runtime_admission_compaction_capacity"):
        compaction.capacity(root, binding, PRIVACY)


def test_the_worker_checks_p9_again_before_k_is_planned():
    import inspect
    source = inspect.getsource(oc.run_materialization)
    assert source.index("_check_free_space(snapshot, destination)") < source.index("\"materializing\", {")


def _identity_world(tmp_path, k_rows):
    destination = tmp_path / "k"
    (destination / "data").mkdir(parents=True)
    with closing(sqlite3.connect(destination / "data" / "x.db")) as db, db:
        db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, label TEXT, score REAL)")
        db.executemany("INSERT INTO t VALUES (?,?,?)", k_rows)
    with closing(sqlite3.connect(destination / "data" / "x.db")) as db:
        db.row_factory = sqlite3.Row
        schema = [dict(row) for row in db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
    rows = [{"id": 1, "label": "a", "score": 0.5}, {"id": 2, "label": "b", "score": None}]
    manifest = {"files": {"state/data/x.db": {"kind": "sqlite_backup", "sha256": "0" * 64, "size": 1}},
                "databases": {"data/x.db": {"schema": schema, "tables": {"t": {"rows": rows}}}}}
    plan = SimpleNamespace(latest=SimpleNamespace(manifest=manifest, snapshot_root=tmp_path / "snapshot"),
                           privacy_relative="data/privacy.db", destination=destination, source_roots=(),
                           evidence_map={})
    return plan, destination, SimpleNamespace(registered_target_ids=(), target_id_map={})


@pytest.mark.parametrize("k_rows,code", [
    ([(1, "a", 0.5), (2, "b", None)], None),
    ([(2, "b", None), (1, "a", 0.5)], None),
    ([(1, "a", 0.5), (2, "c", None)], "payload_changed:t"),
    ([(1, "a", 0.5), (2, "b", 0.0)], "payload_changed:t"),
    ([(1, "a", 0.5)], "payload_changed:t"),
    ([(1, "a", 0.5), (2, "b", None), (3, "b", None)], "payload_changed:t"),
])
def test_verify_identity_on_a_real_database_compares_every_value(tmp_path, k_rows, code):
    plan, destination, journal = _identity_world(tmp_path, k_rows)
    if code is None:
        assert oc.verify_identity(plan, destination, journal) == {"databases": 1, "tables": 1, "rows": 2, "files": 0}
        (destination / "data" / "stray.txt").write_text("x")
        code = "payload_changed:extra_file"
    with pytest.raises(oc.CompactionCloneError, match="runtime_admission_compaction_" + code):
        oc.verify_identity(plan, destination, journal)


@pytest.mark.parametrize("salt_row", [False, True])
def test_verify_identity_accepts_an_added_ledger_only_while_it_is_empty(tmp_path, salt_row):
    from src.memory.deletion_ledger import deletion_ledger_schema_objects
    plan, destination, journal = _identity_world(tmp_path, [(1, "a", 0.5), (2, "b", None)])
    with closing(sqlite3.connect(destination / "data" / "x.db")) as db, db:
        for row in deletion_ledger_schema_objects():
            if row["type"] == "table":
                db.execute(row["sql"])
        if salt_row:
            db.execute("INSERT INTO deletion_ledger_salt_v1 VALUES (1, 'synthetic')")
    if salt_row:
        with pytest.raises(oc.CompactionCloneError, match="runtime_admission_compaction_payload_changed:added_rows"):
            oc.verify_identity(plan, destination, journal)
    else:
        assert oc.verify_identity(plan, destination, journal)["tables"] == 1
