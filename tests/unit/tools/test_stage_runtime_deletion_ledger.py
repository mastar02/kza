"""U3b-0d3 admission: deletion-ledger registration, constant-size binding, receipt schema 2.

The reference controller is this stage_runtime with exactly the ledger
registrations removed (the pre-0d3 contract), so these cases never depend on
what 'HEAD' is.
"""

from contextlib import closing, contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import pytest

from scripts import stage_runtime as stage
from src.memory.deletion_ledger import (
    DELETION_LEDGER_TABLES, DeletionKind, append_deletion, initialize_deletion_ledger, row_sha256,
)
from src.memory.fact_ledger import FactLedger, MemoryReviewer
from tests.mocks.memory_sources import memory_source

REPO = Path(__file__).resolve().parents[3]
PRIVACY = "data/privacy.db"
NOW = 2_000_000_000.0
LEDGER = set(DELETION_LEDGER_TABLES)


@contextmanager
def reference_controller(monkeypatch):
    """This controller without the U3b-0d3 registrations: the pre-0d3 admission contract."""
    with monkeypatch.context() as patch:
        patch.setattr(stage, "_TABLE_COLUMNS", {name: value for name, value in stage._TABLE_COLUMNS.items()
                                                if name not in LEDGER})
        patch.setattr(stage, "_PROTECTED_TABLES", stage._PROTECTED_TABLES - LEDGER)
        yield


def test_registration_strings_are_exact():
    assert {name: stage._TABLE_COLUMNS[name] for name in LEDGER} == {
        "deletion_ledger_v1": "seq:INTEGER kind:TEXT subject_sha256:TEXT at:REAL prev_sha256:TEXT row_sha256:TEXT",
        "deletion_ledger_salt_v1": "id:INTEGER salt:TEXT"}
    assert LEDGER <= stage._PROTECTED_TABLES and not LEDGER & stage._SCHEMA_MARKERS
    assert stage._DELETION_LEDGER_ADDITION == DELETION_LEDGER_TABLES
    assert stage._SCHEMA_ADDITIONS[-1] == stage._DELETION_LEDGER_ADDITION
    assert stage._LEDGER_GENESIS == "0" * 64


def _state(tmp_path, *, pre_0d3=False, deletions=0):
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    (state / "config").mkdir()
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    try:
        for index in range(deletions):
            fact = ledger.store("Synthetic fact %d" % index, "fact", .9, (memory_source(now=NOW - 1),))
            ledger.delete_fact(fact.id, MemoryReviewer("synthetic-reviewer"))
    finally:
        ledger.close()
    with closing(sqlite3.connect(state / PRIVACY)) as db, db:
        db.execute("INSERT INTO voice_group_schema VALUES (1)")
        if pre_0d3:
            for table in DELETION_LEDGER_TABLES:
                db.execute("DROP TABLE " + table)
    return state


def test_projection_is_a_constant_size_chain_not_rows(tmp_path):
    state = _state(tmp_path, deletions=3)
    tables = stage.capture_state_authority(state)["projection"]["tables"][PRIVACY]
    projection = tables["deletion_ledger_v1"]
    assert set(projection) == {"count", "max_seq", "chain_head", "chain_valid"}
    assert projection["count"] == projection["max_seq"] == 3 and projection["chain_valid"] is True
    with closing(sqlite3.connect(state / PRIVACY)) as db:
        assert projection["chain_head"] == db.execute(
            "SELECT row_sha256 FROM deletion_ledger_v1 ORDER BY seq DESC LIMIT 1").fetchone()[0]
    assert len(tables["deletion_ledger_salt_v1"]["row_hashes"]) == 1  # the salt row is pinned


def _bulk_ledger(path, rows):
    """A ledger of `rows` valid chained rows, written in bulk (synthetic subjects)."""
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        initialize_deletion_ledger(db)
        db.execute("BEGIN IMMEDIATE")
        append_deletion(db, DeletionKind.FACT_DELETED, "synthetic-0", NOW)  # creates the salt, seq 1
        head = db.execute("SELECT row_sha256 FROM deletion_ledger_v1").fetchone()[0]
        batch = []
        for seq in range(2, rows + 1):
            subject = "%064x" % seq
            at = NOW + seq
            digest = row_sha256(head, seq, "fact_deleted", subject, at)
            batch.append((seq, "fact_deleted", subject, at, head, digest))
            head = digest
            if len(batch) == 50_000:
                db.executemany("INSERT INTO deletion_ledger_v1 VALUES (?,?,?,?,?,?)", batch)
                batch.clear()
        db.executemany("INSERT INTO deletion_ledger_v1 VALUES (?,?,?,?,?,?)", batch)
        db.commit()


def test_binding_size_is_the_same_for_one_and_a_million_rows(tmp_path):
    """Design v3 N3 test: 1 row vs 1,000,000 rows give the same binding size (within a few bytes)."""
    sizes, seconds = {}, {}
    for rows in (1, 1_000_000):
        state = tmp_path / ("rows-%d" % rows)
        (state / "data").mkdir(parents=True)
        _bulk_ledger(state / "data" / "ledger.db", rows)
        started = time.monotonic()
        authority = stage.capture_state_authority(state)
        seconds[rows] = time.monotonic() - started
        assert authority["projection"]["tables"]["data/ledger.db"]["deletion_ledger_v1"]["count"] == rows
        sizes[rows] = len(stage._canonical_json(authority))
    assert abs(sizes[1_000_000] - sizes[1]) <= 12, sizes  # only the count/max_seq digits differ
    evidence = os.environ.get("KZA_0D3_EVIDENCE")
    if evidence:
        Path(evidence).write_text(json.dumps({"binding_bytes": sizes, "capture_seconds": seconds}, indent=2))


def test_a_per_row_projection_would_grow_with_rows(tmp_path, monkeypatch):
    """Mutant guard: projecting the ledger row by row breaks the constant size."""
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    _bulk_ledger(state / "data" / "ledger.db", 2000)
    constant = len(stage._canonical_json(stage.capture_state_authority(state)))
    monkeypatch.setattr(stage, "_LEDGER_TABLE", "not_the_ledger")
    assert len(stage._canonical_json(stage.capture_state_authority(state))) > constant + 2000 * 64


@pytest.mark.parametrize("statement", [
    "UPDATE deletion_ledger_v1 SET at=at+1 WHERE seq=2",
    "DELETE FROM deletion_ledger_v1 WHERE seq=2",
    "UPDATE deletion_ledger_v1 SET row_sha256=prev_sha256 WHERE seq=3",
])
def test_update_or_delete_refuses_capture_with_chain_broken(tmp_path, statement):
    state = _state(tmp_path, deletions=3)
    with closing(sqlite3.connect(state / PRIVACY)) as db, db:
        db.execute(statement)
    with pytest.raises(ValueError, match="runtime_admission_ledger_chain_broken"):
        stage.capture_state_authority(state)


def test_forward_evolution_accepts_appends_and_refuses_a_restored_older_ledger(tmp_path):
    state = _state(tmp_path, deletions=2)
    older = tmp_path / "older.db"
    shutil.copy2(state / PRIVACY, older)
    binding = {"authority": stage.capture_state_authority(state)}
    assert stage.deletion_ledger_regressions(binding, state) == []
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW + 1)
    ledger.initialize()
    try:
        fact = ledger.store("Synthetic later fact", "fact", .9, (memory_source(now=NOW),))
        ledger.delete_fact(fact.id, MemoryReviewer("synthetic-reviewer"))
    finally:
        ledger.close()
    later = {"authority": stage.capture_state_authority(state)}
    assert stage.deletion_ledger_regressions(binding, state) == []  # forward: count grew, prefix equal
    shutil.copy2(older, state / PRIVACY)  # an out-of-band restore truncates the ledger
    assert stage.deletion_ledger_regressions(later, state) == [PRIVACY]
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW + 2)
    ledger.initialize()
    try:  # the restored ledger grows again past the bound count, on a different history
        for index in range(2):
            fact = ledger.store("Synthetic regrown fact %d" % index, "fact", .9, (memory_source(now=NOW),))
            ledger.delete_fact(fact.id, MemoryReviewer("synthetic-reviewer"))
    finally:
        ledger.close()
    regrown = stage.capture_state_authority(state)["projection"]["tables"][PRIVACY]["deletion_ledger_v1"]
    assert regrown["count"] > later["authority"]["projection"]["tables"][PRIVACY]["deletion_ledger_v1"]["count"]
    assert stage.deletion_ledger_regressions(later, state) == [PRIVACY]  # the bound prefix head differs
    with closing(sqlite3.connect(state / PRIVACY)) as db, db:
        db.execute("DROP TABLE deletion_ledger_v1")
    assert stage.deletion_ledger_regressions(binding, state) == [PRIVACY]
    (state / PRIVACY).unlink()  # review K4/Z3: a missing database is a regression too
    assert stage.deletion_ledger_regressions(binding, state) == [PRIVACY]


def test_unmigrated_pre_0d3_state_has_the_same_digest_under_both_controllers(tmp_path, monkeypatch):
    state = _state(tmp_path, pre_0d3=True)
    new = stage.capture_state_authority(state)
    with reference_controller(monkeypatch):
        assert stage.capture_state_authority(state) == new


def test_initialisation_adds_the_empty_ledger_and_the_reference_controller_refuses_it(tmp_path, monkeypatch):
    """The rollback rejection point: a pre-0d3 controller refuses the migrated state."""
    state = _state(tmp_path, pre_0d3=True)
    source = stage.capture_state_authority(state)["projection"]
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    ledger.close()
    derived = stage.capture_state_authority(state)["projection"]
    assert derived["tables"][PRIVACY]["deletion_ledger_v1"] == {
        "count": 0, "max_seq": 0, "chain_head": "0" * 64, "chain_valid": True}
    assert derived["tables"][PRIVACY]["deletion_ledger_salt_v1"] == {"columns": ["id", "salt"], "row_hashes": []}
    expected = copy.deepcopy(source)
    stage._expect_empty_schema_additions(expected, derived)
    assert expected == derived
    with reference_controller(monkeypatch):
        with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
            stage.capture_state_authority(state)
    # Any ledger row in the migrated group still mismatches the empty expectation.
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    try:
        fact = ledger.store("Synthetic fact", "fact", .9, (memory_source(now=NOW - 1),))
        ledger.delete_fact(fact.id, MemoryReviewer("synthetic-reviewer"))
    finally:
        ledger.close()
    seeded = stage.capture_state_authority(state)["projection"]
    expected = copy.deepcopy(source)
    stage._expect_empty_schema_additions(expected, seeded)
    assert expected != seeded


# --- state manifest and closed-state digest -------------------------------------------------------

def _manifest_state(tmp_path):
    state = _state(tmp_path, deletions=1)
    (state / "data" / "graph").mkdir()
    (state / "data" / "graph" / "context.json").write_text("{}")
    (state / "data" / "privacy.db-wal").write_bytes(b"")
    return state


def test_manifest_covers_files_and_excludes_only_the_seal_records(tmp_path):
    state = _manifest_state(tmp_path)
    digest = stage.state_manifest_sha256(state)
    assert digest == stage.state_manifest_sha256(state)  # lstat only: reading changes nothing
    for name in (stage._MARKER, ".kza-target-ownership.json"):
        (state / name).write_text("{}")
    assert stage.state_manifest_sha256(state) == digest
    (state / "data" / stage._MARKER).write_text("{}")  # only the top-level seal records are excluded
    assert stage.state_manifest_sha256(state) != digest
    (state / "data" / stage._MARKER).unlink()
    assert stage.state_manifest_sha256(state) == digest
    (state / "data" / "privacy.db-wal").write_bytes(b"x")  # -wal files are payload state
    assert stage.state_manifest_sha256(state) != digest
    digest = stage.state_manifest_sha256(state)
    (state / "data" / "privacy.db-shm").write_bytes(b"y" * 32768)  # -shm never is (review K1)
    assert stage.state_manifest_sha256(state) == digest


def _wal_database(path):
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE payload (x)")
        db.execute("INSERT INTO payload VALUES (1)")
        db.commit()


def _read_only_image(path):
    with closing(sqlite3.connect(":memory:")) as image:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
            source.backup(image)


def test_a_second_read_only_reader_leaves_the_manifest_equal(tmp_path):
    """Review K1: the second ro reader rewrites -shm (mtime) but nothing the manifest covers."""
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    _wal_database(state / "data" / "payload.db")
    _read_only_image(state / "data" / "payload.db")  # the child's own snapshot
    digest = stage.state_manifest_sha256(state)
    shm = state / "data" / "payload.db-shm"
    before = shm.stat().st_mtime_ns if shm.exists() else None
    time.sleep(0.05)
    _read_only_image(state / "data" / "payload.db")  # an operator or backup reader before the seal
    assert stage.state_manifest_sha256(state) == digest
    if before is not None:
        assert shm.exists()  # the reader really touched the excluded file


def test_manifest_detects_a_restored_older_file_and_a_touch(tmp_path):
    state = _manifest_state(tmp_path)
    backup = tmp_path / "backup.json"
    shutil.copy2(state / "data" / "graph" / "context.json", backup)
    time.sleep(0.01)
    (state / "data" / "graph" / "context.json").write_text("[]")  # a later version, same size
    digest = stage.state_manifest_sha256(state)
    shutil.copy2(backup, state / "data" / "graph" / "context.json")  # restored in place (same inode), cp -p
    assert stage.state_manifest_sha256(state) != digest
    digest = stage.state_manifest_sha256(state)
    replacement = tmp_path / "replacement.json"
    shutil.copy2(state / "data" / "graph" / "context.json", replacement)
    os.replace(replacement, state / "data" / "graph" / "context.json")  # same bytes and mtime, new inode
    assert stage.state_manifest_sha256(state) != digest
    digest = stage.state_manifest_sha256(state)
    info = (state / "data" / "graph" / "context.json").stat()
    os.utime(state / "data" / "graph" / "context.json", ns=(info.st_atime_ns, info.st_mtime_ns + 1))
    assert stage.state_manifest_sha256(state) != digest  # touch-only is refused too (conservative)


def test_manifest_refuses_a_symlink(tmp_path):
    state = _manifest_state(tmp_path)
    (state / "data" / "link").symlink_to(state / "data" / "graph")
    with pytest.raises(ValueError, match="runtime_admission_state_symlink"):
        stage.state_manifest_sha256(state)


def test_closed_state_digest_covers_exactly_the_unsealed_subset():
    binding = {"state_identity": {"device": 1, "inode": 2}, "store_roots": {".": {"device": 1, "inode": 2}},
               "schemas": {"data/a.db": "a" * 64}, "sdk_versions": {}, "ownership_sha256": None,
               "authority": {"digest": "b" * 64, "projection": {}}, "slot": "x", "release_sha256": "c" * 64}
    digest = stage.closed_state_sha256(binding)
    assert stage.closed_state_sha256({**binding, "slot": "y", "release_sha256": "d" * 64}) == digest
    for key, value in (("state_identity", {}), ("store_roots", {}), ("schemas", {}),
                       ("sdk_versions", {"chromadb": "1"}), ("ownership_sha256", "e" * 64),
                       ("authority", {"digest": "f" * 64})):
        assert stage.closed_state_sha256({**binding, key: value}) != digest


# --- receipt schema 2 -----------------------------------------------------------------------------

BASE = {"schema_version": 1, "binding_id": "a" * 64, "generation": 2, "run_id": "b" * 32,
        "child_pid": 42, "status": "closed", "resources": dict.fromkeys(stage._CLOSED_RESOURCES, True)}
RECORD = {"schema_version": 2, "binding_id": "a" * 64, "generation": 2, "run_id": "b" * 32, "child_pid": 42,
          "closed_state_sha256": "c" * 64, "state_manifest_sha256": "d" * 64}


def _write_record(root, value):
    path = stage._receipt_state_path(root, "b" * 32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_absent_record_is_an_older_slot_and_the_receipt_keeps_schema_one(tmp_path):
    assert stage.read_receipt_state(tmp_path, BASE) is None
    proof = {"running_head": {"binding_id": "a" * 64, "generation": 2, "run_id": "b" * 32}, "child_pid": 42}
    stage._validate_closure_receipt(proof, BASE)  # the shared closure format is unchanged
    with pytest.raises(ValueError, match="runtime_admission_source_close_proof"):
        stage._validate_closure_receipt(proof, {**BASE, "schema_version": 2})


@pytest.mark.parametrize("record, accepted", [
    (RECORD, True),
    ({**RECORD, "schema_version": 1}, False),
    ({key: value for key, value in RECORD.items() if key != "state_manifest_sha256"}, False),
    ({**RECORD, "closed_state_sha256": "C" * 64}, False),
    ({**RECORD, "extra": True}, False),
    ({**RECORD, "child_pid": 43}, False),
    ({**RECORD, "generation": 3}, False),
    ({**RECORD, "binding_id": "e" * 64}, False),
])
def test_receipt_state_record_is_exact_and_names_the_receipts_run(tmp_path, record, accepted):
    _write_record(tmp_path, record)
    if accepted:
        assert stage.read_receipt_state(tmp_path, BASE) == record
    else:
        with pytest.raises(ValueError, match="runtime_admission_receipt_state"):
            stage.read_receipt_state(tmp_path, BASE)


def test_verify_receipt_state_checks_the_manifest_before_the_binding(tmp_path):
    state = _manifest_state(tmp_path)
    binding = {"state_identity": {}, "store_roots": {}, "schemas": {}, "sdk_versions": {},
               "ownership_sha256": None, "authority": {"digest": "b" * 64}}
    record = {**RECORD, "state_manifest_sha256": stage.state_manifest_sha256(state),
              "closed_state_sha256": stage.closed_state_sha256(binding)}
    stage.verify_receipt_state(record, state, binding)
    with pytest.raises(ValueError, match="runtime_admission_closed_state_mismatch"):
        stage.verify_receipt_state(record, state, {**binding, "schemas": {"x": "y"}})
    with pytest.raises(ValueError, match="runtime_admission_receipt_state"):
        stage.verify_receipt_state(BASE, state)
    (state / "data" / "graph" / "context.json").write_text("{} ")
    with pytest.raises(ValueError, match="runtime_admission_state_manifest_mismatch"):
        stage.verify_receipt_state(record, state, {**binding, "schemas": {"x": "y"}})


# --- the controller's immediate seal with a schema-2 child (working-tree stage_runtime) -----------

@pytest.fixture(scope="module")
def admission_archive(tmp_path_factory):
    from scripts.build_runtime import build_runtime
    destination = tmp_path_factory.mktemp("admission-artifact") / "runtime.tar.gz"
    build_runtime(REPO, destination)
    return destination


# The child closes with the working tree's stage_runtime (schema 2). Optionally
# it first writes a new version of a state file, then restores the older backup
# in place (cp -p: same inode, the backup's mtime) after its receipt.
_SCHEMA_TWO_CHILD = "\n".join([
    "import importlib.util, os, shutil, sys",
    "spec = importlib.util.spec_from_file_location('working_stage', sys.argv[1])",
    "module = importlib.util.module_from_spec(spec)",
    "spec.loader.exec_module(module)",
    "if len(sys.argv) > 2:",
    "    with open(sys.argv[3], 'w') as handle:",
    "        handle.write('{\"v\": 2}')  # the version the child closes with",
    "module.write_admission_receipt(os.environ['KZA_RUNTIME_ADMISSION'], dict.fromkeys(",
    "    ['graph_closed', 'privacy_closed', 'sdk_closed', 'native_closed'], True))",
    "if len(sys.argv) > 2:",
    "    shutil.copy2(sys.argv[2], sys.argv[3])  # an older backup restored in place after the receipt",
])


def _prepared(archive, tmp_path):
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text("schema: admission-test\n")
    (source / "data" / "notes.json").write_text("{\"v\": 1}")
    (source / "data" / "empty.db").write_bytes(b"")  # a database the seal opens as an image
    root = (tmp_path / "staging")
    stage.prepare_slot(archive, root, "candidate", source)
    root = root.resolve()
    return root, root / "slots" / "candidate" / "state"


def _child(*arguments):
    return [sys.executable, "-I", "-B", "-c", _SCHEMA_TWO_CHILD, str(REPO / "scripts" / "stage_runtime.py"),
            *map(str, arguments)]


def test_schema_two_receipt_seals_and_the_manifest_comes_before_any_image(admission_archive, tmp_path,
                                                                          monkeypatch):
    root, state = _prepared(admission_archive, tmp_path)
    events = []
    original_manifest, original_image, original_popen = (stage.state_manifest_sha256, stage._database_image,
                                                         stage.subprocess.Popen)

    def manifest(path):
        events.append("manifest")
        return original_manifest(path)

    def image(path):
        events.append("image")
        return original_image(path)

    def popen(*args, **kwargs):
        events.append("spawn")
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(stage, "state_manifest_sha256", manifest)
    monkeypatch.setattr(stage, "_database_image", image)
    monkeypatch.setattr(stage.subprocess, "Popen", popen)
    stage.run_slot(root, "candidate", _child(), timeout=60)
    head = stage.read_admission_head(root)
    assert head["status"] == "closed"
    receipt = json.loads((root / "admission" / "receipts" / (
        stage._closure_proof(root, head["binding_id"])["running_head"]["run_id"] + ".json")).read_text())
    assert receipt["schema_version"] == 1  # the shared receipt format is unchanged
    record = stage.read_receipt_state(root, receipt)
    assert record["schema_version"] == 2 and set(stage._RECEIPT_STATE_KEYS) <= record.keys()
    after_child = events[len(events) - events[::-1].index("spawn"):]
    assert after_child and after_child[0] == "manifest" and "image" in after_child, after_child
    monkeypatch.undo()
    # The closed state verifies offline too (the recovery helper), and closure proofs accept schema 2.
    stage.verify_receipt_state(record, state)
    with stage.AdmissionController(root) as controller:
        controller._require_lease()


def test_a_file_restored_after_the_receipt_is_refused_as_unknown(admission_archive, tmp_path):
    root, state = _prepared(admission_archive, tmp_path)
    older = tmp_path / "older-notes.json"
    shutil.copy2(state / "data" / "notes.json", older)
    with pytest.raises(ValueError, match="runtime_admission_state_manifest_mismatch"):
        stage.run_slot(root, "candidate", _child(older, state / "data" / "notes.json"), timeout=60)
    head = stage.read_admission_head(root)
    assert head["status"] == "unknown"


def test_schema_one_child_is_still_sealed_by_run(admission_archive, tmp_path):
    """Design v3 N7: run() still seals an older slot that writes no closed-state record."""
    root, _state_root = _prepared(admission_archive, tmp_path)
    child = "\n".join([
        "import importlib.util, json, os",
        "from pathlib import Path",
        "ticket = json.loads(os.environ['KZA_RUNTIME_ADMISSION'])",
        "receipt = {'schema_version': 1, 'binding_id': ticket['binding_id'], 'generation': ticket['generation'],",
        "           'run_id': ticket['run_id'], 'child_pid': os.getpid(), 'status': 'closed',",
        "           'resources': dict.fromkeys(['graph_closed', 'privacy_closed', 'sdk_closed', 'native_closed'], True)}",
        "path = Path(ticket['admission_root']) / 'admission' / 'receipts' / (ticket['run_id'] + '.json')",
        "path.parent.mkdir(parents=True, exist_ok=True)",
        "path.write_text(json.dumps(receipt, sort_keys=True, separators=(',', ':')))",
    ])
    stage.run_slot(root, "candidate", [sys.executable, "-I", "-B", "-c", child], timeout=60)
    assert stage.read_admission_head(root)["status"] == "closed"


def test_a_policy_change_after_the_receipt_is_refused_by_the_closed_state_digest(admission_archive, tmp_path):
    """The ownership policy is outside the manifest but inside closed_state_sha256."""
    root, state = _prepared(admission_archive, tmp_path)
    policy = state / ".kza-target-ownership.json"
    child = _SCHEMA_TWO_CHILD + "\n" + "\n".join([
        "import json",
        "path = os.path.join(os.path.realpath('data'), '..', '.kza-target-ownership.json')",
        "value = json.load(open(path)) if os.path.exists(path) else None",
        "if value is not None:",
        "    open(path, 'w').write(json.dumps(value, indent=1))  # same policy, different bytes",
        "else:",
        "    open(path, 'w').write('{}')",
    ])
    with pytest.raises(ValueError, match="runtime_admission_closed_state_mismatch"):
        stage.run_slot(root, "candidate", [sys.executable, "-I", "-B", "-c", child,
                                           str(REPO / "scripts" / "stage_runtime.py")], timeout=60)
    assert stage.read_admission_head(root)["status"] == "unknown"
    assert policy.exists()


def test_a_read_only_reader_between_close_and_seal_does_not_cause_unknown(admission_archive, tmp_path):
    """Review K1, end to end: after the receipt, two ro readers open a WAL database before the seal."""
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text("schema: admission-test\n")
    root = tmp_path / "staging"
    stage.prepare_slot(admission_archive, root, "candidate", source)
    root = root.resolve()
    state = root / "slots" / "candidate" / "state"
    with closing(sqlite3.connect(state / "data" / "wal.db")) as db:
        db.execute("PRAGMA journal_mode=WAL")  # an empty WAL database (no table to classify)
    child = _SCHEMA_TWO_CHILD + "\n" + "\n".join([
        "import sqlite3, time",
        "from contextlib import closing",
        "path = os.path.join(os.path.realpath('data'), 'wal.db')",
        "for _ in range(2):",
        "    with closing(sqlite3.connect(':memory:')) as image:",
        "        with closing(sqlite3.connect('file:' + path + '?mode=ro', uri=True)) as source:",
        "            source.backup(image)",
        "    time.sleep(0.05)",
    ])
    stage.run_slot(root, "candidate", [sys.executable, "-I", "-B", "-c", child,
                                       str(REPO / "scripts" / "stage_runtime.py")], timeout=60)
    assert stage.read_admission_head(root)["status"] == "closed"


def test_a_created_record_directory_is_made_durable_in_its_parent(tmp_path, monkeypatch):
    """Review K6: the first receipt-state record fsyncs admission/ after creating receipt-states/."""
    admission = tmp_path / "admission"
    admission.mkdir()
    synced, original = [], stage.os.fsync

    def fsync(descriptor):
        synced.append(os.fstat(descriptor).st_ino)
        return original(descriptor)

    monkeypatch.setattr(stage.os, "fsync", fsync)
    stage._atomic_record(admission / "receipt-states" / "r.json", {"schema_version": 2}, immutable=True)
    assert admission.stat().st_ino in synced
