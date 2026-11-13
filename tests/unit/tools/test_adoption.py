"""U3b-0e2 adoption library: the pieces, on synthetic trees (the b0866e8 pipeline is integration).

Adoption design v3 sections 1-5; reviews v1 F9/F10, v2 N2/N7/N11/N12d.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
import yaml

from scripts import adoption

REPOSITORY = Path(__file__).resolve().parents[3]
MARKER = b"KZA-UNIT-DROP-MARKER"


def _layout(root: Path) -> adoption.SourceLayout:
    root = root.resolve()
    legacy = root / "b" / "state"
    for name in ("config", "data", "logs"):
        (legacy / name).mkdir(parents=True)
    (legacy / ".env").write_text("A=1\n")
    (legacy / "config" / "settings.yaml").write_text("observability:\n  metrics_path: /home/kza/logs/kza-metrics.jsonl\n")
    container = root / "c" / "state"
    container.mkdir(parents=True)
    (root / "kza" / "models").mkdir(parents=True)
    (root / "logs").mkdir()
    for name in ("config", "data", ".env", "logs"):
        os.symlink(Path("..") / ".." / "b" / "state" / name, container / name)
    os.symlink(Path("..") / ".." / "kza" / "models", container / "models")
    return adoption.SourceLayout(container=container, legacy_state=legacy, models_root=root / "kza" / "models",
                                 legacy_checkout=root / "kza", metrics_root=root / "logs", code_index=root / "ci")


def _db(path: Path, statements=(), *, wal: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        if wal:
            db.execute("PRAGMA journal_mode=WAL")
        for statement in statements:
            db.execute(statement)


# --- pins: the controller never imports src, so its copies are pinned here -------------------------

def test_controller_copies_match_r1():
    from src.analytics.event_evidence import event_target_id
    from src.core import entry_guard, schema_only
    from src.core.adoption_schema import RESIDUE_REASON
    from src.memory.erasure_journal import ErasureError, ErasureParticipant
    from src.memory.index_deletions import IndexCleanupError

    assert adoption.SCHEMA_ONLY_BOOTSTRAP == entry_guard.SCHEMA_ONLY_BOOTSTRAP
    assert adoption.schema_only_command("p", "r", "x") == entry_guard.schema_only_command("p", "r", "x")
    assert adoption.SCHEMA_ONLY_ENV == entry_guard.SCHEMA_ONLY_ENV
    assert adoption.COPY_MARKER == schema_only.COPY_MARKER
    assert adoption.ERASURE_PARTICIPANTS == tuple(item.value for item in ErasureParticipant)
    assert adoption.RESIDUE_REASON == RESIDUE_REASON == ErasureError.RESTARTED.value
    assert adoption.INDEX_RECHECK == IndexCleanupError.RECHECK.value
    descriptor = {"path": "/x/events.db", "table": "events_policy_v1"}
    assert adoption.event_target_id(descriptor) == event_target_id(descriptor)


def test_target_id_is_the_journal_key(tmp_path):
    from src.memory.fact_ledger import FactLedger
    from src.memory.erasure_journal import ErasureParticipant

    ledger = FactLedger(str(tmp_path / "p.db"))
    ledger.initialize()
    try:
        key = ledger.erasures.register_target(ErasureParticipant.EVENTS, 1, {"path": "/x/e.db", "table": "events_policy_v1"})
    finally:
        ledger.close()
    assert key == adoption.target_id("events", 1, {"path": "/x/e.db", "table": "events_policy_v1"})


def test_the_drop_list_matches_the_pre_policy_ddl():
    from tests.mocks import adoption_legacy

    if not adoption_legacy.available():
        pytest.skip("8558ff1 is not available in this clone")
    db = sqlite3.connect(":memory:")
    db.execute(adoption_legacy.legacy_events_ddl())
    assert tuple(row[1] for row in db.execute("PRAGMA table_info(events)")) == \
        adoption.DROP_LIST["data/events.db"]["events"] == adoption.LEGACY_EVENTS_COLUMNS


# --- the source layout ----------------------------------------------------------------------------

def test_the_five_readlinks_are_read_and_classed(tmp_path):
    layout = _layout(tmp_path)
    result = adoption.read_source_layout(layout)
    assert set(result["readlinks"]) == set(adoption.CONTAINER_LINKS)
    assert result["classes"] == {"config": "legacy_state", "data": "legacy_state", ".env": "legacy_state",
                                 "models": "models", "logs": "legacy_state"}


@pytest.mark.parametrize("change", ["foreign_data", "extra_entry", "real_directory", "models_elsewhere"])
def test_an_unexpected_container_refuses(tmp_path, change):
    layout = _layout(tmp_path)
    if change == "foreign_data":
        (tmp_path / "other").mkdir()
        (layout.container / "data").unlink()
        os.symlink(tmp_path / "other", layout.container / "data")
    elif change == "extra_entry":
        (layout.container / "notes").write_text("x")
    elif change == "real_directory":
        (layout.container / "config").unlink()
        (layout.container / "config").mkdir()
    else:
        (layout.container / "models").unlink()
        (tmp_path / "m2").mkdir()
        os.symlink(tmp_path / "m2", layout.container / "models")
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_source_layout$"):
        adoption.read_source_layout(layout)


# --- the source guard (T-D2) ------------------------------------------------------------------------

def test_the_guard_refuses_every_write_and_connect_on_a_source(tmp_path):
    layout = _layout(tmp_path)
    source = layout.legacy_state / "data" / "x.db"
    _db(source, ["CREATE TABLE t(a)"])
    copy = tmp_path / "copy.db"
    attempts = [lambda: sqlite3.connect(source), lambda: sqlite3.connect(source.as_uri() + "?mode=ro", uri=True),
                lambda: sqlite3.connect(layout.container / "data" / "x.db"),
                lambda: open(source, "ab"), lambda: open(source, "r+b"),
                lambda: os.open(source, os.O_RDWR), lambda: os.remove(source),
                lambda: os.rename(source, tmp_path / "moved"), lambda: os.mkdir(layout.legacy_state / "d"),
                lambda: os.chmod(source, 0o600), lambda: os.utime(source)]
    with adoption.SourceGuard(*adoption.source_roots(layout)) as guard:
        for attempt in attempts:
            with pytest.raises(adoption.AdoptionRefusal, match="runtime_admission_adoption_source_access"):
                attempt()
        with open(source, "rb") as stream:  # reading is the copy discipline
            assert stream.read(16) == adoption.SQLITE_HEADER
        sqlite3.connect(copy).close()
    assert guard.refused == len(attempts) and guard.connects == 1
    sqlite3.connect(source).close()  # inactive once the guard exits


# --- the byte copy (N7, F9) -------------------------------------------------------------------------

def _tree(layout: adoption.SourceLayout) -> None:
    data = layout.legacy_state / "data"
    _db(data / "lists.db", ["CREATE TABLE lists(id)", "INSERT INTO lists VALUES (1)"])
    _db(data / "routines.db", ["CREATE TABLE r(id)", "INSERT INTO r VALUES (1)"], wal=True)
    writer = sqlite3.connect(data / "routines.db")
    writer.execute("INSERT INTO r VALUES (2)")
    writer.commit()  # stays in the -wal while the writer is open
    _db(data / "store", ["CREATE TABLE s(id)"])  # SQLite by header only
    (data / "store-wal").write_bytes(b"")
    _db(data / "audit.db", ["CREATE TABLE events(timestamp, kind, payload_json)"])
    (data / "audit.db-journal").write_bytes(b"")
    _db(data / "events.db", ["CREATE TABLE e(id)"])
    (data / "users.json").write_text("{}")
    return writer


def test_every_file_is_hashed_and_each_database_travels_with_its_side_files(tmp_path):
    layout = _layout(tmp_path)
    writer = _tree(layout)
    try:
        before = adoption.file_manifest(layout.legacy_state)
        result = adoption.copy_source(layout, tmp_path / "work", tmp_path / "scratch")
        wal = (layout.legacy_state / "data" / "routines.db-wal").read_bytes()
    finally:
        writer.close()  # the last close checkpoints and removes the source -wal
    names = [item[0] for item in result["manifest"]]
    assert [item[:4] + [item[4]] for item in result["manifest"]] == before  # the same pass, same digests
    assert "data/audit.db" in names and "data/audit.db-journal" in names and "logs" not in names
    work = tmp_path / "work" / "data"
    assert wal and (work / "routines.db-wal").read_bytes() == wal
    assert not (work / "routines.db-shm").exists() and "data/routines.db-shm" in names
    assert (work / "store-wal").exists() and not (work / "audit.db").exists()
    assert not (work / "audit.db-journal").exists() and not (work / "events.db").exists()
    assert (tmp_path / "scratch" / "data" / "events.db").exists()
    assert result["counts"]["sqlite_files"] == 5 and result["counts"]["not_carried_files"] == 2
    assert result["counts"]["shm_not_copied"] == 1


def test_a_copy_that_differs_from_its_source_refuses(tmp_path, monkeypatch):
    layout = _layout(tmp_path)
    (layout.legacy_state / "data" / "users.json").write_text("{}")
    monkeypatch.setattr(adoption, "_sha256_file", lambda path: "0" * 64)
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_copy_mismatch$"):
        adoption.copy_source(layout, tmp_path / "work", tmp_path / "scratch")


@pytest.mark.parametrize("bad", ["symlink", "fifo", "orphan_journal"])
def test_links_specials_and_orphan_journals_refuse(tmp_path, bad):
    layout = _layout(tmp_path)
    data = layout.legacy_state / "data"
    if bad == "symlink":
        os.symlink(tmp_path, data / "escape")
        code = "source_symlink"
    elif bad == "fifo":
        os.mkfifo(data / "pipe")
        code = "source_special"
    else:
        (data / "notes.txt-journal").write_bytes(b"x")
        code = "orphan_journal"
    with pytest.raises(adoption.AdoptionRefusal, match="runtime_admission_adoption_" + code):
        adoption.copy_source(layout, tmp_path / "work", tmp_path / "scratch")


def test_the_manifest_covers_non_sqlite_files(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "settings.yaml").write_text("a: 1\n")
    first = adoption.manifest_digest(adoption.file_manifest(tmp_path))
    info = (tmp_path / "config" / "settings.yaml").stat()
    (tmp_path / "config" / "settings.yaml").write_text("a: 2\n")  # same inode and size
    os.utime(tmp_path / "config" / "settings.yaml", ns=(info.st_atime_ns, info.st_mtime_ns))  # same mtime
    assert adoption.manifest_digest(adoption.file_manifest(tmp_path)) != first
    os.symlink(tmp_path / "config", tmp_path / "link")
    with pytest.raises(adoption.AdoptionRefusal, match="source_symlink"):
        adoption.file_manifest(tmp_path)


# --- recovery and the drop (F9, N2) ----------------------------------------------------------------

def test_a_hot_journal_is_rolled_back_and_counted(tmp_path):
    path = tmp_path / "d.db"
    _db(path, ["CREATE TABLE t(a)", "INSERT INTO t VALUES ('committed')"])
    work = tmp_path / "w.db"
    import shutil
    shutil.copyfile(path, work)
    db = sqlite3.connect(work, isolation_level=None)
    db.execute("PRAGMA cache_size=1")
    db.execute("BEGIN IMMEDIATE")
    db.executemany("INSERT INTO t VALUES (?)", [("x" * 500,)] * 300)
    shutil.copyfile(work, path)
    shutil.copyfile(str(work) + "-journal", str(path) + "-journal")
    db.execute("ROLLBACK")
    db.close()
    assert adoption.recover_copies(tmp_path) == {"hot_journals_recovered": 1, "wal_checkpointed": 0, "databases": 2}
    with closing(sqlite3.connect(path)) as check:
        assert check.execute("SELECT a FROM t").fetchall() == [("committed",)]
    assert not Path(str(path) + "-journal").exists()


def test_a_journal_beside_a_wal_database_refuses(tmp_path):
    path = tmp_path / "d.db"
    _db(path, ["CREATE TABLE t(a)"], wal=True)
    Path(str(path) + "-journal").write_bytes(b"\xd9\xd5\x05\xf9\x20\xa1\x63\xd7" + b"\0" * 504)
    with pytest.raises(adoption.AdoptionRefusal, match="journal_unexpected"):
        adoption.recover_sqlite_file(path)


def _legacy_events(path: Path, rows: int = 300) -> None:
    columns = ", ".join(name + (" INTEGER PRIMARY KEY AUTOINCREMENT" if name == "id" else "")
                        for name in adoption.LEGACY_EVENTS_COLUMNS)
    _db(path, ["CREATE TABLE events (" + columns + ")", "CREATE INDEX idx_user ON events(user_id)",
               "CREATE TABLE events_policy_v1(id INTEGER PRIMARY KEY, sources_json TEXT)"])
    with closing(sqlite3.connect(path)) as db, db:
        db.executemany("INSERT INTO events(timestamp, trigger_phrase, context_json) VALUES (?,?,?)",
                       [(index, MARKER.decode() + str(index), MARKER.decode()) for index in range(rows)])
        db.execute("INSERT INTO events_policy_v1 VALUES (1, '[]')")


def test_the_drop_runs_in_the_scratch_and_only_vacuum_into_reaches_the_destination(tmp_path):
    scratch, destination = tmp_path / "scratch", tmp_path / "work"
    _legacy_events(scratch / "data" / "events.db")
    assert MARKER in (scratch / "data" / "events.db").read_bytes()
    assert adoption.drop_and_vacuum(scratch, destination) == {"data/events.db": {"tables_dropped": 1}}
    target = destination / "data" / "events.db"
    assert MARKER not in target.read_bytes()
    assert not any(scratch.rglob("events.db*"))
    with closing(sqlite3.connect(target)) as db:
        assert db.execute("PRAGMA freelist_count").fetchone() == (0,)
        assert db.execute("SELECT COUNT(*) FROM events_policy_v1").fetchone() == (1,)
        assert db.execute("SELECT name FROM sqlite_master WHERE name LIKE 'events%' OR name LIKE 'idx_%'"
                          ).fetchall() == [("events_policy_v1",)]
    assert oct(target.stat().st_mode & 0o777) == "0o600"


def test_a_drop_of_an_unexpected_shape_refuses(tmp_path):
    _db(tmp_path / "scratch" / "data" / "events.db", ["CREATE TABLE events(timestamp, kind, payload_json)"])
    with pytest.raises(adoption.AdoptionRefusal, match="drop_shape:data/events.db:events"):
        adoption.drop_and_vacuum(tmp_path / "scratch", tmp_path / "work")
    assert not (tmp_path / "work" / "data" / "events.db").exists()


# --- contracts --------------------------------------------------------------------------------------

@pytest.mark.parametrize("relative, table, code", [
    ("data/ambient.db", "events", "legacy_table:data/ambient.db"),
    ("data/latency.db", "notes", "unknown_table:data/latency.db:notes"),
    ("data/memory_privacy.db", "latency_records", "unknown_table:data/memory_privacy.db:latency_records"),
])
def test_tables_outside_the_contracts_refuse(tmp_path, relative, table, code):
    _db(tmp_path / relative, ["CREATE TABLE " + table + "(a)"])
    with pytest.raises(adoption.AdoptionRefusal, match="runtime_admission_adoption_" + code):
        adoption.contract_census(tmp_path)
    if table != "events":
        assert adoption.contract_census(tmp_path, refuse=False)["outside_contracts"] == {relative: [table]}


def test_latency_records_is_scoped_to_its_database(tmp_path):
    from src.monitoring.latency_monitor import LatencyMonitor

    LatencyMonitor(str(tmp_path / "data" / "latency.db"))
    assert adoption.contract_census(tmp_path) == {"outside_contracts": {}}
    assert adoption.verify_contracts(tmp_path) == 1


# --- 2a: the configuration editor -------------------------------------------------------------------

def _b0866e8_settings() -> str:
    result = subprocess.run(["git", "show", "b0866e8:config/settings.yaml"], cwd=REPOSITORY,
                            capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip("b0866e8 is not available in this clone")
    return result.stdout


def test_the_editor_applies_the_table_and_keeps_every_comment():
    text = _b0866e8_settings()
    rows = adoption.transform_table()
    new = adoption.transform_config_text(text, rows)
    value = yaml.safe_load(new)
    assert value == adoption.apply_transform(yaml.safe_load(text), rows)
    assert value["paths"]["models_root"] == "/home/kza/kza/models"
    assert value["llm"]["model_path"].startswith("/home/kza/kza/models/Qwen2.5-72B-Instruct-Q8_0/")
    assert value["llm"]["lora_path"] == "" and value["hooks"]["audit_sqlite"] == {"enabled": False}
    assert "code_index" not in value and "audit_sqlite_path" not in value["hooks"]
    assert value["hooks"]["enabled"] is yaml.safe_load(text)["hooks"]["enabled"]
    old_comments = [line for line in text.split("\n") if line.lstrip().startswith("#")]
    new_comments = [line for line in new.split("\n") if line.lstrip().startswith("#")]
    deleted = text[text.index("\ncode_index:"):]
    assert new_comments == [line for line in old_comments
                            if line not in [item for item in deleted.split("\n") if item.lstrip().startswith("#")]
                            or line in new_comments]
    assert "# OFF por default" in new  # a trailing comment on an untouched line survives
    # Idempotent: the transformed file is a fixpoint.
    assert adoption.transform_config_text(new, rows) == new


def test_the_transformed_settings_pass_the_controller_preflight(tmp_path):
    from tests.unit.tools.test_adoption_preflight import preflight

    value = yaml.safe_load(adoption.transform_config_text(_b0866e8_settings(), adoption.transform_table()))
    problems = preflight(tmp_path, value)
    assert not [item for item in problems if item.split(":", 1)[1] in {
        ".".join(row.key) for row in adoption.transform_table()}]


@pytest.mark.parametrize("key, value", [
    (("stt", "model"), "./models/other"), (("hooks", "audit_sqlite_path"), "/elsewhere/audit.db"),
    (("paths", "models_root"), "/srv/models"), (("llm", "model_path"), "/opt/qwen.gguf"),
])
def test_a_value_matching_neither_side_refuses_by_key(key, value):
    config = yaml.safe_load(_b0866e8_settings())
    owner = config
    for part in key[:-1]:
        owner = owner.setdefault(part, {})
    owner[key[-1]] = value
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_config_unexpected:"
                       + ".".join(key) + "$"):
        adoption.transform_config_text(yaml.safe_dump(config), adoption.transform_table())


def test_the_post_check_refuses_an_editor_slip(monkeypatch):
    monkeypatch.setattr(adoption, "_yaml_scalar", lambda value: '"slip"' if value == "" else
                        (json.dumps(value) if type(value) is str else str(value).lower()))
    with pytest.raises(adoption.AdoptionRefusal, match="config_postcheck"):
        adoption.transform_config_text(_b0866e8_settings(), adoption.transform_table())


# --- .env (N1, N12d) ----------------------------------------------------------------------------------

def test_env_reports_classes_and_refuses_by_key_only(tmp_path):
    layout = _layout(tmp_path)
    env = tmp_path / "env"
    env.write_text("HF_HOME=" + str(layout.models_root) + "/hf\nTOKEN=abc\n# MODELS=/x\nexport X=./data/y\n")
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_env_path:X$"):
        adoption.env_census(env, layout)  # relative values resolve against the container (the service cwd)
    env.write_text("HF_HOME=" + str(layout.models_root) + "/hf\nTOKEN=abc\n")
    assert adoption.env_census(env, layout) == {"keys": {"HF_HOME": "models"}}
    secret = str(layout.legacy_state / "data" / "secret-name")
    env.write_text("CHROMA_PATH=" + secret + "\n")
    with pytest.raises(adoption.AdoptionRefusal) as refusal:
        adoption.env_census(env, layout)
    assert str(refusal.value) == "runtime_admission_adoption_env_path:CHROMA_PATH" and "secret" not in str(refusal.value)
    env.write_text("KZA_AUDIT_DB=/tmp/a.db\n")
    with pytest.raises(adoption.AdoptionRefusal, match="^runtime_admission_adoption_audit_env$"):
        adoption.env_census(env, layout)


# --- legacy_training (P1=D) -------------------------------------------------------------------------

def test_legacy_training_roots(tmp_path):
    layout = _layout(tmp_path)
    lora = layout.models_root / "lora_adapters"
    roots = [str(lora), str(lora / "nightly"), str(lora / "whisper")]
    result = adoption.legacy_training_disposition(layout, roots)
    assert result["retirement"] == "not_needed"
    assert sorted(result["mapping"].values()) == ["data/training_outputs/lora", "data/training_outputs/nightly",
                                                  "data/training_outputs/whisper"]
    (lora / "nightly").mkdir(parents=True)  # empty: still no retirement needed
    assert adoption.legacy_training_disposition(layout, roots)["retirement"] == "not_needed"
    with pytest.raises(adoption.AdoptionRefusal, match="legacy_training_root"):
        adoption.legacy_training_disposition(layout, [str(layout.models_root / "whisper-v3-turbo")])
    os.symlink(tmp_path, lora / "link")
    with pytest.raises(adoption.AdoptionRefusal, match="legacy_training_special"):
        adoption.legacy_training_disposition(layout, roots)


@pytest.mark.parametrize("record, code", [
    ({"schema_version": 1, "decision_id": "", "roots": []}, "retirement_invalid"),
    ({"schema_version": 2, "decision_id": "d", "roots": [{}]}, "retirement_invalid"),
    ({"schema_version": 1, "decision_id": "d", "roots": [{"root": "/elsewhere", "absence": "enoent"}]},
     "retirement_invalid"),
    ({"schema_version": 1, "decision_id": "d", "roots": [{"root": "LORA", "absence": "empty"}]}, "retirement_absence"),
])
def test_a_retirement_record_must_be_valid_and_still_true(tmp_path, record, code):
    layout = _layout(tmp_path)
    lora = str(layout.models_root.resolve() / "lora_adapters")
    record = json.loads(json.dumps(record).replace("LORA", lora))
    with pytest.raises(adoption.AdoptionRefusal, match=code):
        adoption.legacy_training_disposition(layout, [lora], record)
    valid = {"schema_version": 1, "decision_id": "d", "roots": [{"root": lora, "absence": "enoent"}]}
    assert adoption.legacy_training_disposition(layout, [lora], valid)["retirement"] == "d"


# --- the metrics store set (F10) --------------------------------------------------------------------

def test_the_metrics_store_set_and_its_refusals(tmp_path):
    layout = _layout(tmp_path)
    logs = layout.metrics_root
    name = "kza-metrics_policy_v1.jsonl"
    for item in (name, name + ".1", name + ".quarantine-1", name + ".1.quarantine-2", name + ".sqlite3.lock",
                 "kza-metrics.jsonl"):
        (logs / item).write_text("{}\n")
    _db(logs / (name + ".sqlite3"), ["CREATE TABLE staged(name TEXT PRIMARY KEY)"])
    found = adoption.metrics_store_set(layout, "/home/kza/logs/kza-metrics.jsonl")
    assert sorted(found["members"]) == sorted([name, name + ".1", name + ".quarantine-1", name + ".1.quarantine-2",
                                               name + ".sqlite3"])
    assert found["outside"] == ["kza-metrics.jsonl"]
    with adoption.SourceGuard(logs):
        result = adoption.copy_metrics_store(layout, "/home/kza/logs/kza-metrics.jsonl", tmp_path / "work")
    assert result["members"] == 5 and (tmp_path / "work" / "data" / "obs" / (name + ".quarantine-1")).exists()
    with closing(sqlite3.connect(logs / (name + ".sqlite3"))) as db, db:
        db.execute("INSERT INTO staged VALUES ('.diag-x.tmp')")
    with pytest.raises(adoption.AdoptionRefusal, match="metrics_unsettled"):
        adoption.copy_metrics_store(layout, "/home/kza/logs/kza-metrics.jsonl", tmp_path / "work2")
    (logs / ".diag-abc.tmp").write_text("")
    with pytest.raises(adoption.AdoptionRefusal, match="metrics_unsettled"):
        adoption.metrics_store_set(layout, "/home/kza/logs/kza-metrics.jsonl")


# --- the ledger seed (R1 code) ----------------------------------------------------------------------

def test_the_seed_is_deterministic_and_only_on_an_empty_saltless_ledger():
    from src.memory.deletion_ledger import (
        DeletionKind, DeletionLedgerError, append_deletion, initialize_deletion_ledger, seed_adoption_baseline,
        subject_digest, verify_chain,
    )

    entries = [(DeletionKind.VOICE_PROFILE_DELETED, "v", 5.0), (DeletionKind.FACT_DELETED, "f2", 9.0),
               (DeletionKind.CONVERSATION_ERASED, "c", 7.0), (DeletionKind.FACT_DELETED, "f1", 9.0)]
    db = sqlite3.connect(":memory:", isolation_level=None)
    initialize_deletion_ledger(db)
    db.execute("BEGIN IMMEDIATE")
    chain = seed_adoption_baseline(db, entries)
    db.execute("COMMIT")
    assert chain.count == 4 and verify_chain(db).chain_head == chain.chain_head
    [(salt,)] = db.execute("SELECT salt FROM deletion_ledger_salt_v1").fetchall()
    rows = db.execute("SELECT kind, subject_sha256, at FROM deletion_ledger_v1 ORDER BY seq").fetchall()
    assert rows == [("conversation_erased", subject_digest(salt, "c"), 7.0),
                    ("fact_deleted", subject_digest(salt, "f1"), 9.0),
                    ("fact_deleted", subject_digest(salt, "f2"), 9.0),
                    ("voice_profile_deleted", subject_digest(salt, "v"), 5.0)]
    db.execute("BEGIN IMMEDIATE")
    with pytest.raises(DeletionLedgerError, match="seed_not_empty"):
        seed_adoption_baseline(db, [])
    db.execute("ROLLBACK")
    other = sqlite3.connect(":memory:", isolation_level=None)
    initialize_deletion_ledger(other)
    other.execute("BEGIN IMMEDIATE")
    with pytest.raises(DeletionLedgerError, match="seed_duplicate"):
        seed_adoption_baseline(other, [(DeletionKind.FACT_DELETED, "f", 1.0)] * 2)
    with pytest.raises(DeletionLedgerError, match="kind_invalid"):
        seed_adoption_baseline(other, [(DeletionKind.CREDENTIAL_REVOKED, "x", 1.0)])
    append_deletion(other, DeletionKind.FACT_DELETED, "f", 1.0)
    other.execute("DELETE FROM deletion_ledger_v1")  # a salt without rows is not 'saltless'
    with pytest.raises(DeletionLedgerError, match="seed_not_empty"):
        seed_adoption_baseline(other, [])
    other.execute("ROLLBACK")


# --- the child ------------------------------------------------------------------------------------

def test_the_childs_refusal_becomes_the_adoptions(tmp_path):
    copy = tmp_path / "copy"
    (copy / "config").mkdir(parents=True)
    (copy / "config" / "settings.yaml").write_text("{}\n")
    with pytest.raises(adoption.AdoptionRefusal, match="^schema_only_state_unmarked$"):
        adoption.run_child(sys.executable, REPOSITORY, copy, "schema")
    adoption.write_copy_marker(copy, "e" * 64)
    assert adoption.run_child(sys.executable, REPOSITORY, copy, "schema")["initializers"] == 2
    with pytest.raises(adoption.AdoptionRefusal, match="^schema_only_adoption_time_invalid$"):
        adoption.run_child(sys.executable, REPOSITORY, copy, "seed")
    with pytest.raises(adoption.AdoptionRefusal, match="^schema_only_state_root_invalid$"):
        adoption.run_child(sys.executable, REPOSITORY, copy, "residue", at=1.0)
    with pytest.raises(adoption.AdoptionRefusal, match="plan_invalid"):
        adoption.write_copy_marker(tmp_path, "nothex")
