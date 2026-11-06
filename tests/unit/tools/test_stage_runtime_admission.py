"""Admission requires real owned closure, not a successful process exit alone."""

from concurrent.futures import ThreadPoolExecutor
import ast
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import time

import pytest

from scripts.build_runtime import build_runtime
from scripts import stage_runtime as stage


@pytest.fixture(scope="module")
def admission_archive(tmp_path_factory):
    destination = tmp_path_factory.mktemp("admission-artifact") / "runtime.tar.gz"
    build_runtime(Path(__file__).resolve().parents[3], destination)
    return destination


@pytest.fixture
def admission_root(admission_archive, tmp_path):
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text("schema: admission-test\n")
    root = tmp_path / "staging"
    stage.prepare_slot(admission_archive, root, "candidate", source)
    return root


def _command(source, *arguments):
    return [sys.executable, "-I", "-S", "-B", "-c", source, *map(str, arguments)]


def _assert_unknown(root, *, expected="unknown"):
    head = stage.read_admission_head(root)
    assert head["status"] == expected
    assert head["run_id"] is not None
    assert not (root / "admission" / "receipts" / (head["run_id"] + ".json")).exists()
    with pytest.raises(ValueError, match="runtime_admission_not_closed"):
        with stage.AdmissionController(root):
            pytest.fail("UNKNOWN must remain non-serving")


def _observe_real_processes(monkeypatch):
    original = subprocess.Popen
    observed = []

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        observed.append(process)
        return process

    # This is a passive handle recorder; every launch, signal and wait is native.
    monkeypatch.setattr(stage.subprocess, "Popen", launch)
    return observed


def test_zero_exit_without_owned_receipt_is_unknown(admission_root):
    with pytest.raises(ValueError, match="runtime_admission_close_receipt_missing"):
        stage.run_slot(admission_root, "candidate", _command("raise SystemExit(0)"), timeout=5)
    _assert_unknown(admission_root)


@pytest.mark.parametrize("metadata_failure", [False, True])
def test_timeout_with_zero_signal_exit_is_unknown(admission_root, tmp_path, monkeypatch, metadata_failure):
    processes = _observe_real_processes(monkeypatch)
    if metadata_failure:
        original_cas = stage._cas_head

        def fail_unknown(root, expected, binding_id, status, run_id=None):
            if status == "unknown":
                raise OSError("synthetic-control-write-failure")
            return original_cas(root, expected, binding_id, status, run_id)

        monkeypatch.setattr(stage, "_cas_head", fail_unknown)
    handled = tmp_path / "signal-handled"
    source = "\n".join([
        "from pathlib import Path",
        "import signal, sys, time",
        "def stop(signum, frame):",
        "    Path(sys.argv[1]).write_text('native-sigterm')",
        "    raise SystemExit(0)",
        "signal.signal(signal.SIGTERM, stop)",
        "time.sleep(60)",
    ])
    with pytest.raises(BaseExceptionGroup if metadata_failure else subprocess.TimeoutExpired):
        stage.run_slot(admission_root, "candidate", _command(source, handled), timeout=2)
    assert handled.read_text() == "native-sigterm"
    assert len(processes) == 1 and processes[0].returncode == 0
    assert processes[0].wait(timeout=0) == 0
    with pytest.raises(ProcessLookupError):
        os.kill(processes[0].pid, 0)
    _assert_unknown(admission_root, expected="running" if metadata_failure else "unknown")


def test_signal_with_zero_exit_is_unknown(admission_root, monkeypatch):
    processes = _observe_real_processes(monkeypatch)
    source = "\n".join([
        "import os, signal, sys",
        "signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))",
        "os.kill(os.getpid(), signal.SIGTERM)",
    ])
    with pytest.raises(ValueError, match="runtime_admission_close_receipt_missing"):
        stage.run_slot(admission_root, "candidate", _command(source), timeout=5)
    assert len(processes) == 1 and processes[0].wait(timeout=0) == 0
    _assert_unknown(admission_root)


def test_live_child_holds_root_admission_exclusively(admission_root, tmp_path):
    ready, release = tmp_path / "ready", tmp_path / "release"
    child = "\n".join([
        "from pathlib import Path",
        "import sys, time",
        "Path(sys.argv[1]).write_text('ready')",
        "while not Path(sys.argv[2]).exists(): time.sleep(0.01)",
    ])
    contender = "\n".join([
        "import importlib.util, pathlib, sys",
        "spec = importlib.util.spec_from_file_location('trusted_stage', sys.argv[1])",
        "module = importlib.util.module_from_spec(spec)",
        "spec.loader.exec_module(module)",
        "try:",
        "    with module.AdmissionController(pathlib.Path(sys.argv[2])):",
        "        raise AssertionError('concurrent admission accepted')",
        "except ValueError as error:",
        "    assert str(error) == 'runtime_admission_busy'",
        "    print('native-lock-denied')",
    ])
    with stage.AdmissionController(admission_root) as owner:
        with ThreadPoolExecutor(max_workers=1) as executor:
            job = executor.submit(owner.run, "candidate", _command(child, ready, release), timeout=10)
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert ready.read_text() == "ready"
                result = subprocess.run(
                    _command(contender, Path(stage.__file__).resolve(), admission_root),
                    capture_output=True, text=True, timeout=5, check=True,
                )
                assert result.stdout.strip() == "native-lock-denied"
                assert stage.read_admission_head(admission_root)["status"] == "running"
            finally:
                release.write_text("release")
            with pytest.raises(ValueError, match="runtime_admission_close_receipt_missing"):
                job.result(timeout=5)
    _assert_unknown(admission_root)


def _source_sql(filename, *, assignment=None, table=None):
    """Read actual schema literals without importing any application owner."""
    path = Path(__file__).resolve().parents[3] / filename
    tree = ast.parse(path.read_text())
    if assignment:
        return next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == assignment for target in node.targets))
    import re
    expression = re.compile(r"CREATE TABLE(?: IF NOT EXISTS)?\s+" + table + r"\s*\(", re.I)
    return next(node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                and type(node.value) is str and expression.search(node.value))


@pytest.fixture
def projection_state(tmp_path):
    state = tmp_path / "projection"
    (state / "data").mkdir(parents=True)
    (state / "config").mkdir()
    sources = json.dumps([{"capture_id": "capture", "conversation_id": "U",
                           "captured_at": 100, "expires_at": 200}])
    with sqlite3.connect(state / "data" / "authority.db") as db:
        for filename, assignment in (
            ("src/memory/fact_ledger.py", "_SCHEMA"),
            ("src/memory/voice_schema.py", "VOICE_SCHEMA"),
            ("src/wakeword/clip_store.py", "_SCHEMA"),
            ("src/ambient/store.py", "_SCHEMA"),
            ("src/routines/routine_store.py", "_SCHEMA"),
        ):
            db.executescript(_source_sql(filename, assignment=assignment))
        for filename, table in (
            ("src/memory/capture_control.py", "capture_runtime_control"),
            ("src/ambient/control.py", "ambient_runtime_control"),
            ("src/memory/capture_recorders.py", "capture_browser_recorders"),
            ("src/wakeword/clip_store.py", "stt_corrections_v1"),
            ("src/wakeword/clip_store.py", "conversation_turns_v1"),
            ("src/dashboard/live_receipts.py", "live_deliveries"),
            ("src/reminders/reminder_store.py", "reminders"),
            ("src/reminders/reminder_store.py", "reminder_home_actions"),
            ("src/lists/list_store.py", "lists"),
            ("src/lists/list_store.py", "list_items"),
        ):
            db.executescript(_source_sql(filename, table=table))
        db.execute("ALTER TABLE pending_audio_deletions ADD COLUMN conversation_id TEXT")
        db.execute("INSERT INTO voice_group_schema VALUES(1)")
        db.execute("INSERT INTO capture_runtime_control VALUES(1,'running',0)")
        db.execute("INSERT INTO ambient_runtime_control VALUES(1,0,0)")
        db.execute("INSERT INTO capture_browser_recorders VALUES('token','active','browser')")
        db.execute("INSERT INTO facts VALUES('f','ordinary','general',0.8,'automatic',100,200,1,NULL,NULL)")
        db.execute("INSERT INTO fact_sources VALUES('f','capture','U',100,200,'voice','[]')")
        db.execute("INSERT INTO wake_clips_v1 VALUES(1,'clip','target','bucket','room',0.5,?,100,200,1)", (sources,))
        db.execute("INSERT INTO stt_corrections_v1 VALUES('clip','original','corrected','u',16000,0)")
        db.execute("INSERT INTO conversation_turns_v1 VALUES('clip','group',?,'unmarked','digest')", (sources,))
        db.execute("INSERT INTO voice_groups VALUES('group',1,'unknown','automatic',0.5,100,200,?)", (sources,))
        db.execute("INSERT INTO utterances(id,room_id,t0,t1,text,created_at,capture_id,conversation_id,expires_at) "
                   "VALUES(1,'room',100,101,'ordinary',100,'capture','U',172900)")
        db.execute("INSERT INTO live_deliveries VALUES(1,'ticket','client','connection','event','nonce',100,200,'namespace')")
        db.execute("INSERT INTO live_delivery_sources VALUES('ticket','U')")
        db.execute("INSERT INTO live_delivery_lineage VALUES('ticket',1,'digest')")
        db.execute("INSERT INTO reminder_home_actions VALUES('request','{}','done',NULL)")
        db.execute("INSERT INTO routines VALUES('routine',1,'admin','now','{}')")
        db.execute("INSERT INTO routine_runs VALUES('run','routine','occurrence','admin','manual','done',NULL,'{}',NULL)")
        db.execute("INSERT INTO routine_run_actions VALUES('run',0,'light.x','done',NULL)")
        db.execute("INSERT INTO lists VALUES('list','shopping','user','u',100,100)")
        db.execute("INSERT INTO list_items VALUES('item','list','milk',0,'u',100,NULL)")
    with sqlite3.connect(state / "data" / "diagnostics.db") as db:
        db.executescript(_source_sql("src/obs/diagnostic_store.py", table="metrics"))
        db.execute("PRAGMA application_id=1264206897")
        db.execute("PRAGMA user_version=2")
        db.execute("INSERT INTO metrics VALUES('metric',100,200,'{}',0,2)")
    return state


@pytest.mark.parametrize("statement", [
    "UPDATE capture_runtime_control SET mode='muted',revision=1",
    "UPDATE ambient_runtime_control SET paused=1,revision=1",
    "UPDATE capture_browser_recorders SET state='unknown'",
    "UPDATE stt_corrections_v1 SET used=1",
    "UPDATE stt_corrections_v1 SET corrected_text='manual correction'",
    "UPDATE conversation_turns_v1 SET quality='corrected',digest='replacement'",
    "UPDATE wake_clips_v1 SET expires_at=150",
    "UPDATE utterances SET expires_at=150",
    "UPDATE utterances SET distilled=1",
    "UPDATE facts SET state='quarantined',revision=2",
    "UPDATE voice_groups SET state='manual'",
    "DELETE FROM live_delivery_sources",
    "DELETE FROM live_delivery_lineage",
    "DELETE FROM live_deliveries",
    "UPDATE live_deliveries SET nonce='new-nonce'",
    "UPDATE reminder_home_actions SET status='uncertain'",
    "UPDATE routines SET revision=2",
    "UPDATE routine_runs SET status='cancelled'",
    "UPDATE routine_run_actions SET status='uncertain'",
    "UPDATE list_items SET completed=1",
    "INSERT INTO audio_write_intents VALUES('pending.flac',1,'U',200)",
    "INSERT INTO pending_audio_deletions VALUES('pending.flac',100,0,NULL,'U')",
])
def test_protected_family_mutations_change_authority(projection_state, statement):
    before = stage.capture_state_authority(projection_state)
    with sqlite3.connect(projection_state / "data" / "authority.db") as db:
        db.execute(statement)
    assert stage.capture_state_authority(projection_state)["digest"] != before["digest"]


@pytest.mark.parametrize("statement", [
    "UPDATE facts SET content='different ordinary text'",
    "UPDATE wake_clips_v1 SET score=0.9",
    "UPDATE stt_corrections_v1 SET original_text='different raw transcript'",
    "UPDATE conversation_turns_v1 SET digest='different unmarked payload'",
    "UPDATE utterances SET text='different raw transcript'",
    "UPDATE voice_groups SET similarity=0.8",
])
def test_ordinary_payload_mutations_preserve_authority(projection_state, statement):
    before = stage.capture_state_authority(projection_state)
    with sqlite3.connect(projection_state / "data" / "authority.db") as db:
        db.execute(statement)
    assert stage.capture_state_authority(projection_state) == before


@pytest.mark.parametrize("statement,protected", [
    ("UPDATE metrics SET payload='123',bytes=3", False),
    ("UPDATE state SET value=0 WHERE key='dirty'", False),
    ("INSERT INTO staged VALUES('pending.jsonl')", True),
    ("INSERT INTO quarantines VALUES('opaque','quarantined',0)", True),
])
def test_diagnostic_obligations_are_distinct_from_numeric_cache(projection_state, statement, protected):
    before = stage.capture_state_authority(projection_state)
    with sqlite3.connect(projection_state / "data" / "diagnostics.db") as db:
        db.execute(statement)
    assert (stage.capture_state_authority(projection_state) != before) is protected


def test_environment_is_opaque_protected_configuration(projection_state):
    path = projection_state / ".env"
    path.write_text("KZA_DASHBOARD_TOKEN=synthetic-before\n")
    before = stage.capture_state_authority(projection_state)
    assert "synthetic-before" not in json.dumps(before)
    path.write_text("KZA_DASHBOARD_TOKEN=synthetic-after\n")
    assert stage.capture_state_authority(projection_state)["digest"] != before["digest"]


@pytest.mark.parametrize("statement", [
    "ALTER TABLE capture_runtime_control ADD COLUMN unclassified TEXT",
    "CREATE TABLE unknown_authority (permission TEXT)",
    "PRAGMA user_version=99",
])
def test_unknown_schema_is_rejected(projection_state, statement):
    with sqlite3.connect(projection_state / "data" / "authority.db") as db:
        db.execute(statement)
    with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
        stage.capture_state_authority(projection_state)
