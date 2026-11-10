"""Admission requires real owned closure, not a successful process exit alone."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import ast
import hashlib
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


def _probe_then_child(processes):
    """run() activates managed first, so the U3b-0a candidate probe precedes the child.

    Exactly two native launches: the probe (read-only, reaped, exit 0, no
    ticket) and the admitted child. Child assertions keep their full strength.
    """
    assert len(processes) == 2
    probe, child = processes
    assert probe.args[:5] == [sys.executable, "-I", "-B", "-c", stage._PROBE_SOURCE]
    assert probe.returncode == 0 and probe.wait(timeout=0) == 0
    assert child.args[:5] != probe.args[:5]
    return probe, child


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
    probe, child = _probe_then_child(processes)
    assert child.returncode == 0
    assert child.wait(timeout=0) == 0
    with pytest.raises(ProcessLookupError):
        os.kill(child.pid, 0)
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
    probe, child = _probe_then_child(processes)
    assert child.wait(timeout=0) == 0
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
            ("src/memory/fact_ledger.py", "FACT_OWNER_SCHEMA"),  # includes marker row 1
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
        db.execute("INSERT INTO facts VALUES('g','returned','general',1,'confirmed',100,NULL,2,150,'admin')")
        # An owned AUTOMATIC fact: impossible through the 1a API (assignment confirms); the voice
        # stage will own proposals. Admission pins it in any state, like the payload cut (R4).
        db.execute("INSERT INTO facts VALUES('h','owned','general',0.8,'automatic',100,200,1,NULL,NULL)")
        db.execute("INSERT INTO fact_sources VALUES('h','capture-h','U',100,200,'voice','[]')")
        db.execute("INSERT INTO fact_owners VALUES('h','user_a','voice_proposal',1,150,'admin','{}')")
        db.execute("INSERT INTO fact_owner_revisions VALUES('h',1,'voice_proposed',0,150)")
        db.execute("INSERT INTO fact_owner_revisions VALUES('g',2,'owner_returned',1,160)")
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
    "UPDATE fact_owners SET person_id='user_b'",
    "DELETE FROM fact_owner_revisions",
    "INSERT INTO fact_owners VALUES('g','user_b','operator_assignment',2,170,'admin','{}')",
    "UPDATE fact_owner_revisions SET last_change='operator_unassigned' WHERE fact_id='g'",
    "UPDATE facts SET content='different owned text' WHERE id='h'",
    "UPDATE fact_sources SET expires_at=150 WHERE fact_id='h'",
])
def test_protected_family_mutations_change_authority(projection_state, statement):
    before = stage.capture_state_authority(projection_state)
    with sqlite3.connect(projection_state / "data" / "authority.db") as db:
        db.execute(statement)
    assert stage.capture_state_authority(projection_state)["digest"] != before["digest"]


@pytest.mark.parametrize("statement", [
    "UPDATE facts SET content='different ordinary text' WHERE id='f'",  # 'g' confirmed, 'h' owned: protected
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
    "ALTER TABLE fact_owners ADD COLUMN unclassified TEXT",
    "UPDATE fact_owner_schema SET version=2",
    "INSERT INTO fact_owner_schema VALUES(2)",
    "DELETE FROM fact_owner_schema",
])
def test_unknown_schema_is_rejected(projection_state, statement):
    with sqlite3.connect(projection_state / "data" / "authority.db") as db:
        db.execute(statement)
    with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
        stage.capture_state_authority(projection_state)


# --- U3b-0a: candidate probe before any head/marker/binding write -------------

_CLOSING_CHILD = "\n".join([
    "import importlib.util, json, os, sqlite3, sys",
    "from contextlib import closing",
    "spec = importlib.util.spec_from_file_location('candidate_stage', os.path.abspath('scripts/stage_runtime.pyc'))",
    "module = importlib.util.module_from_spec(spec)",
    "spec.loader.exec_module(module)",
    "for database, statement, parameters in json.loads(sys.argv[1]):",
    "    with closing(sqlite3.connect(database)) as db, db:",
    "        db.execute(statement, parameters)",
    "module.write_admission_receipt(os.environ['KZA_RUNTIME_ADMISSION'], dict.fromkeys(",
    "    ['graph_closed', 'privacy_closed', 'sdk_closed', 'native_closed'], True))",
])
_TARGETS_DDL = ("CREATE TABLE erasure_targets (id TEXT PRIMARY KEY, participant TEXT NOT NULL, "
                "version INTEGER NOT NULL, descriptor TEXT NOT NULL)")


def _registration(participant, version, descriptor):
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    identity = hashlib.sha256(json.dumps([participant, version, encoded]).encode()).hexdigest()
    return [identity, participant, version, encoded]


def _closed_head(archive, tmp_path, statements=lambda state: [], *, slot="old"):
    """Establish a real closed head through an admitted child and its receipt."""
    source = tmp_path / ("offline-" + slot)
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text("schema: admission-test\n")
    root = tmp_path / "staging"
    stage.prepare_slot(archive, root, slot, source)
    root = root.resolve()
    state = root / "slots" / slot / "state"
    stage.run_slot(root, slot, _command(_CLOSING_CHILD, json.dumps(statements(state))), timeout=30)
    assert stage.read_admission_head(root)["status"] == "closed"
    return root, state


def _witness(root, *states):
    """Everything a refused activation/delegation must leave byte-identical.

    root/.admission.lock and root/.stage.lock are excluded: every controller
    entry rewrites the lease header and _locked_root creates the stage lock.
    """
    tree = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in {".admission.lock", ".stage.lock"} or relative.startswith("slots/"):
            continue
        if path.is_symlink():
            tree[relative] = ("link", os.readlink(path))
        elif path.is_dir():
            tree[relative] = ("dir",)
        else:
            tree[relative] = ("file", path.read_bytes())
    files = {}
    for state in states:
        for name in (stage._MARKER, ".kza-target-ownership.json"):
            path = state / name
            files[str(path)] = path.read_bytes() if path.exists() else None
    return {"root": tree, "states": files}


def _copy_candidate(archive, root, state, name="candidate"):
    stage.prepare_slot(archive, root, name, state)
    return root / "slots" / name / "state"


def _refused(root, states, action, match):
    before = _witness(root, *states)
    with pytest.raises(ValueError, match=match):
        action()
    assert _witness(root, *states) == before
    return before


def test_candidate_whose_own_code_rejects_state_is_refused_before_head_moves(
        admission_archive, tmp_path, monkeypatch):
    # W3: the controller knows a table the candidate's own companion rejects.
    monkeypatch.setitem(stage._TABLE_COLUMNS, "u3b_probe_extra", "id:TEXT")
    root, state = _closed_head(admission_archive, tmp_path, lambda state: [
        ["data/extra.db", "CREATE TABLE u3b_probe_extra (id TEXT)", []]])
    candidate = _copy_candidate(admission_archive, root, state)
    head = stage.read_admission_head(root)
    with stage.AdmissionController(root) as controller:
        _refused(root, (state, candidate), lambda: controller.activate("candidate"),
                 "runtime_admission_candidate_rejected: runtime_admission_unknown_schema")
    assert stage.read_admission_head(root) == head
    assert os.readlink(root / "current") == "slots/old/runtime"


def test_equal_authority_but_different_binding_is_refused(admission_archive, tmp_path, monkeypatch):
    # 0a-F2: the child compares the whole binding, so the probe must too.
    root, state = _closed_head(admission_archive, tmp_path)
    candidate = _copy_candidate(admission_archive, root, state)
    original = stage._binding_snapshot

    def controller_view(root, name, **kwargs):
        value = original(root, name, **kwargs)
        return dict(value, sdk_versions={"synthetic": "controller-only"}) if name == "candidate" else value

    monkeypatch.setattr(stage, "_binding_snapshot", controller_view)
    with stage.AdmissionController(root) as controller:
        _refused(root, (state, candidate), lambda: controller.activate("candidate"),
                 "runtime_admission_candidate_rejected: binding_mismatch")


def test_authority_digest_disagreement_is_refused(admission_archive, tmp_path, monkeypatch):
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    with closing(sqlite3.connect(source / "data" / "lineage.db")) as db, db:
        db.execute("CREATE TABLE lineage (record_id TEXT, conversation_id TEXT)")
        db.execute("INSERT INTO lineage VALUES('record','U')")
    root = tmp_path / "staging"
    stage.prepare_slot(admission_archive, root, "candidate", source)
    root = root.resolve()
    monkeypatch.setattr(stage, "_PROTECTED_TABLES", stage._PROTECTED_TABLES | {"lineage"})
    with stage.AdmissionController(root) as controller:
        _refused(root, (root / "slots/candidate/state",),
                 lambda: controller.activate("candidate", managed=True),
                 "runtime_admission_candidate_rejected: binding_mismatch")
    assert stage.read_admission_head(root) is None and not (root / "admission").exists()


def test_probe_timeout_is_refused_and_reaped(admission_archive, tmp_path, monkeypatch):
    root, state = _closed_head(admission_archive, tmp_path)
    candidate = _copy_candidate(admission_archive, root, state)
    processes = _observe_real_processes(monkeypatch)
    monkeypatch.setattr(stage, "_PROBE_SOURCE", "import time\ntime.sleep(60)")
    started = time.monotonic()
    with stage.AdmissionController(root, probe_timeout=1) as controller:
        _refused(root, (state, candidate), lambda: controller.activate("candidate"),
                 "runtime_admission_candidate_rejected: probe_timeout")
    assert time.monotonic() - started < 30
    assert len(processes) == 1 and processes[0].poll() is not None


@pytest.mark.parametrize("source,match", [
    ("print('not json')", "candidate_rejected: probe_output exit=0"),
    ("raise SystemExit(0)", "candidate_rejected: probe_output exit=0"),
    ("import sys\nsys.stdout.write('[]')", "candidate_rejected: probe_output exit=0"),
    ("import sys\nsys.stdout.write('{\"probe\": \"error\", \"code\": \"x\"}')",
     "candidate_rejected: probe_output exit=0"),
    ("raise RuntimeError('crash')", "candidate_rejected: probe_output exit=1"),
    ("import sys\nsys.stdout.write('{\"probe\": \"unmanaged\"}')\nraise SystemExit(3)",
     "runtime_admission_candidate_unmanaged"),
    # R3: a correct binding never excuses a failed exit or an unexpected shape.
    (stage._PROBE_SOURCE + "\nraise SystemExit(1)", "candidate_rejected: probe_output exit=1$"),
    (stage._PROBE_SOURCE + "\nraise SystemExit(2)", "candidate_rejected: probe_output exit=2$"),
    (stage._PROBE_SOURCE + "\nsys.stdout.flush()\nimport os, signal\nos.kill(os.getpid(), signal.SIGKILL)",
     "candidate_rejected: probe_output exit=-9$"),
    (stage._PROBE_SOURCE.replace("{'probe': 'binding', 'binding': binding}",
                                 "{'probe': 'binding', 'binding': binding, 'extra': 1}"),
     "candidate_rejected: probe_output exit=0$"),
])
def test_malformed_probe_output_is_refused(admission_archive, tmp_path, monkeypatch, source, match):
    root, state = _closed_head(admission_archive, tmp_path)
    candidate = _copy_candidate(admission_archive, root, state)
    monkeypatch.setattr(stage, "_PROBE_SOURCE", source)
    with stage.AdmissionController(root) as controller:
        _refused(root, (state, candidate), lambda: controller.activate("candidate"), match)


@pytest.fixture(scope="module")
def legacy_archive(tmp_path_factory):
    """A real pre-80b47f5 artifact: ships scripts/stage_runtime, lacks runtime_admission."""
    repo = Path(__file__).resolve().parents[3]
    if subprocess.run(["git", "cat-file", "-e", "b0866e8^{commit}"], cwd=repo).returncode != 0:
        pytest.skip("b0866e8 is not available in this clone")
    base = tmp_path_factory.mktemp("legacy-artifact")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(clone)],
                   check=True, timeout=120)
    destination = base / "runtime.tar.gz"
    manifest = build_runtime(clone, destination, "b0866e8")
    assert manifest["git_commit"].startswith("b0866e8")
    assert "scripts/stage_runtime.pyc" in manifest["files"]
    assert "src/core/runtime_admission.pyc" not in manifest["files"]
    return destination


def test_legacy_candidate_without_admission_companion_is_unmanaged(
        admission_archive, legacy_archive, tmp_path, monkeypatch):
    root, state = _closed_head(admission_archive, tmp_path)
    legacy = _copy_candidate(legacy_archive, root, state, "legacy")
    processes = _observe_real_processes(monkeypatch)
    with stage.AdmissionController(root) as controller:
        _refused(root, (state, legacy), lambda: controller.activate("legacy"),
                 "runtime_admission_candidate_unmanaged")
    assert processes == []  # Decided from the verified manifest before spawning anything.


def test_python_magic_mismatch_is_refused_before_the_probe(admission_archive, tmp_path, monkeypatch):
    # 0a-F5: _validate_manifest refuses first; no separate python check exists.
    root, state = _closed_head(admission_archive, tmp_path)
    candidate = _copy_candidate(admission_archive, root, state)
    release = root / "slots/candidate/runtime/release.json"
    release.chmod(0o600)
    manifest = json.loads(release.read_text())
    release.write_text(json.dumps(dict(manifest, python_magic="00000000")))
    processes = _observe_real_processes(monkeypatch)
    with stage.AdmissionController(root) as controller:
        _refused(root, (state, candidate), lambda: controller.activate("candidate"),
                 "Runtime manifest or file integrity mismatch")
    assert processes == []


def test_same_version_candidate_activates_after_one_probe(admission_archive, tmp_path, monkeypatch):
    root, state = _closed_head(admission_archive, tmp_path)
    candidate = _copy_candidate(admission_archive, root, state)
    head = stage.read_admission_head(root)
    launches = []
    original = subprocess.Popen

    def launch(*args, **kwargs):
        launches.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(stage.subprocess, "Popen", launch)
    monkeypatch.setenv("KZA_RUNTIME_ADMISSION", "{\"inherited\": true}")
    with stage.AdmissionController(root) as controller:
        assert controller.activate("candidate") == "slots/old/runtime"
    assert len(launches) == 1
    assert "pass_fds" not in launches[0] and launches[0].get("close_fds", True) is True
    assert "KZA_RUNTIME_ADMISSION" not in launches[0]["env"]
    assert set(launches[0]["env"]) <= set(stage._PROBE_ENVIRONMENT)
    assert os.readlink(root / "current") == "slots/candidate/runtime"
    closed = stage.read_admission_head(root)
    assert closed["status"] == "closed" and closed["generation"] == head["generation"] + 1
    marker = json.loads((candidate / stage._MARKER).read_text())
    assert marker["binding_id"] == closed["binding_id"]


# --- U3b-0a: one-hop 'delegate' CLI ------------------------------------------------

def _delegate_cli(root, name):
    repo = Path(__file__).resolve().parents[3]
    return subprocess.run([sys.executable, "-B", str(repo / "scripts" / "stage_runtime.py"),
                           "--root", str(root), "--offline", "delegate", name],
                          cwd=repo, capture_output=True, text=True, timeout=60)


def _registered_targets(state):
    (state / "data" / "clips").mkdir()
    local = _registration("collectors", 1, {"kind": "wake_clips", "root": str(state / "data" / "clips")})
    pathless = _registration("producers", 1, {"kind": "runtime_producers", "protocol": 1})
    return [["data/privacy.db", _TARGETS_DDL, []],
            ["data/privacy.db", "INSERT INTO erasure_targets VALUES(?,?,?,?)", local],
            ["data/privacy.db", "INSERT INTO erasure_targets VALUES(?,?,?,?)", pathless]]


@pytest.fixture
def delegated_pair(admission_archive, tmp_path):
    registered = []

    def statements(state):
        rows = _registered_targets(state)
        registered.extend(row[2][0] for row in rows[1:])
        return rows

    root, state = _closed_head(admission_archive, tmp_path, statements)
    candidate = _copy_candidate(admission_archive, root, state)
    return {"root": root, "state": state, "candidate": candidate,
            "local": registered[0], "pathless": registered[1]}


def test_activate_without_delegate_is_foreign_and_changes_nothing(delegated_pair):
    root, state, candidate = delegated_pair["root"], delegated_pair["state"], delegated_pair["candidate"]
    with stage.AdmissionController(root) as controller:
        _refused(root, (state, candidate), lambda: controller.activate("candidate"),
                 "runtime_admission_foreign_target")


def test_delegate_cli_grants_only_local_path_targets_then_activate(delegated_pair):
    root, candidate = delegated_pair["root"], delegated_pair["candidate"]
    head = stage.read_admission_head(root)
    result = _delegate_cli(root, "candidate")
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["slot"] == "candidate" and output["source_binding"] == head["binding_id"]
    assert output["targets"] == [delegated_pair["local"]]  # Path-less targets stay owned.
    record = root / "admission" / "delegations" / (output["capability"] + ".json")
    assert set(json.loads(record.read_text())["targets"]) == {delegated_pair["local"]}
    policy = json.loads((candidate / ".kza-target-ownership.json").read_text())
    assert policy["delegated_targets"] == [delegated_pair["local"]]
    assert policy["owned_targets"] == [delegated_pair["pathless"]]
    assert stage.activate_slot(root, "candidate") == "slots/old/runtime"
    assert os.readlink(root / "current") == "slots/candidate/runtime"


def test_delegate_cli_refuses_a_second_hop(delegated_pair, admission_archive):
    root, candidate = delegated_pair["root"], delegated_pair["candidate"]
    assert _delegate_cli(root, "candidate").returncode == 0
    stage.run_slot(root, "candidate", _command(_CLOSING_CHILD, "[]"), timeout=30)
    assert stage.read_admission_head(root)["status"] == "closed"
    third = _copy_candidate(admission_archive, root, candidate, "third")
    before = _witness(root, delegated_pair["state"], candidate, third)
    result = _delegate_cli(root, "third")
    assert result.returncode != 0
    assert "runtime_admission_delegation_one_hop_only" in result.stderr and "U3b-0b" in result.stderr
    assert _witness(root, delegated_pair["state"], candidate, third) == before


def test_delegate_never_grants_a_target_recorded_outside_the_source(delegated_pair, tmp_path):
    root, state, candidate = delegated_pair["root"], delegated_pair["state"], delegated_pair["candidate"]
    outside = tmp_path / "outside-clips"
    outside.mkdir()
    foreign = _registration("collectors", 1, {"kind": "conversations", "root": str(outside)})
    with closing(sqlite3.connect(state / "data" / "privacy.db")) as db, db:
        db.execute("INSERT INTO erasure_targets VALUES(?,?,?,?)", foreign)
    # The full source verification refuses before the CLI derives any target id.
    with stage.AdmissionController(root) as controller:
        _refused(root, (state, candidate), lambda: controller.delegate("candidate"),
                 "^runtime_admission_foreign_target$")
    before = _witness(root, state, candidate)
    result = _delegate_cli(root, "candidate")
    assert result.returncode != 0 and "ValueError: runtime_admission_foreign_target" in result.stderr
    assert _witness(root, state, candidate) == before


_SENTINEL = "u3b0a-sentinel-secret-7f3c"


def test_probe_environment_is_an_allow_list_without_secrets(admission_archive, tmp_path, monkeypatch):
    """R1: the candidate's code never sees controller secrets, even when it is refused."""
    root, state = _closed_head(admission_archive, tmp_path)
    candidate = _copy_candidate(admission_archive, root, state)
    for name in ("HOME_ASSISTANT_TOKEN", "SPOTIFY_CLIENT_SECRET", "KZA_DASHBOARD_TOKEN", "KZA_PROBE_SENTINEL"):
        monkeypatch.setenv(name, _SENTINEL)
    monkeypatch.setenv("KZA_RUNTIME_ADMISSION", _SENTINEL)
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    observed = tmp_path / "probe-environment.json"
    monkeypatch.setattr(stage, "_PROBE_SOURCE", "import json, os\n"
                        "open(" + repr(str(observed)) + ", 'w').write(json.dumps(dict(os.environ)))")
    with stage.AdmissionController(root) as controller:
        _refused(root, (state, candidate), lambda: controller.activate("candidate"),
                 "candidate_rejected: probe_output exit=0")
    seen = json.loads(observed.read_text())
    assert _SENTINEL not in json.dumps(seen)
    assert seen["LANG"] == "en_US.UTF-8"
    # macOS injects __CF_USER_TEXT_ENCODING into every process; nothing else is added.
    assert set(seen) - {"__CF_USER_TEXT_ENCODING"} <= set(stage._PROBE_ENVIRONMENT)


@pytest.mark.parametrize("value", [0, 1e12, 1e308, float("nan"), -1, -1e-9, float("inf"),
                                   0.5, 3600.5, True, "60", None])
def test_probe_timeout_outside_bounds_is_a_coded_refusal(tmp_path, value):
    """R2: 1-3600 s; nothing outside reaches subprocess (1e12/1e308 used to raise OverflowError)."""
    with pytest.raises(ValueError, match="^runtime_admission_probe_timeout$"):
        stage.AdmissionController(tmp_path / "root", probe_timeout=value)
    with pytest.raises(ValueError, match="^runtime_admission_probe_timeout$"):
        stage.activate_slot(tmp_path / "root", "candidate", probe_timeout=value)
    assert not (tmp_path / "root").exists()


@pytest.mark.parametrize("value", [1, 1.0, 60, 3600])
def test_probe_timeout_bounds_are_inclusive(tmp_path, value):
    assert stage.AdmissionController(tmp_path / "root", probe_timeout=value).probe_timeout == float(value)


@pytest.mark.parametrize("text", ["0", "1e12", "1e308", "nan", "-5", "inf", "0.5", "3601", "soon"])
def test_cli_probe_timeout_outside_bounds_is_refused(tmp_path, text):
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, "-B", str(repo / "scripts" / "stage_runtime.py"),
                             "--root", str(tmp_path / "root"), "--offline", "activate", "candidate",
                             "--probe-timeout", text], cwd=repo, capture_output=True, text=True, timeout=60)
    assert result.returncode == 2
    assert "runtime_admission_probe_timeout" in result.stderr and "Traceback" not in result.stderr
    assert not (tmp_path / "root").exists()


def test_delegation_api_refuses_a_second_hop_even_for_pathless_targets(delegated_pair, admission_archive):
    """R5: one hop is enforced by authorize_target_delegation itself, not only by the CLI."""
    root, candidate = delegated_pair["root"], delegated_pair["candidate"]
    head = stage.read_admission_head(root)
    with stage.AdmissionController(root) as controller:
        # The public API (unlike the CLI derivation) may also delegate a path-less id.
        controller.authorize_target_delegation(candidate, head["binding_id"],
                                               [delegated_pair["local"], delegated_pair["pathless"]])
        assert controller.activate("candidate") == "slots/old/runtime"
    stage.run_slot(root, "candidate", _command(_CLOSING_CHILD, "[]"), timeout=30)
    closed = stage.read_admission_head(root)
    assert closed["status"] == "closed"
    third = _copy_candidate(admission_archive, root, candidate, "third")
    with stage.AdmissionController(root) as controller:
        _refused(root, (delegated_pair["state"], candidate, third),
                 lambda: controller.authorize_target_delegation(third, closed["binding_id"],
                                                                [delegated_pair["pathless"]]),
                 "^runtime_admission_delegation_one_hop_only$")
