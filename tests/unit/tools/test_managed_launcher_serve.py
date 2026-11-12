"""U3b-0d managed launcher: serve, recovery and installers on a compiled slot, in process.

The slot is compiled from this working tree (clone overlay); the controller
is this process (scripts.managed_launcher.Serve). Stops are real SIGTERMs to
the test process, which the controller's own handler must turn into a byte.
The child is tests/mocks/launcher_child.py running the slot's compiled
process_lifecycle and receipt writer.
"""

import json
import os
import shutil
from pathlib import Path
import signal
import sqlite3
from contextlib import closing
import time

import pytest

from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from tests.mocks import fake_systemd
from tests.mocks import launcher_rehearsal as rehearsal

REPO = Path(__file__).resolve().parents[3]
TEMPLATE = (REPO / "systemd" / "kza-voice.service.d" / "managed.conf.template").read_text()
RESOURCES = dict.fromkeys(["graph_closed", "privacy_closed", "sdk_closed", "native_closed"], True)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    return rehearsal.build_artifacts(tmp_path_factory.mktemp("launcher-artifacts"))


@pytest.fixture
def root(artifacts, tmp_path):
    root, _state = rehearsal.managed_root(artifacts["archives"]["x"], tmp_path)
    return root


def _ready(phases):
    return any(name == "ready" for name, _ in phases)


def _serve(root, tmp_path, scenario, *, stop_when=_ready, seconds=0.0, touch=None, name="phases.txt",
           execute=None, **kwargs):
    phase_file = tmp_path / name
    script = None
    if execute is not None:
        script = tmp_path / (name + ".py")
        script.write_text(execute)
    command = rehearsal.child_command(phase_file, scenario, seconds=seconds, touch=touch, execute=script)
    return rehearsal.run_serve(root, command, phase_file=phase_file, stop_when=stop_when, **kwargs)


def _head(root):
    return stage.read_admission_head(root)


def _admission_files(root):
    directory = root / "admission"
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _exit_record(root, result):
    run_id = next(iter(sorted((root / "admission" / "runs").glob("*.exit.json")), None))
    return json.loads(run_id.read_text()) if run_id else None


def _last_exit(root):
    records = [json.loads(path.read_text()) for path in (root / "admission" / "runs").glob("*.exit.json")]
    return max(records, key=lambda item: item["ended_at"]) if records else None


# --- the clean lifecycle -------------------------------------------------------------------------

def test_clean_stop_seals_and_the_child_received_the_forwarded_sigterm(root, tmp_path):
    result = _serve(root, tmp_path, "clean")
    assert result.code == 0, result.serve.events
    head = _head(root)
    assert head["status"] == "closed" and head["run_id"] is None
    phases = dict(result.phases)
    assert phases["mask"] == "empty"  # v3 N1: nothing blocked across exec
    assert phases["stop_received"] == "ready"  # the child itself received SIGTERM (T-C1)
    assert phases["env"] == "clean"  # descriptors popped by the child itself
    assert phases["pythonpath"] == str(root / "slots" / "s" / "runtime")
    assert phases["config_path"] == "config/settings.yaml"
    sent = result.serve.notifier.sent
    assert sent.count("READY=1") == 1 and sent.index("READY=1") < sent.index("STOPPING=1")
    assert "WATCHDOG=1" not in sent  # no WATCHDOG_USEC from the manager: no pings
    receipts = rehearsal.receipts(root)
    assert len(receipts) == 1 and receipts[0]["child_pid"] == int(phases["pid"])
    run = launcher.read_run_record(root, receipts[0]["run_id"])
    assert run["child_pid"] == int(phases["pid"]) and run["child_start_time"]
    exit_record = launcher.read_exit_record(root, receipts[0]["run_id"])
    assert exit_record["reason"] == "stop" and exit_record["sealed"] and exit_record["stop_requested"]
    assert stage._closure_proof(root, head["binding_id"])["child_pid"] == int(phases["pid"])


def test_child_environment_has_no_manager_variables(root, tmp_path):
    environ = dict(os.environ, NOTIFY_SOCKET="/nonexistent/notify", WATCHDOG_USEC="4000000",
                   WATCHDOG_PID=str(os.getpid()), PYTHONPATH="/inherited", CONFIG_PATH="/inherited.yaml")
    result = _serve(root, tmp_path, "clean", environ=environ)
    assert result.code == 0
    phases = dict(result.phases)
    # The child popped its own descriptors; it never saw NOTIFY_SOCKET or WATCHDOG_*.
    assert phases["env"] == "clean"
    assert phases["pythonpath"] == str(root / "slots" / "s" / "runtime")


def test_watchdog_is_relayed_only_after_ready(root, tmp_path):
    environ = {key: value for key, value in os.environ.items() if key != "NOTIFY_SOCKET"}
    environ.update(WATCHDOG_USEC="200000", WATCHDOG_PID=str(os.getpid()))

    def after_pings(phases):
        return _ready(phases) and time.sleep(0.5) is None

    result = _serve(root, tmp_path, "clean", environ=environ, stop_when=after_pings)
    sent = result.serve.notifier.sent
    assert result.code == 0 and sent.count("WATCHDOG=1") >= 3
    assert sent.index("READY=1") < sent.index("WATCHDOG=1")


def test_watchdog_silent_child_sends_no_pings(root, tmp_path):
    environ = {key: value for key, value in os.environ.items() if key != "NOTIFY_SOCKET"}
    environ.update(WATCHDOG_USEC="200000")
    result = _serve(root, tmp_path, "watchdog_silent", environ=environ,
                    stop_when=lambda phases: _ready(phases) and time.sleep(0.4) is None)
    assert result.code == 0 and "WATCHDOG=1" not in result.serve.notifier.sent


def test_child_ending_by_itself_with_a_receipt_exits_75(root, tmp_path):
    result = _serve(root, tmp_path, "self_end", stop_when=None, seconds=0.2)
    assert result.code == 75 and _head(root)["status"] == "closed"


# --- stop windows (v2 controller_loop_M1.windows, v3 N1) ------------------------------------------

def _kill_self():
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(0.05)


@pytest.mark.parametrize("window", ["lease", "activate", "preflight"])
def test_stop_before_cas_running_exits_zero_and_starts_nothing(root, tmp_path, monkeypatch, window):
    before = _head(root)
    if window == "lease":
        original = launcher.check_controller_epoch

        def hook(*args):
            _kill_self()
            return original(*args)
        monkeypatch.setattr(launcher, "check_controller_epoch", hook)
    elif window == "activate":
        original = stage.AdmissionController.activate

        def hook(self, *args, **kwargs):
            _kill_self()
            return original(self, *args, **kwargs)
        monkeypatch.setattr(stage.AdmissionController, "activate", hook)
    else:
        original = launcher.path_preflight

        def hook(*args, **kwargs):
            _kill_self()
            return original(*args, **kwargs)
        monkeypatch.setattr(launcher, "path_preflight", hook)
    result = _serve(root, tmp_path, "clean", stop_when=None)
    assert result.code == 0 and "stop before CAS running; head unchanged" in result.serve.events
    head = _head(root)
    assert head["status"] == "closed" and head["binding_id"] == before["binding_id"]
    assert not (root / "admission" / "runs").exists() and not rehearsal.receipts(root)


def test_stop_between_cas_and_popen_is_forwarded_after_the_run_record(root, tmp_path, monkeypatch):
    original = launcher.Serve._spawn

    def spawn(self, *args):
        _kill_self()
        return original(self, *args)
    monkeypatch.setattr(launcher.Serve, "_spawn", spawn)
    result = _serve(root, tmp_path, "clean", stop_when=None)
    assert result.code == 0, result.serve.events
    assert _head(root)["status"] == "closed"
    # Deferred until the child armed its handler (ARMED=1), then a clean close before READY.
    assert dict(result.phases).get("stop_received") == "entry"
    assert "ready_skipped" in dict(result.phases) and "READY=1" not in result.serve.notifier.sent
    assert result.serve.events.index("child started") < result.serve.events.index("stop forwarded")


def test_stop_during_the_seal_exits_after_the_seal(root, tmp_path, monkeypatch):
    original = stage.seal_admission_run

    def seal(*args, **kwargs):
        _kill_self()
        return original(*args, **kwargs)
    monkeypatch.setattr(stage, "seal_admission_run", seal)
    result = _serve(root, tmp_path, "self_end", stop_when=None, seconds=0.1)
    # The child ended by itself (75), but the stop byte read after the seal makes it a clean stop.
    assert result.code == 0 and _head(root)["status"] == "closed"


# --- failures, exit codes and recovery -----------------------------------------------------------

def test_crash_gives_unknown_68_then_bounded_adoption_then_a_clean_close(root, tmp_path):
    crashed = _serve(root, tmp_path, "crash", stop_when=None, seconds=0.2, name="a.txt")
    assert crashed.code == 68 and _head(root)["status"] == "unknown"
    record = _last_exit(root)
    assert record["reason"] == "child_exit" and record["returncode"] == 1 and record["sealed"] is False
    served = _serve(root, tmp_path, "clean", name="b.txt")
    assert served.code == 0, served.serve.events
    assert "recovery R-b closed" in served.serve.events
    recoveries = launcher._recoveries(root)
    assert len(recoveries) == 1 and recoveries[0]["mode"] == "automatic" and recoveries[0]["class"] == "crash"
    assert recoveries[0]["checks"]["regressed"] == [] and recoveries[0]["head_before"]["status"] == "unknown"
    assert _head(root)["status"] == "closed"


def test_a_recovered_head_cannot_delegate_until_a_clean_close(root, tmp_path, artifacts):
    assert _serve(root, tmp_path, "crash", stop_when=None, seconds=0.1, name="a.txt").code == 68
    with stage.AdmissionController(root, allow_open_head=True):
        head = _head(root)
        decision = launcher.recover_open_head(root, head, mode="bounded", boot_id="b", wall=time.time())
    assert decision.closed_head is not None
    record = stage.read_manual_head(root, decision.closed_head)
    assert record["kind"] == "recovery"
    stage.prepare_slot(artifacts["archives"]["y"], root, "next", root / "slots" / "s" / "state")
    with pytest.raises(FileNotFoundError, match="admission/closures"):  # no closure proof for it
        stage.delegate_slot(root, "next")
    assert _serve(root, tmp_path, "clean", name="b.txt").code == 0  # a clean close...
    with stage.AdmissionController(root):  # ...gives the head a closure proof again
        assert stage.verify_closed_binding(root, _head(root)["binding_id"])


def test_second_crash_within_24h_and_second_in_a_row_are_69(root, tmp_path):
    assert _serve(root, tmp_path, "crash", stop_when=None, seconds=0.1, name="a.txt").code == 68
    again = _serve(root, tmp_path, "crash", stop_when=None, seconds=0.1, name="b.txt")
    assert "recovery R-b closed" in again.serve.events and again.code == 69  # budget spent within 24 h
    third = _serve(root, tmp_path, "clean", name="c.txt")
    assert third.code == 69 and "recovery R-b runtime_admission_recovery_crash_budget_spent" in third.serve.events
    later = time.time() + 25 * 3600
    # A day later the 24 h budget is free, but the crashed run started from a recovered head.
    fourth = _serve(root, tmp_path, "clean", name="d.txt", wall=lambda: later)
    assert fourth.code == 69
    assert "recovery R-b runtime_admission_recovery_second_in_a_row" in fourth.serve.events


def test_exit_zero_without_receipt_is_unknown(root, tmp_path):
    result = _serve(root, tmp_path, "exit_no_receipt", stop_when=None, seconds=0.1)
    assert result.code == 68 and _head(root)["status"] == "unknown"
    assert _last_exit(root)["seal_error"] == "runtime_admission_close_receipt_missing"


def test_stop_grace_exceeded_kills_and_gives_unknown(root, tmp_path):
    result = _serve(root, tmp_path, "hang", stop_grace=1.0)
    assert result.code == 68 and _head(root)["status"] == "unknown"
    record = _last_exit(root)
    assert record["reason"] == "stop_grace_exceeded" and record["returncode"] == -signal.SIGKILL


def test_receipt_then_hung_teardown_is_killed_and_still_sealed(root, tmp_path):
    result = _serve(root, tmp_path, "receipt_then_hang", stop_grace=2.0)
    assert result.code == 0 and _head(root)["status"] == "closed"
    assert _last_exit(root)["reason"] == "stop_grace_exceeded"


def test_missing_receipt_state_record_is_refused_for_a_declaring_slot(root, tmp_path, monkeypatch):
    """Review 0d3 K2: deleting the sidecar does not downgrade the slot to 'older'."""
    original = stage.seal_admission_run

    def seal(root_path, running, child_pid, previous):
        stage._receipt_state_path(root_path, running["run_id"]).unlink()
        return original(root_path, running, child_pid, previous)
    monkeypatch.setattr(stage, "seal_admission_run", seal)
    result = _serve(root, tmp_path, "clean")
    assert result.code == 68 and _head(root)["status"] == "unknown"
    assert _last_exit(root)["seal_error"] == "runtime_admission_receipt_state"


def test_state_changed_after_the_receipt_bars_automatic_recovery_and_needs_the_operator(root, tmp_path):
    state = root / "slots" / "s" / "state"
    result = _serve(root, tmp_path, "restore_after_receipt", touch=state / "data" / "notes.json")
    assert result.code == 69 and _head(root)["status"] == "unknown"
    record = _last_exit(root)
    assert record["state_changed_after_close"] is True
    assert record["seal_error"] == "runtime_admission_state_manifest_mismatch"
    refused = _serve(root, tmp_path, "clean", name="b.txt")
    assert refused.code == 69
    assert "recovery R-b runtime_admission_recovery_state_changed_after_close" in refused.serve.events
    report = launcher.adopt(root, confirm="", regressions=[], dry_run=True)
    with pytest.raises(launcher.LauncherRefusal, match="runtime_admission_adopt_confirmation"):
        launcher.adopt(root, confirm="0" * 64, regressions=["state_changed_after_close"])
    with pytest.raises(launcher.LauncherRefusal, match="runtime_admission_recovery_confirmation"):
        launcher.adopt(root, confirm=report["head_sha256"], regressions=[])
    adopted = launcher.adopt(root, confirm=report["head_sha256"], regressions=["state_changed_after_close"])
    assert adopted["path"] == "R-b" and _head(root)["status"] == "closed"
    recovery = launcher._recoveries(root)[-1]
    assert recovery["mode"] == "operator" and recovery["confirmation"]["regressions"] == ["state_changed_after_close"]


def test_recover_off_and_operator_modes_never_adopt_automatically(root, tmp_path):
    assert _serve(root, tmp_path, "crash", stop_when=None, seconds=0.1, name="a.txt", recover="off").code == 69
    off = _serve(root, tmp_path, "clean", name="b.txt", recover="off")
    assert off.code == 69 and "recovery R-b runtime_admission_recovery_off" in off.serve.events
    operator = _serve(root, tmp_path, "clean", name="c.txt", recover="operator")
    assert operator.code == 69 and "recovery R-b runtime_admission_recovery_operator" in operator.serve.events


def test_controller_killed_after_the_receipt_is_resumed_by_r_a(root, tmp_path, monkeypatch):
    class Killed(BaseException):
        pass

    def die(*args, **kwargs):
        raise Killed
    monkeypatch.setattr(stage, "seal_admission_run", die)
    with pytest.raises(Killed):
        _serve(root, tmp_path, "clean", name="a.txt")
    monkeypatch.undo()
    head = _head(root)
    assert head["status"] == "running"  # a SIGKILLed controller leaves 'running'
    result = _serve(root, tmp_path, "clean", name="b.txt")
    assert result.code == 0 and "recovery R-a closed" in result.serve.events
    assert launcher._recoveries(root) == []  # R-a consumes no budget


APPEND = "\n".join([
    "from contextlib import closing",
    "import sqlite3",
    "from src.memory.deletion_ledger import DeletionKind, append_deletion, initialize_deletion_ledger",
    "with closing(sqlite3.connect(state / 'data' / 'privacy.db', isolation_level=None)) as db:",
    "    initialize_deletion_ledger(db)",
    "    db.execute('BEGIN IMMEDIATE')",
    "    append_deletion(db, DeletionKind.CONVERSATION_ERASED, SUBJECT, 2_000_000_000.0)",
    "    db.execute('COMMIT')",
])


def _append(subject):
    return "SUBJECT = %r\n" % subject + APPEND


def _utterance_store(state, conversation):
    """An out-of-band events store holding one utterance of ``conversation`` (exact contract)."""
    with closing(sqlite3.connect(state / "data" / "events.db")) as db, db:
        db.execute(rehearsal.table_ddl("utterances"))
        db.execute("INSERT INTO utterances (id, conversation_id) VALUES (1, ?)", (conversation,))


def test_restored_older_database_after_a_crash_is_refused_by_r_b(root, tmp_path):
    """v3 N2 restore-after-crash: forward evolution refuses automatic R-b."""
    state = root / "slots" / "s" / "state"
    database = state / "data" / "privacy.db"
    assert _serve(root, tmp_path, "clean", name="a.txt", execute=_append("c-1")).code == 0
    older = tmp_path / "older.db"
    shutil.copy2(database, older)  # a byte copy of the closed database (no SQLite open)
    assert _serve(root, tmp_path, "clean", name="b.txt", execute=_append("c-2")).code == 0
    assert _serve(root, tmp_path, "crash", stop_when=None, seconds=0.1, name="c.txt").code == 68
    os.replace(older, database)  # an older privacy.db restored after the crash
    refused = _serve(root, tmp_path, "clean", name="d.txt")
    assert refused.code == 69
    assert "recovery R-b runtime_admission_recovery_regression" in refused.serve.events
    report = launcher.adopt(root, confirm="", regressions=[], dry_run=True)["report"]
    assert report["regressions"] == ["deletion_ledger_v1:data/privacy.db"]


def test_resurrected_payload_after_a_crash_is_refused(root, tmp_path):
    state = root / "slots" / "s" / "state"
    assert _serve(root, tmp_path, "clean", name="a.txt", execute=_append("c-erased")).code == 0
    assert _serve(root, tmp_path, "crash", stop_when=None, seconds=0.1, name="b.txt").code == 68
    _utterance_store(state, "c-erased")  # a restored events store after the crash
    refused = _serve(root, tmp_path, "clean", name="c.txt")
    assert refused.code == 69 and "recovery R-b runtime_admission_recovery_resurrection" in refused.serve.events
    report = launcher.adopt(root, confirm="", regressions=[], dry_run=True)
    assert report["report"]["resurrected"]["conversation"] == 1 and report["report"]["regressions"] == []
    adopted = launcher.adopt(root, confirm=report["head_sha256"], regressions=["resurrected_conversation"])
    assert adopted["checks"]["resurrected"]["conversation"] == 1


def test_an_unrelated_payload_after_a_crash_is_adopted(root, tmp_path):
    state = root / "slots" / "s" / "state"
    assert _serve(root, tmp_path, "clean", name="a.txt", execute=_append("c-erased")).code == 0
    assert _serve(root, tmp_path, "crash", stop_when=None, seconds=0.1, name="b.txt").code == 68
    _utterance_store(state, "c-live")
    served = _serve(root, tmp_path, "clean", name="c.txt")
    assert served.code == 0 and "recovery R-b closed" in served.serve.events


TOMBSTONE = "\n".join([
    "from contextlib import closing",
    "import sqlite3",
    "with closing(sqlite3.connect(state / 'data' / 'privacy.db')) as db, db:",
    "    db.execute('CREATE TABLE conversation_tombstones (conversation_id TEXT, deleted_at REAL, deleted_by TEXT)')",
    "    db.execute(\"INSERT INTO conversation_tombstones VALUES ('c', 1.0, 'admin')\")",
])


def test_a_protected_state_without_a_deletion_ledger_needs_the_operator(root, tmp_path):
    """v2 R-b: a pre-0d3 state (protected tables, no ledger) allows only operator adoption."""
    assert _serve(root, tmp_path, "clean", name="a.txt", execute=TOMBSTONE).code == 0
    assert _serve(root, tmp_path, "crash", stop_when=None, seconds=0.1, name="b.txt").code == 68
    refused = _serve(root, tmp_path, "clean", name="c.txt")
    assert refused.code == 69 and "recovery R-b runtime_admission_recovery_regression" in refused.serve.events
    report = launcher.adopt(root, confirm="", regressions=[], dry_run=True)
    assert report["report"]["ledger_bound"] is False
    adopted = launcher.adopt(root, confirm=report["head_sha256"], regressions=["no_deletion_ledger"])
    assert adopted["path"] == "R-b"


def test_boot_class_adoption_does_not_spend_the_crash_budget(root, tmp_path):
    boot = tmp_path / "boot_id"
    boot.write_text("boot-1\n")
    manager = rehearsal.FakeManager(budget=5.0, state="stopping")
    # A reboot: the manager is stopping, the child is too slow, and the cap kills the run.
    rebooted = _serve(root, tmp_path, "slow_stop", seconds=30, name="a.txt", manager=manager,
                      boot_id_file=boot, stop_grace=60, unit_timeout_stop=90)
    assert rebooted.code == 68 and _last_exit(root)["manager_stopping"] is True
    boot.write_text("boot-2\n")
    served = _serve(root, tmp_path, "clean", name="b.txt", boot_id_file=boot)
    assert served.code == 0 and "recovery R-b closed" in served.serve.events
    assert launcher._recoveries(root)[-1]["class"] == "boot"
    crashed = _serve(root, tmp_path, "crash", stop_when=None, seconds=0.1, name="c.txt", boot_id_file=boot)
    assert crashed.code == 68  # the 24 h crash budget is still available
    again = _serve(root, tmp_path, "clean", name="d.txt", boot_id_file=boot)
    assert again.code == 0  # records are keyed by head digest, not time: compare the multiset
    assert sorted(item["class"] for item in launcher._recoveries(root)) == ["boot", "crash"]


def test_shutdown_mode_forwards_first_then_probes_and_writes_fast(root, tmp_path, monkeypatch):
    events = []
    manager = rehearsal.FakeManager(budget=12.0, state="stopping", events=events)
    forward = launcher.Serve._forward

    def recorded_forward(self, process, relay, stop_at):
        sent = forward(self, process, relay, stop_at)
        if sent:
            events.append("forward")
        return sent

    monkeypatch.setattr(launcher.Serve, "_forward", recorded_forward)
    result = _serve(root, tmp_path, "slow_stop", seconds=1.0, manager=manager, stop_grace=60, unit_timeout_stop=90)
    assert result.code == 0, result.serve.events
    serve_events = result.serve.events
    assert serve_events.index("stop forwarded") < serve_events.index("system state stopping")
    assert events == ["budget", "forward", "system_state"]  # the SIGTERM precedes the systemctl probe
    assert dict(result.phases)["fast_after"] == "True"
    assert "STATUS=reboot budget insufficient (12s)" in result.serve.notifier.sent
    assert any(item.startswith("shutdown mode grace=") for item in serve_events)
    assert _last_exit(root)["manager_stopping"] is True


def test_clean_reboot_budget_uses_the_normal_grace(root, tmp_path):
    manager = rehearsal.FakeManager(budget=300.0, state="stopping")
    result = _serve(root, tmp_path, "slow_stop", seconds=1.0, manager=manager, stop_grace=60, unit_timeout_stop=90)
    assert result.code == 0 and "shutdown mode grace=60.0" in result.serve.events
    assert not any("reboot budget" in item for item in result.serve.notifier.sent)


def test_busy_lease_is_refused_78_without_touching_the_head(root, tmp_path):
    before = _admission_files(root)
    with stage.AdmissionController(root):
        result = _serve(root, tmp_path, "clean", stop_when=None)
    assert result.code == 78 and "refused runtime_admission_busy" in result.serve.events
    assert _admission_files(root) == before


# --- S3 refusals (78, admission bytes unchanged) --------------------------------------------------

def _refused(root, tmp_path, code):
    before = _admission_files(root)
    result = _serve(root, tmp_path, "clean", stop_when=None, timeout=30.0)
    assert result.code == 78 and f"refused {code}" in result.serve.events, result.serve.events
    assert _admission_files(root) == before


def test_selection_mismatch_is_refused(root, tmp_path):
    (root / "current").unlink()
    (root / "current").symlink_to("slots/other/runtime")
    _refused(root, tmp_path, "runtime_admission_launcher_selection_mismatch")


def test_controller_downgrade_and_missing_install_are_refused(root, tmp_path):
    path = root / "admission" / "controller.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps(dict(record, controller_epoch=stage.CONTROLLER_EPOCH + 1)))
    _refused(root, tmp_path, "runtime_admission_controller_downgrade")
    path.unlink()
    _refused(root, tmp_path, "runtime_admission_controller_not_installed")


def test_format_epoch_below_the_minimum_is_refused(root, tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "MIN_FORMAT_EPOCH", stage.FORMAT_EPOCH + 1)
    _refused(root, tmp_path, "runtime_admission_rollback_format_closed")


def test_pre_launcher_slot_is_refused_by_serve(artifacts, tmp_path):
    if "pre" not in artifacts["archives"]:
        pytest.skip("no pre-0d ancestor")
    root, _state = rehearsal.managed_root(artifacts["archives"]["pre"], tmp_path, controller=False)
    launcher.install_controller(root, artifacts["archives"]["x"])
    _refused(root, tmp_path, "runtime_admission_launcher_contract_missing")


@pytest.mark.parametrize("settings, code", [
    ("whisper:\n  model: ./models/whisper-v3-turbo\n", "relative_path:whisper.model"),
    ("logging:\n  file: ./logs/voice_assistant.log\n", "relative_path:logging.file"),
    ("logging:\n  file: ./data/voice_assistant.log\n", "log_under_state:logging.file"),
    ("memory:\n  db_path: /tmp/elsewhere.db\n", "foreign_serving_store:memory.db_path"),
])
def test_path_preflight_refuses_relative_and_foreign_paths(artifacts, tmp_path, settings, code):
    root, _state = rehearsal.managed_root(artifacts["archives"]["x"], tmp_path, settings="schema: t\n" + settings)
    _refused(root, tmp_path, "runtime_admission_launcher_path:" + code)


# --- install-controller ---------------------------------------------------------------------------

def test_install_controller_is_monotonic_and_reports_the_rollback_window(artifacts, tmp_path):
    root, _state = rehearsal.managed_root(artifacts["archives"]["x"], tmp_path)
    if "pre" in artifacts["archives"]:
        stage.prepare_slot(artifacts["archives"]["pre"], root, "old", root / "slots" / "s" / "state")
    result = launcher.install_controller(root, artifacts["archives"]["y"])
    assert result["commit"] == artifacts["commits"]["y"]
    assert os.readlink(root / "controller" / "current") == artifacts["commits"]["y"]
    assert (root / "controller" / "current" / "scripts" / "managed_launcher.pyc").is_file()
    if "pre" in artifacts["archives"]:
        assert result["closed_for_rollback"] == ["old"]
    path = root / "admission" / "controller.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), controller_epoch=99)))
    with pytest.raises(launcher.LauncherRefusal, match="runtime_admission_controller_downgrade"):
        launcher.install_controller(root, artifacts["archives"]["x"])
    if "pre" in artifacts["archives"]:
        with pytest.raises(launcher.LauncherRefusal, match="runtime_admission_controller_unmanaged"):
            launcher.install_controller(root, artifacts["archives"]["pre"])


# --- install-launcher / uninstall-launcher (v3 N8, L4) --------------------------------------------

BASE_UNIT = {"Type": "simple", "Restart": "on-failure", "RestartSec": "10s", "TimeoutStartSec": "120",
             "Environment": ["HOME_ASSISTANT_URL=http://ha.invalid"]}
RELEASE_CONF = ("[Service]\nExecStart=\nExecStart=/home/kza/kza/.venv/bin/python -B -P -m src.main\n"
                "Environment=PYTHONPATH=/home/kza/releases/current\nEnvironment=CONFIG_PATH=config/settings.yaml\n"
                "WorkingDirectory=/home/kza/releases/slots/legacy/state\n")


def _systemctl(tmp_path, active="inactive"):
    unit_dir = tmp_path / "unit.d"
    unit_dir.mkdir()
    (unit_dir / "release.conf").write_text(RELEASE_CONF)
    (unit_dir / "credentials.conf").write_text("[Service]\nEnvironmentFile=/home/kza/secrets/kza.env\n")
    shim = fake_systemd.write_systemctl_shim(tmp_path / "bin", tmp_path / "sd-state", unit_dir, base=BASE_UNIT,
                                             active_state=active)
    return unit_dir, launcher.ServiceManager(executable=str(shim))


def _settings(root, **overrides):
    values = dict(python="/home/kza/kza/.venv/bin/python", stop_grace=120, timeout_start=180, timeout_stop=200,
                  watchdog_sec=0, console_log=str(root.parent / "logs" / "console.log"))
    values.update(overrides)
    return launcher.LauncherSettings(**values)


def test_install_launcher_renders_managed_conf_and_records_pre_managed_properties(root, tmp_path):
    unit_dir, manager = _systemctl(tmp_path)
    head_before = _head(root)
    result = launcher.install_launcher(root, unit_dir=unit_dir, template=TEMPLATE, settings=_settings(root),
                                       manager=manager)
    assert result["verification_mismatches"] == [] and result["reloaded"]
    assert not (unit_dir / "release.conf").exists()
    assert (unit_dir / "release.conf.unmanaged").read_text() == RELEASE_CONF
    assert (unit_dir / "credentials.conf").exists()  # credential drop-ins untouched
    record = launcher._latest_install(root)
    assert record["properties"]["Type"] == "simple" and "Environment" not in record["properties"]
    assert record["legacy_exec_start"] == ["/home/kza/kza/.venv/bin/python", "-B", "-P", "-m", "src.main"]
    assert record["stop_timeout_dropin"]["value_seconds"] == 200 + 90 + 15  # kza-llm-ik after kza-voice
    assert not Path(record["stop_timeout_dropin"]["path"]).exists()
    assert _head(root) == head_before


@pytest.mark.parametrize("problem, code", [
    ("active", "runtime_admission_launcher_unit_active"),
    ("console", "runtime_admission_launcher_console_under_slots"),
    ("budget", "runtime_admission_launcher_stop_budget"),
])
def test_install_launcher_refusals(root, tmp_path, problem, code):
    unit_dir, manager = _systemctl(tmp_path, active="active" if problem == "active" else "inactive")
    settings = _settings(root, **({"console_log": str(root / "slots" / "s" / "state" / "console.log")}
                                  if problem == "console" else {"timeout_stop": 100} if problem == "budget" else {}))
    with pytest.raises(launcher.LauncherRefusal, match=code):
        launcher.install_launcher(root, unit_dir=unit_dir, template=TEMPLATE, settings=settings, manager=manager)
    assert (unit_dir / "release.conf").exists() and not (unit_dir / "managed.conf").exists()


def test_uninstall_launcher_pins_the_legacy_slot_and_leaves_admission_untouched(artifacts, root, tmp_path):
    unit_dir, manager = _systemctl(tmp_path)
    launcher.install_launcher(root, unit_dir=unit_dir, template=TEMPLATE, settings=_settings(root), manager=manager)
    legacy_archive = artifacts["archives"].get("pre", artifacts["archives"]["x"])
    legacy_commit = artifacts["commits"]["pre"] or artifacts["commits"]["x"]
    source = rehearsal.offline_source(tmp_path / "legacy-source")
    stage.prepare_slot(legacy_archive, root, "legacy", source)
    before = _admission_files(root)
    current = os.readlink(root / "current")
    for kwargs, code in (({"phrase": "revivir"}, "runtime_admission_uninstall_phrase"),
                         ({"decision_id": "x"}, "runtime_admission_uninstall_decision")):
        values = dict(decision_id="user-decision-2026-11-12", phrase=launcher.UNINSTALL_PHRASE,
                      legacy_slot="legacy", legacy_commits=(legacy_commit[:7],))
        values.update(kwargs)
        with pytest.raises(launcher.LauncherRefusal, match=code):
            launcher.uninstall_launcher(root, unit_dir=unit_dir, manager=manager, **values)
    with pytest.raises(launcher.LauncherRefusal, match="runtime_admission_uninstall_legacy_commit"):
        launcher.uninstall_launcher(root, unit_dir=unit_dir, manager=manager, decision_id="d-123",
                                    phrase=launcher.UNINSTALL_PHRASE, legacy_slot="legacy")
    result = launcher.uninstall_launcher(root, unit_dir=unit_dir, manager=manager, decision_id="d-123",
                                         phrase=launcher.UNINSTALL_PHRASE, legacy_slot="legacy",
                                         legacy_commits=(legacy_commit[:7],))
    assert result["mismatches"] == []
    assert not (unit_dir / "managed.conf").exists() and (unit_dir / "managed.conf.disabled").exists()
    legacy = (unit_dir / "legacy.conf").read_text()
    assert "Environment=PYTHONPATH=" + str(root / "slots" / "legacy" / "runtime") in legacy
    assert "WorkingDirectory=" + str(root / "slots" / "legacy" / "state") in legacy
    assert "current" not in legacy and "Type=simple" in legacy and "WatchdogSec=0" in legacy
    after = _admission_files(root)
    added = set(after) - set(before)
    assert len(added) == 1 and next(iter(added)).startswith("launcher/uninstall-")
    assert {key: after[key] for key in before} == before
    assert os.readlink(root / "current") == current
    record = json.loads(after[next(iter(added))])
    assert record["decision_id"] == "d-123" and record["legacy_commit"] == legacy_commit
    assert record["phrase_sha256"] != launcher.UNINSTALL_PHRASE


# --- K7 / resumable seal ---------------------------------------------------------------------------

def test_closed_state_match_accepts_only_its_own_seal_intent(tmp_path):
    root = tmp_path
    binding = {"state_identity": {"a": 1}, "store_roots": {}, "schemas": {}, "sdk_versions": {},
               "ownership_sha256": "1" * 64, "authority": {"digest": "2" * 64}}
    record = {"closed_state_sha256": stage.closed_state_sha256(binding)}
    sealed = dict(binding, ownership_sha256="3" * 64)
    run_id = "a" * 32
    assert stage._closed_state_matches(root, run_id, binding, record)
    assert not stage._closed_state_matches(root, run_id, sealed, record)
    stage._atomic_record(stage._seal_intent_path(root, run_id), {
        "schema_version": 1, "run_id": run_id, "pre_seal_ownership_sha256": "1" * 64,
        "sealed_ownership_sha256": "3" * 64}, immutable=True)
    assert stage._closed_state_matches(root, run_id, sealed, record)
    assert not stage._closed_state_matches(root, run_id, dict(sealed, ownership_sha256="4" * 64), record)
    assert not stage._closed_state_matches(root, "b" * 32, sealed, record)


def test_manifest_is_checked_under_the_root_lock(root, tmp_path, monkeypatch):
    """Review 0d3 K7: the manifest check runs inside _locked_root, before the pre-seal snapshot."""
    events = []
    original_lock, original_manifest, original_snapshot = (stage._locked_root, stage.state_manifest_sha256,
                                                           stage._binding_snapshot)

    class Lock:
        def __init__(self, path):
            self.inner = original_lock(path)

        def __enter__(self):
            events.append("lock")
            return self.inner.__enter__()

        def __exit__(self, *exc):
            events.append("unlock")
            return self.inner.__exit__(*exc)

    def manifest(path):
        events.append("manifest")
        return original_manifest(path)

    def snapshot(*args, **kwargs):
        events.append("snapshot")
        return original_snapshot(*args, **kwargs)

    original_seal = stage.seal_admission_run

    def seal(*args, **kwargs):
        events.append("seal")
        monkeypatch.setattr(stage, "_locked_root", Lock)
        monkeypatch.setattr(stage, "state_manifest_sha256", manifest)
        monkeypatch.setattr(stage, "_binding_snapshot", snapshot)
        return original_seal(*args, **kwargs)
    monkeypatch.setattr(stage, "seal_admission_run", seal)
    assert _serve(root, tmp_path, "clean").code == 0
    tail = events[events.index("seal"):]
    assert tail[1:4] == ["lock", "manifest", "snapshot"], tail


def test_a_live_recorded_child_is_refused_78(root, tmp_path):
    """R-c: the head is running and its recorded child (pid plus start time) is alive."""
    import subprocess
    import sys
    assert _serve(root, tmp_path, "crash", stop_when=None, seconds=0.1, name="a.txt").code == 68
    head = _head(root)
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        run_path = launcher.run_record_path(root, head["run_id"])
        record = json.loads(run_path.read_text())
        run_path.unlink()
        stage._atomic_record(run_path, dict(record, child_pid=alive.pid,
                                            child_start_time=launcher.process_start_time(alive.pid)))
        with stage.AdmissionController(root, allow_open_head=True):
            decision = launcher.recover_open_head(root, head, mode="bounded", boot_id="b", wall=time.time())
        assert decision.closed_head is None and decision.exit_code == 78
        assert decision.code == "runtime_admission_recovery_child_alive"
    finally:
        alive.kill()
        alive.wait()


def test_a_removed_recovery_record_invalidates_the_recovered_head(root, tmp_path):
    assert _serve(root, tmp_path, "crash", stop_when=None, seconds=0.1, name="a.txt").code == 68
    with stage.AdmissionController(root, allow_open_head=True):
        decision = launcher.recover_open_head(root, _head(root), mode="bounded", boot_id="b", wall=time.time())
    assert decision.closed_head is not None
    for path in (root / "admission" / "recoveries").glob("*.json"):
        path.unlink()
    with pytest.raises(ValueError, match="runtime_admission_recovery_record_missing"):
        stage.read_manual_head(root, decision.closed_head)
    refused = _serve(root, tmp_path, "clean", name="b.txt")
    assert refused.code == 78 and "refused runtime_admission_recovery_record_missing" in refused.serve.events
