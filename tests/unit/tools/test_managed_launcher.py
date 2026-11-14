"""U3b-0d managed launcher: pure units (no compiled slot).

Relay, self-pipe, child environment, reboot budget, drop-in and template
rendering, release integers, resurrection digests and the child-side
watchdog/control additions to src/core/process_lifecycle.
"""

import asyncio
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time

import pytest

from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from scripts.build_runtime import release_integers
from src.core import process_lifecycle as lifecycle_module
from src.memory.deletion_ledger import (
    DeletionKind, append_deletion, consent_withdrawal_subject, initialize_deletion_ledger, subject_digest,
)
from src.memory.erasure_journal import ERASURE_SCHEMA
from tests.mocks import fake_systemd

REPO = Path(__file__).resolve().parents[3]
TEMPLATE = REPO / "systemd" / "kza-voice.service.d" / "managed.conf.template"


class Recorder(launcher.SystemdNotifier):
    def __init__(self):
        super().__init__(None)


# --- notify relay (v2 notify_relay, NotifyAccess=main) -----------------------------------------

def test_relay_sends_ready_once_and_watchdog_only_after_ready():
    notifier = Recorder()
    relay = launcher.NotifyRelay(notifier)
    relay.feed(b"WATCHDOG=1\nREADY=1\nREADY=1\nWATCHDOG=1\n")
    assert notifier.sent == ["READY=1", "WATCHDOG=1"]
    assert relay.dropped == 1 and relay.watchdog_relayed == 1


def test_relay_validates_status_extend_and_drops_everything_else():
    notifier = Recorder()
    relay = launcher.NotifyRelay(notifier)
    relay.feed(b"STATUS=ok\nSTATUS=" + b"x" * 201 + b"\nSTATUS=\nEXTEND_TIMEOUT_USEC=5000000\n"
               b"EXTEND_TIMEOUT_USEC=-1\nMAINPID=1\nFDSTORE=1\nSTOPPING=1\n\xff\n")
    assert notifier.sent == ["STATUS=ok", "EXTEND_TIMEOUT_USEC=5000000", "STOPPING=1"]
    assert relay.dropped == 6


def test_relay_consumes_armed_without_relaying_it():
    notifier = Recorder()
    relay = launcher.NotifyRelay(notifier)
    assert relay.armed is False
    relay.feed(b"ARMED=1\nREADY=1\n")
    assert relay.armed is True and notifier.sent == ["READY=1"] and relay.dropped == 0


def test_arm_process_entry_sends_armed_first():
    read, write = os.pipe()
    previous = signal.getsignal(signal.SIGTERM)
    try:
        lifecycle = lifecycle_module.arm_process_entry({"KZA_NOTIFY_FD": str(write)})
        os.set_blocking(read, False)  # a missing line must fail, not hang
        try:
            sent = os.read(read, 100)
        except BlockingIOError:
            sent = b""
        assert sent == b"ARMED=1\n" and lifecycle.notifier.sent == ("ARMED=1",)
    finally:
        signal.signal(signal.SIGTERM, previous)
        os.close(read)
        os.close(write)


def test_relay_reassembles_split_lines():
    notifier = Recorder()
    relay = launcher.NotifyRelay(notifier)
    relay.feed(b"REA")
    relay.feed(b"DY=1\nSTAT")
    assert notifier.sent == ["READY=1"]
    relay.feed(b"US=up\n")
    assert notifier.sent == ["READY=1", "STATUS=up"]


def test_notifier_sends_datagrams_to_notify_socket(tmp_path):
    path = Path(os.path.realpath("/tmp")) / ("kza-notify-%d" % os.getpid())
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        server.bind(str(path))
        notifier = launcher.SystemdNotifier(str(path))
        notifier.send("READY=1")
        notifier.status("bad\nline")
        assert server.recv(100) == b"READY=1"
        assert server.recv(100) == b"STATUS=bad?line"
        notifier.close()
    finally:
        server.close()
        path.unlink(missing_ok=True)


# --- self-pipe (v2 M1, v3 N1) --------------------------------------------------------------------

def test_stop_pipe_handler_only_writes_a_byte_and_never_raises():
    pipe = launcher.StopPipe().install()
    try:
        assert not pipe.drain()
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.05)  # the handler runs on the main thread between bytecodes
        assert pipe.drain() and pipe.signals == 1
        os.kill(os.getpid(), signal.SIGINT)  # no KeyboardInterrupt
        time.sleep(0.05)
        assert pipe.drain() and pipe.signals == 2
        assert signal.getsignal(signal.SIGTERM) == pipe._handler
        assert launcher.StopPipe is not None and lifecycle_module.blocked_stop_signals() == frozenset()
    finally:
        pipe.close()
    assert signal.getsignal(signal.SIGTERM) is not pipe._handler


def test_launcher_never_imports_slot_code():
    """v3 N7: release integers are read as JSON; candidate code is never loaded by the controller."""
    source = (REPO / "scripts" / "managed_launcher.py").read_text()
    for forbidden in ("spec_from_file_location", "import_module", "exec_module", "__import__"):
        assert forbidden not in source


def test_launcher_never_masks_signals():
    """v3 N1: no pthread_sigmask anywhere in the controller (the mask is inherited through exec)."""
    source = (REPO / "scripts" / "managed_launcher.py").read_text()
    assert "pthread_sigmask" not in source and "SIG_BLOCK" not in source


# --- child environment ---------------------------------------------------------------------------

def test_child_environment_drops_manager_and_loader_variables():
    environ = {"NOTIFY_SOCKET": "/run/x", "WATCHDOG_USEC": "1", "WATCHDOG_PID": "2", "PYTHONPATH": "/old",
               "CONFIG_PATH": "/old.yaml", "PYTHONHOME": "/h", "KZA_RUNTIME_ADMISSION": "stale",
               "KZA_RUNTIME_OTHER": "x", "KZA_NOTIFY_FD": "9", "KZA_CONTROL_FD": "8",
               "KZA_WATCHDOG_INTERVAL_USEC": "7", "HOME_ASSISTANT_TOKEN": "secret", "PATH": "/bin"}
    ticket = {"run_id": "r"}
    child = launcher.child_environment(environ, Path("/root/slots/s/runtime"), ticket, 5, 6, 30_000_000)
    assert child["PYTHONPATH"] == "/root/slots/s/runtime" and child["CONFIG_PATH"] == "config/settings.yaml"
    assert child["KZA_RUNTIME_ADMISSION"] == stage._canonical_json(ticket).decode()
    assert child["KZA_NOTIFY_FD"] == "5" and child["KZA_CONTROL_FD"] == "6"
    assert child["KZA_WATCHDOG_INTERVAL_USEC"] == "30000000"
    assert child["HOME_ASSISTANT_TOKEN"] == "secret" and child["PATH"] == "/bin"
    for key in ("NOTIFY_SOCKET", "WATCHDOG_USEC", "WATCHDOG_PID", "PYTHONHOME", "KZA_RUNTIME_OTHER"):
        assert key not in child
    assert "KZA_WATCHDOG_INTERVAL_USEC" not in launcher.child_environment(environ, Path("/r"), ticket, 5, 6, None)


def test_watchdog_interval_is_half_and_only_for_this_pid():
    assert launcher.watchdog_interval({"WATCHDOG_USEC": "60000000"}) == 30_000_000
    assert launcher.watchdog_interval({"WATCHDOG_USEC": "60000000", "WATCHDOG_PID": str(os.getpid())}) == 30_000_000
    assert launcher.watchdog_interval({"WATCHDOG_USEC": "60000000", "WATCHDOG_PID": "1"}) is None
    assert launcher.watchdog_interval({}) is None and launcher.watchdog_interval({"WATCHDOG_USEC": "0"}) is None


# --- reboot budget (N1 = B; v3 N4/N9, review D4) ------------------------------------------------

@pytest.mark.parametrize("text, seconds", [("5s", 5.0), ("1min 30s", 90.0), ("2min 30.5s", 150.5), ("500ms", .5),
                                           ("3min", 180.0), ("infinity", None), ("", None), ("5 parsecs", None)])
def test_parse_timespan(text, seconds):
    assert launcher.parse_timespan(text) == seconds


def test_reboot_budget_clean_and_fallback():
    clean = launcher.RebootBudget(300.0, 200.0)
    assert clean.mode == "clean" and clean.status() is None
    assert clean.deadline_seconds(150.0, 1.0) == 150.0
    fallback = launcher.RebootBudget(5.0, 200.0)
    assert fallback.mode == "fallback" and fallback.status() == "reboot budget insufficient (5s)"
    assert fallback.deadline_seconds(150.0, 0.5) == pytest.approx(2.5)
    assert fallback.deadline_seconds(150.0, 10.0) == 0.0
    unknown = launcher.RebootBudget(None, 200.0)
    assert unknown.mode == "fallback" and unknown.status() == "reboot budget unknown"
    assert launcher.RebootBudget(209.0, 200.0).mode == "fallback"
    assert launcher.RebootBudget(210.0, 200.0).mode == "clean"


def test_stop_timeout_dropin_is_the_instance_zz_file_and_is_never_written(tmp_path):
    dropin = launcher.render_stop_timeout_dropin(1000, 200.0, chain_extra=90.0)
    assert dropin["path"] == "/etc/systemd/system/user@1000.service.d/zz-kza-stop-timeout.conf"
    assert dropin["text"] == "[Service]\nTimeoutStopSec=305s\n" and dropin["value_seconds"] == 305
    # zz- sorts after the vendor timeout.conf and 10-login-barrier.conf (and any 50-*.conf).
    assert sorted(["10-login-barrier.conf", "timeout.conf", "50-TimeoutStopSec.conf",
                   launcher.DROP_IN_NAME])[-1] == launcher.DROP_IN_NAME
    assert "user@.service.d" not in dropin["path"]
    assert all("restart" not in step for step in dropin["apply"])
    assert all(step.startswith("sudo ") for step in dropin["apply"])  # applied by the user, never by the tool


# --- template and installers ---------------------------------------------------------------------

def _settings(**overrides):
    values = dict(python="/home/kza/kza/.venv/bin/python", stop_grace=120, timeout_start=180, timeout_stop=200,
                  watchdog_sec=0, console_log="/home/kza/releases/logs/console.log")
    values.update(overrides)
    return launcher.LauncherSettings(**values)


def test_managed_conf_renders_the_notify_service():
    text = launcher.render_managed_conf(TEMPLATE.read_text(), Path("/home/kza/releases"), _settings())
    directives = fake_systemd.parse_dropins([text])
    assert directives["Type"] == "notify" and directives["NotifyAccess"] == "main"
    assert directives["KillMode"] == "mixed" and directives["RestartPreventExitStatus"] == "69 78"
    assert directives["Restart"] == "on-failure" and directives["StartLimitBurst"] == "5"
    assert directives["TimeoutAbortSec"] == directives["TimeoutStopSec"] == "200"
    assert directives["WatchdogSec"] == "0" and directives["WatchdogSignal"] == "SIGTERM"
    assert directives["ExecStart"] == [
        "/home/kza/kza/.venv/bin/python -I -B /home/kza/releases/controller/current/scripts/stage_runtime.pyc"
        " --root /home/kza/releases --offline serve --stop-grace 120 --recover bounded --unit-timeout-stop 200"]
    assert "PYTHONPATH" not in text and "CONFIG_PATH" not in text
    assert "child-command" not in text and "boot-id" not in text


def test_managed_conf_refuses_injected_values():
    with pytest.raises(launcher.LauncherRefusal, match="runtime_admission_launcher_value:console_log"):
        launcher.render_managed_conf(TEMPLATE.read_text(), Path("/r"), _settings(console_log="/x\nExecStart=/evil"))
    with pytest.raises(launcher.LauncherRefusal, match="runtime_admission_launcher_template"):
        launcher.render_managed_conf("ExecStart={unknown}", Path("/r"), _settings())


def test_managed_properties_need_the_exact_exit_status_set_and_the_rendered_exec_start():
    """Review P9 (exact set, never a subset test) and P7 (a later release.conf must be visible)."""
    root = Path("/home/kza/releases")
    text = launcher.render_managed_conf(TEMPLATE.read_text(), root, _settings())
    shown = fake_systemd.show_properties(fake_systemd.parse_dropins([text]))
    assert launcher.verify_managed_properties(shown, text, root) == []
    for status in ("69 1", "69", "78 69 1", ""):
        assert launcher.verify_managed_properties(dict(shown, RestartPreventExitStatus=status), text, root) == [
            "RestartPreventExitStatus"]
    restored = ("[Service]\nExecStart=\nExecStart=/home/kza/kza/.venv/bin/python -B -P -m src.main\n"
                "WorkingDirectory=/home/kza/releases/slots/legacy/state\n")
    later = fake_systemd.show_properties(fake_systemd.parse_dropins([text, restored]))
    assert launcher.verify_managed_properties(later, text, root) == ["ExecStart", "WorkingDirectory"]
    assert launcher.verify_managed_properties(None, text, root) == ["properties_unavailable"]


def test_console_logrotate_stanza_is_rendered_and_detected(tmp_path):
    """Review P5: the managed console log must be rotated with copytruncate."""
    console = "/home/kza/releases/logs/console.log"
    stanza = launcher.render_console_logrotate(console)
    assert stanza.splitlines()[0] == console + " {" and "    copytruncate" in stanza.splitlines()
    conf = tmp_path / "console-logrotate.conf"
    assert not launcher.console_rotated(conf, console)  # missing file
    conf.write_text("/home/kza/releases/slots/old/state/logs/console.log {\n    copytruncate\n}\n")
    assert not launcher.console_rotated(conf, console)  # only the old slot's console
    conf.write_text(conf.read_text() + "/home/kza/releases/logs/console.log {\n    rotate 24\n}\n")
    assert not launcher.console_rotated(conf, console)  # no copytruncate: systemd keeps the old fd
    conf.write_text(conf.read_text() + stanza)
    assert launcher.console_rotated(conf, console)
    with pytest.raises(launcher.LauncherRefusal, match="runtime_admission_launcher_value:console_log"):
        launcher.render_console_logrotate("logs/console.log")


def test_legacy_conf_replaces_only_the_pythonpath_assignment_and_keeps_other_directives(tmp_path):
    """Review P8: a mixed Environment line keeps CONFIG_PATH; directives outside the list carry over."""
    unmanaged = tmp_path / "release.conf.unmanaged"
    unmanaged.write_text("[Unit]\nDescription=kza voice\n[Service]\nExecStart=\nExecStart=/py -m src.main\n"
                         "Environment=PYTHONPATH=/home/kza/releases/current CONFIG_PATH=config/settings.yaml\n"
                         "Environment=\"KZA_LABEL=a b\"\nLimitNOFILE=4096\nUnsetEnvironment=HTTP_PROXY\n"
                         "WorkingDirectory=/home/kza/releases/slots/old/state\nKillMode=control-group\n")
    record = {"properties": {"TimeoutStartUSec": "2min", "TimeoutStopUSec": "1min 30s",
                             "StandardOutput": "append:/c.log", "StandardError": "append:/c.log"},
              "legacy_exec_start": ["/py", "-B", "-P", "-m", "src.main"]}
    runtime, state = Path("/r/slots/legacy/runtime"), Path("/r/slots/legacy/state")
    text = launcher.render_legacy_conf(record, runtime, state, unmanaged)
    directives = fake_systemd.parse_dropins([text])
    assert directives["Environment"] == ["CONFIG_PATH=config/settings.yaml", "KZA_LABEL=a b",
                                         "PYTHONPATH=/r/slots/legacy/runtime"]
    assert directives["LimitNOFILE"] == "4096" and directives["UnsetEnvironment"] == "HTTP_PROXY"
    assert directives["KillMode"] == "control-group" and directives["WorkingDirectory"] == str(state)
    assert directives["ExecStart"] == ["/py -B -P -m src.main"]
    assert "current" not in text and text.count("WorkingDirectory=") == 1
    assert text.index("[Unit]") > text.index("LimitNOFILE") and "Description=kza voice" in text
    assert launcher.carried_directives(unmanaged) == ["[Service]KillMode", "[Service]LimitNOFILE",
                                                      "[Service]UnsetEnvironment", "[Unit]Description"]


def test_exec_start_argv_parses_systemctl_show():
    value = ("{ path=/home/kza/kza/.venv/bin/python ; argv[]=/home/kza/kza/.venv/bin/python -B -P -m src.main ;"
             " ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }")
    assert launcher.exec_start_argv(value) == ["/home/kza/kza/.venv/bin/python", "-B", "-P", "-m", "src.main"]
    with pytest.raises(launcher.LauncherRefusal):
        launcher.exec_start_argv("")


# --- release integers (v3 N7, review 0d3 K2) -----------------------------------------------------

def test_release_integers_come_from_the_built_revision_text():
    source = (REPO / "scripts" / "stage_runtime.py").read_bytes()
    assert release_integers(source) == {"launcher_contract": stage.LAUNCHER_CONTRACT,
                                        "format_epoch": stage.FORMAT_EPOCH,
                                        "receipt_state": stage.RECEIPT_STATE_SCHEMA,
                                        "controller_epoch": stage.CONTROLLER_EPOCH}
    assert release_integers(b"import os\n") == {}
    assert release_integers(b"    LAUNCHER_CONTRACT = 3\n") == {}  # column 0 only
    with pytest.raises(ValueError, match="Ambiguous"):
        release_integers(b"FORMAT_EPOCH = 1\nFORMAT_EPOCH = 2\n")


def test_release_integers_and_receipt_state_requirement(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    def binding(manifest: bytes) -> dict:
        (runtime / "release.json").write_bytes(manifest)
        return {"runtime_root": str(runtime), "release_sha256": hashlib.sha256(manifest).hexdigest()}

    old = binding(b'{"files": {}}')
    assert stage.release_integers(old) == dict.fromkeys(stage._RELEASE_INTEGER_KEYS, 0)
    assert stage.receipt_state_required(old) is False
    new = binding(b'{"launcher_contract": 3, "receipt_state": 2}')
    assert stage.receipt_state_required(new) is True
    assert stage.receipt_state_required(binding(b'{"receipt_state": 2}')) is True
    changed = dict(new, release_sha256="0" * 64)
    with pytest.raises(ValueError, match="runtime_admission_release_changed"):
        stage.release_integers(changed)
    with pytest.raises(ValueError, match="runtime_admission_release_integer"):
        stage.release_integers(binding(b'{"format_epoch": "1"}'))


# --- recovery classification ---------------------------------------------------------------------

def test_classify_run_boot_and_crash():
    assert launcher.classify_run(None, None, "b2") == "crash"
    assert launcher.classify_run({"boot_id": "b1"}, None, "b2") == "boot"  # power loss / SIGKILL at reboot
    assert launcher.classify_run({"boot_id": "b1"}, {"manager_stopping": True}, "b2") == "boot"
    assert launcher.classify_run({"boot_id": "b1"}, {"manager_stopping": False}, "b2") == "crash"
    assert launcher.classify_run({"boot_id": "b2"}, None, "b2") == "crash"
    assert launcher.classify_run({"boot_id": "unknown"}, None, "b2") == "crash"


def test_process_start_time_identifies_a_live_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        token = launcher.process_start_time(child.pid)
        assert token is not None and launcher.child_alive(child.pid, token)
        assert not launcher.child_alive(child.pid, token + "x")
    finally:
        child.kill()
        child.wait()
    assert launcher.process_start_time(child.pid) is None and not launcher.child_alive(child.pid, None)


def test_kill_stragglers_is_not_checked_outside_a_unit_cgroup(tmp_path, monkeypatch):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    assert launcher._cgroup_procs_path() is None and launcher.kill_stragglers(None) is None
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    procs = tmp_path / "cgroup.procs"
    procs.write_text(f"{os.getpid()}\n{victim.pid}\n")
    try:
        assert launcher.kill_stragglers(procs, wait=0.2) == 1
        assert victim.wait(timeout=5) == -signal.SIGKILL
    finally:
        if victim.poll() is None:
            victim.kill()


# --- resurrection (v3 N2 R-b) --------------------------------------------------------------------

def test_subject_digest_matches_the_ledger_module():
    salt = "ab" * 32
    assert launcher.subject_digest(salt, "conv-1") == subject_digest(salt, "conv-1")


def _ledger_db(path: Path, deletions: list[tuple[DeletionKind, str]]) -> None:
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        initialize_deletion_ledger(db)
        db.execute("BEGIN IMMEDIATE")
        for kind, subject in deletions:
            append_deletion(db, kind, subject, 2_000_000_000.0)
        db.execute("COMMIT")


def _target(db, participant: str, descriptor: dict) -> str:
    """The erasure_targets row ErasureJournal._register_target writes (version 1)."""
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    key = hashlib.sha256(json.dumps([participant, 1, encoded]).encode()).hexdigest()
    db.execute("INSERT INTO erasure_targets VALUES (?,?,?,?)", (key, participant, 1, encoded))
    return key


def test_resurrection_counts_each_family_and_skips_pending_erasures(tmp_path):
    """Fact, voice and consent rows as in U3b-0d; the journal now has the real ERASURE_SCHEMA shape (0d4a)."""
    state = tmp_path.resolve() / "state"
    (state / "data" / "contexts").mkdir(parents=True)
    _ledger_db(state / "data" / "privacy.db", [
        (DeletionKind.CONVERSATION_ERASED, "c-erased"), (DeletionKind.CONVERSATION_ERASED, "c-pending"),
        (DeletionKind.FACT_DELETED, "f-gone"), (DeletionKind.VOICE_PROFILE_DELETED, "v-gone"),
        (DeletionKind.VOICE_CONSENT_WITHDRAWN, consent_withdrawal_subject("p1", 3))])
    with closing(sqlite3.connect(state / "data" / "privacy.db")) as db, db:
        db.execute("CREATE TABLE facts (id TEXT, state TEXT)")
        db.executemany("INSERT INTO facts VALUES (?,?)", [("f-gone", "confirmed"), ("f-ok", "confirmed"),
                                                          ("f-tomb", "deleted")])
        db.execute("CREATE TABLE voice_profiles (id TEXT, deleted_at REAL)")
        db.executemany("INSERT INTO voice_profiles VALUES (?,?)", [("v-gone", None), ("v-ok", None)])
        db.execute("CREATE TABLE voice_attribution_consents (person_id TEXT, revision INTEGER)")
        db.executemany("INSERT INTO voice_attribution_consents VALUES (?,?)", [("p1", 2), ("p2", 2)])
        db.execute("CREATE TABLE conversation_tombstones (conversation_id TEXT PRIMARY KEY)")
        db.executescript(ERASURE_SCHEMA)
        events = _target(db, "events", {"path": str(state / "data" / "events.db"), "table": "events_policy_v1"})
        captures = _target(db, "captures", {"db_path": str(state / "data" / "events.db"),
                                            "audio_dir": str(state / "data" / "audio")})
        literal = _target(db, "diagnostics", {"kind": "unresolved_diagnostic_inventory"})
        db.executemany("INSERT INTO conversation_erasures VALUES (?,?,0,'synthetic')",
                       [("o1", "c-erased"), ("o2", "c-pending")])
        db.executemany("INSERT INTO erasure_steps(operation_id, participant, target_id, version, state)"
                       " VALUES (?,?,?,1,?)", [
                           ("o1", "events", events, "complete"), ("o1", "captures", captures, "complete"),
                           ("o1", "diagnostics", literal, "pending"),  # never exempts another store
                           ("o2", "events", events, "pending"), ("o2", "captures", captures, "pending")])
    with closing(sqlite3.connect(state / "data" / "events.db")) as db, db:
        db.execute("CREATE TABLE utterances (conversation_id TEXT)")
        db.executemany("INSERT INTO utterances VALUES (?)", [("c-erased",), ("c-pending",), ("c-live",)])
        db.execute("CREATE TABLE events_policy_v1 (sources_json TEXT)")
        db.execute("INSERT INTO events_policy_v1 VALUES (?)", ('[{"conversation_id": "c-erased"}]',))
    (state / "data" / "contexts" / "u.json").write_text('{"history": [{"conversation_id": "c-erased"}]}')
    counts = launcher.resurrection_counts(state)
    assert counts == {"conversation": 1, "fact": 1, "voice_profile": 1, "voice_consent": 1}
    with closing(sqlite3.connect(state / "data" / "events.db")) as db, db:
        db.execute("DELETE FROM utterances WHERE conversation_id='c-erased'")
        db.execute("DELETE FROM events_policy_v1")
    (state / "data" / "contexts" / "u.json").write_text("{}")
    assert launcher.resurrection_counts(state)["conversation"] == 0


def _conversation_columns() -> set[str]:
    """Every contract column that can carry a conversation id: conversation_id or *sources_json."""
    found = set()
    for table, columns in stage._TABLE_COLUMNS.items():
        for item in columns.split():
            name = item.split(":")[0]
            if name == "conversation_id" or name.endswith("sources_json"):
                found.add(f"{table}.{name}")
    return found


# Payload-bearing columns outside the conversation_id/*sources_json name predicate (review F1),
# listed by hand: each must stay classified while the column exists.
PAYLOAD_COLUMNS = {
    "stt_corrections_v1.original_text", "stt_corrections_v1.corrected_text",
    "conversation_export_parents_v1.parent_id",
    "voice_capture_reviews.transcript", "voice_capture_reviews.audio_wav",
    "voice_capture_reviews.audio_metadata_json", "voice_capture_reviews.transcript_metadata_json",
    "legacy_quarantine_records_v1.document", "legacy_quarantine_records_v1.metadata_json",
}
# One representative path per file kind of the state (the integration test checks every file the
# real code writes: test_every_file_kind_the_real_code_writes_is_mapped_or_classified).
FILE_KINDS = {
    "data/contexts/policy_v3/u.json": "contexts:*.json",
    "data/conversations/policy_v1/0b6f3c1e-1111-4222-8333-444455556666.json": "collectors:policy_v1/*.json",
    "data/exports/policy_v1/0b6f3c1e-1111-4222-8333-444455556666.jsonl": "collectors:policy_v1/*.jsonl",
    "data/wake/policy_v1/0b6f3c1e-1111-4222-8333-444455556666.wav": "file:policy_v1/*.wav",
    "data/audio/policy_v1/0b6f3c1e-1111-4222-8333-444455556666.flac": "file:policy_v1/*.flac",
    "data/stt/policy_v1/0b6f3c1e-1111-4222-8333-444455556666.wav.part": "file:policy_v1/*.part",
    "data/metrics_policy_v1.jsonl.1": "file:diagnostics *.jsonl[.N]",
    "data/chroma/0b6f3c1e-1111-4222-8333-444455556666/header.bin": "file:chroma segment",
    "data/privacy.db.lock": "file:*.lock / *.lease",
    "data/privacy.db-wal": "sqlite",
    "data/chroma/chroma.sqlite3": "sqlite",
}


def test_every_conversation_source_is_mapped_or_classified_not_scanned():
    """0d4a build-time classification: a new unclassified column or file kind fails here."""
    mapped = {entry.source for entry in launcher.CONVERSATION_SOURCES}
    classified = set(launcher.CONVERSATION_NOT_SCANNED)
    assert not mapped & classified
    columns = _conversation_columns() | PAYLOAD_COLUMNS
    unclassified = (columns | {"chroma:embedding_metadata.conversation_id"} | set(FILE_KINDS.values())) \
        - {"sqlite"} - mapped - classified
    assert unclassified == set()
    # Column keys name real contract columns; file keys are produced by conversation_file_kind.
    contract = {f"{table}.{item.split(':')[0]}" for table, items in stage._TABLE_COLUMNS.items()
                for item in items.split()}
    assert {key for key in classified if not key.startswith("file:")} <= contract
    assert PAYLOAD_COLUMNS <= contract
    for relative, kind in FILE_KINDS.items():
        assert launcher.conversation_file_kind(Path(relative)) == kind, relative
    assert {key for key in classified if key.startswith("file:")} <= set(FILE_KINDS.values())
    assert launcher.conversation_file_kind(Path("data/new_store/x.bin")).startswith("unclassified:")
    assert mapped == {"utterances.conversation_id", "capture_conversations.id", "events_policy_v1.sources_json",
                      "wake_clips_v1.sources_json", "conversation_turns_v1.input_sources_json",
                      "chroma:embedding_metadata.conversation_id", "contexts:*.json",
                      "collectors:policy_v1/*.json", "collectors:policy_v1/*.jsonl"}
    for entry in launcher.CONVERSATION_SOURCES:
        assert entry.version == 1 and callable(entry.resolver) and callable(entry.scan)
        assert entry.participant in {"captures", "events", "collectors", "semantic_index", "contexts"}


def test_an_unclassified_conversation_column_fails_the_classification(monkeypatch):
    monkeypatch.setitem(stage._TABLE_COLUMNS, "new_store_v1", "id:TEXT conversation_id:TEXT")
    with pytest.raises(AssertionError):
        test_every_conversation_source_is_mapped_or_classified_not_scanned()


def _memory(state: Path, *, delete_error: Exception | None = None):
    """The real long-term memory: FactLedger in privacy.db and a real Chroma policy index."""
    from src.memory.long_term import LongTermMemory
    from tests.mocks.memory_graph import MemoryClient
    client = MemoryClient(state / "data" / "memory_db")
    memory = LongTermMemory(client, ledger_path=str(state / "data" / "privacy.db"), close_client=client.close)
    memory.initialize()
    if delete_error is not None:
        client.collections[memory.index_collection_name].delete_error = delete_error
    return memory


def test_resurrection_sees_fact_ids_as_the_memory_index_stores_them(tmp_path):
    """Review P1: LongTermMemory writes ids=[fact.id] with {fact_policy, revision}; no fact_id metadata."""
    from src.memory.fact_ledger import MemoryReviewer
    from tests.mocks.memory_sources import memory_source
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    memory = _memory(state)
    kept = memory.store_fact("Synthetic kept fact", "fact", sources=(memory_source(),))
    gone = memory.store_fact("Synthetic deleted fact", "fact", sources=(memory_source(),))
    asyncio.run(memory.aclose())
    older = tmp_path / "memory_db.older"
    shutil.copytree(state / "data" / "memory_db", older)  # a backup holding both facts' embeddings
    memory = _memory(state)
    assert memory.delete_fact(gone, MemoryReviewer("synthetic-reviewer")) is True  # reconciled: index cleaned
    asyncio.run(memory.aclose())
    with closing(sqlite3.connect(state / "data" / "memory_db" / "chroma.sqlite3")) as db:
        assert {row[0] for row in db.execute("SELECT embedding_id FROM embeddings")} == {kept}
        keys = {row[0] for row in db.execute("SELECT DISTINCT key FROM embedding_metadata")}
        assert "fact_id" not in keys  # the shape the old check looked for is never written
    assert launcher.resurrection_counts(state)["fact"] == 0
    shutil.rmtree(state / "data" / "memory_db")
    shutil.copytree(older, state / "data" / "memory_db")  # the older index restored after the deletion
    assert launcher.resurrection_counts(state)["fact"] == 1


def test_a_pending_index_deletion_is_not_a_resurrection(tmp_path):
    """A crash before reconciliation leaves the embedding legitimately: index_jobs still holds it."""
    from src.memory.fact_ledger import MemoryReviewer
    from tests.mocks.memory_sources import memory_source
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    memory = _memory(state, delete_error=RuntimeError("synthetic index outage"))
    gone = memory.store_fact("Synthetic deleted fact", "fact", sources=(memory_source(),))
    assert memory.delete_fact(gone, MemoryReviewer("synthetic-reviewer")) is True
    asyncio.run(memory.aclose())
    with closing(sqlite3.connect(state / "data" / "privacy.db")) as db:
        assert db.execute("SELECT fact_id, operation FROM index_jobs").fetchall() == [(gone, "delete")]
    with closing(sqlite3.connect(state / "data" / "memory_db" / "chroma.sqlite3")) as db:
        assert [row[0] for row in db.execute("SELECT embedding_id FROM embeddings")] == [gone]
    assert launcher.resurrection_counts(state)["fact"] == 0
    with closing(sqlite3.connect(state / "data" / "privacy.db")) as db, db:
        db.execute("DELETE FROM index_jobs")  # acknowledged, yet the embedding is back: a resurrection
    assert launcher.resurrection_counts(state)["fact"] == 1


def test_a_pending_index_deletion_never_exempts_the_fact_row(tmp_path):
    """Found by U3b-0e4 (T-A5): the pending job exempts only the embedding; a deleted fact row turned
    live again is a resurrection whatever the index queue holds."""
    from src.memory.fact_ledger import MemoryReviewer
    from tests.mocks.memory_sources import memory_source
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    memory = _memory(state, delete_error=RuntimeError("synthetic index outage"))
    gone = memory.store_fact("Synthetic deleted fact", "fact", sources=(memory_source(),))
    assert memory.delete_fact(gone, MemoryReviewer("synthetic-reviewer")) is True
    asyncio.run(memory.aclose())
    assert launcher.resurrection_counts(state)["fact"] == 0
    with closing(sqlite3.connect(state / "data" / "privacy.db")) as db, db:
        assert db.execute("SELECT fact_id FROM index_jobs").fetchall() == [(gone,)]
        db.execute("UPDATE facts SET state='confirmed' WHERE id=?", (gone,))
    assert launcher.resurrection_counts(state)["fact"] == 1
    assert launcher.resurrection_detail(state)["fact"] == {"facts": 1, "semantic_index": 0}


def test_resurrection_without_any_ledger_counts_nothing(tmp_path):
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    with closing(sqlite3.connect(state / "data" / "events.db")) as db, db:
        db.execute("CREATE TABLE utterances (conversation_id TEXT)")
        db.execute("INSERT INTO utterances VALUES ('c')")
    assert launcher.resurrection_counts(state) == dict.fromkeys(launcher._RESURRECTION_KINDS, 0)


# --- child side: watchdog pings and the control pipe (process_lifecycle) -------------------------

def test_watchdog_pings_start_only_after_ready_and_stop_on_stop():
    read, write = os.pipe()
    os.set_blocking(read, False)
    notifier = lifecycle_module.ReadinessNotifier.from_environment({"KZA_NOTIFY_FD": str(write),
                                                                    "KZA_WATCHDOG_INTERVAL_USEC": "20000"})
    lifecycle = lifecycle_module.ProcessLifecycle(lifecycle_module.StopSignal(), notifier)

    async def scenario():
        await asyncio.sleep(0.08)
        before = os.read(read, 4096) if _readable(read) else b""
        lifecycle.ready()
        await asyncio.sleep(0.1)
        lifecycle.stop.request("test")
        after_ready = os.read(read, 4096)
        await asyncio.sleep(0.08)
        tail = os.read(read, 4096) if _readable(read) else b""
        return before, after_ready, tail

    before, after_ready, tail = asyncio.run(scenario())
    os.close(read)
    os.close(write)
    assert before == b""
    lines = after_ready.decode().split()
    assert lines[0] == "READY=1" and lines.count("WATCHDOG=1") >= 3 and lines[-1] == "STOPPING=1"
    assert b"WATCHDOG" not in tail


def _readable(descriptor):
    import select
    return bool(select.select([descriptor], [], [], 0)[0])


def test_control_channel_is_popped_non_inheritable_and_reads_fast():
    read, write = os.pipe()
    os.set_inheritable(read, True)
    environ = {"KZA_CONTROL_FD": str(read)}
    channel = lifecycle_module.ControlChannel.from_environment(environ)
    try:
        assert "KZA_CONTROL_FD" not in environ and os.get_inheritable(read) is False
        assert channel.fast_close() is False
        os.write(write, b"fast\n")
        assert channel.fast_close() is True and channel.fast_close() is True
    finally:
        os.close(read)
        os.close(write)
    assert lifecycle_module.ControlChannel.from_environment({"KZA_CONTROL_FD": "1"}).fast_close() is False
    assert lifecycle_module.ControlChannel.from_environment({}).fast_close() is False


# --- fake supervisor self-checks -----------------------------------------------------------------

def _shell_unit(script: str, **overrides) -> fake_systemd.UnitConfig:
    values = dict(exec_start=["/bin/sh", "-c", script], type="simple", restart="on-failure", restart_sec=0.05,
                  restart_prevent=frozenset({69, 78}), start_limit_interval=30, start_limit_burst=5)
    values.update(overrides)
    return fake_systemd.UnitConfig(**values)


@pytest.mark.parametrize("code, restarts", [(0, False), (68, True), (69, False), (70, True), (75, True), (78, False)])
def test_fake_supervisor_restart_policy(tmp_path, code, restarts):
    unit = fake_systemd.FakeSupervisor(_shell_unit(f"exit {code}", start_limit_burst=2), state_dir=tmp_path)
    try:
        unit.start()
        unit.wait_for(lambda item: len(item.exits) >= (2 if restarts else 1) or item.state == "failed", 10)
        unit.wait_inactive(10)
        assert (len(unit.events("start")) > 1) is restarts
        if restarts:
            assert unit.events("start_limit_hit")
    finally:
        unit.close()


def test_fake_supervisor_kill_mode_mixed_and_timeout_stop(tmp_path):
    script = "trap '' TERM; sleep 30 & wait"
    unit = fake_systemd.FakeSupervisor(_shell_unit(script, timeout_stop=0.5), state_dir=tmp_path)
    try:
        unit.start()
        time.sleep(0.2)
        code = unit.stop(wait=10)
        assert code == -signal.SIGKILL
        assert [event for _, event, _ in unit.log if event in {"sigterm_main", "sigkill_group"}][:2] == [
            "sigterm_main", "sigkill_group"]
    finally:
        unit.close()


def test_dropins_overlay_in_lexical_order(tmp_path):
    (tmp_path / "release.conf").write_text("[Service]\nExecStart=\nExecStart=/py -m src.main\n"
                                           "Environment=PYTHONPATH=/cur CONFIG_PATH=config/settings.yaml\n")
    (tmp_path / "zz.conf").write_text("[Service]\nTimeoutStopSec=20\nEnvironment=A=1\n")
    directives = fake_systemd.parse_dropins(fake_systemd.unit_dropins(tmp_path), {"ExecStart": ["/old"]})
    assert directives["ExecStart"] == ["/py -m src.main"]
    assert directives["Environment"] == ["PYTHONPATH=/cur", "CONFIG_PATH=config/settings.yaml", "A=1"]
    shown = fake_systemd.show_properties(directives)
    assert shown["TimeoutStopUSec"] == "20s" and "argv[]=/py -m src.main ;" in shown["ExecStart"]
