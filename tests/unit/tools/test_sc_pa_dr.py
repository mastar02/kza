"""Rows PA and DR of sc-release-design v4 (SC5B-Q1/Q2: implemented before day 14 of R1).

PA (mechanism.closed_witness, after_acceptance, R5-1): an accepted release that
dies on every boot without a receipt leaves an unknown head; R-b or adopt closes
it with an immutable recovery record. A clean record is a closed witness, so a
rollback or hotfix transition starts from that head (CR-2, PA-4). A record with
operator-confirmed regressions is a witness only for the H of a log planned with
``--accept-regressed-source <12+ hex of its sha256>``: the plan digest and the
intent bind the full sha256, and plan P1/P2, A1, A2's activate (its compaction
capability), W0/_h_intact, the snapshot (its recorded source_witness), the copy
catalog and the journal derivation admit it
only with that value; never K', never another record.

DR (mechanism.dry_run_shadow): adopt --dry-run copies the state only under the
lease, into <root>/admission/dry-run/<random id>/ (0700), never TMPDIR; the copy
is removed when the checks end, and a killed dry run's copy is removed by the
next lease holder right after the flock, before any other check.

Worlds are the first post-R1 transition of test_sc4b_release_exits (an epoch-4
head, T0 to X5, plan and start in release mode, V run by serve).
"""

from __future__ import annotations

import argparse
from contextlib import closing, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import stat
import subprocess
import sys
import time

import pytest

from scripts import compaction
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from src.memory import owner_compaction as oc
from src.memory import state_journal_derivation as derivation
from src.memory.state_snapshot import ClosedStateSnapshot, _json as snapshot_json, snapshot_closed_state
from tests.mocks import compaction_world as cw
from tests.mocks.epoch4 import epoch4_in_process
from tests.mocks import launcher_rehearsal as rehearsal
from tests.unit.tools.test_sc1b_release_plan import (  # noqa: F401
    _QUEUE, _install, _post_r1_world, _refusal, clone, x4, x5)
from tests.unit.tools.test_sc4b_release_exits import SMOKE, _scaled, _tombstone, _world

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold for an in-process controller."""
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)


# --- helpers ------------------------------------------------------------------------------------------

def _head(root: Path) -> dict:
    return stage.read_admission_head(root)


def _run_v(root: Path, tmp_path: Path, name: str):
    """Start the unit once: serve runs the armed V and stops it itself."""
    phase_file = tmp_path / ("phases-" + name)
    return rehearsal.run_serve(root, rehearsal.child_command(phase_file, "verify"), phase_file=phase_file,
                               stop_grace=200.0, timeout=60.0, clock=_scaled(20.0))


def _crash(root: Path, tmp_path: Path, name: str, *, execute: str | None = None) -> dict:
    """The accepted release dies after READY without a receipt (row PA): the head is left open.

    With ``execute`` the run first changes the state (e.g. a deletion-ledger append, a forward evolution), so the
    recovered head gets a new binding that no closure proof covers; without it R-b may re-adopt the
    binding the last clean close already proved.
    """
    phase_file = tmp_path / ("phases-" + name)
    script = None
    if execute is not None:
        script = tmp_path / ("exec-" + name + ".py")
        script.write_text(execute)
    result = rehearsal.run_serve(root, rehearsal.child_command(phase_file, "crash", seconds=0.2, execute=script),
                                 phase_file=phase_file, stop_when=None, timeout=60.0)
    head = _head(root)
    assert result.code in (68, 69) and head["status"] in {"unknown", "running"}, result.serve.events
    return head


def _transition(root: Path, x5: dict, tmp_path: Path, k: str, **options) -> str:
    """plan, start, arm V and run it; returns the compaction id (the log is 'verified')."""
    planned = cw.plan(root, x5["archive"], k, **options)
    started = cw.start(root, x5["archive"], k, planned=planned, **options)
    compaction_id = started["compaction_id"]
    assert compaction.compact_verify(root, compaction_id, smoke_seconds=SMOKE)["armed"] is True
    assert _run_v(root, tmp_path, "v-" + k).code == 0
    assert stage.read_compaction_log(root, compaction_id)["events"][-1]["kind"] == "verified"
    return compaction_id


def _retire(root: Path, compaction_id: str) -> dict:
    return compaction.compact_retire(root, compaction_id, scan=cw.no_units)


def _accepted(x4: dict, x5: dict, tmp_path: Path, *first: str) -> dict:
    """The first transition V'd and retired: NEW (X5) serves K' (the accepted release).

    ``first`` runs in H's first run (e.g. _QUEUE: queued audio rows whose paths K relocates, so a later
    K's authority differs from its H's and activate takes its compaction-capability branch).
    """
    world = _world(x4, x5, tmp_path, *first)
    root = world["root"]
    assert _run_v(root, tmp_path, "v").code == 0
    assert _retire(root, world["id"])["phase"] == "complete"
    world["state"] = Path(stage.read_binding(root, _head(root)["binding_id"])["state_root"])
    return world


def _rb(root: Path) -> dict:
    """Bounded R-b of the open head (what the next unit start runs before the child)."""
    with stage.AdmissionController(root, allow_open_head=True):
        decision = launcher.recover_open_head(root, _head(root), mode="bounded", boot_id="b", wall=time.time())
    assert decision.closed_head is not None, decision.code
    return decision.closed_head


def _regressed_head(world: dict, tmp_path: Path) -> dict:
    """The accepted release closes once with a tombstone, dies without a receipt, and the tombstone row
    set is older than that binding: R-b refuses, adopt closes it with the confirmed regression."""
    root = world["root"]
    assert cw.serve(root, tmp_path, "tombstone", execute=_tombstone("conv-pa")).code == 0
    _crash(root, tmp_path, "crash")
    with closing(sqlite3.connect(world["state"] / "data" / "privacy.db")) as db, db:
        db.execute("DELETE FROM conversation_tombstones")
    dry = launcher.adopt(root, confirm="", regressions=[], dry_run=True)
    families = sorted({item.split(":", 1)[0] for item in dry["report"]["regressions"]})
    assert families == ["conversation_tombstones"], dry["report"]
    with pytest.raises(launcher.LauncherRefusal, match="runtime_admission_recovery_confirmation"):
        launcher.adopt(root, confirm=dry["head_sha256"], regressions=[])
    adopted = launcher.adopt(root, confirm=dry["head_sha256"], regressions=families)
    head = _head(root)
    assert head == adopted["head"] and head["status"] == "closed"
    sha256 = adopted["recovery_record_sha256"]
    assert sha256 == stage.recovery_record_sha256(root, head) == stage._digest(
        stage._read_recovery_record(root, head))
    assert stage._read_recovery_record(root, head)["checks"]["regressed"] == families
    return {"head": head, "sha256": sha256}


# --- PA: closed_witness -------------------------------------------------------------------------------

def test_a_clean_recovery_head_after_acceptance_starts_a_rollback_transition(x4, x5, tmp_path):
    """Row PA, CR-2, PA-4 (test plan 'accepted release killed without receipt, R-b, rollback transition'):
    the clean recovery record is the closed witness at plan P1/P2, A1, the snapshot and the copy catalog
    (the worker), W0 and retire; the transition from that head completes and K'' serves. A regressed-source
    flag on that clean head is refused (it names no regressed record)."""
    world = _accepted(x4, x5, tmp_path)
    root = world["root"]
    _crash(root, tmp_path, "crash", execute=cw.ledger_append("fact_deleted", "fact-pa"))
    closed = _rb(root)
    assert closed == _head(root) and not (root / "admission" / "closures" / (closed["binding_id"] + ".json")).exists()
    record = stage._read_recovery_record(root, closed)
    assert record["checks"]["regressed"] == [] and record["mode"] == "automatic"
    with stage.AdmissionController(root):
        witness = stage.closed_witness(root, closed["binding_id"])
        assert witness == {"kind": "recovery", "record_sha256": stage._digest(record), "record": record}
        assert stage.verify_closed_binding(root, closed["binding_id"])["slot"] == "k"
    error = _refusal(compaction.REGRESSED_SOURCE_MISMATCH, lambda: cw.plan(
        root, x5["archive"], "k2", accept_regressed_source=stage._digest(record)[:12]))
    assert error.report == {"recovery_record_sha256": stage._digest(record)}
    compaction_id = _transition(root, x5, tmp_path, "k2")
    log = stage.read_compaction_log(root, compaction_id)
    assert "source_recovery_sha256" not in log["intent"] and log["intent"]["head"] == closed
    [snapshot] = [event for event in log["events"] if event["kind"] == "snapshot"]
    assert snapshot["data"]["manifest_sha256"]
    assert _retire(root, compaction_id)["phase"] == "complete"
    head = _head(root)
    assert stage.read_binding(root, head["binding_id"])["slot"] == "k2"
    assert cw.serve(root, tmp_path, "k2-serves").code == 0


def test_a_regressed_recovery_head_is_a_source_only_through_its_bound_sha256(x4, x5, tmp_path):
    """Row PA, R5-1 (test plan 'accepted release dying on every boot with a regressed R-b: adopt with
    confirmation, plan with --accept-regressed-source, V, retire succeeds'; 'recovery record with confirmed
    regressions refused as a witness'): plan refuses without the flag (naming the record's sha256 and its
    families), with a short, foreign or non-hex prefix; with the prefix the plan and the intent bind the
    full sha256; start needs the same flag. A V'd log returns to H losslessly through W0 (H's witness is
    the bound record), the next plan still needs the flag, and that transition retires; one more ordinary
    transition from the accepted head then needs no flag (R5V-2)."""
    world = _accepted(x4, x5, tmp_path)
    root = world["root"]
    regressed = _regressed_head(world, tmp_path)
    sha256, h_head = regressed["sha256"], regressed["head"]
    # The regression restored the state K' had before the tombstone exactly: an older closure proves that
    # binding, and the head's own regressed record still decides (the operator's bound sha256, R5-1).
    assert (root / "admission" / "closures" / (h_head["binding_id"] + ".json")).exists()
    with stage.AdmissionController(root):
        _refusal(stage.RECOVERY_REGRESSED_SOURCE, lambda: stage.verify_closed_binding(root, h_head["binding_id"]))
        _refusal(stage.RECOVERY_REGRESSED_SOURCE, lambda: stage.verify_closed_binding(
            root, h_head["binding_id"], regressed_source_sha256="0" * 64))
        assert stage.verify_closed_binding(root, h_head["binding_id"], regressed_source_sha256=sha256)
        with stage.regressed_source_bound(sha256):  # the worker's binding: that record only
            assert stage.closed_witness(root, h_head["binding_id"])["record_sha256"] == sha256
        with stage.regressed_source_bound("f" * 64):
            _refusal(stage.RECOVERY_REGRESSED_SOURCE, lambda: stage.closed_witness(root, h_head["binding_id"]))
        _refusal(stage.RECOVERY_REGRESSED_SOURCE, lambda: stage.closed_witness(root, h_head["binding_id"]))
    witness = cw.witness(root)
    error = _refusal(compaction.REGRESSED_SOURCE_REQUIRED, lambda: cw.plan(root, x5["archive"], "k2"))
    assert error.report == {"recovery_record_sha256": sha256, "regressed": ["conversation_tombstones"]}
    for prefix in (sha256[:11], ("0" if sha256[0] != "0" else "1") + sha256[1:12], sha256[:12].upper(),
                   sha256 + "0"):
        _refusal(compaction.REGRESSED_SOURCE_MISMATCH,
                 lambda: cw.plan(root, x5["archive"], "k2", accept_regressed_source=prefix))
    assert cw.witness(root) == witness  # a refused plan writes nothing
    planned = cw.plan(root, x5["archive"], "k2", accept_regressed_source=sha256[:12])
    assert planned["source_recovery_sha256"] == sha256
    assert planned["plan_sha256"] == cw.plan(root, x5["archive"], "k2", accept_regressed_source=sha256)["plan_sha256"]
    _refusal(compaction.REGRESSED_SOURCE_REQUIRED, lambda: cw.start(root, x5["archive"], "k2", planned=planned))
    assert cw.witness(root) == witness
    started = cw.start(root, x5["archive"], "k2", planned=planned, accept_regressed_source=sha256[:12])
    compaction_id = started["compaction_id"]
    log = stage.read_compaction_log(root, compaction_id)
    assert log["intent"]["source_recovery_sha256"] == sha256 and log["intent"]["head"] == h_head
    assert compaction.compact_verify(root, compaction_id, smoke_seconds=SMOKE)["armed"] is True
    assert _run_v(root, tmp_path, "v-k2").code == 0
    # W0 (H's witness after start is the intent's head and its bound record): a lossless return to H.
    returned = compaction.compact_abort(root, compaction_id, to_h=True, confirm_h=h_head["binding_id"][:12])
    assert returned["phase"] == "aborted"
    head = _head(root)
    assert head["binding_id"] == h_head["binding_id"] and head != h_head  # the return record's head
    assert stage.recovery_witness_head(root, head) == h_head
    _refusal(compaction.REGRESSED_SOURCE_REQUIRED, lambda: cw.plan(root, x5["archive"], "k3"))
    compaction_id = _transition(root, x5, tmp_path, "k3", accept_regressed_source=sha256[:16])
    assert _retire(root, compaction_id)["phase"] == "complete"
    assert stage.read_binding(root, _head(root)["binding_id"])["slot"] == "k3"
    assert cw.serve(root, tmp_path, "k3-serves").code == 0
    # R5V-2: the next ordinary transition from the accepted head needs no flag.
    _refusal(compaction.REGRESSED_SOURCE_MISMATCH,
             lambda: cw.plan(root, x5["archive"], "k4", accept_regressed_source=sha256[:12]))
    compaction_id = _transition(root, x5, tmp_path, "k4")
    assert _retire(root, compaction_id)["phase"] == "complete"


def test_a_regressed_source_activates_a_k_whose_authority_differs_through_the_logs_bound_sha256(
        x4, x5, tmp_path, monkeypatch):
    """Row PA, R5-1, review SCPADR-R1: with queued audio rows K's authority differs from H's (relocated
    queue paths), so A2's activate checks the compaction capability in the controller process, outside the
    worker's regressed_source_bound. H's witness there is the pending log's own bound record: the
    capability holds, K activates, V, retire, and K'' serves (it refused privacy_authority_stale)."""
    world = _accepted(x4, x5, tmp_path, _QUEUE)
    root = world["root"]
    regressed = _regressed_head(world, tmp_path)
    sha256, h_head = regressed["sha256"], regressed["head"]
    checked = []
    capability = stage._validate_compaction_capability

    def recorded(root_, head, source, candidate):
        result = capability(root_, head, source, candidate)
        checked.append((head["binding_id"], candidate["slot"], result))
        return result
    monkeypatch.setattr(stage, "_validate_compaction_capability", recorded)
    compaction_id = _transition(root, x5, tmp_path, "k2", accept_regressed_source=sha256[:12])
    assert checked == [(h_head["binding_id"], "k2", True)]
    assert stage.read_compaction_log(root, compaction_id)["intent"]["source_recovery_sha256"] == sha256
    assert _retire(root, compaction_id)["phase"] == "complete"
    assert stage.read_binding(root, _head(root)["binding_id"])["slot"] == "k2"
    assert cw.serve(root, tmp_path, "k2-serves").code == 0


def test_a_recovery_record_witnesses_only_the_head_it_closed(x4, x5, tmp_path):
    """closed_witness admits a recovery record only for the closed head whose binding it closed: another
    head with that binding (no record of its own, no return to H) has no witness, the closure proof is
    missing as before, and a bound sha256 does not change that; once a clean close moves the head on, its
    closure is the witness and an older recovery binding has none."""
    world = _accepted(x4, x5, tmp_path)
    root = world["root"]
    _crash(root, tmp_path, "crash", execute=cw.ledger_append("fact_deleted", "fact-pa"))
    closed = _rb(root)
    sha256 = stage.recovery_record_sha256(root, closed)
    other = {**closed, "generation": closed["generation"] + 9}
    with stage.AdmissionController(root):
        assert stage.closed_witness(root, closed["binding_id"])["record_sha256"] == sha256
        with pytest.raises(FileNotFoundError, match="admission/closures"):
            stage.closed_witness(root, closed["binding_id"], head=other, regressed_source_sha256=sha256)
    assert cw.serve(root, tmp_path, "clean-close").code == 0  # a later clean close: a closure head
    head = _head(root)
    with stage.AdmissionController(root):
        assert stage.closed_witness(root, head["binding_id"])["kind"] == "closure"
        if head["binding_id"] != closed["binding_id"]:
            with pytest.raises(FileNotFoundError, match="admission/closures"):
                stage.closed_witness(root, closed["binding_id"], regressed_source_sha256=sha256)


def test_journal_derivation_requires_the_snapshot_witness_to_be_the_source_witness(x4, x5, tmp_path):
    """Row PA: the snapshot records source_witness (here the clean recovery record); the derivation re-derives
    it and refuses a manifest whose recorded witness differs."""
    world = _accepted(x4, x5, tmp_path)
    root = world["root"]
    _crash(root, tmp_path, "crash", execute=cw.ledger_append("fact_deleted", "fact-pa"))
    closed = _rb(root)
    privacy = compaction._privacy_relative(world["state"])
    destination = tmp_path.resolve() / "snapshot"
    with stage.AdmissionController(root) as controller:
        snapshot = snapshot_closed_state(controller, closed["binding_id"], destination,
                                         chroma_collections=oc._snapshot_class)
        recorded = snapshot.manifest["source_witness"]
        assert recorded["kind"] == "recovery" and recorded["record_sha256"] == stage.recovery_record_sha256(
            root, closed)
        assert "source_closure" not in snapshot.manifest
        source = derivation._source(stage, root, snapshot, privacy, latest=True)
        assert source["closure"] == {"recovery": {"record_sha256": recorded["record_sha256"],
                                                  "record": recorded["record"]}}
        forged = dict(snapshot.manifest, source_witness=dict(recorded, record_sha256="0" * 64))
        path = destination / "manifest.json"
        os.chmod(path, 0o600)
        path.write_bytes(snapshot_json(forged) + b"\n")
        changed = ClosedStateSnapshot(destination, closed["binding_id"], snapshot.source_binding, forged)
        with pytest.raises(Exception, match="source_closure"):
            derivation._source(stage, root, changed, privacy, latest=True)


def test_the_regressed_source_flag_is_release_mode_only(x4, tmp_path):
    """Row PA, review SCPADR-R2 (MB): in plain mode (an epoch-4 controller and root) plan with
    --accept-regressed-source refuses runtime_admission_transition_requires_release_mode, before the prefix
    is compared with any record, and writes nothing."""
    with epoch4_in_process():
        world = cw.world(x4["archive"], tmp_path, hops=())
        root = world["root"]
        assert compaction.release_mode() is False
        witness = cw.witness(root)
        for prefix in ("0" * 12, stage.recovery_record_sha256(root, _head(root)) or "f" * 64):
            _refusal(stage.TRANSITION_REQUIRES_RELEASE_MODE,
                     lambda: cw.plan(root, x4["archive"], "k2", accept_regressed_source=prefix))
        assert cw.witness(root) == witness



def _operator(root: Path, *argv: str) -> tuple[int, dict]:
    """One operator command through stage_runtime's parser (its launcher and compact sub-commands), as the
    rehearsals type it; returns the exit code and the JSON it printed (a refusal included)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    launcher.add_commands(commands)
    compaction.add_commands(commands)
    args = parser.parse_args(["--root", str(root), "--offline", *argv])
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = (compaction.run_command(args, scan=cw.no_units) if args.command == "compact"
                else launcher.run_command(args))
    [line] = (out.getvalue() or err.getvalue()).strip().splitlines()
    return code, json.loads(line)


def test_a_head_adopted_with_confirmed_regressions_is_refused_by_plan_and_start_without_the_flag(
        x4, x5, tmp_path):
    """Row PA, R5-1, SC-6b finding SC6B-A-12 (plan admitted a recovery-closed head because an older closure
    proved the same binding): the operator's adopt --dry-run, adopt --confirm with the confirmed regression
    closes the head with the binding the last clean close proved; compact plan without
    --accept-regressed-source refuses (78) naming the record's sha256 and its families, and so does compact
    start with a plan digest made with the flag; neither writes anything. With the flag the plan binds the
    full sha256."""
    world = _accepted(x4, x5, tmp_path)
    root = world["root"]
    assert cw.serve(root, tmp_path, "tombstone", execute=_tombstone("conv-a12")).code == 0
    _crash(root, tmp_path, "crash")
    with closing(sqlite3.connect(world["state"] / "data" / "privacy.db")) as db, db:
        db.execute("DELETE FROM conversation_tombstones")
    code, dry = _operator(root, "adopt", "--dry-run")
    assert code == 0 and sorted({item.split(":", 1)[0] for item in dry["report"]["regressions"]}) == [
        "conversation_tombstones"], dry
    code, adopted = _operator(root, "adopt", "--confirm", dry["head_sha256"],
                              "--confirm-regressions", "conversation_tombstones")
    assert code == 0, adopted
    head, sha256 = _head(root), adopted["recovery_record_sha256"]
    assert head["status"] == "closed" and sha256 == stage.recovery_record_sha256(root, head)
    assert stage._read_recovery_record(root, head)["checks"]["regressed"] == ["conversation_tombstones"]
    # What SC6B-A-12 admitted through: an older clean closure proves this very binding.
    assert (root / "admission" / "closures" / (head["binding_id"] + ".json")).exists()
    witness = cw.witness(root)
    code, refused = _operator(root, "compact", "plan", "k2", "--archive", str(x5["archive"]))
    assert (code, refused) == (78, {"refused": compaction.REGRESSED_SOURCE_REQUIRED, "report": {
        "recovery_record_sha256": sha256, "regressed": ["conversation_tombstones"]}})
    assert cw.witness(root) == witness
    code, planned = _operator(root, "compact", "plan", "k2", "--archive", str(x5["archive"]),
                              "--accept-regressed-source", sha256[:12])
    assert code == 0 and planned["source_recovery_sha256"] == sha256, planned
    code, refused = _operator(root, "compact", "start", "k2", "--archive", str(x5["archive"]),
                              "--plan-sha256", planned["plan_sha256"],
                              "--acknowledge-untouched", planned["untouched_sha256"], "--destroy-rollback-states")
    assert (code, refused["refused"]) == (78, compaction.REGRESSED_SOURCE_REQUIRED), refused
    assert cw.witness(root) == witness
    assert stage.read_admission_head(root) == head


def test_a_head_adopted_clean_is_a_source_through_its_recovery_record_without_the_flag(x4, x5, tmp_path):
    """Row PA (SC6B-A-12's clean case): a killed accepted release adopted by the operator with no
    regression (adopt --dry-run, adopt --confirm) is closed by a clean recovery record with the binding
    the last clean close proved; that record is the witness, compact plan admits it without the flag, and
    the flag is refused (the record names no regression)."""
    world = _accepted(x4, x5, tmp_path)
    root = world["root"]
    _crash(root, tmp_path, "crash")
    code, dry = _operator(root, "adopt", "--dry-run")
    assert code == 0 and dry["report"]["regressions"] == [], dry
    code, adopted = _operator(root, "adopt", "--confirm", dry["head_sha256"])
    assert code == 0, adopted
    head = _head(root)
    record = stage._read_recovery_record(root, head)
    assert head["status"] == "closed" and record["checks"]["regressed"] == []
    assert (root / "admission" / "closures" / (head["binding_id"] + ".json")).exists()
    with stage.AdmissionController(root):
        assert stage.closed_witness(root, head["binding_id"]) == {
            "kind": "recovery", "record_sha256": stage._digest(record), "record": record}
    code, planned = _operator(root, "compact", "plan", "k2", "--archive", str(x5["archive"]))
    assert code == 0 and "source_recovery_sha256" not in planned, planned
    code, refused = _operator(root, "compact", "plan", "k2", "--archive", str(x5["archive"]),
                              "--accept-regressed-source", stage._digest(record)[:12])
    assert (code, refused["refused"]) == (78, compaction.REGRESSED_SOURCE_MISMATCH), refused

# --- DR: the dry-run shadow ----------------------------------------------------------------------------

KILLED_DRY_RUN = """
import os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts import managed_launcher as launcher
copied = []
wanted = launcher._shadow_wanted
def kill_mid_copy(relative):
    if not wanted(relative):
        return False
    copied.append(relative)
    if len(copied) == 3:  # two files copied, the third not: a dry run killed mid-copy
        os.kill(os.getpid(), signal.SIGKILL)
    return True
launcher._shadow_wanted = kill_mid_copy
launcher.adopt(Path(sys.argv[2]), confirm="", regressions=[], dry_run=True)
"""


def _killed_dry_run(root: Path, tmpdir: Path) -> Path:
    environment = {key: value for key, value in os.environ.items() if key not in {"TMPDIR", "SQLITE_TMPDIR"}}
    environment["TMPDIR"] = str(tmpdir)
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", KILLED_DRY_RUN, str(REPO), str(root)],
                               env=environment, capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == -signal.SIGKILL, completed.stderr[-2000:]
    shadows = list(stage.dry_run_directory(root).iterdir())
    assert len(shadows) == 1 and len(shadows[0].name) == 32
    assert stat.S_IMODE(os.lstat(shadows[0]).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(stage.dry_run_directory(root)).st_mode) == 0o700
    assert [path for path in (shadows[0] / "state").rglob("*") if path.is_file()]
    assert list(tmpdir.iterdir()) == []  # nothing in TMPDIR
    return shadows[0]


def test_a_killed_dry_run_leaves_its_shadow_only_under_the_root_until_the_next_lease(x4, x5, tmp_path):
    """Row DR, R3-PRIV-2 (test plan 'adopt --dry-run killed mid-copy: the shadow is under
    <root>/admission/dry-run/ and the next serve start removes it before anything else, nothing in
    TMPDIR'): a lease entry that is then refused (open head) has already removed it; a second killed dry
    run's shadow is removed by the next serve, which then recovers the head and serves."""
    world = _post_r1_world(x4, x5, tmp_path)
    root = world["root"]
    _crash(root, tmp_path, "crash")
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    _killed_dry_run(root, tmpdir)
    with pytest.raises(ValueError, match="^runtime_admission_not_closed$"):
        with stage.AdmissionController(root):
            pass
    assert not os.path.lexists(stage.dry_run_directory(root))
    _killed_dry_run(root, tmpdir)
    assert cw.serve(root, tmp_path, "after-kill").code == 0
    assert not os.path.lexists(stage.dry_run_directory(root))
    assert _head(root)["status"] == "closed" and list(tmpdir.iterdir()) == []


def test_a_finished_dry_run_checks_a_copy_under_the_root_and_leaves_nothing(x4, x5, tmp_path, monkeypatch):
    """Row DR: the checks read <root>/admission/dry-run/<32 hex>/state (0700, created under the lease);
    afterwards nothing is left there or in TMPDIR. Without the lease the dry-run checks refuse first."""
    world = _post_r1_world(x4, x5, tmp_path)
    root = world["root"]
    head = _crash(root, tmp_path, "crash")
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    monkeypatch.setattr("tempfile.tempdir", str(tmpdir))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    seen = []
    checks = launcher._recovery_checks

    def recorded(state, checked, binding, report, step):
        seen.append((Path(checked), stat.S_IMODE(os.lstat(Path(checked).parent).st_mode),
                     sorted(path.name for path in Path(checked).iterdir())))
        return checks(state, checked, binding, report, step)
    monkeypatch.setattr(launcher, "_recovery_checks", recorded)
    dry = launcher.adopt(root, confirm="", regressions=[], dry_run=True)
    assert dry["report"]["failures"] == [] and dry["report"]["databases"] >= 1
    [(checked, mode, names)] = seen
    assert checked.parent.parent == stage.dry_run_directory(root) and len(checked.parent.name) == 32
    assert checked.name == "state" and mode == 0o700 and names
    assert not os.path.lexists(stage.dry_run_directory(root)) and list(tmpdir.iterdir()) == []
    lease = root / ".admission.lock"  # another controller's lease header: this process holds no lease
    os.chmod(lease, 0o600)
    lease.write_text('{"controller_pid":1,"lease_fd":0}')
    with pytest.raises(ValueError, match="^runtime_admission_controller_lease_required$"):
        launcher.recovery_report(root, head, read_only=True)
    assert not os.path.lexists(stage.dry_run_directory(root)) and len(seen) == 1


def _leftover(root: Path, outside: Path) -> Path:
    """A dry run's leftover under <root>/admission/dry-run/ (a copied file and a link outside)."""
    shadow = stage.dry_run_directory(root)
    (shadow / "a" / "state").mkdir(parents=True)
    (shadow / "a" / "state" / "privacy.db").write_bytes(b"copy")
    (shadow / "a" / "link").symlink_to(outside)
    return shadow


def test_install_controller_removes_a_killed_dry_runs_copy_at_its_lease_even_when_it_refuses(x4, x5, tmp_path):
    """Row DR, review SCPADR-R2 (MA): install-controller (gate 'install', named in the runbook) removes a
    killed dry run's copy right after its flock: T0 over an unknown head refuses runtime_admission_not_closed
    under the lease and the copy is gone; T0 itself (the first install over an epoch-4 head) removes a
    leftover and installs."""
    world = _post_r1_world(x4, x5, tmp_path)
    root = world["root"]
    _crash(root, tmp_path, "crash")
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    _killed_dry_run(root, tmpdir)
    with pytest.raises(launcher.LauncherRefusal, match="^runtime_admission_not_closed$"):
        _install(root, x5)
    assert not os.path.lexists(stage.dry_run_directory(root)) and list(tmpdir.iterdir()) == []
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("kept")
    (tmp_path / "t0").mkdir()
    with epoch4_in_process():
        first = cw.world(x4["archive"], tmp_path / "t0", hops=())
    shadow = _leftover(first["root"], outside)
    _install(first["root"], x5)
    assert stage.installed_controller_triple(first["root"]) == x5["triple"]
    assert not os.path.lexists(shadow) and (outside / "keep.txt").read_text() == "kept"


def test_a_compaction_gate_removes_a_dry_run_leftover_at_its_lease_even_when_it_refuses(x4, x5, tmp_path):
    """Row DR, review SCPADR-R2 (ME): under a pending release-mode log the 'compaction:<id>' gate removes a
    leftover right after its flock: retire before V refuses runtime_admission_compaction_not_verified under
    the lease and the leftover is gone; after V, retire removes another and completes."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("kept")
    shadow = _leftover(root, outside)
    _refusal("runtime_admission_compaction_not_verified", lambda: _retire(root, world["id"]))
    assert not os.path.lexists(shadow)
    assert _run_v(root, tmp_path, "v").code == 0
    shadow = _leftover(root, outside)
    assert _retire(root, world["id"])["phase"] == "complete"
    assert not os.path.lexists(shadow) and (outside / "keep.txt").read_text() == "kept"


def test_the_lease_removes_any_dry_run_leftover_without_following_links(tmp_path):
    """Row DR: whatever is under admission/dry-run/ goes at the flock (a file, a symlink to a directory
    outside, whose content stays); a symlinked admission/ refuses DRY_RUN_SHADOW_UNSAFE first."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("kept")
    shadow = stage.dry_run_directory(root)
    (shadow / "a" / "state").mkdir(parents=True)
    (shadow / "a" / "state" / "privacy.db").write_bytes(b"copy")
    (shadow / "a" / "link").symlink_to(outside)
    with stage.AdmissionController(root):
        assert not os.path.lexists(shadow)
    shadow.symlink_to(outside)
    with stage.AdmissionController(root):
        assert not os.path.lexists(shadow)
    assert (outside / "keep.txt").read_text() == "kept"
    os.rename(root / "admission", tmp_path / "real-admission")
    (root / "admission").symlink_to(tmp_path / "real-admission")
    with pytest.raises(ValueError, match="^" + stage.DRY_RUN_SHADOW_UNSAFE + "$"):
        with stage.AdmissionController(root):
            pass
