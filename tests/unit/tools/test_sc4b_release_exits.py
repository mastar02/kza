"""SC-4b: every outcome of the release-mode verification run V has an exit (crash_table rows).

sc-release-design v4 mechanism.w_abort_predicate (abort --to-h with W, the
recheck), mechanism.retire (closed head, verified, W0 and W1 exactly equal,
the resurrection scan), mechanism.safe_stop (SS and its SC-D8 A exit with
--discard-k-changes, the terminal HD), R5-PRIV-3 (adopt refused under a
pending release-mode log) and the status view (armed run consumed or not,
verified, 'verify branch reached'). The rows: A0, V1-V4, AT, AB, SS, HD,
RS, RX, C. Worlds are the first post-R1 transition (test_sc1b_release_plan):
an epoch-4 head (R1's stand-in), T0 to X5, plan and start in release mode,
V run by serve with the rehearsal child as NEW; ``--exec`` makes that NEW
change K' at boot as a defective NEW (or an isolation breach) would.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import time

import pytest

from scripts import compaction
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from tests.mocks import compaction_world as cw
from tests.mocks import launcher_rehearsal as rehearsal
from tests.mocks.epoch4 import epoch4_in_process
from tests.mocks.sc4b_states import CANARY
from tests.unit.tools.test_sc1b_release_plan import _install, _refusal, clone, x4, x5  # noqa: F401

SMOKE = 10


@pytest.fixture(autouse=True)
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold for an in-process controller."""
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)


def _world(x4: dict, x5: dict, tmp_path: Path, *first: str) -> dict:
    """The first post-R1 transition: an epoch-4 head (``first`` run in its first run), T0 to X5,
    then plan and start in release mode and arm V."""
    with epoch4_in_process():
        world = cw.world(x4["archive"], tmp_path, hops=(), first=first)
    root = world["root"]
    _install(root, x5)
    world["h_head"] = stage.read_admission_head(root)
    world["h_binding"] = world["h_head"]["binding_id"]
    world["h_tree"] = cw.tree(world["h"]["state"])
    started = cw.start(root, x5["archive"], planned=cw.plan(root, x5["archive"]))
    assert started["next"] == "compact verify " + started["compaction_id"]  # SC4A-N13 gate lifted
    world["id"] = started["compaction_id"]
    assert compaction.compact_verify(root, world["id"], smoke_seconds=SMOKE)["armed"] is True
    return world


def _scaled(factor: float):
    origin = time.monotonic()
    return lambda: origin + (time.monotonic() - origin) * factor


def _v(world: dict, tmp_path: Path, name: str, *, scenario: str = "verify", execute: str | None = None):
    """Start the unit once: serve runs the armed V and stops it itself (N on a scaled monotonic clock)."""
    phase_file = tmp_path / ("phases-" + name)
    script = None
    if execute is not None:
        script = tmp_path / ("exec-" + name + ".py")
        script.write_text(execute)
    return rehearsal.run_serve(world["root"], rehearsal.child_command(phase_file, scenario, execute=script),
                               phase_file=phase_file, stop_grace=200.0, timeout=60.0, clock=_scaled(20.0))


def _log(world: dict) -> dict:
    return stage.read_compaction_log(world["root"], world["id"])


def _kinds(world: dict) -> list[str]:
    return [event["kind"] for event in _log(world)["events"]]


def _status(world: dict) -> dict:
    [status] = compaction.compact_status(world["root"], world["id"])["compactions"]
    return status


def _abort(world: dict, **options) -> dict:
    return compaction.compact_abort(world["root"], world["id"], to_h=True, confirm_h=world["h_binding"][:12],
                                    **options)


def _retire(world: dict) -> dict:
    return compaction.compact_retire(world["root"], world["id"], scan=cw.no_units)


def _refused_writes_nothing(world: dict, code: str, action):
    """A refusal (transient, SS, HD, RX) writes nothing: the root is byte-identical afterwards."""
    witness = cw.witness(world["root"])
    error = _refusal(code, action)
    assert cw.witness(world["root"]) == witness
    return error


def _back_on_h(world: dict, tmp_path: Path, name: str) -> None:
    """Row exits that return to H: H's head and state as before, and the unit serves H."""
    root = world["root"]
    head = stage.read_admission_head(root)
    assert head["status"] == "closed" and head["binding_id"] == world["h_binding"]
    assert cw.tree(world["h"]["state"]) == world["h_tree"]
    assert not (root / "slots" / "k").exists()
    assert _log(world)["terminal"] and _kinds(world)[-1] == "aborted"
    assert cw.serve(root, tmp_path, name).code == 0


def _no_canary(world: dict) -> None:
    for path in (world["root"] / "compaction").rglob("*.json"):
        assert CANARY not in path.read_text()


def _tombstone(conversation: str) -> str:
    return cw.sql(("INSERT INTO conversation_tombstones VALUES (?, 1.0, ?)", (conversation, CANARY)))


# --- V4 -> retire -> C ------------------------------------------------------------------------------

def test_a_verified_equal_v_is_retired_and_k_prime_serves(x4, x5, tmp_path):
    """Rows A0 (status), V4, C: retire with strict W equality keeps K', destroys H; then K' serves."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    assert {key: _status(world)[key] for key in ("armed_run", "verified", "verify_branch_reached")} == {
        "armed_run": "not consumed", "verified": False, "verify_branch_reached": "not run"}
    assert _v(world, tmp_path, "v").code == 0
    assert _kinds(world)[-1] == "verified"
    assert {key: _status(world)[key] for key in ("armed_run", "verified", "verify_branch_reached")} == {
        "armed_run": "consumed", "verified": True, "verify_branch_reached": "yes"}
    retired = _retire(world)
    assert retired["phase"] == "complete"
    [started] = [event for event in _log(world)["events"] if event["kind"] == "retire_started"]
    assert started["data"]["w"]["class"] == compaction.W_EQUAL and len(started["data"]["w"]["sha256"]) == 64
    head = stage.read_admission_head(root)
    assert stage.read_binding(root, head["binding_id"])["slot"] == "k"
    assert not world["h"]["state"].exists()
    assert x5["release_sha256"] in compaction.served_releases(root, stage.read_binding(root, head["binding_id"]))
    assert cw.serve(root, tmp_path, "k-prime").code == 0  # C: enable and start the unit


# --- SS: a restrictive change during V, and its SC-D8 A exit -----------------------------------------

def test_a_tombstone_written_during_v_stops_safely_until_the_discard(x4, x5, tmp_path):
    """Rows V4, RX, SS: NEW writes a tombstone at boot. Retire refuses, abort --to-h refuses with the
    content-free W report (same digest each time), adopt and the unit are refused; a wrong digest is
    refused; the right one returns to H and records the counts and digest in 'abort_started'."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    assert _v(world, tmp_path, "v", execute=_tombstone("conv-" + CANARY)).code == 0
    assert _kinds(world)[-1] == "verified"
    retired = _refused_writes_nothing(world, compaction.RETIRE_PROTECTED_CHANGED, lambda: _retire(world))
    assert retired.report["w"]["class"] == compaction.W_RESTRICTIVE
    first = _refused_writes_nothing(world, compaction.ABORT_PROTECTED_CHANGED, lambda: _abort(world))
    second = _refused_writes_nothing(world, compaction.ABORT_PROTECTED_CHANGED, lambda: _abort(world))
    digest = first.report["w_sha256"]
    assert second.report == first.report and retired.report["w_sha256"] == digest  # no clock input
    report = first.report["w"]
    assert report["verify_branch_reached"] == "yes"
    assert report["tables"]["data/privacy.db"]["conversation_tombstones"] == {"added": 1, "removed": 0}
    assert CANARY not in json.dumps(first.report)
    status = _status(world)
    assert status["armed_run"] == "consumed" and status["verified"] is True
    assert not {"w", "w_sha256", "class", "report"} & set(status)  # status does not compute W
    # R5-PRIV-3: no adoption of K' and no dry-run copy under the pending log; serve refuses (unit down).
    for dry_run in (True, False):
        _refused_writes_nothing(world, "runtime_admission_compaction_pending",
                                lambda: launcher.adopt(root, confirm="", regressions=[], dry_run=dry_run))
    assert cw.serve(root, tmp_path, "refused").code == 78
    _refused_writes_nothing(world, compaction.DISCARD_MISMATCH, lambda: _abort(world, discard_k_changes="0" * 12))
    _refused_writes_nothing(world, compaction.DISCARD_MISMATCH, lambda: _abort(world, discard_k_changes=digest[:11]))
    _refused_writes_nothing(world, compaction.DISCARD_MISMATCH, lambda: compaction.compact_abort(
        root, world["id"], discard_k_changes=digest[:12]))  # a discard belongs to the return to H
    _refused_writes_nothing(world, "runtime_admission_compaction_abort_after_run",
                            lambda: compaction.compact_abort(root, world["id"]))  # no plain abort after V
    assert _abort(world, discard_k_changes=digest[:12])["phase"] == "aborted"
    [started] = [event for event in _log(world)["events"] if event["kind"] == "abort_started"]
    assert started["data"]["w"] == {"class": compaction.W_RESTRICTIVE, "sha256": digest}
    assert started["data"]["discarded"]["w_sha256"] == digest
    assert started["data"]["discarded"]["counts"]["tables"] == {
        "data/privacy.db": {"conversation_tombstones": {"added": 1, "removed": 0}}}
    _no_canary(world)
    _back_on_h(world, tmp_path, "h")


# --- loss-only changes: retire refuses, abort returns to H (V4R2-P1, V4R2-P2, CR2-2) ----------------

_H_ROWS = cw.sql(
    ("INSERT INTO conversation_tombstones VALUES ('conv-h', 1.0, 'owner')", ()),
    ("INSERT INTO facts VALUES ('fact-h', 'x', 'c', 0.5, 'confirmed', 1.0, NULL, 1, 1.0, 'owner')", ()),
    ("INSERT INTO fact_owner_revisions VALUES ('fact-h', 1, 'assigned', 0, 1.0)", ()),
)


@pytest.mark.parametrize("table", ["conversation_tombstones", "fact_owner_revisions"])
def test_a_loss_only_change_is_never_retired_and_abort_returns_to_h(x4, x5, tmp_path, table):
    """NEW removes a tombstone or a fact_owner_revisions row at boot: W1 'loss_only'. Retire refuses
    (exact equality); abort --to-h proceeds, H keeps the row and serves."""
    world = _world(x4, x5, tmp_path, _H_ROWS)
    assert _v(world, tmp_path, "v", execute=cw.sql((f"DELETE FROM {table}", ()))).code == 0
    retired = _refused_writes_nothing(world, compaction.RETIRE_PROTECTED_CHANGED, lambda: _retire(world))
    assert retired.report["w"]["class"] == compaction.W_LOSS_ONLY
    assert retired.report["w"]["tables"]["data/privacy.db"][table] == {"added": 0, "removed": 1}
    assert _abort(world)["phase"] == "aborted"
    [started] = [event for event in _log(world)["events"] if event["kind"] == "abort_started"]
    assert started["data"]["w"]["class"] == compaction.W_LOSS_ONLY and "discarded" not in started["data"]
    _back_on_h(world, tmp_path, "h")


# --- V3: sealed, not verified -----------------------------------------------------------------------

def test_a_run_without_the_acknowledgement_is_aborted_through_w(x4, x5, tmp_path):
    """Row V3 (R5-PRIV-2): READY without VERIFY; serve stops it, sealed and never verified. Retire
    refuses not_verified; abort --to-h with W ('equal') returns to H; the header shows 'no'."""
    world = _world(x4, x5, tmp_path)
    assert _v(world, tmp_path, "v", scenario="verify_ignored").code == 0
    assert stage.read_admission_head(world["root"])["status"] == "closed" and "verified" not in _kinds(world)
    _refusal("runtime_admission_compaction_not_verified", lambda: _retire(world))
    assert _status(world)["verify_branch_reached"] == "no"
    assert compaction._w(world["root"], _log(world))["report"]["verify_branch_reached"] == "no"
    assert _abort(world)["phase"] == "aborted"
    _back_on_h(world, tmp_path, "h")


# --- V2: killed without receipt inside a write transaction (CR-3), status, adopt -------------------

_KILLED_IN_TRANSACTION = "\n".join([
    "import os, sqlite3",
    "db = sqlite3.connect(state / 'data' / 'privacy.db', isolation_level=None)",
    "db.execute('PRAGMA journal_mode=DELETE')",
    "db.execute('PRAGMA cache_size=1')",
    "db.execute('BEGIN IMMEDIATE')",
    "for index in range(400):",
    "    db.execute('INSERT INTO conversation_tombstones VALUES (?, 1.0, ?)', ('kill-%d' % index, 'x' * 200))",
    "os._exit(9)",
])


def test_v_killed_inside_a_write_transaction_is_aborted_losslessly(x4, x5, tmp_path, monkeypatch):
    """Rows V2, CR-3, R3-OPS-4's status, R5-PRIV-3: the child dies inside a write transaction before
    READY (no receipt). status returns without error and without W; adopt and adopt --dry-run refuse
    and copy nothing; abort --to-h recovers the hot journal in place, W1 'equal', H serves."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    result = _v(world, tmp_path, "v", scenario="verify", execute=_KILLED_IN_TRANSACTION)
    assert result.code != 0 and stage.read_admission_head(root)["status"] != "closed"
    journal = root / "slots" / "k" / "state" / "data" / "privacy.db-journal"
    assert journal.exists() and journal.stat().st_size > 0
    witness = cw.witness(root)
    status = _status(world)
    assert status["armed_run"] == "consumed" and status["verified"] is False and "invalid" not in status
    assert cw.witness(root) == witness  # status never writes
    shadows = tmp_path / "tmpdir"
    shadows.mkdir()
    monkeypatch.setenv("TMPDIR", str(shadows))
    for dry_run in (True, False):
        _refused_writes_nothing(world, "runtime_admission_compaction_pending",
                                lambda: launcher.adopt(root, confirm="", regressions=[], dry_run=dry_run))
    assert list(shadows.iterdir()) == []
    monkeypatch.delenv("TMPDIR")
    assert _abort(world)["phase"] == "aborted"
    [started] = [event for event in _log(world)["events"] if event["kind"] == "abort_started"]
    assert started["data"]["w"]["class"] == compaction.W_EQUAL
    _back_on_h(world, tmp_path, "h")


# --- A0: armed, run not consumed (R3-OPS-4) --------------------------------------------------------

def test_serve_dying_between_verify_started_and_begin_run_leaves_the_run_unconsumed(x4, x5, tmp_path):
    """Row A0: verify_started logged, begin_run's CAS never made. status 'not consumed'; abort --to-h
    refuses (nothing ran); starting the unit runs V; retire completes."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    head = stage.read_admission_head(root)
    stage.record_compaction_run(root, _log(world), "verify", head, wall=time.time())  # serve dies after this
    assert _status(world)["armed_run"] == "not consumed"
    _refused_writes_nothing(world, "runtime_admission_compaction_abort_to_h_before_run", lambda: _abort(world))
    assert _v(world, tmp_path, "v").code == 0 and _kinds(world)[-1] == "verified"
    assert _retire(world)["phase"] == "complete"


# --- HD: H drift ------------------------------------------------------------------------------------

def test_h_drift_is_a_terminal_stop_for_every_command(x4, x5, tmp_path):
    """Row HD (CR2-5, V4R2-P3, R3-OPS-1, R4-OPS-10b): H's closure proof deleted after V. abort --to-h,
    plain abort, --discard-k-changes, retire, adopt and L4's uninstall all refuse and write nothing
    ('verified' was already appended by serve)."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    assert _v(world, tmp_path, "v").code == 0 and _kinds(world)[-1] == "verified"
    (root / "admission" / "closures" / (world["h_binding"] + ".json")).unlink()
    error = _refused_writes_nothing(world, compaction.H_CHANGED, lambda: _abort(world))
    assert error.report == {"w0": "drift", "code": "errno:ENOENT"}
    _refused_writes_nothing(world, compaction.H_CHANGED, lambda: _abort(world, discard_k_changes="0" * 12))
    _refused_writes_nothing(world, "runtime_admission_compaction_abort_after_run",
                            lambda: compaction.compact_abort(root, world["id"]))
    _refused_writes_nothing(world, compaction.H_CHANGED, lambda: _retire(world))
    _refused_writes_nothing(world, "runtime_admission_compaction_pending",
                            lambda: launcher.adopt(root, confirm="", regressions=[], dry_run=True))
    manager = rehearsal.FakeManager(properties={"ActiveState": "inactive"})
    _refused_writes_nothing(world, "runtime_admission_compaction_pending", lambda: launcher.uninstall_launcher(
        root, unit_dir=tmp_path, decision_id="d-123", phrase=launcher.UNINSTALL_PHRASE, legacy_slot="a",
        manager=manager))
    assert _status(world)["phase"] == "verified"


# --- AT: transient refusals, then the exit; RS ------------------------------------------------------

def test_transient_refusals_write_nothing_and_clear_with_their_cause(x4, x5, tmp_path, monkeypatch):
    """Row AT (R4-PRIV-1, R5-2, R3-OPS-5, R4-OPS-3): chromadb version changed (abort and retire refuse
    sdk_changed naming both versions), EIO reading H, a stray process holding a lock in H. Each writes
    nothing; once the cause is gone, retire runs; a cut after retire_started resumes (row RS)."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    assert _v(world, tmp_path, "v").code == 0 and _kinds(world)[-1] == "verified"
    layout = stage._state_layout

    def other_sdk(state):
        roots, schemas, _sdk = layout(state)
        return roots, schemas, {"chromadb": "9.9.9"}

    monkeypatch.setattr(stage, "_state_layout", other_sdk)
    for action in (lambda: _abort(world), lambda: _retire(world)):
        error = _refused_writes_nothing(world, compaction.SDK_CHANGED, action)
        assert error.report["installed"] == {"chromadb": "9.9.9"} != error.report["bound"]
    monkeypatch.setattr(stage, "_state_layout", layout)
    proof = stage._closure_proof

    def failing_io(root_path, binding_id):
        if binding_id == world["h_binding"]:
            raise OSError(5, "Input/output error")
        return proof(root_path, binding_id)

    monkeypatch.setattr(stage, "_closure_proof", failing_io)
    error = _refused_writes_nothing(world, compaction.H_TRANSIENT, lambda: _abort(world))
    assert error.report == {"w0": "transient", "errno": "EIO"}
    monkeypatch.setattr(stage, "_closure_proof", proof)
    stray = world["h"]["state"] / "data" / "stray.lock"
    stray.write_bytes(b"")
    with open(stray, "rb") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        _refused_writes_nothing(world, "runtime_admission_writer_active", lambda: _retire(world))
    stray.unlink()

    def cut(path):
        raise RuntimeError("cut")
    with pytest.raises(RuntimeError, match="cut"):
        compaction.compact_retire(root, world["id"], scan=cw.no_units, unlink_hook=cut)
    assert "retire_started" in _kinds(world) and not _log(world)["terminal"]
    assert compaction.compact_resume(root, world["id"])["phase"] == "complete"  # RS
    assert stage.read_binding(root, stage.read_admission_head(root)["binding_id"])["slot"] == "k"


# --- RX: a resurrected id in K' ---------------------------------------------------------------------

def test_a_resurrected_fact_is_refused_at_retire_and_abort_returns_to_h(x4, x5, tmp_path):
    """Row RX (retire.gates[3]): H's ledger records fact-x deleted; NEW revives it as an automatic
    fact (not a protected row, so W1 is 'equal'). Retire refuses with content-free counts; abort
    returns to H without it."""
    world = _world(x4, x5, tmp_path, cw.ledger_append("fact_deleted", "fact-x"))
    revive = cw.sql(("INSERT INTO facts VALUES ('fact-x', ?, 'c', 0.5, 'automatic', 1.0, NULL, 1, NULL, NULL)",
                     (CANARY,)))
    assert _v(world, tmp_path, "v", execute=revive).code == 0 and _kinds(world)[-1] == "verified"
    error = _refused_writes_nothing(world, compaction.RETIRE_RESURRECTED, lambda: _retire(world))
    assert error.report["resurrected"]["fact"] == 1 and CANARY not in json.dumps(error.report)
    assert _abort(world)["phase"] == "aborted"
    _no_canary(world)
    _back_on_h(world, tmp_path, "h")


# --- V1 + SS + AB: the recheck after the head moved (R5-OPS-1) -------------------------------------

def test_a_discard_interrupted_after_abort_started_finishes_through_the_w_recheck(x4, x5, tmp_path, monkeypatch):
    """Rows V1, SS, AB (R5-OPS-1): V writes a tombstone and its seal fails after the receipt (head
    left open with a receipt, as a power loss would). abort --to-h refuses (SS); the discard is cut
    after 'abort_started' before the return's CAS; the unit is started, so serve's R-a seals K' and
    the head moves; resume and plain abort refuse recheck_required; abort --to-h refuses with the
    recheck's own report, and --discard-k-changes with that digest completes, logged in
    'abort_rechecked'."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    seal = stage.seal_admission_run

    def failing_seal(*args, **kwargs):
        raise ValueError("runtime_admission_synthetic_seal_failure")

    monkeypatch.setattr(stage, "seal_admission_run", failing_seal)
    assert _v(world, tmp_path, "v", execute=_tombstone("conv-" + CANARY)).code != 0
    monkeypatch.setattr(stage, "seal_admission_run", seal)
    open_head = stage.read_admission_head(root)
    assert open_head["status"] != "closed"
    first = _refused_writes_nothing(world, compaction.ABORT_PROTECTED_CHANGED, lambda: _abort(world))
    digest = first.report["w_sha256"]
    original = stage.compaction_return_to_h

    def cut(*args, **kwargs):
        raise RuntimeError("cut")

    monkeypatch.setattr(stage, "compaction_return_to_h", cut)
    with pytest.raises(RuntimeError, match="cut"):
        _abort(world, discard_k_changes=digest[:12])
    monkeypatch.setattr(stage, "compaction_return_to_h", original)
    assert _kinds(world)[-1] == "abort_started"
    assert cw.serve(root, tmp_path, "unit-started").code == 78  # R-a seals K', then the gate refuses
    moved = stage.read_admission_head(root)
    assert moved["status"] == "closed" and moved["generation"] > open_head["generation"]
    _refusal("runtime_admission_compaction_abort_recheck_required",
             lambda: compaction.compact_resume(root, world["id"]))
    _refusal("runtime_admission_compaction_abort_recheck_required",
             lambda: compaction.compact_abort(root, world["id"]))
    again = _refused_writes_nothing(world, compaction.ABORT_PROTECTED_CHANGED, lambda: _abort(world))
    recheck = again.report["w_sha256"]
    assert _abort(world, discard_k_changes=recheck[:12])["phase"] == "aborted"
    [rechecked] = [event for event in _log(world)["events"] if event["kind"] == "abort_rechecked"]
    assert rechecked["data"]["discarded"]["w_sha256"] == recheck and rechecked["data"]["head"] == moved
    _no_canary(world)
    _back_on_h(world, tmp_path, "h")
