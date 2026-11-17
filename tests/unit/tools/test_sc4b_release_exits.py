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

from contextlib import closing, contextmanager
import errno
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import stat
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


@contextmanager
def _sqlite_held(path: Path):
    """An operator's sqlite3 session holding a lock on a database (no .lock file): readers get BUSY in
    rollback-journal and in WAL mode (exclusive locking mode)."""
    holder = sqlite3.connect(path, isolation_level=None)
    try:
        holder.execute("PRAGMA locking_mode=EXCLUSIVE")
        holder.execute("BEGIN EXCLUSIVE")
        yield
    finally:
        holder.close()


@contextmanager
def _restored(*paths: Path):
    """Put back the bytes and modes of ``paths`` (and drop the SQLite side files a change left)."""
    saved = {path: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode)) for path in paths}
    sides = {path: {suffix for suffix in ("-journal", "-wal", "-shm")
                    if os.path.lexists(str(path) + suffix)} for path in paths}
    try:
        yield
    finally:
        for path, (raw, mode) in saved.items():
            for suffix in ("-journal", "-wal", "-shm"):
                if suffix not in sides[path] and os.path.lexists(str(path) + suffix):
                    os.unlink(str(path) + suffix)
            os.chmod(path, 0o600)
            path.write_bytes(raw)
            os.chmod(path, mode)


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

_H_TOMBSTONE = cw.sql(
    ("INSERT INTO conversation_tombstones VALUES ('conv-h', 1.0, 'owner')", ()),
    ("INSERT INTO conversation_erasures VALUES ('op-h', 'conv-h', 1.0, 'owner')", ()),  # its operation
)
LOSS_CASES = {
    # NEW removes a tombstone at boot (V4R2-P1).
    "tombstone": ((_H_TOMBSTONE,), cw.sql(("DELETE FROM conversation_tombstones", ())),
                  ("tables", "conversation_tombstones", {"added": 0, "removed": 1})),
    # NEW truncates the deletion ledger to a strict prefix and changes nothing else.
    "ledger_prefix": ((cw.ledger_append("fact_deleted", "fact-a"), cw.ledger_append("fact_deleted", "fact-b")),
                      cw.sql(("DELETE FROM deletion_ledger_v1 WHERE seq=(SELECT MAX(seq) FROM deletion_ledger_v1)",
                              ())),
                      ("ledgers", None, "strict_prefix")),
}


@pytest.mark.parametrize("case", sorted(LOSS_CASES))
def test_a_loss_only_change_is_never_retired_and_abort_returns_to_h(x4, x5, tmp_path, case):
    """W1 'loss_only': retire refuses (exact equality); abort --to-h proceeds, H keeps what K' lost
    and serves (V4R2-P1, CR2-2). fact_owner_revisions, suspensions and the other listed tables in
    both directions are in test_sc4b_w (the 16 parametrized cases)."""
    first, change, (part, key, expected) = LOSS_CASES[case]
    world = _world(x4, x5, tmp_path, *first)
    assert _v(world, tmp_path, "v", execute=change).code == 0 and _kinds(world)[-1] == "verified"
    retired = _refused_writes_nothing(world, compaction.RETIRE_PROTECTED_CHANGED, lambda: _retire(world))
    report = retired.report["w"]
    assert report["class"] == compaction.W_LOSS_ONLY
    found = report[part]["data/privacy.db"]
    assert (found[key] if key else found) == expected
    # SC4B-N6 (review RA4B-3 R1): with the current digest, a discard of a loss-only K' is refused.
    _refused_writes_nothing(world, compaction.DISCARD_MISMATCH,
                            lambda: _abort(world, discard_k_changes=retired.report["w_sha256"][:12]))
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
    for state, where in ((root / "slots" / "k" / "state", "k_prime"), (world["h"]["state"], None)):
        stray = state / "data" / "stray.lock"
        stray.write_bytes(b"")
        with open(stray, "rb") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
            error = _refused_writes_nothing(world, "runtime_admission_writer_active", lambda: _abort(world))
            assert error.report == ({"w0": "transient", "state": "k_prime"} if where else {"w0": "transient"})
        stray.unlink()
    # Review RA4B-5/RB-7: a live holder's SQLite lock on a K' database (no .lock file) and a resource errno
    # in K's writer check are transient refusals of K', not the safe stop.
    with _sqlite_held(root / "slots" / "k" / "state" / "data" / "privacy.db"):
        error = _refused_writes_nothing(world, compaction.K_TRANSIENT, lambda: _abort(world))
    assert error.report == {"w1": "transient", "state": "k_prime", "database": "data/privacy.db",
                            "sqlite": "SQLITE_BUSY"}
    writers, k_state = stage._assert_writers_closed, root / "slots" / "k" / "state"

    def no_descriptors(state):
        if Path(state) == k_state:
            raise OSError(errno.EMFILE, "Too many open files")
        return writers(state)

    monkeypatch.setattr(stage, "_assert_writers_closed", no_descriptors)
    error = _refused_writes_nothing(world, compaction.K_TRANSIENT, lambda: _abort(world))
    assert error.report == {"w0": "transient", "state": "k_prime", "errno": "EMFILE"}
    monkeypatch.setattr(stage, "_assert_writers_closed", writers)

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
    """Rows V1, SS, AB (R5-OPS-1): V writes a tombstone and its seal fails with an I/O error after the
    receipt (serve leaves the head running with the receipt, as a power loss would). abort --to-h refuses (SS); the discard is cut
    after 'abort_started' before the return's CAS; the unit is started, so serve's R-a seals K' and
    the head moves; resume and plain abort refuse recheck_required; abort --to-h refuses with the
    recheck's own report, and --discard-k-changes with that digest completes, logged in
    'abort_rechecked'."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    seal = stage.seal_admission_run

    def failing_seal(*args, **kwargs):
        raise OSError(5, "Input/output error")  # transient: serve leaves the head running for R-a

    monkeypatch.setattr(stage, "seal_admission_run", failing_seal)
    assert _v(world, tmp_path, "v", execute=_tombstone("conv-" + CANARY)).code != 0
    monkeypatch.setattr(stage, "seal_admission_run", seal)
    open_head = stage.read_admission_head(root)
    assert open_head["status"] == "running"
    alive = launcher.child_alive
    monkeypatch.setattr(launcher, "child_alive", lambda pid, start_time: True)  # V's child still running
    for options in ({}, {"discard_k_changes": "0" * 12}):  # review RA4B-3/RB-2(a): W0 refuses it first
        error = _refused_writes_nothing(world, "runtime_admission_recovery_child_alive",
                                        lambda: _abort(world, **options))
        assert error.report == {"w0": "transient"}
    monkeypatch.setattr(launcher, "child_alive", alive)
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


# --- N12: a protected table only in K' --------------------------------------------------------------

_DEFAULT_CONTROL_ROW = cw.sql(
    ("CREATE TABLE IF NOT EXISTS ambient_runtime_control (id INTEGER PRIMARY KEY CHECK(id=1), paused INTEGER "
     "NOT NULL CHECK(paused IN (0,1)), revision INTEGER NOT NULL CHECK(revision>=0))", ()),
    ("INSERT OR IGNORE INTO ambient_runtime_control VALUES (1,0,0)", ()))


def test_a_protected_table_created_only_in_k_prime_is_never_retired(x4, x5, tmp_path):
    """SC4A-N12 with an H that never booted ambient (no ambient_runtime_control): NEW creates the table
    and its default row in V. W1 ignores objects only in K' ('equal'), but retire also needs K's whole
    authority unchanged (SC4B-N4) and refuses; abort returns to H, which never had the row."""
    world = _world(x4, x5, tmp_path)
    assert _v(world, tmp_path, "v", execute=_DEFAULT_CONTROL_ROW).code == 0 and _kinds(world)[-1] == "verified"
    error = _refused_writes_nothing(world, compaction.RETIRE_PROTECTED_CHANGED, lambda: _retire(world))
    assert error.report["w"]["class"] == compaction.W_EQUAL and error.report["authority_equal"] is False
    assert _abort(world)["phase"] == "aborted"
    _back_on_h(world, tmp_path, "h")


# --- HD: each real change to H's own files after V (review RA4B-2, RB-2(d)(e)) ------------------------

def _old_copy_write(world: dict) -> None:
    with closing(sqlite3.connect(world["h"]["state"] / "data" / "privacy.db")) as db, db:
        db.execute("INSERT INTO conversation_tombstones VALUES ('conv-old-copy', 1.0, 'owner')")


def _dropped_table(world: dict) -> None:
    with closing(sqlite3.connect(world["h"]["state"] / "data" / "privacy.db")) as db, db:
        db.execute("DROP TABLE conversation_tombstones")


def _not_a_database(world: dict) -> None:
    (world["h"]["state"] / "data" / "privacy.db").write_bytes(b"garbage" * 1000)


def _marker_rewritten(world: dict) -> None:
    path = world["h"]["state"] / stage._MARKER
    value = json.loads(path.read_text())
    os.chmod(path, 0o600)
    path.write_text(json.dumps({**value, "binding_id": "0" * 64}))


def _closure_malformed(world: dict) -> None:
    path = world["root"] / "admission" / "closures" / (world["h_binding"] + ".json")
    value = json.loads(path.read_text())
    os.chmod(path, 0o600)
    path.write_text(json.dumps({**value, "running_head": {**value["running_head"], "run_id": 5}}))


H_CHANGES = {
    # (change, the files it touches relative to the world, the coded underlying reason)
    "old_copy_writes_a_protected_row": (_old_copy_write, ("h", "data/privacy.db"), "privacy_authority_stale"),
    "table_dropped": (_dropped_table, ("h", "data/privacy.db"), "privacy_authority_stale"),
    "database_overwritten": (_not_a_database, ("h", "data/privacy.db"), "sqlite:SQLITE_NOTADB"),
    "marker_rewritten": (_marker_rewritten, ("h", stage._MARKER), "runtime_admission_marker_mismatch"),
    "closure_proof_malformed": (_closure_malformed, ("closure", None), "record_invalid"),
}


def test_each_change_to_h_after_v_is_hd_and_nothing_is_written(x4, x5, tmp_path):
    """Row HD (design: 'a real change to H's own files: an unauthorized writer, a manual edit or storage
    damage'; test plan R4-OPS-3): after a verified V, an old copy writing a protected row to H, a dropped
    H table, an H database overwritten (SQLITE_NOTADB), H's marker rewritten and a malformed H closure
    proof (a TypeError while reading it). Each refuses h_changed with its code in abort --to-h (with and
    without the discard) and in retire, writing nothing; with H put back, abort returns to H."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    assert _v(world, tmp_path, "v").code == 0 and _kinds(world)[-1] == "verified"
    for name, (change, (where, relative), code) in sorted(H_CHANGES.items()):
        path = (world["h"]["state"] / relative if where == "h"
                else root / "admission" / "closures" / (world["h_binding"] + ".json"))
        with _restored(path):
            change(world)
            for action in (lambda: _abort(world), lambda: _abort(world, discard_k_changes="0" * 12),
                           lambda: _retire(world)):
                error = _refused_writes_nothing(world, compaction.H_CHANGED, action)
                assert error.report == {"w0": "drift", "code": code}, name
    assert _abort(world)["phase"] == "aborted"
    _back_on_h(world, tmp_path, "h")


# --- SC-R follow-ups: a later install over a closed K' (SCR-C-1), DL during the transition (SCR-C-2) ---

def test_a_later_install_over_a_closed_k_prime_checks_h_binding_too(x4, x5, tmp_path, monkeypatch):
    """SCR-C-1 (row CI): after a verified V the head is K', closed. A controller whose binding output
    differs only on H's slot passes SR-1 on K' but is refused controller_binding_changed before current
    moves, writing nothing; without it the install is admitted and abort --to-h returns to H."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    assert _v(world, tmp_path, "v").code == 0 and _kinds(world)[-1] == "verified"
    assert stage.read_admission_head(root)["binding_id"] != world["h_binding"]
    h_slot = stage.read_binding(root, world["h_binding"])["slot"]
    current = os.readlink(root / "controller" / "current")
    snapshot = stage._binding_snapshot

    def changed_on_h(root_, name, **options):
        value = snapshot(root_, name, **options)
        if name == h_slot:  # the new projection differs only on rows H holds and K' does not
            value = dict(value, authority=dict(value["authority"], digest="0" * 64))
        return value

    with monkeypatch.context() as patch:
        patch.setattr(stage, "_binding_snapshot", changed_on_h)
        _refused_writes_nothing(world, launcher.CONTROLLER_BINDING_CHANGED, lambda: _install(root, x5))
    assert os.readlink(root / "controller" / "current") == current
    _install(root, x5)  # the same controller, H unchanged: an equal triple repeat is admitted
    assert _status(world)["phase"] == "verified"
    assert _abort(world)["phase"] == "aborted"
    _back_on_h(world, tmp_path, "h")


def test_a_later_install_during_a_cut_retire_skips_the_h_check(x4, x5, tmp_path):
    """SCRF-2 (rows RS, CI): a retire cut after H's slot is renamed has logged 'retire_started', so its
    gates passed and H is being destroyed. A later install over the closed K' passes SR-1 on K' and skips
    the SCR-C-1 H check (admitted, as before SCR-C-1, instead of a raw errno naming H's slot path), and
    compact resume completes the retire."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    assert _v(world, tmp_path, "v").code == 0 and _kinds(world)[-1] == "verified"
    h_slot = stage.read_binding(root, world["h_binding"])["slot"]
    current = os.readlink(root / "controller" / "current")

    def cut(path):
        if not (root / "slots" / h_slot).exists():
            raise RuntimeError("cut")
    with pytest.raises(RuntimeError, match="cut"):
        compaction.compact_retire(root, world["id"], scan=cw.no_units, unlink_hook=cut)
    assert "retire_started" in _kinds(world) and not (root / "slots" / h_slot).exists()
    _install(root, x5)  # the same controller: an equal triple repeat, admitted
    assert os.readlink(root / "controller" / "current") == current
    assert compaction.compact_resume(root, world["id"])["phase"] == "complete"
    assert stage.read_binding(root, stage.read_admission_head(root)["binding_id"])["slot"] == "k"
    assert cw.serve(root, tmp_path, "k-prime").code == 0


def test_a_delegation_record_during_the_transition_refuses_every_exit_before_any_event(x4, x5, tmp_path,
                                                                                         monkeypatch):
    """SCR-C-2 (row DL during a pending release-mode log): a record planted under admission/delegations
    after V refuses abort --to-h (with and without the discard), the plain abort and retire with
    delegation_frozen, writing nothing (no 'abort_started'). An abort cut after 'abort_started' and
    resumed with a record present refuses the same way. Row DL's exit (remove the record by hand) then
    returns to H."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    assert _v(world, tmp_path, "v").code == 0 and _kinds(world)[-1] == "verified"
    record = root / "admission" / "delegations" / ("0" * 64 + ".json")
    record.parent.mkdir(exist_ok=True)
    record.write_text("{}")
    for action in (lambda: _abort(world), lambda: _abort(world, discard_k_changes="0" * 12),
                   lambda: compaction.compact_abort(root, world["id"]), lambda: _retire(world)):
        _refused_writes_nothing(world, stage.DELEGATION_FROZEN, action)
    assert _kinds(world)[-1] == "verified" and _status(world)["phase"] == "verified"
    record.unlink()
    original = stage.compaction_return_to_h

    def cut(*args, **kwargs):
        raise RuntimeError("cut")

    monkeypatch.setattr(stage, "compaction_return_to_h", cut)
    with pytest.raises(RuntimeError, match="cut"):
        _abort(world)
    monkeypatch.setattr(stage, "compaction_return_to_h", original)
    assert _kinds(world)[-1] == "abort_started"
    record.write_text("{}")
    for action in (lambda: compaction.compact_resume(root, world["id"]), lambda: _abort(world),
                   lambda: compaction.compact_abort(root, world["id"])):
        _refused_writes_nothing(world, stage.DELEGATION_FROZEN, action)
    assert _kinds(world)[-1] == "abort_started"
    record.unlink()
    assert compaction.compact_resume(root, world["id"])["phase"] == "aborted"
    _back_on_h(world, tmp_path, "h")


def test_a_resumed_abort_past_the_return_to_h_refuses_a_delegation_record(x4, x5, tmp_path, monkeypatch):
    """SCRF-3 (row DL): an abort cut after its return to H ('returned' logged, the head is H) still has
    K to destroy. A record planted then makes compact resume and every abort refuse delegation_frozen,
    writing nothing, so K is not destroyed with a record present; row DL's exit then finishes it."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    assert _v(world, tmp_path, "v").code == 0 and _kinds(world)[-1] == "verified"

    def cut(*args, **kwargs):
        raise RuntimeError("cut")
    with monkeypatch.context() as patch:
        patch.setattr(compaction, "_destroy_own_staging", cut)
        with pytest.raises(RuntimeError, match="cut"):
            _abort(world)
    assert _kinds(world)[-1] == "returned" and (root / "slots" / "k").exists()
    assert stage.read_admission_head(root)["binding_id"] == world["h_binding"]
    record = root / "admission" / "delegations" / ("0" * 64 + ".json")
    record.parent.mkdir(exist_ok=True)
    record.write_text("{}")
    for action in (lambda: compaction.compact_resume(root, world["id"]), lambda: _abort(world),
                   lambda: compaction.compact_abort(root, world["id"])):
        _refused_writes_nothing(world, stage.DELEGATION_FROZEN, action)
    assert _kinds(world)[-1] == "returned" and (root / "slots" / "k").exists()
    record.unlink()
    assert compaction.compact_resume(root, world["id"])["phase"] == "aborted"
    _back_on_h(world, tmp_path, "h")


# --- the return to H's own authority checks after V (review RA4B-3, RB-2(b)(c), RB-3) ---------------

def test_the_return_to_h_checks_its_confirmation_the_head_and_k_prime_and_writes_nothing(x4, x5, tmp_path,
                                                                                        monkeypatch):
    """After a verified V with W 'equal': --confirm-h missing, short or another binding's prefix refuses
    abort_confirmation; a discard with the current digest refuses (nothing to discard, SC4B-N6); a
    readable closure proof of another run refuses head_moved; K' head's closure proof gone refuses
    k_closure_unreadable in abort and retire (RB-3, an HD-like stop); a K' binding change outside
    sdk_versions refuses retire with k_binding_changed and its keys, and abort then returns to H."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    assert _v(world, tmp_path, "v").code == 0 and _kinds(world)[-1] == "verified"
    k_head = stage.read_admission_head(root)
    for confirm in (None, world["h_binding"][:11], k_head["binding_id"][:12]):
        _refused_writes_nothing(world, "runtime_admission_compaction_abort_confirmation",
                                lambda: compaction.compact_abort(root, world["id"], to_h=True, confirm_h=confirm))
    digest = compaction._w(root, _log(world))["sha256"]
    _refused_writes_nothing(world, compaction.DISCARD_MISMATCH, lambda: _abort(world, discard_k_changes=digest[:12]))
    proof = stage._closure_proof

    def another_run(root_path, binding_id):
        found = proof(root_path, binding_id)
        if binding_id != k_head["binding_id"]:
            return found
        return {**found, "running_head": {**found["running_head"],
                                          "generation": found["running_head"]["generation"] + 1}}

    monkeypatch.setattr(stage, "_closure_proof", another_run)
    _refused_writes_nothing(world, "runtime_admission_compaction_head_moved", lambda: _abort(world))
    monkeypatch.setattr(stage, "_closure_proof", proof)
    closure = root / "admission" / "closures" / (k_head["binding_id"] + ".json")
    aside = tmp_path / "k-closure-aside.json"
    os.rename(closure, aside)
    for action in (lambda: _abort(world), lambda: _retire(world)):
        error = _refused_writes_nothing(world, compaction.K_CLOSURE_UNREADABLE, action)
        assert error.report == {"code": "errno:ENOENT"}
    os.rename(aside, closure)
    snapshot = stage._binding_snapshot

    def another_commit(root_path, name, **options):
        found = snapshot(root_path, name, **options)
        return {**found, "code_commit": "0" * 40} if name == "k" else found

    monkeypatch.setattr(stage, "_binding_snapshot", another_commit)
    error = _refused_writes_nothing(world, compaction.K_BINDING_CHANGED, lambda: _retire(world))
    assert error.report == {"code": "privacy_authority_stale", "changed": ["code_commit"]}
    assert _abort(world)["phase"] == "aborted"  # the route of k_binding_changed (design test plan R5-2)
    monkeypatch.setattr(stage, "_binding_snapshot", snapshot)
    _back_on_h(world, tmp_path, "h")


# --- RB-1: a K'-only lock entry that cannot be opened (row V2) ----------------------------------------

_STRAY_LOCK_LINK = "import os\nos.symlink(state / 'data' / 'privacy.db', state / 'data' / 'stray.lock')"


def test_a_k_prime_lock_entry_that_cannot_be_opened_stops_safely_until_the_discard(x4, x5, tmp_path):
    """Review RB-1: NEW plants data/stray.lock in K' as a symlink at boot (an isolation breach); serve's
    seal fails (ELOOP) and the head stays open. abort --to-h is the safe stop SS, its W report naming the
    K' locks 'unverifiable'; the discard returns to H, K's destruction refuses the link without following
    it (compaction_symlink, runbook: remove it by hand) and resume then finishes."""
    world = _world(x4, x5, tmp_path)
    root = world["root"]
    assert _v(world, tmp_path, "v", execute=_STRAY_LOCK_LINK).code != 0
    assert stage.read_admission_head(root)["status"] != "closed"
    first = _refused_writes_nothing(world, compaction.ABORT_PROTECTED_CHANGED, lambda: _abort(world))
    report = first.report["w"]
    assert report["k_writers"] == {"status": "unverifiable", "error": "errno:ELOOP"}
    assert report["class"] == compaction.W_RESTRICTIVE
    assert all(found == {"status": "ok"} for found in report["databases"].values())
    _refused_writes_nothing(world, compaction.ABORT_PROTECTED_CHANGED, lambda: _abort(world))
    _refusal("runtime_admission_compaction_symlink",
             lambda: _abort(world, discard_k_changes=first.report["w_sha256"][:12]))
    head = stage.read_admission_head(root)
    assert head["status"] == "closed" and head["binding_id"] == world["h_binding"]  # returned to H
    [started] = [event for event in _log(world)["events"] if event["kind"] == "abort_started"]
    assert started["data"]["discarded"]["counts"]["k_writers"] == {"status": "unverifiable", "error": "errno:ELOOP"}
    (root / "slots" / "k" / "state" / "data" / "stray.lock").unlink()
    assert compaction.compact_resume(root, world["id"])["phase"] == "aborted"
    _back_on_h(world, tmp_path, "h")


def test_a_discard_on_a_plain_log_is_refused(x4_archive, tmp_path):
    """--discard-k-changes answers a W report: a plain-mode log (epoch 4, no 'mode') refuses it and
    writes nothing; its own exits are unchanged (the plain abort before the run)."""
    world = cw.world(x4_archive, tmp_path, hops=())
    root = world["root"]
    compaction_id = cw.start(root, world["archive"])["compaction_id"]
    world["id"] = compaction_id
    witness = cw.witness(root)
    _refusal(compaction.RELEASE_MODE_ONLY, lambda: compaction.compact_abort(
        root, compaction_id, to_h=True, confirm_h=world["h"]["binding"][:12], discard_k_changes="0" * 12))
    assert cw.witness(root) == witness
    assert compaction.compact_abort(root, compaction_id)["phase"] == "aborted"

