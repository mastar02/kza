"""U3b-0c owner-state compaction on real managed roots (owner-compaction design v2-v5).

Every world is built from X (HEAD plus this working tree, compiled): real
closed runs, real delegate hops, the real compaction worker subprocess (G1
capability) and the managed launcher's in-process serve under a fake
manager for the verification run. The controller-location check (P7) is
exercised directly and by the compiled rehearsals, where the controller
runs from controller/current; in this process the controller is the
checkout, so the fixture below replaces only that one check.
"""

import errno
import gzip
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from scripts import compaction
from scripts import stage_runtime as stage
from src.core import entry_guard
from tests.mocks import compaction_world as cw


@pytest.fixture(scope="module")
def x_archive(tmp_path_factory):
    return cw.build_x(tmp_path_factory.mktemp("compaction-x"))


@pytest.fixture(autouse=True)
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold for an in-process controller (see module doc)."""
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)


def _refusal(code, action):
    with pytest.raises(ValueError) as caught:
        action()
    assert str(caught.value) == code or str(caught.value).startswith(code + ":"), str(caught.value)
    return caught.value


def _events(root, compaction_id):
    return [event["kind"] for event in stage.read_compaction_log(root, compaction_id)["events"]]


def _verified(world, tmp_path, *, execute=None):
    """start -> verify -> the one armed serve run, closed cleanly."""
    root = world["root"]
    started = cw.start(root, world["archive"])
    compaction.compact_verify(root, started["compaction_id"])
    result = cw.serve(root, tmp_path, "verify", execute=execute)
    assert result.code == 0, result.phases
    return started["compaction_id"]


def _raw_hits(root: Path, needle: bytes) -> set:
    """Every file under ``root`` whose raw bytes (or gunzipped bytes) contain ``needle``."""
    hits = set()
    for directory, _names, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            raw = path.read_bytes()
            if raw[:2] == b"\x1f\x8b":
                raw += gzip.decompress(raw)
            if needle in raw:
                hits.add(path.relative_to(root).as_posix())
    return hits


# --- T-U3, T-U4, T-U6, E3, T-Q4-1, T-Q4-4, F5, F6, T-U7: the whole flow on a production-like root ---

def test_full_compaction_on_a_production_like_root(x_archive, tmp_path, monkeypatch):
    world = cw.world(x_archive, tmp_path, env="KZA_TEST=1\n")
    root, states = world["root"], world["states"]
    outside = tmp_path / "outside-canonical"
    outside.mkdir()
    (outside / "keep.txt").write_text("CANARY-OUTSIDE")
    # Q4: everything that is not a proven member stays byte-identical.
    (root / "backups").mkdir()
    with sqlite3.connect(root / "backups" / "old.db") as db:
        db.execute("CREATE TABLE t(x)")
        db.execute("INSERT INTO t VALUES ('CANARY-BACKUP')")
    (root / "logs").mkdir()
    (root / "logs" / "console.log.1.gz").write_bytes(gzip.compress(b"CANARY-LOG"))
    (root / "deployments").mkdir()
    (root / "dashboard-proxy").mkdir()
    (root / "console-logrotate.conf").write_text("/x/logs/console.log {}\n")
    (root / "kza-x.tar.gz").write_bytes(b"archive")
    legacy = root / "slots" / "legacy"
    (legacy / "runtime").mkdir(parents=True)
    (legacy / "state").mkdir()
    (legacy / "state" / "data").symlink_to(outside)
    unbound = root / "slots" / "old"  # a pre-0c slot with no binding, marker or prepare record
    (unbound / "runtime").mkdir(parents=True)
    (unbound / "state" / "data").mkdir(parents=True)
    (unbound / "state" / "data" / "x.txt").write_text("CANARY-UNBOUND")
    # F6: a proven prepare leftover (staging record by identity), an unproven one and selection leftovers.
    proven = root / "slots" / ".prepare-proven"
    (proven / "state").mkdir(parents=True)
    (proven / "state" / "x.txt").write_text("CANARY-PREPARE")
    stage._atomic_record(stage.prepared_directory(root) / ".staging-feed.json", {
        "schema_version": 1, "kind": "staging", "staging_id": "feed", "path": str(proven),
        "identity": stage._identity(proven), "compaction_id": None, "created_at": 1.0}, immutable=True)
    unproven = root / "slots" / ".prepare-unproven"
    unproven.mkdir()
    (unproven / "x.txt").write_text("CANARY-UNPROVEN")
    (root / ".activate-empty").mkdir()
    (root / ".activate-other").mkdir()
    (root / ".activate-other" / "x").write_text("y")
    # A prepared, never-activated slot (proof b) and its owner data.
    stage.prepare_slot(x_archive, root, "abandoned", states["a"]["state"])
    untouched_names = ["backups", "logs", "deployments", "dashboard-proxy", "console-logrotate.conf",
                       "kza-x.tar.gz", "slots/legacy", "slots/old", "slots/.prepare-unproven", ".activate-other",
                       "controller"]
    before = {name: cw.tree(root / name) if (root / name).is_dir() else (root / name).read_bytes()
              for name in untouched_names}
    outside_before = cw.tree(outside)

    witness = cw.witness(root)
    planned = cw.plan(root, x_archive, retire_prepare_leftovers=True)
    assert cw.witness(root) == witness  # T-U1: plan is read-only
    classes = {(member["class"], member.get("slot") or Path(member["path"]).name) for member in planned["members"]}
    assert classes == {("managed_slot", "a"), ("managed_slot", "b"), ("managed_slot", "c"),
                       ("managed_slot", "abandoned"), ("prepare_leftover", ".prepare-proven"),
                       ("selection_leftover", ".activate-empty")}
    proofs = {member.get("slot"): member["proof"] for member in planned["members"] if member.get("slot")}
    assert proofs == {"a": "binding", "b": "binding", "c": "binding", "abandoned": "prepare_record"}
    assert {entry["path"] for entry in planned["untouched"]} == set(untouched_names)
    # P10: the only member-only data is the prepare leftover's file; it needs the explicit acceptance.
    unexplained = {member["class"]: planned["coverage_members"][str(member["n"])]["unexplained"]
                   for member in planned["members"]}
    assert planned["owners"] == 2 and planned["coverage"]["unexplained"] == 1
    assert unexplained["prepare_leftover"] == 1
    text = json.dumps(planned)
    assert "CANARY" not in text and "KZA_TEST" not in text

    _refusal("runtime_admission_compaction_owner_only_data",
             lambda: cw.start(root, x_archive, planned=planned, retire_prepare_leftovers=True))
    started = cw.start(root, x_archive, planned=planned, retire_prepare_leftovers=True, accept_owner_only_loss=True)
    compaction_id = started["compaction_id"]
    head = stage.read_admission_head(root)
    k = stage.read_binding(root, head["binding_id"])
    k_state = root / "slots" / "k" / "state"
    assert k["slot"] == "k" and os.readlink(root / "current") == "slots/k/runtime"
    # T-U3: K owns everything, no delegation; many-to-one relocation into K; .env link resolves.
    policy = json.loads((k_state / ".kza-target-ownership.json").read_text())
    assert policy["delegated_targets"] == [] and not list(stage._delegation_records(root, k_state))
    with sqlite3.connect(k_state / "data" / "privacy.db") as db:
        roots = sorted(json.loads(row[0])["root"] for row in db.execute(
            "SELECT descriptor FROM erasure_targets WHERE participant='collectors'"))
    assert roots == [str(k_state / "data" / name) for name in ("clips-a", "clips-b", "clips-c")]
    assert os.readlink(root / "slots" / "k" / "runtime" / ".env") == "../state/.env"
    assert (root / "slots" / "k" / "runtime" / ".env").read_text() == "KZA_TEST=1\n"
    # E3: identity mode never appends to the deletion ledger.
    for state in (states["c"]["state"], k_state):
        with sqlite3.connect(state / "data" / "privacy.db") as db:
            assert db.execute("SELECT COUNT(*) FROM deletion_ledger_v1").fetchone()[0] == 0

    # Gate (T-U5): serve before verify, operator entries and prepare are refused.
    assert cw.serve(root, tmp_path, "early").code == 78
    _refusal("runtime_admission_compaction_pending", lambda: stage.AdmissionController(root).__enter__())
    _refusal("runtime_admission_compaction_pending",
             lambda: stage.prepare_slot(x_archive, root, "zz", states["a"]["state"]))
    _refusal("runtime_admission_compaction_not_verified",
             lambda: compaction.compact_retire(root, compaction_id, scan=cw.no_units))  # P3: K' must close first
    assert compaction.compact_verify(root, compaction_id)["armed"] is True
    result = cw.serve(root, tmp_path, "verify")
    assert result.code == 0
    assert "verified" in _events(root, compaction_id)
    assert cw.serve(root, tmp_path, "again").code == 78  # consumed: retire, allow-run or abort-to-H

    catalog = cw.tree(root / "admission" / "copies")
    calls = []
    for name in ("open", "listdir", "scandir", "rename", "unlink", "rmdir"):
        original = getattr(compaction.os, name)

        def spy(path, *args, _original=original, _name=name, **kwargs):
            calls.append((_name, os.fspath(path) if isinstance(path, (str, os.PathLike)) else None))
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(compaction.os, name, spy)
    retired = compaction.compact_retire(root, compaction_id, scan=cw.no_units)
    monkeypatch.undo()
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)
    assert retired["phase"] == "complete" and retired["destroyed"] == retired["members"] == 8
    # T-Q4-4: no call ever names a path inside an untouched entry.
    for _name, path in calls:
        if path is not None and os.path.isabs(path):
            assert not any(Path(path).is_relative_to(root / name) for name in untouched_names), path
    assert sorted(os.listdir(root / "slots")) == [".prepare-unproven", "k", "legacy", "old"]
    assert not (root / ".activate-empty").exists()
    for name, value in before.items():
        assert (cw.tree(root / name) if (root / name).is_dir() else (root / name).read_bytes()) == value, name
    assert cw.tree(outside) == outside_before
    # T-U6: tombstones, copies-retired records, catalog chain bytes unchanged.
    tombstones = [json.loads(path.read_text()) for path in stage.tombstone_directory(root).glob("*.json")]
    assert {item["slot"] for item in tombstones if item["slot"]} == {"a", "b", "c", "abandoned"}
    assert all(item["status"] == "destroyed" and item["compaction_id"] == compaction_id for item in tombstones)
    retired_copies = sorted(path.stem for path in (root / "admission" / "copies-retired").glob("*.json"))
    assert retired_copies and cw.tree(root / "admission" / "copies") == catalog
    assert _raw_hits(root, b"CANARY-PREPARE") == set()
    assert _raw_hits(root, b"CANARY-BACKUP") == {"backups/old.db"}
    # Serving resumes normally; K' is a self-contained head.
    assert cw.serve(root, tmp_path, "after").code == 0
    # T-U4: the next delegate from K' has exactly one owner.
    stage.prepare_slot(x_archive, root, "next", k_state)
    assert cw.delegate(root, "next")["owners_after_grant"] == 1
    # T-U7: retired locators are never reused, by name or by path.
    _refusal("runtime_admission_retired_locator", lambda: stage.prepare_slot(x_archive, root, "b", k_state))
    # F5: a directory recreated under a retired name, without a valid marker, is untouched.
    fake = root / "slots" / "b"
    (fake / "runtime").mkdir(parents=True)
    (fake / "state").mkdir()
    members, untouched = compaction.retirement_set(root, k_slot="k2", retire_prepare_leftovers=False)
    assert "slots/b" in {entry["path"] for entry in untouched}
    assert all(member.get("slot") != "b" for member in members)
    [status] = compaction.compact_status(root, compaction_id)["compactions"]
    assert status["terminal"] and status["phase"] == "complete"


# --- T-U2 / P1-P14: coded refusals that change nothing ---

def test_preconditions_are_coded_refusals_that_change_nothing(x_archive, tmp_path, monkeypatch):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root, archive = world["root"], world["archive"]
    witness = cw.witness(root)

    def refused(code, **options):
        _refusal(code, lambda: cw.plan(root, archive, **options))
        assert cw.witness(root) == witness

    refused("runtime_admission_compaction_k_exists", k="b")
    refused("runtime_admission_compaction_k_exists", k="a")
    # P6: an absolute locator of H or an owner in config/ or .env refuses (H4).
    cw.run(root, "b", cw.write_file("config/extra.yaml", f"root: {world['states']['a']['state']}\n"))
    witness = cw.witness(root)
    refused("runtime_admission_compaction_absolute_locator")
    cw.run(root, "b", "(state / 'config' / 'extra.yaml').unlink()")
    witness = cw.witness(root)
    # P9: capacity with the archive formula, reported content-free.
    monkeypatch.setattr(compaction, "BINDING_MARGIN", 1000)
    error = _refusal("runtime_admission_compaction_capacity", lambda: cw.plan(root, archive))
    assert set(error.report) >= {"binding_bytes", "history_archive_bytes", "limit_bytes"}
    monkeypatch.setattr(compaction, "BINDING_MARGIN", 12 * 1024 * 1024)
    # P5: a writer holding a member state's lock.
    import fcntl
    lock = world["states"]["a"]["state"] / "data" / "privacy.db.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        witness = cw.witness(root)
        refused("runtime_admission_writer_active")
    finally:
        os.close(descriptor)
    # P13 (G4): contract 2, an uninstall after the install, no controller.
    newer = cw.install_record(root, contract=2, stamp="20261114T000001Z000000")
    witness = cw.witness(root)
    refused("runtime_admission_compaction_launcher_contract")
    newer.unlink()
    removed = root / "admission" / "launcher" / "uninstall-20261114T000002Z000000.json"
    stage._atomic_record(removed, {"schema_version": 1}, immutable=True)
    witness = cw.witness(root)
    refused("runtime_admission_compaction_launcher_missing")
    removed.unlink()
    # One deletion at a time: a pending legacy retirement intent refuses (stricter, recorded).
    intent = root / "admission" / "legacy-retirement" / ("intent-" + "0" * 64 + ".json")
    stage._atomic_record(intent, {"schema_version": 1, "kind": "legacy_state_retire_intent",
                                  "plan_sha256": "0" * 64, "ineligible_slots": ["x"]}, immutable=True)
    witness = cw.witness(root)
    refused("runtime_admission_compaction_legacy_retirement_pending")
    intent.unlink()
    # P14 runs on every member root and refuses with the scanner's code.
    seen = []

    def scanner(systemctl, *, root, members, **options):
        seen.extend(item["path"] for item in members)
        raise compaction.units.UnitScanRefusal("runtime_admission_compaction_unit_path:kza-proxy.service")
    monkeypatch.setattr(compaction.units, "scan_units", scanner)
    monkeypatch.setattr(compaction.units, "strict_systemctl", lambda manager: None)
    witness = cw.witness(root)
    _refusal("runtime_admission_compaction_unit_path", lambda: compaction.compact_plan(root, "k", archive))
    assert cw.witness(root) == witness
    assert str(world["states"]["a"]["state"]) in seen and str(root / "slots" / "a") in seen
    # P11/P12 and T-Q4-3: acknowledgements are exact; a changed untouched list is plan_changed.
    planned = cw.plan(root, archive)
    _refusal("runtime_admission_compaction_not_acknowledged", lambda: compaction.compact_start(
        root, "k", archive, plan_sha256=planned["plan_sha256"], acknowledge_untouched=planned["untouched_sha256"],
        destroy_rollback_states=False, scan=cw.no_units))
    (root / "new-entry").mkdir()
    witness = cw.witness(root)
    _refusal("runtime_admission_compaction_plan_changed", lambda: compaction.compact_start(
        root, "k", archive, plan_sha256=planned["plan_sha256"], acknowledge_untouched=planned["untouched_sha256"],
        destroy_rollback_states=True, scan=cw.no_units))
    assert cw.witness(root) == witness
    replanned = cw.plan(root, archive)
    _refusal("runtime_admission_compaction_plan_changed", lambda: compaction.compact_start(
        root, "k", archive, plan_sha256=replanned["plan_sha256"], acknowledge_untouched="0" * 64,
        destroy_rollback_states=True, scan=cw.no_units))
    assert cw.witness(root) == witness
    assert stage.compaction_logs(root) == []


def test_unknown_head_refuses_before_anything(x_archive, tmp_path):
    """T-U9 (F8): an UNKNOWN head refuses compaction at the lease."""
    world = cw.world(x_archive, tmp_path, hops=())
    root = world["root"]
    with pytest.raises(ValueError):
        stage.run_slot(root, "a", [sys.executable, "-I", "-B", "-c", "raise SystemExit(0)"], timeout=30)
    assert stage.read_admission_head(root)["status"] == "unknown"
    witness = cw.witness(root)
    _refusal("runtime_admission_not_closed", lambda: cw.plan(root, world["archive"]))
    assert cw.witness(root) == witness


def test_controller_location_refuses_member_modules_and_an_uninstalled_location(tmp_path, monkeypatch):
    monkeypatch.undo()
    root = tmp_path / "root"
    member = root / "slots" / "a"
    (member / "runtime" / "src").mkdir(parents=True)
    fake = type(sys)("kza_member_module")
    fake.__file__ = str(member / "runtime" / "src" / "x.pyc")
    monkeypatch.setitem(sys.modules, "kza_member_module", fake)
    _refusal("runtime_admission_compaction_controller_location", lambda: compaction.controller_location(
        root, [{"path": str(member)}]))
    monkeypatch.delitem(sys.modules, "kza_member_module")
    (root / "admission").mkdir()
    compaction.controller_location(root, [{"path": str(member)}])  # not installed: the checkout may run it
    (root / "admission" / "controller.json").write_text("{}")
    _refusal("runtime_admission_compaction_controller_location", lambda: compaction.controller_location(
        root, [{"path": str(member)}]))


# --- T-U5, T-Q7, F1: the gate, allow-run and recovery under the gate ---

def test_allow_run_lets_runs_through_until_the_deadline_but_never_releases(x_archive, tmp_path):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    compaction_id = _verified(world, tmp_path)
    for hours in (0, 73):
        _refusal("runtime_admission_compaction_allow_run_hours", lambda: compaction.compact_allow_run(
            root, compaction_id, reason="smoke", hours=hours))
    _refusal("runtime_admission_compaction_reason", lambda: compaction.compact_allow_run(
        root, compaction_id, reason=" ", hours=1))
    clock = [1_900_000_000.0]
    granted = compaction.compact_allow_run(root, compaction_id, reason="guests tonight", hours=1,
                                           wall=lambda: clock[0])
    assert granted["hours"] == 1
    log = stage.read_compaction_log(root, compaction_id)
    assert "guests" not in json.dumps(log["events"])  # the reason is stored as a digest only
    assert cw.serve(root, tmp_path, "allowed", wall=lambda: clock[0] + 10).code == 0
    assert stage.compaction_events(stage.read_compaction_log(root, compaction_id), "allow_run_started")
    _refusal("runtime_admission_compaction_pending",
             lambda: stage.prepare_slot(x_archive, root, "release", world["h"]["state"]))
    assert cw.serve(root, tmp_path, "expired", wall=lambda: clock[0] + 3601).code == 78
    # F1 abort scope: abort-to-H is never available once an allow-run run happened.
    _refusal("runtime_admission_compaction_abort_after_allow_run", lambda: compaction.compact_abort(
        root, compaction_id, to_h=True, confirm_h=stage.read_compaction_log(root, compaction_id)["intent"][
            "source_binding_id"][:16]))
    assert compaction.compact_retire(root, compaction_id, scan=cw.no_units)["phase"] == "complete"


def test_failed_verification_run_refuses_retire_and_returns_to_h(x_archive, tmp_path):
    """V4 / C6: the armed run crashes (69); retire and resume refuse; abort --to-h restores H."""
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    started = cw.start(root, world["archive"])
    compaction_id = started["compaction_id"]
    compaction.compact_verify(root, compaction_id)
    result = cw.serve(root, tmp_path, "crash", scenario="crash", seconds=0.2)
    assert result.code in {69, 70}, result.code
    assert stage.read_admission_head(root)["status"] != "closed"
    _refusal("runtime_admission_not_closed",  # the lease itself: no destruction from a non-closed head
             lambda: compaction.compact_retire(root, compaction_id, scan=cw.no_units))
    # A second start is refused: the armed run is consumed and the head is not closed.
    assert cw.serve(root, tmp_path, "restart").code in {69, 78}
    assert stage.read_admission_head(root)["status"] != "closed"  # automatic R-b stays off while pending
    _refusal("runtime_admission_compaction_abort_confirmation", lambda: compaction.compact_abort(
        root, compaction_id, to_h=True, confirm_h="abc"))
    members = {name: cw.tree(world["states"][name]["state"]) for name in ("a", "b")}
    aborted = compaction.compact_abort(root, compaction_id, to_h=True, confirm_h=h_head["binding_id"][:16])
    assert aborted["phase"] == "aborted"
    head = stage.read_admission_head(root)
    assert head["status"] == "closed" and head["binding_id"] == h_head["binding_id"]
    assert os.readlink(root / "current") == "slots/b/runtime"
    assert not (root / "slots" / "k").exists() and not (root / "compaction" / compaction_id).exists()
    assert {name: cw.tree(world["states"][name]["state"]) for name in ("a", "b")} == members
    # Serving H again works, and a new compaction may start (K's name is retired).
    assert cw.serve(root, tmp_path, "h-again").code == 0
    _refusal("runtime_admission_retired_locator", lambda: cw.plan(root, world["archive"], k="k"))
    assert cw.plan(root, world["archive"], k="k2")["owners"] == 1


# --- T-U18 / T-Q6 / F7: abort-to-H predicate ---

_LOGIN = "\n".join([
    "import sqlite3",
    "with sqlite3.connect(state / 'data' / 'privacy.db') as db:",
    "    db.execute(\"INSERT INTO sessions VALUES ('d', 'c', NULL, 1.0, 2.0, 1)\")",
])


@pytest.mark.parametrize("during_run, code", [
    (None, None),
    ("login", None),
    ("fact_deleted", "runtime_admission_compaction_abort_revives"),
    ("voice_profile_deleted", "runtime_admission_compaction_abort_revives"),
    ("conversation_erased", "runtime_admission_compaction_abort_revives"),
    ("ordinary_rows", "runtime_admission_compaction_discard_k_writes_required"),
])
def test_abort_to_h_predicate(x_archive, tmp_path, during_run, code):
    first = [cw.sql((stage_ddl("sessions"), ()))]
    world = cw.world(x_archive, tmp_path, hops=("b",), first=first)
    root = world["root"]
    h_binding = world["h"]["binding"]
    execute = {
        None: None, "login": _LOGIN,
        "fact_deleted": cw.ledger_append("fact_deleted", "fact-1"),
        "voice_profile_deleted": cw.ledger_append("voice_profile_deleted", "profile-1"),
        "conversation_erased": cw.sql(("INSERT INTO conversation_tombstones VALUES ('conv-1', 1.0, 'owner')", ())),
        "ordinary_rows": cw.sql(("INSERT INTO live_connections VALUES ('c1', 'client', 'runtime', 'ns')", ())),
    }[during_run]
    compaction_id = _verified(world, tmp_path, execute=execute)
    action = lambda: compaction.compact_abort(root, compaction_id, to_h=True, confirm_h=h_binding[:12])
    if code is None:
        assert action()["phase"] == "aborted"
        assert stage.read_admission_head(root)["binding_id"] == h_binding
        return
    witness = cw.witness(root)
    _refusal(code, action)
    assert cw.witness(root) == witness and stage.read_admission_head(root)["status"] == "closed"
    if during_run == "ordinary_rows":
        assert compaction.compact_abort(root, compaction_id, to_h=True, discard_k_writes=True,
                                        confirm_h=h_binding[:12])["phase"] == "aborted"


def stage_ddl(name):
    columns = ", ".join(" ".join(item.split(":")) for item in stage._TABLE_COLUMNS[name].split())
    return f"CREATE TABLE IF NOT EXISTS {name} ({columns})"


# --- T-U17 and crash injection C2a-C5: abort before the run restores H from every phase ---

@pytest.mark.parametrize("phase", ["k_runtime", "worker", "authorize", "activate"])
def test_crash_then_resume_or_abort_from_each_start_phase(x_archive, tmp_path, monkeypatch, phase):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    members = {name: cw.tree(world["states"][name]["state"]) for name in ("a", "b")}
    crash = RuntimeError("synthetic-crash")
    if phase == "k_runtime":
        original = stage.append_compaction_event

        def append(root_, log, kind, data, *, at):
            if kind == "k_runtime":
                raise crash
            return original(root_, log, kind, data, at=at)
        monkeypatch.setattr(stage, "append_compaction_event", append)
    elif phase == "worker":
        def worker(root_, compaction_id):
            compaction.run_worker(root_, compaction_id)
            raise crash  # after 'materialized': resume must not redo it
        options = {"worker": worker}
    elif phase == "authorize":
        original = stage.compaction_authority_record

        def record(*args, **kwargs):
            raise crash  # between _store_binding and the authority record (C4)
        monkeypatch.setattr(stage, "compaction_authority_record", record)
    else:
        original = stage._cas_head

        def cas(root_, expected, binding_id, status, run_id=None):
            original(root_, expected, binding_id, status, run_id)
            raise crash  # after the CAS, before 'activated' (C5)
        monkeypatch.setattr(stage, "_cas_head", cas)
    with pytest.raises(RuntimeError, match="synthetic-crash"):
        cw.start(root, world["archive"], **(options if phase == "worker" else {}))
    monkeypatch.undo()
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)
    [log] = stage.compaction_logs(root)
    compaction_id = log["compaction_id"]
    if phase in {"worker", "authorize"}:
        resumed = compaction.compact_resume(root, compaction_id)
        assert resumed["phase"] == "activated"
        assert compaction.compact_resume(root, compaction_id)["noop"] is True
    aborted = compaction.compact_abort(root, compaction_id)
    assert aborted["phase"] == "aborted"
    assert compaction.compact_resume(root, compaction_id)["noop"] is True
    head = stage.read_admission_head(root)
    assert head["status"] == "closed" and head["binding_id"] == h_head["binding_id"]
    assert os.readlink(root / "current") == "slots/b/runtime"
    assert sorted(os.listdir(root / "slots")) == ["a", "b"]
    assert {name: cw.tree(world["states"][name]["state"]) for name in ("a", "b")} == members
    assert cw.serve(root, tmp_path, "h").code == 0


@pytest.mark.parametrize("phase", ["A1", "A2"])
def test_capacity_is_checked_again_before_the_head_moves(x_archive, tmp_path, monkeypatch, phase):
    """T-U16 / M6: at authorize (A1) and again at activate (A2)."""
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    options = {}
    if phase == "A1":
        def worker(root_, compaction_id):
            compaction.run_worker(root_, compaction_id)
            monkeypatch.setattr(compaction, "BINDING_MARGIN", 100)
        options["worker"] = worker
    else:
        original = compaction._authorize

        def authorize(*args, **kwargs):
            value = original(*args, **kwargs)
            monkeypatch.setattr(compaction, "BINDING_MARGIN", 100)
            return value
        monkeypatch.setattr(compaction, "_authorize", authorize)
    _refusal("runtime_admission_compaction_capacity", lambda: cw.start(root, world["archive"], **options))
    assert stage.read_admission_head(root) == h_head
    [log] = stage.compaction_logs(root)
    assert ("authorized" in _events(root, log["compaction_id"])) is (phase == "A2")
    assert "activated" not in _events(root, log["compaction_id"])


def test_d5_refuses_a_k_that_differs_from_h(x_archive, tmp_path):
    """D5 (and again after VACUUM): one changed row in K refuses before authorize; abort restores H."""
    from src.memory import owner_compaction
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)

    def hook():  # between the first D5 and VACUUM: the second D5 must catch it
        with sqlite3.connect(root / "slots" / "k" / "state" / "data" / "privacy.db") as db:
            db.execute("INSERT INTO live_connections VALUES ('x', 'client', 'runtime', 'ns')")

    def worker(root_, compaction_id):
        owner_compaction.run_materialization(root_, compaction_id, wall=lambda: 1_900_000_000.0, before_vacuum=hook)
    _refusal("runtime_admission_compaction_payload_changed:live_connections",
             lambda: cw.start(root, world["archive"], worker=worker))
    [log] = stage.compaction_logs(root)
    assert "materialized" not in _events(root, log["compaction_id"])
    assert stage.read_admission_head(root) == h_head
    assert compaction.compact_abort(root, log["compaction_id"])["phase"] == "aborted"
    assert sorted(os.listdir(root / "slots")) == ["a", "b"]


def test_ra_on_the_armed_run_records_verified_then_the_gate_refuses(x_archive, tmp_path):
    """F1: receipt-proven R-a runs before the gate; it seals the armed run, which counts as verified."""
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    started = cw.start(root, world["archive"])
    compaction_id = started["compaction_id"]
    compaction.compact_verify(root, compaction_id)
    phase_file = tmp_path / "phases-hang"
    result = cw.rehearsal.run_serve(root, cw.rehearsal.child_command(phase_file, "receipt_then_hang"),
                                    phase_file=phase_file, stop_grace=2.0,
                                    stop_when=lambda phases: any(key == "receipt" for key, _ in phases))
    assert result.code == 68 and stage.read_admission_head(root)["status"] == "unknown"
    assert "verified" not in _events(root, compaction_id)
    assert cw.serve(root, tmp_path, "after-crash").code == 78
    assert "verified" in _events(root, compaction_id)
    assert stage.read_admission_head(root)["status"] == "closed"
    assert compaction.compact_retire(root, compaction_id, scan=cw.no_units)["phase"] == "complete"


# --- C7/C8 crash injection in retire, accept-identity-change (T-U19) ---

def test_retire_crashes_resume_to_the_exact_residue(x_archive, tmp_path, monkeypatch):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    compaction_id = _verified(world, tmp_path)
    crash = RuntimeError("synthetic-crash")
    calls = []

    def hook(count):
        calls.append(count)
        if len(calls) == 3:
            raise crash  # after k of n unlinks inside one member
    with pytest.raises(RuntimeError):
        compaction.compact_retire(root, compaction_id, scan=cw.no_units, unlink_hook=hook)
    events = _events(root, compaction_id)
    assert "retire_started" in events and "complete" not in events
    assert any(path.name for path in (root / "compaction" / compaction_id / "retiring").iterdir())
    _refusal("runtime_admission_compaction_pending", lambda: stage.AdmissionController(root).__enter__())
    original = stage.append_compaction_event

    def append(root_, log, kind, data, *, at):
        if kind == "destroyed" and not getattr(append, "done", False):
            append.done = True
            raise crash  # after the tombstone, before 'destroyed'
        if kind == "complete":
            raise crash  # C8
        return original(root_, log, kind, data, at=at)
    monkeypatch.setattr(stage, "append_compaction_event", append)
    with pytest.raises(RuntimeError):
        compaction.compact_resume(root, compaction_id)
    with pytest.raises(RuntimeError):
        compaction.compact_resume(root, compaction_id)
    monkeypatch.setattr(stage, "append_compaction_event", original)
    assert "complete" not in _events(root, compaction_id)
    assert compaction.compact_resume(root, compaction_id)["phase"] == "complete"
    assert compaction.compact_resume(root, compaction_id)["noop"] is True
    assert sorted(os.listdir(root / "slots")) == ["k"]
    assert not (root / "compaction" / compaction_id).exists()  # every quarantined member was walked
    destroyed = [event for event in stage.read_compaction_log(root, compaction_id)["events"]
                 if event["kind"] == "destroyed"]
    assert len({event["data"]["member"] for event in destroyed}) == len(destroyed)


def test_identity_change_refuses_until_the_narrow_logged_acceptance(x_archive, tmp_path):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    compaction_id = _verified(world, tmp_path)
    intent = stage.read_compaction_log(root, compaction_id)["intent"]
    [member] = [item for item in intent["members"] if item.get("slot") == "a"]
    slot = Path(member["path"])
    replacement = root / "slots" / ".swap"
    os.rename(slot, replacement)
    (slot / "state").mkdir(parents=True)  # same name, different inode, nothing member-only inside
    witness = cw.witness(root)
    _refusal("runtime_admission_compaction_identity",
             lambda: compaction.compact_retire(root, compaction_id, scan=cw.no_units))
    assert cw.witness(root) == witness
    _refusal("runtime_admission_compaction_member", lambda: compaction.compact_retire(
        root, compaction_id, member=999, accept_identity_change=True, scan=cw.no_units))
    done = compaction.compact_retire(root, compaction_id, member=member["n"], accept_identity_change=True,
                                     scan=cw.no_units)
    assert done["phase"] == "complete"
    accepted = stage.compaction_events(stage.read_compaction_log(root, compaction_id), "identity_accepted")
    assert [event["data"]["member"] for event in accepted] == [member["n"]]
    assert replacement.exists() and not slot.exists()  # the swapped-out original was never a member


def test_symlink_container_member_is_never_walked(x_archive, tmp_path):
    """T-Q4-2 (stricter): a link inside a member after start refuses the walk; nothing is followed."""
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    compaction_id = _verified(world, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    (world["states"]["a"]["state"] / "data" / "link").symlink_to(outside)
    _refusal("runtime_admission_compaction_symlink",
             lambda: compaction.compact_retire(root, compaction_id, scan=cw.no_units))
    assert (outside / "keep").read_text() == "keep"
    (world["states"]["a"]["state"] / "data" / "link").unlink()
    assert compaction.compact_resume(root, compaction_id)["phase"] == "complete"
    assert (outside / "keep").read_text() == "keep"


def test_hard_link_into_k_and_cross_device_rename_refuse(x_archive, tmp_path, monkeypatch):
    """T-U15: nlink/inode-in-K before the walk, EXDEV at the rename."""
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    compaction_id = _verified(world, tmp_path)
    k_file = root / "slots" / "k" / "state" / "config" / "settings.yaml"
    link = world["states"]["a"]["state"] / "data" / "hard"
    os.link(k_file, link)
    _refusal("runtime_admission_compaction_hard_link",
             lambda: compaction.compact_retire(root, compaction_id, scan=cw.no_units))
    link.unlink()
    original = compaction.os.rename

    def rename(source, destination, *args, **kwargs):
        if "retiring" in os.fspath(destination):
            raise OSError(errno.EXDEV, "cross-device")
        return original(source, destination, *args, **kwargs)
    monkeypatch.setattr(compaction.os, "rename", rename)
    _refusal("runtime_admission_compaction_cross_device",
             lambda: compaction.compact_retire(root, compaction_id, scan=cw.no_units))
    monkeypatch.setattr(compaction.os, "rename", original)
    assert k_file.read_text().startswith("schema")
    assert compaction.compact_resume(root, compaction_id)["phase"] == "complete"


# --- proofs on a bare root (F5, F6) ---

def _bare_root(tmp_path):
    root = tmp_path / "bare"
    (root / "admission" / "bindings").mkdir(parents=True)
    (root / "admission" / "head.json").write_text("{}")
    return root


def test_marker_proof_needs_a_valid_marker_of_this_root(tmp_path):
    root = _bare_root(tmp_path)
    state = root / "slots" / "s" / "state"
    state.mkdir(parents=True)
    binding = {"slot": "s", "state_root": str(state), "runtime_root": str(state.parent / "runtime"),
               "state_identity": stage._identity(state)}
    bindings = {"b" * 64: binding}
    assert compaction._marker_proof(root, "s", state, bindings) is None  # no marker
    marker = {"admission_root": str(root), "binding_id": "b" * 64, "state_root": str(state)}
    (state / stage._MARKER).write_text(json.dumps(dict(marker, admission_root="/elsewhere")))
    assert compaction._marker_proof(root, "s", state, bindings) is None
    (state / stage._MARKER).write_text(json.dumps(dict(marker, binding_id="c" * 64)))
    assert compaction._marker_proof(root, "s", state, bindings) is None
    (state / stage._MARKER).unlink()
    (state / stage._MARKER).symlink_to(tmp_path / "elsewhere.json")
    (tmp_path / "elsewhere.json").write_text(json.dumps(marker))
    assert compaction._marker_proof(root, "s", state, bindings) is None  # a link is never a marker
    (state / stage._MARKER).unlink()
    (state / stage._MARKER).write_text(json.dumps(marker))
    assert compaction._marker_proof(root, "s", state, bindings)["proof"] == "binding"


def test_prepare_leftovers_are_proven_by_identity_never_by_age(tmp_path):
    root = _bare_root(tmp_path)
    old = root / "slots" / ".prepare-old"
    (old / "state").mkdir(parents=True)
    os.utime(old, (1_000_000.0, 1_000_000.0))  # decades old, and still unproven
    proven = root / "slots" / ".prepare-new"
    proven.mkdir()
    stage._atomic_record(stage.prepared_directory(root) / ".staging-ab.json", {
        "schema_version": 1, "kind": "staging", "staging_id": "ab", "path": str(proven),
        "identity": stage._identity(proven), "compaction_id": None, "created_at": 2.0}, immutable=True)
    members, untouched = compaction.retirement_set(root, k_slot="k", retire_prepare_leftovers=True)
    assert [Path(member["path"]).name for member in members] == [".prepare-new"]
    assert {entry["path"]: entry["class"] for entry in untouched}["slots/.prepare-old"] == "possible_state_copy"
    members, untouched = compaction.retirement_set(root, k_slot="k", retire_prepare_leftovers=False)
    assert members == [] and "slots/.prepare-new" in {entry["path"] for entry in untouched}


# --- the walk on plain trees (T-U15) ---

def _member_tree(tmp_path):
    slot = tmp_path / "slot"
    (slot / "runtime").mkdir(parents=True)
    (slot / "state" / "data").mkdir(parents=True)
    (slot / "state" / "data" / "a.db").write_bytes(b"x")
    for name in ("config", "data"):
        (slot / "runtime" / name).symlink_to("../state/" + name)
    (slot / "runtime" / "code.pyc").write_bytes(b"c")
    os.chmod(slot / "runtime", 0o500)
    return slot


def test_walk_unlinks_only_the_runtime_links_and_refuses_everything_else(tmp_path, monkeypatch):
    slot = _member_tree(tmp_path)
    assert compaction._unlink_tree(tmp_path, slot, k_inodes=set(), runtime_links=True) == 2
    assert not slot.exists()
    for case in ("symlink", "fifo", "nlink", "k_inode", "mount"):
        slot = _member_tree(tmp_path / case)
        target = slot / "state" / "data"
        if case == "symlink":
            (target / "link").symlink_to(tmp_path)
            code = "runtime_admission_compaction_symlink"
        elif case == "fifo":
            os.mkfifo(target / "pipe")
            code = "runtime_admission_compaction_special_file"
        elif case == "nlink":
            os.link(target / "a.db", tmp_path / case / "other")
            code = "runtime_admission_compaction_hard_link"
        elif case == "k_inode":
            info = os.stat(target / "a.db")
            code = "runtime_admission_compaction_hard_link"
        else:
            original = compaction.os.stat

            def stat(path, *args, **kwargs):
                info = original(path, *args, **kwargs)
                if path == "data" and kwargs.get("dir_fd") is not None:
                    values = list(info)
                    values[2] = info.st_dev + 1
                    return os.stat_result(values)
                return info
            monkeypatch.setattr(compaction.os, "stat", stat)
            code = "runtime_admission_compaction_mount"
        k_inodes = {(info.st_dev, info.st_ino)} if case == "k_inode" else set()
        _refusal(code, lambda: compaction._unlink_tree(tmp_path, slot, k_inodes=k_inodes, runtime_links=True))
        monkeypatch.undo()
        assert (tmp_path / case / "slot" / "state").exists()
    slot = _member_tree(tmp_path / "no-links")
    _refusal("runtime_admission_compaction_symlink",
             lambda: compaction._unlink_tree(tmp_path, slot, k_inodes=set(), runtime_links=False))


def test_walk_fsyncs_every_directory_and_the_parent(tmp_path, monkeypatch):
    slot = _member_tree(tmp_path)
    synced = []
    original = compaction.os.fsync

    def fsync(descriptor):
        synced.append(os.fstat(descriptor).st_ino)
        return original(descriptor)
    parent = os.stat(tmp_path).st_ino
    inner = {os.stat(path).st_ino for path in (slot, slot / "state", slot / "state" / "data", slot / "runtime")}
    monkeypatch.setattr(compaction.os, "fsync", fsync)
    compaction._unlink_tree(tmp_path, slot, k_inodes=set(), runtime_links=True)
    observed = set(synced)
    assert parent in observed and inner <= observed


# --- the compaction worker capability (G1) ---

def test_worker_command_is_the_entry_guard_bootstrap_and_needs_its_capability(tmp_path):
    root = tmp_path / "root"
    command = compaction.worker_command(root, "0" * 32)
    runtime = Path(stage.__file__).resolve().parents[1]
    assert command == entry_guard.compaction_worker_command(sys.executable, runtime, "--root", str(root),
                                                            "--compaction", "0" * 32, "--phase", "materialize")
    environment = {key: value for key, value in os.environ.items() if key != entry_guard.COMPACTION_WORKER_ENV}
    refused = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
    assert refused.returncode == 78 and refused.stderr.strip().endswith(entry_guard.CAPABILITY_CODE)
    altered = command[:4] + [command[4] + "\n"] + command[5:]
    refused = subprocess.run(altered, capture_output=True, text=True, timeout=60,
                             env=dict(environment, KZA_COMPACTION_WORKER="1"))
    assert refused.returncode != 0
    from src.core import compaction_worker
    slots = tmp_path / "r2"
    assert compaction_worker.code_location_refusal(slots) is None


# --- canaries, VACUUM and free pages (T-U13, T-U14, F3) ---

def test_free_page_canaries_are_gone_from_k_and_member_only_rows_need_acceptance(x_archive, tmp_path):
    canary = "CANARY-DELETED-IN-H-" + "x" * 64
    insert = cw.sql(("INSERT INTO live_connections VALUES (?, 'client', 'runtime', 'ns')", (canary,)))
    world = cw.world(x_archive, tmp_path, hops=("b",), first=[insert])
    root = world["root"]
    # H (b) removes the row a still holds (K5: rows erased in H survive in older slots before compaction).
    cw.run(root, "b", cw.sql(("DELETE FROM live_connections WHERE connection_id=?", (canary,))))
    needle = canary.encode()
    assert any(hit.startswith("slots/a/") for hit in _raw_hits(root, needle))
    planned = cw.plan(root, world["archive"])
    assert planned["coverage"]["unexplained"] >= 1
    _refusal("runtime_admission_compaction_owner_only_data", lambda: cw.start(root, world["archive"],
                                                                              planned=planned))
    started = compaction.compact_start(root, "k", world["archive"], plan_sha256=planned["plan_sha256"],
                                       acknowledge_untouched=planned["untouched_sha256"],
                                       destroy_rollback_states=True, accept_owner_only_loss=True,
                                       scan=cw.no_units)
    k_hits = {hit for hit in _raw_hits(root, needle) if hit.startswith("slots/k/")}
    assert k_hits == set()  # VACUUM on K: no free-page residue of rows deleted in H
    log = stage.read_compaction_log(root, started["compaction_id"])
    assert stage.compaction_events(log, "materialized")[0]["data"]["vacuumed"] == ["data/privacy.db"]
    compaction.compact_verify(root, started["compaction_id"])
    assert cw.serve(root, tmp_path, "verify").code == 0
    compaction.compact_retire(root, started["compaction_id"], scan=cw.no_units)
    assert _raw_hits(root, needle) == set()


def test_vacuum_refuses_any_path_outside_k(tmp_path):
    from src.memory import owner_compaction
    (tmp_path / "k").mkdir()
    (tmp_path / "member").mkdir()
    plan = type("Plan", (), {"latest": type("Snapshot", (), {"manifest": {"files": {}}})()})()
    _refusal("runtime_admission_compaction_vacuum_scope", lambda: owner_compaction.vacuum_compacted_state(
        plan, tmp_path / "member", tmp_path / "k"))
    assert owner_compaction.vacuum_compacted_state(plan, tmp_path / "k", tmp_path / "k") == []


# --- G2: the ledger stays constant-size in K's binding ---

def test_k_binding_size_is_constant_in_the_deletion_ledger_size(x_archive, tmp_path):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    cw.run(root, "b", cw.ledger_append("fact_deleted", "one"))
    small = cw.plan(root, world["archive"])["capacity"]["binding_bytes"]
    bulk = "\n".join([
        "import sqlite3",
        "from contextlib import closing",
        "from src.memory.deletion_ledger import DeletionKind, append_deletion",
        "with closing(sqlite3.connect(state / 'data' / 'privacy.db', isolation_level=None)) as db:",
        "    db.execute('BEGIN IMMEDIATE')",
        "    for index in range(20000):",
        "        append_deletion(db, DeletionKind.FACT_DELETED, 'fact-%d' % index, 1.0 + index)",
        "    db.execute('COMMIT')",
    ])
    cw.run(root, "b", bulk, timeout=300)
    large = cw.plan(root, world["archive"])["capacity"]["binding_bytes"]
    assert abs(large - small) <= 16
    started = cw.start(root, world["archive"])
    k = stage.read_binding(root, stage.read_admission_head(root)["binding_id"])
    assert abs(len(stage._canonical_json(k)) - large) <= 64
    with sqlite3.connect(root / "slots" / "k" / "state" / "data" / "privacy.db") as db:
        assert db.execute("SELECT COUNT(*) FROM deletion_ledger_v1").fetchone()[0] == 20001
    assert started["phase"] == "activated"


# --- T-U8, T-U7b: retired owners and inode recycling ---

def test_owner_retired_and_inode_recycling_guards(x_archive, tmp_path):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    a = world["states"]["a"]
    paths, identities = stage._delegation_states(root)
    a_identity = stage._canonical_json(stage.read_binding(root, a["binding"])["state_identity"])
    assert str(a["state"]) in paths and a_identity in identities
    stage._atomic_record(stage.tombstone_directory(root) / "synthetic.json", {
        "schema_version": 1, "compaction_id": "0" * 32, "member": 1, "class": "managed_slot",
        "path": str(a["state"].parent), "identity": stage._identity(a["state"].parent), "slot": "a",
        "state_root": str(a["state"]), "state_identity": stage._identity(a["state"]), "binding_ids": [a["binding"]],
        "copy_ids": [], "status": "destroyed"}, immutable=True)
    paths, identities = stage._delegation_states(root)
    assert str(a["state"]) in paths and a_identity not in identities  # H1
    binding = stage.read_binding(root, world["h"]["binding"])
    _refusal("runtime_admission_delegation_owner_retired",
             lambda: stage._target_ownership_validation(root, binding))


# --- the advisory (T-Q1, F4, G3) ---

def _advisory_state(tmp_path, name, *, facts=(), deleted=(), ledger=(), profiles=()):
    from src.memory.deletion_ledger import DeletionKind, append_deletion, initialize_deletion_ledger
    state = tmp_path / name
    (state / "data").mkdir(parents=True)
    with sqlite3.connect(state / "data" / "privacy.db", isolation_level=None) as db:
        db.execute("CREATE TABLE facts(id TEXT PRIMARY KEY, state TEXT)")
        db.execute("CREATE TABLE voice_profiles(id TEXT PRIMARY KEY, deleted_at REAL)")
        db.execute("CREATE TABLE conversation_tombstones(conversation_id TEXT PRIMARY KEY, deleted_at REAL)")
        for identity in facts:
            db.execute("INSERT INTO facts VALUES (?, 'automatic')", (identity,))
        for identity in deleted:
            db.execute("INSERT INTO facts VALUES (?, 'deleted')", (identity,))
        for identity in profiles:
            db.execute("INSERT INTO voice_profiles VALUES (?, NULL)", (identity,))
        if ledger is not None:
            initialize_deletion_ledger(db)
            db.execute("BEGIN IMMEDIATE")
            for kind, subject in ledger:
                append_deletion(db, DeletionKind(kind), subject, 1.0)
            db.execute("COMMIT")
    return state


def test_advisory_counts_ledger_deletions_and_never_absences(tmp_path):
    older = _advisory_state(tmp_path, "older", facts=("f1", "f2"), profiles=("p1",))
    expiry = compaction.advisory_counts(_advisory_state(tmp_path, "h1"), [older], owners=1)
    assert expiry["compaction_recommended"] is False and expiry["absent"]["facts"] == 2
    assert expiry["ledger"] == "present"
    deleted = compaction.advisory_counts(_advisory_state(tmp_path, "h2", facts=("f2",), ledger=[
        ("fact_deleted", "f1"), ("voice_profile_deleted", "p1")]), [older], owners=1)
    assert deleted["compaction_recommended"] is True
    reasons = {item["kind"]: item["pending"] for item in deleted["reasons"]}
    assert reasons == {"memory_deleted": 1, "voice_profile_deleted": 1}
    assert deleted["absent"]["facts"] == 0
    # Equal counts with different id sets are still detected (F4).
    swapped = compaction.advisory_counts(_advisory_state(tmp_path, "h3", facts=("f2", "f3"), ledger=[
        ("fact_deleted", "f1")]), [older], owners=1)
    assert {item["kind"] for item in swapped["reasons"]} == {"memory_deleted"}
    legacy = compaction.advisory_counts(_advisory_state(tmp_path, "h4", deleted=("f1",), ledger=None), [older],
                                        owners=6)
    assert legacy["ledger"] == "ledger_absent"
    assert {item["kind"] for item in legacy["reasons"]} == {"owner_count", "memory_deleted"}
    text = json.dumps([expiry, deleted, swapped, legacy])
    assert "f1" not in text and "p1" not in text


def test_delegate_output_carries_the_advisory(x_archive, tmp_path):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    advice = world["states"]["b"]["delegate"]["compaction_advisory"]
    assert advice["available"] is True and advice["owners"] == 1 and advice["compaction_recommended"] is False
