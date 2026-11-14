"""U3b-0c third independent review (R3-1, R3-3, R3-4): the reviewer's probes as tests.

probe_abort_missing.py (A1: an own-work item deleted by hand blocks the abort;
A2: a member moved by hand between 'retiring' and the rename) and
probe_owner_alive.py (N10's pid-only liveness) on light roots with the real
log writers, plus one managed world for the operator's path through
``compact abort --member n --accept-missing``. The P14 probes (R3-2, R3-5,
R3-6) are in test_compaction_units.py and test_adoption_commands.py.
"""

import os
from pathlib import Path
import shutil
import time

import pytest

from scripts import compaction
from scripts import stage_runtime as stage
from tests.mocks import compaction_world as cw
from tests.unit.tools.test_stage_runtime_compaction_review_r2 import (
    CID, HEAD, H, _Stage, _catalog, _copy_record, _dead_pid, _fill, _h_restored, _log_with)


@pytest.fixture(scope="module")
def x_archive(tmp_path_factory):
    return cw.build_x(tmp_path_factory.mktemp("compaction-review-r3-x"))


@pytest.fixture(autouse=True)
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold for an in-process controller."""
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)


def _refusal(code, action):
    with pytest.raises(ValueError) as caught:
        action()
    assert str(caught.value) == code or str(caught.value).startswith(code + ":"), str(caught.value)
    return caught.value


def _snapshot_root(tmp_path: Path) -> tuple[Path, Path]:
    """probe_abort_missing's world: a completed snapshot, K, and 'abort_started' (head recorded)."""
    root = tmp_path.resolve() / "root"
    (root / "admission").mkdir(parents=True)
    stage._atomic_record(root / "admission" / "head.json", HEAD)
    (root / "slots" / "h" / "runtime").mkdir(parents=True)
    (root / "current").symlink_to("slots/h/runtime")
    _fill(root / "slots" / "k", 3)
    snapshot = root / "compaction" / CID / "snapshot-1"
    _fill(snapshot, 2)
    intent = {"schema_version": 1, "compaction_id": CID, "kind": "compaction_intent", "head": HEAD,
              "k_slot": "k", "members": [], "source_binding_id": H,
              "k_state_root": str(root / "slots" / "k" / "state")}
    _log_with(root, intent, [
        ("snapshot_reserved", {"n": 1, "path": str(snapshot)}),
        ("snapshot", {"n": 1, "path": str(snapshot), "identity": compaction._identity(snapshot),
                      "manifest_sha256": "0" * 64}),
        ("k_runtime", {"slot": "k", "slot_identity": compaction._identity(root / "slots" / "k")}),
        ("abort_started", {"to_h": False, "head": HEAD})])
    return root, snapshot


def _log(root):
    return stage.read_compaction_log(root, CID)


def _accept(root, number):
    log = _log(root)
    return compaction._accept_missing(root, log, number, lambda: 3.0, members=compaction._abort_members(root, log))


# --- R3-1: an own-work item deleted by hand no longer blocks the abort ------------------------------

def test_an_abort_accepts_a_snapshot_deleted_by_hand_and_claims_no_destruction(tmp_path, monkeypatch):
    monkeypatch.setattr(compaction, "stage", _Stage())
    root, snapshot = _snapshot_root(tmp_path)
    shutil.rmtree(snapshot)
    for _attempt in (1, 2):
        refusal = _refusal("runtime_admission_compaction_member_missing",
                           lambda: compaction._finish_abort(root, _log(root), lambda: 2.0))
    number = refusal.report["member"]
    assert (root / "slots" / "k").is_dir()  # nothing destroyed while it refuses
    _refusal("runtime_admission_compaction_member", lambda: _accept(root, compaction.K_ABORT_NUMBER))  # K exists
    _refusal("runtime_admission_compaction_member", lambda: _accept(root, 999))  # not an abort member
    _accept(root, number)
    result = compaction._finish_abort(root, _log(root), lambda: 4.0)
    assert result["phase"] == "aborted" and result["destroyed"] == 1
    log = _log(root)
    assert [event["data"]["member"] for event in stage.compaction_events(log, "destroyed")] == [
        compaction.K_ABORT_NUMBER]
    assert [event["data"]["member"] for event in stage.compaction_events(log, "missing_accepted")] == [number]
    tombstones = list(stage.tombstone_directory(root).iterdir())
    assert len(tombstones) == 1 and not (root / "slots" / "k").exists()


def test_compact_abort_member_accept_missing_on_a_managed_root(x_archive, tmp_path):
    """R3-1 through the operator's command: snapshot and placeholder deleted by hand, then abort."""
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    members = {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")}
    compaction_id = cw.start(root, world["archive"])["compaction_id"]
    scratch = root / "compaction" / compaction_id
    removed = sorted(name for name in os.listdir(scratch))
    assert "placeholder" in removed and any(name.startswith("snapshot-") for name in removed)
    for name in removed:
        shutil.rmtree(scratch / name)
    accepted, number = [], None
    for _round in range(len(removed) + 1):
        options = {} if number is None else {"member": number, "accept_missing": True}
        try:
            compaction.compact_abort(root, compaction_id, **options)
            break
        except compaction.CompactionRefusal as refusal:
            assert str(refusal) == "runtime_admission_compaction_member_missing"
            assert refusal.report["member"] != number  # the acknowledged one is left out
            number = refusal.report["member"]
            accepted.append(number)
    # compact_abort with the acknowledgement finishes the abort itself once nothing else is missing.
    log = stage.read_compaction_log(root, compaction_id)
    assert log["terminal"] and log["events"][-1]["kind"] == "aborted"
    assert sorted(event["data"]["member"] for event in stage.compaction_events(log, "missing_accepted")) == \
        sorted(accepted) and len(accepted) == len(removed)
    assert [event["data"]["member"] for event in stage.compaction_events(log, "destroyed")] == [
        len(log["intent"]["members"]) + compaction.K_ABORT_NUMBER]
    _h_restored(world, tmp_path, h_head, members)


# --- R3-4: 'retiring' alone (before the rename) never proves the walk ran -------------------------

def test_a_member_moved_between_retiring_and_the_rename_is_not_reported_destroyed(tmp_path, monkeypatch):
    """probe_abort_missing A2: before r3 the resumed abort wrote a tombstone for the moved snapshot."""
    monkeypatch.setattr(compaction, "stage", _Stage())
    root, snapshot = _snapshot_root(tmp_path)
    [own] = compaction.own_work(root, _log(root))
    stage.append_compaction_event(root, _log(root), "retiring", {"member": own["n"], "identity": own["identity"]},
                                  at=1.5)
    kept = root / "kept-snapshot"
    snapshot.rename(kept)
    refusal = _refusal("runtime_admission_compaction_member_missing",
                       lambda: compaction._finish_abort(root, _log(root), lambda: 2.0))
    assert refusal.report["member"] == own["n"]
    assert not stage.tombstone_directory(root).exists() or not list(stage.tombstone_directory(root).iterdir())
    _accept(root, own["n"])  # 'retiring' without 'renamed': this compaction never moved it
    assert compaction._finish_abort(root, _log(root), lambda: 3.0)["phase"] == "aborted"
    assert len(list(stage.tombstone_directory(root).iterdir())) == 1  # K only
    assert len(list(kept.rglob("*.db"))) == 2  # the moved copy is left where the operator put it


def test_a_cut_between_the_rename_and_renamed_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(compaction, "stage", _Stage())
    root, _snapshot = _snapshot_root(tmp_path)
    original = stage.append_compaction_event

    def append(root_, log, kind, data, *, at):
        if kind == "renamed" and not getattr(append, "done", False):
            append.done = True
            raise SystemExit("cut")  # renamed into the quarantine, 'renamed' not yet logged
        return original(root_, log, kind, data, at=at)
    monkeypatch.setattr(stage, "append_compaction_event", append)
    with pytest.raises(SystemExit):
        compaction._finish_abort(root, _log(root), lambda: 2.0)
    monkeypatch.setattr(stage, "append_compaction_event", original)
    log = _log(root)
    number = stage.compaction_events(log, "retiring")[-1]["data"]["member"]
    assert not stage.compaction_events(log, "renamed")
    _refusal("runtime_admission_compaction_member", lambda: _accept(root, number))  # it is in the quarantine
    assert compaction._finish_abort(root, _log(root), lambda: 3.0)["phase"] == "aborted"
    log = _log(root)
    assert {event["data"]["member"] for event in stage.compaction_events(log, "renamed")} == \
        {event["data"]["member"] for event in stage.compaction_events(log, "destroyed")}
    assert not list(root.rglob("*.db"))


def test_accept_missing_refuses_after_renamed(tmp_path, monkeypatch):
    monkeypatch.setattr(compaction, "stage", _Stage())
    root, snapshot = _snapshot_root(tmp_path)
    [own] = compaction.own_work(root, _log(root))
    log = stage.append_compaction_event(root, _log(root), "retiring", {"member": own["n"],
                                                                       "identity": own["identity"]}, at=1.5)
    stage.append_compaction_event(root, log, "renamed", {"member": own["n"], "identity": own["identity"]}, at=1.6)
    shutil.rmtree(snapshot)  # renamed and walked by this compaction (absent everywhere): a destruction
    _refusal("runtime_admission_compaction_member", lambda: _accept(root, own["n"]))
    assert compaction._finish_abort(root, _log(root), lambda: 3.0)["phase"] == "aborted"
    assert len(list(stage.tombstone_directory(root).iterdir())) == 2


# --- R3-3: the owner check resists pid reuse; stale records never block (probe_owner_alive) --------

def test_a_reused_pid_is_not_the_owner_that_reserved_the_copy():
    me = {"process_id": os.getpid()}
    now = time.time()
    started = compaction.process_started_at(os.getpid())
    assert started is not None and started <= now + 1
    assert compaction._owner_alive(me, now) is True  # this process reserved it 'now': still the owner
    assert compaction._owner_alive(me, started - 3600) is False  # reserved before this process existed
    assert compaction._owner_alive({"process_id": _dead_pid()}, now) is False
    assert compaction._owner_alive(me, None) is False  # no reservation time: abandoned, never blocking


def test_the_reservation_time_is_the_first_revision_file(tmp_path):
    root = tmp_path.resolve() / "root"
    destination = root / "backups" / "copy"
    destination.mkdir(parents=True)
    _catalog(root, _copy_record(destination, preparation="writing",
                                preparation_owner={"process_id": os.getpid()}))
    [revision] = [path for path in (root / "admission" / "copies" / ("1" * 32)).iterdir()
                  if path.name != "head.json"]
    os.utime(revision, (1000.0, 1000.0))
    [record] = compaction.catalog_records(root)
    assert record["reserved_at"] == 1000.0
    assert compaction._owner_alive(record["preparation_owner"], record["reserved_at"]) is False
