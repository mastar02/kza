"""U3b-0c fourth independent review (R4-2, R4-6, R4-7): the reviewer's probes as tests.

probe_abort4.py (A: K moved by hand after it was built; B4: the quarantine moved
out after 'renamed'; B7/B8: an item accepted as missing put back) and
probe_owner_alive4.py (ps lstart read in the caller's TZ) on light roots with
the real log writers, plus one managed world for ``compact abort --member n
--accept-missing`` on K. The P14 probes (R4-1, R4-3, R4-4, R4-5) are in
test_compaction_units.py and test_adoption_commands.py.
"""

import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

from scripts import compaction
from scripts import stage_runtime as stage
from tests.mocks import compaction_world as cw
from tests.unit.tools.test_stage_runtime_compaction_review_r2 import _h_restored, _Stage
from tests.unit.tools.test_stage_runtime_compaction_review_r3 import _accept, _log, _refusal, _snapshot_root


@pytest.fixture(scope="module")
def x_archive(tmp_path_factory):
    return cw.build_x(tmp_path_factory.mktemp("compaction-review-r4-x"))


@pytest.fixture(autouse=True)
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold for an in-process controller."""
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)


def _finish(root):
    return compaction._finish_abort(root, _log(root), lambda: 2.0)


def _tombstones(root) -> int:
    directory = stage.tombstone_directory(root)
    return len(list(directory.iterdir())) if directory.exists() else 0


def _events(root, kind) -> list:
    return [event["data"]["member"] for event in stage.compaction_events(_log(root), kind)]


def _private_files(path: Path) -> int:
    return len(list(path.rglob("*.db")))


# --- R4-2: K moved by hand after it was built is never skipped silently -----------------------------

def test_k_moved_by_hand_after_it_was_built_refuses_and_can_be_accepted(tmp_path, monkeypatch):
    """probe_abort4 A/B8: before r4 the abort returned 'aborted' with K's private copy left at the moved
    path, no record of it, and '--member <K> --accept-missing' refused compaction_member."""
    monkeypatch.setattr(compaction, "stage", _Stage())
    root, snapshot = _snapshot_root(tmp_path)
    k_number = compaction.K_ABORT_NUMBER  # the intent has no members
    kept = root / "backups-k"
    (root / "slots" / "k").rename(kept)
    assert [member["n"] for member in compaction._abort_members(root, _log(root))][-1] == k_number
    refusal = _refusal("runtime_admission_compaction_member_missing", lambda: _finish(root))
    assert refusal.report["member"] == k_number
    _accept(root, k_number)
    result = _finish(root)
    assert result["phase"] == "aborted" and result["destroyed"] == 1  # the snapshot only
    assert _events(root, "missing_accepted") == [k_number] and k_number not in _events(root, "destroyed")
    assert _tombstones(root) == 1 and _private_files(kept) == 3  # K's copy is where the operator put it


def test_k_never_built_is_not_an_abort_member(tmp_path, monkeypatch):
    monkeypatch.setattr(compaction, "stage", _Stage())
    root, _snapshot = _snapshot_root(tmp_path)
    log = _log(root)
    log["events"] = [event for event in log["events"] if event["kind"] != "k_runtime"]
    shutil.rmtree(root / "slots" / "k")
    assert compaction._k_abort_member(root, log) is None


def test_compact_abort_accepts_k_moved_by_hand_on_a_managed_root(x_archive, tmp_path):
    """R4-2 through the operator's command, after a start that built K."""
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    members = {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")}
    compaction_id = cw.start(root, world["archive"])["compaction_id"]
    log = stage.read_compaction_log(root, compaction_id)
    k_slot = root / "slots" / log["intent"]["k_slot"]
    k_number = len(log["intent"]["members"]) + compaction.K_ABORT_NUMBER
    kept = tmp_path / "k-kept-by-hand"
    k_slot.rename(kept)
    with pytest.raises(compaction.CompactionRefusal) as caught:
        compaction.compact_abort(root, compaction_id)
    assert str(caught.value) == "runtime_admission_compaction_member_missing"
    assert caught.value.report["member"] == k_number
    compaction.compact_abort(root, compaction_id, member=k_number, accept_missing=True)
    log = stage.read_compaction_log(root, compaction_id)
    assert log["terminal"] and log["events"][-1]["kind"] == "aborted"
    assert [event["data"]["member"] for event in stage.compaction_events(log, "missing_accepted")] == [k_number]
    assert k_number not in [event["data"]["member"] for event in stage.compaction_events(log, "destroyed")]
    assert kept.is_dir()
    _h_restored(world, tmp_path, h_head, members)


# --- R4-6: only the end of the walk proves a destruction; an accepted item put back refuses clearly ---

def _renamed_into_quarantine(root, snapshot):
    [own] = compaction.own_work(root, _log(root))
    log = stage.append_compaction_event(root, _log(root), "retiring", {"member": own["n"],
                                                                       "identity": own["identity"]}, at=1.5)
    quarantine = compaction._quarantine(root, log["compaction_id"], own)
    quarantine.parent.mkdir(parents=True)
    snapshot.rename(quarantine)
    stage.append_compaction_event(root, log, "renamed", {"member": own["n"], "identity": own["identity"]}, at=1.6)
    return own, quarantine


def test_a_quarantine_moved_out_after_renamed_is_not_reported_destroyed(tmp_path, monkeypatch):
    """probe_abort4 B4: before r4 the resumed abort tombstoned it while its files survived."""
    monkeypatch.setattr(compaction, "stage", _Stage())
    root, snapshot = _snapshot_root(tmp_path)
    own, quarantine = _renamed_into_quarantine(root, snapshot)
    (quarantine / "state" / "data" / "f0.db").unlink()  # half-walked
    kept = root / "kept-from-quarantine"
    quarantine.rename(kept)
    refusal = _refusal("runtime_admission_compaction_member_missing", lambda: _finish(root))
    assert refusal.report["member"] == own["n"] and _tombstones(root) == 0
    _accept(root, own["n"])  # renamed, never walked: accepted, nothing claims it destroyed
    assert _finish(root)["phase"] == "aborted"
    assert own["n"] not in _events(root, "destroyed") and _tombstones(root) == 1  # K only
    assert _private_files(kept) == 1


def test_walked_is_logged_while_the_emptied_quarantine_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(compaction, "stage", _Stage())
    root, snapshot = _snapshot_root(tmp_path)
    own, quarantine = _renamed_into_quarantine(root, snapshot)
    original = stage.append_compaction_event
    seen = {}

    def append(root_, log, kind, data, *, at):
        if kind == "walked":
            seen[data["member"]] = (os.path.isdir(quarantine) and not os.listdir(quarantine)) \
                if data["member"] == own["n"] else True
            if data["member"] == own["n"] and not getattr(append, "cut", False):
                append.cut = True
                original(root_, log, kind, data, at=at)
                raise SystemExit("cut after 'walked', before the quarantine's rmdir")
        return original(root_, log, kind, data, at=at)
    monkeypatch.setattr(stage, "append_compaction_event", append)
    with pytest.raises(SystemExit):
        _finish(root)
    monkeypatch.setattr(stage, "append_compaction_event", original)
    assert seen[own["n"]] is True and quarantine.is_dir()
    _refusal("runtime_admission_compaction_member", lambda: _accept(root, own["n"]))  # walked: a destruction
    assert _finish(root)["phase"] == "aborted"
    assert sorted(_events(root, "destroyed")) == sorted(_events(root, "walked")) and _tombstones(root) == 2
    assert not list(root.rglob("*.db"))


def test_walked_then_gone_is_a_destruction(tmp_path, monkeypatch):
    monkeypatch.setattr(compaction, "stage", _Stage())
    root, snapshot = _snapshot_root(tmp_path)
    own, quarantine = _renamed_into_quarantine(root, snapshot)
    shutil.rmtree(quarantine)
    stage.append_compaction_event(root, _log(root), "walked", {"member": own["n"], "identity": own["identity"]},
                                  at=1.7)
    assert _finish(root)["phase"] == "aborted" and own["n"] in _events(root, "destroyed")


@pytest.mark.parametrize("item", ["snapshot", "k"])
def test_an_item_accepted_as_missing_and_put_back_refuses_with_its_path(tmp_path, monkeypatch, item):
    """probe_abort4 B7: before r4 a snapshot put back refused only retiring_residue (and on retire, a member
    back at its path refused compaction_identity), naming neither the item nor what to do."""
    monkeypatch.setattr(compaction, "stage", _Stage())
    root, snapshot = _snapshot_root(tmp_path)
    if item == "snapshot":
        [own] = compaction.own_work(root, _log(root))
        number, path = own["n"], snapshot
    else:
        number, path = compaction.K_ABORT_NUMBER, root / "slots" / "k"
    away = root / "away"
    path.rename(away)
    _accept(root, number)
    away.rename(path)
    refusal = _refusal("runtime_admission_compaction_accepted_member_back", lambda: _finish(root))
    assert refusal.report == {"member": number, "path": str(path)}
    path.rename(away)  # the runbook: move it away again (or delete it by hand), then repeat
    assert _finish(root)["phase"] == "aborted" and number not in _events(root, "destroyed")
    assert _private_files(away) in (2, 3)


# --- R4-7: the start time does not depend on the caller's TZ (probe_owner_alive4) -------------------

@pytest.mark.parametrize("zone", ["UTC0", "Pacific/Kiritimati", "America/Montevideo"])
def test_process_started_at_reads_ps_in_utc_whatever_the_callers_zone(monkeypatch, zone):
    """ps prints lstart in its own TZ; read without it, a TZ in the caller's environment shifted the start
    by the zone difference (-10800 s with TZ=UTC on a -03 Mac). The ps reading is forced on Linux too."""
    monkeypatch.setattr(compaction, "PROC_SELF", "/nonexistent-proc/self")
    original = os.environ.get("TZ")
    os.environ["TZ"] = zone
    time.tzset()
    try:
        before = time.time()
        child = subprocess.Popen(["sleep", "30"])
        try:
            started = compaction.process_started_at(child.pid)
        finally:
            child.kill()
            child.wait()
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()
    assert started is not None and before - 2 <= started <= before + 3, (zone, started - before)
