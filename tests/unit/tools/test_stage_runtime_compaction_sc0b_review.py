"""SC-0b review round 1 (F3, F4) on real managed roots: D5 refusals in each pass, the snapshot's P9 check.

Same worlds as test_stage_runtime_compaction (X compiled from HEAD plus this
working tree, real closed runs, the real worker in process). Each D5 case
corrupts K once, either right after the column relocations (so the first
D5, before VACUUM, must refuse) or in the before_vacuum hook (so the second
D5, after VACUUM, must refuse), and checks which pass refused.
"""

from contextlib import closing
import json
import sqlite3
from types import SimpleNamespace

import pytest

from scripts import compaction
from scripts import stage_runtime as stage
from src.memory import owner_compaction as oc
from tests.mocks import compaction_world as cw
from tests.unit.tools.test_stage_runtime_compaction_review import _A_CLIPS

WALL = 1_900_000_000.0


@pytest.fixture(scope="module")
def x_archive(tmp_path_factory):
    return cw.build_x(tmp_path_factory.mktemp("compaction-sc0b-review-x"))


@pytest.fixture(autouse=True)
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold for an in-process controller."""
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)


def _refusal(code, action):
    with pytest.raises(ValueError) as caught:
        action()
    assert str(caught.value) == code, str(caught.value)
    return caught.value


def _events(root, compaction_id):
    return [event["kind"] for event in stage.read_compaction_log(root, compaction_id)["events"]]


def _k_privacy(root):
    return root / "slots" / "k" / "state" / "data" / "privacy.db"


def _unrelocate_one_clip(root, h_state):
    """A wrongly relocated column: one K wake clip points at H's (owner a's) target id again."""
    with closing(sqlite3.connect(h_state / "data" / "privacy.db")) as db:
        original = dict(db.execute("SELECT id, target_id FROM wake_clips_v1").fetchall())
    with closing(sqlite3.connect(_k_privacy(root))) as db, db:
        relocated = [(identity, target) for identity, target in db.execute("SELECT id, target_id FROM wake_clips_v1")
                     if original.get(identity) not in (None, target)]
        assert relocated, "the world must hold relocated wake clips"
        identity, _target = relocated[0]
        db.execute("UPDATE wake_clips_v1 SET target_id=? WHERE id=?", (original[identity], identity))


def _bump_sqlite_sequence(root, h_state):
    """A changed AUTOINCREMENT counter of a table that is not a rebuilt scheduling table."""
    for database in sorted((root / "slots" / "k" / "state").rglob("*.db")):
        with closing(sqlite3.connect(database)) as db, db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='sqlite_sequence'").fetchone():
                continue
            names = [name for (name,) in db.execute("SELECT name FROM sqlite_sequence ORDER BY name")
                     if name not in oc.SCHEDULING_TABLES]
            if names:
                db.execute("UPDATE sqlite_sequence SET seq=seq+1 WHERE name=?", (names[0],))
                return
    raise AssertionError("the world must hold a non-scheduling AUTOINCREMENT counter")


def _drop_one_target(root, h_state):
    """erasure_targets ids that differ from the journal's registered ids (one registered target missing)."""
    with closing(sqlite3.connect(_k_privacy(root))) as db, db:
        [(identity,)] = db.execute("SELECT id FROM erasure_targets ORDER BY id LIMIT 1").fetchall()
        db.execute("DELETE FROM erasure_targets WHERE id=?", (identity,))


CORRUPTIONS = {
    "wake_clips_v1": _unrelocate_one_clip,
    "sqlite_sequence": _bump_sqlite_sequence,
    "erasure_targets": _drop_one_target,
}


@pytest.mark.parametrize("table", sorted(CORRUPTIONS))
@pytest.mark.parametrize("phase", ["before_vacuum", "after_vacuum"])
def test_d5_refuses_each_rewritten_check_in_each_pass(x_archive, tmp_path, monkeypatch, table, phase):
    """F3: wrongly relocated column, changed sqlite_sequence, erasure_targets ids != journal ids."""
    world = cw.world(x_archive, tmp_path, hops=("b",), first=(_A_CLIPS,))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    h_state = world["states"]["b"]["state"]
    corrupt = CORRUPTIONS[table]
    steps = []
    verify, vacuum = oc.verify_identity, oc.vacuum_compacted_state

    def counted(plan, destination, journal):
        steps.append("d5")
        return verify(plan, destination, journal)

    def vacuumed(*args):
        steps.append("vacuum")
        return vacuum(*args)
    monkeypatch.setattr(oc, "verify_identity", counted)
    monkeypatch.setattr(oc, "vacuum_compacted_state", vacuumed)
    hook = None
    if phase == "before_vacuum":
        relocate = oc._relocate_columns

        def relocated_then_corrupted(plan, destination):
            counts = relocate(plan, destination)
            corrupt(root, h_state)
            return counts
        monkeypatch.setattr(oc, "_relocate_columns", relocated_then_corrupted)
    else:
        def hook():  # between the first D5 and VACUUM: only the second D5 can see it
            corrupt(root, h_state)

    def worker(root_, compaction_id):
        oc.run_materialization(root_, compaction_id, wall=lambda: WALL, before_vacuum=hook)
    _refusal("runtime_admission_compaction_payload_changed:" + table,
             lambda: cw.start(root, world["archive"], worker=worker))
    assert steps == (["d5"] if phase == "before_vacuum" else ["d5", "vacuum", "d5"])
    [log] = stage.compaction_logs(root)
    assert "materialized" not in _events(root, log["compaction_id"])
    assert stage.read_admission_head(root) == h_head


def test_the_snapshot_refuses_before_its_manifest_when_the_exact_need_does_not_fit(x_archive, tmp_path,
                                                                                  monkeypatch):
    """F4: the worker's exact P9 check runs inside the snapshot, with the serialized manifest's size,
    before manifest.json is written; the refusal is resumable (the partial snapshot is destroyed)."""
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    seen, short = [], []
    check = oc._check_free_files
    usage = oc.shutil.disk_usage
    monkeypatch.setattr(oc.shutil, "disk_usage", lambda path: SimpleNamespace(free=0) if short else usage(path))

    def short_of_space(files, destination, pending_bytes=0):
        seen.append(pending_bytes)
        if pending_bytes and len(seen) == 1:
            short.append(True)  # no free space only for the first snapshot's pre-manifest check
        try:
            return check(files, destination, pending_bytes)
        finally:
            short.clear()
    monkeypatch.setattr(oc, "_check_free_files", short_of_space)

    def worker(root_, compaction_id):
        oc.run_materialization(root_, compaction_id, wall=lambda: WALL)
    _refusal("runtime_admission_compaction_capacity", lambda: cw.start(root, world["archive"], worker=worker))
    [log] = stage.compaction_logs(root)
    compaction_id = log["compaction_id"]
    events = _events(root, compaction_id)
    assert "snapshot_reserved" in events and "snapshot" not in events
    reserved = stage.read_compaction_log(root, compaction_id)["events"][events.index("snapshot_reserved")]
    snapshot = root / "compaction" / compaction_id / "snapshot-1"
    assert reserved["data"]["path"] == str(snapshot)
    assert not (snapshot / "manifest.json").exists()
    assert seen[0] > 0
    resumed = compaction.compact_resume(root, compaction_id, worker=worker)
    assert resumed["phase"] == "activated"
    events = _events(root, compaction_id)
    assert "snapshot_destroyed" in events and events.count("snapshot") == 1
    # The second snapshot checked with exactly the size it then wrote, and the post-snapshot check ran too.
    second = root / "compaction" / compaction_id / "snapshot-2" / "manifest.json"
    assert seen[1] == second.stat().st_size and seen[2] == 0
    assert json.loads(second.read_bytes())["schema_version"] == 1
