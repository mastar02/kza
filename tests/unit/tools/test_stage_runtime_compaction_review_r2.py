"""U3b-0c second independent review (N1-N10): the reviewer's probes as tests.

The probes (0c-review2/probe_*.py) ran the real writers on small roots; the
light tests here do the same (a hash-chained log, a head, a few directories).
The world tests use the managed roots of test_stage_runtime_compaction (X
compiled from HEAD plus this working tree, real closed runs, in-process
serve under a fake manager). The P14 probes (N4-N6) are in
test_compaction_units.py and test_adoption_commands.py.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import compaction
from scripts import stage_runtime as stage
from tests.mocks import compaction_world as cw

H = "a" * 64
K = "b" * 64
CID = "c" * 32
HEAD = {"schema_version": 1, "generation": 5, "binding_id": H, "status": "closed", "run_id": None}


@pytest.fixture(scope="module")
def x_archive(tmp_path_factory):
    return cw.build_x(tmp_path_factory.mktemp("compaction-review-r2-x"))


@pytest.fixture(autouse=True)
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold for an in-process controller."""
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)


def _refusal(code, action):
    with pytest.raises(ValueError) as caught:
        action()
    assert str(caught.value) == code or str(caught.value).startswith(code + ":"), str(caught.value)
    return caught.value


def _events(root, compaction_id):
    return [event["kind"] for event in stage.read_compaction_log(root, compaction_id)["events"]]


class _Stage:
    """stage_runtime, except H's binding (a light root has no bindings): the probes' proxy."""

    def __getattr__(self, name):
        return getattr(stage, name)

    def read_binding(self, root, binding_id):
        return {"slot": "h", "state_root": str(Path(root) / "slots" / "h" / "state")}


def _log_with(root: Path, intent: dict, events) -> dict:
    stage._atomic_record(stage.compaction_log_path(root, CID) / "intent.json", intent, immutable=True)
    log = stage.read_compaction_log(root, CID)
    for kind, data in events:
        log = stage.append_compaction_event(root, log, kind, data, at=1.0)
    return log


def _catalog(root: Path, record: dict) -> None:
    """One admission/copies chain of one revision, as src.memory.copy_catalog writes it."""
    raw = json.dumps({"record": record, "previous_sha256": None}).encode()
    digest = hashlib.sha256(raw).hexdigest()
    entry = root / "admission" / "copies" / record["copy_id"]
    entry.mkdir(parents=True)
    (entry / f"{1:012d}-{digest}.json").write_bytes(raw)
    (entry / "head.json").write_text(json.dumps({"revision": 1, "sha256": digest}))


# --- N1: no K run once an abort or a retire has started (probe_gate_abort) -------------------------

def _head(generation, binding=K):
    return {"schema_version": 1, "generation": generation, "binding_id": binding, "status": "closed", "run_id": None}


_ARMED = [("k_runtime", {}), ("materialized", {}), ("authorized", {"candidate_binding_id": K}), ("activated", {}),
          ("verify_armed", {"binding_id": K, "generation": 6, "smoke_seconds": 120})]
_VERIFIED = _ARMED + [("verify_started", {"binding_id": K, "generation": 7}),
                      ("verified", {"generation": 7, "binding_id": K, "run_id": "d" * 32}),
                      ("allow_run", {"reason_sha256": "e" * 64, "hours": 72, "until": 10_000.0})]


@pytest.mark.parametrize("case, events, generation, code", [
    ("A plain abort, armed run unconsumed", _ARMED + [("abort_started", {"to_h": False})], 6,
     "runtime_admission_compaction_abort_pending"),
    ("B abort --to-h, allow-run granted", _VERIFIED + [("abort_started", {"to_h": True})], 8,
     "runtime_admission_compaction_abort_pending"),
    ("C retire started, allow-run granted", _VERIFIED + [("retire_started", {})], 8,
     "runtime_admission_compaction_retire_in_progress"),
])
def test_serve_gate_refuses_once_an_abort_or_a_retire_has_started(tmp_path, case, events, generation, code):
    root = tmp_path.resolve()
    (root / "admission").mkdir()
    stage._atomic_record(root / "admission" / "head.json", _head(generation))
    intent = {"schema_version": 1, "compaction_id": CID, "kind": "compaction_intent", "head": HEAD}
    _log_with(root, intent, events)
    decision = stage.compaction_serve_decision(root, stage.read_admission_head(root), wall=10.0)
    assert (decision["allow"], decision["code"], decision["kind"]) == (False, code, None), case
    earlier = root / "earlier"
    (earlier / "admission").mkdir(parents=True)
    stage._atomic_record(earlier / "admission" / "head.json", _head(generation))
    _log_with(earlier, intent, events[:-1])
    allowed = stage.compaction_serve_decision(earlier, stage.read_admission_head(earlier), wall=10.0)
    assert allowed["allow"] is True  # the same log one event earlier: the event is what refuses


def _verified(world, tmp_path, *, execute=None):
    root = world["root"]
    started = cw.start(root, world["archive"])
    compaction.compact_verify(root, started["compaction_id"])
    assert cw.serve(root, tmp_path, "verify", execute=execute).code == 0
    return started["compaction_id"]


def _h_restored(world, tmp_path, h_head, members):
    root = world["root"]
    head = stage.read_admission_head(root)
    assert head["status"] == "closed" and head["binding_id"] == h_head["binding_id"]
    assert os.readlink(root / "current") == "slots/b/runtime"
    assert sorted(os.listdir(root / "slots")) == ["a", "b"]
    assert {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")} == members
    assert cw.serve(root, tmp_path, "h").code == 0


def test_a_cut_inside_the_return_to_h_keeps_every_k_run_out(x_archive, tmp_path, monkeypatch):
    """N1: allow-run granted, abort --to-h cut after its manual head (before the CAS): K is still the
    closed head and an allow-run run would have started; the gate holds until the abort finishes."""
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    members = {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")}
    compaction_id = _verified(world, tmp_path)
    compaction.compact_allow_run(root, compaction_id, reason="guests", hours=1)
    assert cw.serve(root, tmp_path, "allowed").code == 0  # an allow-run run, before any abort
    options = {"to_h": True, "confirm_h": h_head["binding_id"][:12]}
    _refusal("runtime_admission_compaction_abort_after_allow_run",
             lambda: compaction.compact_abort(root, compaction_id, **options))
    # A second compaction state: verified, allow-run granted but unused, then the cut.
    world = cw.world(x_archive, tmp_path / "second", hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    members = {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")}
    compaction_id = _verified(world, tmp_path / "second")
    compaction.compact_allow_run(root, compaction_id, reason="guests", hours=1)
    options = {"to_h": True, "confirm_h": h_head["binding_id"][:12]}
    original = stage._atomic_record

    def record(path, value, *, immutable=False):
        original(path, value, immutable=immutable)
        if value.get("kind") == "compaction_return":
            raise RuntimeError("synthetic-crash")
    monkeypatch.setattr(stage, "_atomic_record", record)
    with pytest.raises(RuntimeError, match="synthetic-crash"):
        compaction.compact_abort(root, compaction_id, **options)
    monkeypatch.setattr(stage, "_atomic_record", original)
    assert stage.read_admission_head(root)["binding_id"] != h_head["binding_id"]  # K, closed
    k_before = cw.tree(root / "slots" / "k" / "state")
    assert cw.serve(root, tmp_path / "second", "while-aborting").code == 78
    assert cw.tree(root / "slots" / "k" / "state") == k_before
    assert compaction.compact_abort(root, compaction_id, **options)["phase"] == "aborted"
    _h_restored(world, tmp_path / "second", h_head, members)


_ORDINARY = cw.sql(("INSERT INTO live_connections VALUES ('c1', 'client', 'runtime', 'ns')", ()))
_ERASED = cw.sql(("INSERT INTO conversation_tombstones VALUES ('conv-1', 1.0, 'owner')", ()))


@pytest.mark.parametrize("during_run", ["ordinary_rows", "conversation_erased"])
def test_a_k_run_after_abort_started_is_checked_again(x_archive, tmp_path, monkeypatch, during_run):
    """N1: a K run the gate should have refused (the pre-fix gate, emulated) happens after
    'abort_started': resume refuses, and the repeated abort applies Q6 and the discard to what ran."""
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    members = {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")}
    compaction_id = _verified(world, tmp_path)
    compaction.compact_allow_run(root, compaction_id, reason="guests", hours=1)
    options = {"to_h": True, "confirm_h": h_head["binding_id"][:12]}
    original_append = stage.append_compaction_event

    def append(root_, log, kind, data, *, at):
        value = original_append(root_, log, kind, data, at=at)
        if kind == "abort_started":
            raise RuntimeError("synthetic-crash")
        return value
    monkeypatch.setattr(stage, "append_compaction_event", append)
    with pytest.raises(RuntimeError, match="synthetic-crash"):
        compaction.compact_abort(root, compaction_id, **options)
    monkeypatch.setattr(stage, "append_compaction_event", original_append)
    decide = stage.compaction_serve_decision

    def pre_fix(root_, head, *, wall):
        decision = decide(root_, head, wall=wall)
        if decision["code"] == "runtime_admission_compaction_abort_pending":
            return {"allow": True, "code": None, "kind": "allow_run", "log": decision["log"]}
        return decision
    monkeypatch.setattr(stage, "compaction_serve_decision", pre_fix)
    execute = {"ordinary_rows": _ORDINARY, "conversation_erased": _ERASED}[during_run]
    assert cw.serve(root, tmp_path, "behind-the-abort", execute=execute).code == 0
    monkeypatch.setattr(stage, "compaction_serve_decision", decide)
    witness = cw.witness(root)
    _refusal("runtime_admission_compaction_abort_recheck_required",
             lambda: compaction.compact_resume(root, compaction_id))
    _refusal("runtime_admission_compaction_abort_recheck_required",
             lambda: compaction.compact_abort(root, compaction_id))
    if during_run == "conversation_erased":
        _refusal("runtime_admission_compaction_abort_revives", lambda: compaction.compact_abort(
            root, compaction_id, **options, discard_k_writes=True))
        assert cw.witness(root) == witness
        return
    _refusal("runtime_admission_compaction_discard_k_writes_required",
             lambda: compaction.compact_abort(root, compaction_id, **options))
    assert cw.witness(root) == witness
    assert compaction.compact_abort(root, compaction_id, **options, discard_k_writes=True)["phase"] == "aborted"
    assert "abort_rechecked" in _events(root, compaction_id)
    _h_restored(world, tmp_path, h_head, members)


# --- N2: a resumed abort finds every member by its recorded identity (probe_abort_resume) ----------

def _fill(path: Path, files: int) -> None:
    (path / "runtime").mkdir(parents=True)
    (path / "runtime" / "release.json").write_text(json.dumps({"files": []}))
    (path / "state" / "data").mkdir(parents=True)
    for index in range(files):
        (path / "state" / "data" / f"f{index}.db").write_bytes(b"private")


def _abort_root(tmp_path: Path, *, partial_snapshot: bool) -> Path:
    root = tmp_path.resolve() / "root"
    (root / "admission").mkdir(parents=True)
    stage._atomic_record(root / "admission" / "head.json", HEAD)
    (root / "slots" / "h" / "runtime").mkdir(parents=True)
    os.symlink("slots/h/runtime", root / "current")
    _fill(root / "slots" / "k", 5)
    scratch = root / "compaction" / CID
    (scratch / "placeholder" / "config").mkdir(parents=True)
    intent = {"schema_version": 1, "compaction_id": CID, "kind": "compaction_intent", "head": HEAD,
              "k_slot": "k", "members": [], "source_binding_id": H,
              "k_state_root": str(root / "slots" / "k" / "state")}
    events = [("own_work", {"path": str(scratch / "placeholder"),
                            "identity": compaction._identity(scratch / "placeholder")})]
    if partial_snapshot:
        _fill(scratch / "snapshot-1", 2)
        events.append(("snapshot_reserved", {"n": 1, "path": str(scratch / "snapshot-1")}))
    events += [("k_runtime", {"slot": "k", "slot_identity": compaction._identity(root / "slots" / "k")}),
               ("abort_started", {"to_h": False, "head": HEAD})]
    _log_with(root, intent, events)
    return root


def _aborted_clean(root: Path) -> list:
    log = stage.read_compaction_log(root, CID)
    assert log["terminal"] and log["events"][-1]["kind"] == "aborted"
    assert not (root / "slots" / "k").exists() and not (root / "compaction" / CID).exists()
    assert not list(root.rglob("*.db"))  # every private file of K and the snapshot is gone
    destroyed = [event["data"]["member"] for event in stage.compaction_events(log, "destroyed")]
    assert len(destroyed) == len(set(destroyed))
    return destroyed


@pytest.mark.parametrize("partial_snapshot", [False, True])
@pytest.mark.parametrize("victim", ["placeholder", "snapshot", "k"])
@pytest.mark.parametrize("cut", ["retiring", "renamed", "mid_walk", "tombstone", "destroyed"])
def test_a_resumed_abort_finishes_each_member_from_every_boundary(tmp_path, monkeypatch, partial_snapshot, victim,
                                                                    cut):
    if victim == "snapshot" and not partial_snapshot:
        pytest.skip("no partial snapshot in this root")
    monkeypatch.setattr(compaction, "stage", _Stage())
    root = _abort_root(tmp_path, partial_snapshot=partial_snapshot)
    marker = {"placeholder": "placeholder", "snapshot": "snapshot-1", "k": "k"}[victim]
    victim_path = {"placeholder": root / "compaction" / CID / "placeholder",
                   "snapshot": root / "compaction" / CID / "snapshot-1", "k": root / "slots" / "k"}[victim]
    original_retire, original_walk = compaction._retire_member, compaction._unlink_tree
    original_append, original_record = stage.append_compaction_event, stage._atomic_record
    current = {}

    def retire(root_, log, member, k_inodes, wall, unlink_hook=None):
        current["member"] = member if Path(member["path"]).name == marker else None
        return original_retire(root_, log, member, k_inodes, wall, unlink_hook)

    def append(root_, log, kind, data, *, at):
        if cut == "retiring" and kind == "retiring" and current.get("member"):
            value = original_append(root_, log, kind, data, at=at)
            raise SystemExit("cut")
        value = original_append(root_, log, kind, data, at=at)
        if cut == "destroyed" and kind == "destroyed" and current.get("member"):
            raise SystemExit("cut")
        return value

    def walk(root_, path, **kwargs):
        if current.get("member") and cut in {"renamed", "mid_walk"}:
            if cut == "mid_walk":
                victim_file = next(iter(sorted(Path(path).rglob("*"), key=lambda item: -len(item.parts))))
                if victim_file.is_file():
                    victim_file.unlink()
            raise SystemExit("cut")
        return original_walk(root_, path, **kwargs)

    def record(path, value, *, immutable=False):
        original_record(path, value, immutable=immutable)
        if cut == "tombstone" and value.get("status") == "destroyed" and current.get("member"):
            raise SystemExit("cut")

    monkeypatch.setattr(compaction, "_retire_member", retire)
    monkeypatch.setattr(stage, "append_compaction_event", append)
    monkeypatch.setattr(compaction, "_unlink_tree", walk)
    monkeypatch.setattr(stage, "_atomic_record", record)
    with pytest.raises(SystemExit):
        compaction._finish_abort(root, stage.read_compaction_log(root, CID), lambda: 2.0)
    for name, value in (("_retire_member", original_retire), ("_unlink_tree", original_walk)):
        monkeypatch.setattr(compaction, name, value)
    monkeypatch.setattr(stage, "append_compaction_event", original_append)
    monkeypatch.setattr(stage, "_atomic_record", original_record)
    assert not stage.read_compaction_log(root, CID)["terminal"]
    if cut != "retiring":
        assert not victim_path.exists()  # it is in the quarantine (or already gone)
    result = compaction._finish_abort(root, stage.read_compaction_log(root, CID), lambda: 3.0)
    assert result["phase"] == "aborted"
    destroyed = _aborted_clean(root)
    assert len(destroyed) == (3 if partial_snapshot else 2)
    assert max(destroyed) == compaction.K_ABORT_NUMBER  # K's fixed number (intent members: none)


def test_an_abort_never_logs_aborted_over_a_quarantine_residue(tmp_path, monkeypatch):
    """N2 invariant: whatever is still in compaction/<id>/retiring refuses 'aborted' (and 'complete')."""
    monkeypatch.setattr(compaction, "stage", _Stage())
    root = _abort_root(tmp_path, partial_snapshot=False)
    stray = root / "compaction" / CID / "retiring" / "77"
    (stray / "state").mkdir(parents=True)
    _refusal("runtime_admission_compaction_retiring_residue",
             lambda: compaction._finish_abort(root, stage.read_compaction_log(root, CID), lambda: 2.0))
    assert not stage.read_compaction_log(root, CID)["terminal"]
    assert (stray / "state").is_dir()


# --- N3: leftover links are unlinked as links, never followed (probe_leftover_links) ---------------

def _leftover_root(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "root"
    (root / "admission" / "bindings").mkdir(parents=True)
    stage._atomic_record(root / "admission" / "head.json", HEAD)
    target = root / "slots" / "a" / "runtime"
    target.mkdir(parents=True)
    (target / "keep.pyc").write_bytes(b"kept")
    activate = root / ".activate-abc123"
    activate.mkdir()
    (activate / "current").symlink_to("slots/a/runtime")
    staging = root / "slots" / ".prepare-xyz"
    (staging / "runtime" / "scripts").mkdir(parents=True)
    (staging / "runtime" / "scripts" / "stage_runtime.pyc").write_bytes(b"x")
    (staging / "state" / "config").mkdir(parents=True)
    (staging / "state" / "config" / "settings.yaml").write_text("a: 1\n")
    (staging / "state" / "data").mkdir()
    (staging / "state" / "data" / "privacy.db").write_bytes(b"partial")
    (staging / "runtime" / "config").symlink_to("../state/config")
    stage._atomic_record(stage.prepared_directory(root) / ".staging-xyz.json", {
        "schema_version": 1, "kind": "staging", "staging_id": "xyz", "path": str(staging),
        "identity": stage._identity(staging), "compaction_id": None, "created_at": 2.0}, immutable=True)
    return root


def _retire_all(root: Path, members: list) -> dict:
    intent = {"schema_version": 1, "compaction_id": CID, "kind": "compaction_intent", "head": HEAD, "k_slot": "k",
              "members": members, "source_binding_id": H, "owners": [],
              "k_state_root": str(root / "slots" / "k" / "state")}
    log = _log_with(root, intent, [("retire_started", {})])
    for member in members:
        log = compaction._retire_member(root, log, member, set(), lambda: 5.0)
    return log


def test_retire_unlinks_the_links_that_make_leftovers_members(tmp_path):
    root = _leftover_root(tmp_path)
    members, _untouched = compaction.retirement_set(root, k_slot="k", retire_prepare_leftovers=True)
    assert sorted((member["class"], Path(member["path"]).name) for member in members) == [
        ("prepare_leftover", ".prepare-xyz"), ("selection_leftover", ".activate-abc123")]
    log = _retire_all(root, members)
    assert len(stage.compaction_events(log, "destroyed")) == 2
    assert not (root / ".activate-abc123").exists() and not (root / "slots" / ".prepare-xyz").exists()
    assert (root / "slots" / "a" / "runtime" / "keep.pyc").read_bytes() == b"kept"  # never followed


@pytest.mark.parametrize("where", ["activate", "prepare"])
def test_any_other_link_in_a_leftover_refuses_before_retiring_is_logged(tmp_path, where):
    root = _leftover_root(tmp_path)
    members, _untouched = compaction.retirement_set(root, k_slot="k", retire_prepare_leftovers=True)
    if where == "activate":
        member = next(item for item in members if item["class"] == "selection_leftover")
        os.unlink(root / ".activate-abc123" / "current")
        (root / ".activate-abc123" / "current").symlink_to("/etc")
    else:
        member = next(item for item in members if item["class"] == "prepare_leftover")
        (root / "slots" / ".prepare-xyz" / "state" / "data" / "elsewhere").symlink_to("/etc")
    intent = {"schema_version": 1, "compaction_id": CID, "kind": "compaction_intent", "head": HEAD, "k_slot": "k",
              "members": members, "source_binding_id": H, "owners": [],
              "k_state_root": str(root / "slots" / "k" / "state")}
    log = _log_with(root, intent, [("retire_started", {})])
    _refusal("runtime_admission_compaction_symlink",
             lambda: compaction._retire_member(root, log, member, set(), lambda: 5.0))
    assert not stage.compaction_events(stage.read_compaction_log(root, CID), "retiring")


# --- N7 (F6 residual): only what a proof covers is destroyed (probe_catalog_copy) -------------------

def _copy_record(destination: Path, **fields) -> dict:
    return dict({"copy_id": "1" * 32, "destination": str(destination),
                 "root_identity": compaction._identity(destination), "preparation": "prepared",
                 "kind": "snapshot"}, **fields)


def _snapshot_copy(root: Path) -> Path:
    snap = root / "backups" / "snapshot-2026-11-01"
    (snap / "state" / "data").mkdir(parents=True)
    (snap / "state" / "data" / "privacy.db").write_bytes(b"private copy")
    (snap / "manifest.json").write_text("{}")
    return snap


def test_a_catalogued_copy_holding_hand_placed_entries_is_kept_whole(tmp_path):
    root = tmp_path.resolve() / "root"
    (root / "admission").mkdir(parents=True)
    (root / "slots").mkdir()
    snap = _snapshot_copy(root)
    proven = {"extents": {"e1": {"relative": "state", "family": "x", "representation": "y"}},
              "prepared_record": {"relative": "manifest.json", "inventory": {"manifest.json": {},
                                                                             "state/data/privacy.db": {}}}}
    _catalog(root, _copy_record(snap, **proven))
    members, _untouched = compaction.retirement_set(root, k_slot="k", retire_prepare_leftovers=False)
    assert [(member["class"], member["allowed_entries"]) for member in members] == [
        ("catalog_copy", ["manifest.json", "state"])]
    (snap / "hand-backup-2026-11-10").mkdir()
    (snap / "hand-backup-2026-11-10" / "notes.txt").write_text("not the controller's")
    members, untouched = compaction.retirement_set(root, k_slot="k", retire_prepare_leftovers=False)
    assert members == []
    [kept] = [entry for entry in untouched if entry["class"] == "unexpected_entries"]
    assert kept == {"path": "backups/snapshot-2026-11-01", "type": "directory", "class": "unexpected_entries",
                    "names": ["hand-backup-2026-11-10"]}


def test_a_copy_record_without_extents_proves_no_entry(tmp_path):
    """The probe's record (no extents, no prepared inventory): nothing in it is proven, so it is kept."""
    root = tmp_path.resolve() / "root"
    (root / "admission").mkdir(parents=True)
    (root / "slots").mkdir()
    snap = _snapshot_copy(root)
    _catalog(root, _copy_record(snap))
    members, untouched = compaction.retirement_set(root, k_slot="k", retire_prepare_leftovers=False)
    assert members == []
    assert {entry["path"]: entry.get("names") for entry in untouched}["backups/snapshot-2026-11-01"] == [
        "manifest.json", "state"]
    # An extent '.' (the whole destination) adds no name: the inventory still has to list each one.
    record = _copy_record(snap, extents={"e": {"relative": ".", "family": "x", "representation": "y"}})
    assert compaction._copy_entries(record) == []


def _staged_slot(root: Path, *, extra: bool) -> Path:
    slot = root / "slots" / "s"
    (slot / "runtime").mkdir(parents=True)
    (slot / "runtime" / "code.pyc").write_bytes(b"c")
    (slot / "runtime" / "release.json").write_text(json.dumps({"files": ["code.pyc"]}))
    (slot / "state" / "data").mkdir(parents=True)
    (slot / "runtime" / "data").symlink_to("../state/data")
    if extra:
        (slot / "runtime" / "notes-by-hand.txt").write_text("x")
    stage._atomic_record(stage.prepared_directory(root) / ".staging-s.json", {
        "schema_version": 1, "kind": "staging", "staging_id": "s", "path": str(root / "slots" / ".prepare-s"),
        "identity": stage._identity(slot), "compaction_id": None, "created_at": 2.0}, immutable=True)
    return slot


@pytest.mark.parametrize("extra", [False, True])
def test_runtime_extras_and_staging_proven_slots_need_more_than_the_proof(tmp_path, extra):
    root = tmp_path.resolve() / "root"
    (root / "admission" / "bindings").mkdir(parents=True)
    _staged_slot(root, extra=extra)
    members, untouched = compaction.retirement_set(root, k_slot="k", retire_prepare_leftovers=False)
    kept = {entry["path"]: entry for entry in untouched}["slots/s"]
    assert members == []
    if extra:
        assert kept["class"] == "unexpected_entries" and kept["names"] == ["runtime/notes-by-hand.txt"]
        return
    assert kept["class"] == "prepare_leftover_kept"  # a staging record does not prove state/'s contents
    members, _untouched = compaction.retirement_set(root, k_slot="k", retire_prepare_leftovers=True)
    assert [(member["class"], member["proof"]) for member in members] == [("managed_slot", "staging_record")]


# --- N8: a member moved whole after the plan is never reported destroyed (probe_moved_member) -------

def test_a_member_moved_after_the_plan_is_found_and_never_tombstoned(tmp_path, monkeypatch):
    monkeypatch.setattr(compaction, "stage", _Stage())
    root = tmp_path.resolve() / "root"
    (root / "admission").mkdir(parents=True)
    (root / "slots").mkdir()
    (root / "logs").mkdir()
    snap = root / "backups" / "snapshot-2026-11-01"
    (snap / "state" / "data").mkdir(parents=True)
    (snap / "state" / "data" / "privacy.db").write_text("private copy")
    _catalog(root, _copy_record(snap, extents={"e": {"relative": "state", "family": "x", "representation": "y"}}))
    members, untouched = compaction.retirement_set(root, k_slot="k", retire_prepare_leftovers=False,
                                                   compaction_id=CID)
    assert [Path(member["path"]).relative_to(root).as_posix() for member in members] == [
        "backups/snapshot-2026-11-01"]
    intent = {"schema_version": 1, "compaction_id": CID, "kind": "compaction_intent", "k_slot": "k",
              "members": members, "untouched": untouched, "retire_prepare_leftovers": False, "owners": [],
              "head": HEAD, "source_binding_id": H, "k_state_root": str(root / "slots" / "k" / "state")}
    log = _log_with(root, intent, [])
    moved = root / "logs" / "snapshot-2026-11-01"
    snap.rename(moved)
    _refusal("runtime_admission_compaction_member_missing", lambda: compaction._same_plan(root, log, members))
    _refusal("runtime_admission_compaction_member_missing",
             lambda: compaction._retire_member(root, log, members[0], set(), lambda: 5.0))
    log = compaction._accept_missing(root, log, members[0]["n"], lambda: 6.0)
    compaction._same_plan(root, log, members)  # the acknowledged plan again
    log = compaction._retire_member(root, log, members[0], set(), lambda: 7.0)
    assert not stage.compaction_events(log, "destroyed")
    assert not (stage.tombstone_directory(root).exists() and list(stage.tombstone_directory(root).iterdir()))
    assert not (root / "admission" / "copies-retired").exists()
    assert (moved / "state" / "data" / "privacy.db").read_text() == "private copy"  # reported, left alone
    _refusal("runtime_admission_compaction_member", lambda: compaction._accept_missing(root, log, 99, lambda: 8.0))


@pytest.mark.parametrize("logged, accepted", [(("retiring", "renamed", "walked"), False),
                                              (("retiring", "renamed"), True), (("retiring",), True)])
def test_accept_missing_refuses_a_member_this_compaction_renamed(tmp_path, monkeypatch, logged, accepted):
    """A member gone from its path after 'walked' was destroyed by this compaction: never 'accepted missing'.
    'retiring' alone is logged before the rename (review r3 R3-4) and 'renamed' before the walk (review r4
    R4-6: its quarantine can be moved out by hand mid-walk), so neither proves a destruction: accepted."""
    monkeypatch.setattr(compaction, "stage", _Stage())
    root = tmp_path.resolve() / "root"
    (root / "admission").mkdir(parents=True)
    (root / "slots").mkdir()
    snap = _snapshot_copy(root)
    (snap / "manifest.json").unlink()
    _catalog(root, _copy_record(snap, extents={"e": {"relative": "state", "family": "x", "representation": "y"}}))
    members, untouched = compaction.retirement_set(root, k_slot="k", retire_prepare_leftovers=False,
                                                   compaction_id=CID)
    intent = {"schema_version": 1, "compaction_id": CID, "kind": "compaction_intent", "k_slot": "k",
              "members": members, "untouched": untouched, "retire_prepare_leftovers": False, "owners": [],
              "head": HEAD, "source_binding_id": H, "k_state_root": str(root / "slots" / "k" / "state")}
    data = {"member": members[0]["n"], "identity": members[0]["identity"]}
    log = _log_with(root, intent, [(kind, data) for kind in logged])
    snap.rename(root / "elsewhere")  # gone from its path, and not in the quarantine
    if accepted:
        log = compaction._accept_missing(root, log, members[0]["n"], lambda: 6.0)
        assert stage.compaction_events(log, "missing_accepted")
    else:
        _refusal("runtime_admission_compaction_member",
                 lambda: compaction._accept_missing(root, log, members[0]["n"], lambda: 6.0))


# --- N9: a placeholder cut before its 'own_work' is still this compaction's ------------------------

@pytest.mark.parametrize("then", ["resume", "abort"])
def test_a_placeholder_cut_before_own_work_is_found(x_archive, tmp_path, monkeypatch, then):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    members = {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")}
    original = stage.private_directories

    def private(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if Path(path).name == "data" and Path(path).parent.name == "placeholder":
            raise RuntimeError("synthetic-crash")  # both directories made, 'own_work' not logged
        return value
    monkeypatch.setattr(stage, "private_directories", private)
    with pytest.raises(RuntimeError, match="synthetic-crash"):
        cw.start(root, world["archive"])
    monkeypatch.setattr(stage, "private_directories", original)
    [log] = stage.compaction_logs(root)
    compaction_id = log["compaction_id"]
    kinds = _events(root, compaction_id)
    assert "own_work_reserved" in kinds and "own_work" not in kinds
    placeholder = root / "compaction" / compaction_id / "placeholder"
    assert placeholder.is_dir()
    if then == "resume":
        assert compaction.compact_resume(root, compaction_id, archive=world["archive"])["phase"] == "activated"
        assert "own_work" in _events(root, compaction_id)
    assert compaction.compact_abort(root, compaction_id)["phase"] == "aborted"
    assert not (root / "compaction" / compaction_id).exists()
    _h_restored(world, tmp_path, h_head, members)


# --- N10: a copy being written refuses; an abandoned one is listed ---------------------------------

def _dead_pid() -> int:
    child = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True,
                           check=True)
    return int(child.stdout)


def test_a_copy_in_preparation_refuses_the_plan_and_an_abandoned_one_is_listed(x_archive, tmp_path):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    destination = root / "backups" / "copy-in-flight"
    (destination / "state").mkdir(parents=True)
    owner = {"epoch": "f" * 32, "process_id": os.getpid(), "lease_fd": 3,
             "lease_identity": {"device": 1, "inode": 2}}
    record = _copy_record(destination, copy_id="2" * 32, preparation="writing", preparation_owner=owner)
    _catalog(root, record)
    _refusal("runtime_admission_compaction_copy_in_progress", lambda: cw.plan(root, world["archive"]))
    for name in os.listdir(root / "admission" / "copies" / record["copy_id"]):
        os.unlink(root / "admission" / "copies" / record["copy_id"] / name)
    os.rmdir(root / "admission" / "copies" / record["copy_id"])
    _catalog(root, dict(record, preparation_owner=dict(owner, process_id=_dead_pid())))
    planned = cw.plan(root, world["archive"])
    assert planned["abandoned_copies"] == [record["copy_id"]]
    for preparation in ("bogus", None):
        for name in os.listdir(root / "admission" / "copies" / record["copy_id"]):
            os.unlink(root / "admission" / "copies" / record["copy_id"] / name)
        os.rmdir(root / "admission" / "copies" / record["copy_id"])
        _catalog(root, dict(record, preparation=preparation))
        _refusal("runtime_admission_compaction_copy_in_progress", lambda: cw.plan(root, world["archive"]))
