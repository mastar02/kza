"""U3b-0c independent review (F1-F12, D1-D4) and its pre-production gaps, on real managed roots.

Same worlds as test_stage_runtime_compaction (X compiled from HEAD plus this
working tree, real closed runs, the real worker, in-process serve under a
fake manager). Crash injection raises at one boundary, then the operator's
next command must finish the step: nothing may strand the root.
"""

import errno
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace

import pytest

from scripts import compaction
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from tests.mocks import compaction_world as cw


@pytest.fixture(scope="module")
def x_archive(tmp_path_factory):
    return cw.build_x(tmp_path_factory.mktemp("compaction-review-x"))


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


def _verified(world, tmp_path, *, execute=None, **options):
    root = world["root"]
    started = cw.start(root, world["archive"], **options)
    compaction.compact_verify(root, started["compaction_id"])
    result = cw.serve(root, tmp_path, "verify", execute=execute)
    assert result.code == 0, result.phases
    return started["compaction_id"]


def _h_restored(world, tmp_path, h_head, members, name="h"):
    root = world["root"]
    head = stage.read_admission_head(root)
    assert head["status"] == "closed" and head["binding_id"] == h_head["binding_id"]
    assert os.readlink(root / "current") == "slots/b/runtime"
    assert sorted(os.listdir(root / "slots")) == ["a", "b"]
    assert {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")} == members
    assert cw.serve(root, tmp_path, name).code == 0


def _after_append(monkeypatch, kind):
    """Raise right after the first ``kind`` event is durable (the event is logged, its work is not)."""
    original = stage.append_compaction_event

    def append(root_, log, kind_, data, *, at):
        value = original(root_, log, kind_, data, at=at)
        if kind_ == kind and not getattr(append, "done", False):
            append.done = True
            raise RuntimeError("synthetic-crash")
        return value
    monkeypatch.setattr(stage, "append_compaction_event", append)


# --- F1: every boundary of abort, including the window between the return CAS and 'returned' ---

def _cut_abort(monkeypatch, cut, h_binding):
    if cut in {"abort_started", "returned", "retiring", "destroyed"}:
        _after_append(monkeypatch, cut)
    elif cut in {"return_record", "manual_head"}:
        original = stage._atomic_record

        def record(path, value, *, immutable=False):
            original(path, value, immutable=immutable)
            if (cut == "return_record" and Path(path).parent.name == "returns") or \
                    (cut == "manual_head" and value.get("kind") == "compaction_return"):
                raise RuntimeError("synthetic-crash")
        monkeypatch.setattr(stage, "_atomic_record", record)
    elif cut == "cas":
        original = stage._cas_head

        def cas(root_, expected, binding_id, status, run_id=None):
            value = original(root_, expected, binding_id, status, run_id)
            if binding_id == h_binding and status == "closed":
                raise RuntimeError("synthetic-crash")  # H is the head again; 'current' still names K
            return value
        monkeypatch.setattr(stage, "_cas_head", cas)
    elif cut == "publish":
        def publish(root_, name):
            raise RuntimeError("synthetic-crash")  # inside the publication, after the CAS
        monkeypatch.setattr(stage, "_publish_selection", publish)


@pytest.mark.parametrize("to_h, cut", [
    (False, "abort_started"), (False, "return_record"), (False, "manual_head"), (False, "cas"),
    (False, "publish"), (False, "returned"), (False, "retiring"), (False, "destroyed"),
    (True, "cas"), (True, "publish"),
])
def test_abort_crash_at_every_boundary_finishes_on_the_next_command(x_archive, tmp_path, monkeypatch, to_h, cut):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    members = {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")}
    if to_h:
        compaction_id = _verified(world, tmp_path)
        options = {"to_h": True, "confirm_h": h_head["binding_id"][:12]}
    else:
        compaction_id = cw.start(root, world["archive"])["compaction_id"]
        options = {}
    _cut_abort(monkeypatch, cut, h_head["binding_id"])
    with pytest.raises(RuntimeError, match="synthetic-crash"):
        compaction.compact_abort(root, compaction_id, **options)
    monkeypatch.undo()
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)
    [status] = compaction.compact_status(root, compaction_id)["compactions"]
    assert status["terminal"] is False
    assert cw.serve(root, tmp_path, "while-pending").code == 78  # the gate holds until abort finishes
    if cut == "abort_started":
        finished = compaction.compact_resume(root, compaction_id)
    else:
        finished = compaction.compact_abort(root, compaction_id, **options)  # the operator repeats it
    assert finished["phase"] == "aborted"
    assert compaction.compact_resume(root, compaction_id)["noop"] is True
    assert stage.compaction_events(stage.read_compaction_log(root, compaction_id), "returned")
    _h_restored(world, tmp_path, h_head, members)


# --- F2 and the neighbouring C2a cuts: K's prepare, before and after its rename ---

class _NoCleanupOnError(tempfile.TemporaryDirectory):
    """A SIGKILL leaves the staging directory behind; a Python exception would clean it up."""

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self._finalizer.detach()
            return False
        return super().__exit__(kind, value, traceback)


def _cut_prepare(monkeypatch, cut):
    if cut == "placeholder":
        def prepare(*args, **kwargs):
            raise RuntimeError("synthetic-crash")  # after the placeholder's own_work event
        monkeypatch.setattr(stage, "_prepare_slot", prepare)
    elif cut == "extraction":
        monkeypatch.setattr(tempfile, "TemporaryDirectory", _NoCleanupOnError)

        def read_only(runtime):
            raise RuntimeError("synthetic-crash")  # files extracted, staging not renamed
        monkeypatch.setattr(stage, "make_runtime_read_only", read_only)
    elif cut == "renamed":
        original = stage._atomic_record

        def record(path, value, *, immutable=False):
            if value.get("kind") == "prepared" and value.get("slot") == "k":
                raise RuntimeError("synthetic-crash")  # slots/k exists, its prepare record does not
            return original(path, value, immutable=immutable)
        monkeypatch.setattr(stage, "_atomic_record", record)
    elif cut == "prepared":
        original = stage.append_compaction_event

        def append(root_, log, kind, data, *, at):
            if kind == "k_runtime":
                raise RuntimeError("synthetic-crash")  # prepare record written, k_runtime not logged
            return original(root_, log, kind, data, at=at)
        monkeypatch.setattr(stage, "append_compaction_event", append)


@pytest.mark.parametrize("then", ["resume", "abort"])
@pytest.mark.parametrize("cut", ["placeholder", "extraction", "renamed", "prepared"])
def test_k_prepare_crash_resumes_or_aborts(x_archive, tmp_path, monkeypatch, cut, then):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    members = {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")}
    _cut_prepare(monkeypatch, cut)
    with pytest.raises(RuntimeError, match="synthetic-crash"):
        cw.start(root, world["archive"])
    monkeypatch.undo()
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)
    [log] = stage.compaction_logs(root)
    compaction_id = log["compaction_id"]
    leftovers = [name for name in os.listdir(root / "slots") if name.startswith(".prepare-")]
    assert bool(leftovers) is (cut == "extraction")
    assert (root / "slots" / "k").exists() is (cut in {"renamed", "prepared"})
    assert stage.read_admission_head(root) == h_head
    if then == "resume":
        assert compaction.compact_resume(root, compaction_id, archive=world["archive"])["phase"] == "activated"
        assert sorted(os.listdir(root / "slots")) == ["a", "b", "k"]  # the leftover was this compaction's
        compaction.compact_verify(root, compaction_id)
        assert cw.serve(root, tmp_path, "verify").code == 0
        assert compaction.compact_retire(root, compaction_id, scan=cw.no_units)["phase"] == "complete"
        assert sorted(os.listdir(root / "slots")) == ["k"]
        return
    assert compaction.compact_abort(root, compaction_id)["phase"] == "aborted"
    _h_restored(world, tmp_path, h_head, members)


# --- F4: the armed run failed before begin_run's CAS (never consumed) ---

def test_an_unconsumed_armed_run_can_be_aborted_and_allow_run_says_not_verified(x_archive, tmp_path, monkeypatch):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    members = {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")}
    compaction_id = cw.start(root, world["archive"])["compaction_id"]
    compaction.compact_verify(root, compaction_id)
    k_head = stage.read_admission_head(root)
    original = stage._cas_head

    def cas(root_, expected, binding_id, status, run_id=None):
        if status == "running":
            raise OSError(errno.ENOSPC, "synthetic-disk-full")  # begin_run fails before the head moves
        return original(root_, expected, binding_id, status, run_id)
    monkeypatch.setattr(stage, "_cas_head", cas)
    assert cw.serve(root, tmp_path, "begin-fails").code != 0
    monkeypatch.setattr(stage, "_cas_head", original)
    log = stage.read_compaction_log(root, compaction_id)
    [started] = stage.compaction_events(log, "verify_started")
    head = stage.read_admission_head(root)
    # serve's activate re-sealed K (closed -> closed); no running head of the logged generation existed.
    assert head["status"] == "closed" and head["binding_id"] == k_head["binding_id"]
    assert head["generation"] < started["data"]["generation"]
    _refusal("runtime_admission_compaction_not_verified", lambda: compaction.compact_allow_run(
        root, compaction_id, reason="guests", hours=1))
    _refusal("runtime_admission_compaction_abort_to_h_before_run", lambda: compaction.compact_abort(
        root, compaction_id, to_h=True, confirm_h=h_head["binding_id"][:12]))
    assert compaction.compact_abort(root, compaction_id)["phase"] == "aborted"
    _h_restored(world, tmp_path, h_head, members)


# --- F5 / D2: retire compares against H; K's own deletions are counted, never blocking ---

_LIVE_ROW = cw.sql(("INSERT INTO live_connections VALUES ('c1', 'client', 'runtime', 'ns')", ()))
_K_MAINTENANCE = "\n".join([
    cw.sql(("DELETE FROM live_connections", ())),
    cw.ledger_append("fact_deleted", "fact-1"),
])


def test_k_run_maintenance_is_removed_by_k_run_and_never_blocks_retire(x_archive, tmp_path):
    world = cw.world(x_archive, tmp_path, hops=("b",), first=(_LIVE_ROW,))
    root = world["root"]
    planned = cw.plan(root, world["archive"])
    assert planned["coverage"]["unexplained"] == 0 and planned["coverage"]["removed_by_k_run"] == 0
    compaction_id = _verified(world, tmp_path, execute=_K_MAINTENANCE)
    done = compaction.compact_retire(root, compaction_id, scan=cw.no_units)
    assert done["phase"] == "complete"
    [started] = stage.compaction_events(stage.read_compaction_log(root, compaction_id), "retire_started")
    coverage = started["data"]["coverage"]
    assert coverage["unexplained"] == 0
    assert coverage["removed_by_k_run"] >= 2  # the live_connections row in a and in H, gone from K'
    assert started["data"]["accepted_new_unexplained"] == 0


def test_data_first_seen_at_retire_needs_its_own_acknowledgement(x_archive, tmp_path):
    """The start flag covers the plan's counts only (review F5, second part)."""
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    compaction_id = _verified(world, tmp_path, accept_owner_only_loss=True)
    (world["states"]["a"]["state"] / "data" / "late-note.txt").write_text("written after the plan")
    witness = cw.witness(root)
    refused = _refusal("runtime_admission_compaction_owner_only_data",
                       lambda: compaction.compact_retire(root, compaction_id, scan=cw.no_units))
    assert sum(refused.report["new_unexplained"].values()) == 1
    digest = refused.report["coverage_sha256"]
    assert "late-note" not in json.dumps(refused.report)
    _refusal("runtime_admission_compaction_owner_only_data", lambda: compaction.compact_retire(
        root, compaction_id, accept_owner_only_loss="0" * 64, scan=cw.no_units))
    assert cw.witness(root) == witness
    done = compaction.compact_retire(root, compaction_id, accept_owner_only_loss=digest, scan=cw.no_units)
    assert done["phase"] == "complete"
    [started] = stage.compaction_events(stage.read_compaction_log(root, compaction_id), "retire_started")
    assert started["data"]["coverage_sha256"] == digest and started["data"]["accepted_new_unexplained"] == 1


# --- F6: the proof covers the slot's contents, and state/ is the planned one ---

def test_hand_made_entries_keep_a_slot_and_a_swapped_state_refuses(x_archive, tmp_path):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    stage.prepare_slot(world["archive"], root, "spare", world["states"]["a"]["state"])
    stash = root / "slots" / "spare" / "hand-backup-2026-10-29"
    stash.mkdir()
    (stash / "notes.txt").write_text("CANARY-STASH")
    (root / "slots" / "b" / "extra").mkdir()
    _refusal("runtime_admission_compaction_head_unproven", lambda: cw.plan(root, world["archive"]))
    (root / "slots" / "b" / "extra").rmdir()
    planned = cw.plan(root, world["archive"])
    [entry] = [item for item in planned["untouched"] if item["path"] == "slots/spare"]
    assert entry == {"path": "slots/spare", "type": "directory", "class": "unexpected_entries",
                     "names": ["hand-backup-2026-10-29"]}
    assert all(member.get("slot") != "spare" for member in planned["members"])
    compaction_id = _verified(world, tmp_path)
    a_slot = root / "slots" / "a"
    # (stash) after the plan: retire refuses before destroying anything.
    (a_slot / "late").mkdir()
    witness = cw.witness(root)
    _refusal("runtime_admission_compaction_identity",
             lambda: compaction.compact_retire(root, compaction_id, scan=cw.no_units))
    assert cw.witness(root) == witness
    (a_slot / "late").rmdir()
    # (swap) state/ replaced after the plan: refused; the original is never destroyed by name.
    original = tmp_path / "a-state"
    os.rename(a_slot / "state", original)
    (a_slot / "state").mkdir()
    witness = cw.witness(root)
    _refusal("runtime_admission_compaction_identity",
             lambda: compaction.compact_retire(root, compaction_id, scan=cw.no_units))
    assert cw.witness(root) == witness and original.exists()
    (a_slot / "state").rmdir()
    os.rename(original, a_slot / "state")
    assert compaction.compact_retire(root, compaction_id, scan=cw.no_units)["phase"] == "complete"
    assert (stash / "notes.txt").read_text() == "CANARY-STASH"
    assert sorted(os.listdir(root / "slots")) == ["k", "spare"]


# --- F7: the untouched list and the retirement set are recomputed at retire ---

def test_a_copy_made_after_the_plan_refuses_retire_as_plan_changed(x_archive, tmp_path):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    compaction_id = _verified(world, tmp_path)
    late = root / "slots" / "zz"  # what an epoch-2 controller's prepare would have left (review F8 probe)
    (late / "runtime").mkdir(parents=True)
    (late / "state" / "data").mkdir(parents=True)
    (late / "state" / "data" / "copy.txt").write_text("CANARY-LATE")
    witness = cw.witness(root)
    refused = _refusal("runtime_admission_compaction_plan_changed",
                       lambda: compaction.compact_retire(root, compaction_id, scan=cw.no_units))
    assert refused.report["untouched_added"] == ["slots/zz"]
    assert cw.witness(root) == witness and "retire_started" not in _events(root, compaction_id)
    (late / "state" / "data" / "copy.txt").unlink()
    for path in (late / "state" / "data", late / "state", late / "runtime", late):
        path.rmdir()
    assert compaction.compact_retire(root, compaction_id, scan=cw.no_units)["phase"] == "complete"


# --- F8 / D1: forward epoch; D4 / F3: install-controller is the way out ---

def test_a_newer_installed_epoch_refuses_this_controller(x_archive, tmp_path):
    world = cw.world(x_archive, tmp_path, hops=())
    root = world["root"]
    path = root / "admission" / "controller.json"
    record = json.loads(path.read_text())
    assert record["controller_epoch"] == stage.CONTROLLER_EPOCH == 4  # SC-2
    stage._atomic_record(path, dict(record, controller_epoch=stage.CONTROLLER_EPOCH + 1))
    _refusal("runtime_admission_controller_downgrade", lambda: stage.AdmissionController(root).__enter__())
    _refusal("runtime_admission_controller_downgrade",
             lambda: stage.prepare_slot(world["archive"], root, "zz", world["states"]["a"]["state"]))
    _refusal("runtime_admission_controller_downgrade", lambda: cw.plan(root, world["archive"]))
    assert not (root / "slots" / "zz").exists()
    stage._atomic_record(path, record)
    with stage.AdmissionController(root):
        pass


def test_install_controller_enters_a_pending_or_invalid_compaction(x_archive, tmp_path):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    h_head = stage.read_admission_head(root)
    members = {key: cw.tree(world["states"][key]["state"]) for key in ("a", "b")}
    compaction_id = cw.start(root, world["archive"])["compaction_id"]
    _refusal("runtime_admission_compaction_pending", lambda: stage.AdmissionController(root).__enter__())
    assert launcher.install_controller(root, world["archive"])["controller_epoch"] == 4  # D4 (SC-2: epoch 4)
    events = stage.compaction_log_path(root, compaction_id) / "events"
    bad = events / "000099.json"
    bad.write_text("{}")  # a gap: the whole log is invalid (fail closed)
    [status] = compaction.compact_status(root)["compactions"]
    assert status == {"compaction_id": compaction_id, "invalid": True,
                      "code": "runtime_admission_compaction_log_invalid"}
    _refusal("runtime_admission_compaction_log_invalid", lambda: stage.AdmissionController(root).__enter__())
    assert cw.serve(root, tmp_path, "invalid").code == 78
    assert launcher.install_controller(root, world["archive"])["controller_epoch"] == 4
    bad.unlink()  # the runbook's recovery: only after status names the log
    assert compaction.compact_abort(root, compaction_id)["phase"] == "aborted"
    _h_restored(world, tmp_path, h_head, members)


# --- F12: coded refusals, never tracebacks ---

def test_unreadable_databases_and_os_errors_are_coded(x_archive, tmp_path, capsys, monkeypatch):
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    (world["states"]["a"]["state"] / "data" / "junk.db").write_bytes(b"not sqlite" * 100)
    args = SimpleNamespace(compact_command="plan", root=root, k_slot="k", archive=world["archive"],
                           retire_prepare_leftovers=False)
    assert compaction.run_command(args, scan=cw.no_units) == 78
    output = json.loads(capsys.readouterr().out)
    assert output["refused"] == "runtime_admission_compaction_database_unreadable"
    assert "junk" not in json.dumps(output) and str(root) not in json.dumps(output)
    status = SimpleNamespace(compact_command="status", root=root, compaction_id=None)
    for error, expected in ((sqlite3.DatabaseError("file is not a database"),
                             {"refused": "runtime_admission_compaction_database_unreadable"}),
                            (PermissionError(errno.EACCES, "denied", str(root / "x")),
                             {"refused": "runtime_admission_compaction_io_error", "errno": "EACCES"})):
        def raising(*args, _error=error, **kwargs):
            raise _error
        monkeypatch.setattr(compaction, "compact_status", raising)
        assert compaction.run_command(status) == 78
        assert json.loads(capsys.readouterr().out) == expected


# --- Gaps: T-U10 and K' vs H maintenance on relocated collector targets ---

ERASED = "00000000-0000-4000-8000-0000000000e1"
EXPIRING = "00000000-0000-4000-8000-0000000000e2"
KEPT = "00000000-0000-4000-8000-0000000000e3"
_GRAPH = "\n".join([
    "import asyncio, json, sys, threading",
    "from pathlib import Path",
    f"sys.path.append({str(cw.REPO)!r})",
    "from tests.mocks.collector_erasure import ACTOR, graph",
    "from tests.mocks.memory_sources import memory_source",
    # The run's child stands in for src.main without building the application graph (as the world's
    # LEDGER child does): the admission's output registration is not under test here, so its path
    # check is answered as for an unmanaged state. The erasure and TTL code paths run unchanged.
    "import src.core.runtime_admission as _admission",
    "_admission._validate_path = lambda path, *, initial: None",
])
# Owner a: three wake clips in a's own wake target (a's run, real WakeClipWriter).
_A_CLIPS = _GRAPH + "\n" + "\n".join([
    "async def _clips():",
    "    async with graph(state / 'data') as g:",
    f"        for conversation, ttl in (({ERASED!r}, 48 * 3600), ({EXPIRING!r}, 600), ({KEPT!r}, 48 * 3600)):",
    "            await g.add('wake', memory_source(now=g.clock.now, ttl=ttl, conversation_id=conversation))",
    "asyncio.run(_clips())",
])
# H: the erasure of ERASED is requested and left pending (no driver runs in H).
_H_REQUEST = _GRAPH + "\n" + "\n".join([
    "from src.memory.erasure_coordinator import ErasureCoordinator",
    "from src.memory.privacy_services import PrivacyServices",
    "async def _request():",
    "    privacy = PrivacyServices(str(state / 'data' / 'privacy.db'))",
    "    await privacy.acall(privacy.initialize)",
    "    owner = ErasureCoordinator(privacy)",
    f"    await owner.request({ERASED!r}, ACTOR)",
    "    await owner.aclose()",
    "    await privacy.aclose()",
    "asyncio.run(_request())",
])


def _k_run(out: Path) -> str:
    """K's verification run: the erasure step and the TTL sweep, on K's relocated targets."""
    return _GRAPH + "\n" + "\n".join([
        "result = {}",
        "async def _maintain():",
        "    async with graph(state / 'data') as g:",
        f"        steps = await g.settle(memory_source(conversation_id={ERASED!r}))",
        "        result['erasure'] = sorted({row['state'] for row in steps})",
        "        g.clock.now += 3600",
        "        await g.store.reconcile()",
        "        result['sweep'] = str(g.store._status['state'])",
        "def _thread():",
        "    try:",
        "        asyncio.run(_maintain())",
        "    except BaseException as error:",
        "        result['error'] = type(error).__name__ + ':' + str(error)[:200]",
        "worker = threading.Thread(target=_thread)",
        "worker.start()",
        "worker.join()",
        f"Path({str(out)!r}).write_text(json.dumps(result))",
    ])


def _clips(state: Path) -> dict:
    """conversation -> clip ids, and the files present, of one state's wake target rows."""
    with sqlite3.connect(state / "data" / "privacy.db") as db:
        rows = db.execute("SELECT id, sources_json FROM wake_clips_v1").fetchall()
    found = {}
    for identifier, sources in rows:
        for source in json.loads(sources):
            found.setdefault(source["conversation_id"], set()).add(identifier)
    return found


def _files(state: Path, identifiers: set) -> set:
    directory = state / "data" / "wake" / "policy_v1"  # clip_directory's layout
    return {path.name for path in directory.iterdir()
            if path.name.split(".")[0] in identifiers} if directory.exists() else set()


def test_t_u10_pending_erasure_and_ttl_on_inherited_targets_run_on_k_prime(x_archive, tmp_path):
    """T-U10: H's pending erasure of a clip held in an inherited (owner a) target is done by K's run,
    on K's relocated copy; the TTL sweep finds a relocated clip too; the kept clip stays."""
    world = cw.world(x_archive, tmp_path, hops=("b",), first=(_A_CLIPS,))
    root = world["root"]
    a_state = world["states"]["a"]["state"]
    clips = _clips(a_state)
    assert set(clips) >= {ERASED, EXPIRING, KEPT}
    b_binding = cw.run(root, "b", _H_REQUEST)  # H closes again with the erasure pending
    assert stage.read_admission_head(root)["binding_id"] == b_binding
    h_state = world["states"]["b"]["state"]
    with sqlite3.connect(h_state / "data" / "privacy.db") as db:
        pending = db.execute("SELECT COUNT(*) FROM erasure_steps WHERE participant='collectors' "
                             "AND state!='complete'").fetchone()[0]
    assert pending >= 1
    compaction_id = cw.start(root, world["archive"])["compaction_id"]
    k_state = root / "slots" / "k" / "state"
    with sqlite3.connect(k_state / "data" / "privacy.db") as db:
        [k_root] = [json.loads(row[0])["root"] for row in db.execute(
            "SELECT descriptor FROM erasure_targets WHERE participant='collectors'")
            if json.loads(row[0])["root"].endswith("/data/wake")]
    assert k_root == str(k_state / "data" / "wake")  # relocated from a's state into K
    for conversation in (ERASED, EXPIRING, KEPT):
        assert _files(k_state, clips[conversation]), conversation
    compaction.compact_verify(root, compaction_id)
    out = tmp_path / "k-run.json"
    result = cw.serve(root, tmp_path, "verify", execute=_k_run(out), timeout=120)
    assert result.code == 0, result.phases
    report = json.loads(out.read_text())
    assert "error" not in report, report
    assert report["erasure"] == ["complete"]
    assert not _files(k_state, clips[ERASED]) and not _files(k_state, clips[EXPIRING])
    assert _files(k_state, clips[KEPT])
    assert ERASED not in _clips(k_state) and EXPIRING not in _clips(k_state)
    done = compaction.compact_retire(root, compaction_id, scan=cw.no_units)
    assert done["phase"] == "complete"
    [started] = stage.compaction_events(stage.read_compaction_log(root, compaction_id), "retire_started")
    assert started["data"]["coverage"]["unexplained"] == 0 and started["data"]["coverage"]["removed_by_k_run"] > 0
    gone = clips[ERASED] | clips[EXPIRING]
    leftover = [path for path in root.rglob("*") if path.name.split(".")[0] in gone]
    assert leftover == []


# --- F9 follow-up: kza's stock user units (IBus, podman, unity) do not block compaction ---

def test_stock_user_units_with_unresolved_references_do_not_block_compaction(x_archive, tmp_path, monkeypatch):
    """The U3b-0e4 rule in P14: the five stock units the coordinator found in kza's manager pass the
    real scanner in plan, start and retire; a named unit with the same reference still refuses."""
    from tests.unit.tools.test_compaction_units import KZA_MANAGER, FakeUserManager, _stock_units, exec_value, unit
    world = cw.world(x_archive, tmp_path, hops=("b",))
    root = world["root"]
    (root / "logs").mkdir()  # the install record's console rotation, as install-launcher leaves it
    (root / "console-logrotate.conf").write_text(launcher.render_console_logrotate(str(root / "logs" / "console.log")))
    home = (tmp_path / "account-home").resolve()
    user_dir = home / ".config" / "systemd" / "user"
    user_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR"):
        monkeypatch.delenv(name, raising=False)
    stock = _stock_units(SimpleNamespace(user_dir=user_dir), podman_environment=False)
    manager = FakeUserManager(stock, manager_environment=KZA_MANAGER)
    scan = compaction._default_scan(manager=manager)
    planned = cw.plan(root, world["archive"], scan=scan)
    assert planned["units"]["units_scanned"] == 5
    assert len(planned["units"]["tolerated_unresolved"]) == 5
    compaction_id = cw.start(root, world["archive"], planned=planned, scan=scan)["compaction_id"]
    compaction.compact_verify(root, compaction_id)
    assert cw.serve(root, tmp_path, "verify").code == 0
    assert compaction.compact_retire(root, compaction_id, scan=scan)["phase"] == "complete"
    named = dict(stock, **{"kza-code-index.service": unit(str(user_dir / "kza-code-index.service"), loaded=True,
                                                         cat="[Service]\nExecStart=/bin/tool $LOGGING\n",
                                                         ExecStart=exec_value("/bin/tool", "$LOGGING"))})
    _refusal("runtime_admission_compaction_unit_unparsed:kza-code-index.service:ExecStart", lambda: cw.plan(
        root, world["archive"], k="k2", scan=compaction._default_scan(manager=FakeUserManager(
            named, manager_environment=KZA_MANAGER))))
