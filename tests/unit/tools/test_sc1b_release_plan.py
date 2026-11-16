"""SC-1b: release-mode plan and start (sc-release-design v4 implementation_units SC-1b).

mechanism.mode (release mode is the only mode for new logs at controller
epoch >= 5), mechanism.plan.p6_prime (a)-(g) with served(root), the equal
triple rule and the named P6'(c) refusals t0_required and controller_behind
(R5-3), plan.other (forced leftovers, the unproven-copy refusal and its
legacy exemption), start (intent.mode, _k_runtime against intent.archive,
A1 schemas equality and the relocation fix of the queued audio paths) and
crash row P (a refused plan writes nothing; the unit serves H).

Worlds are real managed roots. The post-R1 world is built and served by the
epoch-4 build of this tree (what R1 is to production) and then T0 installs
X5, this working tree at epoch 5, in process. Variants of X5 differ only in
CONTROLLER_REVISION, in one comment of a controller-closure file (X5C) or in
one comment outside the closure (X5S, the equal triple of another release).
Complete release-mode logs (served(root)) are written as immutable logs by
hand, since a release-mode log cannot reach 'complete' before SC-4a/SC-4b.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scripts import compaction
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime
from src.memory import owner_compaction as oc
from tests.mocks import compaction_world as cw
from tests.mocks import launcher_rehearsal as rehearsal
from tests.mocks.epoch4 import edited_commit, epoch4_commit, epoch4_in_process

REVISION_1 = b"\nCONTROLLER_REVISION = 1\n"
FUTURE = b"from __future__ import annotations\n"
OUTSIDE_CLOSURE = "src/health/health_check.py"


@pytest.fixture(scope="module")
def clone(tmp_path_factory):
    """A shared clone with X5 = HEAD + this working tree committed (the base of every variant)."""
    base = tmp_path_factory.mktemp("sc1b-builds")
    path = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(cw.REPO), str(path)],
                   check=True, timeout=120)
    return {"base": base, "path": path, "x5": rehearsal.overlay_commit(cw.REPO, path, message="sc-1b X5")}


def _build(clone, name: str, revision: str, *, sources: bool = False) -> dict:
    archive = clone["base"] / (name + ".tar.gz")
    manifest = build_runtime(clone["path"], archive, revision)
    found = {"archive": archive, "manifest": manifest, "triple": stage.controller_triple(manifest),
             "release_sha256": compaction._sha256_bytes(compaction._release_json_bytes(manifest))}
    if sources:  # the source tree of that revision, for an in-process install of its build (SC-1a _install)
        found["sources"] = clone["base"] / (name + "-src")
        found["sources"].mkdir()
        tar = subprocess.run(["git", "archive", "--format=tar", revision, "src", "scripts"], cwd=clone["path"],
                             check=True, capture_output=True, timeout=120).stdout
        subprocess.run(["tar", "-x", "-C", str(found["sources"])], input=tar, check=True, timeout=120)
    return found


@pytest.fixture(scope="module")
def x5(clone):
    return _build(clone, "x5", clone["x5"])


@pytest.fixture(scope="module")
def x4(clone):
    return _build(clone, "x4", epoch4_commit(clone["path"], clone["x5"]))


@pytest.fixture(scope="module")
def x5s(clone):
    """The equal triple of another release: one comment outside the controller closure."""
    revision = edited_commit(clone["path"], clone["x5"], {OUTSIDE_CLOSURE: [
        (b"Health Check Module\n", b"Health Check Module (sc-1b: outside the controller closure)\n")]},
        message="sc-1b X5 same triple")
    return _build(clone, "x5s", revision)


@pytest.fixture(scope="module")
def x5r2(clone):
    revision = edited_commit(clone["path"], clone["x5"], {"scripts/stage_runtime.py": [
        (REVISION_1, b"\nCONTROLLER_REVISION = 2\n")]}, message="sc-1b X5 revision 2")
    return _build(clone, "x5r2", revision, sources=True)


@pytest.fixture(scope="module")
def x5c(clone):
    revision = edited_commit(clone["path"], clone["x5"], {"src/memory/owner_compaction.py": [
        (FUTURE, FUTURE + b"# sc-1b: a controller-only change\n")]}, message="sc-1b X5 closure change")
    return _build(clone, "x5c", revision)


@pytest.fixture(autouse=True)
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold for an in-process controller."""
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)


def _install(root: Path, build: dict) -> dict:
    """This code installs ``build`` in process, as that build's own T0 command would (SC-1a _install)."""
    if "sources" not in build:
        return launcher.install_controller(root, build["archive"])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(launcher, "_installer_tree", lambda: build["sources"])
        return launcher.install_controller(root, build["archive"])


def _post_r1_world(x4: dict, x5: dict, tmp_path: Path) -> dict:
    """The first post-R1 transition's starting point: an epoch-4 head (R1's stand-in), then T0 to X5."""
    with epoch4_in_process():
        world = cw.world(x4["archive"], tmp_path, hops=())
    _install(world["root"], x5)
    assert stage.installed_controller_triple(world["root"]) == x5["triple"]
    return world


def _world(build: dict, tmp_path: Path) -> dict:
    """An epoch-5 managed root whose head and installed controller are ``build``."""
    with pytest.MonkeyPatch.context() as patch:
        if "sources" in build:
            patch.setattr(launcher, "_installer_tree", lambda: build["sources"])
        return cw.world(build["archive"], tmp_path, hops=())


def _refusal(code: str, action):
    with pytest.raises(ValueError) as caught:
        action()
    assert str(caught.value) == code or str(caught.value).startswith(code + ":"), str(caught.value)
    return caught.value


def _triples(installed: tuple, archive: tuple) -> str:
    return ":installed=" + ".".join(map(str, installed)) + ",archive=" + ".".join(map(str, archive))


def _row_p(root: Path, tmp_path: Path, name: str, code: str, action) -> None:
    """Crash row P: the refused plan writes nothing, and the unit serves H (R4-OPS-10a)."""
    witness = cw.witness(root)
    _refusal(code, action)
    assert cw.witness(root) == witness
    assert cw.serve(root, tmp_path, name).code == 0


def _complete_log(root: Path, release_sha256: str, *, mode: str | None = "release",
                  events: tuple[str, ...] = ("verified", "complete")) -> str:
    """An immutable log as a finished transition leaves it (intent, then events), written by hand."""
    compaction_id = uuid4().hex
    intent = {"schema_version": 1, "compaction_id": compaction_id, "kind": "compaction_intent",
              "archive": {"release_sha256": release_sha256}, "members": [], "owners": []}
    if mode is not None:
        intent["mode"] = mode
    stage._atomic_record(stage.compaction_log_path(root, compaction_id) / "intent.json", intent, immutable=True)
    log = stage.read_compaction_log(root, compaction_id)
    for kind in events:
        log = stage.append_compaction_event(root, log, kind, {}, at=1.0)
    return compaction_id


# --- mechanism.mode ---------------------------------------------------------------------------------

def test_release_mode_is_keyed_on_the_controller_epoch():
    assert compaction.release_mode() is True
    with epoch4_in_process():
        assert compaction.release_mode() is False
    assert compaction.release_log({"intent": {"mode": "release"}})
    assert not compaction.release_log({"intent": {}})


# --- the first post-R1 transition: R1 -> R2 plan admission, intent, start --------------------------

def test_first_post_r1_plan_admits_new_and_start_runs_k_from_new(x4, x5, tmp_path):
    """R1 (epoch 4) is H, X5 is installed by T0 and is NEW: equal triple, so no further T0 is needed.

    The plan binds mode and NEW's release fields (P6'(e)); retire_prepare_leftovers is forced on; the
    intent carries mode 'release'; K's runtime is NEW's archive (P6'(f)); A1 and A2 pass. Until SC-4a
    the log cannot be armed and the plain-mode paths refuse it (V4R2-A1); its plain abort is lossless.
    """
    world = _post_r1_world(x4, x5, tmp_path)
    root = world["root"]
    h_head = stage.read_admission_head(root)
    h_state = cw.tree(world["h"]["state"])
    _row_p(root, tmp_path, "h-epoch4-new", "runtime_admission_compaction_release_epoch",
           lambda: cw.plan(root, x4["archive"]))  # P6'(b): R1 (epoch 4) is never NEW (SC-D7 A)
    witness = cw.witness(root)
    planned = cw.plan(root, x5["archive"])  # retire_prepare_leftovers not given: forced on
    assert cw.witness(root) == witness  # read-only
    started = cw.start(root, x5["archive"], planned=planned)
    # Review B4: until SC-4a the only exit of a release-mode log is the plain abort, and start says so.
    assert started["next"] == "compact abort " + started["compaction_id"]
    assert compaction.compact_resume(root, started["compaction_id"])["next"] == started["next"]
    log = stage.read_compaction_log(root, started["compaction_id"])
    intent = log["intent"]
    assert intent["mode"] == "release" and compaction.release_log(log)
    assert intent["retire_prepare_leftovers"] is True
    assert intent["archive"] == {"release_sha256": x5["release_sha256"], "git_commit": x5["manifest"]["git_commit"],
                                 "schema_contract": 1, "controller_triple": list(x5["triple"])}
    k_runtime = root / "slots" / "k" / "runtime"
    assert compaction._sha256_bytes((k_runtime / "release.json").read_bytes()) == x5["release_sha256"]
    head = stage.read_admission_head(root)
    k = stage.read_binding(root, head["binding_id"])
    h = stage.read_binding(root, h_head["binding_id"])
    assert k["slot"] == "k" and k["release_sha256"] == x5["release_sha256"] != h["release_sha256"]
    assert stage.release_integers(k)["controller_epoch"] == 5
    [status] = compaction.compact_status(root, started["compaction_id"])["compactions"]
    assert status["mode"] == "release" and status["phase"] == "activated"
    compaction_id = started["compaction_id"]
    # SC1B-N3: no verification run before SC-4a; the plain-mode paths refuse a log with 'mode'.
    _refusal(compaction.RELEASE_VERIFY_UNAVAILABLE, lambda: compaction.compact_verify(root, compaction_id))
    _refusal(compaction.PLAIN_MODE_ONLY, lambda: compaction.compact_allow_run(root, compaction_id, reason="x",
                                                                              hours=1))
    _refusal(compaction.PLAIN_MODE_ONLY, lambda: compaction.compact_abort(root, compaction_id,
                                                                          discard_k_writes=True))
    _refusal(compaction.PLAIN_MODE_ONLY, lambda: compaction._q6_checks(
        root, stage.read_compaction_log(root, compaction_id), discard_k_writes=False,
        confirm_h=h_head["binding_id"][:12], verified=True))
    assert not any(event["kind"] in {"verify_armed", "allow_run", "abort_started"}
                   for event in stage.read_compaction_log(root, compaction_id)["events"])
    assert cw.serve(root, tmp_path, "unarmed").code == 78  # K is never served unverified
    # Row A0: the plain abort is lossless; H serves again.
    assert compaction.compact_abort(root, compaction_id)["phase"] == "aborted"
    assert stage.read_admission_head(root)["binding_id"] == h_head["binding_id"]
    assert cw.tree(world["h"]["state"]) == h_state
    assert cw.serve(root, tmp_path, "h-again").code == 0


def test_equal_triple_roll_forward_needs_no_t0_and_a_greater_triple_does(x5, x5s, x5r2, x5c, tmp_path):
    """CR-9, CR2-9, R5-3: equal triple admitted; greater -> t0_required, equal pair with another digest
    -> revision conflict, each naming both triples; after the T0 the greater release is admitted."""
    world = _world(x5, tmp_path)
    root = world["root"]
    assert x5s["triple"] == x5["triple"] and x5s["release_sha256"] != x5["release_sha256"]
    assert OUTSIDE_CLOSURE not in {path for path, _digest in x5s["manifest"]["controller_sources"]}
    assert cw.plan(root, x5s["archive"])["plan_sha256"] != cw.plan(root, x5["archive"])["plan_sha256"]
    _row_p(root, tmp_path, "h-t0", compaction.T0_REQUIRED + _triples(x5["triple"], x5r2["triple"]),
           lambda: cw.plan(root, x5r2["archive"]))
    assert x5c["triple"][:2] == x5["triple"][:2] and x5c["triple"] != x5["triple"]
    _row_p(root, tmp_path, "h-conflict", "runtime_admission_compaction_controller_revision_conflict"
           + _triples(x5["triple"], x5c["triple"]), lambda: cw.plan(root, x5c["archive"]))
    _install(root, x5r2)  # the T0 the refusal asked for
    # The head's own release stays in served(root): planning it again needs no T0 (lower triple).
    assert cw.plan(root, x5["archive"])["plan_sha256"]
    started = cw.start(root, x5r2["archive"])
    intent = stage.read_compaction_log(root, started["compaction_id"])["intent"]
    assert intent["archive"]["release_sha256"] == x5r2["release_sha256"]
    assert intent["archive"]["controller_triple"] == list(x5r2["triple"])


def test_controller_behind_and_the_rollback_admitted_through_served(x5, x5r2, tmp_path):
    """R5-3 and review_resolution_v4 r3[6]: under a newer-revision controller, NEW with a lower triple is
    refused controller_behind unless it is in served(root): a complete, verified release-mode log's archive.
    A log that is not verified, not release mode, or aborted serves nothing; two transitions may share
    a release."""
    world = _world(x5r2, tmp_path)
    root = world["root"]
    behind = compaction.CONTROLLER_BEHIND + _triples(x5r2["triple"], x5["triple"])
    _row_p(root, tmp_path, "h-behind", behind, lambda: cw.plan(root, x5["archive"]))
    for mode, events in ((None, ("verified", "complete")), ("release", ("complete",)),
                         ("release", ("verified", "aborted"))):
        _complete_log(root, x5["release_sha256"], mode=mode, events=events)
        _refusal(behind, lambda: cw.plan(root, x5["archive"]))
    head = stage.read_binding(root, stage.read_admission_head(root)["binding_id"])
    pending = _complete_log(root, x5["release_sha256"], events=("verified",))  # verified, never retired
    assert compaction.served_releases(root, head) == {x5r2["release_sha256"]}
    stage.append_compaction_event(root, stage.read_compaction_log(root, pending), "aborted", {}, at=2.0)
    _complete_log(root, x5["release_sha256"])
    _complete_log(root, x5["release_sha256"])  # two transitions that shared a release
    assert compaction.served_releases(root, head) == {x5r2["release_sha256"], x5["release_sha256"]}
    started = cw.start(root, x5["archive"])  # a rollback with no T0
    log = stage.read_compaction_log(root, started["compaction_id"])
    assert log["intent"]["archive"]["controller_triple"] == list(x5["triple"])
    assert stage.read_binding(root, stage.read_admission_head(root)["binding_id"])["release_sha256"] \
        == x5["release_sha256"]


# --- P6'(a), (b) upper bound, (d), (g) ------------------------------------------------------------------

def test_new_manifest_integers_and_schema_contracts(x5, x5s, tmp_path, monkeypatch):
    world = _world(x5, tmp_path)
    root = world["root"]
    original = compaction._release_manifest

    def with_values(**values):
        monkeypatch.setattr(compaction, "_release_manifest", lambda archive: {**original(archive), **values})

    for values, code in (({"launcher_contract": 2}, "runtime_admission_launcher_contract_missing"),
                         ({"format_epoch": 0}, "runtime_admission_rollback_format_closed"),
                         ({"schema_contract": 2}, "runtime_admission_compaction_schema_contract"),
                         ({"controller_epoch": 6}, "runtime_admission_compaction_release_epoch")):
        with_values(**values)
        _row_p(root, tmp_path, "h-" + next(iter(values)), code, lambda: cw.plan(root, x5s["archive"]))
    monkeypatch.setattr(compaction, "_release_manifest", original)
    assert cw.plan(root, x5s["archive"])["plan_sha256"]

    def h_refused(binding):
        raise ValueError("runtime_admission_launcher_contract_missing")
    monkeypatch.setattr(stage, "_require_launcher_slot", h_refused)  # (d) H passes serve's slot integers
    _refusal("runtime_admission_launcher_contract_missing", lambda: cw.plan(root, x5s["archive"]))


# --- plan.other: unproven copies and the legacy exemption ----------------------------------------------

def _adoption_record(root: Path, container: str, legacy: str, *, aborted: bool = False) -> None:
    head = {"status": "closed", "binding_id": "0" * 64, "generation": 1}
    record = {"kind": "adoption", "head": head, "container_slot": container,
              "layout": {"legacy_state": str(root / "slots" / legacy / "state")}}
    stage._atomic_record(stage.adoption_record_path(root, head), record, immutable=True)
    if aborted:
        stage._atomic_record(stage.abort_record_path(root, head), {"aborted": True}, immutable=True)


def test_release_plan_refuses_an_unproven_copy_under_slots_but_not_the_legacy_slots(x5, tmp_path):
    world = _world(x5, tmp_path)
    root = world["root"]
    old = root / "slots" / "old"
    (old / "state" / "data").mkdir(parents=True)
    (old / "state" / "data" / "x.txt").write_text("CANARY-UNPROVEN")
    error = _refusal("runtime_admission_compaction_release_unproven_copy", lambda: cw.plan(root, x5["archive"]))
    assert error.report == {"paths": ["slots/old"]}
    assert "CANARY" not in json.dumps(error.report)
    _adoption_record(root, "old", "legacy_b0866e8", aborted=True)  # an aborted adoption exempts nothing
    _refusal("runtime_admission_compaction_release_unproven_copy", lambda: cw.plan(root, x5["archive"]))
    for path in (root / "admission" / "adoption").iterdir():
        path.chmod(0o600)
        path.unlink()
    _adoption_record(root, "old", "legacy_b0866e8")  # the committed record's two legacy names
    planned = cw.plan(root, x5["archive"])
    assert {entry["path"] for entry in planned["untouched"]} >= {"slots/old"}
    (root / "slots" / "other").mkdir()
    error = _refusal("runtime_admission_compaction_release_unproven_copy", lambda: cw.plan(root, x5["archive"]))
    assert error.report == {"paths": ["slots/other"]}


# --- start: A1 schemas equality ----------------------------------------------------------------------

def test_a1_refuses_k_whose_schemas_differ_from_h(x5, tmp_path):
    """start.a1_extra: K's binding 'schemas' must equal H's; a refusal at A1 leaves H the head."""
    world = _world(x5, tmp_path)
    root = world["root"]
    h_head = stage.read_admission_head(root)

    def worker(root_path, compaction_id):
        compaction.run_worker(root_path, compaction_id)
        with sqlite3.connect(root_path / "slots" / "k" / "state" / "data" / "privacy.db") as db:
            db.execute("CREATE INDEX sc1b_extra ON erasure_targets(participant)")

    planned = cw.plan(root, x5["archive"])
    _refusal("runtime_admission_compaction_schema_changed",
             lambda: compaction.compact_start(root, "k", x5["archive"], plan_sha256=planned["plan_sha256"],
                                              acknowledge_untouched=planned["untouched_sha256"],
                                              destroy_rollback_states=True, scan=cw.no_units, worker=worker))
    assert stage.read_admission_head(root) == h_head
    [log] = stage.compaction_logs(root)
    assert not any(event["kind"] == "authorized" for event in log["events"])
    assert compaction.compact_abort(root, log["compaction_id"])["phase"] == "aborted"
    assert cw.serve(root, tmp_path, "h").code == 0


# --- start.relocation_fix: queued audio deletions and write intents (all compactions) ----------------

_QUEUE = "\n".join([
    "import asyncio, sqlite3",
    "from contextlib import closing",
    "from src.ambient.store import AmbientStore",
    "async def _init():",
    "    store = AmbientStore(str(state / 'data' / 'ambient.db'), audio_dir=str(state / 'data' / 'ambient_audio'))",
    "    await store.init()",
    "    await store.close()",
    "asyncio.run(_init())",
    "audio = state / 'data' / 'ambient_audio'",
    "audio.mkdir(parents=True, exist_ok=True)",
    "(audio / 'queued.flac').write_bytes(b'fLaC-sc1b-queued')",
    "with closing(sqlite3.connect(state / 'data' / 'ambient.db')) as db, db:",
    # A queued deletion whose unlink failed in H (attempts, last_error), and an open write intent.
    "    db.execute('INSERT INTO pending_audio_deletions(path, requested_at, attempts, last_error)'",
    "               ' VALUES (?,?,?,?)', (str(audio / 'queued.flac'), 1.0, 1, 'PermissionError'))",
    "    db.execute('INSERT INTO audio_write_intents VALUES (?,?,?,?)',",
    "               (str(audio / 'intent.flac'), 7, 'conversation-1', 4102444800.0))",
])


def _k_retry(out: Path) -> str:
    """K's run: the production retry of queued audio deletions, on K's own ambient store."""
    return "\n".join([
        "import asyncio, json, os, sys, threading",
        "from pathlib import Path",
        "sys.path.insert(0, os.getcwd())",
        "from src.ambient.store import AmbientStore",
        "result = {}",
        "async def _retry():",
        "    store = AmbientStore(str(state / 'data' / 'ambient.db'), audio_dir=str(state / 'data' / 'ambient_audio'))",
        "    await store.init()",
        "    try:",
        "        result['removed'] = await store._retry_audio_deletions()",
        "    finally:",
        "        await store.close()",
        "def _thread():",
        "    try:",
        "        asyncio.run(_retry())",
        "    except BaseException as error:",
        "        result['error'] = type(error).__name__ + ':' + str(error)[:200]",
        "worker = threading.Thread(target=_thread)",
        "worker.start()",
        "worker.join()",
        f"Path({str(out)!r}).write_text(json.dumps(result))",
    ])


def _queue_rows(state: Path) -> dict:
    with sqlite3.connect(state / "data" / "ambient.db") as db:
        return {"deletions": [row[0] for row in db.execute("SELECT path FROM pending_audio_deletions")],
                "intents": [row[0] for row in db.execute("SELECT path FROM audio_write_intents")]}


@pytest.mark.parametrize("hops", [(), ("b",)], ids=["h-root", "owner-root"])
def test_queued_audio_paths_are_relocated_into_k_and_k_removes_its_own_file(x4_archive, tmp_path, hops):
    """Before SC-1b _check_locators refused these rows (absolute_locator:...pending_audio_deletions.path).
    Now (plain mode here: the fix covers every compaction) K's rows name K's path, D5 and A1 pass
    (K's protected projection is H's with the path remapped, also from an owner's root, which only an
    epoch-4 world has), and K's run unlinks K's file, never the source's."""
    world = cw.world(x4_archive, tmp_path, hops=hops, first=(_QUEUE,))
    root = world["root"]
    h_state = world["h"]["state"]
    source = world["states"]["a"]["state"]  # the rows name the state that queued them (H, or owner a)
    h_file = source / "data" / "ambient_audio" / "queued.flac"
    assert _queue_rows(h_state) == {"deletions": [str(h_file)],
                                    "intents": [str(source / "data" / "ambient_audio" / "intent.flac")]}
    started = cw.start(root, x4_archive)
    assert started["next"] == "compact verify " + started["compaction_id"]  # plain mode (review B4)
    k_state = root / "slots" / "k" / "state"
    k_file = k_state / "data" / "ambient_audio" / "queued.flac"
    assert _queue_rows(k_state) == {"deletions": [str(k_file)],
                                    "intents": [str(k_state / "data" / "ambient_audio" / "intent.flac")]}
    assert k_file.read_bytes() == h_file.read_bytes() == b"fLaC-sc1b-queued"
    compaction.compact_verify(root, started["compaction_id"])
    out = tmp_path / "k-retry.json"
    result = cw.serve(root, tmp_path, "verify", execute=_k_retry(out), timeout=120)
    assert result.code == 0, result.phases
    report = json.loads(out.read_text())
    assert report == {"removed": 1}, report
    assert not k_file.exists() and _queue_rows(k_state)["deletions"] == []
    assert h_file.read_bytes() == b"fLaC-sc1b-queued"  # the source is never touched by K's run


def _snapshot(rows: list[dict], table: str = "pending_audio_deletions") -> SimpleNamespace:
    return SimpleNamespace(manifest={"databases": {"data/ambient.db": {"tables": {table: {"rows": rows}}}}})


@pytest.mark.parametrize("table", sorted(oc.QUEUE_PATH_TABLES))
def test_a_queue_path_naming_a_source_root_outside_it_still_refuses(table):
    root = "/srv/kza/slots/h/state"
    _ = oc._check_locators(_snapshot([{"path": root + "/data/ambient_audio/a.flac"}], table), (root,))
    _ = oc._check_locators(_snapshot([{"path": "/elsewhere/a.flac"}], table), (root,))  # names no source root
    for outside in (root + "-old/data/a.flac", "/tmp/copy" + root + "/data/a.flac"):
        with pytest.raises(oc.CompactionCloneError) as caught:
            oc._check_locators(_snapshot([{"path": outside}], table), (root,))
        assert str(caught.value) == "runtime_admission_compaction_absolute_locator:" + table + ".path"


def test_two_queue_rows_relocated_to_one_k_path_refuse_before_any_write():
    roots = ("/r/slots/a/state", "/r/slots/b/state")
    rows = [{"path": roots[0] + "/data/x.flac"}, {"path": roots[1] + "/data/x.flac"}]
    with pytest.raises(oc.CompactionCloneError) as caught:
        oc._check_queue_relocations(_snapshot(rows), roots, Path("/r/slots/k/state"))
    assert str(caught.value) == "runtime_admission_compaction_locator_collision:pending_audio_deletions.path"
    oc._check_queue_relocations(_snapshot(rows[:1] + [{"path": "/elsewhere/x.flac"}]), roots,
                                Path("/r/slots/k/state"))


def test_the_controller_remaps_queue_paths_as_the_worker_does():
    roots = ("/r/slots/a/state", "/r/slots/h/state")
    destination = "/r/slots/k/state"
    for value in ("/r/slots/h/state/data/q.flac", "/r/slots/a/state/x/y.flac", "/elsewhere/q.flac",
                  "relative/q.flac", "/r/slots/h/state"):
        assert stage._queue_path_in_k(value, roots, destination) == \
            oc._relocated_value(value, roots, Path(destination))
    nested = ("/r/s", "/r/s/nested")  # the longest matching root wins, on both sides
    assert stage._queue_path_in_k("/r/s/nested/x.flac", nested, destination) == destination + "/x.flac" \
        == oc._relocated_value("/r/s/nested/x.flac", nested, Path(destination))
    assert stage._queue_path_in_k(None, roots, destination) is None
    assert stage.compaction_source_roots({"owners": [{"state_root": "/r/slots/a/state"}]},
                                         {"state_root": "/r/slots/h/state"}) == roots
