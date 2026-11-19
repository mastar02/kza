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
hand here; a real one (V, then retire with SC-4b's strict W equality) is in
test_sc4b_release_exits.
"""

from __future__ import annotations

import json
import os
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
    assert launcher.release_mode_log({"intent": {"mode": "release"}})
    assert not launcher.release_mode_log({"intent": {}})
    # Review RA-2: plain mode is only a log without 'mode'; any other value refuses everywhere.
    for mode in ("verify", "Release", "plain", None, 1):
        log = {"intent": {"mode": mode}}
        _refusal("runtime_admission_compaction_log_invalid", lambda: compaction.release_log(log))
        _refusal("runtime_admission_compaction_log_invalid", lambda: compaction._require_plain_mode(log))
        _refusal("runtime_admission_compaction_log_invalid", lambda: launcher.release_mode_log(log))
    compaction._require_plain_mode({"intent": {}})


# --- the first post-R1 transition: R1 -> R2 plan admission, intent, start --------------------------

def test_first_post_r1_plan_admits_new_and_start_runs_k_from_new(x4, x5, tmp_path):
    """R1 (epoch 4) is H, X5 is installed by T0 and is NEW: equal triple, so no further T0 is needed.

    The plan binds mode and NEW's release fields (P6'(e)); retire_prepare_leftovers is forced on; the
    intent carries mode 'release'; K's runtime is NEW's archive (P6'(f)); A1 and A2 pass. The
    plain-mode paths refuse it (V4R2-A1); before its run is armed its plain abort is lossless (row A0).
    """
    world = _post_r1_world(x4, x5, tmp_path)
    root = world["root"]
    h_head = stage.read_admission_head(root)
    h_state = cw.tree(world["h"]["state"])
    _row_p(root, tmp_path, "h-epoch4-new", "runtime_admission_compaction_release_epoch",
           lambda: cw.plan(root, x4["archive"]))  # P6'(b): R1 (epoch 4) is never NEW (SC-D7 A)
    assert str(_refusal("runtime_admission_compaction_release_epoch", lambda: cw.plan(root, x4["archive"]))) \
        == "runtime_admission_compaction_release_epoch"  # not t0_required: no T0 makes R1 a NEW
    witness = cw.witness(root)
    planned = cw.plan(root, x5["archive"])  # retire_prepare_leftovers not given: forced on
    assert cw.witness(root) == witness  # read-only
    started = cw.start(root, x5["archive"], planned=planned)
    # SC-4b lifted the release gate (SC4A-N13): start names the verification run.
    assert started["next"] == "compact verify " + started["compaction_id"]
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
    # SC4A-N13's gate mechanism, should it be closed again: no arm (with or without N) and start/resume
    # name the lossless abort. As shipped (SC-4b) the log is armable; N stays explicit (SC4A-N14).
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compaction, "RELEASE_EXITS_AVAILABLE", False)
        for smoke in (None, 10):
            _refusal(compaction.RELEASE_EXIT_UNAVAILABLE,
                     lambda: compaction.compact_verify(root, compaction_id, smoke_seconds=smoke))
        assert compaction._next_after_activation(log) == "compact abort " + compaction_id
    assert compaction.RELEASE_EXITS_AVAILABLE is True
    _refusal(compaction.RELEASE_SMOKE_SECONDS, lambda: compaction.compact_verify(root, compaction_id))
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


def _cut_start(root: Path, archive: Path, k: str) -> str:
    """Crash row S: start writes its intent, then is cut before C2a (K's runtime)."""
    def cut(*args, **kwargs):
        raise RuntimeError("cut")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compaction, "_k_runtime", cut)
        with pytest.raises(RuntimeError, match="cut"):
            cw.start(root, archive, k)
    [log] = [log for log in stage.compaction_logs(root) if not log["terminal"]]
    assert [event["kind"] for event in log["events"]] == []
    return log["compaction_id"]


def _resume_refused(root: Path, compaction_id: str, h_head: dict, action, report) -> None:
    error = _refusal("runtime_admission_compaction_artifact", action)
    assert error.report == report
    log = stage.read_compaction_log(root, compaction_id)
    assert not any(event["kind"] == "k_runtime" for event in log["events"])
    assert stage.read_admission_head(root) == h_head


def test_a_cut_start_resumes_only_with_the_intent_archive(x4, x5, x5r2, tmp_path):
    """Row S of the first post-R1 update (reviews RA-1, B1). resume does not re-run P6': an unadmitted
    NEW (x5r2, greater triple, no T0) or H's own archive is refused before anything is written, so the
    resume with the intent's NEW still reaches activation. P6'(f) itself (K's release.json against
    intent.archive) refuses an archive swapped between that check and the extraction."""
    world = _post_r1_world(x4, x5, tmp_path)
    root = world["root"]
    h_head = stage.read_admission_head(root)
    h_state = cw.tree(world["h"]["state"])
    compaction_id = _cut_start(root, x5["archive"], "k")
    witness = cw.witness(root)
    for archive in (x5r2["archive"], x4["archive"]):  # an unadmitted NEW; H's archive ($H_ARCHIVE)
        _resume_refused(root, compaction_id, h_head,
                        lambda: compaction.compact_resume(root, compaction_id, archive=archive),
                        {"archive": "not_the_intent_archive"})
        assert cw.witness(root) == witness and not (root / "slots" / "k").exists()
    resumed = compaction.compact_resume(root, compaction_id, archive=x5["archive"])
    assert resumed["phase"] == "activated" and resumed["next"] == "compact verify " + compaction_id  # SC-4b
    k = stage.read_binding(root, stage.read_admission_head(root)["binding_id"])
    assert k["slot"] == "k" and k["release_sha256"] == x5["release_sha256"]
    assert compaction.compact_abort(root, compaction_id)["phase"] == "aborted"
    assert stage.read_admission_head(root)["binding_id"] == h_head["binding_id"]
    h_head = stage.read_admission_head(root)  # the return to H may publish a new generation
    # P6'(f): the archive passes the pre-check, then another one is extracted (swapped in between), or
    # K's release.json no longer matches its own prepare record: K's file is compared with the intent.
    prepare = stage._prepare_slot

    def other_archive(archive, *args, **kwargs):
        return prepare(x5r2["archive"], *args, **kwargs)

    def release_rewritten(archive, root_path, k_slot, *args, **kwargs):
        result = prepare(archive, root_path, k_slot, *args, **kwargs)
        release = root_path / "slots" / k_slot / "runtime" / "release.json"
        mode = release.stat().st_mode
        release.chmod(0o600)
        release.write_bytes(compaction._release_json_bytes(x5r2["manifest"]))
        release.chmod(mode)
        return result

    for k_slot, swap in (("k2", other_archive), ("k3", release_rewritten)):
        compaction_id = _cut_start(root, x5["archive"], k_slot)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(stage, "_prepare_slot", swap)
            _resume_refused(root, compaction_id, h_head,
                            lambda: compaction.compact_resume(root, compaction_id, archive=x5["archive"]), None)
        assert compaction.compact_abort(root, compaction_id)["phase"] == "aborted"  # lossless, serves H
        assert stage.read_admission_head(root) == h_head
    assert cw.tree(world["h"]["state"]) == h_state
    assert cw.serve(root, tmp_path, "h-after-cut").code == 0


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
    # Review RA-2: a finished log with an unknown mode is neither plain nor served: it refuses.
    odd = _complete_log(root, x5r2["release_sha256"], mode="verify")
    _refusal("runtime_admission_compaction_log_invalid", lambda: compaction.served_releases(root, head))
    [listed] = [entry for entry in compaction.compact_status(root)["compactions"] if entry["compaction_id"] == odd]
    assert listed == {"compaction_id": odd, "invalid": True, "code": "runtime_admission_compaction_log_invalid"}


# --- P6'(a), (b) upper bound, (d), (g) ------------------------------------------------------------------

def test_new_manifest_integers_and_schema_contracts(x5, x5s, tmp_path, monkeypatch):
    world = _world(x5, tmp_path)
    root = world["root"]
    original = compaction._release_manifest

    def with_values(**values):
        monkeypatch.setattr(compaction, "_release_manifest", lambda archive: {**original(archive), **values})

    # Review B3: an epoch above the installed one is (c)'s greater triple: t0_required naming both.
    epoch6 = compaction.T0_REQUIRED + _triples(x5["triple"], (6,) + tuple(x5s["triple"][1:]))
    for values, code in (({"launcher_contract": 2}, "runtime_admission_launcher_contract_missing"),
                         ({"format_epoch": 0}, "runtime_admission_rollback_format_closed"),
                         ({"schema_contract": 2}, "runtime_admission_compaction_schema_contract"),
                         ({"controller_epoch": 6}, epoch6)):
        with_values(**values)
        _row_p(root, tmp_path, "h-" + next(iter(values)), code, lambda: cw.plan(root, x5s["archive"]))
    monkeypatch.setattr(compaction, "_release_manifest", original)
    assert cw.plan(root, x5s["archive"])["plan_sha256"]

    # Review RA-4: SR-2's legs for the installed controller and for H (NEW unchanged).
    installed = compaction._installed_manifest
    integers = stage.release_integers
    for target, name, patched in (
            (compaction, "_installed_manifest", lambda root_path: {**installed(root_path), "schema_contract": 2}),
            (stage, "release_integers", lambda binding: {**integers(binding), "schema_contract": 2})):
        witness = cw.witness(root)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(target, name, patched)
            _refusal("runtime_admission_compaction_schema_contract", lambda: cw.plan(root, x5s["archive"]))
        assert cw.witness(root) == witness
    assert cw.plan(root, x5s["archive"])["plan_sha256"]

    def h_refused(binding):
        raise ValueError("runtime_admission_launcher_contract_missing")
    monkeypatch.setattr(stage, "_require_launcher_slot", h_refused)  # (d) H passes serve's slot integers
    _refusal("runtime_admission_launcher_contract_missing", lambda: cw.plan(root, x5s["archive"]))


# --- plan.other: unproven copies and the legacy exemption ----------------------------------------------

def _adoption_record(root: Path, container: str, legacy: str, *, aborted: bool = False, generation: int = 1,
                     named_as: dict | None = None) -> dict:
    head = {"status": "closed", "binding_id": "0" * 64, "generation": generation}
    record = {"kind": "adoption", "head": head, "container_slot": container,
              "layout": {"legacy_state": str(root / "slots" / legacy / "state")}}
    stage._atomic_record(stage.adoption_record_path(root, named_as or head), record, immutable=True)
    if aborted:
        stage._atomic_record(stage.abort_record_path(root, head), {"aborted": True}, immutable=True)
    return head


def _clear_adoption(root: Path) -> None:
    for path in (root / "admission" / "adoption").iterdir():
        path.chmod(0o600)
        path.unlink()


def test_release_plan_refuses_an_unproven_copy_under_slots_but_not_the_legacy_slots(x5, tmp_path):
    """plan.other: the container and the legacy-state slot of exactly one committed adoption record
    (named by its head digest) are exempt; anything else under slots/ refuses as row P (review RA-3, B2)."""
    world = _world(x5, tmp_path)
    root = world["root"]
    unproven = "runtime_admission_compaction_release_unproven_copy"
    plan = lambda: cw.plan(root, x5["archive"])  # noqa: E731
    both = ["slots/legacy_b0866e8", "slots/old"]
    for name in ("old", "legacy_b0866e8"):  # R1's e9669ec container and its b0866e8 legacy-state slot
        (root / "slots" / name / "state" / "data").mkdir(parents=True)
        (root / "slots" / name / "state" / "data" / "x.txt").write_text("CANARY-UNPROVEN")
    witness = cw.witness(root)
    error = _refusal(unproven, plan)
    assert error.report == {"paths": both}
    assert "CANARY" not in json.dumps(error.report)
    assert cw.witness(root) == witness and cw.serve(root, tmp_path, "h-unproven").code == 0  # row P
    _adoption_record(root, "old", "legacy_b0866e8", aborted=True)  # an aborted adoption exempts nothing
    assert _refusal(unproven, plan).report == {"paths": both}
    _clear_adoption(root)
    first = _adoption_record(root, "old", "legacy_b0866e8")  # the committed record's two legacy names
    planned = plan()
    assert {entry["path"] for entry in planned["untouched"]} >= set(both)
    # Two committed records (naming the same slots, either one sorting first) exempt nothing.
    _adoption_record(root, "old", "legacy_b0866e8", generation=2)
    assert _refusal(unproven, plan).report == {"paths": both}
    _clear_adoption(root)
    # A record whose file name is not its own head's digest exempts nothing.
    _adoption_record(root, "old", "legacy_b0866e8", generation=3, named_as=first)
    assert _refusal(unproven, plan).report == {"paths": both}
    _clear_adoption(root)
    _adoption_record(root, "old", "legacy_b0866e8")
    assert plan()["plan_sha256"]
    # Review B2: an empty, unrecorded .prepare-* (a start cut before its staging record) refuses as
    # row P; the documented exit is rmdir, then the plan again.
    leftover = root / "slots" / ".prepare-sc1bcut"
    leftover.mkdir()
    _row_p(root, tmp_path, "h-prepare", unproven, plan)
    witness = cw.witness(root)  # after row P's serve, which moves the head's run records
    assert _refusal(unproven, plan).report == {"paths": ["slots/.prepare-sc1bcut"]}
    assert cw.witness(root) == witness
    os.rmdir(leftover)
    assert plan()["plan_sha256"]
    (root / "slots" / "other").mkdir()
    assert _refusal(unproven, plan).report == {"paths": ["slots/other"]}


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


@pytest.mark.parametrize("table", sorted(oc.QUEUE_PATH_TABLES))
def test_two_queue_rows_relocated_to_one_k_path_refuse_before_any_write(table):
    """Two relocated rows, or a relocated row and an unchanged row already naming K's path (review RA-5)."""
    roots = ("/r/slots/a/state", "/r/slots/b/state")
    destination = Path("/r/slots/k/state")
    code = "runtime_admission_compaction_locator_collision:" + table + ".path"
    for rows in ([{"path": roots[0] + "/data/x.flac"}, {"path": roots[1] + "/data/x.flac"}],
                 [{"path": roots[1] + "/data/x.flac"}, {"path": str(destination) + "/data/x.flac"}]):
        with pytest.raises(oc.CompactionCloneError) as caught:
            oc._check_queue_relocations(_snapshot(rows, table), roots, destination)
        assert str(caught.value) == code
    oc._check_queue_relocations(_snapshot([{"path": roots[0] + "/data/x.flac"}, {"path": "/elsewhere/x.flac"},
                                           {"path": str(destination) + "/data/y.flac"}], table), roots, destination)


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
