"""SC-1a: controller epoch 5, the controller triple and install-controller's T0 checks.

sc-release-design v4 implementation_units SC-1a, mechanism.t0_controller_install,
mechanism.authz3_1, epochs and crash rows T0, T0S, T0F, DL, DO. Worlds are
real managed roots: the pre-T0 root is built and served by the epoch-4 build
of this tree (tests/mocks/epoch4.py, what R1 is to production), then the
epoch-5 build (X5) is installed by this code in process, as the T0 command
runs NEW's own stage_runtime. Variants of X5 differ only in
CONTROLLER_REVISION or in one comment of a controller-closure file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import compaction
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from scripts.build_runtime import controller_closure
from tests.mocks.runtime_artifact_cache import cached_build_runtime as build_runtime
from tests.mocks import compaction_world as cw
from tests.mocks import launcher_rehearsal as rehearsal
from tests.mocks.epoch4 import edited_commit, epoch4_commit, epoch4_in_process
from tests.unit.tools.test_managed_launcher_serve import TEMPLATE, _settings, _systemctl

BASE_REV = stage.CONTROLLER_REVISION
BASE_REV_LINE = f"\nCONTROLLER_REVISION = {BASE_REV}\n".encode()
NEXT_REV_LINE = f"\nCONTROLLER_REVISION = {BASE_REV + 1}\n".encode()
HIGHER_REV_LINE = f"\nCONTROLLER_REVISION = {BASE_REV + 2}\n".encode()
# The last commit before SC-1a: an epoch-4 controller without the first_install checks (R1's behaviour).
PRE_SC1A = "b15ce86a1ea7e1926e11c7fdb6c67f6831a9d3c1"
# R1, the pinned production release (epoch 4, docs/architecture/DEPLOYMENT.md).
R1_COMMIT = "357dcb5"
OWNER_COMPACTION = "src/memory/owner_compaction.py"
FUTURE = b"from __future__ import annotations\n"


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    """X4 (epoch 4), X5 (base revision) and variants with a bumped revision or changed closure."""
    base = tmp_path_factory.mktemp("sc1a-builds")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(cw.REPO), str(clone)],
                   check=True, timeout=120)
    x5 = rehearsal.overlay_commit(cw.REPO, clone, message="sc-1a X5")
    revisions = {
        "x5": x5,
        "x4": epoch4_commit(clone, x5),
        "x5r2": edited_commit(clone, x5, {"scripts/stage_runtime.py": [(BASE_REV_LINE, NEXT_REV_LINE)]},
                              message=f"sc-1a X5 revision {BASE_REV + 1}"),
        "x5r3": edited_commit(clone, x5, {"scripts/stage_runtime.py": [(BASE_REV_LINE, HIGHER_REV_LINE)]},
                              message=f"sc-1a X5 revision {BASE_REV + 2}"),
        "pre": PRE_SC1A,
        "x5c": edited_commit(clone, x5, {OWNER_COMPACTION: [(FUTURE, FUTURE + b"# sc-1a: a controller-only change\n")]},
                             message="sc-1a X5 closure change"),
        # Next revision with another managed_launcher (review F5: the installer identity covers it too)
        "x5m": edited_commit(clone, x5, {"scripts/stage_runtime.py": [(BASE_REV_LINE, NEXT_REV_LINE)],
                                         "scripts/managed_launcher.py": [(FUTURE, FUTURE + b"# sc-1a: launcher\n")]},
                             message=f"sc-1a X5 revision {BASE_REV + 1}, launcher changed"),
        "r1": R1_COMMIT,
    }
    archives, manifests, sources = {}, {}, {}
    for name, revision in revisions.items():
        archives[name] = base / (name + ".tar.gz")
        manifests[name] = build_runtime(clone, archives[name], revision)
        if name.startswith("x5") and name != "x5":
            # The source tree of that revision, for an in-process install of its build (_install).
            sources[name] = base / (name + "-src")
            sources[name].mkdir()
            tar = subprocess.run(["git", "archive", "--format=tar", revision, "src", "scripts"], cwd=clone,
                                 check=True, capture_output=True, timeout=120).stdout
            subprocess.run(["tar", "-x", "-C", str(sources[name])], input=tar, check=True, timeout=120)
    assert manifests["pre"]["controller_epoch"] == 4 and "controller_revision" not in manifests["pre"]
    assert manifests["r1"]["controller_epoch"] == 4
    return {"archives": archives, "manifests": manifests, "sources": sources, "base": base}


def _extract(archive: Path, destination: Path) -> Path:
    """An extracted archive, as the operator unpacks NEW for T0; returns its stage_runtime.pyc."""
    manifest, files = stage._read_archive(archive)
    for name, data in files.items():
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        (destination / name).write_bytes(data)
    (destination / "release.json").write_text(json.dumps(manifest))
    return destination / "scripts" / "stage_runtime.pyc"


@pytest.fixture(autouse=True)
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold for an in-process controller."""
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)


def _world(builds, tmp_path, *, hops=(), first=()):
    """An epoch-4 managed root (controller, launcher record, closed head 'a' or the last hop)."""
    with epoch4_in_process():
        return cw.world(builds["archives"]["x4"], tmp_path, hops=hops, first=first)


def _refused(code, action):
    with pytest.raises(launcher.LauncherRefusal) as caught:
        action()
    assert caught.value.code == code or caught.value.code.startswith(code + ":"), caught.value.code
    return caught.value


def _install(root, builds, name="x5", **options):
    """This code installs build ``name`` in process, as that build's own T0 command would.

    X5 is this working tree, so the installer runs from the checkout. For the
    other revisions the installer-identity check (review F6) is pointed at
    that revision's sources (launcher._installer_tree), as if their checkout ran.
    """
    if name not in builds["sources"]:
        return launcher.install_controller(root, builds["archives"][name], **options)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(launcher, "_installer_tree", lambda: builds["sources"][name])
        return launcher.install_controller(root, builds["archives"][name], **options)


def _record(root):
    return json.loads((root / "admission" / "controller.json").read_text())


def _current(root):
    return os.readlink(root / "controller" / "current")


def _unchanged_t0(root, witness, action, code):
    """Row T0: refused before current moves; nothing written by the epoch-5 checks."""
    _refused(code, action)
    assert cw.witness(root) == witness


def _serve(root, tmp_path, name):
    return cw.serve(root, tmp_path, name)


# --- epochs and the controller triple (epochs.controller_epoch_5, controller_revision) -------------

def test_controller_epoch_5_is_declared():
    assert stage.CONTROLLER_EPOCH == 5 and BASE_REV >= 1
    assert stage.RELEASE_MODE_EPOCH == 5
    assert (stage.SCHEMA_CONTRACT_EPOCH, stage.FORMAT_EPOCH, stage.LAUNCHER_CONTRACT,
            stage.RECEIPT_STATE_SCHEMA) == (2, 1, 3, 2)


def test_controller_epoch_5_and_revision_are_declared(builds):
    manifest = builds["manifests"]["x5"]
    assert manifest["controller_epoch"] == 5 and manifest["controller_revision"] == BASE_REV
    assert stage.controller_triple(manifest) == (5, BASE_REV, manifest["controller_sources_sha256"])
    assert builds["manifests"]["x5r2"]["controller_revision"] == BASE_REV + 1
    assert builds["manifests"]["x5r3"]["controller_revision"] == BASE_REV + 2
    x4 = builds["manifests"]["x4"]
    assert x4["controller_epoch"] == 4 and x4["controller_revision"] == BASE_REV


def test_the_closure_reaches_the_worker_and_the_named_modules(builds):
    """R4-OPS-1: the computed closure contains the worker's tree and its compaction modules."""
    paths = [path for path, _digest in builds["manifests"]["x5"]["controller_sources"]]
    for required in ("scripts/stage_runtime.py", "scripts/compaction.py", "scripts/managed_launcher.py",
                     "scripts/adoption_commands.py", "scripts/adoption.py", "scripts/compaction_units.py",
                     "src/core/compaction_worker.py", "src/core/runtime_admission.py", "src/core/entry_guard.py",
                     "src/core/own_temporary.py", "src/memory/manual_components.py",
                     *("src/memory/" + name + ".py" for name in (
                         "owner_compaction", "state_snapshot", "copy_catalog", "copy_observation",
                         "state_journal_derivation", "payload_derivation", "fact_ledger", "erasure_journal",
                         "deletion_ledger", "voice_schema")),
                     "src/__init__.py", "src/core/__init__.py", "src/memory/__init__.py"):
        assert required in paths, required
    assert "scripts/build_runtime.py" not in paths
    assert stage.controller_sources_digest(builds["manifests"]["x5"]["controller_sources"]) == \
        builds["manifests"]["x5"]["controller_sources_sha256"]


def test_a_change_only_to_owner_compaction_changes_the_triple(builds):
    """R4-OPS-1: same (epoch, revision), another digest; a T0 with a bumped revision is required."""
    x5, x5c = builds["manifests"]["x5"], builds["manifests"]["x5c"]
    assert stage.controller_triple(x5)[:2] == stage.controller_triple(x5c)[:2]
    assert x5["controller_sources_sha256"] != x5c["controller_sources_sha256"]
    changed = {path for (path, digest), (_path, other) in zip(x5["controller_sources"], x5c["controller_sources"])
               if digest != other}
    assert changed == {OWNER_COMPACTION}


def test_controller_triple_validation():
    sources = [["scripts/stage_runtime.py", "a" * 64], ["src/x.py", "b" * 64]]
    good = {"controller_epoch": 5, "controller_revision": BASE_REV, "controller_sources": sources,
            "controller_sources_sha256": stage.controller_sources_digest(sources)}
    files = {"scripts/stage_runtime.pyc": b"", "src/x.pyc": b""}
    assert stage.controller_triple(good, files)[:2] == (5, BASE_REV)
    for change in ({"controller_revision": 0}, {"controller_revision": str(BASE_REV)}, {"controller_sources_sha256": "0" * 64},
                   {"controller_sources": list(reversed(sources))}, {"controller_sources": sources + [sources[0]]},
                   {"controller_sources": []}, {"controller_sources_sha256": None},
                   {"controller_sources": [["/abs.py", "a" * 64]]}, {"controller_sources": [["tools/x.py", "a" * 64]]}):
        value = dict(good, **change)
        if "controller_sources" in change and "controller_sources_sha256" not in change and change["controller_sources"]:
            value["controller_sources_sha256"] = stage.controller_sources_digest(change["controller_sources"])
        with pytest.raises(ValueError, match="^runtime_admission_controller_unmanaged$"):
            stage.controller_triple(value, files)
    with pytest.raises(ValueError, match="^runtime_admission_controller_unmanaged$"):
        stage.controller_triple(good, {"scripts/stage_runtime.pyc": b""})  # src/x.pyc not compiled in
    assert stage.controller_triple({"controller_epoch": 4}) == (4, 0, None)


def test_build_runtime_records_the_closure_only_for_a_revision_that_declares_it():
    from scripts import build_runtime as builder
    assert builder.RELEASE_INTEGERS["controller_revision"] == "CONTROLLER_REVISION"
    assert "controller_revision" not in builder.release_integers(b"CONTROLLER_EPOCH = 4\n")
    assert "controller_revision" in stage._RELEASE_INTEGER_KEYS
    sources = {path: (cw.REPO / path).read_bytes() for path in subprocess.run(
        ["git", "ls-files", "src", "scripts"], cwd=cw.REPO, capture_output=True, text=True, check=True,
        timeout=60).stdout.split() if path.endswith(".py")}
    closure = controller_closure(sources)
    assert builder.controller_sources_digest(closure) == stage.controller_sources_digest(closure)
    assert closure == sorted(closure)


def test_every_path_loaded_module_of_the_closure_is_listed():
    """SC-R input: the controller loads siblings by path; each such load is in CONTROLLER_PATH_LOADED."""
    from scripts import build_runtime as builder
    sources = {path: (cw.REPO / path).read_bytes() for path in subprocess.run(
        ["git", "ls-files", "src", "scripts"], cwd=cw.REPO, capture_output=True, text=True, check=True,
        timeout=60).stdout.split() if path.endswith(".py")}
    dynamic = {path for path, _digest in controller_closure(sources)
               if any(marker in sources[path] for marker in (b"spec_from_file_location", b"import_module(",
                                                              b"__import__("))}
    # stage_runtime loads managed_launcher, adoption, adoption_commands, compaction_units, compaction and
    # manual_components by path; runtime_admission loads a slot's stage_runtime (the slot's code, not the
    # controller's); piper_tts imports torch by name (third party).
    assert dynamic <= {"scripts/stage_runtime.py", "src/core/runtime_admission.py", "src/tts/piper_tts.py"}, dynamic
    text = sources["scripts/stage_runtime.py"].decode()
    for stem in ("managed_launcher", "adoption", "adoption_commands", "compaction_units", "compaction"):
        assert f'"{stem}"' in text, stem
        assert f"scripts/{stem}.py" in builder.CONTROLLER_PATH_LOADED
    assert '"src/memory/manual_components"' in text
    assert "src/memory/manual_components.py" in builder.CONTROLLER_PATH_LOADED


# Pinned release (epoch, revision) pairs and their closure digest.
# The release procedure (SC-5b) pins the pair to ship, and from then on a closure change without a
# CONTROLLER_REVISION bump fails here, before install-controller would refuse it (CR2-9, CO-5).
RELEASED_CONTROLLERS: dict[tuple[int, int], str] = {
    (5, 1): "8cec6c0477f7b69052a4ae9124c42e27af233733e314b2a3ebbdfd830e6f5b69",
    (5, 2): "03c2eb14c428c23c40c1e235c77e135a6f71e45bede3913af5de6a2aa801049a",
    (5, 3): "a8d934150f69c616cf95dc1ea25fe6ab994ba3e100a27e623458c454e378c345",
    (5, 4): "e05f35beeb8411a1a4640bdfee6d1fd0dbff9790e208cc33baf69779fc337485",
    (5, 5): "cb245eecbaf5178b09fe74a4a592512611059744b90702fd01468c38404faabf",
    (5, 6): "7161ff0e06eac36577a4e41b4e1c400b9a3019bf57ef9f5d45d951f696323768",
    (5, 7): "168f96633caca417d59effac6fcb6c1e70e0b24e0559d07e1b52462b98f71952",
    (5, 8): "137d4c5783bdfe36e2f08f529c62a515c0efe75321df6203026b87f00ec0a0c5",
    (5, 9): "54a7bbb89ce91f972eac180c17e3e702e140f155953e3ea69dac61da7ff4c7d0",
}


def _released_pair_check(released: dict, epoch: int, revision: int, digest: str) -> None:
    pinned = released.get((epoch, revision))
    assert pinned is None or pinned == digest, \
        f"controller ({epoch}, {revision}) was released with another closure: bump CONTROLLER_REVISION"


def test_a_released_controller_pair_keeps_its_digest():
    sources = {path: (cw.REPO / path).read_bytes() for path in subprocess.run(
        ["git", "ls-files", "src", "scripts"], cwd=cw.REPO, capture_output=True, text=True, check=True,
        timeout=60).stdout.split() if path.endswith(".py")}
    digest = stage.controller_sources_digest(controller_closure(sources))
    _released_pair_check(RELEASED_CONTROLLERS, stage.CONTROLLER_EPOCH, stage.CONTROLLER_REVISION, digest)
    pair = (stage.CONTROLLER_EPOCH, BASE_REV)
    _released_pair_check({pair: digest}, *pair, digest)
    with pytest.raises(AssertionError, match="bump CONTROLLER_REVISION"):
        _released_pair_check({pair: "0" * 64}, *pair, digest)


# --- first install (t0_controller_install.first_install, CR2-1) --------------------------------------

def test_first_t0_over_an_epoch4_root_installs_and_h_serves(builds, tmp_path):
    world = _world(builds, tmp_path)
    root = world["root"]
    before = _record(root)
    assert before["controller_epoch"] == 4
    with epoch4_in_process():
        assert _serve(root, tmp_path, "h-old").code == 0  # H under the epoch-4 controller
    result = _install(root, builds)
    assert (result["controller_epoch"], result["controller_revision"]) == (5, BASE_REV)
    assert result["controller_sources_sha256"] == builds["manifests"]["x5"]["controller_sources_sha256"]
    record = _record(root)
    assert set(record) == {"schema_version", "controller_epoch", "commit", "min_format_epoch", "installed_at"}
    assert record["controller_epoch"] == 5 and record["commit"] == builds["manifests"]["x5"]["git_commit"]
    assert stage.installed_controller_triple(root) == stage.controller_triple(builds["manifests"]["x5"])
    assert _serve(root, tmp_path, "h-new").code == 0  # t0_controller_install.after: H serves at epoch 5


def test_first_t0_refuses_a_pending_and_an_unreadable_log(builds, tmp_path):
    """V4R2-L2 and R3-OPS-6a: first_install reads the log itself (the install gate would ignore it)."""
    world = _world(builds, tmp_path)
    root = world["root"]
    with epoch4_in_process():
        started = cw.start(root, builds["archives"]["x4"])
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds), "runtime_admission_compaction_pending")
    events = stage.compaction_log_path(root, started["compaction_id"]) / "events"
    (events / "000099.json").write_text("{}")  # a gap: the log is unreadable
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds), "runtime_admission_compaction_log_invalid")
    assert _record(root)["controller_epoch"] == 4


def test_first_t0_refuses_owners_and_a_delegation_record_before_current_moves(builds, tmp_path):
    """V4R2-A1 (authz3_1 guard (3)): a root with a record or owners never gets the epoch-5 controller."""
    world = _world(builds, tmp_path, hops=("b",))
    root = world["root"]
    current = _current(root)
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds), stage.DELEGATION_FROZEN)
    records = root / "admission" / "delegations"
    moved = tmp_path / "records-aside"
    shutil.move(records, moved)  # only the owners remain: the head's policy still lists delegated targets
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds), stage.OWNERS_PRESENT)
    assert _current(root) == current and _record(root)["controller_epoch"] == 4


def test_first_t0_refuses_an_open_head(builds, tmp_path):
    world = _world(builds, tmp_path)
    root = world["root"]
    head = stage.read_admission_head(root)
    with stage.AdmissionController(root):
        stage._cas_head(root, head, head["binding_id"], "unknown", "0" * 32)
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds), "runtime_admission_not_closed")


def test_sr1_binding_equality_refuses_a_changed_state(builds, tmp_path):
    """R3-OPS-2: this code recomputes H's binding in process; a difference refuses before current moves."""
    world = _world(builds, tmp_path)
    root = world["root"]
    # Only the ownership pin changes: the stored binding names no policy file, the state now has one.
    (world["h"]["state"] / ".kza-target-ownership.json").write_text("{}")
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds), launcher.CONTROLLER_BINDING_CHANGED)


def test_sr1_sdk_versions_only_refuses_sdk_changed_naming_the_bound_version(builds, tmp_path, monkeypatch):
    chroma = "\n".join(["import sqlite3", "from contextlib import closing",
                        "(state / 'data' / 'memory_db').mkdir(parents=True, exist_ok=True)",
                        "with closing(sqlite3.connect(state / 'data' / 'memory_db' / 'chroma.sqlite3')) as db, db:",
                        "    for table in ('collections', 'segments', 'migrations'):",
                        "        db.execute(f'CREATE TABLE IF NOT EXISTS {table} (id TEXT)')"])
    world = _world(builds, tmp_path, first=(chroma,))
    root = world["root"]
    bound = stage.read_binding(root, stage.read_admission_head(root)["binding_id"])["sdk_versions"]["chromadb"]
    import importlib.metadata
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "0.0.0+changed" if name == "chromadb" else original(name))
    witness = cw.witness(root)
    refusal = _refused(launcher.SDK_CHANGED, lambda: _install(root, builds))
    assert refusal.code == f"{launcher.SDK_CHANGED}: chromadb=={bound}"
    assert cw.witness(root) == witness


def test_sr1_refuses_when_the_head_slot_check_fails(builds, tmp_path):
    """CR-6: serve's read-only slot check runs on H's slot before current moves."""
    world = _world(builds, tmp_path)
    root = world["root"]
    target = root / "slots" / "a" / "runtime" / "src" / "main.pyc"
    target.chmod(0o600)
    target.write_bytes(target.read_bytes() + b"\0")
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds), "Runtime manifest or file integrity mismatch")


def test_sr1_refuses_another_schema_contract(builds, tmp_path, monkeypatch):
    world = _world(builds, tmp_path)
    root = world["root"]
    original = stage._read_archive

    def read(archive):
        manifest, files = original(archive)
        return dict(manifest, schema_contract=stage.SCHEMA_CONTRACT_EPOCH + 1), files
    monkeypatch.setattr(stage, "_read_archive", read)
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds), launcher.SCHEMA_CONTRACT_CHANGED)


def test_sr1_runs_the_slot_probe(builds, tmp_path, monkeypatch):
    world = _world(builds, tmp_path)
    root = world["root"]
    calls = []

    def probe(root_, name, binding, *, timeout):
        calls.append(name)
        raise ValueError("runtime_admission_candidate_rejected: probe_timeout")
    monkeypatch.setattr(stage, "_probe_candidate", probe)
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds), "runtime_admission_candidate_rejected")
    assert calls == ["a"]


# --- crash windows (rows T0, T0S; CR2-1, CR-5) ---------------------------------------------------------

def test_kill_after_the_first_install_checks_then_serve_old_then_repeat(builds, tmp_path):
    world = _world(builds, tmp_path)
    root = world["root"]
    record, current = _record(root), _current(root)

    def die(step):
        if step == "checked":
            raise KeyboardInterrupt("synthetic kill")
    with pytest.raises(KeyboardInterrupt):
        _install(root, builds, step_hook=die)
    assert _record(root) == record and _current(root) == current
    with epoch4_in_process():
        assert _serve(root, tmp_path, "old").code == 0  # H under the old controller
    _install(root, builds)
    assert _record(root)["controller_epoch"] == 5
    assert _serve(root, tmp_path, "new").code == 0


def test_kill_between_the_switch_and_controller_json_then_repeat(builds, tmp_path):
    world = _world(builds, tmp_path)
    root = world["root"]
    record = _record(root)

    def die(step):
        if step == "switched":
            raise KeyboardInterrupt("synthetic kill")
    with pytest.raises(KeyboardInterrupt):
        _install(root, builds, step_hook=die)
    assert _record(root) == record and _current(root) == builds["manifests"]["x5"]["git_commit"]
    # CS2-9: every gate but 'install' refuses until the install is repeated.
    served = _serve(root, tmp_path, "t0s")
    assert served.code == 78 and any(stage.CONTROLLER_INSTALL_INCOMPLETE in event for event in served.serve.events)
    with pytest.raises(ValueError, match=stage.CONTROLLER_INSTALL_INCOMPLETE):
        stage.AdmissionController(root).__enter__()
    with pytest.raises(ValueError, match=stage.CONTROLLER_INSTALL_INCOMPLETE):
        cw.plan(root, builds["archives"]["x4"])
    _install(root, builds)  # first_install again: keyed on controller.json epoch < 5
    assert _record(root)["controller_epoch"] == 5
    assert _serve(root, tmp_path, "after").code == 0


def test_cs2_9_unit(tmp_path):
    root = tmp_path
    (root / "admission").mkdir()
    (root / "controller" / "c1").mkdir(parents=True)
    (root / "controller" / "c1" / "release.json").write_text(json.dumps({"git_commit": "a" * 40}))
    (root / "controller" / "current").symlink_to("c1")
    stage._atomic_record(root / "admission" / "controller.json", {"controller_epoch": 5, "commit": "a" * 40})
    stage.require_forward_epoch(root)
    stage._atomic_record(root / "admission" / "controller.json", {"controller_epoch": 5, "commit": "b" * 40})
    with pytest.raises(ValueError, match=stage.CONTROLLER_INSTALL_INCOMPLETE):
        stage.require_forward_epoch(root)
    stage.require_forward_epoch(root, gate="install")
    with stage.AdmissionController(root, gate="install"):
        pass
    with pytest.raises(ValueError, match=stage.CONTROLLER_INSTALL_INCOMPLETE):
        stage.AdmissionController(root, gate="serve").__enter__()
    # serve's own installed-controller check (also run by install-launcher and the adoption commands)
    import hashlib
    (root / "controller" / "c1" / "scripts").mkdir()
    (root / "controller" / "c1" / "scripts" / "x.pyc").write_bytes(b"x")
    (root / "controller" / "c1" / "release.json").write_text(json.dumps(
        {"git_commit": "a" * 40, "files": {"scripts/x.pyc": hashlib.sha256(b"x").hexdigest()}}))
    _refused(stage.CONTROLLER_INSTALL_INCOMPLETE, lambda: launcher.check_controller_epoch(root))
    stage._atomic_record(root / "admission" / "controller.json", {"controller_epoch": 5, "commit": "a" * 40})
    launcher.check_controller_epoch(root)
    # Review F5: an unreadable or missing current manifest never counts as matching at a lease gate.
    manifest = root / "controller" / "c1" / "release.json"
    manifest.write_text("not json")
    with pytest.raises(ValueError, match=stage.CONTROLLER_INSTALL_INCOMPLETE):
        stage.require_forward_epoch(root, gate="normal")
    manifest.unlink()
    with pytest.raises(ValueError, match=stage.CONTROLLER_INSTALL_INCOMPLETE):
        stage.require_forward_epoch(root, gate="normal")
    with pytest.raises(ValueError, match=stage.CONTROLLER_INSTALL_INCOMPLETE):
        stage.AdmissionController(root, gate="serve").__enter__()


# --- later installs (t0_controller_install.later_installs) ---------------------------------------------

@pytest.fixture
def t0_root(builds, tmp_path):
    world = _world(builds, tmp_path)
    _install(world["root"], builds)
    return world


def test_install_is_monotone_in_epoch_and_revision(builds, t0_root, tmp_path):
    """CR2-9: lower revision refused; equal pair needs an equal digest; equal triple is harmless."""
    root = t0_root["root"]
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5c"), launcher.CONTROLLER_REVISION_CONFLICT)
    record = _record(root)
    _install(root, builds, "x5")  # equal triple: a harmless repeat
    assert _record(root)["commit"] == record["commit"] and _current(root) == record["commit"]
    _install(root, builds, "x5r3")
    assert stage.installed_controller_triple(root)[:2] == (5, BASE_REV + 2)
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"), "runtime_admission_controller_downgrade")
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5"), "runtime_admission_controller_downgrade")
    assert _serve(root, tmp_path, "r3").code == 0


def test_later_install_refuses_an_unknown_head_with_no_pending_log(builds, t0_root, tmp_path):
    """R4-OPS-2: close an unknown head with R-b or adopt first; then a greater triple installs (T0F exit)."""
    root = t0_root["root"]
    crashed = cw.serve(root, tmp_path, "crash", scenario="crash", seconds=0.1)
    assert crashed.code == 68 and stage.read_admission_head(root)["status"] == "unknown"
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"), "runtime_admission_not_closed")
    report = launcher.adopt(root, confirm="", regressions=[], dry_run=True)
    adopted = launcher.adopt(root, confirm=report["head_sha256"], regressions=report["report"].get("regressions", []))
    head = stage.read_admission_head(root)
    assert head["status"] == "closed" and stage.read_manual_head(root, head)["kind"] == "recovery"
    assert adopted["head"] == head
    _install(root, builds, "x5r2")  # SR-1 alone gates it; no H-serve step needed (R4-OPS-2)
    assert stage.installed_controller_triple(root)[:2] == (5, BASE_REV + 1)
    assert _serve(root, tmp_path, "after").code == 0


def test_t0f_exit_a_corrected_controller_at_a_higher_revision_serves_h(builds, t0_root, tmp_path):
    root = t0_root["root"]
    _install(root, builds, "x5r2")
    assert _record(root)["commit"] == builds["manifests"]["x5r2"]["git_commit"]
    assert _serve(root, tmp_path, "h").code == 0


def _open_k_under_a_pending_log(builds, root, tmp_path):
    """A compaction at epoch 5 (no owners), then K's head left open as by a run that died.

    Since SC-1b every new log at epoch 5 is a release-mode log (intent.mode = 'release', NEW = X5);
    a carried plain-mode log is emulated by making release_mode_log answer False.
    """
    started = cw.start(root, builds["archives"]["x5"])
    assert stage.pending_compaction(root)["intent"]["mode"] == "release"
    head = stage.read_admission_head(root)
    with stage.AdmissionController(root, allow_open_head=True, gate="compaction:" + started["compaction_id"]):
        stage._cas_head(root, head, head["binding_id"], "unknown", "1" * 32)
    return started


def test_later_install_over_an_open_head_needs_a_release_mode_log_and_h_binding_equality(builds, t0_root, tmp_path,
                                                                                         monkeypatch):
    """CR2-3 and R3-OPS-2 (D4): admitted only under a pending release-mode log, with H's binding equal."""
    root = t0_root["root"]
    started = _open_k_under_a_pending_log(builds, root, tmp_path)
    witness = cw.witness(root)
    with monkeypatch.context() as patch:
        patch.setattr(launcher, "release_mode_log", lambda log: False)  # a carried plain-mode log
        _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"), "runtime_admission_not_closed")
    policy = t0_root["h"]["state"] / ".kza-target-ownership.json"
    policy.write_text("{}")  # H's binding, recomputed by this code, no longer equals the log's source binding
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"), launcher.CONTROLLER_BINDING_CHANGED)
    policy.unlink()
    _install(root, builds, "x5r2")
    assert _current(root) == builds["manifests"]["x5r2"]["git_commit"]
    assert stage.pending_compaction(root)["compaction_id"] == started["compaction_id"]
    events = stage.compaction_log_path(root, started["compaction_id"]) / "events"
    (events / "000099.json").write_text("{}")
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r3"), "runtime_admission_compaction_log_invalid")


def test_release_mode_log_reads_the_intent_mode():
    assert launcher.release_mode_log({"intent": {"mode": "release"}})
    assert not launcher.release_mode_log({"intent": {}})
    # SC-1b review RA-2: plain mode is only a log without 'mode'; any other value is an invalid log.
    with pytest.raises(ValueError, match="^runtime_admission_compaction_log_invalid$"):
        launcher.release_mode_log({"intent": {"mode": "plain"}})


# --- the wrong installer (R3-OPS-6b, R4-OPS-4) ----------------------------------------------------------

def _compiled(st: Path, root: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", "-B", str(st), "--root", str(root), "--offline", *arguments],
                          cwd=st.parents[1], capture_output=True, text=True, timeout=300)


def test_the_epoch4_installer_carries_a_plain_log_and_epoch5_finishes_it(builds, tmp_path):
    world = _world(builds, tmp_path)
    root = world["root"]
    with epoch4_in_process():
        started = cw.start(root, builds["archives"]["x4"])
    old_st = _extract(builds["archives"]["pre"], tmp_path / "pre-sc1a")  # the epoch-4 $ST, by mistake
    installed = _compiled(old_st, root, "install-controller", str(builds["archives"]["x5"]))
    assert installed.returncode == 0, installed.stderr  # no first_install checks in the epoch-4 code
    assert _record(root)["controller_epoch"] == 5
    assert stage.installed_controller_triple(root) == stage.controller_triple(builds["manifests"]["x5"])
    # The guards stay effective: a record refuses plan and serve.
    (root / "admission" / "delegations").mkdir(exist_ok=True)
    (root / "admission" / "delegations" / "x.json").write_text("{}")
    assert _serve(root, tmp_path, "guard").code == 78
    (root / "admission" / "delegations" / "x.json").unlink()
    # The carried plain-mode log is finished by the epoch-5 plain-mode commands.
    compaction.compact_verify(root, started["compaction_id"])
    assert _serve(root, tmp_path, "verify").code == 0
    assert compaction.compact_retire(root, started["compaction_id"], scan=cw.no_units)["phase"] == "complete"
    # The correct T0 is an equal-triple no-op, and a plan is admitted afterwards (R4-OPS-4). Since SC-1b
    # every new log is release mode: NEW is the epoch-5 X5 (equal triple), never the epoch-4 head's release.
    _install(root, builds)
    assert _record(root)["commit"] == builds["manifests"]["x5"]["git_commit"]
    assert cw.plan(root, builds["archives"]["x5"], "k2")["plan_sha256"]
    with pytest.raises(compaction.CompactionRefusal, match="runtime_admission_compaction_release_epoch"):
        cw.plan(root, builds["archives"]["x4"], "k2")


def test_an_epoch4_controller_cannot_lease_an_epoch5_root(builds, t0_root, tmp_path):
    root = t0_root["root"]
    old = _extract(builds["archives"]["pre"], tmp_path / "pre-sc1a")
    witness = cw.witness(root)
    result = _compiled(old, root, "activate", "a")
    assert result.returncode != 0 and "runtime_admission_controller_downgrade" in result.stderr, result.stderr
    with epoch4_in_process():
        with pytest.raises(ValueError, match="runtime_admission_controller_downgrade"):
            stage.AdmissionController(root).__enter__()
    assert cw.witness(root) == witness


def test_a_compiled_install_must_run_the_archives_own_code(builds, t0_root, tmp_path):
    """t0_controller_install.command: never from the installed $ST."""
    root = t0_root["root"]
    installed_st = root / "controller" / "current" / "scripts" / "stage_runtime.pyc"
    witness = cw.witness(root)
    refused = _compiled(installed_st, root, "install-controller", str(builds["archives"]["x5r2"]))
    assert refused.returncode == 78 and launcher.CONTROLLER_INSTALLER_MISMATCH in refused.stderr, refused.stderr
    assert cw.witness(root) == witness
    installed = _compiled(_extract(builds["archives"]["x5r2"], tmp_path / "x5r2"), root, "install-controller",
                          str(builds["archives"]["x5r2"]))
    assert installed.returncode == 0, installed.stderr
    assert json.loads(installed.stdout)["controller_revision"] == BASE_REV + 1


# --- AUTHZ3-1 after T0 (rows DL, DO; L4 stays reachable) -----------------------------------------------

def _l4(root, builds, tmp_path):
    unit_dir, manager = _systemctl(tmp_path)
    launcher.install_launcher(root, unit_dir=unit_dir, template=TEMPLATE, settings=_settings(root), manager=manager)
    legacy = builds["manifests"]["x4"]["git_commit"]
    stage.prepare_slot(builds["archives"]["x4"], root, "legacy", rehearsal.offline_source(tmp_path / "legacy-src"))
    return lambda: launcher.uninstall_launcher(root, unit_dir=unit_dir, manager=manager, decision_id="d-sc1a",
                                               phrase=launcher.UNINSTALL_PHRASE, legacy_slot="legacy",
                                               legacy_commits=(legacy[:7],))


def test_a_record_is_refused_before_serve_recovers_an_open_head(builds, t0_root, tmp_path):
    """Guard (1) in serve runs before R-b or the compaction gate write anything."""
    root = t0_root["root"]
    crashed = cw.serve(root, tmp_path, "crash", scenario="crash", seconds=0.1)
    assert crashed.code == 68 and stage.read_admission_head(root)["status"] == "unknown"
    record = root / "admission" / "delegations" / ("b" * 64 + ".json")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("{}")
    witness = cw.witness(root)
    served = _serve(root, tmp_path, "open")
    assert served.code == 78 and ("refused " + stage.DELEGATION_FROZEN) in served.serve.events
    assert cw.witness(root) == witness  # no recovery record, the head still unknown


def test_dl_a_record_after_t0_refuses_plan_and_serve_until_removed_and_l4_stays(builds, t0_root, tmp_path):
    root = t0_root["root"]
    uninstall = _l4(root, builds, tmp_path)
    record = root / "admission" / "delegations" / ("a" * 64 + ".json")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("{}")
    with pytest.raises(compaction.CompactionRefusal, match=stage.DELEGATION_FROZEN):
        cw.plan(root, builds["archives"]["x4"])
    served = _serve(root, tmp_path, "dl")
    assert served.code == 78 and ("refused " + stage.DELEGATION_FROZEN) in served.serve.events
    with pytest.raises(ValueError, match=stage.DELEGATION_FROZEN):
        stage.validate_target_ownership(root, stage.read_binding(root, stage.read_admission_head(root)["binding_id"]))
    record.unlink()  # a grant removed is never a privacy risk
    assert _serve(root, tmp_path, "dl-removed").code == 0
    record.write_text("{}")
    assert uninstall()["mismatches"] == []  # L4's uninstall is not refused by the guard


def test_do_owners_activated_after_t0_never_serve_again_and_l4_stays(builds, t0_root, tmp_path):
    """R4-OPS-5: an old copy delegated AND activated a delegated head after T0."""
    root = t0_root["root"]
    uninstall = _l4(root, builds, tmp_path)
    with epoch4_in_process(old_copy=True):
        cw.hop(builds["archives"]["x4"], root, t0_root["h"]["state"], "b")
    head = stage.read_binding(root, stage.read_admission_head(root)["binding_id"])
    assert head["slot"] == "b" and stage.delegated_target_ids(head)
    served = _serve(root, tmp_path, "do")
    assert served.code == 78 and ("refused " + stage.DELEGATION_FROZEN) in served.serve.events
    shutil.rmtree(root / "admission" / "delegations")
    served = _serve(root, tmp_path, "do-removed")  # removing the record does not restore serve
    assert served.code == 78 and ("refused " + stage.OWNERS_PRESENT) in served.serve.events
    with pytest.raises(compaction.CompactionRefusal, match=stage.COMPACTION_OWNERS_PRESENT):
        cw.plan(root, builds["archives"]["x4"])
    assert uninstall()["mismatches"] == []


def test_the_guard_is_stateless_and_epoch_keyed(tmp_path, monkeypatch):
    root = tmp_path
    stage.require_no_delegation_records(root)  # no admission/: nothing to refuse
    directory = root / "admission" / "delegations"
    directory.mkdir(parents=True)
    stage.require_no_delegation_records(root)  # an empty directory is the accepted empty set
    (directory / ".admission-tmp").write_text("")
    with pytest.raises(ValueError, match=stage.DELEGATION_FROZEN):
        stage.require_no_delegation_records(root)
    (directory / ".admission-tmp").unlink()
    directory.rmdir()
    directory.write_text("")  # another kind of entry in its place
    with pytest.raises(ValueError, match=stage.DELEGATION_FROZEN):
        stage.require_no_delegation_records(root)
    with pytest.raises(ValueError, match=stage.DELEGATION_FROZEN):
        list(stage._delegation_records(root, root))
    with pytest.raises(ValueError, match=stage.DELEGATION_FROZEN):
        stage._delegation_states(root)
    monkeypatch.setattr(stage, "CONTROLLER_EPOCH", 4)
    stage.require_no_delegation_records(root)
    assert not stage.delegations_frozen()


def test_owners_are_read_from_the_pinned_policy(tmp_path):
    state = tmp_path
    policy = {"schema_version": 1, "source_bindings": [], "owned_targets": [], "delegated_targets": ["t2", "t1"],
              "historical_targets": []}
    raw = json.dumps(policy).encode()
    (state / ".kza-target-ownership.json").write_bytes(raw)
    import hashlib
    binding = {"state_root": str(state), "ownership_sha256": hashlib.sha256(raw).hexdigest()}
    assert stage.delegated_target_ids(binding) == ["t1", "t2"]
    with pytest.raises(ValueError, match="^" + stage.OWNERS_PRESENT + "$"):
        stage.require_no_owners(binding)
    with pytest.raises(ValueError, match="^" + stage.COMPACTION_OWNERS_PRESENT + "$"):
        stage.require_no_owners(binding, stage.COMPACTION_OWNERS_PRESENT)
    with pytest.raises(ValueError, match="runtime_admission_ownership_changed"):
        stage.delegated_target_ids(dict(binding, ownership_sha256="0" * 64))
    assert stage.delegated_target_ids({"state_root": str(state), "ownership_sha256": None}) == []
    stage.require_no_owners({"state_root": str(state), "ownership_sha256": None})


# --- review round 1 ---------------------------------------------------------------------------------------

def test_later_install_refuses_a_delegation_record_and_owners(builds, t0_root, tmp_path):
    """Review F2 (design note N3, guard (3) at every install): an epoch-5 to epoch-5 install refuses too."""
    root = t0_root["root"]
    current = _current(root)
    record = root / "admission" / "delegations" / ("a" * 64 + ".json")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("{}")
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"), stage.DELEGATION_FROZEN)
    record.unlink()
    with epoch4_in_process(old_copy=True):  # an old copy delegates and activates after T0 (row DO)
        cw.hop(builds["archives"]["x4"], root, t0_root["h"]["state"], "b")
    shutil.move(root / "admission" / "delegations", tmp_path / "records-aside")
    head = stage.read_admission_head(root)
    assert head["status"] == "closed" and stage.delegated_target_ids(stage.read_binding(root, head["binding_id"]))
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"), stage.OWNERS_PRESENT)
    assert _current(root) == current


def test_a_missing_or_damaged_controller_json_keeps_the_triple_checks(builds, t0_root, tmp_path):
    """Review F3: the installed triple is read from controller/current whatever controller.json says."""
    root = t0_root["root"]
    record = root / "admission" / "controller.json"
    record.unlink()
    record.write_text("not json")
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5c"), launcher.CONTROLLER_REVISION_CONFLICT)
    record.unlink()  # the record is repaired by repeating the installed triple (first_install checks run)
    _install(root, builds, "x5")
    assert _record(root)["controller_epoch"] == 5 and _current(root) == builds["manifests"]["x5"]["git_commit"]
    _install(root, builds, "x5r3")
    record.unlink()
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5"), "runtime_admission_controller_downgrade")
    record.write_text("not json")
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"), "runtime_admission_controller_downgrade")
    link = root / "controller" / "current"
    link.unlink()
    link.symlink_to("missing")  # an unreadable current manifest is a damaged install, not 'no controller'
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r3"), "runtime_admission_controller_tampered")


def test_later_install_refuses_an_open_head_the_pending_log_does_not_own(builds, t0_root, tmp_path, monkeypatch):
    """Review F4: the open head must be the log's H or its authorized K."""
    root = t0_root["root"]
    started = _open_k_under_a_pending_log(builds, root, tmp_path)  # a release-mode log (SC-1b)
    gate = "compaction:" + started["compaction_id"]
    head = stage.read_admission_head(root)
    with stage.AdmissionController(root, allow_open_head=True, gate=gate):
        stage._cas_head(root, head, "e" * 64, "unknown", "2" * 32)
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"), "runtime_admission_not_closed")
    head = stage.read_admission_head(root)
    source = stage.pending_compaction(root)["intent"]["source_binding_id"]
    with stage.AdmissionController(root, allow_open_head=True, gate=gate):
        stage._cas_head(root, head, source, "unknown", "3" * 32)  # H's binding, open: the log's own head
    _install(root, builds, "x5r2")
    assert _current(root) == builds["manifests"]["x5r2"]["git_commit"]


def test_later_install_over_a_closed_head_refuses_a_plain_or_unreadable_log(builds, t0_root, tmp_path,
                                                                            monkeypatch):
    """Review r2 F2 (stricter reading): over a closed head only a release-mode log is carried."""
    root = t0_root["root"]
    started = cw.start(root, builds["archives"]["x5"])  # a release-mode log (SC-1b); K's head is closed
    assert stage.read_admission_head(root)["status"] == "closed"
    assert stage.pending_compaction(root)["intent"]["mode"] == "release"
    witness = cw.witness(root)
    with monkeypatch.context() as patch:
        patch.setattr(launcher, "release_mode_log", lambda log: False)  # a carried plain-mode log
        _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"),
                      "runtime_admission_compaction_pending")
    _install(root, builds, "x5r2")  # D4 / row CI: a release-mode log, closed head, SR-1
    assert _current(root) == builds["manifests"]["x5r2"]["git_commit"]
    events = stage.compaction_log_path(root, started["compaction_id"]) / "events"
    (events / "000099.json").write_text("{}")
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r3"), "runtime_admission_compaction_log_invalid")


def test_later_install_over_a_closed_k_head_needs_h_binding_equality(builds, t0_root, tmp_path):
    """SCR-C-1 (row CI, A0): over K's closed head under a pending release-mode log, SR-1 checks K and
    H's binding, recomputed by this code, must still equal the log's source binding (as for an open head)."""
    root = t0_root["root"]
    cw.start(root, builds["archives"]["x5"])  # a release-mode log (SC-1b); K's head is closed
    head = stage.read_admission_head(root)
    source = stage.pending_compaction(root)["intent"]["source_binding_id"]
    assert head["status"] == "closed" and head["binding_id"] != source
    policy = t0_root["h"]["state"] / ".kza-target-ownership.json"
    policy.write_text("{}")  # H's binding, recomputed by this code, no longer equals the log's source binding
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"), launcher.CONTROLLER_BINDING_CHANGED)
    policy.unlink()
    _install(root, builds, "x5r2")
    assert _current(root) == builds["manifests"]["x5r2"]["git_commit"]


def test_sr1_runs_serves_slot_check_on_the_head_slot(builds, tmp_path):
    """Review r2 F1 (CR-6, SR-1 step 4): a head slot that only serve's check_slot refuses."""
    world = _world(builds, tmp_path)
    root = world["root"]
    selection = root / "current"
    selection.unlink()
    selection.symlink_to("slots/b/runtime")  # a selection published for another slot
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds), "runtime_admission_launcher_selection_mismatch")
    served = _serve(root, tmp_path, "selection")
    assert served.code == 78


def test_a_compiled_installer_with_another_managed_launcher_is_refused(builds, t0_root, tmp_path):
    """Review F5: the installer identity covers managed_launcher.pyc, not only stage_runtime.pyc."""
    root = t0_root["root"]
    st = _extract(builds["archives"]["x5r2"], tmp_path / "x5r2")
    foreign = stage._read_archive(builds["archives"]["x5m"])[1]["scripts/managed_launcher.pyc"]
    assert foreign != (st.parent / "managed_launcher.pyc").read_bytes()
    (st.parent / "managed_launcher.pyc").write_bytes(foreign)
    witness = cw.witness(root)
    refused = _compiled(st, root, "install-controller", str(builds["archives"]["x5r2"]))
    assert refused.returncode == 78 and launcher.CONTROLLER_INSTALLER_MISMATCH in refused.stderr, refused.stderr
    assert cw.witness(root) == witness


def test_a_source_installer_must_run_the_archives_own_sources(builds, t0_root, tmp_path):
    """Review F6: a T0 from a checkout compares its closure sources with controller_sources."""
    root = t0_root["root"]
    witness = cw.witness(root)
    # This checkout is X5: X5R2 differs in stage_runtime.py, X5C only in owner_compaction.py.
    _unchanged_t0(root, witness, lambda: launcher.install_controller(root, builds["archives"]["x5r2"]),
                  launcher.CONTROLLER_INSTALLER_MISMATCH)
    _unchanged_t0(root, witness, lambda: launcher.install_controller(root, builds["archives"]["x5c"]),
                  launcher.CONTROLLER_INSTALLER_MISMATCH)
    command = subprocess.run([sys.executable, "-B", str(cw.REPO / "scripts" / "stage_runtime.py"), "--root",
                              str(root), "--offline", "install-controller", str(builds["archives"]["x5r2"])],
                             cwd=cw.REPO, capture_output=True, text=True, timeout=300)
    assert command.returncode == 78 and launcher.CONTROLLER_INSTALLER_MISMATCH in command.stderr, command.stderr
    assert cw.witness(root) == witness


def test_kill_between_the_switch_and_controller_json_of_a_later_install_then_repeat(builds, t0_root, tmp_path):
    """Review r2 F7: row T0S for an epoch-5 to epoch-5 install."""
    root = t0_root["root"]
    record = _record(root)

    def die(step):
        if step == "switched":
            raise KeyboardInterrupt("synthetic kill")
    with pytest.raises(KeyboardInterrupt):
        _install(root, builds, "x5r2", step_hook=die)
    assert _record(root) == record and _current(root) == builds["manifests"]["x5r2"]["git_commit"]
    served = _serve(root, tmp_path, "t0s")
    assert served.code == 78 and any(stage.CONTROLLER_INSTALL_INCOMPLETE in event for event in served.serve.events)
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5"), "runtime_admission_controller_downgrade")
    _install(root, builds, "x5r2")  # the repeat: an equal triple read from current
    assert _record(root)["commit"] == builds["manifests"]["x5r2"]["git_commit"]
    assert _serve(root, tmp_path, "after").code == 0


def test_the_epoch4_installer_over_a_pending_owner_compaction_leaves_no_tool_exit(builds, tmp_path):
    """Review F1: R3-OPS-6b's 'plain log finishes' holds only without owners; pinned as a terminal stop.

    SCR-C-2 (SCRF-1): every abort runs the AUTHZ3-1 guard before any event,
    so each one refuses delegation_frozen first and writes nothing; the stop
    stays terminal because the epoch-4 records remain.
    """
    world = _world(builds, tmp_path, hops=("b",))
    root = world["root"]
    with epoch4_in_process():
        uninstall = _l4(root, builds, tmp_path)
        started = cw.start(root, builds["archives"]["x4"])
    old_st = _extract(builds["archives"]["pre"], tmp_path / "pre-sc1a")  # the epoch-4 $ST, by mistake
    installed = _compiled(old_st, root, "install-controller", str(builds["archives"]["x5"]))
    assert installed.returncode == 0, installed.stderr
    compaction_id = started["compaction_id"]
    assert compaction.compact_verify(root, compaction_id)["armed"] is True  # it only arms
    served = _serve(root, tmp_path, "owner-log")
    assert served.code == 78 and ("refused " + stage.DELEGATION_FROZEN) in served.serve.events
    # Before SCR-C-2 abort --to-h refused abort_to_h_before_run and the plain abort logged abort_started
    # before its return to H refused; now the guard refuses first, in both modes, and logs nothing.
    witness = cw.witness(root)
    for to_h in (True, False, False, True):
        with pytest.raises(compaction.CompactionRefusal, match=stage.DELEGATION_FROZEN):
            compaction.compact_abort(root, compaction_id, to_h=to_h)
    assert cw.witness(root) == witness
    assert "abort_started" not in [event["kind"] for event in stage.pending_compaction(root)["events"]]
    with pytest.raises(ValueError, match="runtime_admission_compaction_pending"):
        uninstall()  # L4 takes gate 'normal'
    witness = cw.witness(root)
    _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"), stage.DELEGATION_FROZEN)
    assert stage.pending_compaction(root)["compaction_id"] == compaction_id


def test_sr1_recomputes_the_binding_with_the_staged_code(builds, tmp_path):
    """Review r2 F6: a staged controller whose projection output differs is refused at first and later T0."""
    world = _world(builds, tmp_path)
    root = world["root"]
    original = stage.capture_state_authority

    def changed(state_root):
        return dict(original(state_root), sc1a_projection="changed")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stage, "capture_state_authority", changed)
        witness = cw.witness(root)
        _unchanged_t0(root, witness, lambda: _install(root, builds), launcher.CONTROLLER_BINDING_CHANGED)
    _install(root, builds)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stage, "capture_state_authority", changed)
        witness = cw.witness(root)
        _unchanged_t0(root, witness, lambda: _install(root, builds, "x5r2"), launcher.CONTROLLER_BINDING_CHANGED)
    assert _current(root) == builds["manifests"]["x5"]["git_commit"]


@pytest.mark.parametrize(("archive_name", "schema_contract"), [
    pytest.param("r1", 1, id="historical-r1"),
    pytest.param("x4", 2, id="adopted-epoch-2"),
])
def test_first_t0_respects_the_head_schema_contract(builds, tmp_path, archive_name, schema_contract):
    """Q4: production adopts directly from e9669ec into schema contract 2."""
    assert builds["manifests"]["r1"]["git_commit"].startswith(R1_COMMIT)
    assert builds["manifests"][archive_name]["schema_contract"] == schema_contract
    assert builds["manifests"]["x5"]["schema_contract"] == stage.SCHEMA_CONTRACT_EPOCH == 2
    with epoch4_in_process():
        world = cw.world(builds["archives"][archive_name], tmp_path, hops=())
    root = world["root"]
    head = stage.read_admission_head(root)
    stored = stage.read_binding(root, head["binding_id"])
    runtime = root / "slots" / "a" / "runtime"
    environment = {key: os.environ[key] for key in stage._PROBE_ENVIRONMENT if key in os.environ}
    probe = subprocess.run([sys.executable, "-I", "-B", "-c", stage._PROBE_SOURCE,
                            str(runtime / "scripts" / "stage_runtime.pyc"), str(root), "a"],
                           cwd=runtime, env=environment, capture_output=True, text=True, timeout=300)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    slot_binding = json.loads(probe.stdout)["binding"]
    assert stage._canonical_json(slot_binding) == stage._canonical_json(stored)
    assert stage._canonical_json(stage._binding_snapshot(root, "a")) == stage._canonical_json(stored)
    if schema_contract != stage.SCHEMA_CONTRACT_EPOCH:
        _unchanged_t0(root, cw.witness(root), lambda: _install(root, builds), launcher.SCHEMA_CONTRACT_CHANGED)
        return
    _install(root, builds)
    assert _record(root)["controller_epoch"] == 5
    assert _serve(root, tmp_path, "epoch-2-under-controller-5").code == 0


def test_the_delegated_observation_reader_runs_the_guard_first(tmp_path):
    """Review F7 (defense in depth): output registration never reads a delegation record at epoch 5."""
    from src.memory import output_registration_v2 as registration
    record = tmp_path / "admission" / "delegations" / ("a" * 64 + ".json")
    record.parent.mkdir(parents=True)
    record.write_text("{}")
    with pytest.raises(ValueError, match="^" + stage.DELEGATION_FROZEN + "$"):
        registration._read_delegated_observation(stage, tmp_path, {"state_root": str(tmp_path)}, {"scope": {}}, b"{}")
