"""SC-6a: the pre-R1 rehearsal additions on a production-shaped adopted root (in process, any OS).

sc-release-design v3 epochs.r1_rebuild step 4 and implementation_units SC-6a,
run on the U3b-0e3 world (the b0866e8-written legacy root adopted into R1, the
R1 archive built from this working tree at controller epoch 4, R1's epoch, since SC-1a):

- (i) after the adoption, ``delegate``, ``authorize_derived_state`` and a
  cross-slot ``activate`` are refused, by the library and by the installed
  compiled controller (``controller/current``), and nothing under the root
  changes;
- (ii) ``install-controller`` with an R1 archive built before SC-2 (epoch 3,
  here the pinned 26de262) is refused on the fresh root (W3) and on the
  adopted epoch-4 root, and R1's own prepare refuses it;
- (iii) ``read_r1_archive`` refuses an epoch-5 and an epoch-3 archive (M-6a.1:
  ``>=`` instead of ``==``);
- (iv) the runbook's pinned-sha step, run as written, stops on a stale
  epoch-3 archive and on a rebuilt-but-different archive of the same commit
  (M-6a.2: comparing the commit instead of the archive sha256); right after
  it, the extraction step, run as written, extracts that same archive into a
  new ``mktemp -d`` directory (never a ``$R1`` the shell already had) and
  defines ``$ST`` only when the extracted ``release.json`` names the pinned
  commit (review SC6A-R2).

AUTHZ3-1: b0c322d~1 predates the controller epoch check. On the adopted
root with schema contract 2, it prepares an unbound copy. Delegation
refuses on the source schema and leaves the closed head unchanged.
The first copy with ``require_forward_epoch`` (b0c322d) still refuses
before taking the lease, which pins the range's end.

The same additions run compiled under real systemd in
test_linux_adoption_window.py.
"""
from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

from scripts import adoption_commands as commands
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from tests.mocks.runtime_artifact_cache import cached_build_runtime as build_runtime
from tests.integration import test_adoption_commands as adoption_tests
from tests.mocks import compaction_world as cw
from tests.mocks import launcher_rehearsal
from tests.mocks import r1_runbook
from tests.mocks.epoch4 import apply_epoch4, epoch4_commit
from tests.unit.tools.test_stage_runtime_admission import _witness

REPO = Path(__file__).resolve().parents[2]
CONTAINER = adoption_tests.CONTAINER
EPOCH3_R1 = "26de262"  # the last epoch-3 commit: an R1 built before SC-2
EARLY_COPY = "b0c322d~1"  # the last stage_runtime without require_forward_epoch (AUTHZ3-1 range end)
FIRST_LOCKED = "b0c322d"  # the first one with it
RETIRED = "runtime_admission_delegation_retired"
RELEASE_MODE = "runtime_admission_transition_requires_release_mode"



@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """R1's stand-in: this working tree built at controller epoch 4 (R1 = epoch 4), plus an epoch-1 copy.

    SC-1a raised this tree to epoch 5; R1 stays epoch 4 (the pinned 357dcb5, rebuilt below in
    test_the_runbook_pin_is_the_archive_of_its_commit). The in-process controller runs as the same
    epoch-4 build (the autouse ``r1_epoch`` fixture), so every statement here is about an epoch-4 root.
    """
    base = tmp_path_factory.mktemp("r1-artifact-epoch4")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    revision = epoch4_commit(clone, launcher_rehearsal.overlay_commit(REPO, clone))
    archive = base / "r1.tar.gz"
    manifest = build_runtime(clone, archive, revision)
    assert manifest["controller_epoch"] == 4
    old = base / "epoch1.tar.gz"
    adoption_tests._rewrite_release(archive, old, controller_epoch=1)
    return {"archive": archive, "epoch1": old, "commit": manifest["git_commit"]}


@pytest.fixture(autouse=True)
def r1_epoch(monkeypatch):
    apply_epoch4(monkeypatch)

tree = adoption_tests.tree  # the b0866e8 tree (module scope)
world = adoption_tests.world


def _build(base: Path, revision: str, name: str) -> tuple[Path, dict]:
    if subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"], cwd=REPO).returncode != 0:
        pytest.fail(revision + " is not available in this clone")
    clone = base / "clone"
    if not clone.exists():
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                       check=True, timeout=120)
    archive = base / (name + ".tar.gz")
    return archive, build_runtime(clone, archive, revision)


@pytest.fixture(scope="module")
def old_builds(tmp_path_factory):
    """The epoch-3 R1 (26de262), the early copy (b0c322d~1) and the first locked copy (b0c322d), extracted."""
    base = tmp_path_factory.mktemp("sc6a-old")
    found = {}
    for key, revision in (("epoch3", EPOCH3_R1), ("early", EARLY_COPY), ("locked", FIRST_LOCKED)):
        archive, manifest = _build(base, revision, key)
        assert manifest["controller_epoch"] == 3 and "schema_contract" not in manifest, (revision, manifest)
        extracted = base / (key + "-tree")
        with tarfile.open(archive) as bundle:
            bundle.extractall(extracted, filter="data")
        code = (extracted / "scripts" / "stage_runtime.pyc").read_bytes()
        assert b"delegate_slot" in code
        assert (b"require_forward_epoch" in code) is (key != "early"), key
        found[key] = {"archive": archive, "manifest": manifest, "tree": extracted}
    return found


def _adopted(world) -> dict:
    """W2..W4 in process: prepare, install the epoch-4 controller, commit. The head is closed on r1."""
    adopted = world.adopt()
    head = world.head()
    assert head["status"] == "closed" and stage.read_binding(world.root, head["binding_id"])["slot"] == "r1"
    assert json.loads((world.root / "admission" / "controller.json").read_text())["controller_epoch"] == 4
    return adopted


def _after_w9(world) -> str:
    """W2..W4, then one managed run of r1 that closes cleanly (W8/W9 in process: a receipt and a
    closure proof, the head closed again on r1). Returns the closed binding."""
    _adopted(world)
    cw.install_record(world.root)
    binding = cw.run(world.root, "r1")
    assert world.head()["generation"] > 1
    assert stage.read_binding(world.root, binding)["slot"] == "r1"
    assert list((world.root / "admission" / "closures").iterdir())
    return binding


def _cli(pyc: Path, root: Path, *arguments: str) -> subprocess.CompletedProcess:
    """A compiled stage_runtime as the runbook runs it: python -I -B <pyc> --root R --offline ..."""
    return subprocess.run([sys.executable, "-I", "-B", str(pyc), "--root", str(root), "--offline", *arguments],
                          cwd=pyc.parents[1], capture_output=True, text=True, timeout=600)


def _controller(world) -> Path:
    return world.root / "controller" / "current" / "scripts" / "stage_runtime.pyc"


def _states(world) -> tuple[Path, ...]:
    return (world.root / "slots" / "r1" / "state", world.layout.legacy_state)


# --- (i) after W9 every grant and a cross-slot activate refuse ---------------------------------

def test_after_w9_every_grant_and_a_cross_slot_activate_refuse(world):
    _after_w9(world)
    root, head = world.root, world.head()
    before = _witness(root, *_states(world))
    with pytest.raises(ValueError, match="^" + RETIRED + "$"):
        stage.delegate_slot(root, CONTAINER)
    with stage.AdmissionController(root) as controller:
        with pytest.raises(ValueError, match="^" + RETIRED + "$"):
            controller.authorize_derived_state(CONTAINER, head["binding_id"], target_relocations={})
        with pytest.raises(ValueError, match="^" + RELEASE_MODE + "$"):
            controller.activate(CONTAINER, managed=True)
    with pytest.raises(ValueError, match="^" + RELEASE_MODE + "$"):
        stage.activate_slot(root, CONTAINER)
    assert _witness(root, *_states(world)) == before
    # The installed compiled controller, as an operator would run it.
    delegated = _cli(_controller(world), root, "delegate", CONTAINER)
    assert delegated.returncode == 78 and json.loads(delegated.stdout) == {"refused": RETIRED}, delegated
    activated = _cli(_controller(world), root, "activate", CONTAINER)
    assert activated.returncode != 0 and RELEASE_MODE in activated.stderr, activated.stderr[-2000:]
    assert _witness(root, *_states(world)) == before
    assert world.head() == head and os.readlink(root / "current") == "slots/r1/runtime"
    assert not (root / "admission" / "delegations").exists()
    assert not (root / "admission" / "derived-authority").exists()


# --- (ii) an epoch-3 R1 archive is refused by install-controller and by R1's prepare -----------------------

def test_an_epoch3_r1_archive_is_refused_before_and_after_w3(world, old_builds):
    epoch3 = old_builds["epoch3"]["archive"]
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_archive_epoch$"):
        world.prepare(archive=epoch3)
    assert not (world.root / "admission").exists() and not (world.root / "slots" / "r1").exists()
    with pytest.raises(launcher.LauncherRefusal, match="^runtime_admission_controller_epoch_too_old$"):
        launcher.install_controller(world.root, epoch3)  # W3 on the fresh root
    assert not (world.root / "controller").exists() and not (world.root / "admission").exists()
    _adopted(world)
    record = (world.root / "admission" / "controller.json").read_bytes()
    controllers = sorted(os.listdir(world.root / "controller"))
    with pytest.raises(launcher.LauncherRefusal, match="^runtime_admission_controller_epoch_too_old$"):
        launcher.install_controller(world.root, epoch3)
    installed = _cli(_controller(world), world.root, "install-controller", str(epoch3))
    assert installed.returncode == 78, installed
    assert json.loads(installed.stderr.splitlines()[-1])["refused"] == "runtime_admission_controller_epoch_too_old"
    assert (world.root / "admission" / "controller.json").read_bytes() == record
    assert sorted(os.listdir(world.root / "controller")) == controllers


# --- (iii) read_r1_archive refuses an epoch-5 and an epoch-3 archive (M-6a.1) -----------------------------

def test_read_r1_archive_accepts_only_the_running_epoch(world, artifacts, old_builds, tmp_path):
    epoch5 = tmp_path / "epoch5.tar.gz"
    adoption_tests._rewrite_release(artifacts["archive"], epoch5, controller_epoch=stage.CONTROLLER_EPOCH + 1)
    for archive in (epoch5, old_builds["epoch3"]["archive"]):
        with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_archive_epoch$"):
            commands.read_r1_archive(archive)
    _manifest, _files, summary = commands.read_r1_archive(artifacts["archive"])
    assert summary["controller_epoch"] == stage.CONTROLLER_EPOCH == 4
    # The operator's commands refuse both before writing anything: the dry run and prepare.
    for archive in (epoch5, old_builds["epoch3"]["archive"]):
        with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_archive_epoch$"):
            world.dry_run(archive=archive)
        with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_archive_epoch$"):
            world.prepare(archive=archive)
    assert not (world.root / "admission").exists() and not (world.root / "slots" / "r1").exists()
    assert os.listdir(world.scratch) == []


# --- (iv) the runbook's pinned-sha step stops on a stale or rebuilt archive (M-6a.2) ----------------------

def test_the_runbook_check_stops_on_a_stale_epoch3_archive_and_on_a_rebuilt_one(artifacts, old_builds, tmp_path):
    r1 = artifacts["archive"]
    r1_sha = hashlib.sha256(r1.read_bytes()).hexdigest()
    r1_commit = artifacts["commit"]
    stale = old_builds["epoch3"]["archive"]
    # The runbook's own pin, against the stale epoch-3 archive: stop.
    result = r1_runbook.run_check(stale)
    assert result.returncode != 0 and result.stdout.strip() == r1_runbook.STOP, result
    # Pinned to this rehearsal's R1 (the runbook pins the real one): the R1 passes, the stale one stops.
    passed = r1_runbook.run_check(r1, sha256=r1_sha, commit=r1_commit)
    assert passed.returncode == 0 and passed.stdout.strip() == r1_runbook.OK, passed
    result = r1_runbook.run_check(stale, sha256=r1_sha, commit=r1_commit)
    assert result.returncode != 0 and result.stdout.strip() == r1_runbook.STOP, result
    # A rebuilt-but-different archive of the same commit (same release.json, same files): stop.
    rebuilt = r1_runbook.regzipped(r1, tmp_path / "rebuilt.tar.gz")
    with tarfile.open(rebuilt) as bundle:
        assert json.load(bundle.extractfile("release.json"))["git_commit"] == r1_commit
    result = r1_runbook.run_check(rebuilt, sha256=r1_sha, commit=r1_commit)
    assert result.returncode != 0 and result.stdout.strip() == r1_runbook.STOP, result
    missing = r1_runbook.run_check(tmp_path / "absent.tar.gz", sha256=r1_sha, commit=r1_commit)
    assert missing.returncode != 0 and r1_runbook.STOP in missing.stdout, missing


def test_the_extraction_step_uses_a_new_directory_and_only_the_verified_archive(artifacts, tmp_path):
    """SC6A-R2: $R1 and $ST come from the archive the check just passed, in a new directory."""
    r1 = artifacts["archive"]
    r1_sha = hashlib.sha256(r1.read_bytes()).hexdigest()
    stale = tmp_path / "stale-r1"
    (stale / "scripts").mkdir(parents=True)
    (stale / "scripts" / "stage_runtime.pyc").write_bytes(b"stale")
    shell = {"R1": str(stale), "ST": "stale-st"}  # a previous extraction left in the operator's shell
    for name, commit in (("pinned", artifacts["commit"]), ("other", "0" * 40)):
        temporary = tmp_path / name
        temporary.mkdir()
        # GNU mktemp (the server) honours TMPDIR; macOS mktemp -d uses the user's temporary directory
        # (DARWIN_USER_TEMP_DIR) even when TMPDIR names another one (re-review RR-4). A probe mktemp -d,
        # in the environment the helper gives the runbook's blocks, names the directory it will use.
        probe = Path(subprocess.run(["bash", "-c", "mktemp -d"], capture_output=True, text=True, check=True,
                                    timeout=30, env=dict(os.environ, TMPDIR=str(temporary))).stdout.strip())
        probe.rmdir()
        parents = {probe.resolve().parent}
        existing = {parent: set(os.listdir(parent)) for parent in parents}
        result, final = r1_runbook.run_check_and_extract(r1, python=sys.executable, tmpdir=temporary,
                                                         sha256=r1_sha, commit=commit, environment=shell)
        extracted = Path(final["R1"]) if final.get("R1") else None
        try:
            assert extracted is not None and extracted != stale, (result, final)
            parent = extracted.resolve().parent
            assert parent in parents and extracted.name not in existing[parent]  # created now, never reused
            assert (extracted.stat().st_mode & 0o777) == 0o700
            if name == "pinned":
                assert result.returncode == 0, result
                assert result.stdout.splitlines()[:2] == [r1_runbook.OK, r1_runbook.EXTRACTED], result.stdout
                _assert_extraction_of(r1, extracted, final["ST"])
            else:  # the archive passes the check but names another commit: stop, and $ST stays unset
                assert result.returncode == 1, result
                assert result.stdout.splitlines()[:2] == [r1_runbook.OK, r1_runbook.EXTRACT_STOP], result.stdout
                assert final["ST"] == ""
        finally:
            if extracted is not None and extracted.is_dir() and not extracted.resolve().is_relative_to(
                    tmp_path.resolve()):
                shutil.rmtree(extracted)  # macOS: outside pytest's tmp


def _assert_extraction_of(r1: Path, extracted: Path, st: str) -> None:
    """The new directory holds exactly the archive's files, and $ST runs its stage_runtime."""
    with tarfile.open(r1) as bundle:
        members = sorted(member.name for member in bundle.getmembers() if member.isfile())
        for name in ("release.json", "scripts/stage_runtime.pyc"):
            assert (extracted / name).read_bytes() == bundle.extractfile(name).read(), name
    assert sorted(str(path.relative_to(extracted)) for path in extracted.rglob("*") if path.is_file()) == members
    assert st == (sys.executable + " -I -B " + str(extracted / "scripts" / "stage_runtime.pyc")
                  + " --root /home/kza/releases --offline")


# --- AUTHZ3-1: early stage_runtime on the adopted root with schema contract 2 --------------------

def assert_authz3_1_schema_refusal(
    root: Path, state: Path, archive: Path, command: Callable[..., subprocess.CompletedProcess[str]],
) -> dict:
    """Pin the historical controller's refusal and the unbound copy it leaves."""
    head = stage.read_admission_head(root)
    assert head["status"] == "closed"
    binding = stage.read_binding(root, head["binding_id"])
    assert binding["slot"] == "r1" and Path(binding["state_root"]) == state
    assert stage.release_integers(binding)["schema_contract"] == 2
    tables = binding["authority"]["projection"]["tables"]["data/reminders.db"]["schema_contract_v1"]["columns"]
    assert {"timers", "alert_preferences"} <= tables.keys()
    assert tables["reminders"]["delivery_room"] == "TEXT"

    slot = root / "slots" / "p"
    prepared_state = slot / "state"
    slots_before = set(slot.parent.iterdir())
    assert slot not in slots_before
    prepared = command("prepare", str(archive), "p", "--state-source", str(state))
    assert prepared.returncode == 0, prepared.stderr[-3000:]
    assert set(slot.parent.iterdir()) == slots_before | {slot}
    assert {path.name for path in slot.iterdir()} == {"runtime", "state"}
    [record_path] = stage.prepared_directory(root).glob("p.*.json")
    record = json.loads(record_path.read_text())
    assert record["kind"] == "prepared" and record["slot"] == "p"
    assert Path(record["state_root"]) == prepared_state
    assert Path(record["runtime_root"]) == slot / "runtime"
    assert not os.path.lexists(prepared_state / stage._MARKER)
    assert not os.path.lexists(prepared_state / ".kza-target-ownership.json")

    before = _witness(root, state, prepared_state)
    prepared_tree = cw.tree(slot)
    delegated = command("delegate", "p")
    refusal = "runtime_admission_unknown_schema"
    assert delegated.returncode != 0 and delegated.stderr.strip().endswith("ValueError: " + refusal), (
        delegated.stderr[-3000:])
    delegations = root / "admission" / "delegations"
    assert not os.path.lexists(delegations) or not list(delegations.iterdir())
    assert stage.read_admission_head(root) == head
    assert os.readlink(root / "current") == "slots/r1/runtime"
    assert _witness(root, state, prepared_state) == before
    assert set(slot.parent.iterdir()) == slots_before | {slot} and cw.tree(slot) == prepared_tree
    return {"copy": EARLY_COPY, "prepare_exit": prepared.returncode, "delegate_exit": delegated.returncode,
            "delegate_refusal": refusal, "head_slot_after": binding["slot"], "prepared_slot_after": slot.name}


def test_a_copy_without_the_epoch_check_refuses_to_delegate_on_schema_epoch_2(world, old_builds):
    """The pre-lockout copy reaches the schema guard and leaves r1 selected."""
    _after_w9(world)
    root, early = world.root, old_builds["early"]
    old_pyc = early["tree"] / "scripts" / "stage_runtime.pyc"
    assert_authz3_1_schema_refusal(root, root / "slots" / "r1" / "state", early["archive"],
                                  lambda *args: _cli(old_pyc, root, *args))


def test_the_first_copy_with_the_epoch_check_is_locked_out_of_the_epoch4_root(world, old_builds):
    """The range's end: b0c322d refuses on the same root before the lease (runtime_admission_controller_downgrade)."""
    _after_w9(world)
    root, locked = world.root, old_builds["locked"]
    old_pyc = locked["tree"] / "scripts" / "stage_runtime.pyc"
    before = _witness(root, *_states(world))
    for arguments in (("prepare", str(locked["archive"]), "p", "--state-source", str(root / "slots" / "r1" / "state")),
                      ("delegate", "r1"), ("activate", CONTAINER)):
        result = _cli(old_pyc, root, *arguments)
        assert result.returncode != 0 and "runtime_admission_controller_downgrade" in result.stderr, (
            arguments, result.stderr[-2000:])
    assert _witness(root, *_states(world)) == before
    assert not (root / "slots" / "p").exists()


# --- the runbook's pin is the archive build_runtime makes from the pinned commit -----------------------------

def test_the_runbook_pin_is_the_archive_of_its_commit(tmp_path):
    """Rebuilt here from the pinned commit, the archive has exactly the pinned sha256 and passes the
    runbook's own block (build_runtime is byte-reproducible across the Mac and the server)."""
    pin = r1_runbook.pinned()
    archive, manifest = _build(tmp_path, pin["R1_COMMIT"], "pinned")
    assert manifest["git_commit"] == pin["R1_COMMIT"]
    assert manifest["controller_epoch"] == 4 and manifest["schema_contract"] == 1
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == pin["R1_SHA256"]
    result = r1_runbook.run_check(archive)
    assert result.returncode == 0 and result.stdout.strip() == r1_runbook.OK, result
    _manifest, _files, summary = commands.read_r1_archive(archive)
    assert summary["sha256"] == pin["R1_SHA256"]
