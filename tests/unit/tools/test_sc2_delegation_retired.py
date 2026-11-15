"""SC-2: delegation retired and cross-slot managed activation refused (sc-release-design v3).

delegation_off.refusal_tests: every grant surface refuses with
runtime_admission_delegation_retired before any read, lock, lease or write;
the CLI exits 78 with the code; activating another slot under a managed head
refuses with runtime_admission_transition_requires_release_mode before the
candidate is read, while the head's own slot and compaction's K (under its own
gate) stay allowed; an epoch-3 controller copy cannot take the lease on an
epoch-4 root; an existing owner chain still validates, serves and compacts.
No test here uses the legacy_managed_transitions fixture except to build the
owner chain whose later handling runs with the production flag.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import compaction
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from scripts.build_runtime import build_runtime
from tests.mocks import compaction_world as cw
from tests.unit.tools import test_stage_runtime_admission as admission
from tests.unit.tools.test_stage_runtime_admission import (
    _CLOSING_CHILD,
    _closed_head,
    _command,
    _copy_candidate,
    _refused,
    _registered_targets,
    _witness,
)

REPO = Path(__file__).resolve().parents[3]
# The last epoch-3 commit (sc-release-design v3 checkout): owners of compiled legacy roots are built with it.
PINNED_EPOCH3 = "26de262"
RETIRED = "^runtime_admission_delegation_retired$"
RELEASE_MODE = "^runtime_admission_transition_requires_release_mode$"

admission_archive = admission.admission_archive  # The module-scoped built HEAD artifact fixture.


def test_production_defaults():
    """M-2.6, M-2.7: the flag is off and the controller epoch is 4."""
    assert stage.LEGACY_MANAGED_TRANSITIONS is False
    assert stage.CONTROLLER_EPOCH == 4
    assert stage.DELEGATION_RETIRED == "runtime_admission_delegation_retired"
    assert stage.TRANSITION_REQUIRES_RELEASE_MODE == "runtime_admission_transition_requires_release_mode"


@pytest.fixture
def pair(admission_archive, tmp_path):
    """A real closed head 'old' with a path target and a path-less one, and an undelegated copy."""
    registered = []

    def statements(state):
        rows = _registered_targets(state)
        registered.extend(row[2][0] for row in rows[1:])
        return rows

    root, state = _closed_head(admission_archive, tmp_path, statements)
    candidate = _copy_candidate(admission_archive, root, state)
    head = stage.read_admission_head(root)
    return {"root": root, "state": state, "candidate": candidate, "head": head, "local": registered[0]}


def _no_delegation_records(root: Path, candidate: Path) -> None:
    for name in ("delegations", "delegation-chains", "derived-authority"):
        assert not (root / "admission" / name).exists(), name
    assert not (candidate / ".kza-target-ownership.json").exists()


def _boom(*_args, **_kwargs):
    raise AssertionError("reached past the SC-2 guard")


# --- every grant surface ---------------------------------------------------------------------------

def test_private_grant_refuses_before_any_read(pair, monkeypatch):
    """M-2.1: _authorize_target_delegation's first statement is the guard."""
    root, candidate = pair["root"], pair["candidate"]
    with stage.AdmissionController(root):
        with monkeypatch.context() as patch:
            patch.setattr(stage, "read_admission_head", _boom)
            patch.setattr(stage, "verify_closed_binding", _boom)
            _refused(root, (pair["state"], candidate),
                     lambda: stage._authorize_target_delegation(root, candidate, pair["head"]["binding_id"],
                                                                [pair["local"]]), RETIRED)
    _no_delegation_records(root, candidate)


@pytest.mark.parametrize("surface", ["delegate", "authorize_target_delegation", "authorize_derived_state"])
def test_controller_grants_refuse_before_the_root_lock(pair, monkeypatch, surface):
    """M-2.2, M-2.8 and the method guard: refused after the lease check, before _locked_root."""
    root, candidate, head = pair["root"], pair["candidate"], pair["head"]
    calls = {"delegate": lambda controller: controller.delegate("candidate"),
             "authorize_target_delegation": lambda controller: controller.authorize_target_delegation(
                 candidate, head["binding_id"], [pair["local"]]),
             "authorize_derived_state": lambda controller: controller.authorize_derived_state(
                 "candidate", head["binding_id"], target_relocations={})}
    with stage.AdmissionController(root) as controller:
        with monkeypatch.context() as patch:
            patch.setattr(stage, "_locked_root", _boom)
            _refused(root, (pair["state"], candidate), lambda: calls[surface](controller), RETIRED)
        # And without the injection: the same refusal, nothing written.
        _refused(root, (pair["state"], candidate), lambda: calls[surface](controller), RETIRED)
    _no_delegation_records(root, candidate)
    assert stage.read_admission_head(root) == head
    # Unentered: the lease check still comes first.
    with pytest.raises(ValueError, match="^runtime_admission_controller_not_entered$"):
        calls[surface](stage.AdmissionController(root))


@pytest.mark.parametrize("surface", ["delegate_slot", "authorize_target_delegation"])
def test_module_grants_refuse_before_the_lease(pair, surface):
    """Held lease elsewhere: a function that took the lease would be runtime_admission_busy."""
    root, candidate, head = pair["root"], pair["candidate"], pair["head"]
    calls = {"delegate_slot": lambda: stage.delegate_slot(root, "candidate"),
             "authorize_target_delegation": lambda: stage.authorize_target_delegation(
                 root, candidate, head["binding_id"], [pair["local"]])}
    with stage.AdmissionController(root):
        _refused(root, (pair["state"], candidate), calls[surface], RETIRED)
    _refused(root, (pair["state"], candidate), calls[surface], RETIRED)
    _no_delegation_records(root, candidate)


def test_cli_delegate_exits_78_with_the_code_and_takes_no_lease(pair, tmp_path):
    """M-2.3: the parser entry stays so the operator gets the code, not an argparse error."""
    root, candidate = pair["root"], pair["candidate"]
    before = _witness(root, pair["state"], candidate)
    lock = (root / ".admission.lock").read_bytes()
    command = [sys.executable, "-B", str(REPO / "scripts" / "stage_runtime.py"), "--offline"]
    result = subprocess.run([*command, "--root", str(root), "delegate", "candidate"], cwd=REPO,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 78, result.stderr
    assert json.loads(result.stdout) == {"refused": "runtime_admission_delegation_retired"}
    assert _witness(root, pair["state"], candidate) == before
    assert (root / ".admission.lock").read_bytes() == lock  # the lease header was not rewritten
    _no_delegation_records(root, candidate)
    absent = tmp_path / "no-root"
    result = subprocess.run([*command, "--root", str(absent), "delegate", "x"], cwd=REPO,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 78 and not absent.exists()


# --- activation under a managed head ---------------------------------------------------------------

@pytest.mark.parametrize("managed", [False, True])
def test_activate_of_another_slot_refuses_before_reading_it(pair, monkeypatch, managed):
    """M-2.4: no binding stored, no marker, no head CAS, no selection change, candidate never read."""
    root, candidate = pair["root"], pair["candidate"]
    current = os.readlink(root / "current")
    with stage.AdmissionController(root) as controller:
        with monkeypatch.context() as patch:
            patch.setattr(stage, "_binding_snapshot", _boom)
            patch.setattr(stage, "_probe_candidate", _boom)
            _refused(root, (pair["state"], candidate), lambda: controller.activate("candidate", managed=managed),
                     RELEASE_MODE)
    _refused(root, (pair["state"], candidate), lambda: stage.activate_slot(root, "candidate"), RELEASE_MODE)
    assert stage.read_admission_head(root) == pair["head"] and os.readlink(root / "current") == current
    slots = {json.loads(path.read_text())["slot"] for path in (root / "admission" / "bindings").glob("*.json")}
    assert slots == {"old"}  # no binding of the candidate was ever stored


def test_the_head_slot_still_activates_and_runs(pair):
    root = pair["root"]
    with stage.AdmissionController(root) as controller:
        controller.activate("old", managed=True)
    stage.run_slot(root, "old", _command(_CLOSING_CHILD, json.dumps([])), timeout=30)
    head = stage.read_admission_head(root)
    assert head["status"] == "closed" and stage.read_binding(root, head["binding_id"])["slot"] == "old"


def test_an_unmanaged_root_still_activates_any_slot(admission_archive, tmp_path):
    source = tmp_path / "offline"
    (source / "config").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "config" / "settings.yaml").write_text("schema: sc2\n")
    root = tmp_path / "staging"
    stage.prepare_slot(admission_archive, root, "one", source)
    stage.prepare_slot(admission_archive, root, "two", source)
    assert stage.activate_slot(root, "one") is None
    assert stage.activate_slot(root, "two") == "slots/one/runtime"
    assert stage.read_admission_head(root) is None


# --- compaction keeps its K activation; nothing else under its gate (CS-12, M-2.5) -------------------

@pytest.fixture(scope="module")
def x_archive(tmp_path_factory):
    return cw.build_x(tmp_path_factory.mktemp("sc2-x"))


@pytest.fixture
def checkout_controller(monkeypatch):
    """P7's installed-controller location cannot hold in process (as in the compaction tests)."""
    monkeypatch.setattr(compaction, "controller_location", lambda root, members: None)


def test_compaction_activates_k_but_no_other_slot_under_its_gate(x_archive, tmp_path, checkout_controller):
    world = cw.world(x_archive, tmp_path, hops=())
    root = world["root"]
    stage.prepare_slot(x_archive, root, "other", world["states"]["a"]["state"])
    started = cw.start(root, x_archive)  # A2 activated K with the production flag
    compaction_id = started["compaction_id"]
    log = stage.pending_compaction(root)
    assert log["compaction_id"] == compaction_id and any(event["kind"] == "activated" for event in log["events"])
    head = stage.read_admission_head(root)
    assert stage.read_binding(root, head["binding_id"])["slot"] == log["intent"]["k_slot"] == "k"
    with stage.AdmissionController(root, gate="compaction:" + compaction_id) as controller:
        _refused(root, (world["states"]["a"]["state"], root / "slots" / "other" / "state"),
                 lambda: controller.activate("other", managed=True), RELEASE_MODE)
    assert stage.read_admission_head(root) == head


# --- epochs: an epoch-3 controller copy is locked out of an epoch-4 root ------------------------------

@pytest.fixture(scope="module")
def epoch3_archive(tmp_path_factory):
    if subprocess.run(["git", "cat-file", "-e", PINNED_EPOCH3 + "^{commit}"], cwd=REPO).returncode != 0:
        pytest.fail(PINNED_EPOCH3 + " is not available in this clone")
    base = tmp_path_factory.mktemp("sc2-epoch3")
    clone = base / "clone"
    subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(REPO), str(clone)],
                   check=True, timeout=120)
    destination = base / "epoch3.tar.gz"
    manifest = build_runtime(clone, destination, PINNED_EPOCH3)
    assert manifest["controller_epoch"] == 3 and "schema_contract" not in manifest
    return destination


def _pinned_cli(root: Path, *arguments: str) -> subprocess.CompletedProcess:
    controller = root / "slots" / "p" / "runtime" / "scripts" / "stage_runtime.pyc"
    return subprocess.run([sys.executable, "-B", str(controller), "--root", str(root), "--offline", *arguments],
                          cwd=controller.parents[1], capture_output=True, text=True, timeout=120)


def test_an_epoch3_controller_copy_cannot_take_the_lease_or_be_installed(x_archive, epoch3_archive, tmp_path,
                                                                         checkout_controller):
    world = cw.world(x_archive, tmp_path, hops=())
    root = world["root"]
    record = json.loads((root / "admission" / "controller.json").read_text())
    assert record["controller_epoch"] == 4
    stage.prepare_slot(epoch3_archive, root, "p", world["states"]["a"]["state"])
    states = (world["states"]["a"]["state"], root / "slots" / "p" / "state")
    before = _witness(root, *states)
    for arguments in (("delegate", "p"), ("activate", "p")):
        result = _pinned_cli(root, *arguments)
        assert result.returncode != 0 and "runtime_admission_controller_downgrade" in result.stderr, result.stderr
    assert _witness(root, *states) == before
    with pytest.raises(launcher.LauncherRefusal, match="runtime_admission_controller_epoch_too_old"):
        launcher.install_controller(root, epoch3_archive)
    assert json.loads((root / "admission" / "controller.json").read_text()) == record
    assert _witness(root, *states) == before


# --- an existing owner chain still validates, serves and compacts --------------------------------------

def test_an_existing_two_owner_chain_validates_serves_and_compacts(x_archive, tmp_path, checkout_controller):
    world = cw.world(x_archive, tmp_path, hops=("b", "c"))  # owners built in the legacy context only
    assert stage.LEGACY_MANAGED_TRANSITIONS is False
    root = world["root"]
    head = stage.read_admission_head(root)
    binding = stage.read_binding(root, head["binding_id"])
    owners = {record["source_binding"] for record in stage._delegation_records(root, Path(binding["state_root"]))}
    assert owners == {world["states"]["a"]["binding"], world["states"]["b"]["binding"]}
    stage.validate_target_ownership(root, binding)
    assert cw.serve(root, tmp_path, "owners").code == 0
    stage.prepare_slot(x_archive, root, "next", world["h"]["state"])
    with pytest.raises(ValueError, match=RETIRED):
        stage.delegate_slot(root, "next")
    started = cw.start(root, x_archive)
    compaction.compact_verify(root, started["compaction_id"])
    assert cw.serve(root, tmp_path, "verify").code == 0
    result = compaction.compact_retire(root, started["compaction_id"], scan=cw.no_units)
    assert result["phase"] == "complete", result
    head = stage.read_admission_head(root)
    k_state = Path(stage.read_binding(root, head["binding_id"])["state_root"])
    assert list(stage._delegation_records(root, k_state)) == []
    assert cw.serve(root, tmp_path, "after").code == 0
    with pytest.raises(ValueError, match=RETIRED):
        stage.delegate_slot(root, "next")
