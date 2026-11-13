"""U3b-0e5: Q5 retirement of the frozen b0866e8 state (P7=B) and the rollback-target link check (N4).

Design v3 section 7 'from day 15' and sub-unit U3b-0e5. On the
production-shaped root of the 0e3 tests (written by b0866e8's own code,
adopted by the R1 archive of this tree), with a real clean managed close:
the launcher's begin_run ticket, the child's receipt writer and the
controller's own seal, in process. The compiled T-A17 on Linux stays with
the U3b-0e4 rehearsals.

Groups: the link check (uninstall-launcher and both aborts), T-A17
(retire, then L4 refuses with the record; with only the link check it
refuses with legacy_state; the managed slot is unaffected), the day-15 and
first-close gates, the reference scan, the exclusions, crash safety from
the intent, links never followed, and the CLI.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import adoption_commands as commands
from scripts import managed_launcher as launcher
from scripts import stage_runtime as stage
from tests.integration.test_adoption_commands import CONTAINER, VOICE, World, artifacts, tree, tree_manifest  # noqa: F401
from tests.mocks import adoption_legacy as fixture

LEGACY = fixture.B_SLOT
P7 = "u3b-0e-P7-B-20261112"
DAY = 86400
T0 = 1_800_000_000.0  # the first clean managed close (DAY 0), a fixed wall clock
DUE = T0 + 15 * DAY
RESOURCES = dict.fromkeys(["graph_closed", "privacy_closed", "sdk_closed", "native_closed"], True)
REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def world(tmp_path, tree, artifacts):  # noqa: F811
    return World(tmp_path.resolve(), tree, artifacts)


def clean_close(world: World, ended_at: float = T0, **exit_changes) -> dict:
    """One managed run of the adopted head, sealed by the controller's own seal; then its run/exit records."""
    root = world.root
    with stage.AdmissionController(root) as controller:
        controller.activate("r1", managed=True)
        _binding, running, previous, ticket = controller.begin_run()
        child = os.fork()
        if child == 0:  # the managed child: its own receipt writer, under the inherited lease
            code = 1
            try:
                stage.write_admission_receipt(ticket, RESOURCES)
                code = 0
            finally:
                os._exit(code)
        _pid, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        stage.seal_admission_run(root, running, child, previous)
    run_id = running["run_id"]
    stage._atomic_record(launcher.run_record_path(root, run_id), {
        "schema_version": 1, "run_id": run_id, "generation": running["generation"],
        "binding_id": running["binding_id"], "child_pid": child, "child_start_time": None,
        "controller_pid": os.getpid(), "boot_id": "boot", "started_at": ended_at - 60}, immutable=True)
    stage._atomic_record(launcher.exit_record_path(root, run_id), dict({
        "schema_version": 1, "run_id": run_id, "returncode": 0, "reason": "stop", "manager_stopping": False,
        "boot_id": "boot", "stragglers": [], "sealed": True, "seal_error": None,
        "state_changed_after_close": False, "stop_requested": True, "ended_at": ended_at}, **exit_changes),
        immutable=True)
    return running


def adopted(world: World, *, close: bool = True) -> World:
    world.adopt()
    world.units.properties[VOICE]["ActiveState"] = "inactive"
    if close:
        clean_close(world)
    return world


def retire(world: World, *, wall: float = DUE, **options):
    return commands.retire_legacy_state(world.root, decision_id=options.pop("decision_id", P7),
                                        container_slot=options.pop("container_slot", CONTAINER),
                                        manager=world.units, wall=lambda: wall, **options)


def dry(world: World, wall: float = DUE) -> dict:
    return retire(world, wall=wall, dry_run=True)


def confirm(world: World, wall: float = DUE, **options) -> dict:
    return retire(world, wall=wall, confirm=dry(world, wall)["plan_sha256"], **options)


def legacy_state(world: World) -> Path:
    return world.root / "slots" / LEGACY / "state"


def kept_trees(world: World) -> dict:
    """Everything Q5 must leave byte-identical: the adopted slot, the container, the checkout, the metrics."""
    return {"r1": tree_manifest(world.root / "slots" / "r1"), "container": tree_manifest(world.root / "slots" / CONTAINER),
            "checkout": tree_manifest(world.base / "legacy" / "kza"),
            "logs": tree_manifest(world.base / "legacy" / "logs")}


def install_record(world: World) -> None:
    (world.unit_dir / "managed.conf").write_text("[Service]\nExecStart=managed\n")
    stage._atomic_record(world.root / "admission" / "launcher" / "install-20261120T000000Z000000.json", {
        "schema_version": 1, "legacy_exec_start": ["/home/kza/kza/.venv/bin/python", "-B", "-P", "-m", "src.main"],
        "properties": {"Type": "simple", "Environment_sha256": "0" * 64}}, immutable=True)


def uninstall(world: World, slot: str = CONTAINER):
    return launcher.uninstall_launcher(world.root, unit_dir=world.unit_dir, decision_id="l4-decision-20261201",
                                       phrase=launcher.UNINSTALL_PHRASE, legacy_slot=slot, manager=world.units)


def unit_dir_files(world: World) -> dict:
    return {path.name: path.read_bytes() for path in world.unit_dir.iterdir()}


# --- the rollback-target link check (N4a) -----------------------------------------------------------

def _legacy_slot_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "releases"
    runtime, state = root / "slots" / "legacy" / "runtime", root / "slots" / "legacy" / "state"
    runtime.mkdir(parents=True)
    (runtime / "release.json").write_text(json.dumps({"git_commit": "63ea8d3" + "0" * 33, "files": {}}))
    frozen = root / "slots" / "frozen" / "state"
    for name in ("config", "data"):
        (frozen / name).mkdir(parents=True)
    (frozen / ".env").write_text("A=1\n")
    state.mkdir()
    for name in ("config", "data", ".env"):
        os.symlink(Path("..") / ".." / "frozen" / "state" / name, state / name)
    os.symlink(Path("..") / "state" / "config", runtime / "config")
    return root, runtime, state


@pytest.mark.parametrize("case", ["intact", "dangling", "missing_config", "loop", "runtime_link", "config_is_file",
                                  "chain_broken"])
def test_the_link_check_resolves_every_link_of_the_rollback_target(tmp_path, case):
    root, runtime, state = _legacy_slot_root(tmp_path)
    frozen = root / "slots" / "frozen" / "state"
    if case == "dangling":
        (frozen / ".env").unlink()
    elif case == "missing_config":
        (state / "config").unlink()
    elif case == "loop":
        os.symlink("loop-b", state / "loop-a")
        os.symlink("loop-a", state / "loop-b")
    elif case == "runtime_link":
        os.symlink(Path("..") / "state" / "models", runtime / "models")
    elif case == "config_is_file":
        (state / "config").unlink()
        (state / "config").write_text("not a directory")
    elif case == "chain_broken":  # the container link resolves to a link that dangles (two hops)
        os.rename(frozen / "data", frozen / "data-moved")
        os.symlink("gone", frozen / "data")
    if case == "intact":
        assert launcher._verify_legacy_slot(root, "legacy", ("63ea8d3",))[1] == state
        return
    with pytest.raises(launcher.LauncherRefusal, match="^runtime_admission_uninstall_legacy_state$"):
        launcher._verify_legacy_slot(root, "legacy", ("63ea8d3",))


def test_link_check_only_l4_refuses_with_legacy_state_and_changes_nothing(world):
    """T-A17, second half: the b0866e8 state is gone without a Q5 record; L4 refuses on the dangling links."""
    adopted(world)
    install_record(world)
    stage.make_writable_then_remove(legacy_state(world))
    units, admission = unit_dir_files(world), tree_manifest(world.root / "admission")
    with pytest.raises(launcher.LauncherRefusal, match="^runtime_admission_uninstall_legacy_state$"):
        uninstall(world)
    assert unit_dir_files(world) == units and tree_manifest(world.root / "admission") == admission
    assert world.units.reloads == 0


@pytest.mark.parametrize("mode", ["abort", "abort_prepared"])
def test_both_aborts_run_the_link_check_before_any_change(world, mode):
    models = world.layout.models_root
    if mode == "abort":
        world.adopt()
        world.units.properties[VOICE]["UnitFileState"] = "disabled"
    else:
        world.prepare()
    head, current = world.head(), os.readlink(world.root / "current")
    slots = sorted(os.listdir(world.root / "slots"))
    os.rename(models, models.with_name("models-away"))  # the container's models link now dangles
    try:
        with pytest.raises(commands.CommandRefusal, match="^runtime_admission_uninstall_legacy_state$"):
            if mode == "abort":
                world.abort()
            else:
                commands.adoption_abort_prepared(world.root, slot="r1", manager=world.units)
    finally:
        os.rename(models.with_name("models-away"), models)
    assert world.head() == head and os.readlink(world.root / "current") == current
    assert sorted(os.listdir(world.root / "slots")) == slots and world.units.enabled == []
    assert not os.listdir(world.unit_dir)


def test_abort_refuses_a_slot_a_q5_intent_names(world):
    world.adopt()
    world.units.properties[VOICE]["UnitFileState"] = "disabled"
    plan = {"x": 1}
    digest = stage._digest(plan)
    stage._atomic_record(launcher.legacy_retirement_directory(world.root) / ("intent-" + digest + ".json"), {
        "schema_version": 1, "kind": "legacy_state_retire_intent", "plan_sha256": digest, "plan": plan,
        "ineligible_slots": [CONTAINER, LEGACY], "intended_at": "x"}, immutable=True)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_uninstall_legacy_retired$") as refused:
        world.abort()
    assert refused.value.report == {"problems": ["legacy-retirement/intent-" + digest + ".json"]}
    assert (world.root / "slots" / "r1").is_dir() and world.units.enabled == []


@pytest.mark.parametrize("name, value", [
    ("notes.json", {"schema_version": 1}),
    ("retired-" + "0" * 64 + ".json", {"schema_version": 1, "kind": "legacy_state_retired",
                                      "plan_sha256": "1" * 64, "ineligible_slots": [CONTAINER]}),
    ("retired-" + "0" * 64 + ".json", {"schema_version": 1, "kind": "legacy_state_retire_intent",
                                      "plan_sha256": "0" * 64, "ineligible_slots": [CONTAINER]}),
    ("retired-" + "0" * 64 + ".json", {"schema_version": 1, "kind": "legacy_state_retired",
                                      "plan_sha256": "0" * 64, "ineligible_slots": []}),
])
def test_a_malformed_retirement_directory_refuses_l4(tmp_path, name, value):
    root, _runtime, _state = _legacy_slot_root(tmp_path)
    directory = launcher.legacy_retirement_directory(root)
    directory.mkdir(parents=True)
    (directory / name).write_text(json.dumps(value))
    with pytest.raises(launcher.LauncherRefusal, match="^runtime_admission_uninstall_legacy_retirement_invalid$"):
        launcher._verify_legacy_slot(root, "legacy", ("63ea8d3",))


# --- T-A17: retire the fixture's b0866e8 state through Q5 -------------------------------------------

def test_t_a17_q5_retires_only_the_frozen_state_and_l4_refuses_with_the_record(world):
    adopted(world)
    install_record(world)
    target = legacy_state(world)
    assert launcher._verify_legacy_slot(world.root, CONTAINER, launcher.LEGACY_COMMITS)  # L4 still valid
    kept, head = kept_trees(world), world.head()
    plan = dry(world)
    assert plan["eligible"] is True and plan["target_slot"] == LEGACY
    assert plan["ineligible_slots"] == sorted([CONTAINER, LEGACY])
    assert plan["census"]["exists"] and plan["census"]["files"] > 0
    assert set(plan["census"]["top_level"]) == {".env", "config", "data", "logs"}
    assert not launcher.legacy_retirement_directory(world.root).exists()  # the dry run writes nothing
    result = retire(world, confirm=plan["plan_sha256"])
    assert not os.path.lexists(target) and (world.root / "slots" / LEGACY).is_dir()
    assert kept_trees(world) == kept and world.head() == head
    directory = launcher.legacy_retirement_directory(world.root)
    record = json.loads((directory / ("retired-" + plan["plan_sha256"] + ".json")).read_text())
    intent = json.loads((directory / ("intent-" + plan["plan_sha256"] + ".json")).read_text())
    assert result["record"] == "legacy-retirement/retired-" + plan["plan_sha256"] + ".json"
    assert record["kind"] == "legacy_state_retired" and record["decision_id"] == P7
    assert record["intent"] == "intent-" + plan["plan_sha256"] + ".json" and record["absence"] == "enoent"
    assert record["ineligible_slots"] == intent["ineligible_slots"] == sorted([CONTAINER, LEGACY])
    assert intent["plan"]["p_answer"] == "P7=B" and stage._digest(intent["plan"]) == plan["plan_sha256"]
    assert intent["plan"]["first_clean_close"]["ended_at"] == T0 and intent["plan"]["eligible_from"] == DUE
    # L4 is gone for both legacy slots, and the refusal names the record.
    units = unit_dir_files(world)
    for slot in (CONTAINER, LEGACY):
        with pytest.raises(launcher.LauncherRefusal, match="^runtime_admission_uninstall_legacy_retired") as refused:
            uninstall(world, slot)
        assert refused.value.code == "runtime_admission_uninstall_legacy_retired"
        assert refused.value.problems == [result["record"]]
    assert unit_dir_files(world) == units
    # The managed service is unaffected: the adopted head still activates and mints a ticket.
    with stage.AdmissionController(world.root) as controller:
        controller.activate("r1", managed=True)
        _binding, running, _previous, _ticket = controller.begin_run()
    assert running["status"] == "running"
    # Twice: refused, nothing changes.
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_already_retired$"):
        dry(world)


def test_the_retirement_never_follows_a_link_inside_the_frozen_state(world):
    adopted(world)
    venv = world.base / "legacy" / "kza" / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("interpreter")
    os.symlink(venv, legacy_state(world) / "venv-link")
    os.symlink(world.layout.models_root / "whisper-v3-turbo" / "model.bin", legacy_state(world) / "data" / "model")
    kept = kept_trees(world)
    result = confirm(world)
    assert result["census"]["links"] == 2
    assert kept_trees(world) == kept and (venv / "bin" / "python").read_text() == "interpreter"


# --- gates: decision, adoption, first clean close, day 15, confirm, unit ---------------------------

def test_a_decision_id_is_required_before_anything(world):
    adopted(world)
    before = tree_manifest(world.root)
    for decision in ("", "x", "with spaces", None):
        with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_decision$"):
            retire(world, decision_id=decision, dry_run=True)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_confirm$"):
        retire(world, confirm="abc")
    assert tree_manifest(world.root) == before


@pytest.mark.parametrize("case", ["no_run", "not_sealed", "not_requested", "crashed", "no_receipt", "seal_error"])
def test_no_retirement_before_the_first_clean_managed_close(world, case):
    adopted(world, close=False)
    if case != "no_run":
        changes = {"not_sealed": {"sealed": False}, "not_requested": {"stop_requested": False},
                   "crashed": {"reason": "child_exit"}, "seal_error": {"seal_error": "x"}}.get(case, {})
        running = clean_close(world, **changes)
        if case == "no_receipt":
            os.unlink(world.root / "admission" / "receipts" / (running["run_id"] + ".json"))
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_no_clean_close$"):
        dry(world, wall=DUE + 365 * DAY)
    assert legacy_state(world).is_dir()


def test_day_zero_is_the_earliest_clean_close(world):
    adopted(world)
    clean_close(world, ended_at=T0 + 5 * DAY)
    assert dry(world, wall=DUE - 1)["eligible"] is False
    assert dry(world)["eligible_from"] == launcher._now_stamp(DUE)


def test_no_retirement_before_day_15(world):
    """DAY 0 + 14 days still serves L4 (until the end of day 14); the deletion is allowed from DAY 0 + 15 x 24 h."""
    adopted(world)
    for wall in (T0 + 14 * DAY, DUE - 1):
        plan = dry(world, wall=wall)
        assert plan["eligible"] is False and plan["eligible_from"] == launcher._now_stamp(DUE)
        with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_too_early$") as refused:
            retire(world, wall=wall, confirm=plan["plan_sha256"])
        assert refused.value.report == {"eligible_from": launcher._now_stamp(DUE)}
    assert legacy_state(world).is_dir() and not launcher.legacy_retirement_directory(world.root).exists()
    assert confirm(world, wall=DUE)["resumed"] is False


def test_the_confirmation_must_be_the_dry_run_plan(world):
    adopted(world)
    plan = dry(world)
    (legacy_state(world) / "data" / "late.txt").write_text("changed after the dry run")
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_confirm_mismatch$"):
        retire(world, confirm=plan["plan_sha256"])
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_confirm_mismatch$"):
        retire(world, decision_id="another-decision-1", confirm=dry(world)["plan_sha256"])
    assert legacy_state(world).is_dir() and not launcher.legacy_retirement_directory(world.root).exists()


@pytest.mark.parametrize("case, code", [
    ("active", "runtime_admission_adoption_unit_active"),
    ("open_head", "runtime_admission_not_closed"),
    ("no_adoption", "runtime_admission_legacy_retire_adoption"),
    ("wrong_container", "runtime_admission_legacy_retire_container_mismatch"),
    ("unexpected_entry", "runtime_admission_legacy_retire_slot_unexpected"),
])
def test_the_run_refuses_outside_its_preconditions(world, case, code):
    if case == "no_adoption":
        with pytest.raises(commands.CommandRefusal, match="^" + code + "$"):
            dry(world)
        assert legacy_state(world).is_dir()
        return
    adopted(world)
    plan = dry(world)["plan_sha256"]
    container = CONTAINER
    if case == "active":
        world.units.properties[VOICE]["ActiveState"] = "active"
    elif case == "open_head":
        with stage.AdmissionController(world.root) as controller:
            controller.begin_run()
    elif case == "wrong_container":
        container = LEGACY
    else:
        (world.root / "slots" / LEGACY / "notes").write_text("x")
    with pytest.raises(ValueError, match="^" + code + "$"):
        retire(world, confirm=plan, container_slot=container)
    assert legacy_state(world).is_dir() and not launcher.legacy_retirement_directory(world.root).exists()


# --- nothing may still point at the copies -----------------------------------------------------------

@pytest.mark.parametrize("case, problem", [
    ("slot_link", "link:slots/r1"),
    ("current", "link:current"),
    ("unit", "unit:kza-legacy-tool.service"),
    ("unit_environment", "unit:kza-legacy-tool.service"),
    ("logrotate", "logrotate"),
    ("config", "config:observability.legacy_path"),
    ("env", "env:LEGACY_DATA"),
    ("admission", "admission:history"),
])
def test_anything_pointing_at_the_copies_refuses(world, case, problem):
    adopted(world)
    container_logs = os.fspath(world.root / "slots" / CONTAINER / "state" / "logs")
    r1_state = world.root / "slots" / "r1" / "state"
    if case == "slot_link":
        os.symlink(legacy_state(world) / "data", r1_state / "data" / "old-data")
    elif case == "current":
        os.unlink(world.root / "current")
        os.symlink(Path("slots") / CONTAINER / "runtime", world.root / "current")
    elif case == "unit":
        world.units.properties["kza-legacy-tool.service"] = {
            "LoadState": "loaded", "ExecStart": "{ path=/bin/true ; argv[]=/bin/true ; ignore_errors=no }",
            "Environment": "", "WorkingDirectory": container_logs}
    elif case == "unit_environment":
        world.units.properties["kza-legacy-tool.service"] = {
            "LoadState": "loaded", "ExecStart": "{ path=/bin/tool ; argv[]=/bin/tool --data ${DATA} ; ignore_errors=no }",
            "Environment": "DATA=" + os.fspath(legacy_state(world) / "data"), "WorkingDirectory": "/"}
    elif case == "logrotate":
        (world.root / launcher.LOGROTATE_NAME).write_text(container_logs + "/console.log {\n  copytruncate\n}\n")
    elif case == "config":
        settings = r1_state / "config" / "settings.yaml"
        os.chmod(settings, 0o600)
        settings.write_text(settings.read_text() + "\nobservability:\n  legacy_path: "
                            + os.fspath(legacy_state(world) / "data") + "\n")
    elif case == "env":
        with open(r1_state / ".env", "a") as stream:
            stream.write("LEGACY_DATA=" + os.fspath(legacy_state(world) / "data") + "\n")
    else:
        stage._atomic_record(world.root / "admission" / "history" / "note.json",
                             {"path": os.fspath(legacy_state(world))})
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_referenced:") as refused:
        dry(world)
    assert problem in refused.value.report["problems"]
    assert legacy_state(world).is_dir() and not launcher.legacy_retirement_directory(world.root).exists()


@pytest.mark.parametrize("key", ["models_root", "venv", "r1_slot", "head_state"])
def test_the_exclusions_refuse_before_any_deletion(world, monkeypatch, key):
    adopted(world)
    name, record = commands.committed_adoption(world.root)
    target = legacy_state(world)
    record = json.loads(json.dumps(record))
    if key == "models_root":
        record["layout"]["models_root"] = os.fspath(target / "data")
    elif key == "venv":
        record["layout"]["legacy_checkout"] = os.fspath(target)
    elif key == "r1_slot":
        record["slot"] = LEGACY
    monkeypatch.setattr(commands, "committed_adoption", lambda root: (name, record))
    if key == "head_state":
        binding = stage.read_binding(world.root, world.head()["binding_id"])
        changed = dict(binding, state_root=os.fspath(target))
        monkeypatch.setattr(stage, "read_binding", lambda root, identity: changed)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_excluded:" + key + "$"):
        dry(world)
    assert target.is_dir()


def test_a_mount_inside_the_frozen_state_refuses(world, monkeypatch):
    adopted(world)
    point = os.path.realpath(legacy_state(world) / "data")
    monkeypatch.setattr(stage, "_mount_points", lambda: [point])
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_mount$"):
        dry(world)
    assert legacy_state(world).is_dir()


# --- crash safety ------------------------------------------------------------------------------------

class _Crash(RuntimeError):
    pass


def test_a_crash_mid_deletion_completes_from_the_intent_and_l4_stays_refused(world, monkeypatch):
    adopted(world)
    install_record(world)
    plan = dry(world)["plan_sha256"]
    original = stage.make_writable_then_remove

    def partial(path):
        original(Path(path) / "data")  # part of the tree is gone, then the power fails
        raise _Crash()
    monkeypatch.setattr(stage, "make_writable_then_remove", partial)
    with pytest.raises(_Crash):
        retire(world, confirm=plan)
    monkeypatch.setattr(stage, "make_writable_then_remove", original)
    intent = "legacy-retirement/intent-" + plan + ".json"
    with pytest.raises(launcher.LauncherRefusal) as refused:
        uninstall(world)
    assert refused.value.code == "runtime_admission_uninstall_legacy_retired" and refused.value.problems == [intent]
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_intent_mismatch$"):
        retire(world, decision_id="another-decision-1", confirm=plan)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_intent_mismatch$"):
        retire(world, confirm="f" * 64)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_intent_mismatch$"):
        dry(world)
    result = retire(world, confirm=plan)
    assert result["resumed"] is True and not os.path.lexists(legacy_state(world))
    with pytest.raises(launcher.LauncherRefusal) as refused:
        uninstall(world)
    assert refused.value.problems == [result["record"]]


def test_a_crash_after_the_deletion_writes_the_record_on_the_rerun(world, monkeypatch):
    adopted(world)
    plan = dry(world)["plan_sha256"]
    original = stage._atomic_record

    def record(path, value, **kwargs):
        if Path(path).name.startswith("retired-"):
            raise _Crash()
        original(path, value, **kwargs)
    monkeypatch.setattr(stage, "_atomic_record", record)
    with pytest.raises(_Crash):
        retire(world, confirm=plan)
    monkeypatch.setattr(stage, "_atomic_record", original)
    assert not os.path.lexists(legacy_state(world))
    result = retire(world, confirm=plan)
    assert result["resumed"] is True and result["census"]["exists"] is True


# --- the CLI ---------------------------------------------------------------------------------------

def _cli(world, *arguments):
    return subprocess.run([sys.executable, "-I", "-B", os.fspath(REPOSITORY / "scripts" / "stage_runtime.py"),
                           "--root", os.fspath(world.root), "--offline", *arguments],
                          capture_output=True, text=True, timeout=300)


def test_the_cli_requires_a_mode_and_refuses_with_78(world):
    assert "retire-legacy-state" in stage.ADOPTION_COMMANDS and stage.ADOPTION_COMMANDS == commands.COMMANDS
    result = _cli(world, "retire-legacy-state", "--decision", P7, "--container-slot", CONTAINER)
    assert result.returncode == 2
    result = _cli(world, "retire-legacy-state", "--decision", P7, "--container-slot", CONTAINER, "--dry-run")
    assert result.returncode == 78
    assert json.loads(result.stderr.splitlines()[-1]) == {"refused": "runtime_admission_legacy_retire_adoption"}
    assert legacy_state(world).is_dir()
