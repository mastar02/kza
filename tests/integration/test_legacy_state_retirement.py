"""U3b-0e5: Q5 retirement of the frozen b0866e8 state (P7=B) and the rollback-target link check (N4).

Design v3 section 7 and sub-unit U3b-0e5, with the user decisions
u3b_0e5_retirement Q1-Q3 (DAY 0 + 14 x 24 h; the 2026-10-29 SQLite backups
retired in the same step; no unmanaged rollback afterwards). On the
production-shaped root of the 0e3 tests (written by b0866e8's own code,
adopted by the R1 archive of this tree), with a real clean managed close:
the launcher's begin_run ticket, the child's receipt writer and the
controller's own seal, in process. The compiled T-A17 on Linux stays with
the U3b-0e4 rehearsals.

Groups: the link check (uninstall-launcher and both aborts), T-A17
(retire, then L4 refuses with the record; with only the link check it
refuses with legacy_state; the managed slot is unaffected), the day-14 and
first-close gates, the reference scan, the exclusions, crash safety from
the intent, links never followed, and the CLI.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
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
DUE = T0 + 14 * DAY  # user decision u3b_0e5_retirement Q1=A: exactly 14 x 24 h
OLD_BACKUP = "release_20261028T144002Z_2c5adb4"  # the /home/kza/kza backup: never part of Q5
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


def make_backups(world: World) -> Path:
    """The 2026-10-29 publication's backup (tools/publish_voice_order.py): release.conf, config, .env and
    databases/ with ten SQLite snapshots of the b0866e8 state's data; plus the older /home/kza/kza backup."""
    backup = world.root / "backups" / CONTAINER
    (backup / "databases").mkdir(parents=True)
    (backup / "release.conf").write_text("[Service]\nWorkingDirectory=/old\n")
    shutil.copytree(legacy_state(world) / "config", backup / "config")
    shutil.copyfile(legacy_state(world) / ".env", backup / ".env")
    data = legacy_state(world) / "data"
    databases = sorted(path.relative_to(data) for path in data.rglob("*")
                       if path.suffix in (".db", ".sqlite3") and path.is_file() and path.name != "lists.db")
    assert len(databases) >= commands.LEGACY_BACKUP_DATABASES
    for relative in databases[:commands.LEGACY_BACKUP_DATABASES]:
        (backup / "databases" / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data / relative, backup / "databases" / relative)
    (world.root / "backups" / OLD_BACKUP).mkdir()
    (world.root / "backups" / OLD_BACKUP / "settings.yaml").write_text("old: 1\n")
    return backup / "databases"


def backup_databases(world: World) -> Path:
    return world.root / "backups" / CONTAINER / "databases"


def adopted(world: World, *, close: bool = True, backups: bool = True) -> World:
    world.adopt()
    if backups:
        make_backups(world)
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
            "logs": tree_manifest(world.base / "legacy" / "logs"),
            "backup_kept": {name: tree_manifest(world.root / "backups" / CONTAINER / name) if (
                world.root / "backups" / CONTAINER / name).is_dir() else (world.root / "backups" / CONTAINER / name).read_bytes()
                for name in ("release.conf", "config", ".env")},
            "old_backup": tree_manifest(world.root / "backups" / OLD_BACKUP)}


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
    assert len(plan["backups"]["databases"]) == 10 and "lists.db" not in plan["backups"]["databases"]
    assert not launcher.legacy_retirement_directory(world.root).exists()  # the dry run writes nothing
    result = retire(world, confirm=plan["plan_sha256"])
    assert not os.path.lexists(target) and (world.root / "slots" / LEGACY).is_dir()
    assert not os.path.lexists(backup_databases(world)) and (world.root / "backups" / CONTAINER).is_dir()
    assert kept_trees(world) == kept and world.head() == head
    directory = launcher.legacy_retirement_directory(world.root)
    record = json.loads((directory / ("retired-" + plan["plan_sha256"] + ".json")).read_text())
    intent = json.loads((directory / ("intent-" + plan["plan_sha256"] + ".json")).read_text())
    assert result["record"] == "legacy-retirement/retired-" + plan["plan_sha256"] + ".json"
    assert record["kind"] == "legacy_state_retired" and record["decision_id"] == P7
    assert record["intent"] == "intent-" + plan["plan_sha256"] + ".json" and record["absence"] == "enoent"
    assert record["ineligible_slots"] == intent["ineligible_slots"] == sorted([CONTAINER, LEGACY])
    assert intent["plan"]["p_answer"] == "P7=B" and stage._digest(intent["plan"]) == plan["plan_sha256"]
    assert record["authorization"] == commands.LEGACY_RETIRE_AUTHORIZATION and "Q2=A" in record["authorization"]
    assert record["removed"] == [{"target": os.fspath(target), "absence": "enoent"},
                                 {"target": os.fspath(backup_databases(world)), "absence": "enoent"}]
    assert record["backups"]["databases"] == plan["backups"]["databases"]
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


# --- gates: decision, adoption, first clean close, day 14, confirm, unit ---------------------------

def test_a_decision_id_is_required_before_anything(world):
    adopted(world)
    before = tree_manifest(world.root)
    for decision in ("", "x", "with spaces", None):
        with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_decision$"):
            retire(world, decision_id=decision, dry_run=True)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_confirm$"):
        retire(world, confirm="abc")
    assert tree_manifest(world.root) == before


@pytest.mark.parametrize("case", ["no_run", "not_sealed", "not_requested", "crashed", "no_receipt", "seal_error",
                                  "foreign_receipt"])
def test_no_retirement_before_the_first_clean_managed_close(world, case):
    adopted(world, close=False)
    if case != "no_run":
        changes = {"not_sealed": {"sealed": False}, "not_requested": {"stop_requested": False},
                   "crashed": {"reason": "child_exit"}, "seal_error": {"seal_error": "x"}}.get(case, {})
        running = clean_close(world, **changes)
        receipt = world.root / "admission" / "receipts" / (running["run_id"] + ".json")
        if case == "no_receipt":
            os.unlink(receipt)
        elif case == "foreign_receipt":  # a receipt that is not this run's child's own
            value = json.loads(receipt.read_text())
            receipt.unlink()
            receipt.write_text(json.dumps(dict(value, child_pid=value["child_pid"] + 1)))
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_no_clean_close$"):
        dry(world, wall=DUE + 365 * DAY)
    assert legacy_state(world).is_dir()


def test_day_zero_is_the_earliest_clean_close(world):
    adopted(world)
    clean_close(world, ended_at=T0 + 5 * DAY)
    assert dry(world, wall=DUE - 1)["eligible"] is False
    assert dry(world)["eligible_from"] == launcher._now_stamp(DUE)


def test_no_retirement_before_day_0_plus_14_x_24_h(world):
    """Q1=A: the deletion is allowed from exactly DAY 0 + 14 x 24 h, not a second earlier."""
    adopted(world)
    for wall in (T0 + 13 * DAY, DUE - 1):
        plan = dry(world, wall=wall)
        assert plan["eligible"] is False and plan["eligible_from"] == launcher._now_stamp(DUE)
        with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_too_early$") as refused:
            retire(world, wall=wall, confirm=plan["plan_sha256"])
        assert refused.value.report == {"eligible_from": launcher._now_stamp(DUE)}
    assert legacy_state(world).is_dir() and not launcher.legacy_retirement_directory(world.root).exists()
    assert dry(world, wall=DUE)["eligible"] is True
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
    ("backup_link", "link:slots/r1"),
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
    elif case == "backup_link":  # Q2: the retired backup set is a target too
        os.symlink(backup_databases(world), r1_state / "data" / "old-backups")
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


@pytest.mark.parametrize("linked", ["backups", "container"])
def test_a_resume_refuses_a_backup_path_turned_into_a_link_after_the_crash(world, monkeypatch, linked):
    """R2-1: the resume skips the census (a partial delete), so backups/ and backups/<C> are rechecked."""
    adopted(world)
    install_record(world)
    plan = dry(world)["plan_sha256"]
    original, removed = stage.make_writable_then_remove, []

    def crash_at_the_backups(path):
        if Path(path) == backup_databases(world):
            raise _Crash()  # The frozen state is gone; the power fails before the backups go.
        removed.append(Path(path))
        original(path)
    monkeypatch.setattr(stage, "make_writable_then_remove", crash_at_the_backups)
    with pytest.raises(_Crash):
        retire(world, confirm=plan)
    monkeypatch.setattr(stage, "make_writable_then_remove", original)
    assert removed == [legacy_state(world)] and backup_databases(world).is_dir()
    # Between the crash and the resume, a link moves the backup target elsewhere.
    moved = world.root / "backups" if linked == "backups" else world.root / "backups" / CONTAINER
    elsewhere = world.base / ("elsewhere-" + linked)
    os.rename(moved, elsewhere)
    moved.symlink_to(elsewhere)
    before = tree_manifest(elsewhere)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_backups_set$"):
        retire(world, confirm=plan)
    assert tree_manifest(elsewhere) == before and moved.is_symlink()
    # Restored to its own real path, the same resume completes.
    moved.unlink()
    os.rename(elsewhere, moved)
    result = retire(world, confirm=plan)
    assert result["resumed"] is True and not os.path.lexists(backup_databases(world))


def test_a_resume_with_the_clock_moved_back_is_still_never_early(world, monkeypatch):
    adopted(world)
    plan = dry(world)["plan_sha256"]

    def crash(path):
        raise _Crash()
    monkeypatch.setattr(stage, "make_writable_then_remove", crash)
    with pytest.raises(_Crash):
        retire(world, confirm=plan)
    monkeypatch.undo()
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_too_early$"):
        retire(world, wall=DUE - 1, confirm=plan)
    assert legacy_state(world).is_dir()
    assert retire(world, confirm=plan)["resumed"] is True


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


# --- review round (0e5 independent review F3-F7, user decisions Q1-Q3) ------------------------------

def test_an_l4_use_after_the_adoption_refuses_whatever_the_unit_view_shows(world):
    """F3: uninstall-launcher served the frozen state again; its data needs a new user decision."""
    adopted(world)
    record = world.root / "admission" / "launcher" / "uninstall-20261125T000000Z000000.json"
    stage._atomic_record(record, {"schema_version": 1, "decision_id": "l4-decision-20261125",
                                  "uninstalled_at": "20261125T000000Z000000"}, immutable=True)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_l4_used$") as refused:
        dry(world)
    assert refused.value.report == {"problems": ["launcher/" + record.name]}
    assert legacy_state(world).is_dir() and backup_databases(world).is_dir()


def test_abort_prepared_after_the_commit_still_removes_a_leftover_work_directory(world):
    """F4: after the commit 'current' is the managed slot; the legacy link check does not apply."""
    adopted(world, close=False, backups=False)
    leftover = world.root / commands.WORK_DIRECTORY / "r2"
    (leftover / "state").mkdir(parents=True)
    result = commands.adoption_abort_prepared(world.root, slot="r2", manager=world.units)
    assert result["removed"] == [commands.WORK_DIRECTORY + "/r2"] and not leftover.exists()
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_adoption_abort_prepared_committed$"):
        commands.adoption_abort_prepared(world.root, slot="r1", manager=world.units)


@pytest.mark.parametrize("case, code", [
    ("checkout", "runtime_admission_legacy_retire_excluded:legacy_checkout"),
    ("kept_backup", "runtime_admission_legacy_retire_excluded:backups_kept"),
    ("kept_beside", "runtime_admission_legacy_retire_excluded:backups_kept"),
    ("head_slot", "runtime_admission_legacy_retire_excluded:head_slot"),
    ("current", "runtime_admission_legacy_retire_excluded:current"),
    ("controller", "runtime_admission_legacy_retire_excluded:controller"),
    ("slot_link", "runtime_admission_legacy_retire_target"),
    ("layout", "runtime_admission_legacy_retire_layout"),
    ("slot_name", "runtime_admission_legacy_retire_slot"),
    ("target_file", "runtime_admission_legacy_retire_target"),
])
def test_the_scope_guards_refuse_before_any_deletion(world, monkeypatch, case, code):
    """F5: /home/kza/kza and every backup outside the retired set are real exclusions; the fixed target holds."""
    adopted(world)
    slot, target = world.root / "slots" / LEGACY, legacy_state(world)
    name, record = commands.committed_adoption(world.root)
    record = json.loads(json.dumps(record))
    if case == "checkout":
        record["layout"]["legacy_checkout"] = os.fspath(slot)
        monkeypatch.setattr(commands, "committed_adoption", lambda root: (name, record))
    elif case == "kept_backup":
        shutil.rmtree(world.root / "backups" / OLD_BACKUP)
        os.symlink(target / "data", world.root / "backups" / OLD_BACKUP)
    elif case == "kept_beside":
        shutil.rmtree(world.root / "backups" / CONTAINER / "config")
        os.symlink(backup_databases(world), world.root / "backups" / CONTAINER / "config")
    elif case == "head_slot":
        binding = stage.read_binding(world.root, world.head()["binding_id"])
        monkeypatch.setattr(stage, "read_binding", lambda root, identity: dict(binding, slot=LEGACY))
    elif case == "current":
        (slot / "runtime").mkdir()
        os.unlink(world.root / "current")
        os.symlink(Path("slots") / LEGACY / "runtime", world.root / "current")
    elif case == "controller":
        os.rename(world.root / "controller", world.root / "controller-aside")
        os.symlink(target, world.root / "controller")
    elif case == "slot_link":
        os.rename(slot, world.root / "elsewhere")
        os.symlink(world.root / "elsewhere", slot)
    elif case == "layout":
        container_config = world.root / "slots" / CONTAINER / "state" / "config"
        os.unlink(container_config)
        os.symlink(world.layout.models_root, container_config)
    elif case == "slot_name":
        record["layout"]["legacy_state"] = os.fspath(world.root / "slots" / "release_x_1234567" / "state")
        monkeypatch.setattr(commands, "committed_adoption", lambda root: (name, record))
    else:
        moved = world.root / "frozen-aside"
        os.rename(target, moved)
        target.write_text("not a directory")
    before = tree_manifest(world.root / "backups")
    with pytest.raises(commands.CommandRefusal, match="^" + code + "$"):
        dry(world)
    assert tree_manifest(world.root / "backups") == before
    assert not launcher.legacy_retirement_directory(world.root).exists()


@pytest.mark.parametrize("case, code", [
    ("missing", "runtime_admission_legacy_retire_backups_set"),
    ("nine", "runtime_admission_legacy_retire_backups_set"),
    ("eleven", "runtime_admission_legacy_retire_backups_set"),
    ("other_file", "runtime_admission_legacy_retire_backups_set"),
    ("journal", "runtime_admission_legacy_retire_backups_set"),
    ("link_inside", "runtime_admission_legacy_retire_backups_set"),
    ("not_sqlite", "runtime_admission_legacy_retire_backups_set"),
    ("not_from_state", "runtime_admission_legacy_retire_backups_set"),
    ("not_from_state_extra", "runtime_admission_legacy_retire_backups_set"),
    ("directory_link", "runtime_admission_legacy_retire_backups_set"),
    ("mount", "runtime_admission_legacy_retire_mount"),
])
def test_the_backup_set_must_be_the_reviewed_one(world, monkeypatch, case, code):
    """Q2=A: exactly the ten SQLite snapshots of the b0866e8 state, regular files, no link, no mount."""
    adopted(world)
    databases = backup_databases(world)
    files = sorted(path for path in databases.rglob("*") if path.is_file())
    if case == "missing":
        shutil.rmtree(databases)
    elif case == "nine":
        files[0].unlink()
    elif case == "eleven":
        shutil.copyfile(legacy_state(world) / "data" / "lists.db", databases / "lists.db")
    elif case == "other_file":
        (databases / "notes.txt").write_text("x")
    elif case == "journal":
        (databases / (files[0].name + "-journal")).write_bytes(b"x")
    elif case == "link_inside":
        files[0].unlink()
        os.symlink(legacy_state(world) / "data" / files[0].relative_to(databases), files[0])
    elif case == "not_sqlite":
        files[0].write_bytes(b"plain text, not a database")
    elif case == "not_from_state":  # still ten files: only the source check can refuse
        files[0].rename(files[0].with_name("renamed.db"))
    elif case == "not_from_state_extra":  # ten would count only if the unknown name were skipped
        files[0].rename(files[0].with_name("renamed.db"))
        shutil.copyfile(legacy_state(world) / "data" / "lists.db", databases / "lists.db")
    elif case == "directory_link":
        os.rename(databases, world.root / "databases-aside")
        os.symlink(world.root / "databases-aside", databases)
    else:
        point = os.fspath(databases / files[0].relative_to(databases).parts[0]) if len(
            files[0].relative_to(databases).parts) > 1 else os.fspath(databases)
        monkeypatch.setattr(stage, "_mount_points", lambda: [point])
    with pytest.raises(commands.CommandRefusal, match="^" + code + "$"):
        dry(world)
    assert legacy_state(world).is_dir() and not launcher.legacy_retirement_directory(world.root).exists()


def test_a_directory_vanishing_during_the_dry_run_walk_is_a_refusal(world, monkeypatch):
    """F7: the dry run walks the live state; a vanished directory gives a code, not a traceback."""
    adopted(world)
    vanishing = world.root / "slots" / "r1" / "state" / "data" / "zz-vanishing" / "inner"
    vanishing.mkdir(parents=True)
    original = os.walk

    def walk(top, *args, **kwargs):
        for item in original(top, *args, **kwargs):
            if vanishing.parent.exists() and item[0] == os.fspath(vanishing.parent.parent):
                shutil.rmtree(vanishing.parent)
            yield item
    monkeypatch.setattr(commands.os, "walk", walk)
    with pytest.raises(commands.CommandRefusal, match="^runtime_admission_legacy_retire_scan$"):
        dry(world)
    monkeypatch.undo()
    assert dry(world)["dry_run"] is True  # the same walk without the vanishing directory reports


# --- the server's umask -----------------------------------------------------------------------------

def created_modes(world: World, before: set) -> dict:
    """Mode of every non-link entry created under the root since ``before`` (the backups are the test's own)."""
    return {path.relative_to(world.root).as_posix(): stat.S_IMODE(os.lstat(path).st_mode)
            for path in set(world.root.rglob("*")) - before
            if not path.is_symlink() and not path.relative_to(world.root).as_posix().startswith("backups")}


def assert_private(world: World, modes: dict) -> None:
    """0700 directories and 0600 files; the slot runtimes keep their read-only 0555/0444 contract."""
    assert modes and any(name.startswith("controller/") for name in modes)
    assert "admission" in modes and ".stage.lock" in modes and any("/state/data/" in name for name in modes)
    for name, mode in modes.items():
        directory = (world.root / name).is_dir()
        if name.startswith("slots/") and name.split("/")[2:3] == ["runtime"]:
            assert mode == (stage.RUNTIME_DIRECTORY_MODE if directory else stage.RUNTIME_FILE_MODE), name
        else:
            assert mode == (0o700 if directory else 0o600), (name, oct(mode))


def test_the_whole_lifecycle_runs_under_the_server_umask(tmp_path, tree, artifacts):  # noqa: F811
    """yo, kza and the kza-voice unit all run with umask 0002: build, adopt, close, retire, serve.

    The production-shaped root is written under that umask too. Every entry the
    commands, the controller, the launcher and stage_runtime create under the
    managed root gets an explicit private mode (0700/0600), never the umask's
    0775/0664; and no later check refuses what an earlier one wrote.
    """
    previous = os.umask(0o002)
    try:
        world = World(tmp_path.resolve(), tree, artifacts, adapters=True)
        assert stat.S_IMODE(os.lstat(world.root).st_mode) == 0o710  # /home/kza/releases
        before = set(world.root.rglob("*"))
        assert world.retire()["record"]  # retire-legacy-training (the LoRA roots)
        adopted(world)  # dry-run inside prepare, install-controller, commit, a clean managed close
        install_record(world)
        assert launcher._verify_legacy_slot(world.root, CONTAINER, launcher.LEGACY_COMMITS)
        plan = dry(world)
        assert plan["eligible"] is True and plan["target_slot"] == LEGACY
        result = retire(world, confirm=plan["plan_sha256"])
        assert result["record"] == "legacy-retirement/retired-" + plan["plan_sha256"] + ".json"
        assert not os.path.lexists(legacy_state(world))
        with stage.AdmissionController(world.root) as controller:
            controller.activate("r1", managed=True)
            _binding, running, _previous, _ticket = controller.begin_run()
        assert running["status"] == "running" and running["generation"] == 6
        assert stat.S_IMODE(os.lstat(world.root).st_mode) == 0o710
        assert_private(world, created_modes(world, before))
    finally:
        os.umask(previous)


def test_a_later_run_accepts_what_an_older_version_left_group_writable(tmp_path, tree, artifacts):  # noqa: F811
    """Entries an earlier controller created under umask 0002 (0775/0664) are accepted as they are.

    The adopted root is loosened to the older modes everywhere outside the
    read-only slot runtime; the next install-controller, managed close, Q5
    retirement and activation all run, leave the older entries unchanged and
    create only private new ones.
    """
    previous = os.umask(0o002)
    try:
        world = World(tmp_path.resolve(), tree, artifacts, adapters=True)
        before = set(world.root.rglob("*"))
        world.retire()
        adopted(world, close=False)
        older = {}
        for name in sorted(created_modes(world, before)):
            if not (name.startswith("slots/") and name.split("/")[2:3] == ["runtime"]):
                path = world.root / name
                os.chmod(path, 0o775 if path.is_dir() else 0o664)
                older[name] = (os.lstat(path).st_ino, stat.S_IMODE(os.lstat(path).st_mode))
        assert older[".stage.lock"][1] == 0o664 and older["admission"][1] == 0o775
        loosened = set(world.root.rglob("*"))
        assert world.install_controller()["controller_epoch"] == 2  # the installed tree is re-verified
        clean_close(world)
        install_record(world)
        result = confirm(world)
        assert result["record"].startswith("legacy-retirement/retired-")
        with stage.AdmissionController(world.root) as controller:
            controller.activate("r1", managed=True)
            _binding, running, _previous, _ticket = controller.begin_run()
        assert running["status"] == "running"
        # Same inode (not replaced by an atomic record rewrite): the older mode is kept as it was.
        kept = {name: (os.lstat(world.root / name).st_ino, stat.S_IMODE(os.lstat(world.root / name).st_mode))
                for name in older if os.path.lexists(world.root / name)}
        kept = {name: value for name, value in kept.items() if value[0] == older[name][0]}
        assert kept == {name: older[name] for name in kept}
        assert {"admission", ".stage.lock", "controller"} <= set(kept) and len(kept) > len(older) // 2
        new = created_modes(world, loosened)
        assert new and all(mode == (0o700 if (world.root / name).is_dir() else 0o600) for name, mode in new.items())
    finally:
        os.umask(previous)
