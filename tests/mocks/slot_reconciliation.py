"""Compiled O/N/R/F orchestration; production modules own all state derivation."""
from __future__ import annotations

import json
from pathlib import Path
import re


def fixture_layout(ordinary, latest, destination):
    """Declare concrete fixture store paths without a generic authority rewrite."""
    from src.memory.payload_derivation import (
        DatabaseRelocation, DerivationLayout, FileRootRelocation, IndexRelocation,
    )

    destination = Path(destination).resolve()
    privacy = "data/graph/privacy.db"
    roots = tuple(Path(snapshot.source_binding["state_root"])
        for snapshot in (ordinary, latest))

    def relocate(value):
        if type(value) is dict:
            return {key: relocate(child) for key, child in value.items()}
        if type(value) is list:
            return [relocate(child) for child in value]
        if type(value) is str and Path(value).is_absolute():
            matches = {Path(value).relative_to(root).as_posix() for root in roots
                if Path(value).is_relative_to(root)}
            if len(matches) != 1:
                raise ValueError("reconciliation_unqualified_target_locator")
            return str(destination / matches.pop())
        return value

    targets = {}
    for snapshot in (ordinary, latest):
        tables = snapshot.manifest["databases"][privacy]["tables"]
        for row in tables["erasure_targets"]["rows"]:
            mapped = relocate(json.loads(row["descriptor"]))
            if row["id"] in targets and targets[row["id"]] != mapped:
                raise ValueError("reconciliation_target_identity_collision")
            targets[row["id"]] = mapped
    database_paths = sorted(latest.manifest["databases"])
    if set(database_paths) != set(ordinary.manifest["databases"]):
        raise ValueError("reconciliation_database_inventory_changed")
    directories = [("ambient_capture", "audio"), ("collectors", "wake"),
        ("collectors", "stt"), ("collectors", "conversations"), ("collectors", "exports"),
        ("contexts", "contexts"), ("diagnostics", "metrics_policy_v1.jsonl")]
    indexes = {(item["relative_path"], item["collection"])
        for item in latest.manifest["index_exports"]}
    if indexes != {(item["relative_path"], item["collection"])
            for item in ordinary.manifest["index_exports"]}:
        raise ValueError("reconciliation_index_inventory_changed")
    return DerivationLayout(destination, privacy,
        tuple(DatabaseRelocation(path, path, path) for path in database_paths),
        tuple(FileRootRelocation(family, "data/graph/" + name, "data/graph/" + name,
            "data/graph/" + name) for family, name in directories),
        tuple(IndexRelocation(path, path, path, collection) for path, collection in sorted(indexes)),
        targets)


def derive_branch(controller, ordinary, latest, slot, *, now):
    """Keep the result non-serving until exact typed authority admission succeeds."""
    from src.memory.payload_derivation import materialize_derived_state, plan_payload_cut
    from scripts.stage_runtime import read_admission_head

    destination = controller.root / "slots" / slot / "state"
    layout = fixture_layout(ordinary, latest, destination)
    before = read_admission_head(controller.root)
    plan = plan_payload_cut(ordinary, latest, now=now, layout=layout)
    derived = materialize_derived_state(plan, destination, controller=controller)
    assert derived.serving is False
    assert read_admission_head(controller.root) == before
    identity = controller.authorize_derived_state(slot, latest.source_binding_id,
        target_relocations=derived.event_target_id_map)
    assert read_admission_head(controller.root) == before
    return derived, identity


def prepare_case(task, evidence, repo, new_revision):
    """Build exact old/new artifacts and declare four independent physical roots."""
    from scripts.build_runtime import build_runtime
    from scripts.stage_runtime import prepare_slot
    from tests.mocks import compiled_restore as compiled
    from tests.mocks import slot_rollback as original
    import yaml

    if re.fullmatch("[0-9a-f]{40}", new_revision) is None:
        raise ValueError("reconciliation_exact_new_commit_required")
    # The older application stays genuine OLD code. Its security bridge uses
    # the exact current production closure observer, with independent provenance.
    prepared = original._prepare(task, evidence, repo, original.REVISION,
        bridge_revision=new_revision)
    settings = prepared["runtimes"]["old"].parent / "state/config/settings.yaml"
    value = yaml.safe_load(settings.read_text())
    value["reasoner"]["destination"] = "cloud"
    value["reasoner"].setdefault("cloud", {})["consent"] = True
    settings.write_text(yaml.safe_dump(value, sort_keys=False))
    archive = task / "new-runtime.tar.gz"
    manifest = build_runtime(repo, archive, revision=new_revision)
    required = {"src/memory/state_snapshot.pyc", "src/memory/payload_derivation.pyc",
        "src/memory/state_journal_derivation.pyc", "src/core/state_configuration.pyc"}
    if not required <= manifest["files"].keys() or new_revision == original.REVISION:
        raise ValueError("reconciliation_new_artifact_missing_production_derivation")
    entry = "tests/mocks/slot_reconciliation_child.py"
    target = prepared["island"] / entry
    target.write_bytes((repo / entry).read_bytes())
    prepared["island_files"][entry] = original._sha(target)
    specs = {"O": ("old", prepared["archive"], original.REVISION, prepared["manifest"]),
        "N": ("candidate", archive, new_revision, manifest),
        "R": ("rollback", prepared["archive"], original.REVISION, prepared["manifest"]),
        "F": ("forward", archive, new_revision, manifest)}
    controls, runtimes, roots, manifests = {}, {}, {}, {}
    plan = {"schema_version": 1, "unit": "T024-F1/material-code-data-reconciliation",
        "controls": {}, "outputs": {}}
    common = dict(prepared["controls"]["old"][0])
    common["island_files"] = prepared["island_files"]
    for branch, (slot, artifact, revision, release) in specs.items():
        if branch != "O":
            prepare_slot(artifact, prepared["staging"], slot,
                prepared["runtimes"]["old"].parent / "state")
        runtime = (prepared["staging"] / "slots" / slot / "runtime").resolve()
        root = runtime.parent / "state/data/graph"
        root.mkdir(exist_ok=True)
        control = {**common, "runtime_root": str(runtime), "state_root": str(root),
            "source_commit": revision, "release_sha256": original._sha(runtime / "release.json")}
        path = evidence / (branch + "-control.json")
        compiled.write_json(path, control)
        controls[branch] = control, path
        plan["controls"][branch] = {"path": str(path), "sha256": original._sha(path)}
        plan["outputs"][branch] = str(prepared["checkpoint"].with_name(branch + "-state.json"))
        runtimes[branch], roots[branch], manifests[branch] = runtime, root, release
    compiled.write_json(evidence / "phase-inputs.json", plan)
    compiled._export_file(archive, evidence / "new-runtime.tar.gz")
    compiled.write_json(evidence / "new-release.json", manifest)
    prepared.update(full_controls=controls, full_runtimes=runtimes, full_roots=roots,
        full_manifests=manifests, phase_inputs=plan,
        phase_inputs_sha256=original._sha(evidence / "phase-inputs.json"),
        new_archive=archive, new_manifest=manifest)
    return prepared


def clear_fresh_placeholder(state, expected_settings):
    """Remove only the known empty state just created by this fixture's installer."""
    state = Path(state)
    files = {path.relative_to(state).as_posix() for path in state.rglob("*") if path.is_file()}
    directories = {path.relative_to(state).as_posix() for path in state.rglob("*") if path.is_dir()}
    if (files != {"config/settings.yaml"} or directories != {"config", "data", "data/graph"}
            or any(path.is_symlink() for path in state.rglob("*"))
            or (state / "config/settings.yaml").read_bytes() != expected_settings):
        raise ValueError("reconciliation_placeholder_contains_unexpected_state")
    (state / "config/settings.yaml").unlink()
    for relative in ("data/graph", "data", "config"):
        (state / relative).rmdir()


def run_phase(controller, prepared, evidence, branch, mode, *, timeout):
    """Require the actual controller wait/closure and audited compiled origins."""
    from scripts.stage_runtime import read_admission_head, verify_closed_binding
    from tests.mocks import compiled_restore as compiled
    from tests.mocks import slot_rollback as original

    control, path = prepared["full_controls"][branch]
    output = prepared["phase_inputs"]["outputs"][branch]
    command = [control["executable_literal"], "-I", "-S", "-B",
        str(prepared["island"] / "tests/mocks/slot_reconciliation_child.py"),
        "--mode", mode, "--control", str(path), "--runtime", control["runtime_root"],
        "--state-root", control["state_root"], "--evidence-root", str(evidence),
        "--checkpoint", output]
    slot = {"O": "old", "N": "candidate", "R": "rollback", "F": "forward"}[branch]
    with (evidence / (mode + "-stdout.log")).open("x") as out, (
            evidence / (mode + "-stderr.log")).open("x") as err:
        completed = controller.run(slot, command, timeout=timeout,
            env=compiled._environment(), stdout=out, stderr=err)
    head = read_admission_head(controller.root)
    verify_closed_binding(controller.root, head["binding_id"])
    closure_path = controller.root / "admission/closures" / (head["binding_id"] + ".json")
    closure = json.loads(closure_path.read_text())
    result_path = evidence / (mode + "-result.json")
    result = json.loads(result_path.read_text())
    assert set(result) == {"schema_version", "unit", "mode", "pid", "source_commit",
        "runtime_root", "state_root", "phase_inputs_sha256", "checkpoint_sha256",
        "module_origins", "resources", "sys_flags"}
    assert completed.returncode == 0 and result["schema_version"] == 1
    assert result["unit"] == "T024-F1/material-code-data-reconciliation" and result["mode"] == mode
    assert result["pid"] == closure["child_pid"]
    for key in ("source_commit", "runtime_root", "state_root"):
        assert result[key] == control[key]
    assert result["phase_inputs_sha256"] == prepared["phase_inputs_sha256"]
    assert result["checkpoint_sha256"] == original._sha(Path(output))
    assert result["sys_flags"] == {"isolated": 1, "no_site": 1, "no_user_site": 1,
        "ignore_environment": 1, "dont_write_bytecode": 1}
    resources = result["resources"]
    assert set(resources) == {"graph_constructed", "graph_closed", "writer_lease_released",
        "sdk_clients_closed", "zero_HA_service_calls", "native_gates_returned"}
    assert all(resources[key] is True for key in ("graph_constructed", "graph_closed",
        "writer_lease_released", "sdk_clients_closed", "zero_HA_service_calls"))
    assert resources["native_gates_returned"] is (True if branch == "N" else None)
    manifest = prepared["full_manifests"][branch]
    origins = result["module_origins"]
    assert origins
    for name, record in origins.items():
        assert set(record) == {"relative_origin", "sha256", "loader", "phase"}
        assert name == "src" or name.startswith("src.")
        assert record["relative_origin"].startswith("src/") and record["relative_origin"].endswith(".pyc")
        assert record["sha256"] == manifest["files"][record["relative_origin"]]
        assert record["loader"] == "SourcelessFileLoader" and record["phase"] == mode
    if branch in {"N", "F"}:
        assert {"src.core.runtime_admission", "src.core.state_configuration"} <= set(origins)
    receipt = {"schema_version": 1, "mode": mode, "actual_wait_returned": True,
        "wait_owner": "AdmissionController.run communicate+Popen.wait", "pid": result["pid"],
        "exit_code": completed.returncode, "closed_binding_id": head["binding_id"],
        "closure_sha256": original._sha(closure_path), "result_sha256": original._sha(result_path),
        "module_count": len(origins), "checkpoint_sha256": result["checkpoint_sha256"]}
    compiled.write_json(evidence / (mode + "-parent-receipt.json"), receipt)
    return receipt


def assert_material_point(expected, actual, reference, *, source_state, destination_state,
                          latest_snapshot, derived):
    """Compare all payload fields, allowing only declared locator/policy differences."""
    from copy import deepcopy

    expected, actual = deepcopy(expected), deepcopy(actual)
    families = {"semantic", "captures", "collectors", "contexts", "events", "suggestions",
        "diagnostics", "voices", "streams"}
    assert set(expected) == set(actual) == families
    for row in expected["captures"]["rows"]:
        if row["audio_path"]:
            original = Path(row["audio_path"])
            assert original.is_relative_to(source_state)
            row["audio_path"] = str(destination_state / original.relative_to(source_state))
    for row in expected["collectors"]["rows"]:
        row["target_id"] = derived.target_id_map[row["target_id"]]
    for row in expected["suggestions"]:
        evidence = json.loads(row["evidence_json"])
        for dependency in evidence:
            dependency["target_id"] = derived.event_dependency_target_map[dependency["target_id"]]
        row["evidence_json"] = evidence
    for row in actual["suggestions"]:
        row["evidence_json"] = json.loads(row["evidence_json"])
    # Partition is a regenerated cache locator, not numeric catalog payload.
    for image in (expected, actual):
        for row in image["diagnostics"]["metrics"]:
            row.pop("partition")
        image["voices"]["review"].pop("observed_at")
    privacy = latest_snapshot.manifest["databases"]["data/graph/privacy.db"]["tables"]
    latest_voices = {row["id"]: row for row in privacy["voice_fingerprints"]["rows"]}
    for row in expected["voices"]["rows"]:
        current = latest_voices.get(row["id"])
        fields = ("suggested_profile", "suggested_revision", "similarity")
        if current is None:
            assert all(row[name] is None for name in fields), "missing_latest_voice_association_proof"
        else:
            for name in fields:
                row[name] = current[name]
    expected_stream, actual_stream = expected.pop("streams"), actual.pop("streams")
    assert actual_stream["frame"]["payload"]["stt"] == reference["label"]
    assert actual_stream["frame"]["receipt"]
    assert actual_stream["reader_namespace"] != expected_stream["reader_namespace"]
    assert actual_stream["client"] != expected_stream["client"]
    for family in sorted(expected):
        assert expected[family] == actual[family], ("material_payload_changed", family)


def exercise_material_reconciliation(task, evidence, repo, *, new_revision):
    """Execute real code/data rollback and forward recovery with retained inputs."""
    import time
    from scripts.stage_runtime import AdmissionController
    from src.memory.state_snapshot import snapshot_closed_state
    from tests.mocks import compiled_restore as compiled
    from tests.mocks.compiled_restore_child import DataCodec
    from tests.mocks.slot_reconciliation_child import read_phase_bag

    started = time.monotonic()
    deadline = started + 240
    task, evidence, repo = Path(task).resolve(), Path(evidence).resolve(), Path(repo).resolve()
    assert not task.is_relative_to(repo) and not repo.is_relative_to(task)
    snapshots = task / "closed-snapshots"
    snapshots.mkdir(mode=0o700)
    prepared = prepare_case(task, evidence, repo, new_revision)
    settings = (prepared["full_roots"]["O"].parents[1] / "config/settings.yaml").read_bytes()
    proof = {"schema_version": 1, "unit": "T024-F1/material-code-data-reconciliation",
        "complete": False, "F1_delivered": False, "goal_complete": False,
        "complete_scope": "material scenario only; guards, broad checks and independent review remain separate",
        "security_bridge_revision": prepared["bridge_manifest"]["git_commit"],
        "security_bridge_archive_sha256": prepared["bridge_archive_sha256"],
        "code_pair": {branch: control[0]["source_commit"]
            for branch, control in prepared["full_controls"].items()}, "phases": {}, "derivations": {}}

    def phase(controller, branch, mode):
        budget = min(55, deadline - time.monotonic() - 20)
        if budget <= 0:
            raise TimeoutError("reconciliation_parent_deadline")
        result = run_phase(controller, prepared, evidence, branch, mode, timeout=budget)
        proof["phases"][branch] = result
        return result

    def load(branch, mode):
        control = prepared["full_controls"][branch][0]
        return read_phase_bag(__import__(DataCodec.__module__, fromlist=["DataCodec"]),
            prepared["phase_inputs"]["outputs"][branch], control, mode)[0]

    try:
        with AdmissionController(prepared["staging"]) as controller:
            receipt_O = phase(controller, "O", "seed-O")
            seed = load("O", "seed-O")
            image_O = snapshot_closed_state(controller, receipt_O["closed_binding_id"], snapshots / "O")
            state_N = prepared["full_roots"]["N"].parents[1]
            clear_fresh_placeholder(state_N, settings)
            derived_N, _ = derive_branch(controller, image_O, image_O, "candidate", now=seed["clock_now"])
            proof["derivations"]["N"] = str(derived_N.preparation_record)
            receipt_N = phase(controller, "N", "advance-N")
            latest = load("N", "advance-N")
            image_N = snapshot_closed_state(controller, receipt_N["closed_binding_id"], snapshots / "N")
            state_R = prepared["full_roots"]["R"].parents[1]
            clear_fresh_placeholder(state_R, settings)
            derived_R, _ = derive_branch(controller, image_O, image_N, "rollback", now=latest["clock_now"])
            proof["derivations"]["R"] = str(derived_R.preparation_record)
            receipt_R = phase(controller, "R", "rollback-R")
            rollback = load("R", "rollback-R")
            assert_material_point(seed["O_observed"], rollback["R_observed"], seed["O"],
                source_state=Path(prepared["full_controls"]["O"][0]["state_root"]).parents[1],
                destination_state=state_R, latest_snapshot=image_N, derived=derived_R)
            image_R = snapshot_closed_state(controller, receipt_R["closed_binding_id"], snapshots / "R")
            state_F = prepared["full_roots"]["F"].parents[1]
            clear_fresh_placeholder(state_F, settings)
            derived_F, _ = derive_branch(controller, image_N, image_R, "forward", now=rollback["clock_now"])
            proof["derivations"]["F"] = str(derived_F.preparation_record)
            phase(controller, "F", "forward-F")
            forward = load("F", "forward-F")
            for field, result_field in (("N", "F_observed"), ("N_only", "F_N_only_observed")):
                assert_material_point(latest[field + "_observed"], forward[result_field], latest[field],
                    source_state=Path(prepared["full_controls"]["N"][0]["state_root"]).parents[1],
                    destination_state=state_F, latest_snapshot=image_R, derived=derived_F)
            assert forward["post_R_revoked_credential"] == rollback["post_R_revoked_credential"]
            assert proof["code_pair"]["O"] == proof["code_pair"]["R"]
            assert proof["code_pair"]["N"] == proof["code_pair"]["F"] != proof["code_pair"]["O"]
            proof["complete"] = True
        return proof
    finally:
        proof["elapsed_s"] = time.monotonic() - started
        compiled.write_json(evidence / "material-parent-proof.json", proof)
