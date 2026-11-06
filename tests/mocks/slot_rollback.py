"""Source-free public-Graph characterization of independently copied slots."""
from contextlib import closing
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import sysconfig
import time
from uuid import uuid4

from tests.mocks import compiled_restore as compiled


CHILD = "tests/mocks/slot_rollback_child.py"
REVISION = "9b3170294eb69a53dd27b63949c09a63a2efed54"
EVIDENCE_ENV = "KZA_SLOT_ROLLBACK_EVIDENCE_ROOT"
PARENT_SECONDS, PHASE_SECONDS, CLEANUP_SECONDS = 240, 65, 20


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _typed(value):
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    return value


def _quoted(name):
    return '"' + name.replace('"', '""') + '"'


def offline_state(root: Path) -> dict:
    """Read SQL logical fields and non-SDK files without constructing owners."""
    root = Path(root).resolve(strict=True)
    schemas, tables, files = {}, {}, {}
    sdk = root / "chroma"
    databases = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("state_snapshot_must_not_follow_symlinks")
        if not path.is_file() or path.suffix not in {".db", ".sqlite", ".sqlite3"}:
            continue
        databases.add(path)
        relative = path.relative_to(root).as_posix()
        with closing(sqlite3.connect(":memory:")) as image:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
                source.backup(image)
            schemas[relative] = list(image.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"))
            if path.is_relative_to(sdk):
                continue  # Real SDK get(), not its internal SQL rows, compares vectors.
            values = {}
            for (name,) in image.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
                cursor = image.execute("SELECT * FROM " + _quoted(name))
                columns = [column[0] for column in cursor.description]
                rows = [[_typed(value) for value in row] for row in cursor]
                rows.sort(key=lambda row: json.dumps(row, sort_keys=True, allow_nan=False))
                values[name] = {"columns": columns, "rows": rows}
            tables[relative] = values
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or path.is_relative_to(sdk) or path in databases
                or path.name.endswith(("-wal", "-shm", ".lock"))):
            continue
        files[path.relative_to(root).as_posix()] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    return {"schemas": schemas, "tables": tables, "files": files}


def copied_targets(root: Path, old_root: Path) -> list[dict]:
    """Inventory immutable target descriptors before candidate initialization."""
    root, old_root = Path(root).resolve(strict=True), Path(old_root).resolve(strict=True)
    with closing(sqlite3.connect((root / "privacy.db").as_uri() + "?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT id,participant,version,descriptor FROM erasure_targets ORDER BY id").fetchall()

    def paths(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from paths(child, prefix + "/" + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from paths(child, prefix + "/" + str(index))
        elif isinstance(value, str) and Path(value).is_absolute():
            path = Path(value).resolve(strict=False)
            branch = "old" if path.is_relative_to(old_root) else (
                "candidate" if path.is_relative_to(root) else "outside_both_synthetic_branches")
            yield {"field": prefix, "path": value, "branch": branch}

    records = []
    for identity, participant, version, raw in rows:
        descriptor = json.loads(raw)
        found = list(paths(descriptor))
        if any(item["branch"] == "outside_both_synthetic_branches" for item in found):
            raise ValueError("registered_target_outside_synthetic_branches")
        records.append({"id": identity, "participant": participant, "version": version,
            "descriptor": descriptor, "absolute_paths": found})
    if not records:
        raise ValueError("copied_erasure_target_inventory_empty")
    return records


def _result(path, mode, process_pid, control, manifest, checkpoint):
    result = json.loads(path.read_text())
    fields = {"schema_version", "unit", "mode", "source_commit", "runtime_root", "state_root",
        "pid", "checkpoint_sha256", "candidate_checkpoint_sha256", "module_origins", "sys_flags",
        "resources", "observations"}
    if (type(result) is not dict or set(result) != fields or result["schema_version"] != 1
            or result["unit"] != "T024-F1/privacy-safe-runtime-state-rollback"):
        raise compiled.CompiledRestoreFailure("slot_child_result_schema")
    if (result["mode"] != mode or result["source_commit"] != control["source_commit"]
            or result["pid"] != process_pid or result["runtime_root"] != control["runtime_root"]
            or result["state_root"] != control["state_root"] or result["checkpoint_sha256"] != _sha(checkpoint)):
        raise compiled.CompiledRestoreFailure("slot_child_result_identity")
    candidate = checkpoint.with_name("candidate-state.json")
    if result["candidate_checkpoint_sha256"] != (None if mode == "seed" else _sha(candidate)):
        raise compiled.CompiledRestoreFailure("slot_candidate_checkpoint_identity")
    expected_flags = {"isolated": 1, "ignore_environment": 1, "no_user_site": 1,
        "no_site": 1, "dont_write_bytecode": 1}
    if result["sys_flags"] != expected_flags:
        raise compiled.CompiledRestoreFailure("slot_child_not_isolated")
    resources = result["resources"]
    keys = {"graph_constructed", "graph_closed", "writer_lease_released", "sdk_clients_closed",
        "native_gates_returned", "zero_HA_service_calls"}
    if type(resources) is not dict or set(resources) != keys or resources["sdk_clients_closed"] is not True:
        raise compiled.CompiledRestoreFailure("slot_child_resource_contract")
    graph = mode != "inspect-old"
    if (resources["graph_constructed"] is not graph
            or resources["graph_closed"] is not (True if graph else None)
            or resources["writer_lease_released"] is not (True if graph else None)
            or resources["zero_HA_service_calls"] is not (True if graph else None)
            or resources["native_gates_returned"] is not (True if mode == "candidate" else None)):
        raise compiled.CompiledRestoreFailure("slot_child_native_close_not_proven")
    origins = result["module_origins"]
    if type(origins) is not dict or not origins:
        raise compiled.CompiledRestoreFailure("slot_child_origin_inventory_empty")
    for name, origin in origins.items():
        if (type(name) is not str or type(origin) is not dict
                or set(origin) != {"relative_origin", "sha256", "loader", "phase"}):
            raise compiled.CompiledRestoreFailure("slot_child_origin_record_type")
        relative = origin.get("relative_origin", "")
        actual = origin.get("sha256")
        if (not (name == "src" or name.startswith("src.")) or type(relative) is not str
                or not relative.startswith("src/") or relative not in manifest["files"]
                or not relative.endswith(".pyc") or origin.get("loader") != "SourcelessFileLoader"
                or type(actual) is not str or re.fullmatch("[0-9a-f]{64}", actual) is None
                or actual != manifest["files"][relative] or origin.get("phase") != mode):
            raise compiled.CompiledRestoreFailure("slot_child_source_origin_unproven")
    if type(result["observations"]) is not dict:
        raise compiled.CompiledRestoreFailure("slot_child_observation_schema")
    return result


def _phase(mode, *, control, control_path, island, checkpoint, evidence, deadline, manifest,
           controller=None, slot=None):
    available = deadline - time.monotonic() - CLEANUP_SECONDS
    if available <= 0:
        raise compiled.CompiledRestoreFailure("slot_parent_deadline_before_child")
    budget = min(PHASE_SECONDS, available)
    command = [control["executable_literal"], "-I", "-S", "-B", str(island / CHILD),
        "--mode", mode, "--control", str(control_path), "--runtime", control["runtime_root"],
        "--state-root", control["state_root"], "--evidence-root", str(evidence),
        "--checkpoint", str(checkpoint)]
    result_path = evidence / (mode + "-result.json")
    if result_path.exists():
        raise compiled.CompiledRestoreFailure("slot_result_already_exists")
    record = {"command": command, "environment_names": sorted(compiled._environment()),
        "budget_s": budget, "waits": [], "signals": [], "actual_wait_returned": False,
        "containment_only": False, "completed": False}
    process, primary, result = None, None, None
    cleanup = []
    with (evidence / (mode + "-stdout.log")).open("x") as out, (
            evidence / (mode + "-stderr.log")).open("x") as err:
        try:
            if mode != "inspect-old":
                from scripts.stage_runtime import read_admission_head, verify_closed_binding

                if controller is None or slot is None:
                    raise compiled.CompiledRestoreFailure("slot_managed_controller_required")
                completed = controller.run(slot, command, timeout=budget,
                    env=compiled._environment(), stdout=out, stderr=err)
                head = read_admission_head(controller.root)
                verify_closed_binding(controller.root, head["binding_id"])
                closure_path = controller.root / "admission/closures" / (head["binding_id"] + ".json")
                closure = json.loads(closure_path.read_text())
                child_pid = closure["child_pid"]
                code = completed.returncode
                record.update(pid=child_pid, actual_wait_returned=True,
                    wait_owner="AdmissionController.run communicate+Popen.wait",
                    closed_binding_id=head["binding_id"], closure_sha256=_sha(closure_path))
            else:
                process = subprocess.Popen(command, cwd=control["runtime_root"], env=compiled._environment(),
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
                record.update(pid=process.pid, owned_pgid=process.pid)
                code = compiled._wait(process, budget, record)
                child_pid = process.pid
            record["exit_code"] = code
            if code != 0:
                raise compiled.CompiledRestoreFailure("slot_child_exit_nonzero:" + mode)
            result = _result(result_path, mode, child_pid, control, manifest, checkpoint)
            record["completed"] = True
        except BaseException as error:
            primary = error
        finally:
            if process is not None and not record["actual_wait_returned"]:
                record["containment_only"] = True
                try:
                    compiled._stop_owned(process, record)
                except BaseException as error:
                    cleanup.append(error)
            record.update(error_kind=None if primary is None else type(primary).__name__,
                cleanup_error_kinds=[type(error).__name__ for error in cleanup])
            if result is not None:
                record.update(result_sha256=_sha(result_path), module_count=len(result["module_origins"]),
                    observations=result["observations"])
            try:
                compiled.write_json(evidence / (mode + "-parent-receipt.json"), record)
            except BaseException as error:
                cleanup.append(error)
    if primary is not None and cleanup:
        raise BaseExceptionGroup("slot child and cleanup", [primary, *cleanup])
    if primary is not None:
        raise primary
    if cleanup:
        raise BaseExceptionGroup("slot child cleanup", cleanup)
    return result, record


def _privacy_oracle(proof):
    observations = proof["observations"]
    seed, candidate, old = (observations[name] for name in ("seed", "candidate", "inspect-old"))
    seed_keys = {"ten_drivers", "sources_live", "foreign_source_exact", "manual_current",
        "consent_current", "reader_namespace"}
    candidate_keys = {"offline_inventory_before_graph", "producer_completed_while_U_held",
        "source_revoked", "three_guards_pending", "foreign_source_exact", "foreign_payload_preserved",
        "manual_current", "consent_current", "reader_namespace_changed", "historical_targets_retained"}
    old_keys = {"old_U_preserved", "old_manual_preserved", "latest_tombstone_present",
        "latest_profile_deletion_present", "latest_credential_revocation_present",
        "latest_account_permission_present", "consent_current"}
    assert set(seed) == seed_keys and set(candidate) == candidate_keys and set(old) == old_keys
    assert type(seed["reader_namespace"]) is str and seed["reader_namespace"]
    assert all(seed[name] is True for name in seed_keys - {"reader_namespace"}), observations
    assert all(candidate[name] is True for name in candidate_keys), observations
    assert old["old_U_preserved"] is True and old["old_manual_preserved"] is True, observations
    selected_latest = all(old[name] is True for name in old_keys - {"old_U_preserved", "old_manual_preserved"})
    blocked = proof["stale_activation_rejection"] == {"type": "ValueError", "code": "privacy_authority_stale"}
    assert selected_latest or blocked, ("selected_old_state_lost_latest_authority", observations)


def _control_pair(repo, runtimes, roots, island, island_files, evidence, artifact, revision):
    prefix = Path(sys.prefix).resolve()
    dependencies = sorted({str(Path(sysconfig.get_path(name)).resolve()) for name in ("purelib", "platlib")})
    if any(not Path(name).is_dir() or not Path(name).is_relative_to(prefix) for name in dependencies):
        raise ValueError("slot_dependency_root_outside_native_venv")
    common = {"schema_version": 1, "executable_literal": os.path.abspath(sys.executable),
        "parent_prefix": str(prefix), "base_prefix": str(Path(sys.base_prefix).resolve()),
        "python_version": list(sys.version_info[:3]), "cache_tag": sys.implementation.cache_tag,
        "python_magic_hex": importlib.util.MAGIC_NUMBER.hex(), "dependency_roots": dependencies,
        "checkout_root": str(repo), "release_sha256": artifact["release.json"],
        "source_commit": revision, "island_root": str(island), "island_files": island_files,
        "evidence_root": str(evidence)}
    controls = {}
    for name in ("old", "candidate"):
        control = {**common, "runtime_root": str(runtimes[name]), "state_root": str(roots[name])}
        path = evidence / (name + "-control.json")
        compiled.write_json(path, control)
        controls[name] = control, path
    return controls


def _prepare(task, evidence, repo, revision, *,
             bridge_revision="80b47f55716e980b93a7312b6732de0611e3e2e0"):
    from scripts.build_runtime import build_runtime
    from scripts.stage_runtime import activate_slot, prepare_slot
    from tests.mocks.managed_slot_graph import canonical_template
    from tests.mocks.restore_application_graph import configuration
    import yaml

    if (type(bridge_revision) is not str or len(bridge_revision) != 40
            or any(value not in "0123456789abcdef" for value in bridge_revision)):
        raise ValueError("slot_exact_bridge_commit_required")
    archive = task / "runtime.tar.gz"
    manifest = build_runtime(repo, archive, revision=revision)
    if (manifest["git_commit"] != revision or manifest["python_magic"] != importlib.util.MAGIC_NUMBER.hex()
            or manifest["format"] != "kza-cpython-bytecode-v1" or manifest["python"] != "3.13"
            or manifest["entrypoint"] != ["python", "-m", "src.main"]
            or manifest["external_paths"] != ["config", "data", "models", "logs"]):
        raise compiled.CompiledRestoreFailure("slot_artifact_identity")
    staging = task / "staging"
    empty = task / "empty-state"
    for name in ("config", "data"):
        (empty / name).mkdir(mode=0o700, parents=True)
    raw_config = canonical_template(staging / "slots/old/state/data/graph", configuration)
    (empty / "config/settings.yaml").write_text(yaml.safe_dump(raw_config, sort_keys=False))
    assert prepare_slot(archive, staging, "old", empty) == manifest
    assert activate_slot(staging, "old") is None
    runtimes = {name: (staging / "slots" / name / "runtime").resolve()
        for name in ("old", "candidate")}
    roots = {name: runtime.parent / "state/data/graph" for name, runtime in runtimes.items()}
    roots["old"].mkdir(mode=0o700)
    artifact = compiled._artifact_hashes(runtimes["old"], manifest)
    compiled._export_file(archive, evidence / "runtime.tar.gz")
    compiled._export_file(runtimes["old"] / "release.json", evidence / "release.json")
    island = task / "fixture-island"
    closure, island_files = compiled.copy_island(repo, island)
    entry = repo / CHILD
    if not entry.is_file() or entry.is_symlink():
        raise ValueError("slot_child_missing_or_redirected")
    island_files[CHILD] = _sha(entry)
    (island / CHILD).write_bytes(entry.read_bytes())
    managed = "tests/mocks/managed_slot_graph.py"
    (island / managed).write_bytes((repo / managed).read_bytes())
    island_files[managed] = _sha(island / managed)
    bridge_archive = task / "admission-runtime.tar.gz"
    bridge_manifest = build_runtime(repo, bridge_archive, revision=bridge_revision)
    if bridge_manifest["git_commit"] != bridge_revision:
        raise compiled.CompiledRestoreFailure("slot_bridge_commit_mismatch")
    bridge_stage = task / "admission-staging"
    prepare_slot(bridge_archive, bridge_stage, "bridge", empty)
    bridge_runtime = bridge_stage / "slots/bridge/runtime"
    for relative in ("release.json", "src/core/runtime_admission.pyc",
                     "src/memory/runtime_closure.pyc", "scripts/stage_runtime.pyc"):
        target = island / "bridge" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((bridge_runtime / relative).read_bytes())
        island_files["bridge/" + relative] = _sha(target)
    compiled._export_file(bridge_archive, evidence / "admission-runtime.tar.gz")
    compiled.write_json(evidence / "admission-release.json", bridge_manifest)
    if len(island_files) != 25 or any(_sha(island / name) != digest for name, digest in island_files.items()):
        raise compiled.CompiledRestoreFailure("slot_island_hash_inventory")
    compiled.write_json(evidence / "fixture-closure.json", closure)
    exchange = task / "exchange"
    exchange.mkdir(mode=0o700)
    checkpoint = exchange / "seed-state.json"
    controls = _control_pair(repo, runtimes, roots, island, island_files, evidence, artifact, revision)
    return {"archive": archive, "manifest": manifest, "staging": staging, "runtimes": runtimes,
        "roots": roots, "artifact": artifact, "island": island, "island_files": island_files,
        "controls": controls, "checkpoint": checkpoint, "helper_modules": len(closure),
        "bridge_runtime": bridge_runtime, "bridge_manifest": bridge_manifest,
        "bridge_archive": bridge_archive, "bridge_archive_sha256": _sha(bridge_archive)}


def _exercise(prepared, proof, repo, evidence, deadline):
    from scripts.stage_runtime import AdmissionController

    with AdmissionController(prepared["staging"]) as controller:
        return _exercise_managed(prepared, proof, repo, evidence, deadline, controller)


def _exercise_managed(prepared, proof, repo, evidence, deadline, controller):
    from scripts.stage_runtime import prepare_slot, read_admission_head

    manifest, roots = prepared["manifest"], prepared["roots"]
    runtimes, checkpoint = prepared["runtimes"], prepared["checkpoint"]

    def run(mode, branch):
        proof["stage"] = "child:" + mode
        control, control_path = prepared["controls"][branch]
        result, receipt = _phase(mode, control=control, control_path=control_path,
            island=prepared["island"], checkpoint=checkpoint, evidence=evidence,
            deadline=deadline, manifest=manifest, controller=controller, slot=branch)
        proof["phases"][mode] = receipt
        return result

    seed = run("seed", "old")
    source_binding = read_admission_head(controller.root)["binding_id"]
    seed_sha = _sha(checkpoint)
    proof["stage"] = "offline_copy_preflight"
    old_before = offline_state(roots["old"])
    compiled.write_json(evidence / "old-before-copy.json", old_before)
    assert prepare_slot(prepared["archive"], prepared["staging"], "candidate",
        runtimes["old"].parent / "state") == manifest
    assert roots["candidate"].is_dir() and not roots["candidate"].samefile(roots["old"])
    candidate_artifact = compiled._artifact_hashes(runtimes["candidate"], manifest)
    assert candidate_artifact == prepared["artifact"]
    copied_before = offline_state(roots["candidate"])
    assert copied_before == old_before
    inventory = copied_targets(roots["candidate"], roots["old"])
    compiled.write_json(evidence / "copied-target-inventory-before-Graph.json", inventory)
    compiled.write_json(evidence / "candidate-before-Graph.json", copied_before)
    assert offline_state(roots["old"]) == old_before
    delegated = [row["id"] for row in inventory if row["absolute_paths"]]
    controller.authorize_target_delegation(roots["candidate"].parents[1], source_binding, delegated)
    assert controller.activate("candidate") == "slots/old/runtime"
    candidate = run("candidate", "candidate")
    assert _sha(checkpoint) == seed_sha
    old_after = offline_state(roots["old"])
    compiled.write_json(evidence / "old-after-candidate.json", old_after)
    compiled.write_json(evidence / "candidate-after-close.json", offline_state(roots["candidate"]))
    proof["stage"] = "stale_old_selection"
    rejected = None
    try:
        previous = controller.activate("old")
        assert previous == "slots/candidate/runtime"
        assert (prepared["staging"] / "current").resolve(strict=True) == runtimes["old"]
    except ValueError as error:
        rejected = {"type": type(error).__name__, "code": str(error)}
        if str(error) != "privacy_authority_stale":
            raise
        assert (prepared["staging"] / "current").resolve(strict=True) == runtimes["candidate"], (
            "stale_activation_mutated_selection_before_rejecting")
    old = run("inspect-old", "old")
    assert _sha(checkpoint) == seed_sha
    assert compiled._artifact_hashes(runtimes["old"], manifest) == prepared["artifact"]
    assert compiled._artifact_hashes(runtimes["candidate"], manifest) == candidate_artifact
    assert _sha(prepared["archive"]) == proof["archive_sha256"] == _sha(evidence / "runtime.tar.gz")
    assert all(_sha(prepared["island"] / name) == digest
        for name, digest in prepared["island_files"].items())
    assert all(_sha(repo / name) == digest for name, digest in prepared["island_files"].items()
        if not name.startswith("bridge/"))
    assert _sha(prepared["bridge_archive"]) == prepared["bridge_archive_sha256"]
    assert _sha(evidence / "admission-runtime.tar.gz") == prepared["bridge_archive_sha256"]
    for name in ("release.json", "src/core/runtime_admission.pyc",
                 "src/memory/runtime_closure.pyc", "scripts/stage_runtime.pyc"):
        assert _sha(prepared["bridge_runtime"] / name) == prepared["island_files"]["bridge/" + name]
    proof.update(execution_complete=True, all_actual_child_waits_returned_zero=True,
        original_state_logically_unchanged=old_after == old_before,
        stale_activation_rejection=rejected, observations={"seed": seed["observations"],
            "candidate": candidate["observations"], "inspect-old": old["observations"]},
        artifact_and_fixture_inputs_unchanged=True,
        offline_inventory_preceded_candidate_initialization=True)
    if time.monotonic() > deadline:
        raise compiled.CompiledRestoreFailure("slot_parent_deadline_exceeded")
    proof["stage"] = "privacy_oracle"
    _privacy_oracle(proof)
    proof["privacy_contract_passed"] = True


def exercise_slot_rollback(tmp_path: Path) -> dict:
    """Exercise copied state through fresh compiled owners and native closes."""
    started = time.monotonic()
    deadline = started + PARENT_SECONDS
    repo = Path(__file__).resolve().parents[2]
    task = Path(tmp_path).resolve()
    if task.is_relative_to(repo) or repo.is_relative_to(task):
        raise ValueError("slot_task_must_be_outside_checkout")
    task.mkdir(mode=0o700, parents=True, exist_ok=True)
    requested = os.environ.get("KZA_SLOT_ROLLBACK_REVISION", REVISION)
    if re.fullmatch("[0-9a-f]{40}", requested) is None:
        raise ValueError("slot_revision_must_be_exact_commit")
    revision = subprocess.check_output(["git", "rev-parse", "--verify", requested + "^{commit}"],
        cwd=repo, text=True).strip()
    selected = os.environ.get(EVIDENCE_ENV)
    evidence = task / "evidence" if selected is None else Path(selected).absolute() / ("attempt-" + uuid4().hex)
    evidence.mkdir(mode=0o700, parents=True)
    evidence = evidence.resolve()
    proof = {"schema_version": 1, "unit": "T024-F1/copied-slot-characterization",
        "scope": "Same compiled runtime, two copied physical states; not versioned rollback acceptance",
        "execution_complete": False, "privacy_contract_passed": False, "source_commit": revision,
        "parent_budget_s": PARENT_SECONDS, "phase_cap_s": PHASE_SECONDS,
        "cleanup_reserve_s": CLEANUP_SECONDS, "phases": {}, "evidence_root": str(evidence), "stage": "setup"}
    primary = None
    try:
        if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13) or sys.prefix == sys.base_prefix:
            raise ValueError("slot_fixture_requires_cpython313_venv")
        # Keep the older application/data regression, with the current
        # production security bridge and its independently pinned provenance.
        bridge_revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=repo, text=True).strip()
        prepared = _prepare(task, evidence, repo, revision, bridge_revision=bridge_revision)
        proof.update(archive_sha256=_sha(prepared["archive"]), release_sha256=prepared["artifact"]["release.json"],
            bridge_source_commit=prepared["bridge_manifest"]["git_commit"],
            bridge_archive_sha256=prepared["bridge_archive_sha256"],
            helper_modules=prepared["helper_modules"], island_files=len(prepared["island_files"]),
            original_roots={name: str(root) for name, root in prepared["roots"].items()},
            script_hashes={name: _sha(repo / name) for name in ("scripts/build_runtime.py", "scripts/stage_runtime.py")})
        _exercise(prepared, proof, repo, evidence, deadline)
        return proof
    except BaseException as error:
        primary = error
        proof.update(error_kind=type(error).__name__, error_stage=proof["stage"])
        raise
    finally:
        proof.update(elapsed_s=time.monotonic() - started, remaining_s=max(0, deadline - time.monotonic()))
        try:
            compiled.write_json(evidence / "parent-proof.json", proof)
        except BaseException as error:
            if primary is not None:
                raise BaseExceptionGroup("slot parent and evidence", [primary, error]) from None
            raise
