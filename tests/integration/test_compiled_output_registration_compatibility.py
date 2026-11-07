"""Real v1 and v2 graph histories remain readable under their original authority."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from uuid import uuid4

from tests.mocks import compiled_restore as compiled
from tests.mocks import slot_rollback as slots


OLD_REVISION = "ef8bc6db29f9eb3685edcf32b3a1aacfa8896d54"
PRODUCT_REFERENCE = "014b422b0a1585ffd810c01577b2e57e6a9a6604"
CHILD = "tests/mocks/output_registration_compatibility_child.py"
PARENT_SECONDS = 240


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(root: Path) -> dict:
    result = {}
    for path in sorted(root.rglob("*")):
        assert not path.is_symlink(), ("redirected_fixture_state", path)
        result[path.relative_to(root).as_posix()] = "directory" if path.is_dir() else _sha(path)
    return result


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _control(prepared: dict, branch: str, manifest: dict, evidence: Path) -> tuple[dict, Path]:
    # _prepare's initial controls describe the old release for both slots. Make
    # fresh controls with the exact new island closure and each actual artifact.
    value = dict(prepared["controls"][branch][0])
    value.update(island_files=dict(prepared["island_files"]),
                 source_commit=manifest["git_commit"],
                 release_sha256=_sha(prepared["runtimes"][branch] / "release.json"))
    path = evidence / ("compatibility-" + branch + "-control.json")
    compiled.write_json(path, value)
    return value, path


def _command(mode: str, prepared: dict, control: dict, control_path: Path,
             evidence: Path, expectation: Path | None = None) -> list[str]:
    command = [control["executable_literal"], "-I", "-S", "-B", str(prepared["island"] / CHILD),
               "--mode", mode, "--control", str(control_path), "--runtime", control["runtime_root"],
               "--state-root", control["state_root"], "--evidence-root", str(evidence),
               "--admission-root", str(prepared["staging"])]
    if expectation is not None:
        command.extend(["--expectation", str(expectation)])
    return command


def _result(evidence: Path, mode: str, pid: int, control: dict, manifest: dict) -> dict:
    result = json.loads((evidence / (mode + "-result.json")).read_text())
    assert result["schema_version"] == 1 and result["mode"] == mode and result["pid"] == pid
    assert result["source_commit"] == control["source_commit"] == manifest["git_commit"]
    assert result["release_sha256"] == control["release_sha256"]
    assert result["runtime_root"] == control["runtime_root"]
    assert result["state_root"] == control["state_root"]
    assert result["sys_flags"] == dict.fromkeys(("isolated", "no_site", "no_user_site", "ignore_environment"), 1)
    assert result["dont_write_bytecode"] is True
    assert {"src.main", "src.application", "src.memory.output_registration"} <= set(result["module_origins"])
    for value in result["module_origins"].values():
        assert value["loader"] == "SourcelessFileLoader" and value["phase"] == mode
        assert manifest["files"][value["relative_origin"]] == value["sha256"]
    writer = result["module_origins"]["src.memory.output_registration"]
    assert writer["relative_origin"] == "src/memory/output_registration.pyc"
    if mode == "graph-v1":
        assert "src.memory.output_registration_v2" not in result["module_origins"]
    else:
        assert "src.memory.output_registration_v2" in result["module_origins"]
    if mode.startswith("read-"):
        observed = result["observations"]
        assert observed["controller_module"] == str(Path(control["runtime_root"]) / "scripts/stage_runtime.pyc")
        assert observed["controller_sha256"] == manifest["files"]["scripts/stage_runtime.pyc"]
        assert observed["controller_loader"] == "SourcelessFileLoader"
    return result


def _graph_phase(controller: object, prepared: dict, branch: str, mode: str,
                 control_pair: tuple, manifest: dict, evidence: Path, deadline: float) -> dict:
    from scripts.stage_runtime import read_admission_head, verify_closed_binding

    control, control_path = control_pair
    budget = min(55, deadline - time.monotonic() - 20)
    assert budget > 0
    command = _command(mode, prepared, control, control_path, evidence)
    record = {"command": command, "budget_s": budget, "actual_wait_returned": False,
              "completed": False, "containment_only": False}
    try:
        with (evidence / (mode + "-stdout.log")).open("x") as out, (evidence / (mode + "-stderr.log")).open("x") as err:
            completed = controller.run(branch, command, timeout=budget, env=compiled._environment(), stdout=out, stderr=err)
        record.update(actual_wait_returned=True, exit_code=completed.returncode,
                      wait_owner="AdmissionController.run communicate + Popen.wait")
        assert completed.returncode == 0
        head = read_admission_head(controller.root)
        verify_closed_binding(controller.root, head["binding_id"])
        closure_path = controller.root / "admission/closures" / (head["binding_id"] + ".json")
        closure = json.loads(closure_path.read_text())
        result = _result(evidence, mode, closure["child_pid"], control, manifest)
        observed = result["observations"]
        running = closure["running_head"]
        assert all(observed["header"][key] == running[key] for key in ("run_id", "binding_id", "generation"))
        assert running["status"] == "running"
        assert closure["receipt_sha256"] == observed["receipt_sha256"]
        assert observed["receipt"]["child_pid"] == closure["child_pid"]
        assert observed["factory"] == "src.main.build_application" and observed["bridge_supplied_writer"] is False
        record.update(actual_wait_returned=True, wait_owner="AdmissionController.run communicate + Popen.wait",
                      exit_code=completed.returncode, child_pid=closure["child_pid"], completed=True,
                      closed_head=head, closure=closure, closure_sha256=_sha(closure_path),
                      result_sha256=_sha(evidence / (mode + "-result.json")))
        return {"result": result, "terminal": record}
    except BaseException as error:
        record.update(error_kind=type(error).__name__, containment_only=not record["actual_wait_returned"])
        raise
    finally:
        compiled.write_json(evidence / (mode + "-parent-receipt.json"), record)


def _reader_phase(prepared: dict, mode: str, control_pair: tuple, manifest: dict,
                  evidence: Path, expectation: Path, deadline: float) -> dict:
    control, control_path = control_pair
    command = _command(mode, prepared, control, control_path, evidence, expectation)
    budget = min(25, deadline - time.monotonic() - 20)
    assert budget > 0
    environment = compiled._environment()
    assert "KZA_RUNTIME_ADMISSION" not in environment
    record = {"command": command, "budget_s": budget, "waits": [], "signals": [],
              "actual_wait_returned": False, "containment_only": False, "completed": False}
    process, primary = None, None
    cleanup = []
    try:
        with (evidence / (mode + "-stdout.log")).open("x") as out, (evidence / (mode + "-stderr.log")).open("x") as err:
            process = subprocess.Popen(command, cwd=control["runtime_root"], env=environment,
                                       stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
            record.update(pid=process.pid, owned_pgid=process.pid)
            code = compiled._wait(process, budget, record)
            record["exit_code"] = code
            assert code == 0
            result = _result(evidence, mode, process.pid, control, manifest)
            record.update(completed=True, result_sha256=_sha(evidence / (mode + "-result.json")))
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
        try:
            compiled.write_json(evidence / (mode + "-parent-receipt.json"), record)
        except BaseException as error:
            cleanup.append(error)
    if primary is not None and cleanup:
        raise BaseExceptionGroup("compatibility reader and cleanup", [primary, *cleanup])
    if primary is not None:
        raise primary
    if cleanup:
        raise BaseExceptionGroup("compatibility reader cleanup", cleanup)
    return {"result": result, "terminal": record}


def _exercise(task: Path, evidence: Path, repo: Path, revision: str) -> dict:
    from scripts.stage_runtime import AdmissionController, prepare_slot, read_admission_head

    started = time.monotonic()
    deadline = started + PARENT_SECONDS
    assert not task.is_relative_to(repo) and not repo.is_relative_to(task)
    proof = {"schema_version": 1, "unit": "T024-F1/real-mixed-v1-v2-output-registration",
             "complete": False, "source_revision": revision, "old_revision": OLD_REVISION,
             "product_reference": PRODUCT_REFERENCE, "parent_budget_s": PARENT_SECONDS,
             "global_inventory_complete": False, "aggregate_copy_complete": False,
             "cleanup_ack": False, "F1_delivered": False, "goal_complete": False,
             "phases": {}, "scope": "Two real managed graph closures and current compiled closed-controller mixed reader"}
    primary = None
    try:
        before_tree = _git(repo, "ls-tree", "-r", PRODUCT_REFERENCE, "--", "src", "scripts")
        current_tree = _git(repo, "ls-tree", "-r", revision, "--", "src", "scripts")
        proof["product_mapping"] = {
            "reference_revision": PRODUCT_REFERENCE, "runtime_revision": revision,
            "reference_src_scripts_tree_sha256": hashlib.sha256(before_tree.encode()).hexdigest(),
            "current_src_scripts_tree_sha256": hashlib.sha256(current_tree.encode()).hexdigest(),
            "src_scripts_equal": before_tree == current_tree,
            "scope": "Observed equality for this validation; future legitimate product changes still run the behavioral regression"}
        prepared = slots._prepare(task, evidence, repo, OLD_REVISION, bridge_revision=revision)
        entry = repo / CHILD
        assert entry.is_file() and not entry.is_symlink()
        (prepared["island"] / CHILD).write_bytes(entry.read_bytes())
        prepared["island_files"][CHILD] = _sha(entry)
        assert len(prepared["island_files"]) == 26
        compiled.write_json(evidence / "compatibility-island.json", prepared["island_files"])
        old_control = _control(prepared, "old", prepared["manifest"], evidence)
        proof["archives"] = {
            "old": {"revision": OLD_REVISION, "sha256": _sha(prepared["archive"]),
                    "release_sha256": old_control[0]["release_sha256"]},
            "current": {"revision": revision, "sha256": _sha(prepared["bridge_archive"]),
                        "release_sha256": _sha(prepared["bridge_runtime"] / "release.json")}}
        proof["build_calls"] = 2
        with AdmissionController(prepared["staging"]) as controller:
            first = _graph_phase(controller, prepared, "old", "graph-v1", old_control,
                                 prepared["manifest"], evidence, deadline)
            proof["phases"]["graph-v1"] = first
            v1_root = prepared["staging"] / "admission/output-inventory/v1"
            v1_before = _tree(v1_root)
            old_state = prepared["runtimes"]["old"].parent / "state"
            state_before = _tree(old_state)
            source_binding = read_admission_head(controller.root)["binding_id"]
            # The old graph's real closed state is copied through the supported
            # preparation and delegation APIs. No receipt or history is copied.
            assert prepare_slot(prepared["bridge_archive"], prepared["staging"], "candidate", old_state) == prepared["bridge_manifest"]
            new_state = prepared["runtimes"]["candidate"].parent / "state"
            assert new_state != old_state and not new_state.samefile(old_state)
            targets = slots.copied_targets(prepared["roots"]["candidate"], prepared["roots"]["old"])
            delegated = [target["id"] for target in targets if target["absolute_paths"]]
            assert delegated and len(delegated) == len(set(delegated))
            compiled.write_json(evidence / "copied-targets-before-current-graph.json", targets)
            delegation = controller.authorize_target_delegation(new_state, source_binding, delegated)
            proof["delegation"] = {"id": delegation, "source_closed_binding": source_binding,
                                   "destination_state": str(new_state), "target_ids": delegated}
            current_control = _control(prepared, "candidate", prepared["bridge_manifest"], evidence)
            second = _graph_phase(controller, prepared, "candidate", "graph-v2", current_control,
                                  prepared["bridge_manifest"], evidence, deadline)
            proof["phases"]["graph-v2"] = second
            assert _tree(v1_root) == v1_before and _tree(old_state) == state_before
            assert first["result"]["observations"]["header"]["run_id"] != second["result"]["observations"]["header"]["run_id"]
            head = read_admission_head(controller.root)
            expectation = {"head": head, "runs": [phase["result"]["observations"] for phase in (first, second)],
                           "malformed_run_id": "0" * 32}
            assert all(value["header"]["run_id"] != expectation["malformed_run_id"] for value in expectation["runs"])
            expectation_path = evidence / "reader-expectation.json"
            compiled.write_json(expectation_path, expectation)
        # Acquire a new actual compiled controller in a fresh process only after
        # the parent's original lease has closed; no inherited run ticket.
        authority_before = _tree(prepared["staging"] / "admission")
        closed_states = {branch: _tree(runtime.parent / "state") for branch, runtime in prepared["runtimes"].items()}
        positive = _reader_phase(prepared, "read-valid", current_control, prepared["bridge_manifest"],
                                 evidence, expectation_path, deadline)
        proof["phases"]["read-valid"] = positive
        assert _tree(prepared["staging"] / "admission") == authority_before
        assert _tree(v1_root) == v1_before
        namespace = prepared["staging"] / "admission/output-inventory"
        # These are explicitly malformed data in this fresh fixture. They carry
        # no fabricated header, binding, receipt, delegation or capability.
        (namespace / "v999").mkdir(mode=0o700)
        (namespace / "v2/runs" / expectation["malformed_run_id"]).mkdir(mode=0o700)
        injected = _tree(prepared["staging"] / "admission")
        assert set(injected) - set(authority_before) == {
            "output-inventory/v999", "output-inventory/v2/runs/" + expectation["malformed_run_id"]}
        assert all(injected[key] == value for key, value in authority_before.items())
        compiled.write_json(evidence / "negative-injection.json", {
            "unknown_version": "v999", "malformed_v2_run": expectation["malformed_run_id"],
            "original_entries_unchanged": True, "receipt_or_authority_fabricated": False})
        negative = _reader_phase(prepared, "read-invalid", current_control, prepared["bridge_manifest"],
                                 evidence, expectation_path, deadline)
        proof["phases"]["read-invalid"] = negative
        assert _tree(prepared["staging"] / "admission") == injected
        assert _tree(v1_root) == v1_before
        for branch, runtime in prepared["runtimes"].items():
            assert _tree(runtime.parent / "state") == closed_states[branch]
            manifest = prepared["manifest"] if branch == "old" else prepared["bridge_manifest"]
            compiled._artifact_hashes(runtime, manifest)
        assert all(_sha(prepared["island"] / name) == digest for name, digest in prepared["island_files"].items())
        assert _sha(prepared["archive"]) == proof["archives"]["old"]["sha256"]
        assert _sha(prepared["bridge_archive"]) == proof["archives"]["current"]["sha256"]
        assert time.monotonic() <= deadline
        proof.update(complete=True, real_v1_read_only_compatibility=True, real_v2_history_observed=True,
                     original_v1_bytes_unchanged=True, closed_authority_unchanged_by_readers=True,
                     unknown_and_malformed_fail_closed=True, actual_child_waits=4,
                     scope_counts={"inventory_categories": 32, "instrumented_families": 7,
                                   "closed_primary_registration_families": ["lists", "reminders"],
                                   "partial_families": 5, "uncovered_categories": 30})
        return proof
    except BaseException as error:
        primary = error
        proof["error_kind"] = type(error).__name__
        raise
    finally:
        proof["elapsed_s"] = time.monotonic() - started
        try:
            compiled.write_json(evidence / "mixed-compatibility-parent-proof.json", proof)
        except BaseException as error:
            if primary is not None:
                raise BaseExceptionGroup("compatibility and evidence", [primary, error]) from None
            raise


def test_compiled_v1_v2_histories_share_original_closed_controller(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    revision = _git(repo, "rev-parse", "HEAD")
    assert re.fullmatch("[0-9a-f]{40}", revision)
    requested = os.environ.get("KZA_RECONCILIATION_REVISION")
    assert requested is None or requested == revision
    selected = os.environ.get("KZA_OUTPUT_COMPATIBILITY_EVIDENCE_ROOT")
    evidence = (tmp_path / "evidence" if selected is None else Path(selected).absolute() / ("attempt-" + uuid4().hex))
    evidence.mkdir(mode=0o700, parents=True)
    proof = _exercise(tmp_path.resolve(), evidence.resolve(), repo, revision)
    assert proof["complete"] is True
    assert proof["real_v1_read_only_compatibility"] and proof["unknown_and_malformed_fail_closed"]
