"""Parent-only build and process receipts for the compiled restore rehearsal.

The parent reads fixture bytes and JSON metadata. Only fresh isolated children
import the compiled application or reconstruct its DTOs.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
from uuid import uuid4


REVISION_ENV = "KZA_COMPILED_RESTORE_REVISION"
EVIDENCE_ENV = "KZA_COMPILED_RESTORE_EVIDENCE_ROOT"
PARENT_SECONDS = 240
PHASE_SECONDS = 100
CLEANUP_SECONDS = 15
SEED_MODULES = (
    "tests.mocks.application_restore", "tests.mocks.restore_application_graph",
    "tests.mocks.restore_preservation",
)
CHILD = "tests/mocks/compiled_restore_child.py"
GUARDS = {("diagnostics", 1), ("streams", 2), ("voices", 2)}


class CompiledRestoreFailure(AssertionError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _export_file(source, target):
    with Path(source).open("rb") as original, Path(target).open("xb") as copied:
        shutil.copyfileobj(original, copied)
    Path(target).chmod(0o600)
    if digest(source) != digest(target):
        raise CompiledRestoreFailure("evidence_copy_mismatch")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x") as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def _module_file(repo, name):
    base = repo.joinpath(*name.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(repo / "tests"):
                raise CompiledRestoreFailure("fixture_source_outside_tests")
            return path
    return None


def fixture_closure(repo):
    """Resolve only static tests.* imports; never import/collect a fixture."""
    pending, result = list(SEED_MODULES), {}
    while pending:
        name = pending.pop()
        if name in result:
            continue
        path = _module_file(repo, name)
        if path is None:
            raise CompiledRestoreFailure("fixture_module_missing:" + name)
        relative = str(path.relative_to(repo))
        if path.name == "conftest.py" or not relative.startswith("tests/"):
            raise CompiledRestoreFailure("fixture_closure_forbidden")
        tree = ast.parse(path.read_bytes(), filename=relative)
        dependencies = set()
        for count in range(1, len(name.split("."))):
            dependencies.add(".".join(name.split(".")[:count]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.update(alias.name for alias in node.names
                    if alias.name == "tests" or alias.name.startswith("tests."))
            elif isinstance(node, ast.ImportFrom) and not node.level:
                module = node.module or ""
                if module == "tests" or module.startswith("tests."):
                    dependencies.add(module)
                    dependencies.update(module + "." + alias.name for alias in node.names
                        if _module_file(repo, module + "." + alias.name) is not None)
            elif isinstance(node, ast.ImportFrom) and node.level:
                raise CompiledRestoreFailure("fixture_relative_import_needs_review:" + relative)
        result[name] = {"path": relative, "sha256": digest(path),
            "dependencies": sorted(dependencies)}
        pending.extend(dependencies - result.keys())
    return dict(sorted(result.items()))


def copy_island(repo, destination):
    closure = fixture_closure(repo)
    if len(closure) != 18:
        raise CompiledRestoreFailure("fixture_closure_changed_needs_review")
    records = {item["path"]: item["sha256"] for item in closure.values()}
    entry = repo / CHILD
    if not entry.is_file() or entry.is_symlink():
        raise CompiledRestoreFailure("compiled_child_missing")
    records[CHILD] = digest(entry)
    destination.mkdir(mode=0o700)
    for relative, expected in records.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = (repo / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise CompiledRestoreFailure("fixture_changed_during_copy")
        target.write_bytes(raw)
        if digest(target) != expected:
            raise CompiledRestoreFailure("fixture_copy_mismatch")
    assert not (destination / "src").exists()
    assert not list(destination.rglob("conftest.py"))
    return closure, records


def _revision(repo):
    requested = os.environ.get(REVISION_ENV)
    if requested is not None and re.fullmatch(r"[0-9a-f]{40}", requested) is None:
        raise CompiledRestoreFailure("invalid_compiled_restore_revision")
    selected = requested if requested is not None else "HEAD"
    commit = subprocess.check_output(["git", "rev-parse", "--verify", selected + "^{commit}"],
        cwd=repo, text=True).strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise CompiledRestoreFailure("invalid_resolved_revision")
    return requested, commit


def _environment():
    names = ("PATH", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT")
    environment = {name: os.environ[name] for name in names if name in os.environ}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", ANONYMIZED_TELEMETRY="False",
        TOKENIZERS_PARALLELISM="false", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    return environment


def _artifact_hashes(runtime, manifest):
    result = {}
    for relative, expected in manifest["files"].items():
        path = runtime / relative
        if path.is_symlink() or path.resolve() != path or path.suffix == ".py":
            raise CompiledRestoreFailure("artifact_file_not_source_free")
        result[relative] = digest(path)
        if result[relative] != expected:
            raise CompiledRestoreFailure("artifact_hash_mismatch")
    result["release.json"] = digest(runtime / "release.json")
    actual = {str(path.relative_to(runtime)) for path in runtime.rglob("*")
        if path.is_file() or path.is_symlink()}
    if actual != set(manifest["files"]) | {"release.json", "config", "data"}:
        raise CompiledRestoreFailure("unexpected_artifact_file")
    return result


def _slot_targets(staging, runtime):
    result = {}
    for name, link, expected in (
        ("current", staging / "current", runtime),
        ("config", runtime / "config", runtime.parent / "state/config"),
        ("data", runtime / "data", runtime.parent / "state/data"),
    ):
        target = link.resolve(strict=True)
        if not link.is_symlink() or target != expected or not target.is_dir():
            raise CompiledRestoreFailure("slot_state_target_changed:" + name)
        info = target.stat()
        result[name] = {"target": str(target), "link": os.readlink(link),
            "device": info.st_dev, "inode": info.st_ino}
    return result


def _wait(proc, timeout, record):
    item = {"timeout_s": timeout, "entered": True, "returned": False}
    record["waits"].append(item)
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        item["timed_out"] = True
        raise
    item.update(returned=True, exit_code=code)
    record["actual_wait_returned"] = True
    return code


def _stop_owned(proc, record):
    """Contain only this admitted process group, then obtain a real wait."""
    end = time.monotonic() + CLEANUP_SECONDS
    if proc.poll() is not None:
        _wait(proc, max(.1, end - time.monotonic()), record)
        return
    for number, grace in ((signal.SIGTERM, 2), (signal.SIGKILL, 10)):
        item = {"pgid": proc.pid, "signal": int(number), "sent": False}
        record["signals"].append(item)
        try:
            os.killpg(proc.pid, number)
            item["sent"] = True
        except ProcessLookupError:
            item["process_lookup_error"] = True
        try:
            _wait(proc, min(grace, max(.1, end - time.monotonic())), record)
            return
        except subprocess.TimeoutExpired:
            continue
    raise CompiledRestoreFailure("owned_child_exit_unconfirmed")


def _validate_result(path, mode, proc, control, manifest, checkpoint):
    result = json.loads(path.read_text())
    if type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise CompiledRestoreFailure("child_result_schema")
    for name in ("accepted_within_scope", "writer_lease_released", "sdk_clients_closed",
        "native_gates_returned", "zero_HA_service_calls"):
        if result.get(name) is not True:
            raise CompiledRestoreFailure("child_result_not_confirmed:" + name)
    expected = {"mode": mode, "source_commit": control["source_commit"],
        "runtime_root": control["runtime_root"], "state_root": control["state_root"], "pid": proc.pid}
    if any(result.get(key) != value for key, value in expected.items()):
        raise CompiledRestoreFailure("child_result_identity_mismatch")
    if not checkpoint.is_file() or result.get("checkpoint_sha256") != digest(checkpoint):
        raise CompiledRestoreFailure("child_checkpoint_mismatch")
    guards = result.get("three_guards")
    if not isinstance(guards, list) or len(guards) != 3 or {
        (row["participant"], row["version"]) for row in guards} != GUARDS:
        raise CompiledRestoreFailure("child_unresolved_guards_changed")
    oracles = result.get("domain_oracles")
    if not isinstance(oracles, dict) or not oracles or any(value is not True for value in oracles.values()):
        raise CompiledRestoreFailure("child_domain_oracle_missing")
    flags = result.get("sys_flags", {})
    if any(flags.get(name) != 1 for name in (
        "isolated", "ignore_environment", "no_user_site", "no_site", "dont_write_bytecode")):
        raise CompiledRestoreFailure("child_not_isolated")
    origins = result.get("module_origins")
    required = {"src", "src.main", "src.application", "src.memory.privacy_services",
        "src.memory.erasure_coordinator", "src.memory.producer_erasure"}
    if not isinstance(origins, dict) or not required <= origins.keys():
        raise CompiledRestoreFailure("child_origin_inventory_missing")
    for name, entry in origins.items():
        relative = entry["relative_origin"]
        path = Path(relative)
        if (not (name == "src" or name.startswith("src.")) or path.is_absolute()
            or ".." in path.parts or path.suffix != ".pyc" or not relative.startswith("src/")
            or manifest["files"].get(relative) != entry["sha256"]
            or digest(Path(control["runtime_root"]) / relative) != entry["sha256"]):
            raise CompiledRestoreFailure("child_source_origin_mismatch")
    return result


def _run_child(mode, *, control_path, control, island, checkpoint, evidence, deadline, manifest):
    available = deadline - time.monotonic() - CLEANUP_SECONDS
    if available <= 0:
        raise CompiledRestoreFailure("parent_deadline_before_child")
    budget = min(PHASE_SECONDS, available)
    result_path = evidence / (mode + "-result.json")
    if result_path.exists():
        raise CompiledRestoreFailure("child_result_already_exists")
    command = [control["executable_literal"], "-I", "-S", "-B", str(island / CHILD),
        "--mode", mode, "--control", str(control_path), "--runtime", control["runtime_root"],
        "--state-root", control["state_root"], "--evidence-root", str(evidence),
        "--checkpoint", str(checkpoint)]
    environment = _environment()
    record = {"command": command, "environment_names": sorted(environment), "budget_s": budget,
        "waits": [], "signals": [], "actual_wait_returned": False, "containment_only": False}
    proc, primary, result = None, None, None
    cleanup_errors = []
    with (evidence / (mode + "-stdout.log")).open("x") as out, (
        evidence / (mode + "-stderr.log")).open("x") as err:
        try:
            proc = subprocess.Popen(command, cwd=control["runtime_root"], env=environment,
                stdout=out, stderr=err, start_new_session=True)
            record.update(pid=proc.pid, owned_pgid=proc.pid,
                group_basis="Popen(start_new_session=True), known PGID equals admitted PID")
            code = _wait(proc, budget, record)
            if code != 0:
                raise CompiledRestoreFailure("compiled_child_exit_nonzero:" + mode)
            result = _validate_result(result_path, mode, proc, control, manifest, checkpoint)
        except BaseException as error:
            primary = error
        finally:
            if proc is not None and not record["actual_wait_returned"]:
                record["containment_only"] = True
                try:
                    _stop_owned(proc, record)
                except BaseException as error:
                    cleanup_errors.append(error)
            record["accepted_within_scope"] = primary is None and not cleanup_errors
            if primary is not None:
                record["error_kind"] = type(primary).__name__
            record["cleanup_error_kinds"] = [type(error).__name__ for error in cleanup_errors]
            if result is not None:
                record.update(result_sha256=digest(result_path), module_count=len(result["module_origins"]),
                    checkpoint_sha256=result["checkpoint_sha256"], domain_oracles=result["domain_oracles"])
            try:
                write_json(evidence / (mode + "-parent-receipt.json"), record)
            except BaseException as error:
                cleanup_errors.append(error)
    if primary is not None and cleanup_errors:
        raise BaseExceptionGroup("compiled child observation and cleanup", [primary, *cleanup_errors])
    if primary is not None:
        raise primary
    if cleanup_errors:
        raise BaseExceptionGroup("compiled child cleanup", cleanup_errors)
    return record


def exercise_compiled_restore(tmp_path):
    """Build one candidate and run two independent, source-free generations."""
    started = time.monotonic()
    deadline = started + PARENT_SECONDS
    repo = Path(__file__).resolve().parents[2]
    task = Path(tmp_path).resolve()
    if task == repo or task.is_relative_to(repo) or repo.is_relative_to(task):
        raise CompiledRestoreFailure("compiled_task_must_be_outside_checkout")
    task.mkdir(mode=0o700, parents=True, exist_ok=True)
    selected_evidence = os.environ.get(EVIDENCE_ENV)
    if selected_evidence is None:
        evidence = task / "evidence"
    else:
        evidence = Path(selected_evidence).absolute() / ("attempt-" + uuid4().hex)
    evidence.mkdir(mode=0o700, parents=True)
    evidence = evidence.resolve()
    proof = {"schema_version": 1, "scope": "local compiled public graph restore; all criteria open",
        "accepted_within_scope": False, "parent_budget_s": PARENT_SECONDS, "phase_cap_s": PHASE_SECONDS,
        "cleanup_reserve_s": CLEANUP_SECONDS, "evidence_root": str(evidence), "phases": {}}
    primary = None
    try:
        if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13):
            raise CompiledRestoreFailure("compiled_restore_requires_cpython313")
        if sys.prefix == sys.base_prefix:
            raise CompiledRestoreFailure("compiled_restore_requires_native_venv")
        # These existing checkout scripts are parent-only. No src module or
        # copied fixture is imported to run either graph in this process.
        from scripts.build_runtime import build_runtime
        from scripts.stage_runtime import activate_slot, prepare_slot

        requested, revision = _revision(repo)
        proof.update(requested_revision=requested, source_commit=revision,
            script_hashes={str(path.relative_to(repo)): digest(path) for path in (
                repo / "scripts/build_runtime.py", repo / "scripts/stage_runtime.py")})
        archive = task / "runtime.tar.gz"
        manifest = build_runtime(repo, archive, revision=revision)
        if (manifest["git_commit"] != revision
            or manifest["python_magic"] != importlib.util.MAGIC_NUMBER.hex()
            or manifest.get("format") != "kza-cpython-bytecode-v1"
            or manifest.get("python") != "3.13"
            or manifest.get("entrypoint") != ["python", "-m", "src.main"]
            or manifest.get("external_paths") != ["config", "data", "models", "logs"]):
            raise CompiledRestoreFailure("built_candidate_identity_mismatch")
        empty = task / "empty-state"
        (empty / "config").mkdir(mode=0o700, parents=True)
        (empty / "data").mkdir(mode=0o700)
        staging = task / "staging"
        assert prepare_slot(archive, staging, "exact", empty) == manifest
        assert activate_slot(staging, "exact") is None
        runtime = (staging / "slots/exact/runtime").resolve()
        assert (staging / "current").resolve() == runtime
        state = (runtime.parent / "state/data/graph").resolve()
        state.mkdir(mode=0o700)
        assert (runtime / "data").resolve() == runtime.parent / "state/data"
        assert (runtime / "config").resolve() == runtime.parent / "state/config"
        slot_before = _slot_targets(staging, runtime)
        proof["slot_targets_before"] = slot_before
        before = _artifact_hashes(runtime, manifest)
        write_json(evidence / "artifact-before.json", before)
        _export_file(archive, evidence / "runtime.tar.gz")
        _export_file(runtime / "release.json", evidence / "release.json")
        island = task / "fixture-island"
        closure, island_files = copy_island(repo, island)
        write_json(evidence / "fixture-closure.json", closure)
        exchange = task / "exchange"
        exchange.mkdir(mode=0o700)
        checkpoint = exchange / "g1-state.json"
        dependency_roots = sorted({str(Path(sysconfig.get_path(name)).resolve())
            for name in ("purelib", "platlib")})
        prefix = Path(sys.prefix).resolve()
        if any(not Path(path).is_dir() or not Path(path).is_relative_to(prefix) for path in dependency_roots):
            raise CompiledRestoreFailure("dependency_root_outside_native_venv")
        control = {"schema_version": 1, "executable_literal": os.path.abspath(sys.executable),
            "parent_prefix": str(prefix), "base_prefix": str(Path(sys.base_prefix).resolve()),
            "python_version": list(sys.version_info[:3]), "cache_tag": sys.implementation.cache_tag,
            "python_magic_hex": importlib.util.MAGIC_NUMBER.hex(), "dependency_roots": dependency_roots,
            "checkout_root": str(repo), "runtime_root": str(runtime),
            "release_sha256": before["release.json"], "source_commit": revision,
            "island_root": str(island), "island_files": island_files,
            "state_root": str(state), "evidence_root": str(evidence)}
        control_path = evidence / "parent-control.json"
        write_json(control_path, control)
        proof.update(control_sha256=digest(control_path), archive_sha256=digest(archive),
            release_sha256=before["release.json"], helper_modules=len(closure),
            island_files=len(island_files), runtime_root=str(runtime), state_root=str(state))
        for mode in ("generate", "replay"):
            proof["phases"][mode] = _run_child(mode, control_path=control_path, control=control,
                island=island, checkpoint=checkpoint, evidence=evidence,
                deadline=deadline, manifest=manifest)
        assert proof["phases"]["generate"]["checkpoint_sha256"] == proof["phases"]["replay"]["checkpoint_sha256"]
        after = _artifact_hashes(runtime, manifest)
        slot_after = _slot_targets(staging, runtime)
        proof["slot_targets_after"] = slot_after
        assert slot_after == slot_before
        write_json(evidence / "artifact-after.json", after)
        assert after == before
        assert digest(archive) == proof["archive_sha256"]
        assert digest(evidence / "runtime.tar.gz") == proof["archive_sha256"]
        assert digest(evidence / "release.json") == proof["release_sha256"]
        assert digest(control_path) == proof["control_sha256"]
        assert all(digest(island / relative) == expected and digest(repo / relative) == expected
            for relative, expected in island_files.items())
        if time.monotonic() > deadline:
            raise CompiledRestoreFailure("compiled_parent_deadline_exceeded")
        proof.update(accepted_within_scope=True, artifact_unchanged=True,
            fixture_and_child_inputs_unchanged=True, actual_child_waits_returned=True,
            slot_state_targets_unchanged=True)
        return proof
    except BaseException as error:
        primary = error
        proof["error_kind"] = type(error).__name__
        raise
    finally:
        proof["elapsed_s"] = time.monotonic() - started
        proof["remaining_s"] = max(0, deadline - time.monotonic())
        try:
            write_json(evidence / "parent-proof.json", proof)
        except BaseException as error:
            if primary is not None:
                raise BaseExceptionGroup("compiled parent and evidence write", [primary, error]) from None
            raise
