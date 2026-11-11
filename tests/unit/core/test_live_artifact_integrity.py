"""Real built artifacts and native dispatch; inherited authority is a unit seam.

The supported admission_archive/admission_root fixtures build and prepare the
actual runtime. The ticket's controller/FD provenance and source close proof are
explicitly stubbed here; G01 remains the real compiled authority oracle. Code,
physical roots, SQLite target inventory, delegation metadata and all byte/stat
checks use their original implementations.
"""
from __future__ import annotations

from contextlib import closing
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading

import pytest

from scripts import stage_runtime as stage
from src.core import runtime_admission
from src.core.owned_work import OwnedBlockingWork, current_native_execution
from tests.unit.tools.test_stage_runtime_admission import admission_archive, admission_root


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = stage._canonical_json(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _registry(state: Path, target: dict) -> None:
    with closing(sqlite3.connect(state / "data/registry.db")) as database:
        database.execute("CREATE TABLE erasure_targets (id TEXT, participant TEXT, version INTEGER, descriptor TEXT)")
        database.execute("INSERT INTO erasure_targets VALUES (?,?,?,?)", (
            target["id"], target["participant"], target["version"], json.dumps(target["descriptor"])))
        database.commit()


@pytest.fixture
def artifact_admission(admission_root, admission_archive, monkeypatch, request):
    root = admission_root
    state = root / "slots/candidate/state"
    runtime = root / "slots/candidate/runtime"
    (state / ".env").write_text("UNIT_ARTIFACT_OBSERVATION=ordinary\n")
    (runtime / ".env").symlink_to("../state/.env")
    inspected = runtime
    if getattr(request, "param", "current") == "source":
        stage.prepare_slot(admission_archive, root, "source", state)
        source_state = root / "slots/source/state"
        inspected = root / "slots/source/runtime"
        descriptor = {"path": str(source_state / "data/events.db"), "table": "events_policy_v1"}
        encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
        target = {"id": hashlib.sha256(json.dumps(["events", 1, encoded]).encode()).hexdigest(),
                  "participant": "events", "version": 1, "descriptor": descriptor}
        _registry(source_state, target)
        source = stage._binding_snapshot(root, "source")
        source_id = stage._digest(source)
        _write(root / "admission/bindings" / (source_id + ".json"), source)
        _write(source_state / stage._MARKER, {"schema_version": 1, "admission_root": str(root),
               "binding_id": source_id, "state_root": str(source_state)})
        # Not a controller receipt; its running head names the source's own binding so the
        # U3b-0b owner check (the closure belongs to a run of the owner's slot) applies.
        proof = {"unit_close_observation": "not a controller receipt",
                 "running_head": {"schema_version": 1, "generation": 1, "binding_id": source_id,
                                  "status": "running", "run_id": "c" * 32}}

        def closure(actual_root, identity):
            assert actual_root == root and identity == source_id
            return copy.deepcopy(proof)

        monkeypatch.setattr(stage, "_closure_proof", closure)
        record = {"schema_version": 1, "destination_state": str(state),
                  "destination_identity": stage._identity(state), "source_binding": source_id,
                  "source_close_sha256": stage._digest(proof), "targets": {target["id"]: target}}
        _write(root / "admission/delegations" / (stage._digest(record) + ".json"), record)
        # U3b-0b: every honored record carries its provenance; a one-hop chain is [owner].
        hop = {"binding_id": source_id, "close_sha256": stage._digest(proof), "closure": proof,
               "state_root": str(source_state), "state_identity": stage._identity(source_state), "via": None}
        _write(stage._chain_path(root, stage._digest(record)), {
            "schema_version": 1, "record_sha256": stage._digest(record), "destination_state": str(state),
            "destination_identity": stage._identity(state),
            "grantor": {"binding_id": source_id, "close_sha256": stage._digest(proof)},
            "chains": {target["id"]: [hop]}})
        _registry(state, target)
        _write(state / ".kza-target-ownership.json", {"schema_version": 1,
               "source_bindings": [source_id], "owned_targets": [],
               "delegated_targets": [target["id"]], "historical_targets": []})
    # Empty code directories are legal in the original full checker. Their
    # mutation must be visible even though they add no manifest payload leaf.
    (inspected / "src/integrity-empty/nested").mkdir(parents=True)
    binding = stage._binding_snapshot(root, "candidate")
    ticket = {"schema_version": 1, "admission_root": str(root), "binding_id": "a" * 64,
              "run_id": "b" * 32, "generation": 1, "controller_pid": os.getppid(),
              "lease_fd": 0, "lease_identity": {"device": 0, "inode": 0}}

    def metadata(actual_state, actual_ticket):
        assert Path(actual_state) == state and actual_ticket == ticket
        stage._check_roots(binding)
        return root, state, copy.deepcopy(binding)

    monkeypatch.setattr(stage, "_admission_ticket_metadata", metadata)
    return root, state, runtime, inspected, binding, ticket


def _payload(runtime: Path) -> Path:
    manifest = json.loads((runtime / "release.json").read_text())
    return runtime / next(name for name in sorted(manifest["files"]) if name.endswith(".pyc"))


@pytest.mark.parametrize("artifact_admission", ["current", "source"], indirect=True)
@pytest.mark.parametrize("mutation", [
    "same_size_restored_mtime", "replace_inode", "leaf_symlink", "remove_leaf",
    "add_leaf", "empty_directory_addition", "parent_redirect", "mode", "hardlink",
    "state_link_replace", "environment_link_replace", "release_bytes",
])
def test_live_rejects_real_code_or_link_mutation(artifact_admission, mutation, monkeypatch, tmp_path):
    root, state, runtime, inspected, binding, ticket = artifact_admission
    snapshot = stage.admission_validation_snapshot(state, ticket)
    path = _payload(inspected)
    before = path.stat()
    if mutation == "same_size_restored_mtime":
        value = path.read_bytes()
        path.write_bytes(bytes([value[0] ^ 1]) + value[1:])
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert path.stat().st_size == before.st_size and path.stat().st_mtime_ns == before.st_mtime_ns
        assert path.stat().st_ctime_ns != before.st_ctime_ns
    elif mutation == "replace_inode":
        replacement = tmp_path / "same-bytes"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif mutation == "leaf_symlink":
        replacement = tmp_path / "redirected-payload"
        path.replace(replacement)
        path.symlink_to(replacement)
    elif mutation == "remove_leaf":
        path.unlink()
    elif mutation == "add_leaf":
        (path.parent / "unlisted.pyc").write_bytes(b"extra")
    elif mutation == "empty_directory_addition":
        (inspected / "src/integrity-empty/nested/extra").mkdir()
    elif mutation == "parent_redirect":
        parent = inspected.parent
        moved = parent.with_name(parent.name + "-moved")
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
    elif mutation == "mode":
        path.chmod(before.st_mode ^ 0o100)
    elif mutation == "hardlink":
        os.link(path, tmp_path / "extra-hardlink")
    elif mutation == "state_link_replace":
        link = inspected / "data"
        target = os.readlink(link)
        link.unlink()
        link.symlink_to(target, target_is_directory=True)
    elif mutation == "environment_link_replace":
        link = inspected / ".env"
        target = os.readlink(link)
        link.unlink()
        link.symlink_to(target)
    elif mutation == "release_bytes":
        release = inspected / "release.json"
        release.write_bytes(release.read_bytes() + b" ")

    def forbidden(*args, **kwargs):
        raise AssertionError("live code change attempted a fresh full validation")

    monkeypatch.setattr(stage, "_checked_slot", forbidden)
    monkeypatch.setattr(stage, "_binding_snapshot", forbidden)
    monkeypatch.setattr(stage, "_target_inventory", forbidden)
    with pytest.raises((ValueError, OSError)):
        stage.validate_admission_live(state, ticket, snapshot)


@pytest.mark.parametrize("artifact_admission", ["current", "source"], indirect=True)
def test_live_allows_real_mutable_state_writes_without_payload_reads(artifact_admission, monkeypatch, tmp_path):
    root, state, runtime, inspected, binding, ticket = artifact_admission
    snapshot = stage.admission_validation_snapshot(state, ticket)
    original = stage._artifact_read
    reads = []

    def release_only(actual_runtime, relative, observation):
        reads.append((actual_runtime, relative))
        assert relative == "release.json"
        return original(actual_runtime, relative, observation)

    monkeypatch.setattr(stage, "_artifact_read", release_only)
    for destination in {runtime, inspected}:
        with closing(sqlite3.connect(destination / "data/ordinary.db")) as database:
            database.execute("CREATE TABLE memories (value TEXT)")
            database.execute("INSERT INTO memories VALUES ('ordinary payload')")
            database.commit()
        # The fingerprint concerns code, not mutable configuration contents.
        # Original configuration-authority validation remains a separate path.
        (destination / "config/ordinary.txt").write_text("allowed state payload\n")
        (destination / "config/ordinary.txt").write_text("changed state payload\n")
    (tmp_path / "unrelated-sibling").mkdir()
    for _ in range(3):
        assert stage.validate_admission_live(state, ticket, snapshot) == binding
    assert reads and all(relative == "release.json" for _, relative in reads)
    assert snapshot.artifacts_json == stage._canonical_json(json.loads(snapshot.artifacts_json))


@pytest.mark.parametrize("artifact_admission", ["current", "source"], indirect=True)
def test_full_capture_rejects_mutation_after_real_payload_read(artifact_admission, monkeypatch):
    _, state, _, inspected, _, ticket = artifact_admission
    relative = str(_payload(inspected).relative_to(inspected))
    original = stage._artifact_read
    changed = []

    def mutate_after_read(runtime, filename, observation):
        value = original(runtime, filename, observation)
        if runtime == inspected and filename == relative and not changed:
            path = runtime / filename
            info = path.stat()
            path.write_bytes(bytes([value[0] ^ 1]) + value[1:])
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
            changed.append(path)
        return value

    monkeypatch.setattr(stage, "_artifact_read", mutate_after_read)
    with pytest.raises(ValueError, match="runtime_admission_artifact_changed"):
        stage.admission_validation_snapshot(state, ticket)
    assert changed == [inspected / relative]


def test_full_capture_fifo_swap_rejects_without_waiting_and_closes_fds(artifact_admission, monkeypatch):
    _, state, _, inspected, _, ticket = artifact_admission
    leaf = _payload(inspected)
    original_open, original_close = os.open, os.close
    parent_identity = stage._identity(leaf.parent)
    swapped, done = threading.Event(), threading.Event()
    opened, closed, failures = [], [], []

    def observe_open(*args, **kwargs):
        # Inject only at the actual leaf open, after the original helper has
        # checked all parent metadata. A wrapper around _artifact_read would
        # change parent times too early and fail before exercising this open.
        directory = kwargs.get("dir_fd")
        if (threading.current_thread() is worker and args[0] == leaf.name
                and directory is not None and not args[1] & os.O_DIRECTORY
                and not swapped.is_set()):
            info = os.fstat(directory)
            assert {"device": info.st_dev, "inode": info.st_ino} == parent_identity
            leaf.unlink()
            os.mkfifo(leaf, 0o600)
            swapped.set()
        descriptor = original_open(*args, **kwargs)
        if threading.current_thread() is worker:
            opened.append(descriptor)
        return descriptor

    def observe_close(descriptor):
        if threading.current_thread() is worker:
            closed.append(descriptor)
        return original_close(descriptor)

    def capture():
        try:
            stage.admission_validation_snapshot(state, ticket)
        except BaseException as error:
            failures.append(error)
        finally:
            done.set()

    worker = threading.Thread(target=capture)
    monkeypatch.setattr(os, "open", observe_open)
    monkeypatch.setattr(os, "close", observe_close)
    worker.start()
    returned_before_unblock = False
    emergency = None
    try:
        assert swapped.wait(5)
        returned_before_unblock = done.wait(2)
    finally:
        if not done.is_set() and swapped.is_set():
            emergency = original_open(leaf, os.O_RDWR | os.O_NONBLOCK)
        worker.join(5)
        if emergency is not None:
            original_close(emergency)
    assert not worker.is_alive() and returned_before_unblock
    assert len(failures) == 1 and isinstance(failures[0], ValueError)
    assert str(failures[0]) == "runtime_admission_artifact_changed"
    assert sorted(opened) == sorted(closed)


@pytest.mark.asyncio
async def test_each_real_native_invocation_fully_reads_code_then_only_live_metadata(artifact_admission, monkeypatch):
    _, state, _, inspected, binding, ticket = artifact_admission
    admission = runtime_admission.RuntimeAdmission(ticket, binding, str(state / "data/privacy.db"), "unit")
    monkeypatch.setattr(runtime_admission, "_stage", lambda: stage)
    monkeypatch.setattr(runtime_admission, "_verified_admissions", {admission: True})
    original = stage._artifact_read
    reads, executions, snapshots = [], [], []

    def observe(runtime, relative, observation):
        if relative != "release.json":
            reads.append((runtime, relative))
        return original(runtime, relative, observation)

    monkeypatch.setattr(stage, "_artifact_read", observe)
    work = OwnedBlockingWork()

    def original_callback():
        execution = current_native_execution()
        executions.append(execution)
        snapshots.append(admission._native_validations[execution])
        count = len(reads)
        assert count > 0
        for _ in range(4):
            admission._revalidate_native(execution)
        assert len(reads) == count
        return count

    work._install_native_preflight(admission, admission._preflight_native)
    try:
        first = await work.run(original_callback)
        second = await work.run(original_callback)
        assert second == first * 2
        assert executions[0] is not executions[1] and snapshots[0] is not snapshots[1]
        # A fresh invocation still reads bytes even if no live fingerprint was
        # consulted after an in-place corruption.
        path = _payload(inspected)
        value, info = path.read_bytes(), path.stat()
        path.write_bytes(bytes([value[0] ^ 1]) + value[1:])
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        with pytest.raises(ValueError):
            await work.run(original_callback)
        assert len(executions) == 2 and len(reads) > second
    finally:
        await work.aclose()
    assert work.status()["closed"] and work.status()["operations"] == 0
