"""Full/live metadata boundaries and real writer probes.

The metadata cases stub admission provenance and do not issue controller close
receipts or claim native graph acceptance. G01 exercises the real compiled
controller and delegated history. The flock case uses the original probe and
real operating-system locks in both threads and a separate process.
"""
import copy
from contextlib import closing
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import sqlite3
import subprocess
import sys
import threading

import pytest

from scripts import stage_runtime as stage


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stage._canonical_json(value) + b"\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ticket(root):
    # This is an explicit unit metadata seam, never inherited authority.
    return {"schema_version": 1, "admission_root": str(root), "binding_id": "a" * 64,
            "run_id": "b" * 32, "generation": 1, "controller_pid": os.getppid(),
            "lease_fd": 0, "lease_identity": {"device": 0, "inode": 0}}


@pytest.fixture
def local_metadata(tmp_path, monkeypatch):
    root, state = tmp_path / "root", tmp_path / "root/state"
    state.mkdir(parents=True)
    with closing(sqlite3.connect(state / "targets.db")) as db:
        db.execute("CREATE TABLE erasure_targets (id TEXT, participant TEXT, version INTEGER, descriptor TEXT)")
        db.commit()
    binding = {"state_root": str(state), "ownership_sha256": None, "slot": "unit"}
    ticket = _ticket(root)

    def core(actual_state, actual_ticket):
        assert Path(actual_state) == state and actual_ticket == ticket
        return root, state, copy.deepcopy(binding)

    monkeypatch.setattr(stage, "_admission_ticket_core", core)
    return root, state, binding, ticket


@pytest.mark.parametrize("check_state", [False, True])
def test_default_full_validation_keeps_complete_inventory_scan(local_metadata, monkeypatch, check_state):
    root, state, binding, ticket = local_metadata
    inventory = stage._target_inventory
    calls = []

    def observed(path):
        calls.append(("inventory", path))
        return inventory(path)

    def binding_snapshot(actual_root, slot):
        assert actual_root == root and slot == "unit"
        calls.append(("binding", slot))
        return copy.deepcopy(binding)

    monkeypatch.setattr(stage, "_target_inventory", observed)
    monkeypatch.setattr(stage, "_binding_snapshot", binding_snapshot)
    assert stage.validate_admission_ticket(state, ticket, check_state=check_state) == binding
    assert calls == ([("binding", "unit")] if check_state else []) + [("inventory", state)]


def test_snapshot_uses_complete_inventory_once_then_live_never_scans(local_metadata, monkeypatch):
    _, state, binding, ticket = local_metadata
    inventory = stage._target_inventory
    calls = []

    def observed(path):
        calls.append(path)
        return inventory(path)

    monkeypatch.setattr(stage, "_target_inventory", observed)
    snapshot = stage.admission_validation_snapshot(state, ticket)
    assert calls == [state]
    assert json.loads(snapshot.ownership_json)["targets"] == {}

    def forbidden(*args, **kwargs):
        raise AssertionError("live metadata attempted a database scan, probe, or scan mutex")

    class NoScanMutex:
        __enter__ = forbidden
        __exit__ = forbidden

    for name in ("_target_inventory", "_database_image", "_binding_snapshot", "_assert_writers_closed"):
        monkeypatch.setattr(stage, name, forbidden)
    monkeypatch.setattr(stage, "_ADMISSION_VALIDATION_LOCK", NoScanMutex())
    assert stage.validate_admission_live(state, ticket, snapshot) == binding
    with pytest.raises(AttributeError):
        snapshot.ownership_json = b"{}"
    with pytest.raises(TypeError):
        snapshot[2] = b"{}"
    decoded = json.loads(snapshot.ownership_json)
    decoded["policy"] = {"untrusted": True}
    assert json.loads(snapshot.ownership_json)["policy"] is None


@pytest.fixture
def delegated_metadata(tmp_path, monkeypatch):
    """Real metadata files; controller/closure provenance is a deliberate stub."""
    root = tmp_path / "root"
    state = root / "slots/current/state"
    source_state, runtime = root / "slots/source/state", root / "slots/source/runtime"
    for directory in (state, source_state / "data", source_state / "config", runtime):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = {"git_commit": "c" * 40, "python_magic": importlib.util.MAGIC_NUMBER.hex()}
    release_sha = _write(runtime / "release.json", manifest)
    source = {"schema_version": 1, "slot": "source", "runtime_root": str(runtime),
              "state_root": str(source_state), "release_sha256": release_sha,
              "code_commit": manifest["git_commit"], "python_magic_hex": manifest["python_magic"],
              "state_identity": stage._identity(source_state),
              "store_roots": {name: stage._identity(source_state / name) for name in ("data", "config")},
              "schemas": {}, "sdk_versions": {}, "authority": {}, "ownership_sha256": None}
    assert set(source) == stage._BINDING_KEYS
    source_id = stage._digest(source)
    _write(root / "admission/bindings" / (source_id + ".json"), source)
    marker = {"schema_version": 1, "admission_root": str(root),
              "binding_id": source_id, "state_root": str(source_state)}
    _write(source_state / stage._MARKER, marker)
    target = {"id": "events:unit", "participant": "events", "version": 1,
              "descriptor": {"db_path": str(source_state / "data/events.db")}}
    # No receipt/closure file is forged. Only the metadata comparison seam is
    # supplied here; the real closure implementation is exercised by G01.
    proof = {"unit_metadata_observation": "not a controller receipt"}
    record = {"schema_version": 1, "destination_state": str(state),
              "destination_identity": stage._identity(state), "source_binding": source_id,
              "source_close_sha256": stage._digest(proof), "targets": {target["id"]: target}}
    record_path = root / "admission/delegations" / (stage._digest(record) + ".json")
    _write(record_path, record)
    archive = {"schema_version": 1, "source_binding": source_id, "targets": [target]}
    raw_archive = stage._canonical_json(archive) + b"\n"
    archive_sha = hashlib.sha256(raw_archive).hexdigest()
    archive_path = root / "admission/history" / (archive_sha + ".json")
    _write(archive_path, archive)
    policy = {"schema_version": 1, "source_bindings": [source_id], "owned_targets": [],
              "delegated_targets": [target["id"]], "historical_targets": [
                  {"source_binding": source_id, "target_id": target["id"], "archive_sha256": archive_sha}]}
    policy_path = state / ".kza-target-ownership.json"
    binding = {"state_root": str(state), "ownership_sha256": _write(policy_path, policy)}
    ticket = _ticket(root)
    selected = {"record": record, "source": source, "marker": marker, "proof": proof}
    ownership = {"targets": {target["id"]: target}, "policy": policy,
                 "delegations": {target["id"]: selected}, "delegation_records": [record]}
    scan_calls = []

    def full(actual_root, actual_binding, *, allow_new_local, capture):
        assert actual_root == root and actual_binding == binding and allow_new_local and capture
        scan_calls.append("synthetic full-validation observation")
        return copy.deepcopy(ownership)

    def core(actual_state, actual_ticket):
        assert Path(actual_state) == state and actual_ticket == ticket
        return root, state, copy.deepcopy(binding)

    def checked_slot(actual_root, slot):
        assert actual_root == root and slot == "source"
        return runtime, source_state, copy.deepcopy(manifest)

    def close_observation(actual_root, binding_id):
        assert actual_root == root and binding_id == source_id
        return copy.deepcopy(proof)

    monkeypatch.setattr(stage, "_admission_ticket_core", core)
    monkeypatch.setattr(stage, "_target_ownership_validation", full)
    monkeypatch.setattr(stage, "_checked_slot", checked_slot)
    monkeypatch.setattr(stage, "_closure_proof", close_observation)
    snapshot = stage.admission_validation_snapshot(state, ticket)
    assert scan_calls == ["synthetic full-validation observation"]
    return {"root": root, "state": state, "source_state": source_state, "runtime": runtime,
            "ticket": ticket, "snapshot": snapshot, "policy_path": policy_path,
            "record_path": record_path, "archive_path": archive_path, "proof": proof}


@pytest.mark.parametrize("mutation,error", [
    ("policy", "runtime_admission_ownership_changed"),
    ("delegation", "runtime_admission_delegation_changed"),
    ("marker", "runtime_admission_marker_mismatch"),
    ("store_root", "runtime_admission_store_replaced"),
    ("release", "runtime_admission_release_changed"),
    ("source_policy", "runtime_admission_ownership_changed"),
    ("closure", "runtime_admission_source_close_proof"),
    ("history", "runtime_admission_history_mismatch"),
])
def test_live_metadata_rejects_changed_pinned_observation(delegated_metadata, monkeypatch, mutation, error):
    value = delegated_metadata

    def forbidden(*args, **kwargs):
        raise AssertionError("live path attempted a full scan or writer probe")

    for name in ("_target_ownership_validation", "_target_inventory", "_database_image",
                 "_binding_snapshot", "_assert_writers_closed"):
        monkeypatch.setattr(stage, name, forbidden)
    if mutation == "policy":
        path = value["policy_path"]
        policy = json.loads(path.read_bytes())
        policy["owned_targets"].append("unexpected")
        _write(path, policy)
    elif mutation == "delegation":
        path = value["record_path"]
        record = json.loads(path.read_bytes())
        record["source_close_sha256"] = "0" * 64
        _write(path, record)
    elif mutation == "marker":
        _write(value["source_state"] / stage._MARKER, {"binding_id": "0" * 64})
    elif mutation == "store_root":
        directory = value["source_state"] / "data"
        directory.rename(value["source_state"] / "replaced-data")
        directory.mkdir()
    elif mutation == "release":
        path = value["runtime"] / "release.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "source_policy":
        _write(value["source_state"] / ".kza-target-ownership.json", {"unexpected": True})
    elif mutation == "closure":
        value["proof"]["unit_metadata_observation"] = "changed"
    elif mutation == "history":
        path = value["archive_path"]
        archive = json.loads(path.read_bytes())
        archive["targets"] = []
        _write(path, archive)
    else:
        raise AssertionError(mutation)
    with pytest.raises(ValueError, match="^" + error + "$"):
        stage.validate_admission_live(value["state"], value["ticket"], value["snapshot"])


def test_original_writer_probes_serialize_and_still_reject_external_process(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    lock_path = state / "actual.lock"
    lock_path.write_bytes(b"")
    original_flock = fcntl.flock
    original_mutex = stage._ADMISSION_VALIDATION_LOCK
    acquired, release, attempted = threading.Event(), threading.Event(), threading.Event()
    errors, probes = [], []

    class ObservedMutex:
        def __enter__(self):
            if threading.current_thread().name == "validator-B":
                attempted.set()
            original_mutex.acquire()
            return self

        def __exit__(self, *args):
            original_mutex.release()

    def observe(fd, operation):
        actor = threading.current_thread().name
        original_flock(fd, operation)
        info = os.fstat(fd)
        probes.append((actor, fd, info.st_dev, info.st_ino))
        if actor == "validator-A":
            acquired.set()
            assert release.wait(3), "first real lock was not released"

    def probe():
        try:
            stage._assert_writers_closed(state)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(stage, "_ADMISSION_VALIDATION_LOCK", ObservedMutex())
    monkeypatch.setattr(stage.fcntl, "flock", observe)
    first = threading.Thread(target=probe, name="validator-A", daemon=True)
    second = threading.Thread(target=probe, name="validator-B", daemon=True)
    first.start()
    try:
        assert acquired.wait(3)
        second.start()
        assert attempted.wait(3)
        assert [item[0] for item in probes] == ["validator-A"]
        assert first.is_alive() and second.is_alive()
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive() and not errors
    assert [item[0] for item in probes] == ["validator-A", "validator-B"]
    assert probes[0][2:] == probes[1][2:] == (lock_path.stat().st_dev, lock_path.stat().st_ino)
    for _, fd, _, _ in probes:
        with pytest.raises(OSError):
            os.fstat(fd)

    child = "\n".join([
        "import fcntl,json,os,sys",
        "fd=os.open(sys.argv[1],os.O_RDWR|os.O_NOFOLLOW)",
        "try:",
        " fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)",
        " info=os.fstat(fd)",
        " os.write(1,(json.dumps([os.getpid(),info.st_dev,info.st_ino])+'\\n').encode())",
        " os.read(0,1)",
        "finally: os.close(fd)",
    ])
    process = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", child, str(lock_path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(3), "external real writer did not acquire the lock"
        identity = json.loads(os.read(process.stdout.fileno(), 4096))
        assert identity == [process.pid, lock_path.stat().st_dev, lock_path.stat().st_ino]
        assert process.poll() is None
        with pytest.raises(ValueError, match="^runtime_admission_writer_active$"):
            stage._assert_writers_closed(state)
        assert process.poll() is None
        process.stdin.write(b"x")
        process.stdin.flush()
        process.stdin.close()
        assert process.wait(5) == 0
        stage._assert_writers_closed(state)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(5)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
