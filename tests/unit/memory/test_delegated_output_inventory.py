"""Decoder contracts with synthetic authority metadata, never real run receipts.

The original compiled G01 scenario supplies genuine controller/closure lineage.
Here only the controller lease is a seam: record hashes, closed source receipt
validation, journal parsing and public fail-closed classification remain real.
"""
from copy import deepcopy
import errno
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import threading

import pytest

from scripts import stage_runtime as stage
from src.memory import copy_catalog
from src.memory import output_registration as base
from src.memory import output_registration_v2 as registration


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stage._canonical_json(value) + b"\n")


def _binding(state):
    value = dict.fromkeys(stage._BINDING_KEYS)
    value.update(schema_version=1, state_root=str(state), state_identity=stage._identity(state),
                 authority={"projection": {"configuration": {"config/settings.yaml": "b" * 64}}})
    return value


def _target(path):
    descriptor = {"path": str(path), "table": "events_policy_v1"}
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    identity = hashlib.sha256(json.dumps(["events", 1, encoded]).encode()).hexdigest()
    return {"id": identity, "participant": "events", "version": 1, "descriptor": descriptor}


def _metadata_bytes(decoder):
    return {str(path): path.read_bytes()
        for root in (decoder["root"], decoder["state"], decoder["old"])
        for path in root.rglob("*") if path.is_file()}


def _assert_unknown(result):
    assert result["phase"] == "unknown" and result["reason"] == "invalid_or_unfinished_history"
    assert not {"header", "stores", "journal", "configuration", "material_observations"} & result.keys()
    assert result["capability"] is False and result["global_inventory_complete"] is False
    assert result["aggregate_copy_complete"] is False


def _journal(directory, header, event, policy_events=(), *, policy_after_target=False):
    for path in directory.glob("*-*.json"):
        path.unlink()
    raw_header = base._encoded(header)
    (directory / "epoch.json").write_bytes(raw_header)
    previous = base._sha(raw_header)
    owner = {"kind": "owner", "owner": "events.retention", "native_identity": None,
             "plan": {"kind": "worker", "producer": "events", "role": "retention"}}
    observations = [event, *policy_events] if policy_after_target else [*policy_events, event]
    events = [dict(owner, phase="intent"), dict(owner, phase="attached"), *observations,
              dict(owner, phase="active"), dict(owner, phase="closed")]
    for revision, value in enumerate(events, 1):
        raw = base._encoded(dict(value, revision=revision, previous_sha256=previous))
        previous = base._sha(raw)
        (directory / f"{revision:012d}-{previous}.json").write_bytes(raw)
    (directory / "HEAD.json").write_bytes(base._encoded({"revision": revision, "sha256": previous}))


@pytest.fixture
def decoder(tmp_path, monkeypatch, request):
    tmp_path = tmp_path.resolve()  # Match the real controller's canonical roots.
    root, state, old = tmp_path / "controller", tmp_path / "current", tmp_path / "old"
    state.mkdir(); old.mkdir()
    source = _binding(old)
    source_id = stage._digest(source)
    _write(root / "admission/bindings" / f"{source_id}.json", source)
    _write(old / stage._MARKER, {"binding_id": source_id})
    source_run = "1" * 32
    source_head = {"schema_version": 1, "generation": 1, "binding_id": "c" * 64,
                   "status": "running", "run_id": source_run}
    source_receipt = {"schema_version": 1, "binding_id": source_head["binding_id"],
        "generation": 1, "run_id": source_run, "child_pid": 100,
        "status": "closed", "resources": dict.fromkeys(stage._CLOSED_RESOURCES, True)}
    receipt_path = root / "admission/receipts" / f"{source_run}.json"
    _write(receipt_path, source_receipt)
    proof = {"schema_version": 1, "binding_id": source_id, "running_head": source_head,
             "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(), "child_pid": 100}
    _write(root / "admission/closures" / f"{source_id}.json", proof)
    path = old / "events.db"
    path.touch()  # No SQLite access is necessary to inspect a physical plan.
    target = _target(path)
    record = {"schema_version": 1, "destination_state": str(state),
        "destination_identity": stage._identity(state), "source_binding": source_id,
        "source_close_sha256": stage._digest(proof), "targets": {target["id"]: target}}
    delegation_id = stage._digest(record)
    _write(root / "admission/delegations" / f"{delegation_id}.json", record)
    policy = {"schema_version": getattr(request, "param", 1), "source_bindings": [source_id], "owned_targets": [],
              "delegated_targets": [target["id"]], "historical_targets": []}
    policy_path = state / ".kza-target-ownership.json"
    _write(policy_path, policy)
    binding = _binding(state)
    binding["ownership_sha256"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    binding_id = stage._digest(binding)
    _write(root / "admission/bindings" / f"{binding_id}.json", binding)
    run = "2" * 32
    directory = root / "admission/output-inventory/v2/runs" / run
    directory.mkdir(parents=True)
    header = {"schema_version": 2, "run_id": run, "generation": 2, "binding_id": binding_id,
        "config_sha256": "b" * 64, "state_root": str(state), "state_identity": stage._identity(state),
        "namespace_identity": base._directory(directory), "producers": list(base.PRODUCERS),
        "selected": dict.fromkeys(("privacy_authority", "authority_sessions", "routines", "events", "suggestions"), True)}
    _write(root / "admission/receipts" / f"{run}.json", {"run_id": run, "binding_id": binding_id,
        "generation": 2, "status": "closed", "resources": dict.fromkeys(stage._CLOSED_RESOURCES, True)})
    scope = {"scope_schema_version": 1, "source_binding": source_id,
        "delegation_sha256": delegation_id, "source_close_sha256": stage._digest(proof),
        "source_state_root": str(old), "source_state_identity": stage._identity(old)}
    event = {"kind": "delegated_target", "owner": "events.retention", "id": target["id"],
        "status": "attached", "plan": registration._plan("events", "historical", path, old),
        "borrowed_reader": False, "scope": scope}
    policy_event = {"kind": "delegation_policy", "scope_schema_version": 1,
                    "ownership_sha256": binding["ownership_sha256"], "policy": deepcopy(policy)}
    _journal(directory, header, event, [policy_event])
    controller = object()
    head = {"schema_version": 1, "generation": 2, "binding_id": binding_id, "status": "closed", "run_id": run}
    def lease(actual):
        assert actual is controller
        return stage, root, head, None, None
    monkeypatch.setattr(copy_catalog, "_controller_lease", lease)
    def forbidden(*args, **kwargs):
        raise AssertionError("inventory decoder must not open a material database")
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    return {"controller": controller, "directory": directory, "header": header, "event": event,
            "root": root, "state": state, "old": old, "source_receipt": source_receipt,
            "source_receipt_path": receipt_path, "policy": policy, "policy_path": policy_path,
            "source": source, "source_id": source_id, "source_proof": proof,
            "delegation": record, "delegation_id": delegation_id, "policy_event": policy_event}


@pytest.mark.parametrize("form", ["delegated", "legacy_local"])
def test_closed_decoder_preserves_existing_local_form_and_joins_tagged_scope(decoder, form):
    event = deepcopy(decoder["event"])
    if form == "legacy_local":
        path = decoder["state"] / "events.db"
        path.touch()
        event.update(kind="target", id=_target(path)["id"],
                     plan=registration._plan("events", "historical", path, decoder["state"]))
        del event["scope"]
        _journal(decoder["directory"], decoder["header"], event)
    before = _metadata_bytes(decoder)
    result, = registration.read_inventory(decoder["controller"])
    assert result["phase"] == "observed" and result["capability"] is False
    assert result["global_inventory_complete"] is False and result["aggregate_copy_complete"] is False
    observed, = (value for value in result["material_observations"] if value["kind"] in {"target", "delegated_target"})
    assert observed["kind"] == event["kind"] and observed["plan"] == event["plan"]
    policies = [value for value in result["material_observations"] if value["kind"] == "delegation_policy"]
    assert len(policies) == (1 if form == "delegated" else 0)
    assert _metadata_bytes(decoder) == before


def test_closed_decoder_reads_legacy_v1_beside_tagged_v2_without_promotion(decoder):
    """Exercise both original parsers; these are synthetic decoder receipts."""
    run = "3" * 32
    directory = decoder["root"] / "admission/output-inventory/v1/runs" / run
    directory.mkdir(parents=True)
    header = {key: value for key, value in decoder["header"].items() if key != "selected"}
    header.update(schema_version=1, run_id=run, generation=1,
                  namespace_identity=base._identity(directory))
    raw = base._encoded(header)
    (directory / "epoch.json").write_bytes(raw)
    (directory / "HEAD.json").write_bytes(base._encoded({"revision": 0, "sha256": base._sha(raw)}))
    _write(decoder["root"] / "admission/receipts" / (run + ".json"), {
        "run_id": run, "binding_id": header["binding_id"], "generation": 1,
        "status": "closed", "resources": dict.fromkeys(stage._CLOSED_RESOURCES, True)})
    before = _metadata_bytes(decoder)
    records = registration.read_inventory(decoder["controller"])
    assert len(records) == 2
    by_version = {record["version"]: record for record in records}
    assert set(by_version) == {"v1", "v2"}
    legacy = by_version["v1"]
    assert legacy["phase"] == "observed" and legacy["header"]["schema_version"] == 1
    assert legacy["stores"] == {} and "material_observations" not in legacy
    tagged = by_version["v2"]
    assert tagged["phase"] == "observed" and tagged["header"]["schema_version"] == 2
    assert [event["kind"] for event in tagged["material_observations"]] == ["delegation_policy", "delegated_target"]
    assert all(record["capability"] is False and record["global_inventory_complete"] is False
               and record["aggregate_copy_complete"] is False for record in records)
    assert _metadata_bytes(decoder) == before


def test_closed_decoder_uses_captured_policy_after_current_owned_targets_change(decoder):
    """Model the real seal's policy update, without claiming a controller run."""
    changed = deepcopy(decoder["policy"])
    changed["owned_targets"] = ["e" * 64]
    _write(decoder["policy_path"], changed)
    assert base._sha(decoder["policy_path"].read_bytes()) != decoder["policy_event"]["ownership_sha256"]
    before = _metadata_bytes(decoder)
    result, = registration.read_inventory(decoder["controller"])
    assert result["phase"] == "observed" and result["capability"] is False
    assert result["global_inventory_complete"] is False and result["aggregate_copy_complete"] is False
    policy, target = result["material_observations"]
    assert policy["kind"] == "delegation_policy" and policy["policy"] == decoder["policy"]
    assert policy["ownership_sha256"] == decoder["policy_event"]["ownership_sha256"]
    assert target["kind"] == "delegated_target" and target["id"] == decoder["event"]["id"]
    assert _metadata_bytes(decoder) == before


@pytest.mark.parametrize("decoder", [True, 2], indirect=True, ids=["boolean", "unknown"])
def test_closed_decoder_refuses_policy_schema_even_with_matching_synthetic_binding(decoder):
    """Fixture hashes include the invalid version; rejection cannot be a stale hash."""
    assert base._sha(base._encoded(decoder["policy"])) == decoder["policy_event"]["ownership_sha256"]
    before = _metadata_bytes(decoder)
    result, = registration.read_inventory(decoder["controller"])
    _assert_unknown(result)
    assert _metadata_bytes(decoder) == before


@pytest.mark.parametrize("version", [True, 2], ids=["boolean", "unknown"])
def test_closed_decoder_refuses_policy_scope_version(decoder, version):
    policy = deepcopy(decoder["policy_event"])
    policy["scope_schema_version"] = version
    _journal(decoder["directory"], decoder["header"], decoder["event"], [policy])
    before = _metadata_bytes(decoder)
    result, = registration.read_inventory(decoder["controller"])
    _assert_unknown(result)
    assert _metadata_bytes(decoder) == before


@pytest.mark.parametrize("form", ["missing", "duplicate", "after_target"])
def test_closed_decoder_requires_one_policy_before_delegated_targets(decoder, form):
    policies = [] if form == "missing" else [deepcopy(decoder["policy_event"])]
    if form == "duplicate":
        policies.append(deepcopy(policies[0]))
    _journal(decoder["directory"], decoder["header"], decoder["event"], policies,
             policy_after_target=form == "after_target")
    before = _metadata_bytes(decoder)
    result, = registration.read_inventory(decoder["controller"])
    _assert_unknown(result)
    assert _metadata_bytes(decoder) == before


def test_closed_decoder_refuses_symlinked_metadata_parent_without_resolving_it(decoder):
    parent = decoder["root"] / "admission/delegations"
    moved = parent.with_name("original-delegations")
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    before = _metadata_bytes(decoder)
    result, = registration.read_inventory(decoder["controller"])
    _assert_unknown(result)
    assert parent.is_symlink() and os.readlink(parent) == str(moved)
    assert _metadata_bytes(decoder) == before


def test_closed_decoder_drops_all_projection_if_closure_changes_between_joins(decoder, monkeypatch):
    """Change a real metadata file after the first original join has returned."""
    original = registration._read_delegated_observation
    path = decoder["root"] / "admission/closures" / (decoder["source_id"] + ".json")
    changed = deepcopy(decoder["source_proof"])
    changed["child_pid"] += 1
    before = _metadata_bytes(decoder)
    calls = []

    def observe(*args):
        calls.append(args[3]["id"])
        result = original(*args)
        if len(calls) == 1:
            _write(path, changed)
        return result

    monkeypatch.setattr(registration, "_read_delegated_observation", observe)
    result, = registration.read_inventory(decoder["controller"])
    assert calls == [decoder["event"]["id"]] * 2
    assert result["phase"] == "unknown" and result["reason"] == "invalid_or_unfinished_history"
    assert result["capability"] is False and result["global_inventory_complete"] is False
    assert result["aggregate_copy_complete"] is False
    assert not {"header", "stores", "journal", "configuration", "material_observations"} & result.keys()
    # The hook's explicit metadata mutation is the only write; the reader must
    # not repair the receipt/closure or publish a partially recovered record.
    expected = dict(before)
    expected[str(path)] = stage._canonical_json(changed) + b"\n"
    assert _metadata_bytes(decoder) == expected


@pytest.mark.parametrize("mutation", [
    "unknown_scope_version", "boolean_scope_version", "extra_scope_field", "target_id", "sibling_path", "source_binding",
    "source_root", "source_identity", "delegation_digest", "closure_digest", "source_marker",
    "source_receipt", "ownership_policy", "untagged_external", "borrowed_reader_role",
    "delegation_record", "source_binding_record", "source_closure_record",
])
def test_closed_decoder_refuses_unbound_scope_without_positive_projection(decoder, mutation):
    event = deepcopy(decoder["event"])
    policy_event = deepcopy(decoder["policy_event"])
    scope = event["scope"]
    if mutation == "unknown_scope_version":
        scope["scope_schema_version"] = 2
    elif mutation == "boolean_scope_version":
        scope["scope_schema_version"] = True
    elif mutation == "extra_scope_field":
        scope["permit"] = True
    elif mutation == "target_id":
        event["id"] = "f" * 64
    elif mutation == "sibling_path":
        path = decoder["old"] / "sibling.db"
        path.touch()
        event["plan"] = registration._plan("events", "historical", path, decoder["old"])
    elif mutation in {"source_binding", "delegation_digest", "closure_digest"}:
        key = {"source_binding": "source_binding", "delegation_digest": "delegation_sha256",
               "closure_digest": "source_close_sha256"}[mutation]
        scope[key] = "f" * 64
    elif mutation == "source_root":
        scope["source_state_root"] = str(decoder["old"].parent)
    elif mutation == "source_identity":
        scope["source_state_identity"]["inode"] += 1
    elif mutation == "source_marker":
        _write(decoder["old"] / stage._MARKER, {"binding_id": "f" * 64})
    elif mutation == "source_receipt":
        receipt = deepcopy(decoder["source_receipt"])
        receipt["resources"]["native_closed"] = False
        _write(decoder["source_receipt_path"], receipt)
    elif mutation == "ownership_policy":
        # Internally consistent altered policy bytes still cannot replace the
        # historical policy hash bound by the original running header.
        policy_event["policy"]["owned_targets"] = ["e" * 64]
        policy_event["ownership_sha256"] = base._sha(base._encoded(policy_event["policy"]))
    elif mutation == "delegation_record":
        record = deepcopy(decoder["delegation"])
        record["targets"][event["id"]]["descriptor"]["path"] = str(decoder["old"] / "sibling.db")
        _write(decoder["root"] / "admission/delegations" / (decoder["delegation_id"] + ".json"), record)
    elif mutation == "source_binding_record":
        source = deepcopy(decoder["source"])
        source["state_identity"]["inode"] += 1
        _write(decoder["root"] / "admission/bindings" / (decoder["source_id"] + ".json"), source)
    elif mutation == "source_closure_record":
        proof = deepcopy(decoder["source_proof"])
        proof["child_pid"] += 1
        _write(decoder["root"] / "admission/closures" / (decoder["source_id"] + ".json"), proof)
    elif mutation == "untagged_external":
        event["kind"] = "target"
        del event["scope"]
    else:
        event["borrowed_reader"] = True
    # The journal itself remains hash-consistent. Rejection must come from the
    # semantic scope/authority join, not an unrelated broken hash chain.
    _journal(decoder["directory"], decoder["header"], event, [policy_event])
    before = _metadata_bytes(decoder)
    result, = registration.read_inventory(decoder["controller"])
    assert result["phase"] == "unknown" and result["capability"] is False
    assert result["reason"] == "invalid_or_unfinished_history"
    assert all(key not in result for key in ("header", "stores", "journal", "configuration", "material_observations"))
    assert result["global_inventory_complete"] is False and result["aggregate_copy_complete"] is False
    assert _metadata_bytes(decoder) == before


@pytest.mark.parametrize("record", [
    "delegation", "source_binding", "source_closure", "source_receipt", "source_marker",
    "delegation_at_open",
])
def test_closed_decoder_rejects_real_fifo_metadata_without_blocking_or_repair(decoder, monkeypatch, record):
    """A bounded emergency unblock prevents a regressed reader hanging pytest."""
    paths = {
        "delegation": decoder["root"] / "admission/delegations" / (decoder["delegation_id"] + ".json"),
        "source_binding": decoder["root"] / "admission/bindings" / (decoder["source_id"] + ".json"),
        "source_closure": decoder["root"] / "admission/closures" / (decoder["source_id"] + ".json"),
        "source_receipt": decoder["source_receipt_path"],
        "source_marker": decoder["old"] / stage._MARKER,
    }
    race = record == "delegation_at_open"
    path = paths["delegation" if race else record]
    parent_identity = path.parent.stat()
    fifo_identity = []
    if not race:
        path.unlink()
        os.mkfifo(path, 0o600)
        fifo_identity.append(path.lstat())
    before = _metadata_bytes(decoder)
    results, errors = [], []
    original_open, fifo_descriptors = os.open, []

    def observe_open(*args, **kwargs):
        parent = kwargs.get("dir_fd")
        if race and not fifo_identity and args[0] == path.name and parent is not None:
            observed_parent = os.fstat(parent)
            if (observed_parent.st_dev, observed_parent.st_ino) == (parent_identity.st_dev, parent_identity.st_ino):
                # Substitute the actual leaf only after the original reader has
                # traversed and checked its parents; preserve open arguments.
                path.unlink()
                os.mkfifo(path, 0o600)
                fifo_identity.append(path.lstat())
        descriptor = original_open(*args, **kwargs)
        info = os.fstat(descriptor)
        if fifo_identity and (info.st_dev, info.st_ino) == (fifo_identity[0].st_dev, fifo_identity[0].st_ino):
            fifo_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(os, "open", observe_open)

    def read():
        try:
            results.extend(registration.read_inventory(decoder["controller"]))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    worker.join(1)
    blocked = worker.is_alive()
    if blocked:
        # The real FIFO remains the input. Opening and immediately closing its
        # opposite end only releases an incorrect blocking O_RDONLY reader.
        descriptor = os.open(path, os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW)
        os.close(descriptor)
        worker.join(5)
    assert not worker.is_alive(), "metadata reader survived emergency unblock"
    assert not blocked, "metadata reader blocked on a non-regular record"
    assert not errors
    assert len(fifo_identity) == 1, "the actual leaf-open substitution was not reached"
    identity = fifo_identity[0]
    if race:
        assert fifo_descriptors, "the original reader never opened the substituted FIFO input"
    for descriptor in fifo_descriptors:
        try:
            observed = os.fstat(descriptor)
        except OSError as error:
            assert error.errno == errno.EBADF
        else:
            assert (observed.st_dev, observed.st_ino) != (identity.st_dev, identity.st_ino), "FIFO descriptor leaked"
    result, = results
    _assert_unknown(result)
    after = path.lstat()
    assert stat.S_ISFIFO(after.st_mode) and stat.S_IMODE(after.st_mode) == 0o600
    assert (after.st_dev, after.st_ino) == (identity.st_dev, identity.st_ino)
    expected = dict(before)
    if race:
        del expected[str(path)]  # The hook's only write replaced this file.
    assert _metadata_bytes(decoder) == expected
