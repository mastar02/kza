"""Persistence protocol and denied authority, never fabricated Graph receipts.

Direct journal fixtures below exercise only bytes/CAS/recovery. They cannot
mint CopyReservation, SDKEntry or controller/native cleanup permission.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scripts.stage_runtime import AdmissionController
from src.memory import copy_catalog as catalog


@pytest.fixture
def record(tmp_path):
    return {"schema_version": 1, "copy_id": uuid4().hex, "kind": "snapshot",
        "authority": {"schema_version": 1, "generation": 2, "binding_id": "1"*64,
            "status": "closed", "run_id": None}, "source_binding_ids": ["1"*64],
        "source_snapshot_refs": [], "destination": str(tmp_path / "copy"),
        "parent_identity": catalog._identity(tmp_path), "root_identity": None,
        "preparation_owner": {"epoch": uuid4().hex, "process_id": os.getpid(),
            "lease_fd": 0, "lease_identity": catalog._identity(tmp_path)},
        "preparation": "reserved", "cleanup": {"phase": "blocked", "owner": None, "receipt": None},
        "extents": {}, "sdk": {}, "prepared_record": None, "error_code": None, "finished": False}


def directory(tmp_path, record):
    path = tmp_path / "journal" / record["copy_id"]
    path.mkdir(parents=True)
    return path


def test_reservation_bytes_precede_any_destination_payload_and_recover_without_ram(tmp_path, record):
    path = directory(tmp_path, record)
    head = catalog._append(path, record, None)
    assert not Path(record["destination"]).exists()
    observed = catalog._recover_record(path)
    assert observed["destination"] == record["destination"]
    assert observed["source_binding_ids"] == record["source_binding_ids"]
    assert observed["record_sha256"] == head["sha256"]
    assert observed["preparation"] == "unknown" and observed["capability"] is False
    assert observed["cleanup"] == {"phase": "blocked", "owner": None, "receipt": None}


def test_stale_revision_cannot_overwrite_the_committed_copy_owner(tmp_path, record):
    path = directory(tmp_path, record)
    first = catalog._append(path, record, None)
    changed = deepcopy(record)
    destination = Path(record["destination"])
    destination.mkdir()
    changed.update(root_identity=catalog._identity(destination), preparation="writing")
    second = catalog._append(path, changed, first)
    before = {item.name: item.read_bytes() for item in path.iterdir()}
    with pytest.raises(catalog.CopyCatalogError, match="copy_catalog_cas"):
        catalog._append(path, dict(record, preparation="failed"), first)
    assert {item.name: item.read_bytes() for item in path.iterdir()} == before
    assert catalog._read_record(path) == (changed, second)


@pytest.mark.parametrize("boundary", ["initial", "later"])
def test_crash_between_event_and_head_preserves_actual_destination_metadata(tmp_path, record, monkeypatch, boundary):
    path = directory(tmp_path, record)
    head = catalog._append(path, record, None) if boundary == "later" else None
    changed = dict(record, preparation="failed", error_code="SyntheticFailure") if head else record
    original = os.replace

    def interrupted(source, target):
        if Path(target) == path / "head.json":
            raise OSError("synthetic event committed before HEAD")
        return original(source, target)

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(OSError):
        catalog._append(path, changed, head)
    before = {item.name: item.read_bytes() for item in path.iterdir()}
    observed = catalog._recover_record(path)
    assert observed["destination"] == record["destination"]
    assert observed["source_binding_ids"] == record["source_binding_ids"]
    assert observed["parent_identity"] == record["parent_identity"]
    assert observed["preparation"] == "unknown" and observed["capability"] is False
    assert observed["catalog_error"] == ("uncommitted_events" if head else "initial_uncommitted_metadata")
    assert {item.name: item.read_bytes() for item in path.iterdir()} == before


def test_failed_reservation_write_does_not_create_a_payload_root(tmp_path, record, monkeypatch):
    path = directory(tmp_path, record)

    def failed(*_args):
        raise OSError("synthetic durable write failure")

    monkeypatch.setattr(catalog, "_write_new", failed)
    with pytest.raises(OSError):
        catalog._append(path, record, None)
    assert not Path(record["destination"]).exists() and not list(path.iterdir())


@pytest.mark.parametrize("field", ["destination", "authority", "sources", "cleanup"])
def test_copy_ownership_and_authority_cannot_be_rewritten_in_the_hash_chain(tmp_path, record, field):
    path = directory(tmp_path, record)
    head = catalog._append(path, record, None)
    changed = deepcopy(record)
    if field == "destination":
        changed["destination"] = str(tmp_path / "other")
    elif field == "authority":
        changed["authority"]["generation"] += 1
    elif field == "sources":
        changed["source_binding_ids"] = ["2"*64]
    else:
        changed["cleanup"]["owner"] = "not a real cleanup driver"
    with pytest.raises(catalog.CopyCatalogError):
        catalog._append(path, changed, head)
    assert catalog._read_record(path) == (record, head)


@pytest.mark.parametrize("field", ["schema_version", "authority.schema_version", "authority.generation",
                                    "preparation_owner.lease_fd", "finished"])
def test_unknown_boolean_integer_fields_fail_before_a_reservation_is_published(tmp_path, record, field):
    path = directory(tmp_path, record)
    changed = deepcopy(record)
    parts = field.split(".")
    holder = changed if len(parts) == 1 else changed[parts[0]]
    holder[parts[-1]] = 1 if field == "finished" else True
    with pytest.raises(catalog.CopyCatalogError):
        catalog._append(path, changed, None)
    assert not list(path.iterdir()) and not Path(record["destination"]).exists()


@pytest.mark.parametrize("initial", ["writing", "prepared", "unknown"])
def test_a_first_event_cannot_skip_the_before_first_byte_reservation(tmp_path, record, initial):
    path = directory(tmp_path, record)
    with pytest.raises(catalog.CopyCatalogError):
        catalog._append(path, dict(record, preparation=initial), None)
    assert not list(path.iterdir())


def test_protocol_prepared_is_historical_and_failure_before_finish_stays_blocked(tmp_path, record):
    path = directory(tmp_path, record)
    head = catalog._append(path, record, None)
    destination = Path(record["destination"])
    destination.mkdir()
    writing = dict(record, preparation="writing", root_identity=catalog._identity(destination))
    head = catalog._append(path, writing, head)
    manifest = destination / "manifest.json"
    manifest.write_bytes(b'{"protocol_fixture":true}\n')
    extent = {"relative": ".", "family": "managed_payload", "representation": "state"}
    key = catalog._sha(catalog._json(extent))
    preparing = dict(writing, extents={key: extent})
    head = catalog._append(path, preparing, head)
    inventory = catalog._inventory(destination)
    evidence = {"relative": "manifest.json", "sha256": inventory["manifest.json"]["sha256"],
        "size": inventory["manifest.json"]["size"], "inventory": inventory,
        "extent_observations": {key: {"status": "present", "identity": catalog._identity(destination)}}}
    prepared = dict(preparing, preparation="prepared", prepared_record=evidence)
    head = catalog._append(path, prepared, head)
    assert catalog._recover_record(path)["preparation"] == "unknown"
    failed = dict(prepared, preparation="failed", error_code="PublicationFailure")
    catalog._append(path, failed, head)
    observed = catalog._recover_record(path)
    assert observed["prepared_record"] == evidence and observed["finished"] is False
    assert observed["preparation"] == "failed" and observed["cleanup"]["receipt"] is None


@pytest.mark.parametrize("phase", ["not_entered", "constructor_entered", "unknown"])
def test_sdk_return_flags_cannot_skip_actual_protocol_boundaries(tmp_path, record, phase):
    path = directory(tmp_path, record)
    head = catalog._append(path, record, None)
    extent = {"relative": "sdk", "family": "semantic_index", "representation": "sdk_work"}
    key = catalog._sha(catalog._json(extent))
    destination = Path(record["destination"])
    destination.mkdir()
    writing = dict(record, preparation="writing", root_identity=catalog._identity(destination), extents={key: extent})
    head = catalog._append(path, writing, head)
    sdk_id = uuid4().hex
    entered = dict(writing, sdk={sdk_id: {"extent": key, "sdk_version": "0.0.0", "phase": "not_entered"}})
    head = catalog._append(path, entered, head)
    if phase != "not_entered":
        entered = deepcopy(entered)
        entered["sdk"][sdk_id]["phase"] = "constructor_entered"
        head = catalog._append(path, entered, head)
    if phase == "unknown":
        entered = deepcopy(entered)
        entered["sdk"][sdk_id]["phase"] = "unknown"
        entered["preparation"] = "unknown"
        head = catalog._append(path, entered, head)
    forged = deepcopy(entered)
    forged["sdk"][sdk_id]["phase"] = "close_returned"
    with pytest.raises(catalog.CopyCatalogError, match="sdk_history"):
        catalog._append(path, forged, head)
    assert catalog._recover_record(path)["cleanup"]["phase"] == "blocked"


def test_a_catalog_dictionary_or_history_is_not_a_controller_grant(tmp_path, record):
    path = directory(tmp_path, record)
    catalog._append(path, record, None)
    observed = catalog._recover_record(path)
    for purported in (observed, SimpleNamespace(root=tmp_path, _descriptor=0)):
        with pytest.raises(catalog.CopyCatalogError, match="copy_catalog_controller$"):
            catalog.reserve_copy(purported, "snapshot", tmp_path / "payload", "1"*64, ("1"*64,))
    assert not (tmp_path / "payload").exists()


def test_real_controller_without_a_current_closed_graph_cannot_reserve_payload(tmp_path):
    root, destination = tmp_path / "admission", tmp_path / "copy"
    with AdmissionController(root) as controller:
        with pytest.raises(catalog.CopyCatalogError, match="copy_catalog_closed_head$"):
            catalog.reserve_copy(controller, "snapshot", destination, "1"*64, ("1"*64,))
        with pytest.raises(catalog.CopyCatalogError, match="copy_catalog_closed_head$"):
            catalog.recover_copies(controller)
    assert not destination.exists() and not (root / "admission/copies").exists()


@pytest.mark.parametrize("drift", ["fd", "header", "identity", "closed"])
def test_real_controller_fd_header_and_physical_lease_cannot_be_substituted(tmp_path, drift):
    root, destination = tmp_path / "admission", tmp_path / "copy"
    with AdmissionController(root) as controller:
        fd = controller._descriptor
        header = (root / ".admission.lock").read_bytes()
        duplicate = None
        renamed = tmp_path / "original-lease"
        try:
            if drift == "fd":
                duplicate = os.dup(fd)
                controller._descriptor = duplicate
            elif drift == "header":
                replacement = json.dumps({"controller_pid": True, "lease_fd": fd}).encode()
                os.ftruncate(fd, 0)
                os.pwrite(fd, replacement, 0)
            elif drift == "identity":
                (root / ".admission.lock").rename(renamed)
                (root / ".admission.lock").write_bytes(header)
            else:
                controller.__exit__(None, None, None)
            with pytest.raises((ValueError, OSError)):
                catalog.reserve_copy(controller, "snapshot", destination, "1"*64, ("1"*64,))
            assert not destination.exists()
        finally:
            if drift != "closed":
                controller._descriptor = fd
            if drift == "header":
                os.ftruncate(fd, 0)
                os.pwrite(fd, header, 0)
            if drift == "identity":
                (root / ".admission.lock").unlink()
                renamed.rename(root / ".admission.lock")
            if duplicate is not None:
                os.close(duplicate)


@pytest.mark.parametrize("field", ["schema_version", "generation", "binding_id"])
def test_changed_or_unknown_head_never_grants_a_copy_before_binding_validation(tmp_path, field):
    root, destination = tmp_path / "admission", tmp_path / "copy"
    with AdmissionController(root) as controller:
        # Deliberately invalid metadata: no Graph/closure or usable binding is
        # created. This is a rejection before any payload, not a success fixture.
        (root / "admission").mkdir()
        head = {"schema_version": 1, "generation": 2, "binding_id": "1"*64, "status": "closed", "run_id": None}
        head[field] = "2"*64 if field == "binding_id" else True
        (root / "admission/head.json").write_text(json.dumps(head))
        with pytest.raises(ValueError):
            catalog.reserve_copy(controller, "snapshot", destination, "1"*64, ("1"*64,))
        assert not destination.exists() and not (root / "admission/copies").exists()


@pytest.mark.parametrize("kind", [catalog.CopyReservation, catalog.Extent, catalog.SDKEntry])
def test_process_local_capability_cannot_be_reconstituted_from_serialized_identity(kind):
    unusable = object.__new__(kind)
    with pytest.raises(TypeError, match="copy_catalog_capability_not_serializable"):
        pickle.dumps(unusable)
    with pytest.raises(TypeError, match="copy_catalog_capability_not_serializable"):
        deepcopy(unusable)


def test_a_borrowed_real_chroma_client_lambda_is_rejected_before_dispatch(tmp_path):
    import chromadb
    from chromadb.config import Settings

    borrowed = chromadb.PersistentClient(path=str(tmp_path / "borrowed"),
        settings=Settings(anonymized_telemetry=False))
    calls = []

    def wrapped():
        calls.append(True)
        return borrowed

    try:
        with pytest.raises(catalog.CopyCatalogError, match="sdk_factory_unsupported"):
            catalog._sdk_factory(wrapped)
        assert calls == [] and borrowed.list_collections() == []
    finally:
        borrowed.close()


@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_inventory_cannot_follow_an_unowned_payload_representation(tmp_path, link):
    root = tmp_path / "copy"
    root.mkdir()
    outside = tmp_path / "payload"
    outside.write_bytes(b"synthetic bytes")
    if link == "symbolic":
        (root / "data").symlink_to(outside)
    else:
        (root / "data").hardlink_to(outside)
    with pytest.raises(catalog.CopyCatalogError):
        catalog._inventory(root)
    assert outside.read_bytes() == b"synthetic bytes"


def snapshot_protocol(tmp_path, record, *, phase="finished"):
    """A reference-codec journal, not an AdmissionController or Graph witness."""
    root = tmp_path / "controller"
    path = root / "admission/copies" / record["copy_id"]
    path.mkdir(parents=True)
    snapshot = Path(record["destination"])
    snapshot.mkdir()
    manifest = {"source_binding_id": "1"*64, "protocol_only": True}
    raw = catalog._json(manifest) + b"\n"
    (snapshot / "manifest.json").write_bytes(raw)
    refs = ({"source_binding_id": "1"*64, "snapshot_root": str(snapshot),
        "root_identity": catalog._identity(snapshot), "manifest_sha256": catalog._sha(catalog._json(manifest))},)
    if phase == "legacy":
        return root, refs, manifest
    head = catalog._append(path, record, None)
    extent = {"relative": ".", "family": "managed_payload", "representation": "state"}
    key = catalog._sha(catalog._json(extent))
    writing = dict(record, preparation="writing", root_identity=catalog._identity(snapshot), extents={key: extent})
    head = catalog._append(path, writing, head)
    inventory = catalog._inventory(snapshot)
    evidence = {"relative": "manifest.json", "sha256": catalog._sha(raw), "size": len(raw),
        "inventory": inventory, "extent_observations": {key: {"status": "present", "identity": catalog._identity(snapshot)}}}
    prepared = dict(writing, preparation="prepared", prepared_record=evidence)
    head = catalog._append(path, prepared, head)
    if phase == "finished":
        catalog._append(path, dict(prepared, finished=True), head)
    elif phase in {"failed", "unknown"}:
        catalog._append(path, dict(prepared, preparation=phase, error_code="SyntheticFailure"), head)
    return root, refs, manifest


def test_rewritten_manifest_and_dto_cannot_replace_the_external_prepared_snapshot_pin(tmp_path, record):
    root, refs, manifest = snapshot_protocol(tmp_path, record)
    assert catalog._snapshot_refs(root, refs, {"1"*64: {}}) == list(refs)
    # Even updating the local DTO digest cannot replace the separately stored
    # final prepared record. The private codec check itself grants no root lease.
    manifest["rewritten"] = True
    (Path(refs[0]["snapshot_root"]) / "manifest.json").write_bytes(catalog._json(manifest) + b"\n")
    rewritten = (dict(refs[0], manifest_sha256=catalog._sha(catalog._json(manifest))),)
    with pytest.raises(catalog.CopyCatalogError, match="snapshot_not_catalog_prepared"):
        catalog._snapshot_refs(root, rewritten, {"1"*64: {}})


@pytest.mark.parametrize("phase", ["unfinished", "failed", "unknown", "legacy"])
def test_unfinished_failed_unknown_and_unregistered_snapshots_are_not_prepared_refs(tmp_path, record, phase):
    root, refs, _ = snapshot_protocol(tmp_path, record, phase=phase)
    with pytest.raises(catalog.CopyCatalogError, match="snapshot_not_catalog_prepared"):
        catalog._snapshot_refs(root, refs, {"1"*64: {}})


def test_current_snapshot_identity_cannot_be_relabelled_after_directory_replacement(tmp_path, record):
    root, refs, _ = snapshot_protocol(tmp_path, record)
    snapshot = Path(refs[0]["snapshot_root"])
    before = snapshot / "manifest.json"
    raw = before.read_bytes()
    snapshot.rename(tmp_path / "retained-old-snapshot")
    snapshot.mkdir()
    (snapshot / "manifest.json").write_bytes(raw)
    changed = (dict(refs[0], root_identity=catalog._identity(snapshot)),)
    with pytest.raises(catalog.CopyCatalogError, match="snapshot_not_catalog_prepared"):
        catalog._snapshot_refs(root, changed, {"1"*64: {}})


@pytest.mark.parametrize("phase", ["handle_returned", "close_entered"])
def test_interrupted_sdk_metadata_publication_keeps_committed_owner_unknown(tmp_path, record, phase):
    """Protocol bytes only: no real handle, close_returned or admission ACK is issued."""
    path = directory(tmp_path, record)
    head = catalog._append(path, record, None)
    destination = Path(record["destination"])
    destination.mkdir()
    sentinel = destination / "retained"
    sentinel.write_bytes(b"unconfirmed SDK copy remains owned and unserved")
    extent = {"relative": ".", "family": "semantic_index", "representation": "sdk_work"}
    extent_id = catalog._sha(catalog._json(extent))
    sdk_id = uuid4().hex
    current = dict(record, preparation="writing", root_identity=catalog._identity(destination),
        extents={extent_id: extent}, sdk={sdk_id: {
            "extent": extent_id, "sdk_version": "0.0.0", "phase": "not_entered"}})
    head = catalog._append(path, current, head)
    current = deepcopy(current)
    current["sdk"][sdk_id]["phase"] = "constructor_entered"
    head = catalog._append(path, current, head)
    if phase == "close_entered":
        current = deepcopy(current)
        current["sdk"][sdk_id]["phase"] = "handle_returned"
        head = catalog._append(path, current, head)
    committed_phase = current["sdk"][sdk_id]["phase"]
    committed_head = (path / "head.json").read_bytes()
    uncommitted = deepcopy(current)
    uncommitted["sdk"][sdk_id]["phase"] = phase
    catalog._append(path, uncommitted, head)
    # Represent interruption after durable event write but before HEAD became
    # durable. Recovery must neither promote this event nor repair the files.
    (path / "head.json").write_bytes(committed_head)
    before = {entry.name: entry.read_bytes() for entry in path.iterdir()}
    recovered = catalog._recover_record(path)
    assert recovered["preparation"] == "unknown" and recovered["finished"] is False
    assert recovered["catalog_error"] == "uncommitted_events" and recovered["capability"] is False
    assert recovered["sdk"][sdk_id]["phase"] == committed_phase
    assert recovered["cleanup"] == {"phase": "blocked", "owner": None, "receipt": None}
    assert {entry.name: entry.read_bytes() for entry in path.iterdir()} == before
    assert sentinel.read_bytes() == b"unconfirmed SDK copy remains owned and unserved"
