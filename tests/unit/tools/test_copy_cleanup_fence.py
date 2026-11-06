"""An invalidation is a denial, never a fabricated controller or Graph receipt."""
from copy import deepcopy
import os
from uuid import uuid4

import pytest

from scripts import stage_runtime as stage
from src.memory import copy_catalog


@pytest.fixture
def fenced(tmp_path):
    root = tmp_path / "admission-root"
    destination = root / "slots/retained/state"
    destination.mkdir(parents=True)
    directory = root / "admission/copy-cleanup/v1/fences"
    directory.mkdir(parents=True)
    copy_id = uuid4().hex
    record = {"schema_version": 1, "scope": "collector_rows_files_v1", "copy_id": copy_id,
        "catalog_revision": 2, "catalog_record_sha256": "1"*64,
        "admission_root": str(root), "admission_identity": stage._identity(root),
        "destination": str(destination), "root_identity": stage._identity(destination),
        "authority_head": {"schema_version": 1, "generation": 1, "binding_id": "2"*64,
                           "status": "closed", "run_id": None},
        "operation_id": str(uuid4()), "conversation_id": str(uuid4()),
        "layout_sha256": "3"*64, "claim_id": uuid4().hex, "status": "invalidated"}
    path = directory / (copy_id + ".json")
    path.write_bytes(stage._canonical_json(record) + b"\n")
    return root, destination, path, record


def test_historical_fence_blocks_only_its_locator_without_current_head_grant(fenced):
    root, destination, _, _ = fenced
    # No current HEAD or closure exists in this fixture. A historical fence is
    # still sufficient to deny, and does not authorize another copy or run.
    with pytest.raises(ValueError, match="^runtime_admission_copy_invalidated$"):
        stage._require_copy_not_invalidated(root, destination)
    stage._require_copy_not_invalidated(root, root / "slots/unrelated/state")


def test_replacing_a_fenced_root_cannot_adopt_its_old_preparation(fenced):
    root, destination, _, _ = fenced
    destination.rename(destination.with_name("old-state"))
    destination.mkdir()
    with pytest.raises(ValueError, match="^runtime_admission_copy_invalidated$"):
        stage._require_copy_not_invalidated(root, destination)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("catalog_revision", False),
    ("scope", "whole_copy"), ("status", "complete"),
    ("operation_id", "not-an-operation"), ("conversation_id", "not-a-conversation"),
    ("catalog_record_sha256", "unknown"), ("admission_root", "/another-root"),
    ("root_identity", {"device": True, "inode": 1}),
    ("authority_head", {"schema_version": 1, "generation": True, "binding_id": "2"*64,
                        "status": "closed", "run_id": None}),
], ids=["schema-bool", "revision-bool", "unknown-scope", "unknown-status", "invalid-operation",
        "invalid-conversation", "invalid-record-hash", "foreign-admission-root", "identity-bool",
        "head-generation-bool"])
def test_unknown_or_ambiguous_fence_blocks_before_any_store_use(fenced, field, value):
    root, destination, path, record = fenced
    changed = deepcopy(record)
    changed[field] = value
    path.write_bytes(stage._canonical_json(changed) + b"\n")
    with pytest.raises(ValueError, match="^runtime_admission_copy_fence_unknown$"):
        stage._require_copy_not_invalidated(root, destination)


@pytest.mark.parametrize("link", ["symbolic", "hard", "ancestor"])
def test_fence_records_cannot_redirect_or_share_physical_files(fenced, link):
    root, destination, path, _ = fenced
    if link == "ancestor":
        parent = root / "admission/copy-cleanup"
        parent.rename(parent.with_name("saved-cleanup"))
        parent.symlink_to(root / "missing-cleanup", target_is_directory=True)
    else:
        saved = path.with_suffix(".saved")
        path.rename(saved)
        if link == "symbolic":
            path.symlink_to(saved)
        else:
            os.link(saved, path)
    with pytest.raises(ValueError, match="^runtime_admission_copy_fence_unknown$"):
        stage._require_copy_not_invalidated(root, destination)


def test_binding_preflight_denies_before_writer_or_schema_inspection(fenced, monkeypatch):
    root, destination, _, _ = fenced
    opened = []
    monkeypatch.setattr(stage, "_checked_slot", lambda *_: (root / "runtime", destination, {}))
    monkeypatch.setattr(stage, "_assert_writers_closed", lambda *_: opened.append("writer"))
    monkeypatch.setattr(stage, "_state_layout", lambda *_: opened.append("store"))
    with pytest.raises(ValueError, match="^runtime_admission_copy_invalidated$"):
        stage._binding_snapshot(root, "retained")
    assert opened == []


@pytest.mark.parametrize("check_state", [False, True])
def test_child_denies_fenced_copy_even_without_state_revalidation(fenced, check_state):
    root, destination, _, _ = fenced
    ticket = {"schema_version": 1, "admission_root": str(root), "binding_id": "2"*64,
        "generation": 2, "run_id": uuid4().hex, "lease_fd": 99999,
        "lease_identity": stage._identity(root), "controller_pid": os.getpid()}
    # The syntactically valid ticket is never treated as an inherited lease.
    # Denial precedes marker, FD and payload inspection.
    with pytest.raises(ValueError, match="^runtime_admission_copy_invalidated$"):
        stage.validate_admission_ticket(destination, ticket, check_state=check_state)


def test_snapshot_reference_and_new_reservation_deny_invalidated_roots(fenced):
    root, destination, _, _ = fenced
    ref = {"source_binding_id": "2"*64, "snapshot_root": str(destination),
           "root_identity": stage._identity(destination), "manifest_sha256": "4"*64}
    with pytest.raises(ValueError, match="^runtime_admission_copy_invalidated$"):
        copy_catalog._snapshot_refs(root, (ref,), ("2"*64,))
    with pytest.raises(ValueError, match="^runtime_admission_copy_invalidated$"):
        copy_catalog._destination(destination, root, [])
