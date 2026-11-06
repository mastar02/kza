"""Retrospective metadata and denied authority; no fabricated Graph/SDK ACK.

Direct journal inputs exercise protocol/IO only. Actual successful public
registration remains a separate closed-F integration obligation.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from scripts import stage_runtime as stage
from src.core.runtime_admission import _stage
from src.memory import copy_catalog as catalog
from src.memory import copy_observation as observations


def digest(value):
    return hashlib.sha256(catalog._json(value)).hexdigest()


@pytest.fixture
def observation(tmp_path):
    """Prior physical bytes and a protocol intent, never a closed Graph grant."""
    root = (tmp_path / "controller").resolve()
    root.mkdir()
    (root / "admission").mkdir()
    payload = (tmp_path / "existing").resolve()
    payload.mkdir()
    retained = payload / "retained.bin"
    retained.write_bytes(b"synthetic bytes existed before observation")
    identity = uuid4().hex
    intent = {"schema_version": 1, "observation_id": identity, "kind": "forensic_audit",
        "authority": {"schema_version": 1, "generation": 3, "binding_id": "a" * 64,
                      "status": "closed", "run_id": None},
        "admission_root": str(root), "admission_identity": catalog._identity(root),
        "observed_root": str(payload), "parent_identity": catalog._identity(payload.parent),
        "root_identity": catalog._identity(payload),
        "extents": [{"relative": "retained.bin", "family": "history", "representation": "raw",
                     "source_sha256": hashlib.sha256(retained.read_bytes()).hexdigest()}],
        "source_binding_ids": ["a" * 64], "existing_copy": None, "before_first_byte": False}
    directory = observations._namespace(root, create=True) / identity
    directory.mkdir()
    catalog._write_new(directory / "intent.json", catalog._json(intent) + b"\n")
    return root, payload, directory, intent


def published(observation):
    _, payload, directory, intent = observation
    first = observations._publish(directory, intent, None, None)
    inventory = catalog._inventory(payload)
    final = observations._publish(directory, intent, inventory, first)
    return first, final, inventory


def public_arguments(payload):
    return {"kind": "forensic_audit", "existing_root": payload, "latest_binding_id": "a" * 64,
        "extents": (observations.ObservedExtent(".", "history", "raw"),)}


def test_existing_bytes_recover_as_history_never_prepared_or_granted(observation):
    root, payload, directory, intent = observation
    original = (payload / "retained.bin").read_bytes()
    _, final, inventory = published(observation)
    recovered = observations._recover(directory)
    assert recovered["observation_id"] == intent["observation_id"]
    assert recovered["phase"] == "observed" and recovered["revision"] == 2
    assert recovered["record_sha256"] == final["sha256"]
    assert recovered["inventory"] == inventory and recovered["inventory_sha256"] == digest(inventory)
    assert recovered["retrospective"] is True and recovered["before_first_byte"] is False
    assert recovered["non_serving"] is True and recovered["capability"] is False
    assert recovered["lineage_complete"] is False and recovered["SDK_retirement_confirmed"] is False
    assert recovered["cleanup"] == {"phase": "blocked", "owner": None, "receipt": None}
    assert (payload / "retained.bin").read_bytes() == original
    assert not (root / "admission/copies").exists()
    arguments = {"observation_id": intent["observation_id"], "revision": 2,
        "record_sha256": final["sha256"], "observed_root": str(payload), "inventory_sha256": digest(inventory)}
    receipt = observations.ObservationReceipt(**arguments)
    assert asdict(receipt)["capability"] is False
    with pytest.raises(catalog.CopyCatalogError):
        observations._shape_intent(dict(intent, before_first_byte=True))
    for flag in ("retrospective", "before_first_byte", "non_serving", "capability"):
        with pytest.raises(TypeError):
            observations.ObservationReceipt(**arguments, **{flag: True})


@pytest.mark.parametrize("field", ["identity", "path", "inode", "content_pin", "source_binding"])
def test_observation_identity_roots_and_pins_cannot_be_rewritten(observation, field):
    _, payload, directory, intent = observation
    first = observations._publish(directory, intent, None, None)
    changed = deepcopy(intent)
    if field == "identity":
        changed["observation_id"] = uuid4().hex
    elif field == "path":
        changed["observed_root"] += "-replacement"
    elif field == "inode":
        changed["root_identity"]["inode"] += 1
    elif field == "content_pin":
        changed["extents"][0]["source_sha256"] = "b" * 64
    else:
        changed["source_binding_ids"] = ["b" * 64]
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(catalog.CopyCatalogError):
        observations._publish(directory, changed, catalog._inventory(payload), first)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("field", ["schema_version", "authority.generation", "root_identity.inode"])
def test_ambiguous_boolean_integers_fail_before_publication(observation, field):
    _, _, directory, intent = observation
    changed = deepcopy(intent)
    pieces = field.split(".")
    holder = changed if len(pieces) == 1 else changed[pieces[0]]
    holder[pieces[-1]] = True
    with pytest.raises(catalog.CopyCatalogError):
        observations._publish(directory, changed, None, None)
    assert [p.name for p in directory.iterdir()] == ["intent.json"]


def test_stale_revision_and_duplicate_publication_cannot_refresh_history(observation):
    _, _, directory, intent = observation
    first, _, inventory = published(observation)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    for expected in (None, first, dict(first, revision=0), dict(first, sha256="c" * 64)):
        with pytest.raises(catalog.CopyCatalogError):
            observations._publish(directory, intent, inventory, expected)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("problem", ["changed_source", "unregistered_bytes"])
def test_inventory_cannot_hide_bytes_or_relabel_content_pin(observation, problem):
    _, payload, directory, intent = observation
    first = observations._publish(directory, intent, None, None)
    if problem == "changed_source":
        (payload / "retained.bin").write_bytes(b"different physical bytes")
    else:
        (payload / "unregistered.bin").write_bytes(b"outside explicit extent")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(catalog.CopyCatalogError):
        observations._publish(directory, intent, catalog._inventory(payload), first)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert observations._recover(directory)["phase"] == "unknown"


@pytest.mark.parametrize("boundary", ["initial", "later"])
def test_event_before_HEAD_crash_preserves_prefix_without_adoption(observation, monkeypatch, boundary):
    root, payload, directory, intent = observation
    first = observations._publish(directory, intent, None, None) if boundary == "later" else None
    previous = (directory / "head.json").read_bytes() if first else None
    original = os.replace

    def interrupted(source, target):
        if Path(target) == directory / "head.json":
            raise OSError("synthetic event-before-head interruption")
        return original(source, target)

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(OSError):
        observations._publish(directory, intent, catalog._inventory(payload) if first else None, first)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    recovered = observations._recover(directory)
    assert recovered["observed_root"] == str(payload) and recovered["root_identity"] == intent["root_identity"]
    assert recovered["source_binding_ids"] == intent["source_binding_ids"]
    assert recovered["phase"] == "unknown" and recovered["capability"] is False
    assert recovered["before_first_byte"] is False and recovered["non_serving"] is True
    if first:
        assert recovered["revision"] == 1 and recovered["record_sha256"] == first["sha256"]
        assert recovered["inventory"] is None
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert ((directory / "head.json").read_bytes() if first else None) == previous
    with pytest.raises(ValueError, match="^runtime_admission_copy_observation_unknown$"):
        stage._require_copy_not_observed(root, root / "unrelated")


def test_tampered_event_is_unknown_without_repair_or_permission(observation):
    root, payload, directory, _ = observation
    _, head, _ = published(observation)
    event = directory / f"000000000002-{head['sha256']}.json"
    event.write_bytes(event.read_bytes() + b" ")
    retained = {p.name: p.read_bytes() for p in directory.iterdir()}
    recovered = observations._recover(directory)
    assert recovered["phase"] == "unknown" and recovered["capability"] is False
    assert recovered["cleanup"]["receipt"] is None
    with pytest.raises(ValueError, match="^runtime_admission_copy_observation_unknown$"):
        stage._require_copy_not_observed(root, payload)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == retained


def rewritten_protocol(directory, intent):
    """Canonical adversarial metadata with valid hashes, not a registry grant."""
    events = []
    for revision in (1, 2):
        path, = directory.glob(f"{revision:012d}-*.json")
        events.append(stage._read_json(path))
        path.unlink()
    raw = catalog._json(intent) + b"\n"
    (directory / "intent.json").write_bytes(raw)
    previous = None
    for event in events:
        event["intent_sha256"] = hashlib.sha256(raw).hexdigest()
        event["previous_sha256"] = previous
        event_raw = catalog._json(event) + b"\n"
        previous = hashlib.sha256(event_raw).hexdigest()
        (directory / f"{event['revision']:012d}-{previous}.json").write_bytes(event_raw)
    head = {"schema_version": 1, "observation_id": intent["observation_id"], "revision": 2, "sha256": previous}
    (directory / "head.json").write_bytes(catalog._json(head) + b"\n")


@pytest.mark.parametrize("problem", ["source_pin", "coverage"])
def test_valid_hash_chain_cannot_hide_semantic_extent_tamper(observation, problem):
    root, payload, directory, intent = observation
    published(observation)
    changed = deepcopy(intent)
    if problem == "source_pin":
        changed["extents"][0]["source_sha256"] = "c" * 64
    else:
        changed["extents"][0]["relative"] = "not-in-inventory.bin"
        changed["extents"][0]["source_sha256"] = None
    rewritten_protocol(directory, changed)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError, match="^runtime_admission_copy_observation_unknown$"):
        stage._require_copy_not_observed(root, payload)
    assert observations._recover(directory)["phase"] == "unknown"
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("link", ["symbolic", "hard", "ancestor"])
def test_record_and_namespace_aliases_fail_closed(observation, link):
    root, payload, directory, _ = observation
    published(observation)
    if link == "ancestor":
        parent = root / "admission/copy-observations"
        parent.rename(parent.with_name("saved-observations"))
        parent.symlink_to(root / "missing-observations", target_is_directory=True)
    else:
        path = directory / "intent.json"
        saved = root.parent / ("retained-intent-" + uuid4().hex)
        path.rename(saved)
        if link == "symbolic":
            path.symlink_to(saved)
        else:
            os.link(saved, path)
    original = (payload / "retained.bin").read_bytes()
    with pytest.raises(ValueError, match="^runtime_admission_copy_observation_unknown$"):
        stage._require_copy_not_observed(root, payload)
    assert (payload / "retained.bin").read_bytes() == original


@pytest.mark.parametrize("drift", ["fd", "header", "identity", "closed"])
def test_actual_controller_lease_guard_precedes_missing_graph_binding(observation, drift):
    root, payload, directory, _ = observation
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with stage.AdmissionController(root) as controller:
        descriptor = controller._descriptor
        header = (root / ".admission.lock").read_bytes()
        duplicate = None
        renamed = root.parent / ("retained-lease-" + uuid4().hex)
        try:
            if drift == "fd":
                duplicate = os.dup(descriptor)
                controller._descriptor = duplicate
            elif drift == "header":
                os.ftruncate(descriptor, 0)
                os.pwrite(descriptor, catalog._json({"controller_pid": True, "lease_fd": descriptor}), 0)
            elif drift == "identity":
                (root / ".admission.lock").rename(renamed)
                (root / ".admission.lock").write_bytes(header)
            else:
                controller.__exit__(None, None, None)
            expected = "^runtime_admission_controller_not_entered$" if drift == "closed" else "^copy_catalog_controller_fd$"
            # Exact early error prevents missing HEAD/binding from masking a
            # removed FD guard. No successful authority is patched into place.
            with pytest.raises(ValueError, match=expected):
                observations.observe_existing_copy(controller, **public_arguments(payload))
            with pytest.raises(ValueError, match=expected):
                observations.recover_observations(controller)
        finally:
            if drift != "closed":
                controller._descriptor = descriptor
            if drift == "header":
                os.ftruncate(descriptor, 0)
                os.pwrite(descriptor, header, 0)
                os.fsync(descriptor)
            if drift == "identity":
                (root / ".admission.lock").unlink()
                renamed.rename(root / ".admission.lock")
            if duplicate is not None:
                os.close(duplicate)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("state", ["running", "unknown", "stale_binding"])
def test_current_head_must_be_closed_and_exact_before_observation_write(observation, state):
    root, payload, directory, intent = observation
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with stage.AdmissionController(root) as controller:
        head = deepcopy(intent["authority"])
        if state != "stale_binding":
            head.update(status=state, run_id=uuid4().hex)
        # Metadata deliberately lacks any Graph/binding/closure. It is only an
        # invalid input for the early status/pin guard, never a success fixture.
        stage._atomic_record(root / "admission/head.json", head)
        arguments = public_arguments(payload)
        if state == "stale_binding":
            arguments["latest_binding_id"] = "b" * 64
        expected = "^copy_catalog_authority_changed$" if state == "stale_binding" else "^copy_catalog_closed_head$"
        with pytest.raises(catalog.CopyCatalogError, match=expected):
            observations.observe_existing_copy(controller, **arguments)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize("purported", ["history", "namespace"])
def test_recovered_DTO_or_foreign_object_cannot_authorize_registration(observation, purported):
    _, payload, directory, _ = observation
    published(observation)
    value = observations._recover(directory) if purported == "history" else SimpleNamespace(root=payload)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(catalog.CopyCatalogError, match="^copy_catalog_controller$"):
        observations.observe_existing_copy(value, **public_arguments(payload))
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_live_lease_can_recover_UNKNOWN_metadata_without_opening_stores(observation, monkeypatch):
    root, payload, directory, intent = observation
    _, head, _ = published(observation)
    event = directory / f"000000000002-{head['sha256']}.json"
    event.write_bytes(event.read_bytes() + b" ")
    before = {p.name: p.read_bytes() for p in directory.iterdir()}

    def forbidden_store_grant(*_args, **_kwargs):
        pytest.fail("metadata recovery must not request a store-seal grant")

    with stage.AdmissionController(root) as controller:
        # Actual live lease plus a typed metadata HEAD permits observation
        # reads only. No binding, Graph closure or serving receipt is fabricated.
        stage._atomic_record(root / "admission/head.json", intent["authority"])
        monkeypatch.setattr(_stage(), "verify_closed_binding", forbidden_store_grant)
        recovered, = observations.recover_observations(controller)
        assert recovered["observed_root"] == str(payload) and recovered["phase"] == "unknown"
        assert recovered["capability"] is False and recovered["SDK_retirement_confirmed"] is False
        assert recovered["cleanup"] == {"phase": "blocked", "owner": None, "receipt": None}
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_observation_is_not_a_prepared_snapshot_or_destination_grant(observation):
    root, payload, directory, intent = observation
    manifest = {"source_binding_id": "a" * 64, "retrospective_fixture": True}
    (payload / "manifest.json").write_bytes(catalog._json(manifest) + b"\n")
    intent["extents"] = [{"relative": ".", "family": "history", "representation": "raw", "source_sha256": None}]
    (directory / "intent.json").write_bytes(catalog._json(intent) + b"\n")
    published(observation)
    ref = {"source_binding_id": "a" * 64, "snapshot_root": str(payload),
        "root_identity": catalog._identity(payload), "manifest_sha256": digest(manifest)}
    before = catalog._inventory(payload)
    with pytest.raises(ValueError, match="^runtime_admission_copy_observed$"):
        catalog._snapshot_refs(root, (ref,), {"a" * 64: {}})
    with pytest.raises(ValueError, match="^runtime_admission_copy_observed$"):
        catalog._destination(payload, root, [])
    assert catalog._inventory(payload) == before and not (root / "admission/copies").exists()


@pytest.mark.parametrize("selected", ["state", "runtime"])
def test_binding_gate_denies_observed_state_or_runtime_before_store_use(observation, monkeypatch, selected):
    root, payload, _, _ = observation
    published(observation)
    other = root / "unrelated"
    runtime, state = (payload, other) if selected == "runtime" else (other, payload)
    opened = []
    monkeypatch.setattr(stage, "_checked_slot", lambda *_: (runtime, state, {}))
    monkeypatch.setattr(stage, "_assert_writers_closed", lambda *_: opened.append("writers"))
    monkeypatch.setattr(stage, "_state_layout", lambda *_: opened.append("store"))
    with pytest.raises(ValueError, match="^runtime_admission_copy_observed$"):
        stage._binding_snapshot(root, "unadmitted-negative")
    assert opened == []


@pytest.mark.parametrize("check_state", [False, True])
def test_child_gate_denies_observed_state_even_without_state_revalidation(observation, check_state):
    root, payload, _, _ = observation
    published(observation)
    ticket = {"schema_version": 1, "admission_root": str(root), "binding_id": "a" * 64,
        "generation": 3, "run_id": uuid4().hex, "lease_fd": 999999,
        "lease_identity": catalog._identity(root), "controller_pid": os.getpid()}
    # Denial precedes missing marker/current-run/FD validation. This syntactic
    # negative ticket never supplies an inherited lease or successful admission.
    with pytest.raises(ValueError, match="^runtime_admission_copy_observed$"):
        stage.validate_admission_ticket(payload, ticket, check_state=check_state)
