"""Retrospective inventory of existing copies, without execution capabilities.

An observation records bytes which already existed. It cannot attest their
historical completeness or SDK retirement, and its locator remains non-serving.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING
from uuid import uuid4

from src.memory import copy_catalog as catalog

if TYPE_CHECKING:
    from scripts.stage_runtime import AdmissionController

KINDS = frozenset({"forensic_audit", "phase_checkpoint", "failed_work"})
MAX_RECORD_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ObservedExtent:
    """Identify original relative bytes and an optional existing content pin."""

    relative: str
    family: str
    representation: str
    source_sha256: str | None = None


@dataclass(frozen=True)
class ObservationReceipt:
    """Describe a point-in-time inventory; never an execution/erasure grant."""

    observation_id: str
    revision: int
    record_sha256: str
    observed_root: str
    inventory_sha256: str
    retrospective: bool = field(default=True, init=False)
    before_first_byte: bool = field(default=False, init=False)
    non_serving: bool = field(default=True, init=False)
    capability: bool = field(default=False, init=False)


def _require(condition, code):
    catalog._require(condition, "observation_" + code)


def _namespace(root, *, create=False):
    parent = catalog._directory(root / "admission")
    for part in ("copy-observations", "v1"):
        candidate = parent / part
        if create and not candidate.exists():
            candidate.mkdir(mode=0o700)
            catalog._fsync(parent)
        parent = catalog._directory(candidate)
    return parent


def _shape_intent(intent):
    _require(type(intent) is dict and set(intent) == {"schema_version", "observation_id", "kind",
        "authority", "admission_root", "admission_identity", "observed_root", "parent_identity",
        "root_identity", "extents", "source_binding_ids", "existing_copy", "before_first_byte"}, "intent")
    _require(type(intent["schema_version"]) is int and intent["schema_version"] == 1
        and type(intent["observation_id"]) is str and re.fullmatch("[0-9a-f]{32}", intent["observation_id"])
        and type(intent["kind"]) is str and intent["kind"] in KINDS and intent["before_first_byte"] is False, "intent")
    for key in ("root_identity", "parent_identity", "admission_identity"):
        _require(catalog._identity_shape(intent[key]), "identity")
    for key in ("admission_root", "observed_root"):
        _require(type(intent[key]) is str and Path(intent[key]).is_absolute()
            and Path(intent[key]).resolve(strict=False) == Path(intent[key]), "path")
    authority = intent["authority"]
    _require(type(authority) is dict and set(authority) == {"schema_version", "generation", "binding_id", "status", "run_id"}
        and type(authority["schema_version"]) is int and authority["schema_version"] == 1
        and type(authority["generation"]) is int and authority["generation"] >= 1
        and catalog._hash(authority["binding_id"]) and authority["status"] == "closed"
        and authority["run_id"] is None, "authority")
    refs = intent["source_binding_ids"]
    _require(type(refs) is list and all(catalog._hash(ref) for ref in refs)
        and len(set(refs)) == len(refs), "references")
    existing = intent["existing_copy"]
    _require(existing is None or type(existing) is dict
        and set(existing) == {"copy_id", "revision", "record_sha256"}
        and type(existing["copy_id"]) is str and re.fullmatch("[0-9a-f]{32}", existing["copy_id"])
        and type(existing["revision"]) is int and existing["revision"] >= 1
        and catalog._hash(existing["record_sha256"]), "existing_copy")
    extents = intent["extents"]
    _require(type(extents) is list and bool(extents), "extents")
    for extent in extents:
        _require(type(extent) is dict and set(extent) == {"relative", "family", "representation", "source_sha256"}
            and type(extent["family"]) is str and extent["family"] in catalog.FAMILIES
            and type(extent["representation"]) is str and extent["representation"] in catalog.REPRESENTATIONS
            and (extent["source_sha256"] is None or catalog._hash(extent["source_sha256"])), "extent")
        catalog._relative(extent["relative"])
        _require("\x00" not in extent["relative"], "relative_path")
    _require(len({e["relative"] for e in extents}) == len(extents), "extents")
    _require(len(catalog._json(intent)) + 1 <= MAX_RECORD_BYTES, "record_size")


def _publish(directory, intent, inventory, expected):
    """Append one immutable intent/observed event with an exact CAS head."""
    from src.core.runtime_admission import _stage

    _shape_intent(intent)
    if inventory is not None:
        _require(type(inventory) is dict, "inventory")
        for relative, item in inventory.items():
            catalog._relative(relative)
            _require(relative != "." and type(item) is dict and set(item) == {"sha256", "size", "identity"}
                and catalog._hash(item["sha256"]) and type(item["size"]) is int and item["size"] >= 0
                and catalog._identity_shape(item["identity"]), "inventory")
        _check_extents(intent, inventory)
    intent_value, raw = catalog._read(directory / "intent.json")
    _require(intent_value == intent, "intent_changed")
    if expected is None:
        _require(inventory is None and not (directory / "head.json").exists(), "cas")
        revision, previous = 1, None
    else:
        actual_intent, prior, head = _stage()._read_copy_observation(directory, Path(intent["admission_root"]))
        _require(actual_intent == intent and head == expected and head["revision"] == 1
            and prior["phase"] == "intent" and inventory, "cas")
        revision, previous = 2, head["sha256"]
    event = {"schema_version": 1, "observation_id": intent["observation_id"], "revision": revision,
        "previous_sha256": previous, "intent_sha256": catalog._sha(raw),
        "phase": "intent" if inventory is None else "observed", "inventory": inventory,
        "inventory_sha256": None if inventory is None else catalog._sha(catalog._json(inventory))}
    raw = catalog._json(event) + b"\n"
    _require(len(raw) <= MAX_RECORD_BYTES, "record_size")
    digest = catalog._sha(raw)
    catalog._write_new(directory / f"{revision:012d}-{digest}.json", raw)
    head = {"schema_version": 1, "observation_id": intent["observation_id"], "revision": revision, "sha256": digest}
    descriptor, temporary = tempfile.mkstemp(prefix=".head-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(catalog._json(head) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "head.json")
        catalog._fsync(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return head


def _check_extents(intent, inventory):
    _require(bool(inventory), "empty_inventory")
    covered = set()
    for extent in intent["extents"]:
        relative = extent["relative"]
        selected = {name for name in inventory if relative == "." or name == relative
            or Path(name).is_relative_to(relative)}
        _require(bool(selected), "extent_absent")
        if extent["source_sha256"] is not None:
            _require(relative in inventory and inventory[relative]["sha256"] == extent["source_sha256"], "source_pin")
        covered.update(selected)
    _require(covered == set(inventory), "unregistered_extent")


def observe_existing_copy(controller: AdmissionController, *, kind: str, existing_root: Path,
                          latest_binding_id: str, extents: tuple[ObservedExtent, ...],
                          source_binding_ids: tuple[str, ...] = (), existing_copy_id: str | None = None,
                          observation_id: str | None = None, expected_revision: int = 0) -> ObservationReceipt:
    """Record existing regular bytes under current sealed authority.

    Args:
        controller: Actual current controller holding the physical lease.
        kind: Forensic/audit, phase checkpoint, or failed work observation.
        existing_root: Canonical existing directory; its payload is never changed.
        latest_binding_id: Exact current closed binding, not a history grant.
        extents: Explicit original locators, families and optional content pins.
        source_binding_ids: Verifiable binding references, not a lineage grant.
        existing_copy_id: Optional actual catalog identity to retain unchanged.
        observation_id: Optional stable identity for an initial CAS attempt.
        expected_revision: Must be zero; observations cannot be promoted/refreshed.

    Returns:
        Frozen inventory receipt without SDK, admission or cleanup capability.
    """
    stage, root, head, binding, _, _ = catalog._authority(controller, latest_binding_id)
    _require(type(expected_revision) is int and expected_revision == 0, "cas")
    _require(type(existing_root) is Path or isinstance(existing_root, Path), "root")
    existing_root = catalog._directory(existing_root)
    _require(not root.is_relative_to(existing_root)
        and not existing_root.is_relative_to(root / "admission"), "control_overlap")
    for name in ("state_root", "runtime_root"):
        current = Path(binding[name])
        _require(not current.is_relative_to(existing_root) and not existing_root.is_relative_to(current), "current_overlap")
    _require(type(extents) is tuple and all(type(e) is ObservedExtent for e in extents), "extents")
    _require(type(source_binding_ids) is tuple and all(catalog._hash(v) for v in source_binding_ids), "references")
    identity = observation_id if observation_id is not None else uuid4().hex
    existing = None
    with catalog._LOCK, stage._locked_root(root):
        catalog._authority(controller, latest_binding_id)
        for ref in source_binding_ids:
            stage.verify_closed_binding(root, ref)
        if existing_copy_id is not None:
            _require(type(existing_copy_id) is str and re.fullmatch("[0-9a-f]{32}", existing_copy_id), "existing_copy")
            record = catalog._recover_record(catalog._catalog(root) / existing_copy_id)
            _require(record.get("destination") == str(existing_root)
                and record.get("root_identity") == catalog._identity(existing_root)
                and type(record.get("revision")) is int and record["revision"] >= 1
                and catalog._hash(record.get("record_sha256")), "existing_copy")
            existing = {"copy_id": existing_copy_id, "revision": record["revision"],
                "record_sha256": record["record_sha256"]}
        intent = {"schema_version": 1, "observation_id": identity, "kind": kind, "authority": head,
            "admission_root": str(root), "admission_identity": catalog._identity(root),
            "observed_root": str(existing_root), "parent_identity": catalog._identity(existing_root.parent),
            "root_identity": catalog._identity(existing_root), "extents": [asdict(e) for e in extents],
            "source_binding_ids": list(source_binding_ids), "existing_copy": existing, "before_first_byte": False}
        _shape_intent(intent)
        namespace = _namespace(root, create=True)
        for previous in namespace.iterdir():
            prior = _recover(previous)
            _require(type(prior.get("observed_root")) is str, "prior_unknown")
            _require(prior["observed_root"] != str(existing_root)
                and (existing_copy_id is None or (prior.get("existing_copy") or {}).get("copy_id") != existing_copy_id), "duplicate")
        directory = namespace / identity
        _require(not directory.exists(), "cas")
        directory.mkdir(mode=0o700)
        catalog._fsync(namespace)
        catalog._write_new(directory / "intent.json", catalog._json(intent) + b"\n")
        first = _publish(directory, intent, None, None)
        # Any later error leaves the durable intent UNKNOWN and non-serving.
        inventory = catalog._inventory(existing_root)
        _check_extents(intent, inventory)
        _require(catalog._identity(existing_root) == intent["root_identity"]
            and catalog._identity(existing_root.parent) == intent["parent_identity"], "root_changed")
        catalog._authority(controller, latest_binding_id)
        _require(catalog._inventory(existing_root) == inventory, "inventory_changed")
        _require(catalog._identity(existing_root) == intent["root_identity"]
            and catalog._identity(existing_root.parent) == intent["parent_identity"], "root_changed")
        final = _publish(directory, intent, inventory, first)
        return ObservationReceipt(identity, final["revision"], final["sha256"], str(existing_root),
            hashlib.sha256(catalog._json(inventory)).hexdigest())


def _recover(directory):
    from src.core.runtime_admission import _stage

    result = {"observation_id": directory.name, "phase": "unknown", "retrospective": True,
        "before_first_byte": False, "non_serving": True, "capability": False,
        "lineage_complete": False, "SDK_retirement_confirmed": False,
        "cleanup": {"phase": "blocked", "owner": None, "receipt": None}}
    try:
        catalog._directory(directory)
        intent, _ = catalog._read(directory / "intent.json")
        _shape_intent(intent)
        _require(intent["observation_id"] == directory.name, "identity")
        _require(intent["admission_root"] == str(directory.parents[3]), "admission_root")
        result.update(intent)
        _, event, head = _stage()._read_copy_observation(directory, Path(intent["admission_root"]), allow_orphans=True)
        result.update(phase="observed" if event["phase"] == "observed" and not head["uncommitted_entries"] else "unknown",
            revision=head["revision"], record_sha256=head["sha256"],
            inventory=event["inventory"], inventory_sha256=event["inventory_sha256"],
            uncommitted_entries=head["uncommitted_entries"])
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        result["phase"] = "unknown"
    return result


def recover_observations(controller: AdmissionController) -> tuple[dict, ...]:
    """Read historical metadata with a live lease, even after incomplete writes.

    This validates only the controller lease and current closed HEAD; it does
    not open stores or assert a current sealed payload when a denial is unknown.
    """
    stage, root, _, _, _ = catalog._controller_lease(controller)
    with catalog._LOCK, stage._locked_root(root):
        catalog._controller_lease(controller)
        if not (root / "admission/copy-observations").exists():
            return ()
        return tuple(_recover(directory) for directory in sorted(_namespace(root).iterdir()))
