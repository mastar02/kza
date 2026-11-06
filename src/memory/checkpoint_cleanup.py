"""Current-authority retirement of exclusively withdrawn checkpoint blobs.

Observation and checkpoint records remain immutable. This owner never grants
serving, retires shared bytes, or claims aggregate-copy erasure.
"""
from __future__ import annotations

import asyncio
from contextvars import Context
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
from uuid import UUID, uuid4

from src.core.owned_work import OwnedBlockingWork
from src.memory import checkpoint_lineage as lineage
from src.memory import copy_catalog as catalog
from src.memory import copy_observation as observation

SCOPE = "checkpoint_blobs_s_only_v1"
_OWNERS = set()
_MODES = {"O": "seed-O", "N": "advance-N", "R": "rollback-R", "F": "forward-F"}


class CheckpointCleanupError(ValueError):
    """A content-free unsupported scope or unconfirmed local retirement."""


def _require(value, code):
    if not value:
        raise CheckpointCleanupError("checkpoint_cleanup_" + code)


def _sha(value):
    return hashlib.sha256(catalog._json(value)).hexdigest()


def _identifier(value):
    _require(type(value) is str and len(value) == 32
        and all(c in "0123456789abcdef" for c in value), "identity")
    return value


def _uuid(value):
    _require(type(value) is str and str(UUID(value)) == value, "operation_identity")
    return value


@dataclass(frozen=True)
class CheckpointPin:
    """An original relative checkpoint and its independently checked pins."""
    branch: str
    relative: str
    sha256: str
    expected_pins: dict


@dataclass(frozen=True)
class CheckpointBlobLayout:
    """Exact exchange scope; these metadata values convey no authority."""
    privacy_relative: str
    blob_relative: str
    checkpoints: tuple[CheckpointPin, ...]
    source_anchors: tuple[dict, ...]
    preserved_conversation_ids: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointCleanupClaim:
    """A local claim; serializing its identity cannot transfer ownership."""
    observation_id: str
    claim_id: str
    operation_id: str
    conversation_id: str
    observation_revision: int
    observation_record_sha256: str
    cleanup_revision: int
    cleanup_record_sha256: str
    authority_head: dict
    layout_sha256: str
    plan_sha256: str

    def to_dict(self) -> dict:
        """Return metadata without a callable capability."""
        return asdict(self)

    def __reduce_ex__(self, _protocol):
        raise TypeError("checkpoint_cleanup_claim_not_serializable")


@dataclass(frozen=True)
class CheckpointCleanupReceipt:
    """Confirm only the selected physical S-only blobs, after real return."""
    observation_id: str
    claim_id: str
    operation_id: str
    conversation_id: str
    cleanup_revision: int
    record_sha256: str
    plan_sha256: str
    removed_files: int
    newly_removed_files: int
    retained_blobs: int
    preserved_inventory_sha256: str
    analysis_sha256: str
    scope: str = SCOPE
    representation_complete: bool = True
    aggregate_copy_complete: bool = False
    other_representations: str = "blocked"
    native_returned: bool = True

    def to_dict(self) -> dict:
        """Return bounded receipt metadata, never an aggregate grant."""
        return asdict(self)


def _layout(value):
    _require(type(value) is CheckpointBlobLayout and type(value.checkpoints) is tuple
        and len(value.checkpoints) == 4 and all(type(p) is CheckpointPin for p in value.checkpoints), "layout")
    catalog._relative(value.privacy_relative)
    catalog._relative(value.blob_relative)
    _require(value.privacy_relative != "." and value.blob_relative != ".", "layout")
    _require({p.branch for p in value.checkpoints} == set(_MODES), "checkpoint_set")
    names = set()
    for pin in value.checkpoints:
        catalog._relative(pin.relative)
        _require(pin.relative != "." and pin.relative not in names and catalog._hash(pin.sha256)
            and not Path(pin.relative).is_relative_to(value.blob_relative)
            and type(pin.expected_pins) is dict and set(pin.expected_pins) == lineage.PIN_FIELDS
            and all(type(v) is str for v in pin.expected_pins.values())
            and pin.expected_pins["mode"] == _MODES[pin.branch], "checkpoint_pin")
        names.add(pin.relative)
    _require(type(value.source_anchors) is tuple and bool(value.source_anchors)
        and all(type(s) is dict and lineage._source(s) for s in value.source_anchors)
        and type(value.preserved_conversation_ids) is tuple, "source_scope")
    for identifier in value.preserved_conversation_ids:
        _uuid(identifier)
    _require(len(set(value.preserved_conversation_ids)) == len(value.preserved_conversation_ids), "source_scope")
    return _sha(asdict(value))


def _copy_layout(value):
    """Detach validated JSON metadata from the caller's mutable containers."""
    _layout(value)
    return CheckpointBlobLayout(value.privacy_relative, value.blob_relative,
        tuple(CheckpointPin(p.branch, p.relative, p.sha256, json.loads(catalog._json(p.expected_pins)))
            for p in value.checkpoints), tuple(json.loads(catalog._json(s)) for s in value.source_anchors),
        tuple(value.preserved_conversation_ids))


def _mkdir(path):
    if not path.exists():
        _mkdir(path.parent)
        path.mkdir(mode=0o700)
        catalog._fsync(path.parent)
    catalog._directory(path)


def _base(root, identifier):
    _identifier(identifier)
    return root / "admission/checkpoint-cleanup/v1/claims" / identifier


def _head_shape(value):
    return (type(value) is dict and set(value) == {"schema_version", "revision", "sha256"}
        and type(value["schema_version"]) is int and value["schema_version"] == 1
        and type(value["revision"]) is int and value["revision"] > 0 and catalog._hash(value["sha256"]))


def _record_shape(value):
    keys = {"schema_version", "scope", "observation_id", "claim_id", "operation_id", "conversation_id",
        "authority_head", "observation_revision", "observation_record_sha256", "layout_sha256",
        "tombstone_sha256", "plan", "plan_sha256", "phase", "native_started", "native_returned",
        "inflight", "removed", "receipt", "error_code"}
    _require(type(value) is dict and set(value) == keys and type(value["schema_version"]) is int
        and value["schema_version"] == 1 and value["scope"] == SCOPE, "journal_schema")
    for key in ("observation_id", "claim_id"):
        _identifier(value[key])
    for key in ("operation_id", "conversation_id"):
        _uuid(value[key])
    for key in ("observation_record_sha256", "layout_sha256", "tombstone_sha256", "plan_sha256"):
        _require(catalog._hash(value[key]), "journal_pin")
    _require(type(value["observation_revision"]) is int and value["observation_revision"] > 0
        and type(value["native_started"]) is bool and type(value["native_returned"]) is bool
        and value["phase"] in {"claimed", "running", "failed", "unknown", "complete"}, "journal_schema")
    head = value["authority_head"]
    _require(type(head) is dict and set(head) == {"schema_version", "generation", "binding_id", "status", "run_id"}
        and type(head["schema_version"]) is int and head["schema_version"] == 1
        and type(head["generation"]) is int and head["generation"] > 0
        and catalog._hash(head["binding_id"]) and head["status"] == "closed" and head["run_id"] is None, "journal_authority")
    plan = value["plan"]
    _require(type(plan) is dict and set(plan) == {"inventory", "selected", "checkpoints", "blob_relative"}
        and _sha(plan) == value["plan_sha256"] and type(plan["inventory"]) is dict
        and type(plan["selected"]) is dict and bool(plan["selected"])
        and type(plan["checkpoints"]) is dict, "journal_plan")
    catalog._relative(plan["blob_relative"])
    for digest, item in plan["selected"].items():
        _require(catalog._hash(digest) and type(item) is dict
            and set(item) == {"relative", "sha256", "size", "identity"}
            and item["relative"] == plan["blob_relative"] + "/" + digest + ".bin"
            and item["sha256"] == digest and type(item["size"]) is int and item["size"] >= 0
            and catalog._identity_shape(item["identity"])
            and plan["inventory"].get(item["relative"]) == {k: item[k] for k in ("sha256", "size", "identity")}, "journal_plan")
    _require(type(value["removed"]) is list and all(type(v) is str for v in value["removed"])
        and len(set(value["removed"])) == len(value["removed"])
        and set(value["removed"]) <= set(plan["selected"])
        and (value["inflight"] is None or type(value["inflight"]) is str and value["inflight"] in plan["selected"]), "journal_progress")
    _require(value["error_code"] is None or type(value["error_code"]) is str
        and value["error_code"].isascii() and value["error_code"].isidentifier(), "journal_error")
    if value["phase"] == "complete":
        r = value["receipt"]
        _require(type(r) is dict and set(r) == set(CheckpointCleanupReceipt.__dataclass_fields__)
            and all(r[k] == value[k] for k in ("scope", "observation_id", "claim_id", "operation_id",
                                               "conversation_id", "plan_sha256"))
            and r["representation_complete"] is True and r["aggregate_copy_complete"] is False
            and r["native_returned"] is True and r["other_representations"] == "blocked"
            and value["native_started"] is True and value["native_returned"] is True
            and value["inflight"] is None and set(value["removed"]) == set(plan["selected"])
            and all(type(r[k]) is int and r[k] >= 0 for k in ("removed_files", "newly_removed_files", "retained_blobs"))
            and r["removed_files"] == len(plan["selected"])
            and type(r["cleanup_revision"]) is int and r["cleanup_revision"] > 0
            and all(catalog._hash(r[k]) for k in ("record_sha256", "preserved_inventory_sha256", "analysis_sha256")), "journal_receipt")
    else:
        _require(value["receipt"] is None, "journal_receipt")


def _transition_shape(before, after):
    immutable = ("schema_version", "scope", "observation_id", "operation_id", "conversation_id",
        "observation_revision", "observation_record_sha256", "layout_sha256", "plan_sha256", "plan", "tombstone_sha256")
    _require(all(before[k] == after[k] for k in immutable), "journal_immutable")
    if before["claim_id"] != after["claim_id"]:
        _require(before["phase"] != "complete" and after["phase"] == "claimed"
            and after["native_started"] is False and after["native_returned"] is False
            and after["removed"] == before["removed"] and after["inflight"] == before["inflight"], "journal_resume")
    else:
        _require(after["authority_head"] == before["authority_head"]
            and before["phase"] in {"claimed", "running"}
            and after["phase"] in {"running", "failed", "unknown", "complete"}
            and set(before["removed"]) <= set(after["removed"])
            and (not before["native_started"] or after["native_started"]), "journal_transition")


def _read_state(directory):
    catalog._directory(directory)
    head, _ = catalog._read(directory / "head.json")
    _require(_head_shape(head), "journal_head")
    previous, saved = None, None
    committed = {"head.json"}
    for revision in range(1, head["revision"] + 1):
        candidates = list(directory.glob(f"{revision:012d}-*.json"))
        _require(len(candidates) == 1, "journal_chain")
        path = candidates[0]
        event, raw = catalog._read(path)
        digest = hashlib.sha256(raw).hexdigest()
        _require(path.name == f"{revision:012d}-{digest}.json" and type(event) is dict
            and set(event) == {"schema_version", "revision", "previous", "state"}
            and type(event["schema_version"]) is int and event["schema_version"] == 1
            and type(event["revision"]) is int and event["revision"] == revision
            and event["previous"] == previous, "journal_chain")
        _record_shape(event["state"])
        if event["state"]["phase"] == "complete":
            receipt = event["state"]["receipt"]
            _require(receipt["cleanup_revision"] == revision and receipt["record_sha256"] == previous,
                "journal_receipt_pin")
        if saved is not None:
            _transition_shape(saved, event["state"])
        saved, previous = event["state"], digest
        committed.add(path.name)
    _require(previous == head["sha256"], "journal_head")
    return saved, head, sorted({p.name for p in directory.iterdir()} - committed)


def _append(directory, record, expected):
    _record_shape(record)
    if expected is None:
        _require(not (directory / "head.json").exists() and not list(directory.iterdir())
            and record["phase"] == "claimed" and not record["native_started"]
            and not record["native_returned"] and not record["removed"] and record["inflight"] is None, "claim_cas")
        revision, previous = 1, None
    else:
        before, actual, orphans = _read_state(directory)
        _require(not orphans and actual == expected, "claim_cas")
        _transition_shape(before, record)
        revision, previous = actual["revision"] + 1, actual["sha256"]
    if record["phase"] == "complete":
        _require(record["receipt"]["cleanup_revision"] == revision
            and record["receipt"]["record_sha256"] == previous, "journal_receipt_pin")
    event = {"schema_version": 1, "revision": revision, "previous": previous, "state": record}
    raw = catalog._json(event) + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    catalog._write_new(directory / f"{revision:012d}-{digest}.json", raw)
    head = {"schema_version": 1, "revision": revision, "sha256": digest}
    from src.core.runtime_admission import _stage
    _stage()._atomic_record(directory / "head.json", head)
    catalog._fsync(directory)
    return head


def _tombstone(stage, binding, relative, operation_id):
    _require(relative in binding["schemas"], "current_database")
    with stage._database_image(Path(binding["state_root"]) / relative) as db:
        row = db.execute("SELECT o.id,o.conversation_id,o.requested_at,o.requested_by,t.deleted_at,t.deleted_by "
            "FROM conversation_erasures o JOIN conversation_tombstones t "
            "ON t.conversation_id=o.conversation_id WHERE o.id=?", (operation_id,)).fetchone()
        _require(row is not None, "current_tombstone")
        result = dict(row)
    _uuid(result["conversation_id"])
    return result


def _specs(destination, layout):
    return {pin.branch: {"path": destination / pin.relative, "expected_sha256": pin.sha256,
        "expected_pins": pin.expected_pins} for pin in layout.checkpoints}


def _source_scope(destination, layout):
    pin, = (p for p in layout.checkpoints if p.branch == "O")
    scope = lineage.checkpoint_source_scope(destination / pin.relative, expected_sha256=pin.sha256)
    _require(catalog._json(scope["source_anchors"]) == catalog._json(layout.source_anchors)
        and scope["preserved_conversation_ids"] == layout.preserved_conversation_ids, "source_scope")
    return scope


def _verify_bindings(stage, root, observed, layout):
    refs = observed["source_binding_ids"]
    _require(type(refs) is list and len(refs) == len(set(refs)) == 4, "source_bindings")
    bindings = {identity: stage.verify_closed_binding(root, identity) for identity in refs}
    matched = set()
    for pin in layout.checkpoints:
        fields = pin.expected_pins
        matches = [identity for identity, binding in bindings.items()
            if fields["source_commit"] == binding["code_commit"]
            and fields["runtime_root"] == binding["runtime_root"]
            and fields["state_root"] == str(Path(binding["state_root"]) / Path(layout.privacy_relative).parent)
            and fields["release_sha256"] == binding["release_sha256"]]
        _require(len(matches) == 1 and matches[0] not in matched, "checkpoint_binding")
        matched.add(matches[0])
    _require(matched == set(refs), "source_bindings")


def _inventory(destination, original, selected, allowed_absent):
    current = catalog._inventory(destination)
    absent = set(original) - set(current)
    allowed = {selected[d]["relative"] for d in allowed_absent}
    _require(absent <= allowed and current == {k: v for k, v in original.items() if k not in absent}, "inventory_changed")
    expected_dirs = {Path(name).parent.as_posix() for name in original}
    expected_dirs.discard(".")
    expected_dirs |= {p.as_posix() for name in tuple(expected_dirs) for p in Path(name).parents if p.as_posix() != "."}
    actual_dirs = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_dir()}
    _require(actual_dirs == expected_dirs, "inventory_scope")
    return current, {d: {"size": row["size"], "identity": row["identity"]}
        for d, row in selected.items() if row["relative"] in absent}


def _analyze(destination, layout, conversation, unavailable):
    report = lineage.analyze_checkpoints(_specs(destination, layout), destination / layout.blob_relative,
        source_anchors=layout.source_anchors, withdrawn_conversation_id=conversation,
        preserved_conversation_ids=layout.preserved_conversation_ids, unavailable_blobs=unavailable)
    _require(report["complete_reference_scope"] is True and all(item["classification"] != "blocked"
        for item in report["blobs"].values()), "unknown_lineage")
    return report


def _plan(destination, layout, original, conversation):
    report = _analyze(destination, layout, conversation, {})
    checkpoints = {p.relative: {"sha256": p.sha256, "expected_pins": p.expected_pins} for p in layout.checkpoints}
    expected = set(checkpoints) | {layout.blob_relative + "/" + digest + ".bin" for digest in report["blobs"]}
    _require(set(original) == expected, "inventory_scope")
    for relative, pin in checkpoints.items():
        _require(original[relative]["sha256"] == pin["sha256"], "checkpoint_observation_pin")
    selected = {}
    for digest, item in report["blobs"].items():
        relative = layout.blob_relative + "/" + digest + ".bin"
        physical = original[relative]
        _require(physical["sha256"] == digest, "blob_observation_pin")
        if item["classification"] == "S_only":
            selected[digest] = dict(physical, relative=relative)
    _require(bool(selected), "no_new_s_only")
    result = {"inventory": original, "selected": selected, "checkpoints": checkpoints, "blob_relative": layout.blob_relative}
    _inventory(destination, original, selected, set())
    return result


def _unlink_blob(owner, relative: str, pin: dict) -> bool:
    """Unlink one pinned file, fsync its directory and consume both FD closes."""
    path = owner.destination / relative
    catalog._directory(path.parent)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    owner._native_fds.add(directory)
    fd = None
    try:
        info = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            and {"device": info.st_dev, "inode": info.st_ino} == pin["identity"]
            and info.st_size == pin["size"], "unlink_identity")
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        owner._native_fds.add(fd)
        current = os.fstat(fd)
        _require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns), "unlink_identity")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        _require(digest.hexdigest() == pin["sha256"], "unlink_pin")
        current = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        _require(current.st_nlink == 1 and (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns), "unlink_identity")
        os.unlink(path.name, dir_fd=directory)
        os.fsync(directory)
    finally:
        try:
            if fd is not None:
                os.close(fd)
                owner._native_fds.remove(fd)
        finally:
            os.close(directory)
            owner._native_fds.remove(directory)
    return True


def _perform(owner):
    owner._current()
    plan = owner._record["plan"]
    allowed = set(owner._record["removed"])
    if owner._record["inflight"] is not None:
        allowed.add(owner._record["inflight"])
    _, unavailable = _inventory(owner.destination, plan["inventory"], plan["selected"], allowed)
    report = _analyze(owner.destination, owner.layout, owner._claim.conversation_id, unavailable)
    _require(set(plan["selected"]) == {d for d, v in report["blobs"].items()
        if v["classification"] == "S_only"}, "selection_changed")
    new = 0
    for digest, item in sorted(plan["selected"].items()):
        owner._current()
        if digest not in unavailable:
            owner._transition(phase="running", inflight=digest)
            _require(_unlink_blob(owner, item["relative"], item) is True, "unlink_return")
            new += 1
        else:
            # Resume proves current absence durably; a historical intent alone
            # is never interpreted as a completed native unlink.
            directory = os.open(owner.destination / owner.layout.blob_relative,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            owner._native_fds.add(directory)
            try:
                _require(not (owner.destination / item["relative"]).exists(), "resume_file_returned")
                os.fsync(directory)
            finally:
                os.close(directory)
                owner._native_fds.remove(directory)
        owner._transition(phase="running", removed=sorted(set(owner._record["removed"]) | {digest}), inflight=None)
    remaining, unavailable = _inventory(owner.destination, plan["inventory"], plan["selected"], set(plan["selected"]))
    _require(set(unavailable) == set(plan["selected"]) and not owner._native_fds, "retirement_unconfirmed")
    after = _analyze(owner.destination, owner.layout, owner._claim.conversation_id, unavailable)
    _require(set(plan["selected"]) == {d for d, v in after["blobs"].items() if v["classification"] == "S_only"}, "selection_changed")
    owner._current()
    return {"removed_files": len(plan["selected"]), "newly_removed_files": new,
        "retained_blobs": len(after["blobs"]) - len(plan["selected"]),
        "preserved_inventory_sha256": _sha(remaining), "analysis_sha256": _sha(after)}


class CheckpointBlobCleanup:
    """Own a selective physical cleanup under the actual closed controller."""

    def __init__(self, controller, observation_id: str, *, layout: CheckpointBlobLayout) -> None:
        self.controller, self.observation_id = controller, _identifier(observation_id)
        self.layout = layout = _copy_layout(layout)
        self._layout_sha = _layout(layout)
        stage, root, head, binding, fd, identity = catalog._authority(controller)
        self._stage, self.root, self._authority_head, self._binding = stage, root, head, binding
        self._fd, self._fd_identity = fd, identity
        self._observation_directory = observation._namespace(root) / observation_id
        self._intent, self._event, self._observation_head = stage._read_copy_observation(self._observation_directory, root)
        _require(self._intent["kind"] == "phase_checkpoint" and self._event["phase"] == "observed", "observation_required")
        self.destination = catalog._directory(Path(self._intent["observed_root"]))
        _require(catalog._identity(self.destination) == self._intent["root_identity"]
            and catalog._identity(self.destination.parent) == self._intent["parent_identity"], "observation_root")
        for current in (Path(binding["state_root"]), Path(binding["runtime_root"]), root / "admission"):
            _require(not self.destination.is_relative_to(current) and not current.is_relative_to(self.destination), "inactive_exchange_required")
        _verify_bindings(stage, root, self._intent, layout)
        self._directory = _base(root, observation_id)
        self._record = self._head = self._claim = self._claim_sha = self._tombstone_sha = None
        self._task = self._closing = None
        self._closed = False
        self._native_fds = set()
        self._work = OwnedBlockingWork(capacity=1, close_timeout_s=1, isolate_context=True)
        self._lease_fd, self._lock_fd = os.dup(fd), None
        try:
            locks = root / "admission/checkpoint-cleanup/v1/locks"
            _mkdir(locks)
            self._lock_fd = os.open(locks / (observation_id + ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            info = os.fstat(self._lock_fd)
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "claim_lock")
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
            os.close(self._lease_fd)
            self._lease_fd = self._lock_fd = None
            raise
        _OWNERS.add(self)

    def _current(self):
        _require(not self._closed and self._lease_fd is not None and _layout(self.layout) == self._layout_sha, "owner_closed_or_changed")
        stage, root, head, binding, fd, identity = catalog._authority(self.controller)
        _require(stage is self._stage and root == self.root and head == self._authority_head
            and binding == self._binding and fd == self._fd and identity == self._fd_identity, "authority_changed")
        info = os.fstat(self._lease_fd)
        _require({"device": info.st_dev, "inode": info.st_ino} == identity, "lease_changed")
        intent, event, saved = stage._read_copy_observation(self._observation_directory, root)
        _require(intent == self._intent and event == self._event and saved == self._observation_head
            and catalog._identity(self.destination) == intent["root_identity"]
            and catalog._identity(self.destination.parent) == intent["parent_identity"], "observation_changed")
        _verify_bindings(stage, root, intent, self.layout)
        if self._claim is not None:
            tombstone = _tombstone(stage, binding, self.layout.privacy_relative, self._claim.operation_id)
            _require(tombstone["conversation_id"] == self._claim.conversation_id
                and _sha(tombstone) == self._tombstone_sha, "tombstone_changed")

    def claim(self, operation_id: str, *, expected_observation_revision: int,
              expected_record_sha256: str, expected_cleanup_revision: int = 0) -> CheckpointCleanupClaim:
        """Persist an exact current claim and original plan before any unlink."""
        self._current()
        _uuid(operation_id)
        _require(self._claim is None and type(expected_observation_revision) is int
            and expected_observation_revision == self._observation_head["revision"]
            and catalog._hash(expected_record_sha256) and expected_record_sha256 == self._observation_head["sha256"]
            and type(expected_cleanup_revision) is int and expected_cleanup_revision >= 0, "claim_cas")
        tombstone = _tombstone(self._stage, self._binding, self.layout.privacy_relative, operation_id)
        scope = _source_scope(self.destination, self.layout)
        _require(scope["withdrawn_conversation_id"] == tombstone["conversation_id"], "operation_source")
        self._tombstone_sha = _sha(tombstone)
        previous, old = None, None
        if self._directory.exists() or self._directory.is_symlink():
            old, previous, orphans = _read_state(self._directory)
            _require(not orphans and previous["revision"] == expected_cleanup_revision
                and old["phase"] != "complete" and old["observation_id"] == self.observation_id
                and old["operation_id"] == operation_id and old["conversation_id"] == tombstone["conversation_id"]
                and old["layout_sha256"] == self._layout_sha and old["tombstone_sha256"] == self._tombstone_sha
                and old["plan"]["inventory"] == self._event["inventory"], "resume_blocked")
            plan = old["plan"]
            allowed = set(old["removed"]) | ({old["inflight"]} if old["inflight"] is not None else set())
            _, unavailable = _inventory(self.destination, plan["inventory"], plan["selected"], allowed)
            report = _analyze(self.destination, self.layout, tombstone["conversation_id"], unavailable)
            _require(set(plan["selected"]) == {d for d, v in report["blobs"].items()
                if v["classification"] == "S_only"}, "selection_changed")
        else:
            _require(expected_cleanup_revision == 0, "claim_cas")
            plan = _plan(self.destination, self.layout, self._event["inventory"], tombstone["conversation_id"])
        claim_id = uuid4().hex
        record = {"schema_version": 1, "scope": SCOPE, "observation_id": self.observation_id,
            "claim_id": claim_id, "operation_id": operation_id, "conversation_id": tombstone["conversation_id"],
            "authority_head": self._authority_head, "observation_revision": expected_observation_revision,
            "observation_record_sha256": expected_record_sha256, "layout_sha256": self._layout_sha,
            "tombstone_sha256": self._tombstone_sha, "plan": plan, "plan_sha256": _sha(plan),
            "phase": "claimed", "native_started": False, "native_returned": False,
            "inflight": old["inflight"] if old is not None else None,
            "removed": old["removed"] if old is not None else [], "receipt": None, "error_code": None}
        self._current()
        _mkdir(self._directory)
        head = _append(self._directory, record, previous)
        self._record, self._head = record, head
        self._claim = CheckpointCleanupClaim(self.observation_id, claim_id, operation_id, tombstone["conversation_id"],
            expected_observation_revision, expected_record_sha256, head["revision"], head["sha256"],
            json.loads(catalog._json(self._authority_head)), self._layout_sha, record["plan_sha256"])
        self._claim_sha = _sha(self._claim.to_dict())
        return self._claim

    def _transition(self, **changes):
        self._current()
        record = dict(self._record, **changes)
        head = _append(self._directory, record, self._head)
        self._record, self._head = record, head

    def _native(self):
        try:
            return _perform(self), None
        except BaseException as error:
            return None, type(error).__name__

    async def _finish(self):
        native = asyncio.create_task(self._work.run(self._native), context=Context())
        self._native_task = native
        cancelled = False
        while not native.done():
            try:
                await asyncio.shield(native)
            except asyncio.CancelledError:
                cancelled = True
        data, error = native.result()
        if error is not None:
            self._transition(phase="unknown", native_returned=True, error_code=error)
            raise CheckpointCleanupError("checkpoint_cleanup_native_failed")
        self._current()
        _require(not self._native_fds, "native_close_unknown")
        receipt = CheckpointCleanupReceipt(self.observation_id, self._claim.claim_id, self._claim.operation_id,
            self._claim.conversation_id, self._head["revision"] + 1, self._head["sha256"],
            self._claim.plan_sha256, **data)
        self._transition(phase="complete", native_returned=True, receipt=receipt.to_dict())
        if cancelled:
            raise asyncio.CancelledError
        return receipt

    async def clean(self, claim: CheckpointCleanupClaim) -> CheckpointCleanupReceipt:
        """Observe the same owned Task; caller cancellation does not retire it."""
        _require(claim is self._claim and type(claim) is CheckpointCleanupClaim
            and _sha(claim.to_dict()) == self._claim_sha, "local_claim_required")
        if self._task is None:
            self._transition(phase="running", native_started=True)
            self._task = asyncio.create_task(self._finish(), context=Context())
        return await asyncio.shield(self._task)

    async def aclose(self) -> None:
        """Drain actual work before releasing the duplicated authority lease."""
        if self._closing is None:
            async def close():
                if self._task is not None:
                    await asyncio.gather(self._task, return_exceptions=True)
                await self._work.begin_close()
                _require(not self._native_fds, "native_close_unknown")
                if self._lock_fd is not None:
                    os.close(self._lock_fd)
                    self._lock_fd = None
                if self._lease_fd is not None:
                    os.close(self._lease_fd)
                    self._lease_fd = None
                self._closed = True
                _OWNERS.discard(self)
            self._closing = asyncio.create_task(close(), context=Context())
        cancelled = False
        while not self._closing.done():
            try:
                await asyncio.shield(self._closing)
            except asyncio.CancelledError:
                cancelled = True
        self._closing.result()
        if cancelled:
            raise asyncio.CancelledError


def read_checkpoint_cleanup(controller, observation_id: str) -> dict:
    """Read durable metadata under current authority; never return a grant."""
    _, root, _, _, _, _ = catalog._authority(controller)
    directory = _base(root, observation_id)
    view = {"schema_version": 1, "scope": SCOPE, "observation_id": observation_id,
        "phase": "unclaimed", "cleanup_revision": 0, "record_sha256": None, "claim_id": None,
        "operation_id": None, "conversation_id": None, "native_started": False,
        "native_returned": False, "receipt": None, "error_code": None, "capability": False,
        "aggregate_copy_complete": False, "other_representations": "blocked"}
    if directory.exists() or directory.is_symlink():
        try:
            record, head, orphans = _read_state(directory)
            _require(record["observation_id"] == observation_id, "journal_identity")
            view.update({k: record[k] for k in ("phase", "claim_id", "operation_id", "conversation_id",
                "native_started", "native_returned", "receipt", "error_code")})
            view.update(cleanup_revision=head["revision"], record_sha256=head["sha256"])
            if orphans:
                view.update(phase="unknown", receipt=None, error_code="journal_orphan")
            elif record["phase"] in {"claimed", "running"}:
                view["phase"] = "pending"
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            view.update(phase="unknown", receipt=None, error_code="journal_unknown")
    return view
