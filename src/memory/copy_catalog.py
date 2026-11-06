"""Durable, payload-free copy obligations; preparation never means cleanup."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import threading
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

if TYPE_CHECKING:
    from scripts.stage_runtime import AdmissionController

KINDS = frozenset({"snapshot", "derived_state"})
FAMILIES = frozenset({"managed_payload", "semantic_index", "ambient_capture", "collectors",
    "contexts", "events", "suggestions", "diagnostics", "voices", "streams", "history", "authority"})
REPRESENTATIONS = frozenset({"raw", "state", "sqlite", "sdk_work", "blob", "logical_export",
    "manifest", "preparation", "history_archive", "derived_state"})
_LOCK = threading.RLock()
_LIVE = {}
_KEY = object()


class CopyCatalogError(ValueError):
    """A content-free failure to establish copy authority or persistence."""


def _require(value, code):
    if not value:
        raise CopyCatalogError("copy_catalog_" + code)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _hash(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _version(value):
    return type(value) is str and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}(?:(?:a|b|rc|\.post|\.dev)[0-9]+)?(?:\+[A-Za-z0-9.]+)?", value) is not None


def _identity(path):
    info = path.stat()
    return {"device": info.st_dev, "inode": info.st_ino}


def _identity_shape(value):
    return (type(value) is dict and set(value) == {"device", "inode"}
            and all(type(v) is int for v in value.values())
            and value["device"] >= 0 and value["inode"] > 0)


def _directory(path):
    _require(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir(), "directory")
    return path


def _relative(value):
    _require(type(value) is str and value and PurePosixPath(value).as_posix() == value
             and not PurePosixPath(value).is_absolute()
             and ".." not in PurePosixPath(value).parts, "relative_path")
    return value


def _fsync(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync(path.parent)


def _read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "record_file")
        raw = stream.read()
    value = json.loads(raw)
    _require(type(value) is dict and _json(value) + b"\n" == raw, "record_canonical")
    return value, raw


def _catalog(root, *, create=False):
    admission = _directory(root / "admission")
    path = admission / "copies"
    if create and not path.exists():
        path.mkdir(mode=0o700)
        _fsync(admission)
    return _directory(path)


def _record_shape(value):
    _require(type(value) is dict and set(value) == {"schema_version", "copy_id", "kind", "authority",
        "source_binding_ids", "source_snapshot_refs", "destination", "parent_identity", "root_identity",
        "preparation_owner", "preparation", "cleanup", "extents", "sdk", "prepared_record", "error_code", "finished"}, "record_schema")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1
        and type(value["copy_id"]) is str and re.fullmatch("[0-9a-f]{32}", value["copy_id"])
        and value["kind"] in KINDS and _identity_shape(value["parent_identity"])
        and (value["root_identity"] is None or _identity_shape(value["root_identity"]))
        and value["preparation"] in {"reserved", "writing", "prepared", "failed", "unknown"}
        and type(value["finished"]) is bool and (not value["finished"] or value["preparation"] == "prepared")
        and type(value["cleanup"]) is dict and value["cleanup"] == {"phase": "blocked", "owner": None, "receipt": None}, "record_schema")
    _require(value["preparation"] not in {"writing", "prepared"} or value["root_identity"] is not None,
             "root_identity_required")
    head = value["authority"]
    _require(type(head) is dict and set(head) == {"schema_version", "generation", "binding_id", "status", "run_id"}
        and type(head["schema_version"]) is int and head["schema_version"] == 1
        and type(head["generation"]) is int and head["generation"] >= 1
        and _hash(head["binding_id"]) and head["status"] == "closed" and head["run_id"] is None, "record_authority")
    owner = value["preparation_owner"]
    _require(type(owner) is dict and set(owner) == {"epoch", "process_id", "lease_fd", "lease_identity"}
        and type(owner["epoch"]) is str and re.fullmatch("[0-9a-f]{32}", owner["epoch"])
        and type(owner["process_id"]) is int and owner["process_id"] > 0
        and type(owner["lease_fd"]) is int and owner["lease_fd"] >= 0
        and _identity_shape(owner["lease_identity"]), "record_owner")
    _require(type(value["destination"]) is str and Path(value["destination"]).is_absolute()
        and type(value["source_binding_ids"]) is list and bool(value["source_binding_ids"])
        and all(_hash(item) for item in value["source_binding_ids"])
        and len(set(value["source_binding_ids"])) == len(value["source_binding_ids"])
        and type(value["source_snapshot_refs"]) is list, "record_sources")
    for ref in value["source_snapshot_refs"]:
        _require(type(ref) is dict and set(ref) == {"source_binding_id", "snapshot_root", "root_identity", "manifest_sha256"}
            and ref["source_binding_id"] in value["source_binding_ids"] and _hash(ref["manifest_sha256"])
            and type(ref["snapshot_root"]) is str and Path(ref["snapshot_root"]).is_absolute()
            and _identity_shape(ref["root_identity"]), "record_sources")
    _require(type(value["extents"]) is dict and type(value["sdk"]) is dict, "record_extents")
    for key, extent in value["extents"].items():
        _require(_hash(key) and type(extent) is dict and set(extent) == {"relative", "family", "representation"}
            and extent["family"] in FAMILIES and extent["representation"] in REPRESENTATIONS, "record_extents")
        _relative(extent["relative"])
        _require(_sha(_json(extent)) == key, "record_extents")
    for key, sdk in value["sdk"].items():
        _require(type(key) is str and re.fullmatch("[0-9a-f]{32}", key)
            and type(sdk) is dict and set(sdk) == {"extent", "sdk_version", "phase"}
            and sdk["extent"] in value["extents"]
            and value["extents"][sdk["extent"]]["representation"] == "sdk_work"
            and _version(sdk["sdk_version"])
            and sdk["phase"] in {"not_entered", "constructor_entered", "handle_returned",
                "close_entered", "close_returned", "unknown"}, "record_sdk")
    _require(value["error_code"] is None or type(value["error_code"]) is str
        and re.fullmatch("[A-Za-z_][A-Za-z0-9_]{0,79}", value["error_code"]), "record_error")
    if value["prepared_record"] is not None:
        evidence = value["prepared_record"]
        _require(type(evidence) is dict and set(evidence) == {"relative", "sha256", "size", "inventory", "extent_observations"}
            and _hash(evidence["sha256"]) and type(evidence["size"]) is int and evidence["size"] >= 0
            and type(evidence["inventory"]) is dict and type(evidence["extent_observations"]) is dict, "record_prepared")
        _relative(evidence["relative"])
        for relative, item in evidence["inventory"].items():
            _relative(relative)
            _require(type(item) is dict and set(item) == {"sha256", "size", "identity"}
                and _hash(item["sha256"]) and type(item["size"]) is int and item["size"] >= 0
                and _identity_shape(item["identity"]), "record_inventory")
        _require(set(evidence["extent_observations"]) == set(value["extents"]), "record_inventory")
        for item in evidence["extent_observations"].values():
            _require(type(item) is dict and set(item) == {"status", "identity"}
                and (item["status"] == "present" and _identity_shape(item["identity"])
                    or item["status"] == "absent_observed" and item["identity"] is None), "record_inventory")
        _require(value["preparation"] in {"prepared", "failed", "unknown"}, "record_prepared")
        if value["preparation"] == "prepared":
            _require(all(sdk["phase"] == "close_returned" for sdk in value["sdk"].values()), "record_prepared")
    else:
        _require(value["preparation"] != "prepared", "record_prepared")
    _json(value)


def _read_record(directory, *, allow_orphans=False):
    _directory(directory)
    head, _ = _read(directory / "head.json")
    _require(set(head) == {"schema_version", "copy_id", "revision", "sha256"}
        and type(head["schema_version"]) is int and head["schema_version"] == 1
        and type(head["revision"]) is int and head["revision"] >= 1
        and head["copy_id"] == directory.name and _hash(head["sha256"]), "head_schema")
    revision, expected = head["revision"], head["sha256"]
    selected, names, newer = None, set(), None
    while revision:
        name = f"{revision:012d}-{expected}.json"
        event, raw = _read(directory / name)
        names.add(name)
        _require(_sha(raw) == expected and set(event) == {"schema_version", "copy_id", "revision", "previous_sha256", "record"}
            and type(event["schema_version"]) is int and event["schema_version"] == 1
            and type(event["revision"]) is int and event["revision"] == revision
            and event["copy_id"] == directory.name
            and (event["previous_sha256"] is None if revision == 1 else _hash(event["previous_sha256"])), "chain")
        _record_shape(event["record"])
        _require(event["record"]["copy_id"] == directory.name, "chain")
        if revision == 1:
            _initial(event["record"])
        if newer is not None:
            _transition(event["record"], newer)
        newer = event["record"]
        if selected is None:
            selected = event["record"]
        expected, revision = event["previous_sha256"], revision - 1
    _require(allow_orphans or {p.name for p in directory.iterdir()} == names | {"head.json"}, "chain_orphan")
    return selected, head


def _initial(record):
    _require(record["preparation"] == "reserved" and record["root_identity"] is None
        and not record["extents"] and not record["sdk"] and record["prepared_record"] is None
        and record["error_code"] is None and not record["finished"], "initial_reservation")


def _transition(before, after):
    immutable = {"schema_version", "copy_id", "kind", "authority", "source_binding_ids",
        "source_snapshot_refs", "destination", "parent_identity", "preparation_owner", "cleanup"}
    _require(all(before[key] == after[key] for key in immutable), "record_immutable")
    _require(before["root_identity"] is None or before["root_identity"] == after["root_identity"], "record_immutable")
    phases = {"reserved": {"reserved", "writing", "failed", "unknown"},
        "writing": {"writing", "prepared", "failed", "unknown"}, "prepared": {"prepared", "failed", "unknown"},
        "failed": {"failed", "unknown"}, "unknown": {"unknown"}}
    _require(after["preparation"] in phases[before["preparation"]], "phase_transition")
    _require(not before["finished"] or before == after, "finished_immutable")
    _require(not after["finished"] or before["preparation"] == "prepared", "finish_transition")
    _require(all(key in after["extents"] and after["extents"][key] == item
        for key, item in before["extents"].items()), "extent_history")
    sdk_phases = {"not_entered": {"not_entered", "constructor_entered"},
        "constructor_entered": {"constructor_entered", "handle_returned", "close_entered", "unknown"},
        "handle_returned": {"handle_returned", "close_entered", "unknown"},
        "close_entered": {"close_entered", "close_returned", "unknown"},
        "close_returned": {"close_returned"}, "unknown": {"unknown"}}
    for key, item in before["sdk"].items():
        _require(key in after["sdk"] and after["sdk"][key]["extent"] == item["extent"]
            and after["sdk"][key]["sdk_version"] == item["sdk_version"]
            and after["sdk"][key]["phase"] in sdk_phases[item["phase"]], "sdk_history")
    _require(all(item["phase"] == "not_entered" for key, item in after["sdk"].items()
        if key not in before["sdk"]), "sdk_history")
    if before["prepared_record"] is not None:
        _require(before["prepared_record"] == after["prepared_record"], "record_immutable")


def _append(directory, record, expected):
    _record_shape(record)
    if expected is None:
        _require(not (directory / "head.json").exists(), "cas")
        _initial(record)
        revision, previous = 1, None
    else:
        before, actual = _read_record(directory)
        _require(actual == expected, "cas")
        _transition(before, record)
        revision, previous = expected["revision"] + 1, expected["sha256"]
    event = {"schema_version": 1, "copy_id": record["copy_id"], "revision": revision,
             "previous_sha256": previous, "record": record}
    raw = _json(event) + b"\n"
    identity = _sha(raw)
    _write_new(directory / f"{revision:012d}-{identity}.json", raw)
    head = {"schema_version": 1, "copy_id": record["copy_id"], "revision": revision, "sha256": identity}
    fd, temporary = tempfile.mkstemp(prefix=".head-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json(head) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "head.json")
        _fsync(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return head


def _authority(controller, latest_binding_id=None):
    from src.core.runtime_admission import _stage

    stage = _stage()
    module = sys.modules.get(type(controller).__module__)
    _require(module is not None and type(controller) is getattr(module, "AdmissionController", None)
        and Path(module.__file__).resolve() == Path(stage.__file__).resolve(), "controller")
    method = controller._require_lease
    _require(type(method) is MethodType and method.__func__ is module.AdmissionController._require_lease,
             "controller")
    method()
    root = _directory(Path(controller.root))
    fd = controller._descriptor
    _require(type(fd) is int and fd >= 0, "controller_fd")
    header = stage._read_json(root / ".admission.lock")
    _require(set(header) == {"controller_pid", "lease_fd"}
        and type(header["controller_pid"]) is int and header["controller_pid"] == os.getpid()
        and type(header["lease_fd"]) is int and header["lease_fd"] == fd, "controller_fd")
    info = os.fstat(fd)
    _require(stat.S_ISREG(info.st_mode) and {"device": info.st_dev, "inode": info.st_ino}
             == _identity(root / ".admission.lock"), "controller_fd")
    stage._require_root_lease(root)
    head = stage.read_admission_head(root)
    _require(type(head) is dict and type(head.get("schema_version")) is int and head["schema_version"] == 1
        and head.get("status") == "closed" and head.get("run_id") is None
        and type(head.get("generation")) is int and head["generation"] >= 1, "closed_head")
    if latest_binding_id is not None:
        _require(_hash(latest_binding_id) and head["binding_id"] == latest_binding_id, "authority_changed")
    binding = stage.verify_closed_binding(root, head["binding_id"])
    return stage, root, head, binding, fd, _identity(root / ".admission.lock")


def _snapshot_refs(root, refs, bindings):
    _require(type(refs) is tuple, "snapshot_refs")
    result = []
    for ref in refs:
        _require(type(ref) is dict and set(ref) == {"source_binding_id", "snapshot_root", "root_identity", "manifest_sha256"}
            and ref["source_binding_id"] in bindings and _hash(ref["manifest_sha256"])
            and type(ref["snapshot_root"]) is str and _identity_shape(ref["root_identity"]), "snapshot_refs")
        path = _directory(Path(ref["snapshot_root"]))
        _require(_identity(path) == ref["root_identity"], "snapshot_changed")
        # The canonical manifest digest is provenance only. It does not replace
        # current controller authority or the caller's full snapshot preflight.
        fd = os.open(path / "manifest.json", os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "snapshot_changed")
            raw = stream.read()
            manifest = json.loads(raw)
        _require(type(manifest) is dict and manifest.get("source_binding_id") == ref["source_binding_id"]
            and _sha(_json(manifest)) == ref["manifest_sha256"], "snapshot_changed")
        matches = []
        _require((root / "admission/copies").exists(), "snapshot_not_catalog_prepared")
        catalog = _catalog(root)
        for directory in catalog.iterdir():
            _directory(directory)
            try:
                record, _ = _read_record(directory)
            except (OSError, ValueError, TypeError, KeyError):
                raise CopyCatalogError("copy_catalog_snapshot_not_catalog_prepared") from None
            if record["destination"] != str(path):
                continue
            evidence = record["prepared_record"]
            if (record["kind"] == "snapshot" and record["preparation"] == "prepared" and record["finished"] is True
                    and record["root_identity"] == ref["root_identity"]
                    and ref["source_binding_id"] in record["source_binding_ids"]
                    and type(evidence) is dict and evidence["relative"] == "manifest.json"
                    and evidence["sha256"] == _sha(raw) and evidence["size"] == len(raw)):
                matches.append(record["copy_id"])
        _require(len(matches) == 1, "snapshot_not_catalog_prepared")
        result.append(copy.deepcopy(ref))
    _require(len({_json(ref) for ref in result}) == len(result), "snapshot_refs")
    return result


def _destination(destination, root, source_roots):
    _require(isinstance(destination, Path) and destination.is_absolute()
        and destination.resolve(strict=False) == destination, "destination")
    _directory(destination.parent)
    for source in (root / "admission", *source_roots):
        _require(not destination.is_relative_to(source) and not source.is_relative_to(destination), "destination_overlap")
    if destination.exists():
        _directory(destination)
        _require(all(path.is_dir() and not path.is_symlink()
            and path.relative_to(destination).parts[0] in {"config", "data"}
            for path in destination.rglob("*")), "destination_not_empty")


def _inventory(root):
    _directory(root)
    result = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink() and path.resolve(strict=True) == path, "inventory_redirected")
        before = path.stat()
        if stat.S_ISDIR(before.st_mode):
            continue
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "inventory_file")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            _require((before.st_dev, before.st_ino) == (info.st_dev, info.st_ino)
                and info.st_nlink == 1, "inventory_changed")
            digest = hashlib.sha256()
            while block := stream.read(1024*1024):
                digest.update(block)
        after = path.stat()
        _require(after.st_nlink == 1 and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "inventory_changed")
        result[path.relative_to(root).as_posix()] = {"sha256": digest.hexdigest(), "size": after.st_size,
            "identity": {"device": after.st_dev, "inode": after.st_ino}}
    return result


class Extent:
    """Process-local permission to name one already registered representation."""

    def __init__(self, key, operation, identity):
        _require(key is _KEY and type(operation) is CopyReservation
            and _LIVE.get(operation.copy_id) is operation and operation._active, "local_capability_required")
        self._operation, self.identity = operation, identity

    @property
    def path(self) -> Path:
        return self._operation.destination / self._operation._record["extents"][self.identity]["relative"]

    def __reduce_ex__(self, _protocol):
        raise TypeError("copy_catalog_capability_not_serializable")

    def __deepcopy__(self, _memo):
        raise TypeError("copy_catalog_capability_not_serializable")


class CopyReservation:
    """Retain an actual controller lifetime, record CAS and any SDK handles."""

    def __init__(self, key, controller, stage, directory, record, head, descriptor):
        _require(key is _KEY, "local_capability_required")
        self._controller, self._stage, self._directory = controller, stage, directory
        self._record, self._head = record, head
        self._root = Path(controller.root)
        self._root_identity, self._directory_identity = _identity(self._root), _identity(directory)
        self._descriptor, self._original_fd = descriptor, controller._descriptor
        self._active, self._tainted = True, False
        self._sdk = {}
        self.destination = Path(record["destination"])
        self.copy_id = record["copy_id"]

    def __reduce_ex__(self, _protocol):
        raise TypeError("copy_catalog_capability_not_serializable")

    def __deepcopy__(self, _memo):
        raise TypeError("copy_catalog_capability_not_serializable")

    def _check(self):
        _require(self._active and _LIVE.get(self.copy_id) is self, "local_capability_required")
        _, root, head, _, fd, lease_identity = _authority(self._controller, self._record["authority"]["binding_id"])
        _directory(self._directory.parent)
        _directory(self._directory)
        _directory(self.destination.parent)
        _require(root == self._root and _identity(root) == self._root_identity
            and _identity(self._directory) == self._directory_identity and head == self._record["authority"]
            and fd == self._original_fd
            and lease_identity == self._record["preparation_owner"]["lease_identity"]
            and _identity(self.destination.parent) == self._record["parent_identity"], "authority_changed")
        info = os.fstat(self._descriptor)
        _require({"device": info.st_dev, "inode": info.st_ino} == lease_identity, "controller_fd")
        if self._record["root_identity"] is not None:
            _directory(self.destination)
            _require(_identity(self.destination) == self._record["root_identity"], "destination_changed")
        else:
            _require(self.destination.resolve(strict=False) == self.destination, "destination_changed")

    @contextmanager
    def _locked(self):
        with _LOCK, self._stage._locked_root(self._root):
            self._check()
            current, head = _read_record(self._directory)
            _require(head == self._head and current == self._record, "cas")
            yield

    def _change(self, change):
        try:
            with self._locked():
                record = copy.deepcopy(self._record)
                change(record)
                head = _append(self._directory, record, self._head)
                self._record, self._head = record, head
                return copy.deepcopy(record)
        except BaseException:
            self._tainted = True
            raise

    def create_root(self) -> Path:
        _require(not self._tainted and self._record["preparation"] == "reserved", "preparation_state")
        with self._locked():
            _destination(self.destination, self._root, [])
            if not self.destination.exists():
                self.destination.mkdir(mode=0o700)
                _fsync(self.destination.parent)
        physical = _identity(self.destination)

        def changed(record):
            record["root_identity"] = physical
            record["preparation"] = "writing"
        self._change(changed)
        return self.destination

    def extent(self, relative: str, *, family: str, representation: str) -> Extent:
        _relative(relative)
        _require(not self._tainted and self._record["preparation"] == "writing"
            and family in FAMILIES and representation in REPRESENTATIONS, "extent")
        value = {"relative": relative, "family": family, "representation": representation}
        identity = _sha(_json(value))
        _require(all(item["relative"] != relative or item == value
            for item in self._record["extents"].values()), "extent_ambiguous")
        self._change(lambda record: record["extents"].update({identity: value}))
        return Extent(_KEY, self, identity)

    def sdk_enter(self, extent: Extent, *, sdk_version: str) -> SDKEntry:
        _require(type(extent) is Extent and extent._operation is self and not self._tainted
            and self._record["preparation"] == "writing"
            and self._record["extents"][extent.identity]["representation"] == "sdk_work"
            and _version(sdk_version), "sdk_extent")
        _require(not any(value["extent"] == extent.identity for value in self._record["sdk"].values()), "sdk_duplicate")
        identity = uuid4().hex
        self._change(lambda record: record["sdk"].update({identity: {"extent": extent.identity,
            "sdk_version": sdk_version, "phase": "not_entered"}}))
        entry = SDKEntry(_KEY, self, extent, identity, sdk_version)
        self._sdk[identity] = entry
        return entry

    def prepared(self, record_path: Path) -> dict:
        _require(not self._tainted and self._record["preparation"] == "writing"
            and self._record["extents"] and set(self._sdk) == set(self._record["sdk"])
            and all(entry._close_returned and entry._handle is None and not entry._uncertain for entry in self._sdk.values())
            and all(item["phase"] == "close_returned" for item in self._record["sdk"].values()),
            "preparation_unconfirmed")
        _require(isinstance(record_path, Path) and record_path.is_relative_to(self.destination), "prepared_record")
        relative = record_path.relative_to(self.destination).as_posix()
        _relative(relative)
        with self._locked():
            inventory = _inventory(self.destination)
            _require(relative in inventory, "prepared_record")
            _require(all(any(extent["relative"] == "." or Path(name).is_relative_to(extent["relative"])
                for extent in self._record["extents"].values()) for name in inventory), "unregistered_extent")
            observations = {}
            for key, extent in self._record["extents"].items():
                path = self.destination / extent["relative"]
                _require(path.resolve(strict=False) == path and not path.is_symlink(), "extent_redirected")
                observations[key] = {"status": "present" if path.exists() else "absent_observed",
                                     "identity": _identity(path) if path.exists() else None}
        evidence = {"relative": relative, "sha256": inventory[relative]["sha256"],
            "size": inventory[relative]["size"], "inventory": inventory, "extent_observations": observations}

        def changed(record):
            _require(_inventory(self.destination) == inventory, "inventory_changed")
            record["prepared_record"], record["preparation"] = evidence, "prepared"
        result = self._change(changed)
        return result

    def finish(self) -> dict:
        _require(not self._tainted and self._record["preparation"] == "prepared"
            and all(entry._close_returned and not entry._uncertain and entry._handle is None
                    for entry in self._sdk.values()), "preparation_unconfirmed")
        result = self._change(lambda record: record.update(finished=True))
        self._release()
        return result

    def failed(self, code: str) -> dict:
        _require(type(code) is str and re.fullmatch("[A-Za-z_][A-Za-z0-9_]{0,79}", code), "failure_code")
        uncertain = self._tainted or any(entry._uncertain or entry._entered and not entry._close_returned
            for entry in self._sdk.values())

        def changed(record):
            record["preparation"] = "unknown" if uncertain else "failed"
            record["error_code"] = code
            for item in record["sdk"].values():
                if item["phase"] not in {"not_entered", "close_returned"}:
                    item["phase"] = "unknown"
        result = self._change(changed)
        if not uncertain:
            self._release()
        return result

    def _release(self):
        self._active = False
        _LIVE.pop(self.copy_id, None)
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None


class SDKEntry:
    """Retain one nominal Chroma acquisition and actual public close return."""

    def __init__(self, key, operation, extent, identity, version):
        _require(key is _KEY and type(operation) is CopyReservation
            and _LIVE.get(operation.copy_id) is operation and operation._active, "local_capability_required")
        self._operation, self._extent, self.identity, self._version = operation, extent, identity, version
        self._handle = self._close_method = None
        self._trusted_close = self._trusted_code = None
        self._entered = self._attempted_close = self._close_returned = False
        self._uncertain = False

    def __reduce_ex__(self, _protocol):
        raise TypeError("copy_catalog_capability_not_serializable")

    def __deepcopy__(self, _memo):
        raise TypeError("copy_catalog_capability_not_serializable")

    def _phase(self, phase):
        return self._operation._change(lambda record: record["sdk"][self.identity].update(phase=phase))

    def construct(self, factory: Callable[..., object], *args: object, **kwargs: object) -> object:
        _require(not self._entered and not self._operation._tainted, "sdk_constructor_state")
        self._operation._check()
        _require(importlib.metadata.version("chromadb") == self._version, "sdk_version")
        Client = _sdk_factory(factory)
        _require(not args and set(kwargs) <= {"path", "settings", "tenant", "database"}
            and "path" in kwargs and Path(kwargs["path"]) == self._extent.path
            and self._extent.path.resolve(strict=False) == self._extent.path, "sdk_path")
        self._phase("constructor_entered")
        self._entered = True
        try:
            self._handle = factory(*args, **kwargs)
            _require(type(self._handle) is Client, "sdk_handle_unsupported")
            nominal, method = Client.close, self._handle.close
            _require(type(nominal) is FunctionType and nominal.__module__ == "chromadb.api.client"
                and nominal.__qualname__ == "Client.close"
                and Path(nominal.__code__.co_filename).resolve() == Path(sys.modules[Client.__module__].__file__).resolve()
                and type(method) is MethodType and method.__self__ is self._handle
                and method.__func__ is nominal, "sdk_close_unsupported")
            self._trusted_close, self._trusted_code = nominal, nominal.__code__
            self._close_method = method
            settings = self._handle.get_settings()
            _require(settings.is_persistent is True
                and Path(settings.persist_directory).resolve() == self._extent.path, "sdk_path")
            self._phase("handle_returned")
            return self._handle
        except BaseException:
            self._uncertain = True
            raise

    def close(self) -> None:
        if self._close_returned:
            self._operation._check()
            return
        _require(self._entered and self._handle is not None and self._close_method is not None
            and not self._attempted_close, "sdk_retirement_unknown")
        errors = []
        marker = False
        try:
            self._phase("close_entered")
            marker = True
        except BaseException as error:
            errors.append(error)
        self._attempted_close = True
        try:
            from chromadb.api.client import Client
            current = self._handle.close
            _require(type(current) is MethodType and current.__self__ is self._handle
                and current.__func__ is self._trusted_close and Client.close is self._trusted_close
                and self._trusted_close.__code__ is self._trusted_code, "sdk_close_changed")
            result = self._close_method()
            _require(result is None, "sdk_retirement_unknown")
        except BaseException as error:
            self._uncertain = True
            errors.append(error)
        if not errors and marker:
            try:
                self._phase("close_returned")
            except BaseException as error:
                self._uncertain = True
                errors.append(error)
            else:
                self._close_returned = True
                self._handle = self._close_method = None
        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup("copy catalog SDK close unconfirmed", errors)


def _sdk_factory(factory):
    """Recognize construction code, never an arbitrary borrowed-handle lambda."""
    import chromadb
    from chromadb.api.client import Client

    _require(factory is chromadb.PersistentClient and type(factory) is FunctionType
        and factory.__module__ == "chromadb" and factory.__name__ == "PersistentClient"
        and Path(factory.__code__.co_filename).resolve() == Path(chromadb.__file__).resolve(),
        "sdk_factory_unsupported")
    return Client


def reserve_copy(controller: AdmissionController, kind: str, destination: Path,
                 latest_binding_id: str, source_binding_ids: tuple[str, ...],
                 source_snapshot_refs: tuple[dict, ...] = ()) -> CopyReservation:
    """Reserve a non-serving copy under actual current closed authority.

    Commit and fsync the external record before the caller can create payload.
    The returned capability is process-local. Snapshot references establish
    provenance only; neither those DTOs nor this reservation grant cleanup.
    """
    _require(kind in KINDS and type(source_binding_ids) is tuple and bool(source_binding_ids)
        and all(_hash(value) for value in source_binding_ids)
        and len(set(source_binding_ids)) == len(source_binding_ids), "sources")
    stage, root, _, _, _, _ = _authority(controller, latest_binding_id)
    with _LOCK, stage._locked_root(root):
        _, _, head, latest, fd, lease_identity = _authority(controller, latest_binding_id)
        sources = {}
        for identity in source_binding_ids:
            sources[identity] = stage.read_binding(root, identity)
            stage._closure_proof(root, identity)
        refs = _snapshot_refs(root, source_snapshot_refs, sources)
        source_roots = [Path(item["state_root"]) for item in sources.values()]
        source_roots.extend(Path(ref["snapshot_root"]) for ref in refs)
        source_roots.append(Path(latest["state_root"]))
        _destination(destination, root, source_roots)
        catalog = _catalog(root, create=True)
        for directory in catalog.iterdir():
            _directory(directory)
            record, _ = _read_record(directory)
            previous = Path(record["destination"])
            _require(not destination.is_relative_to(previous) and not previous.is_relative_to(destination),
                     "destination_already_cataloged")
        identity = uuid4().hex
        record = {"schema_version": 1, "copy_id": identity, "kind": kind, "authority": copy.deepcopy(head),
            "source_binding_ids": list(source_binding_ids), "source_snapshot_refs": refs,
            "destination": str(destination), "parent_identity": _identity(destination.parent), "root_identity": None,
            "preparation_owner": {"epoch": uuid4().hex, "process_id": os.getpid(), "lease_fd": fd,
                                  "lease_identity": lease_identity},
            "preparation": "reserved", "cleanup": {"phase": "blocked", "owner": None, "receipt": None},
            "extents": {}, "sdk": {}, "prepared_record": None, "error_code": None, "finished": False}
        directory = catalog / identity
        directory.mkdir(mode=0o700)
        _fsync(catalog)
        retained = os.dup(fd)
        try:
            head_record = _append(directory, record, None)
            token = CopyReservation(_KEY, controller, stage, directory, record, head_record, retained)
        except BaseException:
            os.close(retained)
            raise
        _LIVE[identity] = token
        return token


def recover_copies(controller: AdmissionController) -> tuple[dict, ...]:
    """Read durable obligations under current authority without issuing grants.

    Preserve validated metadata even around an interrupted HEAD publication.
    Never repair records, infer native return from process liveness or mark
    cleanup complete; unknown obligations remain blocked without a capability.
    """
    stage, root, _, _, _, _ = _authority(controller)
    with _LOCK, stage._locked_root(root):
        _authority(controller)
        if not (root / "admission/copies").exists():
            return ()
        results = []
        for directory in sorted(_catalog(root).iterdir()):
            _directory(directory)
            _require(re.fullmatch("[0-9a-f]{32}", directory.name), "catalog_entry")
            results.append(_recover_record(directory))
        return tuple(results)


def _recover_record(directory):
    """Preserve verified committed or initial-uncommitted ownership metadata."""
    error = None
    try:
        record, head = _read_record(directory)
    except (OSError, ValueError, TypeError, KeyError):
        error = "record_unconfirmed"
        try:
            record, head = _read_record(directory, allow_orphans=True)
            error = "uncommitted_events"
        except (OSError, ValueError, TypeError, KeyError):
            candidates = []
            for path in sorted(directory.iterdir()):
                match = re.fullmatch(r"000000000001-([0-9a-f]{64})\.json", path.name)
                if match is None:
                    continue
                try:
                    event, raw = _read(path)
                    _require(_sha(raw) == match[1] and set(event) == {"schema_version", "copy_id", "revision", "previous_sha256", "record"}
                        and type(event["schema_version"]) is int and event["schema_version"] == 1
                        and type(event["revision"]) is int and event["revision"] == 1
                        and event["copy_id"] == directory.name and event["previous_sha256"] is None, "chain")
                    _record_shape(event["record"])
                    _initial(event["record"])
                    candidates.append({"record": event["record"], "sha256": match[1]})
                except (OSError, ValueError, TypeError, KeyError):
                    continue
            if len(candidates) != 1:
                return {"copy_id": directory.name, "preparation": "unknown",
                    "cleanup": {"phase": "blocked", "owner": None, "receipt": None},
                    "catalog_error": "record_unconfirmed", "uncommitted_metadata": candidates,
                    "capability": False}
            record, head = candidates[0]["record"], {"revision": 0, "sha256": candidates[0]["sha256"]}
            error = "initial_uncommitted_metadata"
    observed = copy.deepcopy(record)
    if error is not None or observed["preparation"] in {"reserved", "writing"} or (
            not observed["finished"] and observed["preparation"] == "prepared"):
        observed["preparation"] = "unknown"
    observed.update(revision=head["revision"], record_sha256=head["sha256"], capability=False)
    if error is not None:
        observed["catalog_error"] = error
    return observed
