"""Current-authority collector-copy retirement, separate from V1 history."""
from __future__ import annotations

import asyncio
from contextvars import Context
from dataclasses import asdict, dataclass
import hashlib
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import stat
from uuid import UUID, uuid4

from src.core.owned_work import OwnedBlockingWork
from src.memory import copy_catalog as catalog
from src.memory.manual_components import TABLES, protected_collector_records
from src.wakeword.clip_store import clip_sources

SCOPE = "collector_rows_files_v1"
_KINDS = {"wake_clips": (".wav", None), "stt_corrections": (".wav", "stt_correction"),
          "conversations": (".json", "conversation_turn"), "conversation_exports": (".jsonl", "conversation_export")}
_FENCE_KEYS = {"schema_version", "scope", "copy_id", "catalog_revision", "catalog_record_sha256",
    "admission_root", "admission_identity", "destination", "root_identity", "authority_head",
    "operation_id", "conversation_id", "layout_sha256", "claim_id", "status"}


class CopyCleanupError(ValueError):
    """Content-free unsupported scope or unconfirmed retirement."""


def _require(value, code):
    if not value:
        raise CopyCleanupError("copy_cleanup_" + code)


def _json(value):
    return catalog._json(value)


def _sha(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _uuid(value):
    _require(type(value) is str and str(UUID(value)) == value, "operation_identity")
    return value


@dataclass(frozen=True)
class CollectorRoot:
    """An existing local collector target, never a relocation grant."""
    target_id: str
    kind: str
    relative_root: str


@dataclass(frozen=True)
class CollectorCopyLayout:
    """Exact supported database and complete local collector roots."""
    database_relative: str
    roots: tuple[CollectorRoot, ...]


@dataclass(frozen=True)
class CopyCleanupClaim:
    """A local-owner DTO whose serialized identity grants no work."""
    copy_id: str
    claim_id: str
    operation_id: str
    conversation_id: str
    catalog_revision: int
    catalog_record_sha256: str
    cleanup_revision: int
    cleanup_record_sha256: str
    authority_head: dict
    layout_sha256: str
    fence_sha256: str

    def to_dict(self) -> dict:
        """Return metadata only."""
        return asdict(self)

    def __reduce_ex__(self, _protocol):
        raise TypeError("copy_cleanup_claim_not_serializable")


@dataclass(frozen=True)
class CopyCleanupReceipt:
    """An actual local return; other copies and representations remain blocked."""
    copy_id: str
    claim_id: str
    operation_id: str
    conversation_id: str
    cleanup_revision: int
    record_sha256: str
    fence_sha256: str
    removed_records: int
    removed_exports: int
    removed_files: int
    preserved_rows_sha256: str
    preserved_files_sha256: str
    scope: str = SCOPE
    representation_complete: bool = True
    aggregate_copy_complete: bool = False
    other_representations: str = "blocked"
    native_returned: bool = True

    def to_dict(self) -> dict:
        """Return bounded metadata, never aggregate cleanup permission."""
        return asdict(self)


def _layout(value):
    _require(type(value) is CollectorCopyLayout and type(value.roots) is tuple and value.roots, "layout")
    catalog._relative(value.database_relative)
    _require(value.database_relative != ".", "layout")
    ids, paths = set(), set()
    for root in value.roots:
        _require(type(root) is CollectorRoot and catalog._hash(root.target_id)
            and root.kind in _KINDS and root.target_id not in ids, "layout")
        catalog._relative(root.relative_root)
        _require(root.relative_root != "." and root.relative_root not in paths, "layout")
        ids.add(root.target_id)
        paths.add(root.relative_root)
    _require(not any(a != b and Path(a).is_relative_to(b) for a in paths for b in paths), "overlapping_roots")
    return _sha(asdict(value))


def _base(root, copy_id):
    _require(type(copy_id) is str and len(copy_id) == 32 and all(c in "0123456789abcdef" for c in copy_id), "copy_id")
    return root / "admission/copy-cleanup/v1/claims" / copy_id


def _mkdir(path):
    if not path.exists():
        _mkdir(path.parent)
        path.mkdir(mode=0o700)
        catalog._fsync(path.parent)
    catalog._directory(path)


def _state(directory):
    catalog._directory(directory)
    head, _ = catalog._read(directory / "head.json")
    _require(set(head) == {"schema_version", "revision", "sha256"}
        and type(head["schema_version"]) is int and head["schema_version"] == 1
        and type(head["revision"]) is int and head["revision"] > 0 and catalog._hash(head["sha256"]), "journal_head")
    revision, digest, latest, names, newer = head["revision"], head["sha256"], None, {"head.json"}, None
    while revision:
        name = f"{revision:012d}-{digest}.json"
        value, raw = catalog._read(directory / name)
        names.add(name)
        _require(hashlib.sha256(raw).hexdigest() == digest
            and set(value) == {"schema_version", "revision", "previous", "state"}
            and type(value["schema_version"]) is int and value["schema_version"] == 1
            and type(value["revision"]) is int and value["revision"] == revision
            and (value["previous"] is None if revision == 1 else catalog._hash(value["previous"])), "journal_chain")
        _state_shape(value["state"])
        if newer is not None:
            _state_transition(value["state"], newer)
        newer = value["state"]
        if latest is None:
            latest = value["state"]
        digest, revision = value["previous"], revision - 1
    _require({p.name for p in directory.iterdir()} == names, "journal_orphan")
    return latest, head


def _state_shape(value):
    _require(type(value) is dict and set(value) == {"schema_version", "scope", "copy_id", "claim_id",
        "operation_id", "conversation_id", "authority_head", "catalog_revision", "catalog_record_sha256",
        "layout_sha256", "fence_sha256", "phase", "native_started", "native_returned", "receipt", "error_code"}, "journal_schema")
    _require(type(value["schema_version"]) is int and value["schema_version"] == 1 and value["scope"] == SCOPE
        and value["phase"] in {"claimed", "running", "complete", "failed", "unknown"}
        and type(value["native_started"]) is bool and type(value["native_returned"]) is bool
        and type(value["catalog_revision"]) is int and value["catalog_revision"] > 0
        and all(catalog._hash(value[key]) for key in ("catalog_record_sha256", "layout_sha256", "fence_sha256")), "journal_schema")
    _uuid(value["operation_id"])
    _uuid(value["conversation_id"])
    for key in ("copy_id", "claim_id"):
        _require(type(value[key]) is str and len(value[key]) == 32
            and all(char in "0123456789abcdef" for char in value[key]), "journal_identity")
    head = value["authority_head"]
    _require(type(head) is dict and set(head) == {"schema_version", "generation", "binding_id", "status", "run_id"}
        and type(head["schema_version"]) is int and head["schema_version"] == 1
        and type(head["generation"]) is int and head["generation"] > 0
        and catalog._hash(head["binding_id"]) and head["status"] == "closed" and head["run_id"] is None,
        "journal_authority")
    _require(value["error_code"] is None or type(value["error_code"]) is str
        and value["error_code"].isascii() and value["error_code"].isidentifier(), "journal_error")
    _require(value["phase"] != "complete" or value["native_returned"] is True and type(value["receipt"]) is dict,
             "journal_receipt")
    if value["phase"] == "complete":
        receipt = value["receipt"]
        _require(set(receipt) == set(CopyCleanupReceipt.__dataclass_fields__)
            and all(receipt[key] == value[key] for key in ("copy_id", "claim_id", "operation_id", "conversation_id", "scope"))
            and receipt["representation_complete"] is True and receipt["aggregate_copy_complete"] is False
            and receipt["other_representations"] == "blocked" and receipt["native_returned"] is True
            and all(type(receipt[key]) is int and receipt[key] >= 0
                    for key in ("removed_records", "removed_exports", "removed_files"))
            and type(receipt["cleanup_revision"]) is int and receipt["cleanup_revision"] > 0
            and all(catalog._hash(receipt[key]) for key in ("record_sha256", "fence_sha256",
                  "preserved_rows_sha256", "preserved_files_sha256")), "journal_receipt")
    else:
        _require(value["receipt"] is None, "journal_receipt")


def _state_transition(before, after):
    _require(all(before[key] == after[key] for key in ("schema_version", "scope", "copy_id", "layout_sha256",
                                                      "fence_sha256")), "journal_immutable")
    if before["claim_id"] != after["claim_id"]:
        _require(before["phase"] in {"complete", "failed"} and before["native_returned"] is True
            and after["phase"] == "claimed" and not after["native_started"] and not after["native_returned"]
            and after["receipt"] is None, "journal_retry")
        return
    _require(all(before[key] == after[key] for key in ("operation_id", "conversation_id", "authority_head",
        "catalog_revision", "catalog_record_sha256")), "journal_immutable")
    phases = {"claimed": {"running"}, "running": {"complete", "failed", "unknown"},
              "complete": set(), "failed": set(), "unknown": set()}
    _require(after["phase"] in phases[before["phase"]] and (not before["native_started"] or after["native_started"])
        and (not before["native_returned"] or after["native_returned"]), "journal_transition")


def _append(directory, value, expected):
    _state_shape(value)
    if expected is None:
        _require(not (directory / "head.json").exists(), "claim_cas")
        _require(value["phase"] == "claimed" and value["native_started"] is False
            and value["native_returned"] is False and value["error_code"] is None, "initial_claim")
        revision, previous = 1, None
    else:
        before, current = _state(directory)
        _require(current == expected, "claim_cas")
        _state_transition(before, value)
        revision, previous = current["revision"]+1, current["sha256"]
    event = {"schema_version": 1, "revision": revision, "previous": previous, "state": value}
    raw = _json(event)+b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    catalog._write_new(directory / f"{revision:012d}-{digest}.json", raw)
    head = {"schema_version": 1, "revision": revision, "sha256": digest}
    from src.core.runtime_admission import _stage
    _stage()._atomic_record(directory / "head.json", head)
    catalog._fsync(directory)
    return head


def _table_state(db):
    _require(not db.execute("SELECT 1 FROM sqlite_master WHERE type IN ('view','trigger') LIMIT 1").fetchone(), "schema")
    tables = {}
    for name, fields in TABLES.items():
        _require({row[1] for row in db.execute('PRAGMA table_info("'+name+'")')} == fields, "schema")
        tables[name] = [dict(row) for row in db.execute('SELECT * FROM "'+name+'"')]
    return tables


def _other_rows(db):
    values = {}
    names = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for name in sorted(set(names)-set(TABLES)):
        if name.startswith("sqlite_"):
            continue
        quoted = '"'+name.replace('"','""')+'"'
        rows = [dict(row) for row in db.execute("SELECT * FROM "+quoted)]
        values[name] = sorted(_sha({key: ({"blob_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
            if type(value) is bytes else value) for key, value in row.items()}) for row in rows)
    return _sha(values)


def _row_targets(db, destination, layout):
    targets = {}
    for row in db.execute("SELECT * FROM erasure_targets WHERE participant='collectors'"):
        descriptor = json.loads(row["descriptor"])
        if descriptor.get("kind") not in _KINDS:
            continue
        _require(row["version"] == 1 and set(descriptor) == {"kind", "root"}, "target")
        root = Path(descriptor["root"])
        _require(root.is_absolute() and root.resolve(strict=False) == root
            and root.is_relative_to(destination) and root != destination, "foreign_target")
        encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = hashlib.sha256(json.dumps(["collectors", 1, encoded]).encode()).hexdigest()
        _require(row["id"] == expected, "target")
        targets[row["id"]] = CollectorRoot(row["id"], descriptor["kind"], root.relative_to(destination).as_posix())
    _require(targets == {root.target_id: root for root in layout.roots}, "layout_target_coverage")
    return targets


def _files(destination, tables, roots):
    clips = {row["id"]: row for row in tables["wake_clips_v1"]}
    _require(len(clips) == len(tables["wake_clips_v1"]), "duplicate_record")
    found = {}
    for target in roots.values():
        root = destination / target.relative_root
        _require(root.resolve(strict=False) == root and not root.is_symlink(), "collector_root")
        if not root.exists():
            continue
        catalog._directory(root)
        _require(all(path.name == "policy_v1" and path.is_dir() and not path.is_symlink()
            for path in root.iterdir()), "unreviewed_collector_file")
        policy = root / "policy_v1"
        if not policy.exists():
            continue
        catalog._directory(policy)
        for path in policy.iterdir():
            suffix = next((ext for ext in (".part", _KINDS[target.kind][0]) if path.name.endswith(ext)), None)
            _require(suffix is not None, "unreviewed_collector_file")
            identifier = path.name[:-len(suffix)]
            _uuid(identifier)
            _require(identifier in clips and clips[identifier]["target_id"] == target.target_id,
                     "unattributed_collector_file")
            info = path.lstat()
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and path.resolve() == path, "collector_file")
            raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            after = path.stat()
            _require((info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns) ==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns), "collector_file_changed")
            found[path.relative_to(destination).as_posix()] = {"id": identifier, "target_id": target.target_id,
                "suffix": suffix, "sha256": raw_hash, "size": info.st_size,
                "identity": {"device": info.st_dev, "inode": info.st_ino}}
    return found


def _affected(tables, conversation):
    clips = tables["wake_clips_v1"]
    ids = {row["id"] for row in clips if any(source.conversation_id == conversation
        for source in clip_sources(row["sources_json"]))}
    parents = tables["conversation_export_parents_v1"]
    while True:
        expanded = ids | {row["export_id"] for row in parents if row["parent_id"] in ids}
        if expanded == ids:
            break
        ids = expanded
    # A protected component is an audit/preservation boundary, not permission
    # to delete co-parents. Shared U-only manual parents remain byte-exact.
    protected = protected_collector_records(tables)
    return ids, protected-ids


def _preserved(tables, ids):
    result = {}
    for name, rows in tables.items():
        field = "export_id" if name == "conversation_export_parents_v1" else "id"
        result[name] = sorted((_sha(row) for row in rows if row[field] not in ids))
    return _sha(result)


def _capture(stage, destination, layout, conversation):
    database = destination / layout.database_relative
    _require(database.resolve(strict=True) == database and database.is_file(), "database")
    with stage._database_image(database) as db:
        tables = _table_state(db)
        roots = _row_targets(db, destination, layout)
        other = _other_rows(db)
    files = _files(destination, tables, roots)
    ids, manual = _affected(tables, conversation)
    _require(all(row["target_id"] in roots for row in tables["wake_clips_v1"]), "target_coverage")
    for row in tables["wake_clips_v1"]:
        target = roots[row["target_id"]]
        _require(_KINDS[target.kind][1] is None or row["bucket"] == _KINDS[target.kind][1], "bucket")
        # Absence is preserved too: legitimate retention can remove a payload
        # while leaving an expired catalog row. It grants no new read access.
    return {"tables": tables, "roots": roots, "files": files, "ids": ids, "manual": manual,
        "other": other, "preserved_rows": _preserved(tables,ids),
        "preserved_files": _sha({key:item for key,item in files.items() if item["id"] not in ids})}


def _unlink_file(root: Path, identifier: str, suffix: str) -> bool:
    """Unlink one admitted collector file and wait for directory durability."""
    _uuid(identifier)
    _require(suffix in {".part", ".wav", ".json", ".jsonl"}, "file_suffix")
    catalog._directory(root)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    policy_fd = None
    try:
        try:
            policy_fd = os.open("policy_v1", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        except FileNotFoundError:
            return False
        name = identifier + suffix
        try:
            info = os.stat(name, dir_fd=policy_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "file_identity")
        os.unlink(name, dir_fd=policy_fd)
        os.fsync(policy_fd)
        return True
    finally:
        if policy_fd is not None:
            os.close(policy_fd)
        os.close(root_fd)


def _tombstone(stage, binding, layout, operation_id):
    current = Path(binding["state_root"]) / layout.database_relative
    _require(layout.database_relative in binding["schemas"], "current_database")
    with stage._database_image(current) as db:
        row = db.execute("SELECT o.id,o.conversation_id,o.requested_at,o.requested_by,t.deleted_at,t.deleted_by "
            "FROM conversation_erasures o JOIN conversation_tombstones t "
            "ON t.conversation_id=o.conversation_id WHERE o.id=?", (operation_id,)).fetchone()
        _require(row is not None, "current_tombstone")
        value = dict(row)
    _uuid(value["conversation_id"])
    return value


def _perform(owner, plan):
    """Run one selective transaction; the worker return is the drain witness."""
    owner._current()
    fresh = _capture(owner._stage, owner.destination, owner.layout, owner._claim.conversation_id)
    _require(fresh == plan, "beforeimage_changed")
    database = owner.destination / owner.layout.database_relative
    db = sqlite3.connect(database)
    owner._database = db
    db.row_factory = sqlite3.Row
    failure = None
    try:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA secure_delete=ON")
        _require(db.execute("PRAGMA secure_delete").fetchone()[0] == 1, "secure_delete_unconfirmed")
        db.execute("BEGIN IMMEDIATE")
        _require(_table_state(db) == plan["tables"] and _other_rows(db) == plan["other"], "beforeimage_changed")
        clips = {row["id"]: row for row in plan["tables"]["wake_clips_v1"]}
        removed_files = 0
        for identifier in sorted(plan["ids"]):
            owner._current()
            row = clips[identifier]
            target = plan["roots"][row["target_id"]]
            root = owner.destination / target.relative_root
            if root.exists():
                for suffix in (".part", _KINDS[target.kind][0]):
                    removed_files += bool(_unlink_file(root, identifier, suffix))
        # All exports depending on S have been included before deleting parents.
        for identifier in sorted(plan["ids"]):
            db.execute("DELETE FROM wake_clips_v1 WHERE id=?", (identifier,))
        after = _table_state(db)
        _require(_preserved(after, set()) == plan["preserved_rows"] and _other_rows(db) == plan["other"],
                 "preserved_rows_changed")
        db.commit()
        _checkpoint(db)
    except BaseException as error:
        failure = error
        try:
            db.rollback()
        except BaseException as rollback_error:
            failure = BaseExceptionGroup("copy_cleanup_native_failures", [failure, rollback_error])
    # Do not drop a connection whose actual close has not returned, or discard
    # an earlier control/failure if close itself also fails.
    try:
        db.close()
        owner._database = None
    except BaseException as close_error:
        if failure is not None:
            raise BaseExceptionGroup("copy_cleanup_native_failures", [failure, close_error]) from None
        raise
    if failure is not None:
        raise failure
    _check_sidecars(database)
    owner._current()
    after = _capture(owner._stage, owner.destination, owner.layout, owner._claim.conversation_id)
    _require(not after["ids"] and after["preserved_rows"] == plan["preserved_rows"]
        and after["preserved_files"] == plan["preserved_files"] and after["other"] == plan["other"],
        "preservation_failed")
    return {"removed_records": len(plan["ids"]), "removed_exports": sum(
        row["id"] in plan["ids"] and row["bucket"] == "conversation_export" for row in clips.values()),
        "removed_files": removed_files, "preserved_rows_sha256": plan["preserved_rows"],
        "preserved_files_sha256": plan["preserved_files"]}


def _check_sidecars(database):
    for suffix in ("-wal", "-journal"):
        path = Path(str(database) + suffix)
        if path.exists() or path.is_symlink():
            info = path.lstat()
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size == 0,
                     "sidecar_retention_unknown")


def _checkpoint(db):
    checkpoint = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    _require(len(checkpoint) == 1 and checkpoint[0][0] == 0, "checkpoint_unconfirmed")


_CLEANUPS: set = set()


class CollectorCopyCleanup:
    """Own selective collector retirement under the actual current controller.

    The permanent fence invalidates this representation. A successful local
    receipt neither cleans other families nor restores admission to the copy.
    """

    def __init__(self, controller, copy_id: str, *, layout: CollectorCopyLayout) -> None:
        self.controller, self.copy_id, self.layout = controller, copy_id, layout
        self._layout_sha = _layout(layout)
        stage, root, head, binding, fd, identity = catalog._authority(controller)
        self._stage, self.root, self._authority_head = stage, root, head
        self._binding, self._fd, self._fd_identity = binding, fd, identity
        self._directory = _base(root, copy_id)
        self._copy, self._copy_head = catalog._read_record(catalog._catalog(root) / copy_id)
        _require(self._copy["kind"] == "derived_state" and self._copy["preparation"] == "prepared"
            and self._copy["finished"] is True, "unsupported_representation")
        self.destination = catalog._directory(Path(self._copy["destination"]))
        current_root = Path(binding["state_root"])
        _require(catalog._identity(self.destination) == self._copy["root_identity"]
            and not self.destination.is_relative_to(current_root)
            and not current_root.is_relative_to(self.destination), "inactive_copy_required")
        self._stage._assert_writers_closed(self.destination)
        self._claim = self._claim_sha = self._plan = self._head = self._record = None
        self._task = self._database = self._closing = None
        self._closed = False
        self._work = OwnedBlockingWork(capacity=1, close_timeout_s=1, isolate_context=True)
        self._lease_fd = os.dup(fd)
        self._lock_fd = None
        try:
            locks = root / "admission/copy-cleanup/v1/locks"
            _mkdir(locks)
            self._lock_fd = os.open(locks / (copy_id + ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            lock_info = os.fstat(self._lock_fd)
            _require(stat.S_ISREG(lock_info.st_mode) and lock_info.st_nlink == 1, "claim_lock")
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            if self._lock_fd is not None:
                os.close(self._lock_fd)
            os.close(self._lease_fd)
            self._lease_fd = self._lock_fd = None
            raise
        _CLEANUPS.add(self)

    def _current(self):
        _require(not self._closed and self._lease_fd is not None, "owner_closed")
        stage, root, head, binding, fd, identity = catalog._authority(self.controller)
        _require(stage is self._stage and root == self.root and head == self._authority_head
            and binding == self._binding and fd == self._fd and identity == self._fd_identity,
            "authority_changed")
        info = os.fstat(self._lease_fd)
        _require({"device": info.st_dev, "inode": info.st_ino} == identity, "lease_changed")
        record, copy_head = catalog._read_record(catalog._catalog(root) / self.copy_id)
        _require(record == self._copy and copy_head == self._copy_head
            and catalog._identity(self.destination) == record["root_identity"], "catalog_changed")
        self._stage._assert_writers_closed(self.destination)
        if self._claim is not None:
            value = _tombstone(stage, binding, self.layout, self._claim.operation_id)
            _require(value["conversation_id"] == self._claim.conversation_id
                and _sha(value) == self._tombstone_sha, "tombstone_changed")

    def claim(self, operation_id: str, *, expected_catalog_revision: int,
              expected_record_sha256: str, expected_cleanup_revision: int = 0) -> CopyCleanupClaim:
        """Persist the current tombstone claim and invalidation before dispatch."""
        self._current()
        _uuid(operation_id)
        _require(self._claim is None and type(expected_catalog_revision) is int
            and expected_catalog_revision == self._copy_head["revision"]
            and catalog._hash(expected_record_sha256) and expected_record_sha256 == self._copy_head["sha256"]
            and type(expected_cleanup_revision) is int and expected_cleanup_revision >= 0, "claim_cas")
        previous = None
        if self._directory.exists():
            old, previous = _state(self._directory)
            _require(previous["revision"] == expected_cleanup_revision and old["phase"] in {"complete", "failed"}
                and old["native_returned"] is True, "previous_cleanup_blocked")
        else:
            _require(expected_cleanup_revision == 0, "claim_cas")
        tombstone = _tombstone(self._stage, self._binding, self.layout, operation_id)
        self._tombstone_sha = _sha(tombstone)
        conversation = tombstone["conversation_id"]
        plan = _capture(self._stage, self.destination, self.layout, conversation)
        claim_id = uuid4().hex
        fence = {"schema_version": 1, "scope": SCOPE, "copy_id": self.copy_id,
            "catalog_revision": expected_catalog_revision, "catalog_record_sha256": expected_record_sha256,
            "admission_root": str(self.root), "admission_identity": catalog._identity(self.root),
            "destination": str(self.destination), "root_identity": self._copy["root_identity"],
            "authority_head": self._authority_head, "operation_id": operation_id,
            "conversation_id": conversation, "layout_sha256": self._layout_sha,
            "claim_id": claim_id, "status": "invalidated"}
        fence_path = self.root / "admission/copy-cleanup/v1/fences" / (self.copy_id + ".json")
        _mkdir(fence_path.parent)
        if fence_path.exists():
            saved, raw = catalog._read(fence_path)
            _require(set(saved) == _FENCE_KEYS and saved["copy_id"] == self.copy_id
                and saved["destination"] == str(self.destination)
                and saved["root_identity"] == self._copy["root_identity"]
                and saved["layout_sha256"] == self._layout_sha, "fence_changed")
            self._stage._require_copy_not_invalidated(self.root, self.root / "__nonserving_fence_check__")
        else:
            raw = _json(fence) + b"\n"
            catalog._write_new(fence_path, raw)
        fence_sha = hashlib.sha256(raw).hexdigest()
        _mkdir(self._directory)
        record = {"schema_version": 1, "scope": SCOPE, "copy_id": self.copy_id, "claim_id": claim_id,
            "operation_id": operation_id, "conversation_id": conversation, "authority_head": self._authority_head,
            "catalog_revision": expected_catalog_revision, "catalog_record_sha256": expected_record_sha256,
            "layout_sha256": self._layout_sha, "fence_sha256": fence_sha, "phase": "claimed",
            "native_started": False, "native_returned": False, "receipt": None, "error_code": None}
        head = _append(self._directory, record, previous)
        self._claim = CopyCleanupClaim(self.copy_id, claim_id, operation_id, conversation,
            expected_catalog_revision, expected_record_sha256, head["revision"], head["sha256"],
            self._authority_head, self._layout_sha, fence_sha)
        self._claim_sha = _sha(self._claim.to_dict())
        self._record, self._head, self._plan = record, head, plan
        return self._claim

    def _validate_claim(self, claim):
        _require(claim is self._claim and type(claim) is CopyCleanupClaim
            and _sha(claim.to_dict()) == self._claim_sha, "local_claim_required")

    def _transition(self, **changes):
        self._current()
        record = dict(self._record, **changes)
        head = _append(self._directory, record, self._head)
        self._record, self._head = record, head

    def _native(self):
        try:
            return _perform(self, self._plan), None, False
        except BaseException as error:
            # Native controls are data here, never injected into the event loop.
            return None, type(error).__name__, self._database is not None

    async def _finish(self):
        native = asyncio.create_task(self._work.run(self._native), context=Context())
        self._native_task = native
        cancelled = False
        while not native.done():
            try:
                await asyncio.shield(native)
            except asyncio.CancelledError:
                cancelled = True
        outcome = native.result()
        data, failure, uncertain = outcome
        self._plan = None
        if failure is not None:
            self._transition(phase="unknown" if uncertain else "failed", native_returned=True,
                             error_code=failure)
            raise CopyCleanupError("copy_cleanup_native_failed")
        self._current()
        # The receipt pins its preceding running record. Its own durable event
        # is identified by read_copy_cleanup's record_sha256, without a hash cycle.
        receipt = CopyCleanupReceipt(self.copy_id, self._claim.claim_id, self._claim.operation_id,
            self._claim.conversation_id, self._head["revision"] + 1, self._head["sha256"],
            self._claim.fence_sha256, **data)
        self._transition(phase="complete", native_returned=True, receipt=receipt.to_dict())
        if cancelled:
            raise asyncio.CancelledError
        return receipt

    async def clean(self, claim: CopyCleanupClaim) -> CopyCleanupReceipt:
        """Observe the same retained operation; caller cancellation is not drain."""
        self._validate_claim(claim)
        if self._task is None:
            self._current()
            self._transition(phase="running", native_started=True)
            self._task = asyncio.create_task(self._finish(), context=Context())
        return await asyncio.shield(self._task)

    async def aclose(self) -> None:
        """Retain the lease until the Task and actual blocking owner have drained."""
        if self._closing is None:
            async def close():
                if self._task is not None:
                    await asyncio.gather(self._task, return_exceptions=True)
                await self._work.begin_close()
                _require(self._database is None, "connection_close_unknown")
                if self._lock_fd is not None:
                    os.close(self._lock_fd)
                    self._lock_fd = None
                if self._lease_fd is not None:
                    os.close(self._lease_fd)
                    self._lease_fd = None
                self._closed = True
                self._plan = None
                _CLEANUPS.discard(self)
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


def read_copy_cleanup(controller, copy_id: str) -> dict:
    """Recover bounded metadata under current authority, never a capability."""
    _, root, _, _, _, _ = catalog._authority(controller)
    directory = _base(root, copy_id)
    view = {"schema_version": 1, "scope": SCOPE, "copy_id": copy_id, "phase": "unclaimed",
        "cleanup_revision": 0, "record_sha256": None, "claim_id": None,
        "operation_id": None, "conversation_id": None, "native_started": False,
        "native_returned": False, "receipt": None, "error_code": None,
        "capability": False, "aggregate_copy_complete": False, "other_representations": "blocked"}
    if directory.exists() or directory.is_symlink():
        try:
            catalog._directory(directory)
            record, head = _state(directory)
            _require(record["copy_id"] == copy_id, "journal_identity")
            view.update({key: record[key] for key in ("phase", "claim_id", "operation_id", "conversation_id",
                "native_started", "native_returned", "receipt", "error_code")})
            view.update(cleanup_revision=head["revision"], record_sha256=head["sha256"])
            if record["phase"] in {"claimed", "running"}:
                # RAM/process liveness is not inferred by a recovery reader.
                view["phase"] = "pending"
        except (OSError, ValueError, TypeError, KeyError):
            view.update(phase="unknown", error_code="journal_unknown")
    return view
