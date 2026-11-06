"""Source-qualified offline images of actually closed managed state.

SQLite is read through a private copy and its public backup API. Registered
indexes are exported through the public SDK on another isolated copy. Neither
snapshot verification nor snapshot construction opens the original SDK.
"""

from contextlib import closing
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.stage_runtime import AdmissionController


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict:
    value = path.stat()
    return {"device": value.st_dev, "inode": value.st_ino}


def _stage():
    from src.core.runtime_admission import _stage as companion
    return companion()


def _relative(value: str) -> PurePosixPath:
    if type(value) is not str:
        raise ValueError("state_snapshot_unsafe_path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.as_posix() != value:
        raise ValueError("state_snapshot_unsafe_path")
    return path


def _regular(path: Path) -> None:
    if path.resolve(strict=True) != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("state_snapshot_unsafe_file")


def _read_json(path: Path) -> dict:
    _regular(path)
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("state_snapshot_record_schema")
    return value


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class ClosedStateSnapshot:
    """A closed-image DTO; mutable dicts are pinned by the captured manifest hash."""

    snapshot_root: Path
    source_binding_id: str
    source_binding: dict
    manifest: dict
    _manifest_sha256: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_manifest_sha256", hashlib.sha256(_json(self.manifest)).hexdigest())

    @property
    def manifest_sha256(self) -> str:
        """Canonical content identity captured when this immutable DTO was made."""
        return self._manifest_sha256


def _inventory(root: Path) -> dict:
    if root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError("state_snapshot_unsafe_root")
    result = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise ValueError("state_snapshot_symlink")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("state_snapshot_special_file")
        result[path.relative_to(root).as_posix()] = {
            "sha256": _sha(path), "size": info.st_size, "source_path": str(path),
            "source_identity": {"device": info.st_dev, "inode": info.st_ino},
        }
    return result


def _copy(source: Path, destination: Path) -> None:
    _regular(source)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with source.open("rb") as incoming:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _backup(source: Path, destination: Path) -> None:
    """Merge a copied SQLite WAL coherently, without opening original files."""
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as incoming:
        with closing(sqlite3.connect(destination)) as outgoing:
            incoming.backup(outgoing)
    # A backup can inherit WAL header mode. Canonicalize only this new file
    # before any read-only inspection can create unlisted WAL/SHM companions.
    with closing(sqlite3.connect(destination)) as canonical:
        checkpoint = canonical.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint[0] != 0:
            raise ValueError("state_snapshot_backup_busy")
        if canonical.execute("PRAGMA journal_mode=DELETE").fetchone()[0].lower() != "delete":
            raise ValueError("state_snapshot_backup_journal_mode")
    destination.chmod(0o600)


def _encode(value, root: Path):
    if type(value) is bytes:
        digest = hashlib.sha256(value).hexdigest()
        relative = "blobs/" + digest
        path = root / relative
        if path.exists():
            _regular(path)
            if _sha(path) != digest:
                raise ValueError("state_snapshot_blob_collision")
        else:
            _write(path, value)
        return {"$blob": {"path": relative, "sha256": digest, "size": len(value)}}
    if value is None or type(value) in (str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("state_snapshot_unsupported_sql_value")


def _database(path: Path, root: Path) -> dict:
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        schema = [dict(row) for row in db.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
        tables = {}
        for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            columns = [{"name": row[1], "type": row[2], "notnull": row[3],
                        "default": row[4], "pk": row[5]}
                       for row in db.execute("PRAGMA table_info(" + _quote(name) + ")")]
            rows = [{key: _encode(row[key], root) for key in row.keys()}
                    for row in db.execute("SELECT * FROM " + _quote(name))]
            rows.sort(key=_json)
            tables[name] = {"columns": columns,
                "primary_key": [row["name"] for row in sorted(columns, key=lambda item: item["pk"]) if row["pk"]],
                "rows": rows}
        return {"schema": schema, "application_id": db.execute("PRAGMA application_id").fetchone()[0],
                "user_version": db.execute("PRAGMA user_version").fetchone()[0], "tables": tables}


def _index_descriptors(raw: Path, binding: dict) -> list[dict]:
    stage = _stage()
    source = Path(binding["state_root"])
    result = {}
    catalogs = binding["authority"]["projection"]["tables"]
    for relative, projection in catalogs.items():
        columns = projection.get("schema_contract_v1", {}).get("columns", {})
        if "erasure_targets" not in columns:
            continue
        path = raw / _relative(relative)
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            for row in db.execute("SELECT id,participant,version,descriptor FROM erasure_targets "
                                  "WHERE participant='semantic_index'"):
                record = dict(row)
                record["descriptor"] = json.loads(record["descriptor"])
                stage._validate_target_record(record)
                descriptor = record["descriptor"]
                index = Path(descriptor["path"])
                if not index.is_relative_to(source):
                    continue  # A qualified foreign capability is never an SDK input here.
                if index.resolve() != index:
                    raise ValueError("state_snapshot_index_redirected")
                relative_index = index.relative_to(source).as_posix()
                present = index.exists()
                if present and (not index.is_dir()
                        or binding["store_roots"].get(relative_index) != _identity(index)):
                    raise ValueError("state_snapshot_index_identity")
                key = (relative_index, descriptor["collection"])
                result[key] = {"source_path": str(index), "relative_path": relative_index,
                    "source_identity": _identity(index) if present else None,
                    "collection": descriptor["collection"], "present": present}
    return [result[key] for key in sorted(result)]


def _sdk_json(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [_sdk_json(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _sdk_json(item) for key, item in value.items()}
    raise ValueError("state_snapshot_unsupported_sdk_value")


def _close_export_sdk(entry, client) -> None:
    if entry is not None:
        entry.close()
    elif client is not None:
        client.close()


def _export_indexes(raw: Path, descriptors: list[dict], binding: dict, work: Path, *,
                    copy_operation=None) -> list[dict]:
    """Only this function imports SDK code, after all source/path preflight."""
    expected = binding["sdk_versions"].get("chromadb")
    if any(item["present"] for item in descriptors):
        if expected is None or importlib.metadata.version("chromadb") != expected:
            raise ValueError("state_snapshot_sdk_version")
        import chromadb
        from chromadb.config import Settings
    exports = []
    grouped = {}
    for descriptor in descriptors:
        grouped.setdefault(descriptor["relative_path"], []).append(descriptor)
    for relative, entries in sorted(grouped.items()):
        if not entries[0]["present"]:
            for entry in entries:
                exports.append({**entry, "metadata": None, "sdk_version": expected,
                    "fields": ["ids", "documents", "metadatas", "embeddings"], "records": []})
            continue
        private = work / hashlib.sha256(relative.encode()).hexdigest()
        extent = None
        if copy_operation is not None:
            extent = copy_operation.extent(".work/sdk/" + private.name,
                family="semantic_index", representation="sdk_work")
        shutil.copytree(raw / _relative(relative), private, symlinks=False)
        sdk_entry = (copy_operation.sdk_enter(extent, sdk_version=expected)
                     if copy_operation is not None else None)
        client = None
        try:
            client = (sdk_entry.construct(chromadb.PersistentClient, path=str(private),
                        settings=Settings(anonymized_telemetry=False)) if sdk_entry is not None else
                      chromadb.PersistentClient(path=str(private), settings=Settings(anonymized_telemetry=False)))
            for entry in entries:
                collection = client.get_collection(name=entry["collection"], embedding_function=None)
                payload = collection.get(include=["documents", "metadatas", "embeddings"])
                ids = payload["ids"]
                embeddings = payload["embeddings"]
                if hasattr(embeddings, "tolist"):
                    embeddings = embeddings.tolist()
                if (type(ids) is not list or len(set(ids)) != len(ids)
                        or any(type(identity) is not str for identity in ids)
                        or collection.count() != len(ids)):
                    raise ValueError("state_snapshot_incomplete_sdk_export")
                arrays = [payload["documents"], payload["metadatas"], embeddings]
                if any(value is None or len(value) != len(ids) for value in arrays):
                    if ids:
                        raise ValueError("state_snapshot_incomplete_sdk_export")
                    arrays = [[], [], []]
                records = [{"id": identity, "document": _sdk_json(arrays[0][index]),
                            "metadata": _sdk_json(arrays[1][index]),
                            "embedding": _sdk_json(arrays[2][index])}
                           for index, identity in enumerate(ids)]
                for record in records:
                    if (record["document"] is not None and type(record["document"]) is not str
                            or record["metadata"] is not None and type(record["metadata"]) is not dict
                            or type(record["embedding"]) is not list or not record["embedding"]
                            or any(type(value) not in (int, float) for value in record["embedding"])):
                        raise ValueError("state_snapshot_invalid_sdk_record")
                records.sort(key=lambda item: item["id"])
                exports.append({**entry, "metadata": _sdk_json(collection.metadata), "sdk_version": expected,
                    "fields": ["ids", "documents", "metadatas", "embeddings"], "records": records})
        except BaseException as original:
            try:
                _close_export_sdk(sdk_entry, client)
            except BaseException as cleanup:
                raise BaseExceptionGroup("state snapshot SDK export retirement unconfirmed",
                                         [original, cleanup]) from None
            raise
        else:
            _close_export_sdk(sdk_entry, client)
        shutil.rmtree(private)
    return exports


def _history_references(policy_bytes: bytes | None, binding: dict) -> dict:
    expected = binding["ownership_sha256"]
    if expected is None:
        if policy_bytes is not None:
            raise ValueError("state_snapshot_ownership_changed")
        return {}
    if policy_bytes is None or hashlib.sha256(policy_bytes).hexdigest() != expected:
        raise ValueError("state_snapshot_ownership_changed")
    policy = json.loads(policy_bytes)
    if (type(policy) is not dict or set(policy) != {"schema_version", "source_bindings", "owned_targets",
            "delegated_targets", "historical_targets"} or policy["schema_version"] != 1
            or any(type(policy[key]) is not list for key in
                   ("source_bindings", "owned_targets", "delegated_targets", "historical_targets"))):
        raise ValueError("state_snapshot_history_policy")
    for key in ("source_bindings", "owned_targets", "delegated_targets"):
        values = policy[key]
        if (any(type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None for value in values)
                or len(set(values)) != len(values)):
            raise ValueError("state_snapshot_history_policy")
    if set(policy["owned_targets"]) & set(policy["delegated_targets"]):
        raise ValueError("state_snapshot_history_policy")
    references, seen = {}, set()
    for reference in policy["historical_targets"]:
        if (type(reference) is not dict or set(reference) != {"source_binding", "target_id", "archive_sha256"}
                or any(type(reference[key]) is not str or re.fullmatch("[0-9a-f]{64}", reference[key]) is None
                       for key in ("source_binding", "target_id", "archive_sha256"))
                or reference["source_binding"] not in policy["source_bindings"]):
            raise ValueError("state_snapshot_history_reference")
        key = (reference["source_binding"], reference["target_id"], reference["archive_sha256"])
        if key in seen:
            raise ValueError("state_snapshot_history_reference")
        seen.add(key)
        references.setdefault(reference["archive_sha256"], []).append(reference)
    return references


def _qualified_history(payload: bytes, archive_sha: str, references: list) -> dict:
    if (type(archive_sha) is not str or re.fullmatch("[0-9a-f]{64}", archive_sha) is None
            or hashlib.sha256(payload).hexdigest() != archive_sha):
        raise ValueError("state_snapshot_history_hash")
    document = json.loads(payload)
    if (type(document) is not dict or document.get("schema_version") != 1
            or type(document.get("targets")) is not list or not references
            or any(reference["archive_sha256"] != archive_sha
                   or reference["source_binding"] != document.get("source_binding") for reference in references)):
        raise ValueError("state_snapshot_history_qualification")
    for reference in references:
        targets = [target for target in document["targets"]
                   if type(target) is dict and target.get("id") == reference["target_id"]]
        if len(targets) != 1:
            raise ValueError("state_snapshot_history_target")
        target = targets[0]
        if (set(target) != {"id", "participant", "version", "descriptor"}
                or type(target["participant"]) is not str or type(target["version"]) is not int
                or target["version"] < 1 or type(target["descriptor"]) is not dict):
            raise ValueError("state_snapshot_history_target")
        encoded = json.dumps(target["descriptor"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        identity = hashlib.sha256(json.dumps([target["participant"], target["version"], encoded]).encode()).hexdigest()
        if target["id"] != identity:
            raise ValueError("state_snapshot_history_target")
    return document


def _copy_history(root: Path, raw: Path, binding: dict, destination: Path) -> tuple[dict, dict]:
    policy = raw / ".kza-target-ownership.json"
    references = _history_references(policy.read_bytes() if policy.exists() else None, binding)
    archives, files = {}, {}
    for archive_sha, refs in sorted(references.items()):
        original = root / "admission" / "history" / (archive_sha + ".json")
        _regular(original)
        before = _identity(original)
        payload = original.read_bytes()
        document = _qualified_history(payload, archive_sha, refs)
        source_binding = _stage().read_binding(root, document["source_binding"])
        if _json(document.get("source_binding_record", source_binding)) != _json(source_binding):
            raise ValueError("state_snapshot_history_qualification")
        relative = "history/" + archive_sha + ".json"
        _write(destination / relative, payload)
        if _identity(original) != before or _sha(original) != archive_sha:
            raise ValueError("state_snapshot_history_changed")
        record = {"path": relative, "sha256": archive_sha, "size": len(payload),
                  "source_binding": document["source_binding"], "references": refs,
                  "source_path": str(original), "source_identity": before}
        archives[archive_sha] = record
        files[relative] = {key: value for key, value in record.items()
                           if key in {"sha256", "size", "source_path", "source_identity"}}
        files[relative]["kind"] = "history_archive"
    return archives, files


def read_blob(snapshot: ClosedStateSnapshot, value: dict) -> bytes:
    """Read one typed SQL BLOB with exact content/size checks."""
    if type(value) is not dict or set(value) != {"$blob"} or type(value["$blob"]) is not dict:
        raise ValueError("state_snapshot_blob_schema")
    record = value["$blob"]
    if (set(record) != {"path", "sha256", "size"} or type(record["size"]) is not int
            or record["size"] < 0 or type(record["sha256"]) is not str):
        raise ValueError("state_snapshot_blob_schema")
    relative = _relative(record["path"])
    if relative.parts[0] != "blobs" or relative.name != record["sha256"]:
        raise ValueError("state_snapshot_blob_path")
    path = snapshot.snapshot_root / relative
    _regular(path)
    payload = path.read_bytes()
    if len(payload) != record["size"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError("state_snapshot_blob_changed")
    return payload


def read_history_archive(snapshot: ClosedStateSnapshot, archive_sha: str) -> dict:
    """Read only a copied archive qualified by this source's pinned ownership."""
    manifest = verify_snapshot(snapshot)
    policy_path = snapshot.snapshot_root / "state" / ".kza-target-ownership.json"
    references = _history_references(policy_path.read_bytes() if policy_path.exists() else None,
                                     snapshot.source_binding)
    archives = manifest.get("history_archives", {})
    if archive_sha not in references or archive_sha not in archives:
        raise ValueError("state_snapshot_history_unreferenced")
    record = archives[archive_sha]
    relative = "history/" + archive_sha + ".json"
    if (type(record) is not dict or record.get("path") != relative or record.get("sha256") != archive_sha
            or record.get("references") != references[archive_sha]
            or record.get("source_binding") != references[archive_sha][0]["source_binding"]
            or manifest["files"].get(relative, {}).get("kind") != "history_archive"):
        raise ValueError("state_snapshot_history_qualification")
    path = snapshot.snapshot_root / relative
    payload = path.read_bytes()
    if len(payload) != record["size"]:
        raise ValueError("state_snapshot_history_hash")
    return _qualified_history(payload, archive_sha, references[archive_sha])


def verify_snapshot(snapshot: ClosedStateSnapshot) -> dict:
    """Verify a captured DTO and its immutable files without SDK/source access."""
    if type(snapshot) is not ClosedStateSnapshot:
        raise TypeError("state_snapshot_dto_required")
    root = snapshot.snapshot_root
    manifest = _read_json(root / "manifest.json")
    if (hashlib.sha256(_json(manifest)).hexdigest() != snapshot._manifest_sha256
            or manifest != snapshot.manifest or manifest.get("schema_version") != 1
            or manifest.get("source_binding_id") != snapshot.source_binding_id
            or manifest.get("source_binding") != snapshot.source_binding
            or hashlib.sha256(_json(snapshot.source_binding)).hexdigest() != snapshot.source_binding_id):
        raise ValueError("state_snapshot_manifest_changed")
    if (root / "manifest.json").read_bytes() != _json(manifest) + b"\n":
        raise ValueError("state_snapshot_manifest_changed")
    _verify_files(root, manifest["files"], manifest_present=True)
    return manifest


def _verify_files(root: Path, files: dict, *, manifest_present: bool) -> None:
    actual = _inventory(root)
    expected = set(files) | ({"manifest.json"} if manifest_present else set())
    if set(actual) != expected:
        raise ValueError("state_snapshot_inventory_changed")
    for relative, record in files.items():
        _relative(relative)
        if actual[relative]["sha256"] != record["sha256"] or actual[relative]["size"] != record["size"]:
            raise ValueError("state_snapshot_file_changed")


def _record_snapshot_failure(operation, destination: Path, root_identity: dict | None,
                             source_binding_id: str) -> list:
    """Attempt independent durable and local failure records without losing errors."""
    failures = []
    try:
        operation.failed("snapshot_failed")
    except BaseException as error:
        failures.append(error)
    if root_identity is None:
        return failures
    try:
        if (destination.resolve(strict=True) != destination or destination.is_symlink()
                or _identity(destination) != root_identity):
            raise ValueError("state_snapshot_destination_changed")
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            os.replace(manifest_path, destination / "manifest.incomplete.json")
        _write(destination / "incomplete.json", _json({"schema_version": 1,
               "source_binding_id": source_binding_id, "status": "incomplete"}) + b"\n")
    except BaseException as error:
        failures.append(error)
    return failures


def snapshot_closed_state(controller: "AdmissionController", binding_id: str,
                          destination: Path) -> ClosedStateSnapshot:
    """Capture a source-qualified image while the caller retains admission.

    A failed copy remains private and incomplete. No success manifest is
    published before public SDK close and a second closed-source verification.
    Cross-store crash atomicity and payload reconciliation are separate work.
    """
    from src.memory.copy_catalog import reserve_copy

    stage = _stage()
    root = Path(controller.root)
    binding = stage.verify_closed_binding(root, binding_id)
    source = Path(binding["state_root"])
    destination = Path(destination).absolute()
    if (destination.parent.resolve(strict=True) != destination.parent
            or destination.is_relative_to(source) or source.is_relative_to(destination)
            or destination.is_relative_to(root / "slots")):
        raise ValueError("state_snapshot_unsafe_destination")
    inventory = _inventory(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    head = stage.read_admission_head(root)
    operation = reserve_copy(controller, "snapshot", destination, head["binding_id"], (binding_id,))
    root_identity = None
    try:
        created = operation.create_root()
        if created != destination:
            raise ValueError("state_snapshot_destination_changed")
        root_identity = _identity(destination)
        for relative, representation in ((".work/raw", "raw"), ("state", "state"),
                ("blobs", "blob"), ("history", "history_archive"), ("manifest.json", "manifest"),
                ("manifest.incomplete.json", "manifest"), ("incomplete.json", "preparation")):
            operation.extent(relative, family="managed_payload", representation=representation)
        work = destination / ".work"
        work.mkdir(mode=0o700)
        raw = work / "raw"
        raw.mkdir(mode=0o700)
        for relative in sorted(inventory):
            _copy(source / _relative(relative), raw / _relative(relative))
        for relative in binding["store_roots"]:
            if relative != ".":
                (raw / _relative(relative)).mkdir(parents=True, exist_ok=True, mode=0o700)
        if _inventory(source) != inventory:
            raise ValueError("state_snapshot_source_changed")
        history_archives, history_files = _copy_history(root, raw, binding, destination)
        descriptors = _index_descriptors(raw, binding)
        sdk_roots = {PurePosixPath(item["relative_path"]) for item in descriptors if item["present"]}
        for left in sdk_roots:
            if any(left != right and left.is_relative_to(right) for right in sdk_roots):
                raise ValueError("state_snapshot_overlapping_sdk_roots")
        databases = set(binding["schemas"])
        for relative in databases:
            path = _relative(relative)
            if path.name == "chroma.sqlite3" and not any(path.is_relative_to(base) for base in sdk_roots):
                raise ValueError("state_snapshot_unregistered_sdk_root")
        files, images = dict(history_files), {}
        for relative, record in sorted(inventory.items()):
            path = _relative(relative)
            sdk = any(path.is_relative_to(base) for base in sdk_roots)
            if not sdk and any(relative == name + suffix for name in databases
                               for suffix in ("-wal", "-shm", "-journal")):
                continue
            copied = destination / "state" / path
            if relative in databases and not sdk:
                _backup(raw / path, copied)
                images[relative] = _database(copied, destination)
                kind = "sqlite_backup"
            else:
                _copy(raw / path, copied)
                kind = "sdk_opaque" if sdk else "file"
            files["state/" + relative] = {**record, "sha256": _sha(copied),
                                         "size": copied.stat().st_size, "kind": kind}
        for relative in binding["store_roots"]:
            (destination / "state" / relative).mkdir(parents=True, exist_ok=True, mode=0o700)
        exports = _export_indexes(raw, descriptors, binding, work / "sdk", copy_operation=operation)
        if stage.verify_closed_binding(root, binding_id) != binding or _inventory(source) != inventory:
            raise ValueError("state_snapshot_source_changed")
        proof = _read_json(root / "admission" / "closures" / (binding_id + ".json"))
        receipt = _read_json(root / "admission" / "receipts" / (proof["running_head"]["run_id"] + ".json"))
        blobs = destination / "blobs"
        if blobs.exists():
            for relative, record in _inventory(blobs).items():
                files["blobs/" + relative] = {**record, "source_path": None,
                                            "source_identity": None, "kind": "blob"}
        manifest = {"schema_version": 1, "source_binding_id": binding_id,
            "source_binding": binding, "source_closure": {"proof": proof, "receipt": receipt},
            "source_roots": binding["store_roots"], "source_schemas": binding["schemas"],
            "source_inventory": inventory, "files": files, "databases": images, "index_exports": exports,
            "history_archives": history_archives}
        shutil.rmtree(work)  # SDK close already returned; no unknown owner is erased.
        _verify_files(destination, files, manifest_present=False)
        _write(destination / "manifest.json", _json(manifest) + b"\n")
        snapshot = ClosedStateSnapshot(destination, binding_id, binding, manifest)
        verify_snapshot(snapshot)
        if stage.verify_closed_binding(root, binding_id) != binding or _inventory(source) != inventory:
            raise ValueError("state_snapshot_source_changed")
        operation.prepared(destination / "manifest.json")
        operation.finish()
        return snapshot
    except BaseException as original:
        failures = _record_snapshot_failure(operation, destination, root_identity, binding_id)
        if failures:
            raise BaseExceptionGroup("state snapshot failure persistence unconfirmed", [original, *failures]) from None
        raise
