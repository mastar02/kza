"""Qualify closed journal history before registering a derived local journal.

Only scheduling/locator records change. Protected ledger records are copied
from the latest closed image, never merged with an older authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
from types import MappingProxyType

_JOURNAL = ("conversation_tombstones", "conversation_erasures", "erasure_targets",
            "erasure_steps", "policy_index_deletions", "index_jobs")
_SCHEDULING = frozenset({"erasure_targets", "erasure_steps", "policy_index_deletions", "index_jobs"})
_LOCATORS = frozenset({"path", "root", "db_path", "audio_dir"})


def _require(value, code):
    if not value:
        raise ValueError("state_journal_" + code)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _stage():
    from src.core.runtime_admission import _stage as trusted_stage
    return trusted_stage()


def _relative(value):
    _require(type(value) is str and value, "privacy_relative")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts and str(path) == value,
             "privacy_relative")
    return path


def _target_id(participant, version, descriptor):
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(json.dumps([participant, version, encoded]).encode()).hexdigest()


def _quote(value):
    return '"' + value.replace('"', '""') + '"'


def _token(value):
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    return value


def _projection(db, *, immutable=False):
    result = {}
    for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        if immutable and name in _SCHEDULING:
            continue
        records = []
        for row in db.execute("SELECT * FROM " + _quote(name)):
            records.append({key: _token(row[key]) for key in row.keys()
                            if not (immutable and name == "wake_clips_v1" and key == "target_id")})
        result[name] = sorted(records, key=_json)
    return result


def _snapshot_projection(snapshot, database):
    from src.memory.state_snapshot import read_blob

    result = {}
    for name, table in database["tables"].items():
        records = []
        for row in table["rows"]:
            records.append({key: _token(read_blob(snapshot, value))
                            if type(value) is dict and "$blob" in value else value
                            for key, value in row.items()})
        result[name] = sorted(records, key=_json)
    return result


def _source(stage, root, snapshot, privacy_relative, *, latest):
    from src.memory.state_snapshot import verify_snapshot

    manifest = verify_snapshot(snapshot)
    binding = stage.read_binding(root, snapshot.source_binding_id)
    _require(binding == snapshot.source_binding == manifest["source_binding"], "source_binding")
    proof = stage._closure_proof(root, snapshot.source_binding_id)
    receipt = stage._read_json(root / "admission" / "receipts" /
                               (proof["running_head"]["run_id"] + ".json"))
    _require(manifest["source_closure"] == {"proof": proof, "receipt": receipt}, "source_closure")
    if latest:
        head = stage.read_admission_head(root)
        _require(head is not None and head["status"] == "closed"
                 and head["binding_id"] == snapshot.source_binding_id, "latest_not_current")
        _require(stage.verify_closed_binding(root, snapshot.source_binding_id) == binding,
                 "source_binding")
    database = manifest["databases"].get(privacy_relative)
    _require(type(database) is dict and set(_JOURNAL) <= set(database["tables"]), "journal_schema")
    ledgers = {name for name, value in manifest["databases"].items()
               if "erasure_targets" in value["tables"]}
    _require(ledgers == {privacy_relative}, "multiple_active_journals")
    _require(not any(name not in _JOURNAL and name.startswith(
        ("erasure_", "conversation_erasure", "policy_index_deletion", "index_jobs"))
        for name in database["tables"]), "unknown_journal_history")
    targets = []
    for row in database["tables"]["erasure_targets"]["rows"]:
        target = dict(row)
        _require(set(target) == {"id", "participant", "version", "descriptor"}, "target_columns")
        target["descriptor"] = json.loads(target["descriptor"])
        stage._validate_target_record(target)
        targets.append(target)
    _require(targets and len({target["id"] for target in targets}) == len(targets), "target_inventory")
    policy_path = snapshot.snapshot_root / "state" / ".kza-target-ownership.json"
    policy = stage._read_json(policy_path) if policy_path.exists() else None
    expected = binding["ownership_sha256"]
    _require((policy is None and expected is None) or
             (policy is not None and hashlib.sha256(policy_path.read_bytes()).hexdigest() == expected),
             "source_ownership")
    return {"snapshot": snapshot, "manifest": manifest, "binding": binding,
            "database": database, "targets": sorted(targets, key=lambda item: item["id"]),
            "policy": policy, "closure": {"proof": proof, "receipt": receipt},
            "delegations": list(stage._delegation_records(root, Path(binding["state_root"])))}


def _relocations(stage, state, sources, relocations):
    _require(isinstance(relocations, Mapping), "relocations")
    originals = {}
    for source in sources:
        for target in source["targets"]:
            previous = originals.setdefault(target["id"], target)
            _require(previous == target, "source_target_collision")
    _require(set(relocations) == set(originals), "relocation_inventory")
    mapped, records = {}, {}
    for identity, original in originals.items():
        descriptor = dict(relocations[identity])
        old = original["descriptor"]
        _require(set(descriptor) == set(old), "descriptor_shape")
        for name in descriptor:
            if name not in _LOCATORS or not isinstance(old[name], str) or not old[name].startswith("/"):
                _require(descriptor[name] == old[name], "descriptor_semantics")
        target = {"id": _target_id(original["participant"], original["version"], descriptor),
                  "participant": original["participant"], "version": original["version"],
                  "descriptor": descriptor}
        stage._validate_target_record(target)
        paths = tuple(stage._descriptor_paths(descriptor))
        for value in paths:
            path = Path(value)
            _require(path.is_relative_to(state) and path.resolve() == path, "foreign_relocation")
        if tuple(stage._descriptor_paths(old)):
            _require(paths and target["id"] != identity, "old_locator_reused")
        else:
            _require(descriptor == old, "descriptor_semantics")
        mapped[identity] = target["id"]
        previous = records.setdefault(target["id"], target)
        _require(previous == target, "derived_target_collision")
    return mapped, tuple(records[key] for key in sorted(records))


def _event_maps(sources, target_id_map, records):
    """Keep event-only journal IDs distinct from EventDependency target hashes."""
    from src.analytics.event_evidence import event_target_id

    current = {target["id"]: target for target in records}
    journal_ids, evidence_ids = {}, {}
    for source in sources:
        for target in source["targets"]:
            if target["participant"] != "events":
                continue
            destination = current[target_id_map[target["id"]]]
            _require(target["version"] == destination["version"] == 1, "event_target_version")
            journal_ids[target["id"]] = destination["id"]
            old = event_target_id(target["descriptor"])
            new = event_target_id(destination["descriptor"])
            _require(evidence_ids.setdefault(old, new) == new, "event_target_collision")
    return journal_ids, evidence_ids


def _payload_cut_proof(stage, proof, sources, state):
    """Bind preparation evidence to exact source images and ordinary rows."""
    from src.memory.payload_derivation import _proof_source_rows, _validate_cut_proof_shape

    proof = json.loads(_json(proof))
    _validate_cut_proof_shape(proof)
    latest = sources[0]["snapshot"]
    _require(proof["latest_binding_id"] == latest.source_binding_id
             and proof["latest_manifest_sha256"] == hashlib.sha256(_json(latest.manifest)).hexdigest(),
             "cut_latest_source")
    _require(any(proof["ordinary_binding_id"] == source["snapshot"].source_binding_id
                 and proof["ordinary_manifest_sha256"] == hashlib.sha256(
                     _json(source["snapshot"].manifest)).hexdigest() for source in sources),
             "cut_ordinary_source")
    _require(proof["destination_root"] == str(state)
             and proof["destination_identity"] == stage._identity(state), "cut_destination")
    _proof_source_rows(proof, latest)
    return proof


def _archive(source, privacy_relative, *, payload_cut_proof=None):
    snapshot, database = source["snapshot"], source["database"]
    tables = {name: database["tables"][name] for name in _JOURNAL}
    # Quarantine metadata remains history, never a new execution capability.
    tables.update({name: value for name, value in database["tables"].items()
                   if name.startswith("legacy_quarantine")})
    archive = {"schema_version": 1, "source_binding": snapshot.source_binding_id,
            "source_binding_record": source["binding"], "source_closure": source["closure"],
            "snapshot_root": str(snapshot.snapshot_root),
            "snapshot_manifest_sha256": hashlib.sha256(_json(source["manifest"])).hexdigest(),
            "privacy_relative": privacy_relative, "targets": source["targets"],
            "journal": {"schema": database["schema"], "tables": tables},
            "ownership_history": source["policy"],
            "historical_delegations": source.get("delegations", []), "current_execution_grant": False,
            "known_copy_obligations": [
                {"kind": "source_state", "root": source["binding"]["state_root"],
                 "identity": source["binding"]["state_identity"], "status": "historical_not_current_proof"},
                {"kind": "closed_snapshot", "root": str(snapshot.snapshot_root / "state"),
                 "status": "known_retained_nonserving_copy"}]}
    if payload_cut_proof is not None:
        archive["payload_cut_proof"] = payload_cut_proof
    return archive


def _preflight_database(path, source):
    _require(path.is_file() and not path.is_symlink(), "destination_database")
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
             and not info.st_mode & 0o077, "unsafe_database")
    snapshot = source["snapshot"]
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        schema = [dict(row) for row in db.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
        _require(schema == source["database"]["schema"], "destination_schema")
        _require(_projection(db) == _snapshot_projection(snapshot, source["database"]),
                 "destination_not_latest_clone")
        _require(not db.execute("PRAGMA foreign_key_check").fetchall(), "destination_foreign_key")
        tombstones = {row[0] for row in db.execute("SELECT conversation_id FROM conversation_tombstones")}
        operations = {row[0] for row in db.execute("SELECT conversation_id FROM conversation_erasures")}
        _require(tombstones == operations, "operation_inventory")
        for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            for row in db.execute("PRAGMA foreign_key_list(" + _quote(name) + ")"):
                if row[2] == "erasure_targets":
                    _require((name, row[3], row[4]) in {
                        ("wake_clips_v1", "target_id", "id"),
                        ("policy_index_deletions", "target_id", "id")}, "unknown_target_reference")


@contextmanager
def _writer_lease(path):
    descriptor = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                 and not info.st_mode & 0o077, "unsafe_writer_lease")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _replace_current(ledger, records, target_id_map):
    """Register actual pending work, sharing the source ledger transaction API."""
    from src.memory.erasure_journal import ErasureParticipant

    with ledger._transaction() as db:
        immutable = _projection(db, immutable=True)
        db.execute("PRAGMA defer_foreign_keys=ON")
        for table in ("erasure_steps", "policy_index_deletions", "index_jobs", "erasure_targets"):
            db.execute("DELETE FROM " + _quote(table))
        for target in records:
            identity = ledger.erasures._register_target(db, ErasureParticipant(target["participant"]),
                                                       target["version"], target["descriptor"])
            _require(identity == target["id"], "registered_identity")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "wake_clips_v1" in tables:
            for row in list(db.execute("SELECT id,target_id FROM wake_clips_v1")):
                _require(row["target_id"] in target_id_map, "payload_target_missing")
                db.execute("UPDATE wake_clips_v1 SET target_id=? WHERE id=?",
                           (target_id_map[row["target_id"]], row["id"]))
        for row in db.execute("SELECT id,revision,state FROM facts").fetchall():
            _require(row["state"] in {"automatic", "confirmed", "proposed", "deleted", "quarantined"},
                     "fact_state")
            if row["state"] != "proposed":
                operation = "delete" if row["state"] in {"deleted", "quarantined"} else "upsert"
                ledger._queue(db, row["id"], row["revision"], operation)
        _require(immutable == _projection(db, immutable=True), "protected_records_changed")
        _require(not db.execute("PRAGMA foreign_key_check").fetchall(), "derived_foreign_key")
        _require(not db.execute("SELECT 1 FROM erasure_steps WHERE state!='pending' OR attempts!=0 "
                                "OR claim_token IS NOT NULL OR cursor IS NOT NULL LIMIT 1").fetchone(),
                 "inherited_receipt")
        return {"steps": db.execute("SELECT COUNT(*) FROM erasure_steps").fetchone()[0],
                "index_jobs": db.execute("SELECT COUNT(*) FROM index_jobs").fetchone()[0],
                "index_cleanup": db.execute("SELECT COUNT(*) FROM policy_index_deletions").fetchone()[0]}


@dataclass(frozen=True)
class JournalDerivationReceipt:
    source_binding_id: str
    destination_state: Path
    privacy_relative: str
    history_sha256: str
    history_path: Path
    history_archives: tuple
    target_id_map: Mapping
    registered_target_ids: tuple[str, ...]
    pending_counts: Mapping
    event_target_id_map: Mapping = field(default_factory=dict)
    event_dependency_target_map: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "target_id_map", MappingProxyType(dict(self.target_id_map)))
        object.__setattr__(self, "pending_counts", MappingProxyType(dict(self.pending_counts)))
        object.__setattr__(self, "event_target_id_map", MappingProxyType(dict(self.event_target_id_map)))
        object.__setattr__(self, "event_dependency_target_map",
                           MappingProxyType(dict(self.event_dependency_target_map)))

    def to_dict(self):
        """Serialize preparation evidence, never an erasure completion ACK."""
        return {"schema_version": 1, "source_binding_id": self.source_binding_id,
                "destination_state": str(self.destination_state), "privacy_relative": self.privacy_relative,
                "history_sha256": self.history_sha256, "history_path": str(self.history_path),
                "history_archives": [list(item) for item in self.history_archives],
                "target_id_map": dict(self.target_id_map),
                "registered_target_ids": list(self.registered_target_ids),
                "pending_counts": dict(self.pending_counts),
                "event_target_id_map": dict(self.event_target_id_map),
                "event_dependency_target_map": dict(self.event_dependency_target_map),
                "acknowledges_retained_copy_cleanup": False}


def _qualify_locked(controller, latest_snapshot, destination_state, *, privacy_relative,
                    target_relocations, ordinary_snapshot, payload_cut_proof):
    from src.memory.fact_ledger import FactLedger

    stage = _stage()
    controller._require_lease()
    root = Path(controller.root).resolve(strict=True)
    stage._require_root_lease(root)
    relative = _relative(privacy_relative)
    latest = _source(stage, root, latest_snapshot, privacy_relative, latest=True)
    sources = [latest]
    if ordinary_snapshot is not None and ordinary_snapshot is not latest_snapshot:
        sources.append(_source(stage, root, ordinary_snapshot, privacy_relative, latest=False))
    state = Path(destination_state)
    _require(state.is_absolute() and state.resolve(strict=True) == state
             and state.parent.parent == root / "slots" and state.name == "state", "destination_state")
    _require(all(str(state) != source["binding"]["state_root"] for source in sources),
             "destination_is_source")
    _, checked_state, _ = stage._checked_slot(root, state.parent.name)
    _require(checked_state == state, "destination_state")
    stage._assert_writers_closed(state)
    path = state / relative
    _require(path.resolve(strict=True) == path, "destination_database")
    for source in sources:
        for original in (Path(source["binding"]["state_root"]) / relative,
                         source["snapshot"].snapshot_root / "state" / relative):
            _require(not original.exists() or not os.path.samefile(path, original), "aliased_database")
    _preflight_database(path, latest)
    existing_policy = state / ".kza-target-ownership.json"
    if existing_policy.exists() or existing_policy.is_symlink():
        _require(stage._read_json(existing_policy) == latest["policy"], "destination_ownership")
    target_id_map, records = _relocations(stage, state, sources, target_relocations)
    event_ids, evidence_ids = _event_maps(sources, target_id_map, records)
    cut_proof = (_payload_cut_proof(stage, payload_cut_proof, sources, state)
                 if payload_cut_proof is not None else None)
    archives, histories, source_bindings = [], [], set()
    for source in sources:
        archive = _archive(source, privacy_relative,
                           payload_cut_proof=cut_proof if source is latest else None)
        raw = _json(archive) + b"\n"
        _require(len(raw) <= 16 * 1024 * 1024, "history_capacity")
        digest = hashlib.sha256(raw).hexdigest()
        archive_path = root / "admission" / "history" / (digest + ".json")
        archives.append((source["snapshot"].source_binding_id, digest, archive_path, archive))
        source_bindings.add(source["snapshot"].source_binding_id)
        histories.extend({"source_binding": source["snapshot"].source_binding_id,
                          "target_id": target["id"], "archive_sha256": digest}
                         for target in source["targets"])
        if source["policy"] is not None:
            source_bindings.update(source["policy"]["source_bindings"])
            histories.extend(source["policy"]["historical_targets"])
    for identity in source_bindings:
        stage.read_binding(root, identity)
    policy = {"schema_version": 1, "source_bindings": sorted(source_bindings),
              "owned_targets": [], "delegated_targets": [],
              "historical_targets": sorted({_json(item): item for item in histories}.values(), key=_json)}
    blocking_policy = dict(policy)
    head = stage.read_admission_head(root)
    preparing = False
    try:
        with _writer_lease(path):
            _preflight_database(path, latest)
            for _, _, archive_path, archive in archives:
                stage._atomic_record(archive_path, archive, immutable=True)
            # Incomplete preparation cannot satisfy its nonempty active inventory.
            stage._atomic_record(existing_policy, blocking_policy)
            preparing = True
            ledger = FactLedger(str(path))
            try:
                ledger.initialize()
                pending = _replace_current(ledger, records, target_id_map)
            finally:
                ledger.close()
            _require(stage.read_admission_head(root) == head, "head_changed")
            _source(stage, root, latest_snapshot, privacy_relative, latest=True)
        policy["owned_targets"] = sorted(target["id"] for target in records)
        stage._atomic_record(existing_policy, policy)
        binding = stage._binding_snapshot(root, state.parent.name)
        stage.validate_target_ownership(root, binding)
    except BaseException as error:
        if preparing:
            try:
                stage._atomic_record(existing_policy, blocking_policy)
            except BaseException as poison_error:
                raise BaseExceptionGroup("state journal preparation unconfirmed", [error, poison_error])
        raise
    latest_archive = archives[0]
    return JournalDerivationReceipt(latest_snapshot.source_binding_id, state, privacy_relative,
        latest_archive[1], latest_archive[2],
        tuple((identity, digest, str(location)) for identity, digest, location, _ in archives),
        target_id_map, tuple(policy["owned_targets"]), pending, event_ids, evidence_ids)


def qualify_and_register_derived_journal(controller, latest_snapshot, destination_state, *,
                                         privacy_relative: str, target_relocations: Mapping,
                                         ordinary_snapshot=None,
                                         payload_cut_proof=None) -> JournalDerivationReceipt:
    """Archive whole history and register local work under controller exclusion.

    Destination is a non-serving exact latest-state clone. The optional older
    image proves additional target IDs in selected payload. Neither history nor
    this receipt acknowledges retained copies. No child is launched here.
    """
    stage = _stage()
    controller._require_lease()
    root = Path(controller.root).resolve(strict=True)
    stage._require_root_lease(root)
    with stage._locked_root(root):
        return _qualify_locked(controller, latest_snapshot, destination_state,
            privacy_relative=privacy_relative, target_relocations=target_relocations,
            ordinary_snapshot=ordinary_snapshot, payload_cut_proof=payload_cut_proof)
