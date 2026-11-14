"""Owner-state compaction, data side (U3b-0c): the identity clone of H into K.

Owner-compaction design v2 data_model (identity mode, D1-D5), v4 F3 (VACUUM
only on K, scope-guarded) and v5 (review E3: derivation never appends to the
deletion ledger). This module runs in the compaction worker under the
admission lease (src.core.compaction_worker); the controller side lives in
scripts/compaction.py.

Identity mode keeps every row, file and index record of the closed head H.
Nothing is cut and no source deadline is parsed: whatever the live app would
expire is expired by K's own code during its verification run. Only locators
change:

* erasure targets: every path-bearing descriptor (H's own and the inherited
  ones in its owners, the P3 selection recorded in the intent) maps to the
  same relative path under K, many-to-one (relocation by the existing
  journal derivation, which also replays every recorded erasure on K's
  targets and rebuilds the scheduling tables);
* wake_clips_v1.target_id, through that same journal derivation;
* suggestions_policy_v1.evidence_json, by the shared stage function
  compaction_evidence_json (rules a-c);
* utterances.audio_path, a path under H or an owner, to the same relative
  path under K (stricter than the design, which did not list this column:
  without it a retained utterance would point into a destroyed state).

Any other absolute locator of H or an owner inside a database refuses
(compaction_absolute_locator). Refusal codes are content-free.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
from typing import Mapping

from src.memory.state_snapshot import ClosedStateSnapshot, read_blob, verify_snapshot

# Tables the journal derivation deletes and rebuilds on K (state_journal_derivation._SCHEDULING).
SCHEDULING_TABLES = frozenset({"erasure_targets", "erasure_steps", "policy_index_deletions", "index_jobs"})
# Columns compaction may rewrite, with the rule that rewrites them.
RELOCATED_COLUMNS = {("wake_clips_v1", "target_id"): "target_id_map",
                     ("suggestions_policy_v1", "evidence_json"): "evidence",
                     ("utterances", "audio_path"): "path"}
# Files the derivation itself writes into K beside the cloned state.
DERIVATION_FILES = frozenset({".kza-target-ownership.json", ".kza-payload-preparation.json",
                              ".kza-state-binding.json"})
SQLITE_SIDE_SUFFIXES = ("-wal", "-shm", "-journal")


class CompactionCloneError(ValueError):
    """A content-free refusal of the identity clone (``str(error)`` is the code)."""


def _require(value, code: str) -> None:
    if not value:
        raise CompactionCloneError("runtime_admission_compaction_" + code)


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _digest(value) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _stage():
    from src.core.runtime_admission import _stage as trusted_stage
    return trusted_stage()


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class CompactionClonePlan:
    """Read-only plan of the identity clone of H's closed snapshot into K's state."""

    latest: ClosedStateSnapshot = field(repr=False)
    now: float
    destination: Path
    privacy_relative: str
    source_roots: tuple[str, ...]
    target_relocations: Mapping[str, dict]
    evidence_map: Mapping[str, str]
    plan_sha256: str


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def relocate_path(value: str, roots: tuple[str, ...], destination: Path) -> str:
    """One absolute locator under H or an owner, at the same relative path under K."""
    matches = [root for root in roots if _under(value, root)]
    _require(len(matches) >= 1, "target_foreign")
    # The longest root wins (an owner's state is never nested in another, but be exact).
    root = max(matches, key=len)
    relative = PurePosixPath(value).relative_to(root)
    _require(".." not in relative.parts, "target_foreign")
    return str(destination / relative) if relative.parts else str(destination)


def _absolute_strings(value):
    if type(value) is dict:
        for child in value.values():
            yield from _absolute_strings(child)
    elif type(value) is list:
        for child in value:
            yield from _absolute_strings(child)
    elif type(value) is str and Path(value).is_absolute():
        yield value


def _relocate_descriptor(descriptor, roots, destination):
    """Relocate the top-level locator keys the journal derivation accepts; anything else refuses.

    state_journal_derivation._relocations admits a changed value only under
    its locator keys (path, root, db_path, audio_dir) at the top level, so an
    absolute path anywhere else could never be relocated: refuse it here.
    """
    from src.memory.state_journal_derivation import _LOCATORS
    _require(type(descriptor) is dict, "target_unsupported")
    result = {}
    for key, child in descriptor.items():
        if key in _LOCATORS and type(child) is str and child.startswith("/"):
            result[key] = relocate_path(child, roots, destination)
        else:
            _require(not any(True for _ in _absolute_strings(child)), "target_unsupported")
            result[key] = child
    return result


def _targets(snapshot: ClosedStateSnapshot, privacy_relative: str) -> dict:
    database = snapshot.manifest["databases"].get(privacy_relative)
    _require(type(database) is dict and "erasure_targets" in database["tables"], "privacy_missing")
    rows = database["tables"]["erasure_targets"]["rows"]
    return {row["id"]: {**row, "descriptor": json.loads(row["descriptor"])} for row in rows}


def target_relocations(snapshot: ClosedStateSnapshot, privacy_relative: str, roots: tuple[str, ...],
                       destination: Path) -> dict:
    """Every target of H's inventory (owned and inherited) relocated into K (design v2 D2)."""
    return {identity: _relocate_descriptor(target["descriptor"], roots, destination)
            for identity, target in sorted(_targets(snapshot, privacy_relative).items())}


def _evidence_map(snapshot: ClosedStateSnapshot, privacy_relative: str, relocations: dict) -> dict:
    from src.analytics.event_evidence import event_target_id
    result = {}
    for identity, target in _targets(snapshot, privacy_relative).items():
        if target["participant"] != "events" or target["version"] != 1:
            continue
        old, new = event_target_id(target["descriptor"]), event_target_id(relocations[identity])
        _require(result.setdefault(old, new) == new, "event_target_collision")
    return result


def _text_values(database: dict):
    for name, table in database["tables"].items():
        if name in SCHEDULING_TABLES:
            continue
        for row in table["rows"]:
            for column, value in row.items():
                if type(value) is str:
                    yield name, column, value


def _check_locators(snapshot: ClosedStateSnapshot, roots: tuple[str, ...]) -> None:
    """Refuse any absolute locator of H or an owner outside the relocated columns (design v2 P6/H4)."""
    needles = tuple(root.rstrip("/") + "/" for root in roots) + roots
    for relative, database in sorted(snapshot.manifest["databases"].items()):
        for name, column, value in _text_values(database):
            if not any(needle in value for needle in needles):
                continue
            if (name, column) == ("utterances", "audio_path"):
                _require(any(_under(value, root) for root in roots), "absolute_locator:" + name + "." + column)
                continue
            raise CompactionCloneError("runtime_admission_compaction_absolute_locator:" + relative
                                       + ":" + name + "." + column)


def _events_rows(snapshot: ClosedStateSnapshot, relative: str) -> dict:
    database = snapshot.manifest["databases"].get(relative)
    if type(database) is not dict or "events_policy_v1" not in database["tables"]:
        return {}
    return {row["id"]: row for row in database["tables"]["events_policy_v1"]["rows"]}


def _check_evidence(snapshot, privacy_relative, relocations, evidence_map, destination) -> None:
    """Rules (a)-(c): a parent present in K must project the remapped dependency exactly."""
    from src.analytics.event_evidence import EventDependency, event_target_id, project_pattern_event
    stage = _stage()
    new_targets = {}
    for identity, target in _targets(snapshot, privacy_relative).items():
        if target["participant"] == "events" and target["version"] == 1:
            descriptor = relocations[identity]
            path = Path(descriptor["path"])
            _require(path.is_relative_to(destination), "event_target_foreign")
            new_targets[event_target_id(descriptor)] = path.relative_to(destination).as_posix()
    for database in snapshot.manifest["databases"].values():
        table = database["tables"].get("suggestions_policy_v1")
        if table is None:
            continue
        for row in table["rows"]:
            remapped = json.loads(stage.compaction_evidence_json(row["evidence_json"], evidence_map))
            for entry in remapped:
                relative = new_targets.get(entry.get("target_id"))
                if relative is None:
                    continue  # (b) unknown to H's inventory: kept as it is
                parent = _events_rows(snapshot, relative).get(entry.get("event_id"))
                if parent is None:
                    continue  # (c) the live app keeps a suggestion whose parent expired
                decoded = {key: read_blob(snapshot, value) if type(value) is dict else value
                           for key, value in parent.items()}
                _require(project_pattern_event(decoded, entry["target_id"]).dependency
                         == EventDependency.from_dict(entry), "event_lineage_changed")


def plan_compaction_clone(latest: ClosedStateSnapshot, *, now: float, destination: Path, privacy_relative: str,
                          source_roots: tuple[str, ...]) -> CompactionClonePlan:
    """Deterministic, read-only plan of the identity clone (no SDK, no destination I/O)."""
    _require(type(latest) is ClosedStateSnapshot and type(now) in (int, float) and math.isfinite(now),
             "plan_input")
    verify_snapshot(latest)
    destination = Path(destination)
    _require(destination.is_absolute() and ".." not in destination.parts, "destination_invalid")
    head_root = latest.source_binding["state_root"]
    roots = tuple(sorted(set(source_roots) | {head_root}))
    _require(all(Path(root).is_absolute() and not _under(str(destination), root) and not _under(root, str(destination))
                 for root in roots), "destination_invalid")
    _check_locators(latest, roots)
    relocations = target_relocations(latest, privacy_relative, roots, destination)
    evidence_map = _evidence_map(latest, privacy_relative, relocations)
    _check_evidence(latest, privacy_relative, relocations, evidence_map, destination)
    value = {"latest": latest.manifest_sha256, "now": now, "destination": str(destination),
             "privacy_relative": privacy_relative, "source_roots": list(roots),
             "target_relocations": relocations, "evidence_map": evidence_map, "mode": "compaction_identity"}
    return CompactionClonePlan(latest, float(now), destination, privacy_relative, roots, relocations,
                               evidence_map, _digest(value))


def _identity_proof(plan: CompactionClonePlan, destination: Path) -> dict:
    """The exclusion proof of compaction: nothing removed (design v2 D3, kind compaction_identity)."""
    info = destination.stat()
    return {"schema_version": 1, "kind": "compaction_identity",
            "ordinary_binding_id": plan.latest.source_binding_id,
            "latest_binding_id": plan.latest.source_binding_id,
            "ordinary_manifest_sha256": plan.latest.manifest_sha256,
            "latest_manifest_sha256": plan.latest.manifest_sha256, "plan_sha256": plan.plan_sha256,
            "destination_root": str(destination),
            "destination_identity": {"device": info.st_dev, "inode": info.st_ino},
            "removed_rows": [], "removed_files": []}


def _relocate_columns(plan: CompactionClonePlan, destination: Path) -> dict:
    """Rewrite evidence_json and audio_path in K (wake clip targets are the journal's)."""
    stage = _stage()
    counts = {"evidence_json": 0, "audio_path": 0}
    for relative, database in sorted(plan.latest.manifest["databases"].items()):
        tables = database["tables"]
        if "suggestions_policy_v1" not in tables and "utterances" not in tables:
            continue
        path = destination / relative
        _require(path.is_file() and path.resolve() == path, "destination_database")
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("BEGIN IMMEDIATE")
            if "suggestions_policy_v1" in tables:
                for identity, text in db.execute("SELECT id,evidence_json FROM suggestions_policy_v1").fetchall():
                    new = stage.compaction_evidence_json(text, plan.evidence_map)
                    if new != text:
                        db.execute("UPDATE suggestions_policy_v1 SET evidence_json=? WHERE id=?", (new, identity))
                        counts["evidence_json"] += 1
            if "utterances" in tables:
                for identity, value in db.execute(
                        "SELECT id,audio_path FROM utterances WHERE audio_path IS NOT NULL").fetchall():
                    if any(_under(value, root) for root in plan.source_roots):
                        db.execute("UPDATE utterances SET audio_path=? WHERE id=?",
                                   (relocate_path(value, plan.source_roots, destination), identity))
                        counts["audio_path"] += 1
    return counts


def _expected_rows(plan: CompactionClonePlan, name: str, rows: list[dict], journal) -> list[dict]:
    stage = _stage()
    if name == "wake_clips_v1":
        return [dict(row, target_id=journal.target_id_map.get(row["target_id"], row["target_id"])) for row in rows]
    if name == "suggestions_policy_v1":
        return [dict(row, evidence_json=stage.compaction_evidence_json(row["evidence_json"], plan.evidence_map))
                for row in rows]
    if name == "utterances":
        destination = plan.destination
        return [dict(row, audio_path=relocate_path(row["audio_path"], plan.source_roots, destination)
                     if type(row["audio_path"]) is str and any(_under(row["audio_path"], root)
                                                               for root in plan.source_roots)
                     else row["audio_path"]) for row in rows]
    return rows


def verify_identity(plan: CompactionClonePlan, destination: Path, journal) -> dict:
    """D5: K equals H exactly except the relocated columns and the rebuilt scheduling tables.

    Per database and table, row sets compare as typed projections (BLOBs by
    sha256); per file, relative path and sha256; SDK-opaque stores byte for
    byte (stricter than record ids). Extra files in K beyond the derivation
    records and SQLite side files refuse. Returns content-free counts.
    """
    from src.memory.payload_derivation import _privacy_schema_added
    from src.memory.state_journal_derivation import _projection, _snapshot_projection, _token
    counts = {"databases": 0, "tables": 0, "rows": 0, "files": 0}
    manifest = plan.latest.manifest
    expected_files = set()
    for key, record in sorted(manifest["files"].items()):
        if not key.startswith("state/"):
            continue
        relative = key.removeprefix("state/")
        if relative in DERIVATION_FILES:
            continue
        expected_files.add(relative)
        if record["kind"] == "sqlite_backup":
            continue
        path = destination / relative
        _require(path.is_file() and not path.is_symlink(), "payload_changed:file_missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(digest == record["sha256"], "payload_changed:file")
        counts["files"] += 1
    for relative, database in sorted(manifest["databases"].items()):
        path = destination / relative
        _require(path.is_file() and path.resolve() == path, "payload_changed:database_missing")
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            schema = [dict(row) for row in db.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
            actual = _projection(db)
        source_schema = database["schema"]
        added = set()
        if schema != source_schema:
            groups = None if relative == plan.privacy_relative else _authority_groups()
            _require(_privacy_schema_added(source_schema, schema, groups), "payload_changed:schema")
            added = {row["name"] for row in schema if row["type"] == "table"} - set(database["tables"])
        expected = _snapshot_projection(plan.latest, database)
        _require(set(actual) == set(expected) | added, "payload_changed:tables")
        for name in sorted(actual):
            if name in added:
                _require(actual[name] == [], "payload_changed:added_rows")
                continue
            if name in SCHEDULING_TABLES:
                continue
            if name == "sqlite_sequence":
                # The scheduling tables' AUTOINCREMENT counters move with their rebuild.
                kept = [row for row in actual[name] if row["name"] not in SCHEDULING_TABLES]
                wanted = sorted((row for row in expected[name] if row["name"] not in SCHEDULING_TABLES), key=_json)
                _require(sorted(kept, key=_json) == wanted, "payload_changed:sqlite_sequence")
                counts["tables"] += 1
                continue
            rows = _expected_rows(plan, name, [dict(row) for row in database["tables"][name]["rows"]], journal)
            wanted = sorted(({key: _token(read_blob(plan.latest, value)) if type(value) is dict and "$blob" in value
                              else value for key, value in row.items()} for row in rows), key=_json)
            _require(actual[name] == wanted, "payload_changed:" + name)
            counts["tables"] += 1
            counts["rows"] += len(wanted)
        if relative == plan.privacy_relative:
            ids = sorted(row["id"] for row in actual["erasure_targets"])
            _require(ids == sorted(journal.registered_target_ids), "payload_changed:erasure_targets")
        counts["databases"] += 1
    allowed = set(DERIVATION_FILES)
    for relative in manifest["databases"]:
        allowed.update(relative + suffix for suffix in SQLITE_SIDE_SUFFIXES)
    for path in destination.rglob("*"):
        _require(not path.is_symlink(), "payload_changed:symlink")
        if path.is_dir():
            continue
        relative = path.relative_to(destination).as_posix()
        if relative not in expected_files and relative.endswith(".lock") and relative[:-5] in manifest["databases"]:
            # The journal qualification's own writer lease (an empty 0600 file) beside a database.
            _require(path.stat().st_size == 0, "payload_changed:extra_file")
            continue
        _require(relative in expected_files or relative in allowed, "payload_changed:extra_file")
    return counts


def _authority_groups():
    from src.memory.payload_derivation import _authority_schema_additions
    return _authority_schema_additions()


def vacuum_compacted_state(plan: CompactionClonePlan, destination: Path, scope: Path) -> list[str]:
    """D4 (user decision P5_vacuum_new_state): VACUUM only K's own SQLite files.

    ``scope`` is K's state root as the compaction log records it; any path
    not under it refuses (compaction_vacuum_scope), so no retirement member
    can ever be rewritten. SDK-opaque stores (chroma) are not vacuumed: their
    free-page residue is a documented limit. Returns the vacuumed relative paths.
    """
    scope = Path(scope)
    _require(Path(destination) == scope and scope.resolve(strict=True) == scope, "vacuum_scope")
    done = []
    for relative, record in sorted(plan.latest.manifest["files"].items()):
        if not relative.startswith("state/") or record["kind"] != "sqlite_backup":
            continue
        path = scope / relative.removeprefix("state/")
        _require(path.resolve(strict=True).is_relative_to(scope) and not path.is_symlink(), "vacuum_scope")
        with closing(sqlite3.connect(path, isolation_level=None)) as db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.execute("VACUUM")
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        done.append(relative.removeprefix("state/"))
    return done


@dataclass(frozen=True)
class CompactionClone:
    """Materialization evidence of K (non-serving until the controller authorizes it)."""

    destination: Path
    plan_sha256: str
    evidence_map: dict
    target_id_map: dict
    history_sha256: str
    counts: dict
    vacuumed: tuple[str, ...]
    preparation_record: Path


def materialize_compaction_clone(plan: CompactionClonePlan, destination: Path, *, controller, scope: Path,
                                 before_vacuum=None) -> CompactionClone:
    """D3-D5 at K's final locator under the controller lease (keeps K blocked until the end).

    Order: clone H's image; journal derivation with the compaction_identity
    proof (relocates every target, replays every erasure, rebuilds the
    scheduling tables; protected rows, the deletion ledger included, must be
    unchanged); column relocations; D5; VACUUM (D4); D5 again; preparation
    record; the copy-catalog 'prepared' inventory; then the qualified ownership.
    ``before_vacuum`` is a test hook (crash injection mid-phase).
    """
    from src.memory.copy_catalog import reserve_copy
    from src.memory.payload_derivation import _atomic_file, _clone_latest, _prepared_ownership
    from src.memory.state_journal_derivation import _source, qualify_and_register_derived_journal
    _require(type(plan) is CompactionClonePlan and Path(destination) == plan.destination, "destination_mismatch")
    checked = plan_compaction_clone(plan.latest, now=plan.now, destination=plan.destination,
                                    privacy_relative=plan.privacy_relative, source_roots=plan.source_roots)
    _require(checked.plan_sha256 == plan.plan_sha256, "plan_changed")
    controller._require_lease()
    stage = _stage()
    _source(stage, Path(controller.root), plan.latest, plan.privacy_relative, latest=True)
    info = plan.latest.snapshot_root.stat()
    reference = {"source_binding_id": plan.latest.source_binding_id, "snapshot_root": str(plan.latest.snapshot_root),
                 "root_identity": {"device": info.st_dev, "inode": info.st_ino},
                 "manifest_sha256": plan.latest.manifest_sha256}
    operation = reserve_copy(controller, "derived_state", destination, plan.latest.source_binding_id,
                             (plan.latest.source_binding_id,), source_snapshot_refs=(reference,))
    try:
        operation.create_root()
        operation.extent(".", family="managed_payload", representation="derived_state")
        _clone_latest(plan, destination)  # every latest state file except the marker (K24)
        record = destination / ".kza-payload-preparation.json"
        _atomic_file(record, _json({"schema_version": 1, "plan_sha256": plan.plan_sha256, "phase": "preparing",
                                    "serving": False, "mode": "compaction_identity"}) + b"\n")
        journal = qualify_and_register_derived_journal(
            controller, plan.latest, destination, privacy_relative=plan.privacy_relative,
            target_relocations=plan.target_relocations, ordinary_snapshot=None,
            payload_cut_proof=_identity_proof(plan, destination), keep_blocked=True)
        ownership, blocked, qualified = _prepared_ownership(destination, journal)
        relocated = _relocate_columns(plan, destination)
        counts = verify_identity(plan, destination, journal)
        if before_vacuum is not None:
            before_vacuum()
        vacuumed = tuple(vacuum_compacted_state(plan, destination, scope))
        counts = verify_identity(plan, destination, journal)  # D5 again after VACUUM (v4 F3)
        _require(ownership.read_bytes() == blocked, "preparation_changed")
        evidence = {"schema_version": 1, "plan_sha256": plan.plan_sha256, "phase": "materialized",
                    "serving": False, "mode": "compaction_identity", "journal": journal.to_dict(),
                    "relocated": relocated, "counts": counts, "vacuumed": list(vacuumed),
                    "retained_copy_cleanup_ack": False}
        _atomic_file(record, _json(evidence) + b"\n")
        operation.prepared(record)
        _atomic_file(ownership, qualified)
        operation.finish()
    except BaseException as error:
        try:
            operation.failed(type(error).__name__)
        except BaseException as catalog_error:
            raise BaseExceptionGroup("compaction preparation and catalog failure", [error, catalog_error]) from None
        raise
    return CompactionClone(destination, plan.plan_sha256, dict(plan.evidence_map), dict(journal.target_id_map),
                           journal.history_sha256, counts, vacuumed, record)


# --- the worker's phases C2b (snapshot) and C3 (materialize, D4, D5) --------------------------------

def _load_snapshot(path: Path, binding_id: str, manifest_sha256: str | None = None) -> ClosedStateSnapshot:
    manifest = json.loads((path / "manifest.json").read_bytes())
    _require(type(manifest) is dict and manifest.get("source_binding_id") == binding_id, "snapshot_changed")
    snapshot = ClosedStateSnapshot(path, binding_id, manifest["source_binding"], manifest)
    verify_snapshot(snapshot)
    _require(manifest_sha256 is None or snapshot.manifest_sha256 == manifest_sha256, "snapshot_changed")
    return snapshot


def _events(log: dict, kind: str) -> list[dict]:
    return [event for event in log["events"] if event["kind"] == kind]


def run_materialization(root: Path, compaction_id: str, *, wall, before_vacuum=None) -> dict:
    """The worker's work for one compaction: C2b then C3, logged, resumable.

    Runs under its own lease with gate ``compaction:<id>``. Requires the
    controller's 'k_runtime' event (K exists as a placeholder slot) and no
    unresolved snapshot reservation (the controller destroys a partial
    snapshot before it starts the worker again). A materialization that was
    started but not finished is never redone over a partly written K: it
    refuses with compaction_resume_abort_required (design v2 C3), unless K's
    own preparation record already says 'materialized' and D5 holds again.
    """
    from src.memory.state_snapshot import snapshot_closed_state
    stage = _stage()
    root = Path(root).resolve(strict=True)
    with stage.AdmissionController(root, gate="compaction:" + compaction_id) as controller:
        log = controller.compaction
        intent = log["intent"]
        _require(_events(log, "k_runtime") and not _events(log, "materialized"), "phase_order")
        _require(not any(event["kind"] in {"abort_started", "aborted"} for event in log["events"]), "phase_order")
        head = stage.read_admission_head(root)
        _require(head == intent["head"], "head_moved")
        binding = stage.verify_closed_binding(root, intent["source_binding_id"])
        reserved = {event["data"]["n"]: event for event in _events(log, "snapshot_reserved")}
        done = {event["data"]["n"]: event for event in _events(log, "snapshot")}
        destroyed = {event["data"]["n"] for event in _events(log, "snapshot_destroyed")}
        _require(set(reserved) - set(done) <= destroyed, "snapshot_partial")
        live = [event for n, event in sorted(done.items()) if n not in destroyed]
        if live:
            data = live[-1]["data"]
            snapshot = _load_snapshot(Path(data["path"]), intent["source_binding_id"], data["manifest_sha256"])
        else:
            n = max(reserved, default=0) + 1
            path = root / "compaction" / compaction_id / ("snapshot-" + str(n))
            stage.private_directories(path.parent)
            log = stage.append_compaction_event(root, log, "snapshot_reserved", {"n": n, "path": str(path)},
                                                at=wall())
            snapshot = snapshot_closed_state(controller, intent["source_binding_id"], path)
            log = stage.append_compaction_event(root, log, "snapshot", {
                "n": n, "path": str(path), "identity": stage._identity(path),
                "manifest_sha256": snapshot.manifest_sha256}, at=wall())
        destination = Path(intent["k_state_root"])
        scope = Path(_events(log, "k_runtime")[-1]["data"]["state_root"])
        _require(scope == destination, "vacuum_scope")
        roots = tuple(item["state_root"] for item in intent["owners"])
        materializing = _events(log, "materializing")
        if materializing:
            now = materializing[-1]["at"]
            plan = plan_compaction_clone(snapshot, now=now, destination=destination,
                                         privacy_relative=intent["privacy_relative"], source_roots=roots)
            return _finish_interrupted(root, log, plan, destination, wall)
        now = float(wall())
        plan = plan_compaction_clone(snapshot, now=now, destination=destination,
                                     privacy_relative=intent["privacy_relative"], source_roots=roots)
        _require(plan.latest.source_binding == binding, "head_moved")
        log = stage.append_compaction_event(root, log, "materializing", {"plan_sha256": plan.plan_sha256}, at=now)
        clone = materialize_compaction_clone(plan, destination, controller=controller, scope=scope,
                                             before_vacuum=before_vacuum)
        data = {"plan_sha256": clone.plan_sha256, "evidence_map": clone.evidence_map,
                "history_sha256": clone.history_sha256, "counts": clone.counts, "vacuumed": list(clone.vacuumed),
                "state_identity": stage._identity(destination)}
        stage.append_compaction_event(root, log, "materialized", data, at=wall())
        return {"phase": "materialized", **{key: data[key] for key in ("counts", "vacuumed")}}


def _finish_interrupted(root: Path, log: dict, plan: CompactionClonePlan, destination: Path, wall) -> dict:
    """Accept a materialization that finished before its event; anything else needs abort."""
    from src.memory.state_journal_derivation import JournalDerivationReceipt
    stage = _stage()
    _require(plan.plan_sha256 == _events(log, "materializing")[-1]["data"]["plan_sha256"],
             "resume_abort_required")
    record = destination / ".kza-payload-preparation.json"
    try:
        evidence = json.loads(record.read_bytes())
        policy = json.loads((destination / ".kza-target-ownership.json").read_bytes())
    except (OSError, ValueError):
        raise CompactionCloneError("runtime_admission_compaction_resume_abort_required") from None
    _require(type(evidence) is dict and evidence.get("phase") == "materialized"
             and evidence.get("plan_sha256") == plan.plan_sha256 and evidence.get("mode") == "compaction_identity"
             and type(policy) is dict and policy.get("owned_targets"), "resume_abort_required")
    journal = evidence["journal"]
    receipt = JournalDerivationReceipt(journal["source_binding_id"], destination, journal["privacy_relative"],
                                       journal["history_sha256"], Path(journal["history_path"]),
                                       tuple(tuple(item) for item in journal["history_archives"]),
                                       journal["target_id_map"], tuple(journal["registered_target_ids"]),
                                       journal["pending_counts"], journal["event_target_id_map"],
                                       journal["event_dependency_target_map"])
    _require(sorted(policy["owned_targets"]) == sorted(receipt.registered_target_ids), "resume_abort_required")
    counts = verify_identity(plan, destination, receipt)
    data = {"plan_sha256": plan.plan_sha256, "evidence_map": dict(plan.evidence_map),
            "history_sha256": receipt.history_sha256, "counts": counts, "vacuumed": evidence["vacuumed"],
            "state_identity": stage._identity(destination)}
    stage.append_compaction_event(root, log, "materialized", data, at=wall())
    return {"phase": "materialized", "counts": counts, "vacuumed": evidence["vacuumed"], "resumed": True}
