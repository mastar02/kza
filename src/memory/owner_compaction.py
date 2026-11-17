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
* pending_audio_deletions.path and audio_write_intents.path (SC-1b,
  sc-release-design v4 start.relocation_fix), the queued deletions and write
  intents of ambient audio, the same way: a value under H or an owner is
  admitted and relocated; one that names H or an owner without being under
  it refuses. Both tables are protected, so A1 remaps H's expected projection
  the same way (stage_runtime._compaction_authority_equivalent).

Any other absolute locator of H or an owner inside a database refuses
(compaction_absolute_locator). Refusal codes are content-free.

Chroma (SC-4c, sc-release-design v4 start.chroma_allowlist, PRIV-3, PA-8,
V4R2-C1) is the one exception to identity: a closed collection list. Every
Chroma store of H is rebuilt in K through the public SDK from the
snapshot's export (never a byte copy, so nothing left out survives in free
pages or index files):

* the policy memory collection (a registered semantic_index target,
  ``<collection>_policy_v1``) gets only the records of facts get() would
  return at t_clone (live, unexpired, no tombstoned source) whose owner is
  not in retired_users, with the deletion ledger as a second gate;
* the command-index collections (home_assistant_commands, its pointer and
  its versioned staging collections) are copied whole;
* home_assistant_routines is not copied (a named drop: dictated routine
  text with no erasure path; K's boot recreates it empty);
* any other collection, the legacy 'user_memories' included, refuses
  (compaction_unlisted_collection), at plan (scripts/compaction.py) and
  here, in the snapshot and in the plan.

Each store is written by a fresh child interpreter that gets only what K
keeps (CHROMA_REBUILD_CHILD): Chroma persists index buffers whose unused
capacity can hold earlier heap contents, and this worker's heap holds H's
whole snapshot. D5 for rebuilt stores: the child reads its store back and
its afterimage (exactly the planned collections, each with its planned
metadata, configuration and records) must hash to the plan's; then the
store's files are pinned by sha256 and every later D5 (after VACUUM, on
resume) compares that pin. The child never outlives the worker as a writer
(SC4C-RV-1): it dies with the worker on Linux, exits before each write once
the worker is gone, holds the worker's admission lease while it lives (so
abort, resume and plan refuse runtime_admission_busy) and is killed after
CHROMA_REBUILD_TIMEOUT.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
from types import SimpleNamespace
from typing import Mapping

from src.memory.state_snapshot import ClosedStateSnapshot, read_blob, verify_snapshot

# Tables the journal derivation deletes and rebuilds on K (state_journal_derivation._SCHEDULING).
SCHEDULING_TABLES = frozenset({"erasure_targets", "erasure_steps", "policy_index_deletions", "index_jobs"})
# Columns compaction may rewrite, with the rule that rewrites them.
RELOCATED_COLUMNS = {("wake_clips_v1", "target_id"): "target_id_map",
                     ("suggestions_policy_v1", "evidence_json"): "evidence",
                     ("utterances", "audio_path"): "path",
                     ("pending_audio_deletions", "path"): "path",
                     ("audio_write_intents", "path"): "path"}
# Path locators relocated under K (rule "path"); a value naming H or an owner must lie under it.
PATH_COLUMNS = tuple(sorted(key for key, rule in RELOCATED_COLUMNS.items() if rule == "path"))
# SC-1b relocation fix: the two protected queues whose path is their primary key.
QUEUE_PATH_TABLES = frozenset({"pending_audio_deletions", "audio_write_intents"})
# Files the derivation itself writes into K beside the cloned state.
DERIVATION_FILES = frozenset({".kza-target-ownership.json", ".kza-payload-preparation.json",
                              ".kza-state-binding.json"})
SQLITE_SIDE_SUFFIXES = ("-wal", "-shm", "-journal")
FREE_MARGIN = 256 << 20  # P9 (scripts/compaction.FREE_MARGIN)

# SC-4c closed collection list (scripts/compaction.py holds the same list for plan; a test pins both).
CHROMA_COMMAND_INDEX = "home_assistant_commands"  # with "__pointer" and the "__<version>" staging collections
CHROMA_DROPPED = frozenset({"home_assistant_routines"})  # V4R2-C1: not copied; K's boot recreates it empty
CHROMA_KINDS = ("rebuilt", "copied", "dropped")
_FACT_LIVE_STATES = ("automatic", "confirmed")
_FACT_LEDGER_KINDS = ("fact_deleted", "fact_pruned_retired_owner")


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


def collection_kind(name: str, *, policy: bool) -> str:
    """SC-4c: the closed list. 'rebuilt' (policy memory), 'copied' (command index), 'dropped' (routines).

    ``policy`` is True only for a collection H registered as its own
    semantic_index target in that store. Anything else refuses.
    """
    if policy:
        return "rebuilt"
    if name == CHROMA_COMMAND_INDEX or name.startswith(CHROMA_COMMAND_INDEX + "__"):
        return "copied"
    if name in CHROMA_DROPPED:
        return "dropped"
    raise CompactionCloneError("runtime_admission_compaction_unlisted_collection")


def _snapshot_class(relative: str, name: str, registered: bool) -> str:
    """The snapshot's classifier (state_snapshot._export_stores): a dropped collection is never read."""
    return "omit" if collection_kind(name, policy=registered) == "dropped" else "export"


@dataclass(frozen=True)
class ChromaCollectionPlan:
    """One collection of one of H's Chroma stores, as K gets it (records empty when dropped)."""

    store: str
    collection: str
    kind: str
    metadata: dict | None = field(repr=False)
    configuration: dict | None = field(repr=False)
    records: tuple = field(repr=False)
    excluded: int

    def summary(self) -> dict:
        """Content-free: names, counts and the digest of K's records."""
        return {"store": self.store, "collection": self.collection, "kind": self.kind,
                "records": len(self.records), "excluded": self.excluded,
                "records_sha256": _digest([self.metadata, self.configuration, list(self.records)])}


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
    chroma_stores: tuple[str, ...] = ()
    chroma: tuple[ChromaCollectionPlan, ...] = field(default=(), repr=False)


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
            if (name, column) in PATH_COLUMNS:
                _require(any(_under(value, root) for root in roots), "absolute_locator:" + name + "." + column)
                continue
            raise CompactionCloneError("runtime_admission_compaction_absolute_locator:" + relative
                                       + ":" + name + "." + column)


def _relocated_value(value, roots: tuple[str, ...], destination: Path):
    """A path column's value in K: relocated when it lies under H or an owner, else unchanged."""
    if type(value) is str and any(_under(value, root) for root in roots):
        return relocate_path(value, roots, destination)
    return value


def _check_queue_relocations(snapshot: ClosedStateSnapshot, roots: tuple[str, ...], destination: Path) -> None:
    """SC-1b: the queues' paths are primary keys; two H rows that relocate to one K path refuse.

    Many-to-one relocation (an owner's file and H's file at the same relative
    path) would make K's UPDATE violate the key: refused here, read-only and
    content-free, before anything is written.
    """
    for _relative, database in sorted(snapshot.manifest["databases"].items()):
        for name in sorted(QUEUE_PATH_TABLES & set(database["tables"])):
            values = [row["path"] for row in database["tables"][name]["rows"]]
            relocated = [_relocated_value(value, roots, destination) for value in values]
            _require(len(set(relocated)) == len(relocated), "locator_collision:" + name + ".path")


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


def _rows(database: dict, name: str) -> list[dict]:
    table = database["tables"].get(name)
    return table["rows"] if type(table) is dict else []


def policy_allowlist(snapshot: ClosedStateSnapshot, now: float) -> dict:
    """fact id -> (revision, content) of every fact get() would return at ``now`` with a non-retired owner.

    get() (fact_ledger.FactLedger._eligible): state automatic or confirmed,
    not expired, no source conversation tombstoned. Owner (fact_owners)
    not in any retired_users table of H. Second gate: no fact deletion of
    that id in the database's deletion ledger (salted digest).
    """
    from src.memory.deletion_ledger import subject_digest
    retired = {row["user_id"] for database in snapshot.manifest["databases"].values()
               for row in _rows(database, "retired_users")}
    allowed = {}
    for _relative, database in sorted(snapshot.manifest["databases"].items()):
        facts = _rows(database, "facts")
        if not facts:
            continue
        tombstoned = {row["conversation_id"] for row in _rows(database, "conversation_tombstones")}
        conversations = {}
        for row in _rows(database, "fact_sources"):
            conversations.setdefault(row["fact_id"], set()).add(row["conversation_id"])
        owners = {row["fact_id"]: row["person_id"] for row in _rows(database, "fact_owners")}
        salts = [row["salt"] for row in _rows(database, "deletion_ledger_salt_v1") if type(row["salt"]) is str]
        recorded = {row["subject_sha256"] for row in _rows(database, "deletion_ledger_v1")
                    if row["kind"] in _FACT_LEDGER_KINDS}
        for fact in facts:
            identity, expires = fact["id"], fact["expires_at"]
            if (fact["state"] not in _FACT_LIVE_STATES or type(fact["content"]) is not str
                    or type(fact["revision"]) is not int
                    or expires is not None and not (type(expires) in (int, float) and expires > now)
                    or conversations.get(identity, set()) & tombstoned
                    or owners.get(identity) in retired
                    or any(subject_digest(salt, identity) in recorded for salt in salts)):
                continue
            allowed[identity] = (fact["revision"], fact["content"])
    return allowed


def _policy_record(record: dict, allowed: dict) -> bool:
    """A record of the policy collection K keeps: its fact is allowed and it indexes that exact revision."""
    fact = allowed.get(record["id"])
    return (fact is not None and record["document"] == fact[1]
            and record["metadata"] == {"fact_policy": 1, "revision": fact[0]})


def chroma_plan(snapshot: ClosedStateSnapshot, now: float) -> tuple[tuple[str, ...], tuple[ChromaCollectionPlan, ...]]:
    """SC-4c: H's Chroma stores and, per collection, what K gets (refuses an unlisted collection)."""
    listed = snapshot.manifest.get("sdk_collections")
    _require(type(listed) is dict and listed.get("schema_version") == 1 and type(listed.get("stores")) is list,
             "unlisted_collection")
    stores = tuple(listed["stores"])
    allowed = None
    plans = []
    for export in snapshot.manifest["index_exports"]:
        _require(export["relative_path"] in stores, "unlisted_collection")
        kind = collection_kind(export["collection"], policy=export["registered"] is True)
        _require(export["omitted"] is (kind == "dropped"), "unlisted_collection")
        records = tuple(export["records"])
        excluded = 0
        if kind == "dropped":
            records, excluded = (), 0
        elif kind == "rebuilt":
            if allowed is None:
                allowed = policy_allowlist(snapshot, now)
            kept = tuple(record for record in records if _policy_record(record, allowed))
            records, excluded = kept, len(records) - len(kept)
        plans.append(ChromaCollectionPlan(export["relative_path"], export["collection"], kind,
                                          export["metadata"], export["configuration"], records, excluded))
    plans.sort(key=lambda item: (item.store, item.collection))
    _require(len({(item.store, item.collection) for item in plans}) == len(plans), "unlisted_collection")
    return stores, tuple(plans)


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
    _check_queue_relocations(latest, roots, destination)
    relocations = target_relocations(latest, privacy_relative, roots, destination)
    evidence_map = _evidence_map(latest, privacy_relative, relocations)
    _check_evidence(latest, privacy_relative, relocations, evidence_map, destination)
    stores, chroma = chroma_plan(latest, now)
    value = {"latest": latest.manifest_sha256, "now": now, "destination": str(destination),
             "privacy_relative": privacy_relative, "source_roots": list(roots),
             "target_relocations": relocations, "evidence_map": evidence_map, "mode": "compaction_identity",
             "chroma_stores": list(stores), "chroma": [item.summary() for item in chroma]}
    return CompactionClonePlan(latest, float(now), destination, privacy_relative, roots, relocations,
                               evidence_map, _digest(value), stores, chroma)


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


def _in_store(relative: str, stores: tuple[str, ...]) -> bool:
    return any(PurePosixPath(relative).is_relative_to(store) for store in stores)


def _clone_state(plan: CompactionClonePlan, destination: Path) -> None:
    """payload_derivation._clone_latest without H's Chroma stores (SC-4c: rebuilt, never byte-copied)."""
    from src.memory.payload_derivation import _clone_latest
    if not plan.chroma_stores:
        _clone_latest(plan, destination)
        return
    files = {key: record for key, record in plan.latest.manifest["files"].items()
             if not (key.startswith("state/") and _in_store(key.removeprefix("state/"), plan.chroma_stores))}
    # The same verified reads (each file's size and sha256 against its record), over the filtered list.
    view = SimpleNamespace(manifest={"files": files}, snapshot_root=plan.latest.snapshot_root)
    _clone_latest(SimpleNamespace(latest=view), destination)


def _store_pin(destination: Path, store: str) -> dict:
    """relative path -> sha256 of every file of one rebuilt store (no link, no special file)."""
    root = destination / store
    _require(root.is_dir() and not root.is_symlink() and root.resolve() == root, "payload_changed:chroma")
    pin = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "payload_changed:symlink")
        if path.is_dir():
            continue
        _require(path.is_file(), "payload_changed:chroma")
        pin[path.relative_to(destination).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pin


# SC-4c: the rebuild runs in a fresh interpreter that imports no src and gets only what K keeps (stdin).
# Chroma persists HNSW buffers whose unused capacity can hold whatever its allocator handed out, so a
# store written by this worker (whose heap holds H's snapshot: excluded records, expired facts) could
# carry fragments of it into K's files. The child's heap holds K's planned records and the SDK only.
# It builds each collection, re-reads every collection of the store and prints the sha256 of that
# afterimage; the parent compares it with the plan's (D5 for rebuilt stores, in the session).
#
# SC4C-RV-1: the child never outlives the worker as a writer. On Linux the kernel kills it when the
# worker dies (PR_SET_PDEATHSIG, set before anything else); on every platform it exits before each
# write once it is no longer the worker's child (guard); and it holds the worker's admission lease
# (an inherited descriptor of the same flock), so abort, resume and plan refuse runtime_admission_busy
# while it lives, whatever killed the worker. A rebuild longer than CHROMA_REBUILD_TIMEOUT is killed.
CHROMA_REBUILD_CHILD = "\n".join([
    "import hashlib, importlib.metadata, json, os, sys",
    "if sys.platform.startswith('linux'):",
    "    import ctypes, signal",
    "    if ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:",  # PR_SET_PDEATHSIG
    "        raise SystemExit(6)",
    "spec = json.loads(sys.stdin.buffer.read())",
    "def guard():",
    "    if os.getppid() != spec['parent']:",
    "        os._exit(5)",
    "guard()",
    "if importlib.metadata.version('chromadb') != spec['sdk_version']:",
    "    raise SystemExit(3)",
    "import chromadb",
    "from chromadb.config import Settings",
    "from chromadb.api.collection_configuration import load_collection_configuration_from_json",
    "def plain(value):",
    "    if hasattr(value, 'tolist'):",
    "        value = value.tolist()",
    "    if type(value) is list:",
    "        return [plain(item) for item in value]",
    "    if type(value) is dict:",
    "        return {key: plain(item) for key, item in value.items()}",
    "    return value",
    "client = chromadb.PersistentClient(path=spec['path'], settings=Settings(anonymized_telemetry=False))",
    "try:",
    "    batch = max(1, min(int(client.get_max_batch_size()), 1000))",
    "    for item in spec['collections']:",
    "        guard()",
    "        configuration = item['configuration']",
    "        if configuration is not None:",
    "            configuration = load_collection_configuration_from_json(configuration)",
    "        collection = client.create_collection(name=item['collection'], metadata=item['metadata'] or None,",
    "                                              configuration=configuration, embedding_function=None)",
    "        rows = item['records']",
    "        for start in range(0, len(rows), batch):",
    "            part = rows[start:start + batch]",
    "            guard()",
    "            collection.add(ids=[row['id'] for row in part], documents=[row['document'] for row in part],",
    "                           metadatas=[row['metadata'] for row in part],",
    "                           embeddings=[row['embedding'] for row in part])",
    "    guard()",
    "    image = []",
    "    for name in sorted(item.name for item in client.list_collections()):",
    "        collection = client.get_collection(name=name, embedding_function=None)",
    "        payload = collection.get(include=['documents', 'metadatas', 'embeddings'])",
    "        ids = list(payload['ids'])",
    "        if collection.count() != len(ids) or len(set(ids)) != len(ids):",
    "            raise SystemExit(4)",
    "        arrays = [plain(payload[key]) if ids else [] for key in ('documents', 'metadatas', 'embeddings')]",
    "        records = sorted(({'id': identity, 'document': arrays[0][index], 'metadata': arrays[1][index],",
    "                           'embedding': arrays[2][index]} for index, identity in enumerate(ids)),",
    "                         key=lambda row: row['id'])",
    "        image.append({'collection': name, 'metadata': plain(collection.metadata),",
    "                      'configuration': plain(collection.configuration_json), 'records': records})",
    "finally:",
    "    client.close()",
    "print(hashlib.sha256(json.dumps(image, sort_keys=True, separators=(',', ':'), ensure_ascii=True,",
    "                                allow_nan=False).encode()).hexdigest())",
])
_CHILD_ENVIRONMENT = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "HOME", "TMPDIR")
# SC4C-RV-1 (c): one store's rebuild may not take longer than a whole transition's downtime budget
# (SC-D2 A, 600 s); the child is then killed and the start refuses.
CHROMA_REBUILD_TIMEOUT = 600.0


def _store_image(wanted: list) -> list[dict]:
    """What the child must read back from one store: exactly its planned collections, sorted by name."""
    return [{"collection": item.collection, "metadata": item.metadata, "configuration": item.configuration,
             "records": list(item.records)} for item in sorted(wanted, key=lambda item: item.collection)]


def _rebuild_store(path: Path, wanted: list, version: str, lease: int) -> None:
    """One store in the child; refuses (content-free) unless it read back exactly the planned image.

    ``lease`` is the worker's admission lease descriptor: the child inherits
    it, so the lease stays held until the child is gone (SC4C-RV-1).
    """
    import subprocess
    import sys
    spec = {"path": str(path), "sdk_version": version, "parent": os.getpid(),
            "collections": [{"collection": item.collection, "metadata": item.metadata,
                             "configuration": item.configuration, "records": list(item.records)}
                            for item in wanted]}
    environment = {key: os.environ[key] for key in _CHILD_ENVIRONMENT if key in os.environ}
    try:
        completed = subprocess.run([sys.executable, "-I", "-B", "-c", CHROMA_REBUILD_CHILD], input=_json(spec),
                                   capture_output=True, env=environment, check=False, pass_fds=(lease,),
                                   timeout=CHROMA_REBUILD_TIMEOUT)
    except subprocess.TimeoutExpired:  # run() has killed and reaped the child
        raise CompactionCloneError("runtime_admission_compaction_chroma_rebuild_timeout") from None
    _require(completed.returncode != 3, "sdk_changed")
    _require(completed.returncode == 0, "chroma_rebuild_failed")  # stderr is never read back (content)
    lines = completed.stdout.decode("ascii", "replace").split()
    _require(lines[-1:] == [hashlib.sha256(_json(_store_image(wanted))).hexdigest()], "payload_changed:chroma")


def rebuild_chroma(plan: CompactionClonePlan, destination: Path, operation, lease: int | None = None) -> dict:
    """SC-4c: every Chroma store of H made anew in K from the plan; returns the stores' file pins.

    A store whose collections are all dropped is still created (empty), so
    K's schemas equal H's (A1). Each store is written by its own child
    process (CHROMA_REBUILD_CHILD) inside an extent the copy catalog holds;
    the child's exit is the SDK's close, and only after it the store's
    files are pinned. ``lease`` is the worker's admission lease descriptor,
    which each child holds while it lives (SC4C-RV-1).
    """
    if not plan.chroma_stores:
        return {}
    _require(type(lease) is int and lease >= 0, "lease")
    version = plan.latest.source_binding["sdk_versions"].get("chromadb")
    _require(version is not None and importlib.metadata.version("chromadb") == version, "sdk_changed")
    pins = {}
    for store in plan.chroma_stores:
        path = destination / store
        _require(not path.exists() and not path.is_symlink(), "destination_database")
        operation.extent(store, family="semantic_index", representation="sdk_work")
        path.mkdir(parents=True, mode=0o700)
        _rebuild_store(path, [item for item in plan.chroma if item.store == store and item.kind != "dropped"],
                       version, lease)
        pins[store] = _store_pin(destination, store)
    return pins


def _chroma_summary(plan: CompactionClonePlan) -> dict:
    """Content-free counts of the closed list's outcome."""
    counts = {"stores": len(plan.chroma_stores), **{kind: 0 for kind in CHROMA_KINDS},
              "records": 0, "excluded": 0}
    for item in plan.chroma:
        counts[item.kind] += 1
        counts["records"] += len(item.records)
        counts["excluded"] += item.excluded
    return counts


def _relocate_columns(plan: CompactionClonePlan, destination: Path) -> dict:
    """Rewrite evidence_json and the path columns in K (wake clip targets are the journal's)."""
    stage = _stage()
    counts = {"evidence_json": 0, "audio_path": 0}
    counts.update({name + "_path": 0 for name in sorted(QUEUE_PATH_TABLES)})
    for relative, database in sorted(plan.latest.manifest["databases"].items()):
        tables = database["tables"]
        if "suggestions_policy_v1" not in tables and not any(name in tables for name, _column in PATH_COLUMNS):
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
            for name in sorted(QUEUE_PATH_TABLES & set(tables)):
                # SC-1b: rowid, since the path itself is the key being rewritten.
                for rowid, value in db.execute("SELECT rowid,path FROM " + _quote(name)).fetchall():
                    new = _relocated_value(value, plan.source_roots, destination)
                    if new != value:
                        db.execute("UPDATE " + _quote(name) + " SET path=? WHERE rowid=?", (new, rowid))
                        counts[name + "_path"] += 1
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
    if name in QUEUE_PATH_TABLES:
        return [dict(row, path=_relocated_value(row["path"], plan.source_roots, plan.destination)) for row in rows]
    return rows


def _row_digest(values: list) -> bytes:
    return hashlib.sha256(_json(values)).digest()


def _destination_digests(db: sqlite3.Connection, name: str) -> tuple[tuple[str, ...], list[bytes]]:
    """K's rows of one table as sorted per-row SHA-256 digests (BLOBs as _token), and its columns.

    Each digest covers the canonical JSON of the row's typed values in sorted
    column-name order, so equal digest lists mean equal row multisets up to
    SHA-256 collisions (SC-0b O4; the old projection compared the same JSON).
    """
    from src.memory.state_journal_derivation import _token
    cursor = db.execute("SELECT * FROM " + _quote(name))
    names = [item[0] for item in cursor.description]
    order = sorted(range(len(names)), key=names.__getitem__)
    digests = [_row_digest([_token(row[index]) for index in order]) for row in cursor]
    digests.sort()
    return tuple(names[index] for index in order), digests


def _expected_digests(plan: CompactionClonePlan, rows: list[dict], columns: tuple[str, ...]) -> list[bytes] | None:
    """The snapshot side of one table in the same digest form; None if a row's columns differ from K's."""
    from src.memory.state_journal_derivation import _token
    wanted = set(columns)
    digests = []
    for row in rows:
        if row.keys() != wanted:
            return None
        digests.append(_row_digest([_token(read_blob(plan.latest, row[key]))
                                    if type(row[key]) is dict and "$blob" in row[key] else row[key]
                                    for key in columns]))
    digests.sort()
    return digests


def verify_identity(plan: CompactionClonePlan, destination: Path, journal, chroma_pins: dict | None = None) -> dict:
    """D5: K equals H exactly except the relocated columns and the rebuilt scheduling tables.

    Per database and table, row sets compare as typed projections (BLOBs by
    sha256), as sorted per-row SHA-256 digests of their canonical JSON (SC-0b
    O4); per file, relative path and sha256; SDK-opaque stores byte for
    byte (stricter than record ids). Extra files in K beyond the derivation
    records and SQLite side files refuse. Returns content-free counts.
    SC-4c: H's Chroma stores are rebuilt, not cloned: each must match the
    file pin rebuild_chroma took after the child's afterimage check, exactly.
    """
    from src.memory.payload_derivation import _privacy_schema_added
    from src.memory.state_journal_derivation import _token
    counts = {"databases": 0, "tables": 0, "rows": 0, "files": 0}
    manifest = plan.latest.manifest
    expected_files = set()
    stores = plan.chroma_stores
    _require(set(chroma_pins or {}) == set(stores), "payload_changed:chroma")
    for store in stores:
        _require(_store_pin(destination, store) == chroma_pins[store], "payload_changed:chroma")
    for key, record in sorted(manifest["files"].items()):
        if not key.startswith("state/"):
            continue
        relative = key.removeprefix("state/")
        if relative in DERIVATION_FILES or _in_store(relative, stores):
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
            schema = [{"type": row[0], "name": row[1], "tbl_name": row[2], "sql": row[3]} for row in db.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
            names = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            source_schema = database["schema"]
            added = set()
            if schema != source_schema:
                groups = None if relative == plan.privacy_relative else _authority_groups()
                _require(_privacy_schema_added(source_schema, schema, groups), "payload_changed:schema")
                added = {row["name"] for row in schema if row["type"] == "table"} - set(database["tables"])
            _require(set(names) == set(database["tables"]) | added, "payload_changed:tables")
            for name in names:
                if name in added:
                    _require(db.execute("SELECT 1 FROM " + _quote(name) + " LIMIT 1").fetchone() is None,
                             "payload_changed:added_rows")
                    continue
                if name in SCHEDULING_TABLES:
                    continue
                if name == "sqlite_sequence":
                    # The scheduling tables' AUTOINCREMENT counters move with their rebuild.
                    db.row_factory = sqlite3.Row
                    kept = sorted(({key: _token(row[key]) for key in row.keys()}
                                   for row in db.execute("SELECT * FROM sqlite_sequence")
                                   if row["name"] not in SCHEDULING_TABLES), key=_json)
                    db.row_factory = None
                    wanted = sorted(({key: _token(read_blob(plan.latest, value))
                                      if type(value) is dict and "$blob" in value else value
                                      for key, value in row.items()}
                                     for row in database["tables"][name]["rows"]
                                     if row["name"] not in SCHEDULING_TABLES), key=_json)
                    _require(kept == wanted, "payload_changed:sqlite_sequence")
                    counts["tables"] += 1
                    continue
                columns, actual = _destination_digests(db, name)
                rows = _expected_rows(plan, name, [dict(row) for row in database["tables"][name]["rows"]], journal)
                wanted = _expected_digests(plan, rows, columns)
                _require(wanted is not None and actual == wanted, "payload_changed:" + name)
                counts["tables"] += 1
                counts["rows"] += len(wanted)
            if relative == plan.privacy_relative:
                ids = sorted(row[0] for row in db.execute("SELECT id FROM erasure_targets"))
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
        if _in_store(relative, stores):
            continue  # pinned exactly above
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
    chroma: dict = field(default_factory=dict)


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
    from src.memory.payload_derivation import _atomic_file, _prepared_ownership
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
        _clone_state(plan, destination)  # every latest state file except the marker (K24) and Chroma (SC-4c)
        record = destination / ".kza-payload-preparation.json"
        _atomic_file(record, _json({"schema_version": 1, "plan_sha256": plan.plan_sha256, "phase": "preparing",
                                    "serving": False, "mode": "compaction_identity"}) + b"\n")
        journal = qualify_and_register_derived_journal(
            controller, plan.latest, destination, privacy_relative=plan.privacy_relative,
            target_relocations=plan.target_relocations, ordinary_snapshot=None,
            payload_cut_proof=_identity_proof(plan, destination), keep_blocked=True)
        ownership, blocked, qualified = _prepared_ownership(destination, journal)
        relocated = _relocate_columns(plan, destination)
        pins = rebuild_chroma(plan, destination, operation, getattr(controller, "_descriptor", None))  # SC-4c
        counts = verify_identity(plan, destination, journal, pins)
        if before_vacuum is not None:
            before_vacuum()
        vacuumed = tuple(vacuum_compacted_state(plan, destination, scope))
        counts = verify_identity(plan, destination, journal, pins)  # D5 again after VACUUM (v4 F3)
        _require(ownership.read_bytes() == blocked, "preparation_changed")
        evidence = {"schema_version": 1, "plan_sha256": plan.plan_sha256, "phase": "materialized",
                    "serving": False, "mode": "compaction_identity", "journal": journal.to_dict(),
                    "relocated": relocated, "counts": counts, "vacuumed": list(vacuumed),
                    "chroma": {"pins": pins, "counts": _chroma_summary(plan)},
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
                           journal.history_sha256, counts, vacuumed, record, _chroma_summary(plan))


# --- the worker's phases C2b (snapshot) and C3 (materialize, D4, D5) --------------------------------

def _load_snapshot(path: Path, binding_id: str, manifest_sha256: str | None = None) -> ClosedStateSnapshot:
    manifest = json.loads((path / "manifest.json").read_bytes())
    _require(type(manifest) is dict and manifest.get("source_binding_id") == binding_id, "snapshot_changed")
    snapshot = ClosedStateSnapshot(path, binding_id, manifest["source_binding"], manifest)
    verify_snapshot(snapshot)
    _require(manifest_sha256 is None or snapshot.manifest_sha256 == manifest_sha256, "snapshot_changed")
    return snapshot


def free_space_need(snapshot: ClosedStateSnapshot, *, temp_on_root: bool, pending_bytes: int = 0) -> dict:
    """P9, worker side: what K still needs once the snapshot (and its real manifest) is on disk.

    K's state is the snapshot's state files (S); VACUUM of K's largest database (L) needs its
    rollback journal beside it and its temporary database in SQLite's temp directory. K's runtime
    already exists ('k_runtime' precedes the snapshot). ``pending_bytes`` is what the snapshot
    has still to write (its manifest, when checked just before it is written).
    """
    return _files_need(snapshot.manifest["files"], temp_on_root=temp_on_root, pending_bytes=pending_bytes)


def _files_need(files: Mapping, *, temp_on_root: bool, pending_bytes: int = 0) -> dict:
    state_files = [record for key, record in files.items() if key.startswith("state/")]
    state = sum(record["size"] for record in state_files)
    largest = max((record["size"] for record in state_files if record["kind"] == "sqlite_backup"), default=0)
    root = pending_bytes + state + largest + FREE_MARGIN
    if temp_on_root:
        return {"root": root + largest, "temp": 0}
    return {"root": root, "temp": largest + FREE_MARGIN}


def _device(path: Path) -> int:
    return os.stat(path).st_dev


def _sqlite_temp_directory():
    return _stage().sqlite_temp_directory()  # one search order for the controller and the worker


def _check_free_space(snapshot: ClosedStateSnapshot, destination: Path) -> None:
    """P9 again with the manifest's real size: refuses before K is written (resumable after freeing space)."""
    _check_free_files(snapshot.manifest["files"], destination)


def _check_free_files(files: Mapping, destination: Path, pending_bytes: int = 0) -> None:
    anchor = destination.parent
    temp = _sqlite_temp_directory()
    temp_on_root = temp is None or _device(temp) == _device(anchor)
    need = _files_need(files, temp_on_root=temp_on_root, pending_bytes=pending_bytes)
    _require(shutil.disk_usage(anchor).free >= need["root"]
             and (temp_on_root or shutil.disk_usage(temp).free >= need["temp"]), "capacity")


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
    # SC-0b O1: H does not change under the lease, so its repeated authority
    # captures reuse one content-addressed result (every input byte is still hashed).
    with stage.authority_memo(), stage.AdmissionController(root, gate="compaction:" + compaction_id) as controller:
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
            # P9 with the exact manifest size before it is written (SC-0b review F4): the snapshot's
            # backups are on disk, and the manifest plus K's state and VACUUM must still fit.
            k_state = Path(intent["k_state_root"])
            snapshot = snapshot_closed_state(
                controller, intent["source_binding_id"], path,
                before_manifest=lambda size, files: _check_free_files(files, k_state, size),
                chroma_collections=_snapshot_class)
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
        _check_free_space(snapshot, destination)  # P9 with the real manifest, before K's first byte
        now = float(wall())
        plan = plan_compaction_clone(snapshot, now=now, destination=destination,
                                     privacy_relative=intent["privacy_relative"], source_roots=roots)
        _require(plan.latest.source_binding == binding, "head_moved")
        log = stage.append_compaction_event(root, log, "materializing", {"plan_sha256": plan.plan_sha256}, at=now)
        clone = materialize_compaction_clone(plan, destination, controller=controller, scope=scope,
                                             before_vacuum=before_vacuum)
        data = {"plan_sha256": clone.plan_sha256, "evidence_map": clone.evidence_map,
                "history_sha256": clone.history_sha256, "counts": clone.counts, "vacuumed": list(clone.vacuumed),
                "chroma": clone.chroma, "state_identity": stage._identity(destination)}
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
    chroma = evidence.get("chroma")
    _require(type(chroma) is dict and type(chroma.get("pins")) is dict, "resume_abort_required")
    counts = verify_identity(plan, destination, receipt, chroma["pins"])
    data = {"plan_sha256": plan.plan_sha256, "evidence_map": dict(plan.evidence_map),
            "history_sha256": receipt.history_sha256, "counts": counts, "vacuumed": evidence["vacuumed"],
            "chroma": _chroma_summary(plan), "state_identity": stage._identity(destination)}
    stage.append_compaction_event(root, log, "materialized", data, at=wall())
    return {"phase": "materialized", "counts": counts, "vacuumed": evidence["vacuumed"], "resumed": True}
