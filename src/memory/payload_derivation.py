"""Explicit, non-serving ordinary-data cuts over a closed latest authority.

No authority table is merged from the older image.  Planning reads only pinned
snapshot bytes; SDK clients and destination writes belong to materialization.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile
from typing import Mapping

from src.memory.fact_ledger import MemorySource
from src.memory.manual_components import collector_manual_closure
from src.memory.state_snapshot import ClosedStateSnapshot, read_blob, verify_snapshot


FAMILIES = frozenset({"semantic_index", "ambient_capture", "collectors", "contexts",
    "events", "suggestions", "diagnostics", "voices", "streams"})
PAYLOAD_TABLES = frozenset({"facts", "fact_sources", "fact_dependencies", "utterances",
    "capture_conversations", "wake_clips_v1", "stt_corrections_v1", "conversation_turns_v1",
    "conversation_export_parents_v1", "events_policy_v1", "suggestions_policy_v1",
    "metrics", "lineage", "voice_fingerprints", "voice_fingerprint_sources",
    "voice_capture_reviews", "voice_groups", "voice_group_members", "voice_group_lineage"})
_EXTENSIONS = {"wake_clips": ".wav", "stt_corrections": ".wav",
    "conversations": ".json", "conversation_exports": ".jsonl"}
_INCOMPLETE = {}


class PayloadDerivationError(ValueError):
    """A content-free rejection of an unsupported or ambiguous data cut."""


def _require(value, code):
    if not value:
        raise PayloadDerivationError(code)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _relative(value):
    _require(type(value) is str and bool(value), "payload_relative_path_required")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and not {"..", "."} & set(path.parts)
        and path.as_posix() == value, "payload_relative_path_required")
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class DatabaseRelocation:
    ordinary_relative: str
    latest_relative: str
    destination_relative: str


@dataclass(frozen=True)
class FileRootRelocation:
    family: str
    ordinary_relative: str
    latest_relative: str
    destination_relative: str


@dataclass(frozen=True)
class IndexRelocation:
    ordinary_relative: str
    latest_relative: str
    destination_relative: str
    collection: str


@dataclass(frozen=True)
class DerivationLayout:
    destination_root: Path
    privacy_relative: str
    database_relocations: tuple[DatabaseRelocation, ...]
    file_roots: tuple[FileRootRelocation, ...]
    index_relocations: tuple[IndexRelocation, ...]
    target_relocations: Mapping[str, dict]
    config_path_keys: tuple = ()


@dataclass(frozen=True)
class RowImage:
    source: str
    values: dict


@dataclass(frozen=True)
class TableCut:
    database: str
    table: str
    primary_key: tuple[str, ...]
    latest_rows: tuple[dict, ...]
    selected_rows: tuple[RowImage, ...]


@dataclass(frozen=True)
class FileCut:
    destination_relative: str
    source: str | None
    snapshot_relative: str | None


@dataclass(frozen=True)
class IndexCut:
    destination_relative: str
    collection: str
    sdk_version: str | None
    present: bool
    latest_records: tuple[dict, ...]
    selected_records: tuple[dict, ...]
    metadata: dict | None = None


@dataclass(frozen=True)
class PayloadCutPlan:
    ordinary: ClosedStateSnapshot = field(repr=False)
    latest: ClosedStateSnapshot = field(repr=False)
    now: float
    layout: DerivationLayout
    tables: tuple[TableCut, ...]
    files: tuple[FileCut, ...]
    indexes: tuple[IndexCut, ...]
    coverage: tuple[str, ...]
    source_manifest_hashes: tuple[str, str]
    plan_sha256: str


@dataclass(frozen=True)
class DerivedState:
    destination_root: Path
    plan_sha256: str
    journal_history_sha256: str
    target_id_map: dict
    event_target_id_map: dict
    event_dependency_target_map: dict
    relocation_audit: tuple[dict, ...]
    preparation_record: Path
    serving: bool = False


def _root(snapshot):
    root = Path(snapshot.source_binding["state_root"])
    _require(root.is_absolute() and ".." not in root.parts, "payload_source_root_invalid")
    return root


def _read_file(snapshot, relative):
    relative = _relative(relative)
    record = snapshot.manifest["files"].get(relative)
    _require(type(record) is dict, "payload_snapshot_file_missing")
    path = snapshot.snapshot_root / relative
    _require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(),
        "payload_snapshot_file_redirected")
    raw = path.read_bytes()
    _require(len(raw) == record["size"] and hashlib.sha256(raw).hexdigest() == record["sha256"],
        "payload_snapshot_file_changed")
    return raw


def _table(database, name):
    table = database["tables"].get(name)
    _require(type(table) is dict and set(table) == {"columns", "primary_key", "rows"},
        "payload_table_schema_invalid")
    columns = [item["name"] for item in table["columns"]]
    _require(len(columns) == len(set(columns)) and bool(table["primary_key"])
        and set(table["primary_key"]) <= set(columns), "payload_table_schema_invalid")
    keyed = {}
    for row in table["rows"]:
        _require(type(row) is dict and set(row) == set(columns), "payload_row_columns_invalid")
        for value in row.values():
            if type(value) is dict:
                _require(set(value) == {"$blob"} and type(value["$blob"]) is dict
                    and set(value["$blob"]) == {"path", "sha256", "size"}
                    and type(value["$blob"]["size"]) is int and value["$blob"]["size"] >= 0,
                    "payload_sql_value_invalid")
            else:
                _require(value is None or type(value) in (str, int) or _number(value),
                    "payload_sql_value_invalid")
        key = tuple(row[name] for name in table["primary_key"])
        _require(all(value is not None and type(value) in (str, int) for value in key)
            and key not in keyed, "payload_row_identity_invalid")
        keyed[key] = row
    return table, keyed


def _sources(encoded):
    try:
        raw = json.loads(encoded) if type(encoded) is str else encoded
        _require(type(raw) is list and 1 <= len(raw) <= 256, "payload_sources_invalid")
        result = []
        for item in raw:
            _require(type(item) is dict and {"capture_id", "conversation_id", "captured_at", "expires_at"} <= set(item)
                and set(item) <= {"capture_id", "conversation_id", "captured_at", "expires_at", "origin", "fact_dependencies"}
                and _number(item["captured_at"]) and _number(item["expires_at"]), "payload_sources_invalid")
            source = MemorySource.from_dict(item)
            _require(0 < source.expires_at - source.captured_at <= 48 * 3600,
                "payload_source_deadline_invalid")
            result.append(source)
        _require(len({source.capture_id for source in result}) == len(result), "payload_sources_invalid")
        return tuple(result)
    except (KeyError, TypeError, ValueError, OverflowError):
        raise PayloadDerivationError("payload_sources_invalid") from None


def _sha_token(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_cut_proof_shape(proof):
    fields = {"schema_version", "kind", "ordinary_binding_id", "latest_binding_id",
        "ordinary_manifest_sha256", "latest_manifest_sha256", "plan_sha256", "destination_root",
        "destination_identity", "removed_rows", "removed_files"}
    _require(type(proof) is dict and set(proof) == fields and type(proof["schema_version"]) is int
        and proof["schema_version"] == 1 and proof["kind"] == "ordinary_payload_cut",
        "payload_cut_history_schema_invalid")
    _require(all(_sha_token(proof[name]) for name in ("ordinary_binding_id", "latest_binding_id",
        "ordinary_manifest_sha256", "latest_manifest_sha256", "plan_sha256")),
        "payload_cut_history_digest_invalid")
    root, identity = proof["destination_root"], proof["destination_identity"]
    _require(type(root) is str and Path(root).is_absolute() and not root.startswith("//")
        and Path(root).as_posix() == root and ".." not in Path(root).parts,
        "payload_cut_history_destination_invalid")
    _require(type(identity) is dict and set(identity) == {"device", "inode"}
        and type(identity["device"]) is int and identity["device"] >= 0
        and type(identity["inode"]) is int and identity["inode"] > 0,
        "payload_cut_history_destination_invalid")
    _require(type(proof["removed_rows"]) is list and type(proof["removed_files"]) is list,
        "payload_cut_history_schema_invalid")
    seen = set()
    for row in proof["removed_rows"]:
        _require(type(row) is dict and set(row) == {"database", "table", "primary_key", "identity",
            "row_sha256", "family"} and type(row["table"]) is str and row["table"] in PAYLOAD_TABLES
            and row["table"] != "stt_corrections_v1" and row["family"] == _family(row["table"])
            and _sha_token(row["row_sha256"]), "payload_cut_history_row_invalid")
        _relative(row["database"])
        keys, values = row["primary_key"], row["identity"]
        _require(type(keys) is list and bool(keys) and all(type(key) is str and key for key in keys)
            and len(keys) == len(set(keys)) and type(values) is list and len(values) == len(keys)
            and all(type(value) in (str, int) for value in values), "payload_cut_history_identity_invalid")
        key = (row["database"], row["table"], _json(values))
        _require(key not in seen, "payload_cut_history_duplicate")
        seen.add(key)
    seen = set()
    for item in proof["removed_files"]:
        _require(type(item) is dict and set(item) == {"relative_path", "sha256", "family"}
            and item["family"] in {"ambient_capture", "collectors", "contexts"}
            and _sha_token(item["sha256"]), "payload_cut_history_file_invalid")
        _relative(item["relative_path"])
        _require(item["relative_path"] not in seen, "payload_cut_history_duplicate")
        seen.add(item["relative_path"])


def _protected_collector_records(tables):
    """Return the indivisible latest manual/export component, without I/O."""
    return collector_manual_closure(tables.get("stt_corrections_v1", {}).get("rows", ()),
        tables.get("conversation_turns_v1", {}).get("rows", ()),
        tables.get("conversation_export_parents_v1", {}).get("rows", ()))


def _proof_source_rows(proof, source):
    """The exclusion describes real ordinary rows of N, never authority rows."""
    collector_closures = {}
    for item in proof["removed_rows"]:
        database = source.manifest["databases"].get(item["database"])
        _require(type(database) is dict, "payload_cut_history_database_unknown")
        table, keyed = _table(database, item["table"])
        _require(table["primary_key"] == item["primary_key"], "payload_cut_history_identity_invalid")
        row = keyed.get(tuple(item["identity"]))
        _require(row is not None and _digest(row) == item["row_sha256"], "payload_ordinary_absence_ambiguous")
        tables, name = database["tables"], item["table"]
        if name == "facts":
            _require(row["state"] in {"automatic", "proposed"}, "payload_cut_history_manual_row")
        elif name in {"fact_sources", "fact_dependencies"}:
            parents = {value["id"]: value for value in tables["facts"]["rows"]}
            _require(parents[row["fact_id"]]["state"] in {"automatic", "proposed"},
                "payload_cut_history_manual_row")
        elif name == "utterances":
            _require(row["distilled"] == 0, "payload_cut_history_manual_row")
        elif name in {"wake_clips_v1", "conversation_turns_v1", "conversation_export_parents_v1"}:
            if item["database"] not in collector_closures:
                collector_closures[item["database"]] = _protected_collector_records(tables)
            protected = collector_closures[item["database"]]
            identity = row["export_id"] if name == "conversation_export_parents_v1" else row["id"]
            _require(identity not in protected, "payload_cut_history_manual_row")
        elif name == "suggestions_policy_v1":
            _require(row["status"] == "pending" and row["snooze_until"] is None,
                "payload_cut_history_manual_row")
        elif name == "events_policy_v1":
            for other in source.manifest["databases"].values():
                for suggestion in other["tables"].get("suggestions_policy_v1", {}).get("rows", ()):
                    if suggestion["status"] != "pending" or suggestion["snooze_until"] is not None:
                        _require(all(event["event_id"] != row["id"] for event in
                            json.loads(suggestion["evidence_json"])), "payload_cut_history_manual_row")
        elif name.startswith("voice_"):
            manual = {value["id"] for value in tables["voice_groups"]["rows"] if _manual_group(value)}
            if name in {"voice_groups", "voice_group_lineage", "voice_group_members"}:
                identity = row["id"] if name == "voice_groups" else row["group_id"]
                _require(identity not in manual, "payload_cut_history_manual_row")
            else:
                identity = row["id"] if name == "voice_fingerprints" else row["fingerprint_id"]
                _require(not any(value["fingerprint_id"] == identity and value["group_id"] in manual
                    for value in tables["voice_group_members"]["rows"]), "payload_cut_history_manual_row")
    for item in proof["removed_files"]:
        record = source.manifest["files"].get("state/" + item["relative_path"])
        _require(type(record) is dict and record["kind"] == "file" and record["sha256"] == item["sha256"],
            "payload_cut_history_file_invalid")


def _qualified_exclusions(ordinary, latest):
    archives = latest.manifest.get("history_archives", {})
    if not archives:
        return {}
    from src.memory.state_snapshot import read_history_archive
    documents = [read_history_archive(latest, sha) for sha in sorted(archives)]
    result = {}
    for archive in documents:
        proof = archive.get("payload_cut_proof")
        if proof is None:
            continue
        _validate_cut_proof_shape(proof)
        record = archive.get("source_binding_record")
        _require(type(record) is dict and _digest(record) == proof["latest_binding_id"]
            and archive.get("source_binding") == proof["latest_binding_id"]
            and archive.get("snapshot_manifest_sha256") == proof["latest_manifest_sha256"]
            and archive.get("current_execution_grant") is False,
            "payload_cut_history_source_mismatch")
        older = [item for item in documents if item.get("source_binding") == proof["ordinary_binding_id"]
            and item.get("snapshot_manifest_sha256") == proof["ordinary_manifest_sha256"]
            and type(item.get("source_binding_record")) is dict
            and _digest(item["source_binding_record"]) == proof["ordinary_binding_id"]]
        _require(bool(older), "payload_cut_history_ordinary_source_missing")
        # A previous cut only explains exclusions of its *latest* image. The
        # binding of the present snapshot pins this archive through ownership.
        if proof["latest_binding_id"] != ordinary.source_binding_id:
            continue
        _require(proof["latest_manifest_sha256"] == ordinary.manifest_sha256
            and record == ordinary.source_binding, "payload_cut_history_source_mismatch")
        _require(proof["destination_root"] == latest.source_binding["state_root"]
            and proof["destination_identity"] == latest.source_binding["state_identity"],
            "payload_cut_history_destination_mismatch")
        _proof_source_rows(proof, ordinary)
        for row in proof["removed_rows"]:
            key = (_relative(row["database"]), row["table"], row["row_sha256"])
            _require(key not in result or result[key] == row, "payload_cut_history_conflict")
            result[key] = row
        for item in proof["removed_files"]:
            key = ("file", _relative(item["relative_path"]), item["sha256"])
            _require(key not in result or result[key] == item, "payload_cut_history_conflict")
            result[key] = item
    return result


class _Policy:
    def __init__(self, ordinary, latest, privacy_old, privacy_new, now):
        self.now = now
        self.old, self.new = privacy_old["tables"], privacy_new["tables"]
        self.facts = {row["id"]: row for row in self.new["facts"]["rows"]}
        self.revoked = {row["conversation_id"] for row in self.new["conversation_tombstones"]["rows"]}
        self.sources = {}
        self.conversations = {}
        self.exclusions = _qualified_exclusions(ordinary, latest)
        self.database = None
        self.protected_events = set()
        for database in latest.manifest["databases"].values():
            for row in database["tables"].get("suggestions_policy_v1", {}).get("rows", ()):
                if row["status"] != "pending" or row["snooze_until"] is not None:
                    evidence = json.loads(row["evidence_json"])
                    _require(type(evidence) is list and bool(evidence), "payload_event_evidence_missing")
                    self.protected_events.update(item["event_id"] for item in evidence)
        for snapshot in (ordinary, latest):
            for database in snapshot.manifest["databases"].values():
                for table in database["tables"].values():
                    for row in table["rows"]:
                        if type(row.get("sources_json")) is str:
                            for source in _sources(row["sources_json"]):
                                key, encoded = source.capture_id, source.to_dict()
                                _require(key not in self.sources or self.sources[key] == encoded,
                                    "payload_source_identity_changed")
                                self.sources[key] = encoded
                                self.conversations.setdefault(source.conversation_id, {})[key] = source
                for row in database["tables"].get("fact_sources", {}).get("rows", ()):
                    data = {key: row[key] for key in ("capture_id", "conversation_id", "captured_at", "expires_at", "origin")}
                    data["fact_dependencies"] = json.loads(row["dependencies"])
                    source, = _sources([data])
                    key, encoded = source.capture_id, source.to_dict()
                    _require(key not in self.sources or self.sources[key] == encoded,
                        "payload_source_identity_changed")
                    self.sources[key] = encoded
                    self.conversations.setdefault(source.conversation_id, {})[key] = source

    def activity(self, sources):
        """Snapshot equivalent of FactLedger.activity_sources_available.

        This does not authorize audio or text past a capture's raw deadline.
        Only the numeric diagnostic branch pairs it with its own row expiry.
        """
        for source in sources:
            if source.conversation_id in self.revoked:
                return False
            for dependency in source.fact_dependencies:
                fact = self.facts.get(dependency.fact_id)
                if (fact is None or fact["state"] not in {"automatic", "confirmed"}
                        or fact["revision"] != dependency.revision
                        or fact["expires_at"] != dependency.expires_at
                        or dependency.expires_at is not None and dependency.expires_at <= self.now):
                    return False
        return True

    def live(self, sources):
        return all(source.captured_at <= self.now < source.expires_at for source in sources) and self.activity(sources)

    def row_live(self, row, *, sources=None):
        if sources is None:
            sources = _sources(row["sources_json"])
        if not self.live(sources):
            return False
        expiry = row.get("expires_at", row.get("expires"))
        _require(expiry is None or _number(expiry), "payload_deadline_invalid")
        return expiry is None or self.now < expiry

    def fact_sources(self, database, identity):
        rows = [row for row in database["fact_sources"]["rows"] if row["fact_id"] == identity]
        _require(bool(rows), "payload_fact_lineage_missing")
        result = []
        for row in rows:
            source = {key: row[key] for key in ("capture_id", "conversation_id", "captured_at", "expires_at", "origin")}
            source["fact_dependencies"] = json.loads(row["dependencies"])
            result.extend(_sources([source]))
        return tuple(result)

    def absent(self, table, row):
        expected = self.exclusions.get((self.database, table, _digest(row)))
        _require(expected is not None, "payload_ordinary_absence_ambiguous")
        _require(expected.get("identity") == [row[key] for key in expected["primary_key"]],
            "payload_cut_history_identity_mismatch")
        if table == "voice_fingerprints" and row.get("suggested_profile") is not None:
            profiles = {item["id"]: item for item in self.new.get("voice_profiles", {}).get("rows", ())}
            profile = profiles.get(row["suggested_profile"])
            if profile is None or profile["deleted_at"] is not None or profile["revision"] != row["suggested_revision"]:
                return dict(row, suggested_profile=None, suggested_revision=None, similarity=None)
        return row


def _images(table, source):
    return {tuple(row[name] for name in table["primary_key"]): RowImage(source, dict(row))
        for row in table["rows"]}


def _blend(row, current, now, *, table):
    value = dict(row)
    for name in ("expires_at", "text_expires_at", "expires"):
        old, new = value.get(name), current.get(name)
        if _number(new) and (old is None or _number(old) and new < old):
            value[name] = new
    for name in ("suggested_profile", "suggested_revision", "similarity"):
        if name in value:
            value[name] = current[name]
    if table == "events_policy_v1":
        _require({"text_expires_at", "trigger_phrase", "user_name", "context_json", "metrics_json"}
            <= value.keys(), "payload_event_text_schema_invalid")
        deadline = value["text_expires_at"]
        _require(deadline is None or _number(deadline), "payload_event_text_deadline_invalid")
        if deadline is not None and deadline <= now:
            value["trigger_phrase"] = None
            value["user_name"] = None
            value["context_json"] = value["metrics_json"]
    return value


def _ordinary(old, new, predicate, *, protected=(), now, policy, table):
    old_table, older = _table({"tables": {"x": old}}, "x")
    _new_table, current = _table({"tables": {"x": new}}, "x")
    _require(old_table["columns"] == new["columns"] and old_table["primary_key"] == new["primary_key"],
        "payload_table_schema_changed")
    result = {key: RowImage("latest", dict(current[key])) for key in protected}
    for key, row in older.items():
        if key in result or not predicate(row):
            continue
        candidate = _blend(row, current.get(key) or policy.absent(table, row), now, table=table)
        if predicate(candidate):
            result[key] = RowImage("ordinary", candidate)
    return result


def _select_database(old, new, policy):
    older, current, selected = old["tables"], new["tables"], {}
    for name in sorted(PAYLOAD_TABLES & current.keys()):
        _require(name in older, "payload_table_missing_from_ordinary")
        _table(old, name)
        _table(new, name)
        selected[name] = _images(current[name], "latest")

    def keep(name, chosen):
        selected[name] = {tuple(row.values[key] for key in current[name]["primary_key"]): row
            for row in chosen}

    def related(name, parent, foreign_key):
        chosen = []
        for image in selected[parent].values():
            tables = older if image.source == "ordinary" else current
            for row in tables[name]["rows"]:
                if row[foreign_key] != image.values["id"]:
                    continue
                if image.source == "ordinary":
                    keys = current[name]["primary_key"]
                    if not any(all(item[key] == row[key] for key in keys) for item in current[name]["rows"]):
                        policy.absent(name, row)
                chosen.append(RowImage(image.source, dict(row)))
        keep(name, chosen)

    _select_facts(older, current, selected, policy, related)
    _select_captures(older, current, selected, policy)
    _select_collectors(older, current, selected, policy, related)
    _select_events_suggestions(older, current, selected, policy)
    _select_voices(older, current, selected, policy, keep, related)
    _select_diagnostics(older, current, selected, policy, related)
    return selected


_OWNER_TABLES = ("fact_owners", "fact_owner_revisions")


def _owned_fact_ids(tables):
    """Fact ids in fact_owners ∪ fact_owner_revisions of the latest image.

    A pre-U3b image has neither table (empty set); one without the other is
    an invalid image. Present tables are validated like any other table.
    """
    present = [name for name in _OWNER_TABLES if name in tables]
    if not present:
        return set()
    _require(len(present) == len(_OWNER_TABLES), "payload_table_schema_invalid")
    owned = set()
    for name in _OWNER_TABLES:
        table, keyed = _table({"tables": tables}, name)
        _require(list(table["primary_key"]) == ["fact_id"], "payload_table_schema_invalid")
        owned |= {key[0] for key in keyed}
    return owned


def _select_facts(old, new, selected, policy, related):
    if "facts" not in new:
        return
    protected = {row["id"] for row in new["facts"]["rows"]
        if row["state"] in {"confirmed", "deleted", "quarantined"}}
    # U3b W1: a fact with an owner row or any ownership history is never cut,
    # or the latest owner rows (not payload; cloned from latest) would dangle.
    protected |= _owned_fact_ids(new)
    dependencies = new["fact_dependencies"]["rows"]
    while True:
        closure = protected | {row["source_fact_id"] for row in dependencies if row["fact_id"] in protected}
        if closure == protected:
            break
        protected = closure
    latest = {row["id"]: row for row in new["facts"]["rows"]}
    _require(protected <= latest.keys(), "payload_protected_fact_dependency_missing")

    def eligible(row):
        _require(row["state"] in {"automatic", "proposed", "confirmed", "deleted", "quarantined"},
            "payload_fact_state_unknown")
        if row["state"] not in {"automatic", "proposed"} or row["id"] in protected:
            return False
        if not policy.row_live(row, sources=policy.fact_sources(old, row["id"])):
            return False
        current = latest.get(row["id"])
        if current is None:
            current = policy.absent("facts", row)
        _require(current["revision"] == row["revision"], "payload_ordinary_fact_revision_changed")
        return current["state"] in {"automatic", "proposed"}

    selected["facts"] = _ordinary(old["facts"], new["facts"], eligible,
        protected={(identity,) for identity in protected}, now=policy.now, policy=policy, table="facts")
    related("fact_sources", "facts", "fact_id")
    related("fact_dependencies", "facts", "fact_id")


def _select_captures(old, new, selected, policy):
    if "utterances" not in new:
        return

    def eligible(row):
        source = policy.sources.get(row["capture_id"])
        _require(source is not None and source["conversation_id"] == row["conversation_id"],
            "payload_capture_lineage_missing")
        _require(row["expires_at"] <= source["expires_at"], "payload_deadline_renewal")
        return policy.row_live(row, sources=_sources([source]))

    protected = {(row["id"],) for row in new["utterances"]["rows"] if row["distilled"] != 0}
    for row in new["utterances"]["rows"]:
        if (row["id"],) in protected and row["expires_at"] > policy.now:
            _require(eligible(row), "payload_protected_capture_requires_cleanup")
    selected["utterances"] = _ordinary(old["utterances"], new["utterances"], eligible,
        protected=protected, now=policy.now, policy=policy, table="utterances")
    conversations = {image.values["conversation_id"] for image in selected["utterances"].values()}
    parents = {}
    for source, tables in (("ordinary", old), ("latest", new)):
        for row in tables["capture_conversations"]["rows"]:
            if row["id"] in conversations:
                parents[(row["id"],)] = RowImage(source, dict(row))
    selected["capture_conversations"] = parents


def _select_collectors(old, new, selected, policy, related):
    if "wake_clips_v1" not in new:
        return
    protected = _protected_collector_records(new)
    current = {row["id"]: row for row in new["wake_clips_v1"]["rows"]}
    _require(protected <= current.keys(), "payload_manual_clip_closure_missing")
    for identity in protected:
        if current[identity]["expires_at"] > policy.now:
            _require(policy.row_live(current[identity]), "payload_protected_clip_requires_cleanup")
    selected["wake_clips_v1"] = _ordinary(old["wake_clips_v1"], new["wake_clips_v1"],
        lambda row: row["bucket"] != "stt_correction" and bool(row["published"]) and policy.row_live(row),
        protected={tuple(row[key] for key in new["wake_clips_v1"]["primary_key"])
            for row in new["wake_clips_v1"]["rows"] if row["id"] in protected},
        now=policy.now, policy=policy, table="wake_clips_v1")
    for name in ("stt_corrections_v1", "conversation_turns_v1"):
        if name in new:
            related(name, "wake_clips_v1", "id")
    if "conversation_export_parents_v1" in new:
        related("conversation_export_parents_v1", "wake_clips_v1", "export_id")
        chosen = {image.values["id"] for image in selected["wake_clips_v1"].values()}
        _require(all(image.values["parent_id"] in chosen for image in
            selected["conversation_export_parents_v1"].values()), "payload_export_parent_missing")


def _select_events_suggestions(old, new, selected, policy):
    if "events_policy_v1" in new:
        protected = {(row["id"],) for row in new["events_policy_v1"]["rows"]
            if row["id"] in policy.protected_events}
        _require(all(policy.row_live(row) for row in new["events_policy_v1"]["rows"]
            if (row["id"],) in protected), "payload_protected_event_requires_cleanup")
        selected["events_policy_v1"] = _ordinary(old["events_policy_v1"],
            new["events_policy_v1"], policy.row_live, protected=protected, now=policy.now,
            policy=policy, table="events_policy_v1")
    if "suggestions_policy_v1" in new:
        protected = {tuple(row[name] for name in new["suggestions_policy_v1"]["primary_key"])
            for row in new["suggestions_policy_v1"]["rows"]
            if row["status"] != "pending" or row["snooze_until"] is not None}
        _require(all(policy.row_live(row) for row in new["suggestions_policy_v1"]["rows"]
            if tuple(row[name] for name in new["suggestions_policy_v1"]["primary_key"]) in protected),
            "payload_protected_suggestion_requires_cleanup")
        selected["suggestions_policy_v1"] = _ordinary(old["suggestions_policy_v1"],
            new["suggestions_policy_v1"], lambda row: row["status"] == "pending"
                and row["snooze_until"] is None and policy.row_live(row),
            protected=protected, now=policy.now, policy=policy, table="suggestions_policy_v1")


def _manual_group(row):
    return row["state"] == "manual" or type(row["reason"]) is str and row["reason"].startswith("manual")


def _select_voices(old, new, selected, policy, keep, related):
    if "voice_fingerprints" not in new:
        return
    groups = {row["id"] for row in new["voice_groups"]["rows"] if _manual_group(row)}
    manual_members = {row["fingerprint_id"] for row in new["voice_group_members"]["rows"]
        if row["group_id"] in groups}
    for row in new["voice_fingerprints"]["rows"]:
        if row["id"] in manual_members:
            _require(policy.row_live(row), "payload_protected_voice_requires_cleanup")
    selected["voice_fingerprints"] = _ordinary(old["voice_fingerprints"], new["voice_fingerprints"],
        policy.row_live, protected={(identity,) for identity in manual_members}, now=policy.now,
        policy=policy, table="voice_fingerprints")
    for name in ("voice_fingerprint_sources", "voice_capture_reviews"):
        related(name, "voice_fingerprints", "fingerprint_id")
    fingerprints = {image.values["id"]: image.values for image in selected["voice_fingerprints"].values()}
    chosen_groups, members, lineage, assigned, chosen_ids = [], [], [], set(), set()
    candidates = [("latest", new, row) for row in new["voice_groups"]["rows"] if _manual_group(row)]
    candidates += [("ordinary", old, row) for row in old["voice_groups"]["rows"] if not _manual_group(row)]
    candidates += [("latest", new, row) for row in new["voice_groups"]["rows"] if not _manual_group(row)]
    for source, tables, group in candidates:
        if group["id"] in chosen_ids:
            continue
        linked = [row for row in tables["voice_group_members"]["rows"] if row["group_id"] == group["id"]]
        identities = {row["fingerprint_id"] for row in linked}
        if source == "latest" and _manual_group(group) and not identities:
            chosen_groups.append(RowImage(source, dict(group)))
            lineage.extend(RowImage(source, dict(row)) for row in tables["voice_group_lineage"]["rows"]
                if row["group_id"] == group["id"])
            chosen_ids.add(group["id"])
            continue
        if not identities or identities & assigned or not identities <= fingerprints.keys():
            continue
        if source == "ordinary":
            for name, rows in (("voice_groups", [group]), ("voice_group_members", linked)):
                keys = new[name]["primary_key"]
                present = {tuple(item[key] for key in keys) for item in new[name]["rows"]}
                for row in rows:
                    if tuple(row[key] for key in keys) not in present:
                        policy.absent(name, row)
        original = {row["id"]: row for row in tables["voice_fingerprints"]["rows"]}
        ignored = {"suggested_profile", "suggested_revision", "similarity"}
        if any({k: v for k, v in fingerprints[identity].items() if k not in ignored}
                != {k: v for k, v in original[identity].items() if k not in ignored} for identity in identities):
            continue
        proofs = [row for row in tables["voice_group_lineage"]["rows"] if row["group_id"] == group["id"]]
        _require(bool(proofs), "payload_group_lineage_missing")
        if source == "ordinary":
            keys = new["voice_group_lineage"]["primary_key"]
            present = {tuple(item[key] for key in keys) for item in new["voice_group_lineage"]["rows"]}
            for row in proofs:
                if tuple(row[key] for key in keys) not in present:
                    policy.absent("voice_group_lineage", row)
        chosen_groups.append(RowImage(source, dict(group)))
        chosen_ids.add(group["id"])
        members.extend(RowImage(source, dict(row)) for row in linked)
        lineage.extend(RowImage(source, dict(row)) for row in proofs)
        assigned |= identities
    _require(assigned == fingerprints.keys(), "payload_group_membership_incomplete")
    keep("voice_groups", chosen_groups)
    keep("voice_group_members", members)
    keep("voice_group_lineage", lineage)


def _select_diagnostics(old, new, selected, policy, related):
    if "metrics" not in new:
        return
    by_id = {}
    for row in old["lineage"]["rows"]:
        by_id.setdefault(row["record_id"], set()).add(row["conversation_id"])

    def eligible(row):
        from src.obs.metrics_emitter import project_diagnostic
        body = json.loads(row["payload"])
        _require(project_diagnostic(body) == body and type(row["partition"]) is int and row["partition"] in (0, 1)
            and _number(row["born"]) and _number(row["expires"])
            and row["born"] < row["expires"] and row["born"] <= policy.now,
            "payload_diagnostic_record_invalid")
        conversations = by_id.get(row["id"], set())
        _require(bool(conversations), "payload_diagnostic_lineage_missing")
        _require(all(identity in policy.conversations for identity in conversations),
            "payload_diagnostic_source_unknown")
        sources = tuple(source for identity in conversations for source in policy.conversations[identity].values())
        # Numeric metadata has the store's anchored retention, commonly 90d.
        # Neither reuse the raw 48h cap nor renew born/expires at derivation.
        return policy.now < row["expires"] and policy.activity(sources)

    selected["metrics"] = _ordinary(old["metrics"], new["metrics"], eligible, now=policy.now,
        policy=policy, table="metrics")
    related("lineage", "metrics", "record_id")


def _validate_layout(layout, ordinary, latest):
    _require(type(layout) is DerivationLayout and isinstance(layout.destination_root, Path),
        "payload_layout_required")
    destination = layout.destination_root
    _require(destination.is_absolute() and ".." not in destination.parts
        and destination not in {_root(ordinary), _root(latest)}, "payload_destination_invalid")
    _relative(layout.privacy_relative)
    _require(not layout.config_path_keys, "payload_configuration_relocation_unsupported")
    destinations = set()
    for item in layout.database_relocations:
        _require(type(item) is DatabaseRelocation, "payload_database_layout_invalid")
        for name in (item.ordinary_relative, item.latest_relative, item.destination_relative):
            _relative(name)
        _require(item.destination_relative == item.latest_relative
            and item.destination_relative not in destinations, "payload_database_layout_invalid")
        destinations.add(item.destination_relative)
    _require(layout.privacy_relative in destinations, "payload_privacy_layout_missing")
    privacy = next(item for item in layout.database_relocations
        if item.destination_relative == layout.privacy_relative)
    _require(privacy.ordinary_relative == privacy.latest_relative == layout.privacy_relative,
        "payload_privacy_layout_unsupported")
    for item in layout.file_roots:
        _require(type(item) is FileRootRelocation and item.family in {
            "ambient_capture", "collectors", "contexts", "diagnostics"}, "payload_file_layout_invalid")
        for name in (item.ordinary_relative, item.latest_relative, item.destination_relative):
            _relative(name)
        _require(item.destination_relative == item.latest_relative, "payload_file_layout_invalid")
    _require({item.family for item in layout.file_roots} >= {
        "ambient_capture", "collectors", "contexts", "diagnostics"}, "payload_family_layout_missing")
    for item in layout.index_relocations:
        _require(type(item) is IndexRelocation and type(item.collection) is str,
            "payload_index_layout_invalid")
        for name in (item.ordinary_relative, item.latest_relative, item.destination_relative):
            _relative(name)
        _require(item.destination_relative == item.latest_relative, "payload_index_layout_invalid")
    _require(bool(layout.index_relocations), "payload_index_layout_missing")


def _targets(snapshot, privacy_relative):
    rows = snapshot.manifest["databases"][privacy_relative]["tables"]["erasure_targets"]["rows"]
    return {row["id"]: {**row, "descriptor": json.loads(row["descriptor"])} for row in rows}


def _map_file(layout, source, relative):
    matches = []
    path = PurePosixPath(relative)
    for entry in layout.file_roots:
        prefix = PurePosixPath(entry.ordinary_relative if source == "ordinary" else entry.latest_relative)
        if path.is_relative_to(prefix):
            matches.append((entry, (PurePosixPath(entry.destination_relative) / path.relative_to(prefix)).as_posix()))
    _require(len(matches) == 1, "payload_file_mapping_ambiguous")
    return matches[0]


def _source_file(snapshot, source, original_path, layout):
    path = Path(original_path)
    _require(path.is_absolute() and path.is_relative_to(_root(snapshot)), "payload_file_source_foreign")
    relative = path.relative_to(_root(snapshot)).as_posix()
    _entry, destination = _map_file(layout, source, relative)
    key = "state/" + relative
    record = snapshot.manifest["files"].get(key)
    _require(type(record) is dict and record["kind"] == "file", "payload_payload_file_missing")
    _read_file(snapshot, key)
    return FileCut(destination, source, key)


def _context_files(ordinary, latest, layout, policy):
    wanted, known_latest = {}, set()
    for mapping in layout.file_roots:
        if mapping.family != "contexts":
            continue
        old_prefix = "state/" + mapping.ordinary_relative + "/"
        new_prefix = "state/" + mapping.latest_relative + "/"
        known_latest.update(key.removeprefix("state/") for key in latest.manifest["files"]
            if key.startswith(new_prefix))
        for key, record in ordinary.manifest["files"].items():
            if not key.startswith(old_prefix):
                continue
            _require(record["kind"] == "file" and key.endswith(".json"), "payload_context_file_unsupported")
            value = json.loads(_read_file(ordinary, key))
            _require(type(value) is dict and type(value.get("version")) is int and value["version"] == 3
                and type(value.get("compacted_summary")) is str, "payload_context_version_unsupported")
            if not policy.live(_sources(value.get("summary_sources"))):
                continue
            new_key = new_prefix + key.removeprefix(old_prefix)
            _require(new_key in latest.manifest["files"] or
                ("file", new_key.removeprefix("state/"), record["sha256"]) in policy.exclusions,
                "payload_ordinary_absence_ambiguous")
            if new_key in latest.manifest["files"]:
                current = json.loads(_read_file(latest, new_key))
                _require(current.get("version") == value["version"], "payload_context_version_unsupported")
            original = str(_root(ordinary) / key.removeprefix("state/"))
            cut = _source_file(ordinary, "ordinary", original, layout)
            _require(cut.destination_relative not in wanted, "payload_file_collision")
            wanted[cut.destination_relative] = cut
    return wanted, known_latest


def _index_plan(ordinary, latest, layout, cuts):
    facts = {image.values["id"]: image for cut in cuts if cut.table == "facts" for image in cut.selected_rows}
    known = {row["id"] for snapshot in (ordinary, latest)
        for database in snapshot.manifest["databases"].values()
        for row in database["tables"].get("facts", {}).get("rows", ())}
    plans = []
    for source, snapshot in (("ordinary", ordinary), ("latest", latest)):
        expected = {(item.ordinary_relative if source == "ordinary" else item.latest_relative, item.collection)
            for item in layout.index_relocations}
        _require(expected == {(item["relative_path"], item["collection"])
            for item in snapshot.manifest["index_exports"]}, "payload_index_layout_incomplete")
    for mapping in layout.index_relocations:
        exports = {}
        for source, snapshot, relative in (("ordinary", ordinary, mapping.ordinary_relative),
                ("latest", latest, mapping.latest_relative)):
            found = [item for item in snapshot.manifest["index_exports"]
                if item["relative_path"] == relative and item["collection"] == mapping.collection]
            _require(len(found) == 1, "payload_index_export_missing")
            exports[source] = found[0]
        old, new = exports["ordinary"], exports["latest"]
        _require(old["sdk_version"] == new["sdk_version"] and old["metadata"] == new["metadata"]
            and old["fields"] == new["fields"] == ["ids", "documents", "metadatas", "embeddings"],
            "payload_index_schema_changed")
        indexed = {}
        dimensions = set()
        for source, export in exports.items():
            indexed[source] = {}
            for row in export["records"]:
                _require(set(row) == {"id", "document", "metadata", "embedding"}
                    and row["id"] in known and row["id"] not in indexed[source],
                    "payload_unattributed_index_record")
                vector, metadata = row["embedding"], row["metadata"]
                _require(type(vector) is list and bool(vector) and all(_number(v) for v in vector)
                    and type(row["document"]) is str and type(metadata) is dict
                    and set(metadata) == {"fact_policy", "revision"}
                    and type(metadata["fact_policy"]) is int and metadata["fact_policy"] == 1
                    and type(metadata["revision"]) is int and metadata["revision"] > 0,
                    "payload_index_record_invalid")
                dimensions.add(len(vector))
                indexed[source][row["id"]] = row
        _require(len(dimensions) <= 1, "payload_index_dimensions_changed")
        chosen = []
        for identity, image in facts.items():
            fact = image.values
            if fact["state"] not in {"automatic", "confirmed"}:
                continue
            row = indexed[image.source].get(identity)
            _require(row is not None and row["document"] == fact["content"]
                and row["metadata"]["revision"] == fact["revision"], "payload_fact_index_lineage_missing")
            chosen.append(dict(row))
        _require(bool(new["present"]) or not chosen, "payload_index_storage_missing")
        plans.append(IndexCut(mapping.destination_relative, mapping.collection, new["sdk_version"],
            new["present"], tuple(new["records"]), tuple(sorted(chosen, key=lambda row: row["id"])), new["metadata"]))
    return tuple(plans)


def _file_plan(ordinary, latest, layout, cuts, policy, target_sets):
    snapshots = {"ordinary": ordinary, "latest": latest}
    wanted, known_latest = _context_files(ordinary, latest, layout, policy)

    def add(image, locator):
        cut = _source_file(snapshots[image.source], image.source, locator, layout)
        previous = wanted.setdefault(cut.destination_relative, cut)
        _require(previous == cut, "payload_file_collision")
        return str(layout.destination_root / cut.destination_relative)

    def remember(locator):
        path = Path(locator)
        if path.is_absolute() and path.is_relative_to(_root(latest)):
            known_latest.add(path.relative_to(_root(latest)).as_posix())

    for cut in cuts:
        if cut.table == "utterances":
            for row in cut.latest_rows:
                if row["audio_path"]:
                    remember(row["audio_path"])
            for image in cut.selected_rows:
                if image.values["audio_path"] and image.values["expires_at"] > policy.now:
                    image.values["audio_path"] = add(image, image.values["audio_path"])
        elif cut.table == "wake_clips_v1":
            for row in cut.latest_rows:
                target = target_sets["latest"][row["target_id"]]["descriptor"]
                _require(target["kind"] in _EXTENSIONS, "payload_collector_kind_unsupported")
                remember(str(Path(target["root"]) / "policy_v1" / (row["id"] + _EXTENSIONS[target["kind"]])))
            for image in cut.selected_rows:
                row = image.values
                if row["expires_at"] <= policy.now or not row["published"]:
                    continue
                target = target_sets[image.source][row["target_id"]]["descriptor"]
                _require(target["kind"] in _EXTENSIONS, "payload_collector_kind_unsupported")
                locator = str(Path(target["root"]) / "policy_v1" / (row["id"] + _EXTENSIONS[target["kind"]]))
                destination = add(image, locator)
                declared = layout.target_relocations[row["target_id"]]
                _require(destination == str(Path(declared["root"]) / "policy_v1" /
                    (row["id"] + _EXTENSIONS[target["kind"]])), "payload_target_file_mapping_changed")
    for key, record in latest.manifest["files"].items():
        if not key.startswith("state/") or record["kind"] != "file":
            continue
        relative = key.removeprefix("state/")
        matches = [item for item in layout.file_roots if PurePosixPath(relative).is_relative_to(item.latest_relative)]
        if not matches:
            continue
        _require(len(matches) == 1, "payload_file_mapping_ambiguous")
        if matches[0].family == "diagnostics":
            continue
        _require(relative in known_latest, "payload_unattributed_file")
        _entry, destination = _map_file(layout, "latest", relative)
        wanted.setdefault(destination, FileCut(destination, None, None))
    return tuple(wanted[key] for key in sorted(wanted))


def _plan_value(ordinary, latest, now, layout, tables, files, indexes, coverage):
    description = {"destination_root": str(layout.destination_root), "privacy_relative": layout.privacy_relative,
        "database_relocations": [asdict(v) for v in layout.database_relocations],
        "file_roots": [asdict(v) for v in layout.file_roots],
        "index_relocations": [asdict(v) for v in layout.index_relocations],
        "target_relocations": {k: dict(v) for k, v in layout.target_relocations.items()},
        "config_path_keys": list(layout.config_path_keys)}
    return {"ordinary": ordinary.manifest_sha256, "latest": latest.manifest_sha256,
        "now": now, "layout": description, "tables": [asdict(item) for item in tables],
        "files": [asdict(item) for item in files], "indexes": [asdict(item) for item in indexes],
        "coverage": list(coverage)}


def _database_contract(snapshot, relative):
    database = snapshot.manifest["databases"][relative]
    schema = [[row[name] for name in ("type", "name", "tbl_name", "sql")] for row in database["schema"]]
    _require(_digest(schema) == snapshot.source_binding["schemas"].get(relative),
        "payload_database_schema_unbound")
    projection = snapshot.source_binding["authority"]["projection"]["tables"].get(relative, {})
    contract = projection.get("schema_contract_v1")
    _require(type(contract) is dict, "payload_database_schema_unsupported")
    columns = {name: {item["name"]: item["type"].upper() for item in table["columns"]}
        for name, table in database["tables"].items() if name not in {"sqlite_sequence", "sqlite_stat1", "sqlite_stat4"}}
    _require(columns == contract["columns"] and database["application_id"] == contract["application_id"]
        and database["user_version"] == contract["user_version"], "payload_database_schema_unsupported")
    return database


def _validate_evidence(cuts, targets, snapshots, layout):
    from src.analytics.event_evidence import EventDependency, event_target_id, project_pattern_event
    evidence_targets = {}
    for source, records in targets.items():
        for target in records.values():
            if target["participant"] == "events" and target["version"] == 1:
                evidence_targets[(source, event_target_id(target["descriptor"]))] = target
    for cut in cuts:
        if cut.table != "suggestions_policy_v1":
            continue
        for image in cut.selected_rows:
            row = image.values
            _require("event_dependencies" not in json.loads(row["pattern_json"]),
                "payload_pattern_protocol_unsupported")
            evidence = json.loads(row["evidence_json"])
            _require(type(evidence) is list and bool(evidence), "payload_event_evidence_missing")
            for encoded in evidence:
                dependency = EventDependency.from_dict(encoded)
                target = evidence_targets.get((image.source, dependency.target_id))
                _require(target is not None, "payload_event_target_unknown")
                original = Path(target["descriptor"]["path"])
                relative = original.relative_to(_root(snapshots[image.source])).as_posix()
                mappings = [item for item in layout.database_relocations if relative ==
                    (item.ordinary_relative if image.source == "ordinary" else item.latest_relative)]
                _require(len(mappings) == 1, "payload_event_database_unknown")
                events = [item for item in cuts if item.database == mappings[0].destination_relative
                    and item.table == "events_policy_v1"]
                _require(len(events) == 1, "payload_event_database_unknown")
                parents = [item.values for item in events[0].selected_rows if item.values["id"] == dependency.event_id]
                _require(len(parents) == 1 and project_pattern_event(parents[0], dependency.target_id).dependency
                    == dependency, "payload_event_revision_or_lineage_changed")


_LOOKUP_INDEX = {"type": "index", "name": "idx_erasure_target_participant_id", "tbl_name": "erasure_targets",
    "sql": "CREATE INDEX idx_erasure_target_participant_id ON erasure_targets(participant, id)"}


def _privacy_schema_additions() -> tuple[tuple[dict, ...], ...]:
    """Each known additive privacy-schema change, as its exact sqlite_master rows.

    The erasure lookup index (014b422), the U3b owner schema (all seven rows
    of FACT_OWNER_SCHEMA together), the U3b-4a-0 voice-attribution schema
    (every row of VOICE_ATTRIBUTION_SCHEMA together) and the U3b-0d3 deletion
    ledger (DELETION_LEDGER_SCHEMA). A group is either wholly absent from an
    older image or wholly present in both. Ledger rows are authority, cloned
    from the latest image like every other non-payload table: derivation never
    appends to the ledger.
    """
    from src.memory.deletion_ledger import deletion_ledger_schema_objects
    from src.memory.fact_ledger import fact_owner_schema_objects
    from src.memory.voice_schema import voice_attribution_schema_objects
    return ((_LOOKUP_INDEX,), tuple(fact_owner_schema_objects()), tuple(voice_attribution_schema_objects()),
            tuple(deletion_ledger_schema_objects()))


def _privacy_schema_added(ordinary: list[dict], latest: list[dict]) -> bool:
    """Recognize latest == ordinary + any non-empty subset of the known additive groups.

    Each source schema is already authenticated by _database_contract. An
    added group must appear exactly once per row in the latest image and not
    at all in the ordinary image; every other row must match in order. This
    rule neither normalizes the schemas nor changes the latest schema that
    materialization clones exactly. Owner rows are not payload; they are
    cloned from the latest image (U3b W9).
    """
    names = {row["name"] for row in ordinary}
    added = []
    for group in _privacy_schema_additions():
        if any(row["name"] in names for row in group):
            continue  # Already in the older image: it must then match like any other row.
        if all(latest.count(row) == 1 for row in group):
            added.extend(group)
    return bool(added) and [row for row in latest if row not in added] == ordinary


def plan_payload_cut(ordinary: ClosedStateSnapshot, latest: ClosedStateSnapshot, *, now: float,
                     layout: DerivationLayout) -> PayloadCutPlan:
    """Deterministic read-only preflight of pinned inputs; no SDK or destination I/O."""
    _require(type(ordinary) is ClosedStateSnapshot and type(latest) is ClosedStateSnapshot
        and _number(now), "payload_snapshot_or_time_invalid")
    verify_snapshot(ordinary)
    verify_snapshot(latest)
    _validate_layout(layout, ordinary, latest)
    privacy = next(item for item in layout.database_relocations
        if item.destination_relative == layout.privacy_relative)
    old_privacy = ordinary.manifest["databases"][privacy.ordinary_relative]
    new_privacy = latest.manifest["databases"][privacy.latest_relative]
    policy = _Policy(ordinary, latest, old_privacy, new_privacy, now)
    targets = {"ordinary": _targets(ordinary, privacy.ordinary_relative),
        "latest": _targets(latest, privacy.latest_relative)}
    _require(set(layout.target_relocations) == set(targets["ordinary"]) | set(targets["latest"]),
        "payload_target_relocation_incomplete")
    for identity, descriptor in layout.target_relocations.items():
        original = targets["latest"].get(identity, targets["ordinary"].get(identity))["descriptor"]
        _require(type(descriptor) is dict and set(descriptor) == set(original), "payload_target_role_changed")
        for key, value in original.items():
            if type(value) is str and Path(value).is_absolute():
                _require(type(descriptor[key]) is str and Path(descriptor[key]).is_absolute()
                    and Path(descriptor[key]).is_relative_to(layout.destination_root)
                    and ".." not in Path(descriptor[key]).parts, "payload_target_destination_invalid")
            else:
                _require(descriptor[key] == value, "payload_target_role_changed")
    tables, observed = [], set()
    matched_databases = set()
    for mapping in layout.database_relocations:
        policy.database = mapping.latest_relative
        old = _database_contract(ordinary, mapping.ordinary_relative)
        new = _database_contract(latest, mapping.latest_relative)
        for snapshot, database in ((ordinary, old), (latest, new)):
            for table in database["tables"].values():
                for row in table["rows"]:
                    for value in row.values():
                        if type(value) is dict:
                            read_blob(snapshot, value)
        compatible_schema = (old["schema"] == new["schema"] or
            mapping.destination_relative == layout.privacy_relative
            and _privacy_schema_added(old["schema"], new["schema"]))
        _require(compatible_schema and old["application_id"] == new["application_id"]
            and old["user_version"] == new["user_version"], "payload_database_schema_changed")
        for name in ("pending_audio_deletions", "audio_write_intents", "quarantines", "staged"):
            _require(not new["tables"].get(name, {}).get("rows"), "payload_store_obligations_unresolved")
        selected = _select_database(old, new, policy)
        matched_databases.add(mapping.latest_relative)
        for name, images in selected.items():
            observed.add(name)
            table = new["tables"][name]
            tables.append(TableCut(mapping.destination_relative, name, tuple(table["primary_key"]),
                tuple(table["rows"]), tuple(images[key] for key in sorted(images))))
    _require(all(not (PAYLOAD_TABLES & data["tables"].keys()) or path in matched_databases
        for path, data in latest.manifest["databases"].items()), "payload_database_layout_incomplete")
    required = {"facts", "utterances", "wake_clips_v1", "stt_corrections_v1", "conversation_turns_v1",
        "conversation_export_parents_v1", "events_policy_v1", "suggestions_policy_v1", "metrics",
        "lineage", "voice_fingerprints", "voice_capture_reviews"}
    _require(required <= observed and {"live_deliveries", "live_delivery_sources", "live_delivery_lineage"}
        <= new_privacy["tables"].keys(), "payload_family_coverage_missing")
    for snapshot in (ordinary, latest):
        for key, record in snapshot.manifest["files"].items():
            if key.startswith("state/config/") and key.endswith((".yaml", ".yml", ".json")):
                raw = _read_file(snapshot, key)
                _require(str(_root(snapshot)).encode() not in raw, "payload_configuration_relocation_unsupported")
    _validate_evidence(tables, targets, {"ordinary": ordinary, "latest": latest}, layout)
    files = _file_plan(ordinary, latest, layout, tables, policy, targets)
    indexes = _index_plan(ordinary, latest, layout, tables)
    for mapping in layout.file_roots:
        if mapping.family == "diagnostics":
            _require(sum(cut.database == mapping.destination_relative + ".sqlite3" and cut.table == "metrics"
                for cut in tables) == 1, "payload_diagnostic_layout_mismatch")
    coverage = tuple(sorted(FAMILIES))
    value = _plan_value(ordinary, latest, now, layout, tables, files, indexes, coverage)
    return PayloadCutPlan(ordinary, latest, now, layout, tuple(tables), files, indexes,
        coverage, (ordinary.manifest_sha256, latest.manifest_sha256), _digest(value))


def _atomic_file(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require(path.parent.resolve() == path.parent and not path.is_symlink(), "payload_destination_redirected")
    descriptor, name = tempfile.mkstemp(prefix=".payload-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _decode_row(snapshot, row):
    return {key: read_blob(snapshot, value) if type(value) is dict else value for key, value in row.items()}


def _relocate_row(row, table, journal, *, evidence=False):
    from src.analytics.event_evidence import EventDependency
    value = dict(row)
    if table == "wake_clips_v1":
        old = value["target_id"]
        _require(old in journal.target_id_map, "payload_target_mapping_missing")
        value["target_id"] = journal.target_id_map[old]
    if table == "suggestions_policy_v1" and evidence:
        dependencies = []
        for raw in json.loads(value["evidence_json"]):
            item = EventDependency.from_dict(raw)
            _require(item.target_id in journal.event_dependency_target_map, "payload_event_target_unknown")
            mapped = journal.event_dependency_target_map[item.target_id]
            dependencies.append(EventDependency(mapped, item.event_id, item.revision, item.expires_at).to_dict())
        value["evidence_json"] = json.dumps(dependencies, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _quote(value):
    return '"' + value.replace('"', '""') + '"'


def _apply_tables(plan, destination, journal):
    by_database = {}
    for cut in plan.tables:
        by_database.setdefault(cut.database, []).append(cut)
    for relative, cuts in by_database.items():
        path = destination / _relative(relative)
        _require(path.is_file() and path.resolve() == path, "payload_destination_database_invalid")
        with closing(sqlite3.connect(path)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            db.execute("PRAGMA defer_foreign_keys=ON")
            prepared = []
            for cut in cuts:
                current = [dict(row) for row in db.execute("SELECT * FROM " + _quote(cut.table))]
                expected = [_relocate_row(_decode_row(plan.latest, row), cut.table, journal) for row in cut.latest_rows]
                def key(row):
                    return tuple(row[name] for name in cut.primary_key)
                _require(sorted(current, key=key) == sorted(expected, key=key), "payload_destination_payload_changed")
                selected = [_relocate_row(_decode_row(plan.ordinary if image.source == "ordinary" else plan.latest,
                    image.values), cut.table, journal, evidence=True) for image in cut.selected_rows]
                prepared.append((cut, selected))
            # A parent deletion can cascade into child payload tables. All
            # beforeimages were checked above, so reconcile their actual rows
            # after each cascade rather than treating it as external mutation.
            for cut, selected in prepared:
                current = [dict(row) for row in db.execute("SELECT * FROM " + _quote(cut.table))]
                _replace_payload_rows(db, cut, current, selected)
            for cut, selected in prepared:
                actual = [dict(row) for row in db.execute("SELECT * FROM " + _quote(cut.table))]
                _require(sorted(actual, key=_json_sql) == sorted(selected, key=_json_sql),
                    "payload_destination_afterimage_changed")
            _require(not list(db.execute("PRAGMA foreign_key_check")), "payload_foreign_key_incomplete")


def _replace_payload_rows(db, cut, current, selected):
    keys = cut.primary_key
    def key(row):
        return tuple(row[name] for name in keys)
    before, after = {key(row): row for row in current}, {key(row): row for row in selected}
    _require(len(after) == len(selected), "payload_duplicate_selected_identity")
    where = " AND ".join(_quote(name) + "=?" for name in keys)
    for identity in sorted(before.keys() - after.keys()):
        if cut.table == "facts":
            db.execute("DELETE FROM index_jobs WHERE fact_id=?", identity)
        db.execute("DELETE FROM " + _quote(cut.table) + " WHERE " + where, identity)
    for identity, row in sorted(after.items()):
        if identity not in before:
            columns = tuple(row)
            db.execute("INSERT INTO " + _quote(cut.table) + " (" + ",".join(map(_quote, columns))
                + ") VALUES (" + ",".join("?" for _ in columns) + ")", tuple(row[name] for name in columns))
        elif row != before[identity]:
            columns = tuple(name for name in row if name not in keys and row[name] != before[identity][name])
            _require(bool(columns), "payload_primary_identity_changed")
            db.execute("UPDATE " + _quote(cut.table) + " SET "
                + ",".join(_quote(name) + "=?" for name in columns) + " WHERE " + where,
                tuple(row[name] for name in columns) + identity)


def _json_sql(row):
    return _json({key: {"blob_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
        if isinstance(value, bytes) else value for key, value in row.items()})


def _family(table):
    for family, names in (
            ("semantic_index", {"facts", "fact_sources", "fact_dependencies"}),
            ("ambient_capture", {"utterances", "capture_conversations"}),
            ("collectors", {"wake_clips_v1", "stt_corrections_v1", "conversation_turns_v1",
                "conversation_export_parents_v1"}),
            ("events", {"events_policy_v1"}), ("suggestions", {"suggestions_policy_v1"}),
            ("diagnostics", {"metrics", "lineage"})):
        if table in names:
            return family
    _require(table.startswith("voice_"), "payload_family_unknown")
    return "voices"


def _exclusion_proof(plan, destination):
    removed = []
    for cut in plan.tables:
        selected = {tuple(image.values[name] for name in cut.primary_key) for image in cut.selected_rows}
        for row in cut.latest_rows:
            identity = tuple(row[name] for name in cut.primary_key)
            if identity not in selected:
                removed.append({"database": cut.database, "table": cut.table,
                    "primary_key": list(cut.primary_key), "identity": list(identity),
                    "row_sha256": _digest(row), "family": _family(cut.table)})
    files = []
    for cut in plan.files:
        if cut.source is not None:
            continue
        record = plan.latest.manifest["files"].get("state/" + cut.destination_relative)
        if record is not None:
            roots = [item for item in plan.layout.file_roots
                if PurePosixPath(cut.destination_relative).is_relative_to(item.destination_relative)]
            _require(len(roots) == 1, "payload_exclusion_file_ambiguous")
            files.append({"relative_path": cut.destination_relative, "sha256": record["sha256"],
                "family": roots[0].family})
    info = destination.stat()
    return {"schema_version": 1, "kind": "ordinary_payload_cut",
        "ordinary_binding_id": plan.ordinary.source_binding_id, "latest_binding_id": plan.latest.source_binding_id,
        "ordinary_manifest_sha256": plan.ordinary.manifest_sha256,
        "latest_manifest_sha256": plan.latest.manifest_sha256, "plan_sha256": plan.plan_sha256,
        "destination_root": str(destination), "destination_identity": {"device": info.st_dev, "inode": info.st_ino},
        "removed_rows": sorted(removed, key=_json), "removed_files": sorted(files, key=_json)}


def _clone_latest(plan, destination):
    if destination.exists():
        _require(destination.is_dir() and not destination.is_symlink() and destination.resolve() == destination,
            "payload_destination_redirected")
        _require(all(path.is_dir() and not path.is_symlink()
            and path.relative_to(destination).parts[0] in {"config", "data"}
            for path in destination.rglob("*")), "payload_destination_not_empty")
    else:
        _require(destination.parent.resolve(strict=True) == destination.parent, "payload_destination_redirected")
        destination.mkdir(mode=0o700)
    for key, record in sorted(plan.latest.manifest["files"].items()):
        if not key.startswith("state/"):
            continue
        relative = key.removeprefix("state/")
        if relative == ".kza-state-binding.json":
            continue
        _require(record["kind"] in {"file", "sqlite_backup", "sdk_opaque"}, "payload_clone_kind_unknown")
        _atomic_file(destination / _relative(relative), _read_file(plan.latest, key))


def _apply_files(plan, destination):
    for cut in plan.files:
        path = destination / _relative(cut.destination_relative)
        if cut.source is None:
            if path.exists():
                _require(path.is_file() and not path.is_symlink() and path.resolve() == path,
                    "payload_destination_redirected")
                path.unlink()
        else:
            source = plan.ordinary if cut.source == "ordinary" else plan.latest
            _atomic_file(path, _read_file(source, cut.snapshot_relative))


def _diagnostic_line(row):
    from datetime import datetime, timezone
    from src.obs.metrics_emitter import project_diagnostic
    payload = json.loads(row["payload"])
    _require(project_diagnostic(payload) == payload, "payload_diagnostic_record_invalid")
    value = dict(payload, record_id=row["id"], recorded_at=row["born"], expires_at=row["expires"],
        **{"@timestamp": datetime.fromtimestamp(row["born"], timezone.utc).isoformat(
            timespec="milliseconds").replace("+00:00", "Z"), "service": "kza-voice"})
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"


def _publish_diagnostics(plan, destination):
    for mapping in plan.layout.file_roots:
        if mapping.family != "diagnostics":
            continue
        catalog = mapping.destination_relative + ".sqlite3"
        cuts = [cut for cut in plan.tables if cut.database == catalog and cut.table == "metrics"]
        _require(len(cuts) == 1, "payload_diagnostic_layout_mismatch")
        rows = sorted((image.values for image in cuts[0].selected_rows), key=lambda row: (row["born"], row["id"]))
        for partition, suffix in ((0, ""), (1, ".1")):
            raw = "".join(_diagnostic_line(row) for row in rows if row["partition"] == partition).encode()
            _atomic_file(destination / (mapping.destination_relative + suffix), raw)
        with closing(sqlite3.connect(destination / catalog)) as db, db:
            db.execute("UPDATE state SET value=1 WHERE key='dirty'")
            db.execute("UPDATE state SET value=? WHERE key='active_bytes'",
                ((destination / mapping.destination_relative).stat().st_size,))


def _sdk_records(collection):
    value = collection.get(include=["documents", "metadatas", "embeddings"])
    ids, vectors = value["ids"], value["embeddings"]
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    _require(type(ids) is list and len(ids) == len(set(ids)) and collection.count() == len(ids),
        "payload_sdk_export_incomplete")
    arrays = [value["documents"], value["metadatas"], vectors]
    if not ids:
        arrays = [[], [], []]
    _require(all(array is not None and len(array) == len(ids) for array in arrays),
        "payload_sdk_export_incomplete")
    records = [{"id": identity, "document": arrays[0][i], "metadata": arrays[1][i],
        "embedding": arrays[2][i]} for i, identity in enumerate(ids)]
    _json(records)
    return sorted(records, key=lambda row: row["id"])


def _apply_indexes(plan, destination, reservation, factory, *, operation=None):
    present = [cut for cut in plan.indexes if cut.present]
    if not present:
        return
    for cut in present:
        _require(importlib.metadata.version("chromadb") == cut.sdk_version, "payload_sdk_version_changed")
    from chromadb.config import Settings
    if factory is None:
        from chromadb import PersistentClient
        factory = PersistentClient
    grouped = {}
    for cut in present:
        grouped.setdefault(cut.destination_relative, []).append(cut)
    for relative, cuts in sorted(grouped.items()):
        owner = {"entered": False, "client": None, "close_attempted": False, "close_returned": False}
        reservation["sdk"].append(owner)
        entry = None
        if operation is not None:
            extent = operation.extent(relative, family="semantic_index", representation="sdk_work")
            entry = operation.sdk_enter(extent, sdk_version=cuts[0].sdk_version)
            owner["catalog_entry"] = entry
        primary = None
        try:
            arguments = {"path": str(destination / relative), "settings": Settings(anonymized_telemetry=False)}
            owner["entered"] = True
            owner["client"] = factory(**arguments) if entry is None else entry.construct(factory, **arguments)
            _edit_collection(owner["client"], cuts)
        except BaseException as error:
            primary = error
        finally:
            # construct retains an acquired handle before persisting its return.
            # Its close must still run when that persistence raised, even though
            # the caller never received the handle. Uncertain returns stay sealed.
            if entry is not None or owner["client"] is not None:
                owner["close_attempted"] = True
                try:
                    if entry is None:
                        owner["client"].close()
                    else:
                        entry.close()
                except BaseException as cleanup:
                    if primary is not None:
                        raise BaseExceptionGroup("payload SDK operation and close failed", [primary, cleanup])
                    raise
                owner["close_returned"] = True
                owner["client"] = None
        if primary is not None:
            raise primary


def _edit_collection(client, cuts):
    for cut in cuts:
        collection = client.get_collection(name=cut.collection, embedding_function=None)
        _require(collection.metadata == cut.metadata, "payload_sdk_collection_metadata_changed")
        _require(_sdk_records(collection) == list(cut.latest_records), "payload_sdk_beforeimage_changed")
        old = {row["id"] for row in cut.latest_records}
        chosen = {row["id"] for row in cut.selected_records}
        if old - chosen:
            collection.delete(ids=sorted(old - chosen))
        if cut.selected_records:
            rows = cut.selected_records
            collection.upsert(ids=[row["id"] for row in rows], documents=[row["document"] for row in rows],
                metadatas=[row["metadata"] for row in rows], embeddings=[row["embedding"] for row in rows])
        _require(_sdk_records(collection) == list(cut.selected_records), "payload_sdk_afterimage_changed")


def _relocation_audit(plan, journal):
    from src.analytics.event_evidence import EventDependency
    audit = []
    for cut in plan.tables:
        if cut.table != "suggestions_policy_v1":
            continue
        for image in cut.selected_rows:
            for raw in json.loads(image.values["evidence_json"]):
                event = EventDependency.from_dict(raw)
                target = journal.event_dependency_target_map[event.target_id]
                if target != event.target_id:
                    audit.append({"database": cut.database, "table": cut.table,
                        "identity": [image.values[key] for key in cut.primary_key],
                        "field": "evidence_json.target_id", "old": event.target_id, "new": target,
                        "event_id": event.event_id, "revision": event.revision, "expires_at": event.expires_at})
    return tuple(sorted(audit, key=_json))


def _prepared_ownership(destination, journal):
    """Build the eventual policy in memory; preparation remains inadmissible."""
    from src.memory.state_journal_derivation import JournalDerivationReceipt
    _require(type(journal) is JournalDerivationReceipt and journal.destination_state == destination,
        "payload_journal_receipt_mismatch")
    path = destination / ".kza-target-ownership.json"
    _require(path.resolve(strict=True) == path and not path.is_symlink(), "payload_preparation_redirected")
    blocked = path.read_bytes()
    policy = json.loads(blocked)
    _require(type(policy) is dict and set(policy) == {"schema_version", "source_bindings", "owned_targets",
        "delegated_targets", "historical_targets"} and type(policy["schema_version"]) is int
        and policy["schema_version"] == 1 and policy["owned_targets"] == [] and policy["delegated_targets"] == [],
        "payload_preparation_not_blocked")
    registered = journal.registered_target_ids
    _require(bool(registered) and len(registered) == len(set(registered)) and all(_sha_token(item) for item in registered),
        "payload_journal_receipt_mismatch")
    database = destination / _relative(journal.privacy_relative)
    _require(database.resolve(strict=True) == database, "payload_destination_redirected")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        actual = {row[0] for row in db.execute("SELECT id FROM erasure_targets")}
    _require(actual == set(registered), "payload_journal_receipt_mismatch")
    qualified = dict(policy, owned_targets=sorted(registered))
    return path, blocked, _json(qualified) + b"\n"


def _finish_prepared_payload(plan, destination, journal, reservation, sdk_client_factory, *, operation=None):
    ownership, blocked, qualified = _prepared_ownership(destination, journal)
    _apply_tables(plan, destination, journal)
    _apply_files(plan, destination)
    _publish_diagnostics(plan, destination)
    _apply_indexes(plan, destination, reservation, sdk_client_factory, operation=operation)
    _require(all(owner["close_returned"] and owner["client"] is None for owner in reservation["sdk"]),
        "payload_sdk_retirement_unconfirmed")
    audit = _relocation_audit(plan, journal)
    privacy = next(item for item in plan.layout.database_relocations
        if item.destination_relative == plan.layout.privacy_relative)
    active_latest = _targets(plan.latest, privacy.latest_relative)
    event_targets = {old: new for old, new in journal.event_target_id_map.items() if old in active_latest}
    _require(ownership.read_bytes() == blocked, "payload_preparation_changed")
    record = destination / ".kza-payload-preparation.json"
    value = {"schema_version": 1, "plan_sha256": plan.plan_sha256, "phase": "materialized",
        "serving": False, "journal": journal.to_dict(), "relocation_audit": list(audit),
        "sdk_close_returns": len(reservation["sdk"]), "retained_copy_cleanup_ack": False}
    _atomic_file(record, _json(value) + b"\n")
    if operation is not None:
        operation.prepared(record)
    _atomic_file(ownership, qualified)
    if operation is not None:
        operation.finish()
    return DerivedState(destination, plan.plan_sha256, journal.history_sha256,
        dict(journal.target_id_map), event_targets, dict(journal.event_dependency_target_map), audit, record)


def materialize_derived_state(plan: PayloadCutPlan, destination: Path, *, controller,
                              sdk_client_factory=None) -> DerivedState:
    """Prepare a non-serving state at its FINAL locator under the controller lease.

    Failure leaves its preparation record and incomplete ownership sealed. A
    successful return is preparation evidence, not admission or copy erasure.
    """
    from src.memory.copy_catalog import reserve_copy
    from src.memory.state_journal_derivation import _source, _stage, qualify_and_register_derived_journal
    _require(type(plan) is PayloadCutPlan and isinstance(destination, Path)
        and destination == plan.layout.destination_root, "payload_destination_mismatch")
    checked = plan_payload_cut(plan.ordinary, plan.latest, now=plan.now, layout=plan.layout)
    _require(checked.plan_sha256 == plan.plan_sha256
        and checked == plan, "payload_plan_changed")
    controller._require_lease()
    privacy = next(item for item in plan.layout.database_relocations
        if item.destination_relative == plan.layout.privacy_relative)
    stage = _stage()
    _source(stage, Path(controller.root), plan.latest, privacy.latest_relative, latest=True)
    _source(stage, Path(controller.root), plan.ordinary, privacy.ordinary_relative, latest=False)
    for cut in plan.indexes:
        if cut.present:
            _require(importlib.metadata.version("chromadb") == cut.sdk_version, "payload_sdk_version_changed")
    key = str(destination)
    _require(key not in _INCOMPLETE, "payload_previous_preparation_unconfirmed")
    refs = []
    for snapshot in (plan.ordinary, plan.latest):
        info = snapshot.snapshot_root.stat()
        reference = {"source_binding_id": snapshot.source_binding_id,
            "snapshot_root": str(snapshot.snapshot_root),
            "root_identity": {"device": info.st_dev, "inode": info.st_ino},
            "manifest_sha256": snapshot.manifest_sha256}
        if reference not in refs:
            refs.append(reference)
    operation = reserve_copy(controller, "derived_state", destination, plan.latest.source_binding_id,
        tuple(dict.fromkeys(snapshot.source_binding_id for snapshot in (plan.ordinary, plan.latest))),
        source_snapshot_refs=tuple(refs))
    reservation = {"phase": "preparing", "sdk": [], "catalog": operation}
    _INCOMPLETE[key] = reservation
    try:
        operation.create_root()
        operation.extent(".", family="managed_payload", representation="derived_state")
        _clone_latest(plan, destination)
        record = destination / ".kza-payload-preparation.json"
        _atomic_file(record, _json({"schema_version": 1, "plan_sha256": plan.plan_sha256,
            "phase": "preparing", "serving": False}) + b"\n")
        proof = _exclusion_proof(plan, destination)
        journal = qualify_and_register_derived_journal(controller, plan.latest, destination,
            privacy_relative=plan.layout.privacy_relative, target_relocations=plan.layout.target_relocations,
            ordinary_snapshot=plan.ordinary, payload_cut_proof=proof, keep_blocked=True)
        result = _finish_prepared_payload(plan, destination, journal, reservation, sdk_client_factory,
            operation=operation)
    except BaseException as error:
        reservation["phase"] = "failed"
        try:
            operation.failed(type(error).__name__)
        except BaseException as catalog_error:
            raise BaseExceptionGroup("payload preparation and catalog failure", [error, catalog_error]) from None
        raise
    reservation["phase"] = "materialized"
    del _INCOMPLETE[key]
    return result
