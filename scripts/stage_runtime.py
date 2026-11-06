#!/usr/bin/env python3
"""Prepare and select source-free runtime slots with independent state copies.

Stop every writer before copying state. The CLI prepares/selects offline slots;
AdmissionController additionally owns a managed child's admission and real wait.
Neither copying nor selecting performs schema or data reconciliation.
"""

import argparse
from contextlib import closing, contextmanager
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import tarfile
import tempfile
from uuid import uuid4


_HEAD_KEYS = {"schema_version", "generation", "binding_id", "status", "run_id"}
_BINDING_KEYS = {"schema_version", "slot", "runtime_root", "state_root", "release_sha256",
                 "code_commit", "python_magic_hex", "state_identity", "store_roots", "schemas",
                 "sdk_versions", "authority", "ownership_sha256"}
_TICKET_KEYS = {"schema_version", "admission_root", "binding_id", "generation", "run_id",
                "lease_fd", "lease_identity", "controller_pid"}
_MARKER = ".kza-state-binding.json"
_CLOSED_RESOURCES = {"graph_closed", "privacy_closed", "sdk_closed", "native_closed"}
_manual_components_module = None
_PROTECTED_TABLES = frozenset({
    "conversation_tombstones", "conversation_erasures", "voice_profiles", "voice_profile_sources",
    "voice_review_requests", "voice_profile_people", "voice_group_requests",
    "voice_group_request_lineage", "credentials", "sessions", "retired_users", "auth_meta",
    "credential_attempts",
    "capture_runtime_control", "ambient_runtime_control", "capture_browser_recorders",
    "live_deliveries", "live_delivery_sources", "live_delivery_lineage",
    "reminders", "reminder_home_actions", "routines", "routine_runs", "routine_run_actions",
    "lists", "list_items", "quarantines", "staged",
    "legacy_quarantine_batches_v1", "legacy_quarantine_records_v1",
    "pending_audio_deletions", "audio_write_intents",
})

# Source-owned v1 column contracts. Unknown additions need an explicit decision
# about authority versus payload before a managed generation may serve them.
_TABLE_COLUMNS = {
    "capture_runtime_control": "id:INTEGER mode:TEXT revision:INTEGER",
    "ambient_runtime_control": "id:INTEGER paused:INTEGER revision:INTEGER",
    "capture_browser_recorders": "token_hash:TEXT state:TEXT kind:TEXT",
    "live_deliveries": "seq:INTEGER ticket:TEXT client_id:TEXT connection_id:TEXT event_id:TEXT nonce:TEXT issued_at:REAL expires_at:REAL owner_namespace:TEXT",
    "live_delivery_sources": "ticket:TEXT conversation_id:TEXT",
    "live_delivery_lineage": "ticket:TEXT version:INTEGER source_digest:TEXT",
    "live_connections": "connection_id:TEXT client_id:TEXT runtime_id:TEXT owner_namespace:TEXT",
    "live_resets": "id:TEXT client_id:TEXT connection_id:TEXT nonce:TEXT cutoff:INTEGER owner_namespace:TEXT",
    "facts": "id:TEXT content:TEXT category:TEXT confidence:REAL state:TEXT extracted_at:REAL expires_at:REAL revision:INTEGER confirmed_at:REAL confirmed_by:TEXT",
    "fact_sources": "fact_id:TEXT capture_id:TEXT conversation_id:TEXT captured_at:REAL expires_at:REAL origin:TEXT dependencies:TEXT",
    "fact_dependencies": "fact_id:TEXT source_fact_id:TEXT",
    "conversation_tombstones": "conversation_id:TEXT deleted_at:REAL deleted_by:TEXT",
    "conversation_erasures": "id:TEXT conversation_id:TEXT requested_at:REAL requested_by:TEXT",
    "erasure_targets": "id:TEXT participant:TEXT version:INTEGER descriptor:TEXT",
    "erasure_steps": "operation_id:TEXT participant:TEXT target_id:TEXT version:INTEGER state:TEXT attempts:INTEGER next_attempt_at:REAL claim_token:TEXT cursor:TEXT pending_count:INTEGER last_error:TEXT",
    "index_jobs": "fact_id:TEXT revision:INTEGER operation:TEXT queued_at:REAL attempts:INTEGER last_error:TEXT",
    "policy_index_deletions": "target_id:TEXT fact_id:TEXT revision:INTEGER state:TEXT attempts:INTEGER next_attempt_at:REAL claim_token:TEXT last_error:TEXT",
    "wake_clips_v1": "sequence:INTEGER id:TEXT target_id:TEXT bucket:TEXT room_id:TEXT score:REAL sources_json:TEXT captured_at:REAL expires_at:REAL published:INTEGER",
    "stt_corrections_v1": "id:TEXT original_text:TEXT corrected_text:TEXT user_id:TEXT sample_rate:INTEGER used:INTEGER",
    "conversation_turns_v1": "id:TEXT group_id:TEXT input_sources_json:TEXT quality:TEXT digest:TEXT",
    "conversation_export_parents_v1": "export_id:TEXT parent_id:TEXT",
    "voice_fingerprints": "id:TEXT model_key:TEXT dimensions:INTEGER embedding:BLOB duration_s:REAL created_at:REAL expires_at:REAL sources_json:TEXT suggested_profile:TEXT suggested_revision:INTEGER similarity:REAL",
    "voice_fingerprint_sources": "fingerprint_id:TEXT conversation_id:TEXT",
    "voice_capture_reviews": "fingerprint_id:TEXT audio_wav:BLOB transcript:TEXT room_id:TEXT duration_s:REAL clip_duration_s:REAL audio_metadata_json:TEXT transcript_metadata_json:TEXT",
    "voice_profiles": "id:TEXT name:TEXT model_key:TEXT dimensions:INTEGER embedding:BLOB sources_json:TEXT confirmed_at:REAL confirmed_by:TEXT revision:INTEGER deleted_at:REAL",
    "voice_profile_sources": "profile_id:TEXT conversation_id:TEXT",
    "voice_review_requests": "actor:TEXT request_id:TEXT digest:TEXT profile_id:TEXT",
    "voice_profile_people": "profile_id:TEXT user_id:TEXT revision:INTEGER updated_at:REAL updated_by:TEXT",
    "voice_groups": "id:TEXT revision:INTEGER state:TEXT reason:TEXT similarity:REAL created_at:REAL expires_at:REAL sources_json:TEXT",
    "voice_group_members": "fingerprint_id:TEXT group_id:TEXT",
    "voice_group_requests": "actor:TEXT request_id:TEXT digest:TEXT expires_at:REAL sources_json:TEXT",
    "voice_group_lineage": "group_id:TEXT source_digest:TEXT member_digest:TEXT",
    "voice_group_request_lineage": "actor:TEXT request_id:TEXT source_digest:TEXT",
    "voice_group_schema": "version:INTEGER", "voice_person_schema": "version:INTEGER",
    "voice_capture_review_schema": "version:INTEGER",
    "auth_meta": "key:TEXT value:TEXT",
    "credentials": "id:TEXT digest:TEXT subject:TEXT role:TEXT user_id:TEXT issued_at:REAL expires_at:REAL revision:INTEGER samples:TEXT",
    "sessions": "digest:TEXT credential:TEXT legacy_digest:TEXT issued_at:REAL expires_at:REAL revision:INTEGER",
    "retired_users": "user_id:TEXT retired_at:REAL",
    "credential_attempts": "request_id:TEXT actor_key:TEXT command_digest:TEXT kind:TEXT credential_id:TEXT expected_revision:INTEGER outcome:TEXT result_revision:INTEGER reason:TEXT completed_at:REAL",
    "suggestions_policy_v1": "row_id:INTEGER id:TEXT pattern_key:TEXT created_at:REAL observed_at:REAL expires_at:REAL status:TEXT snooze_until:REAL pattern_json:TEXT evidence_json:TEXT sources_json:TEXT message:TEXT",
    "utterances": "id:INTEGER room_id:TEXT t0:REAL t1:REAL text:TEXT speaker:TEXT speaker_confidence:REAL azimuth:REAL azimuth_stability:REAL source:TEXT confidence:REAL no_speech_prob:REAL vad_prob:REAL lang:TEXT lang_prob:REAL lang_ok:INTEGER during_tts:INTEGER distilled:INTEGER text_empty:INTEGER audio_path:TEXT capture_id:TEXT expires_at:REAL conversation_id:TEXT created_at:REAL",
    "capture_conversations": "id:TEXT room_id:TEXT started_at:REAL last_seen_at:REAL",
    "pending_audio_deletions": "path:TEXT requested_at:REAL attempts:INTEGER last_error:TEXT conversation_id:TEXT",
    "audio_write_intents": "path:TEXT utterance_id:INTEGER conversation_id:TEXT expires_at:REAL",
    "metrics": "id:TEXT born:REAL expires:REAL payload:TEXT partition:INTEGER bytes:INTEGER",
    "lineage": "record_id:TEXT conversation_id:TEXT",
    "state": "key:TEXT value:INTEGER", "quarantines": "source:TEXT destination:TEXT moved:INTEGER",
    "staged": "name:TEXT",
    "reminders": "id:TEXT user_id:TEXT text:TEXT trigger_at:REAL created_at:REAL recurrence:TEXT recurrence_end:REAL ha_actions:TEXT state:TEXT last_fired_at:REAL fire_count:INTEGER scope:TEXT",
    "reminder_home_actions": "request_id:TEXT body:TEXT status:TEXT reason:TEXT",
    "routines": "routine_id:TEXT revision:INTEGER created_by:TEXT created_at:TEXT definition_json:TEXT",
    "routine_runs": "run_id:TEXT routine_id:TEXT occurrence_key:TEXT actor:TEXT source:TEXT status:TEXT reason:TEXT snapshot_json:TEXT scheduled_for:TEXT",
    "routine_run_actions": "run_id:TEXT action_index:INTEGER entity_id:TEXT status:TEXT reason:TEXT",
    "lists": "id:TEXT name:TEXT owner_type:TEXT owner_id:TEXT created_at:REAL updated_at:REAL",
    "list_items": "id:TEXT list_id:TEXT text:TEXT completed:INTEGER added_by:TEXT created_at:REAL completed_at:REAL",
    "events_policy_v1": "id:INTEGER timestamp:REAL event_type:TEXT entity_id:TEXT action:TEXT user_id:TEXT user_name:TEXT trigger_phrase:TEXT context_json:TEXT hour:INTEGER minute:INTEGER weekday:INTEGER is_weekend:INTEGER time_of_day:TEXT sources_json:TEXT metrics_json:TEXT expires_at:REAL text_expires_at:REAL",
    "legacy_quarantine_batches_v1": "id:TEXT snapshot_sha256:TEXT selection_sha256:TEXT collection_id:TEXT imported_at:TEXT records:INTEGER",
    "legacy_quarantine_records_v1": "batch_id:TEXT record_id:TEXT state:TEXT document:TEXT metadata_json:TEXT vector:BLOB encoding:TEXT digest:TEXT",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _identity(path: Path) -> dict:
    info = path.stat()
    return {"device": info.st_dev, "inode": info.st_ino}


def _safe_directory(path: Path, *, create: bool = False) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.resolve(strict=True) != path or not path.is_dir():
        raise ValueError("runtime_admission_unsafe_directory")
    return path


def _read_json(path: Path) -> dict:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024 * 1024:
            raise ValueError("runtime_admission_unsafe_record")
        value = json.load(stream)
    if type(value) is not dict:
        raise ValueError("runtime_admission_record_schema")
    return value


def _atomic_record(path: Path, value: dict, *, immutable: bool = False) -> None:
    _safe_directory(path.parent, create=True)
    raw = _canonical_json(value) + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=".admission-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if _read_json(path) != value:
                    raise ValueError("runtime_admission_immutable_conflict") from None
        else:
            if path.is_symlink():
                raise ValueError("runtime_admission_redirected_record")
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def read_admission_head(root: Path) -> dict | None:
    """Read the external monotonic head; it is never chosen from a copied ledger."""
    root = Path(root).resolve(strict=True)
    path = root / "admission" / "head.json"
    if not path.exists() and not path.is_symlink():
        return None
    _safe_directory(path.parent)
    head = _read_json(path)
    if (set(head) != _HEAD_KEYS or head["schema_version"] != 1
            or type(head["generation"]) is not int or head["generation"] < 1
            or type(head["binding_id"]) is not str
            or re.fullmatch("[0-9a-f]{64}", head["binding_id"]) is None
            or head["status"] not in {"closed", "running", "unknown"}
            or (head["run_id"] is not None and (type(head["run_id"]) is not str
                or re.fullmatch("[0-9a-f]{32}", head["run_id"]) is None))):
        raise ValueError("runtime_admission_head_schema")
    return head


def _cas_head(root: Path, expected: dict | None, binding_id: str, status: str,
              run_id: str | None = None) -> dict:
    if read_admission_head(root) != expected:
        raise ValueError("runtime_admission_head_changed")
    head = {"schema_version": 1, "generation": 1 if expected is None else expected["generation"] + 1,
            "binding_id": binding_id, "status": status, "run_id": run_id}
    if status == "closed":
        _verify_manual_head(root, head, read_binding(root, binding_id))
    _atomic_record(root / "admission" / "head.json", head)
    return head


def read_binding(root: Path, binding_id: str) -> dict:
    """Read an immutable canonical code/state descriptor by its content identity."""
    root = Path(root).resolve(strict=True)
    if type(binding_id) is not str or re.fullmatch("[0-9a-f]{64}", binding_id) is None:
        raise ValueError("runtime_admission_binding_id")
    path = root / "admission" / "bindings" / (binding_id + ".json")
    _safe_directory(path.parent)
    binding = _read_json(path)
    if set(binding) != _BINDING_KEYS or binding["schema_version"] != 1 or _digest(binding) != binding_id:
        raise ValueError("runtime_admission_binding_schema")
    return binding


def _typed(value: object) -> object:
    if type(value) is bytes:
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    return value


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@contextmanager
def _database_image(path: Path):
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("runtime_admission_redirected_database")
    with closing(sqlite3.connect(":memory:")) as image:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
            source.backup(image)
        image.row_factory = sqlite3.Row
        yield image


def _row_projection(db: sqlite3.Connection, table: str, where: str = "",
                    parameters: tuple = (), *, columns: tuple[str, ...] = ()) -> dict:
    fields = ",".join(map(_quote, columns)) if columns else "*"
    cursor = db.execute("SELECT " + fields + " FROM " + _quote(table) + where, parameters)
    columns = [item[0] for item in cursor.description]
    hashes = sorted(_digest([_typed(value) for value in row]) for row in cursor)
    return {"columns": columns, "row_hashes": hashes}


def _schema_contract(db: sqlite3.Connection, names: set[str]) -> dict | None:
    names = names - {"sqlite_sequence", "sqlite_stat1", "sqlite_stat4"}
    if not names or names == {"memories"}:
        return None
    if not names.intersection(_TABLE_COLUMNS) and {"collections", "segments", "migrations"} <= names:
        return None  # SDK internals are separately bound to their actual version.
    if names - _TABLE_COLUMNS.keys() or db.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('view','trigger') LIMIT 1").fetchone():
        raise ValueError("runtime_admission_unknown_schema")
    columns = {}
    for name in sorted(names):
        actual = {row[1]: row[2].upper() for row in db.execute("PRAGMA table_info(" + _quote(name) + ")")}
        if actual != dict(item.split(":") for item in _TABLE_COLUMNS[name].split()):
            raise ValueError("runtime_admission_unknown_schema")
        columns[name] = actual
        if name in {"voice_group_schema", "voice_person_schema", "voice_capture_review_schema"}:
            if [tuple(row) for row in db.execute("SELECT version FROM " + _quote(name))] != [(1,)]:
                raise ValueError("runtime_admission_unknown_schema")
    application = db.execute("PRAGMA application_id").fetchone()[0]
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if "metrics" in names:
        if application != 1264206897 or version not in (1, 2):
            raise ValueError("runtime_admission_unknown_schema")
    elif application != 0 or version != 0:
        raise ValueError("runtime_admission_unknown_schema")
    if "auth_meta" in names:
        row = db.execute("SELECT value FROM auth_meta WHERE key='schema_version'").fetchone()
        if row is None or row[0] not in {"1", "2", "3"}:
            raise ValueError("runtime_admission_unknown_schema")
    return {"columns": columns, "application_id": application, "user_version": version}


def _source_deadline(encoded: str) -> float:
    sources = json.loads(encoded)
    if (type(sources) is not list or not sources
            or any(type(source) is not dict or type(source.get("expires_at")) not in (int, float)
                   for source in sources)):
        raise ValueError("runtime_admission_unknown_deadline")
    return min(source["expires_at"] for source in sources)


def _deadline_projection(db: sqlite3.Connection, table: str, identity: str,
                         sources: str, deadline: str) -> dict:
    hashes = []
    for row in db.execute("SELECT " + ",".join(map(_quote, (identity, sources, deadline)))
                          + " FROM " + _quote(table)):
        ceiling = _source_deadline(row[sources])
        if row[deadline] is not None and row[deadline] < ceiling:
            hashes.append(_digest([row[identity], row[sources], row[deadline]]))
    return {"columns": [identity, sources, deadline], "row_hashes": sorted(hashes)}


def _mixed_projections(db: sqlite3.Connection, names: set[str], selected: dict) -> None:
    if "wake_clips_v1" in names:
        selected["wake_deadline_fences"] = _deadline_projection(
            db, "wake_clips_v1", "id", "sources_json", "expires_at")
        clauses = []
        if "stt_corrections_v1" in names:
            selected["stt_corrections_v1"] = _row_projection(db, "stt_corrections_v1",
                columns=("id", "corrected_text", "user_id", "used"))
            clauses.append("id IN (SELECT id FROM stt_corrections_v1)")
        if "conversation_turns_v1" in names:
            if db.execute("SELECT 1 FROM conversation_turns_v1 WHERE quality NOT IN "
                          "('unmarked','good','bad','corrected') LIMIT 1").fetchone():
                raise ValueError("runtime_admission_unknown_feedback")
            selected["conversation_feedback"] = _row_projection(db, "conversation_turns_v1",
                " WHERE quality!='unmarked'")
            clauses.append("id IN (SELECT id FROM conversation_turns_v1 WHERE quality!='unmarked')")
        if clauses:
            selected["manual_clip_closure"] = _row_projection(db, "wake_clips_v1",
                " WHERE " + " OR ".join(clauses),
                columns=("id", "bucket", "room_id", "sources_json", "captured_at", "expires_at", "published"))
    elif names & {"stt_corrections_v1", "conversation_turns_v1"}:
        raise ValueError("runtime_admission_missing_manual_closure")
    if "utterances" in names:
        selected["distillation_receipts"] = _row_projection(db, "utterances", " WHERE distilled!=0",
            columns=("id", "capture_id", "conversation_id", "distilled"))
        selected["ambient_deadline_fences"] = _row_projection(db, "utterances",
            " WHERE expires_at < min(t0,created_at)+172800",
            columns=("id", "capture_id", "conversation_id", "t0", "created_at", "expires_at"))
    if "voice_fingerprints" in names:
        selected["voice_deadline_fences"] = _deadline_projection(
            db, "voice_fingerprints", "id", "sources_json", "expires_at")
        if {"voice_groups", "voice_group_members"} <= names:
            selected["manual_group_fingerprints"] = _row_projection(db, "voice_fingerprints",
                " WHERE id IN (SELECT m.fingerprint_id FROM voice_group_members m JOIN voice_groups g "
                "ON g.id=m.group_id WHERE g.state='manual' OR g.reason LIKE 'manual%')")
    if "suggestions_policy_v1" in names:
        selected["suggestion_deadline_fences"] = _deadline_projection(
            db, "suggestions_policy_v1", "id", "sources_json", "expires_at")
    if "events_policy_v1" in names:
        selected["event_text_deadline_fences"] = _deadline_projection(
            db, "events_policy_v1", "id", "sources_json", "text_expires_at")
    if {"facts", "fact_sources"} <= names:
        rows = db.execute("SELECT f.id,f.revision,f.expires_at,MIN(s.expires_at) AS ceiling "
            "FROM facts f JOIN fact_sources s ON s.fact_id=f.id "
            "WHERE f.state IN ('automatic','proposed') GROUP BY f.id")
        selected["fact_deadline_fences"] = {"columns": ["id", "revision", "expires_at"],
            "row_hashes": sorted(_digest([r["id"], r["revision"], r["expires_at"]]) for r in rows
                if r["expires_at"] is not None and r["expires_at"] < r["ceiling"])}
    if "live_deliveries" in names and "sqlite_sequence" in names:
        selected["delivery_sequence_reservation"] = _row_projection(db, "sqlite_sequence",
            " WHERE name='live_deliveries'")


def capture_state_authority(state_root: Path) -> dict:
    """Hash source-owned protected decisions, excluding ordinary payload and SDK bytes.

    Control records retain row digests, not biometric BLOBs or credential material.
    Configuration files are consent/policy inputs; account activity timestamps are
    excluded while role, entity grants, names, preferences and retirement remain.
    """
    state = _safe_directory(Path(state_root).resolve(strict=True))
    projection = {"tables": {}, "accounts": {}, "configuration": {}}
    for path in sorted(state.rglob("*")):
        if path.is_symlink():
            raise ValueError("runtime_admission_state_symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(state).as_posix()
        if path.suffix in {".db", ".sqlite", ".sqlite3"}:
            with _database_image(path) as db:
                names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                contract = _schema_contract(db, names)
                if contract is None:
                    continue
                selected = {name: _row_projection(db, name) for name in sorted(names & _PROTECTED_TABLES)}
                selected["schema_contract_v1"] = contract
                if "facts" in names:
                    condition = " WHERE state IN ('confirmed','deleted','quarantined')"
                    selected["facts"] = _row_projection(db, "facts", condition)
                    for table in ("fact_sources", "fact_dependencies"):
                        if table in names:
                            selected[table] = _row_projection(db, table,
                                " WHERE fact_id IN (SELECT id FROM facts" + condition + ")")
                if "policy_index_deletions" in names:
                    selected["deleted_fact_revisions"] = {"columns": ["fact_id", "revision"],
                        "row_hashes": sorted(_digest(list(row)) for row in db.execute(
                            "SELECT fact_id,MAX(revision) FROM policy_index_deletions GROUP BY fact_id"))}
                if "voice_groups" in names:
                    condition = " WHERE state='manual' OR reason LIKE 'manual%'"
                    selected["voice_groups"] = _row_projection(db, "voice_groups", condition)
                    for table in ("voice_group_members", "voice_group_lineage"):
                        if table in names:
                            selected[table] = _row_projection(db, table,
                                " WHERE group_id IN (SELECT id FROM voice_groups" + condition + ")")
                if "suggestions_policy_v1" in names:
                    selected["suggestions_policy_v1"] = _row_projection(db, "suggestions_policy_v1",
                        " WHERE status!='pending' OR snooze_until IS NOT NULL")
                _mixed_projections(db, names, selected)
                if selected:
                    projection["tables"][relative] = selected
        elif path.name.endswith("_accounts_v1.json"):
            value = _read_json(path)
            if value.get("schema") != "kza-users-accounts-v1" or type(value.get("users")) is not list:
                raise ValueError("runtime_admission_account_schema")
            projection["accounts"][relative] = sorted(_digest({key: item for key, item in user.items()
                if key != "last_seen"}) for user in value["users"])
        elif path == state / ".env" or (path.is_relative_to(state / "config")
                and path.suffix in {".json", ".yaml", ".yml", ".toml"}):
            projection["configuration"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"schema_version": 1, "digest": _digest(projection), "projection": projection}


def _state_layout(state: Path) -> tuple[dict, dict, dict]:
    roots = {".": _identity(state)}
    schemas = {}
    sdk = {}
    for name in ("config", "data"):
        path = state / name
        if path.is_dir():
            roots[name] = _identity(path)
    for path in sorted(state.rglob("*")):
        if path.is_symlink():
            raise ValueError("runtime_admission_state_symlink")
        relative = path.relative_to(state).as_posix()
        if path.is_file() and path.suffix in {".db", ".sqlite", ".sqlite3"}:
            roots[path.parent.relative_to(state).as_posix()] = _identity(path.parent)
            with _database_image(path) as db:
                schemas[relative] = _digest([list(row) for row in db.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")])
                names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "erasure_targets" in names:
                    for (encoded,) in db.execute("SELECT descriptor FROM erasure_targets"):
                        for value in _descriptor_paths(json.loads(encoded)):
                            target = Path(value)
                            if target.is_relative_to(state) and target.is_dir():
                                _safe_directory(target)
                                roots[target.relative_to(state).as_posix()] = _identity(target)
            if path.name == "chroma.sqlite3":
                try:
                    sdk["chromadb"] = importlib.metadata.version("chromadb")
                except importlib.metadata.PackageNotFoundError:
                    raise ValueError("runtime_admission_sdk_version_unavailable") from None
    return roots, schemas, sdk


def _assert_writers_closed(state: Path) -> None:
    descriptors = []
    try:
        for path in sorted(state.rglob("*")):
            if path.is_file() and path.name.endswith((".lock", ".lease")):
                descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
                descriptors.append(descriptor)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError("runtime_admission_writer_active") from None
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _slot_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Invalid slot name")
    return name


@contextmanager
def _locked_root(root: Path):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / ".stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _read_archive(archive: Path) -> tuple[dict, dict[str, bytes]]:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Runtime staging requires CPython 3.13")
    files = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            path = PurePosixPath(member.name)
            if (
                not member.isfile() or path.is_absolute() or ".." in path.parts
                or str(path) != member.name or member.name in files
                or path.suffix == ".py"
                or (member.name != "release.json" and path.parts[0] not in {"src", "scripts", "tools"})
            ):
                raise ValueError("Unexpected or unsafe runtime archive member")
            files[member.name] = bundle.extractfile(member).read()
    manifest = json.loads(files.pop("release.json"))
    _validate_manifest(manifest, files)
    return manifest, files


def _validate_manifest(manifest: dict, files: dict[str, bytes]) -> None:
    if (
        manifest.get("format") != "kza-cpython-bytecode-v1"
        or manifest.get("python") != "3.13"
        or manifest.get("python_magic") != importlib.util.MAGIC_NUMBER.hex()
        or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("git_commit", ""))
        or manifest.get("external_paths") != ["config", "data", "models", "logs"]
        or not {"src/main.pyc", "src/__init__.pyc"} <= files.keys()
        or manifest.get("files") != {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        }
    ):
        raise ValueError("Runtime manifest or file integrity mismatch")


def _copy_private_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError("State source must contain regular config and data directories")
    for entry in source.rglob("*"):
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            raise ValueError("State links and special files require separate review")
    shutil.copytree(source, destination)
    destination.chmod(0o700)


def prepare_slot(archive: Path, root: Path, name: str, state_source: Path) -> dict:
    """Verify an archive and copy offline config/data into a new private slot.

    Args:
        archive: Archive produced by build_runtime.py.
        root: Explicit staging root; no production path is assumed.
        name: New slot name, independent of the commit (allows repeated trials).
        state_source: Offline directory containing config/ and data/. Optional
            .env is copied privately. Models, logs and dependencies stay external.

    Returns:
        Runtime manifest, without the private state's contents.
    """
    name = _slot_name(name)
    root = root.resolve()
    state_source = state_source.resolve()
    # Do not recurse through an output nested in its own input state.
    if root.is_relative_to(state_source):
        raise ValueError("Staging root and state source must be separate trees")
    manifest, files = _read_archive(archive)
    with _locked_root(root):
        slots = root / "slots"
        slots.mkdir(exist_ok=True, mode=0o700)
        target = slots / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=slots) as temporary:
            staging = Path(temporary)
            runtime = staging / "runtime"
            runtime.mkdir()
            for filename, data in files.items():
                destination = runtime / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            (runtime / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
            state = staging / "state"
            state.mkdir(mode=0o700)
            for directory in ("config", "data"):
                _copy_private_tree(state_source / directory, state / directory)
                (runtime / directory).symlink_to(Path("..") / "state" / directory)
            env_file = state_source / ".env"
            if env_file.exists() or env_file.is_symlink():
                if env_file.is_symlink() or not env_file.is_file():
                    raise ValueError("Environment file must be regular")
                shutil.copyfile(env_file, state / ".env")
                (state / ".env").chmod(0o600)
                (runtime / ".env").symlink_to(Path("..") / "state" / ".env")
            os.rename(staging, target)
    return manifest


def _checked_slot(root: Path, name: str) -> tuple[Path, Path, dict]:
    """Validate installed code and same-slot roots without selecting the slot."""
    name = _slot_name(name)
    target = root / "slots" / name
    if target.resolve(strict=True) != target:
        raise ValueError("Slot must not redirect to another directory")
    runtime = target / "runtime"
    manifest = _read_json(runtime / "release.json")
    files = {}
    for filename in manifest["files"]:
        path = PurePosixPath(filename)
        if path.is_absolute() or ".." in path.parts or path.parts[0] not in {"src", "scripts", "tools"}:
            raise ValueError("Unsafe installed runtime path")
        installed = runtime / filename
        if installed.resolve() != installed or installed.suffix == ".py":
            raise ValueError("Installed runtime integrity mismatch")
        files[filename] = installed.read_bytes()
    _validate_manifest(manifest, files)
    expected = set(files) | {"release.json", "config", "data"}
    if (runtime / ".env").is_symlink():
        expected.add(".env")
        if (runtime / ".env").resolve() != target / "state" / ".env":
            raise ValueError("Environment does not belong to the same slot")
    actual = {str(path.relative_to(runtime)) for path in runtime.rglob("*")
              if path.is_file() or path.is_symlink()}
    if actual != expected:
        raise ValueError("Unexpected installed runtime files")
    for directory in ("config", "data"):
        if ((runtime / directory).resolve() != target / "state" / directory
                or not (runtime / directory).is_dir()):
            raise ValueError("Runtime and state do not belong to the same slot")
    return runtime, _safe_directory(target / "state"), manifest


def _descriptor_paths(value: object):
    if type(value) is dict:
        for key, child in value.items():
            if key in {"root", "path", "db_path", "audio_dir", "base_dir", "storage_path",
                       "persist_directory", "clip_root", "contexts_dir", "directory"}:
                if type(child) is not str or not Path(child).is_absolute():
                    raise ValueError("runtime_admission_relative_target")
            yield from _descriptor_paths(child)
    elif type(value) is list:
        for child in value:
            yield from _descriptor_paths(child)
    elif type(value) is str and Path(value).is_absolute():
        yield value


def _target_inventory(state: Path) -> dict:
    targets = {}
    for path in sorted(state.rglob("*")):
        if path.is_file() and path.suffix in {".db", ".sqlite", ".sqlite3"}:
            with _database_image(path) as db:
                names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "erasure_targets" in names:
                    for row in db.execute("SELECT id,participant,version,descriptor FROM erasure_targets"):
                        value = dict(row)
                        value["descriptor"] = json.loads(value["descriptor"])
                        _validate_target_record(value)
                        if row["id"] in targets and targets[row["id"]] != value:
                            raise ValueError("runtime_admission_target_collision")
                        targets[row["id"]] = value
    return targets


def _validate_target_record(target: dict) -> None:
    participant, version, descriptor = target["participant"], target["version"], target["descriptor"]
    versions = {"producers": {1}, "captures": {1}, "contexts": {1}, "semantic_index": {1},
                "events": {1}, "diagnostics": {1, 2}, "collectors": {1},
                "streams": {1, 2, 3, 4}, "suggestions": {1}, "voices": {1, 2}}
    if (participant not in versions or type(version) is not int or version not in versions[participant]
            or type(descriptor) is not dict):
        raise ValueError("runtime_admission_unsupported_target")
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected = hashlib.sha256(json.dumps([participant, version, encoded]).encode()).hexdigest()
    if target["id"] != expected:
        raise ValueError("runtime_admission_target_identity")
    paths = tuple(_descriptor_paths(descriptor))
    if paths:
        keys = set(descriptor)
        supported = (
            participant == "captures" and keys == {"db_path", "audio_dir"}
            or participant == "contexts" and keys == {"kind", "path", "format"}
            and descriptor["kind"] == "snapshots" and descriptor["format"] in (2, 3)
            or participant == "semantic_index" and keys == {"path", "collection"}
            and type(descriptor["collection"]) is str and descriptor["collection"].endswith("_policy_v1")
            or participant in {"events", "suggestions"} and keys == {"path", "table"}
            and descriptor["table"] == ("events_policy_v1" if participant == "events" else "suggestions_policy_v1")
            or participant == "diagnostics" and version == 2 and keys == {"kind", "path"}
            and descriptor["kind"] == "numeric_catalog"
            or participant == "collectors" and keys == {"kind", "root"}
            and descriptor["kind"] in {"wake_clips", "stt_corrections", "conversations",
                                        "conversation_exports", "legacy_training"}
            or participant == "collectors" and keys == {"kind", "root", "batch_id"}
            and descriptor["kind"] == "legacy_quarantine" and type(descriptor["batch_id"]) is str
        )
        if not supported:
            raise ValueError("runtime_admission_unsupported_target")
    else:
        literals = {
            ("producers", 1): [{"kind": "runtime_producers", "protocol": 1}],
            ("contexts", 1): [{"kind": "runtime", "name": "conversation_buffers"}],
            ("suggestions", 1): [{"kind": "runtime", "name": "automation_proposals"}],
            ("diagnostics", 1): [{"kind": "unresolved_diagnostic_inventory"}],
            ("streams", 1): [{"kind": "live_browser_receipts", "protocol": 1}],
            ("streams", 2): [{"kind": "unresolved_stream_inventory"}],
            ("streams", 3): [{"kind": "conversation_rest_and_live_receipts", "protocol": 2}],
            ("streams", 4): [{"kind": "voice_review_rest_and_live_receipts", "protocol": 3}],
            ("voices", 1): [{"kind": "source_bound_voice_registry", "schema": 1}],
            ("voices", 2): [{"kind": "unresolved_voice_inventory"}],
        }
        if descriptor not in literals.get((participant, version), []) and not (
                participant == "collectors" and version == 1
                and set(descriptor) == {"kind", "batch_id"}
                and descriptor["kind"] == "quarantine_locator_coverage"
                and type(descriptor["batch_id"]) is str):
            raise ValueError("runtime_admission_unsupported_target")


def _binding_snapshot(root: Path, name: str) -> dict:
    runtime, state, manifest = _checked_slot(root, name)
    _assert_writers_closed(state)
    roots, schemas, versions = _state_layout(state)
    ownership = state / ".kza-target-ownership.json"
    if ownership.is_symlink():
        raise ValueError("runtime_admission_ownership_redirected")
    return {"schema_version": 1, "slot": name, "runtime_root": str(runtime), "state_root": str(state),
            "release_sha256": hashlib.sha256((runtime / "release.json").read_bytes()).hexdigest(),
            "code_commit": manifest["git_commit"], "python_magic_hex": manifest["python_magic"],
            "state_identity": _identity(state), "store_roots": roots, "schemas": schemas,
            "sdk_versions": versions, "authority": capture_state_authority(state),
            "ownership_sha256": hashlib.sha256(ownership.read_bytes()).hexdigest() if ownership.exists() else None}


def _store_binding(root: Path, binding: dict) -> str:
    identity = _digest(binding)
    _atomic_record(root / "admission" / "bindings" / (identity + ".json"), binding, immutable=True)
    return identity


def _write_marker(root: Path, identity: str, binding: dict) -> None:
    _atomic_record(Path(binding["state_root"]) / _MARKER,
        {"schema_version": 1, "admission_root": str(root), "binding_id": identity,
         "state_root": binding["state_root"]})


def _check_roots(binding: dict) -> None:
    state = _safe_directory(Path(binding["state_root"]))
    if _identity(state) != binding["state_identity"]:
        raise ValueError("runtime_admission_state_replaced")
    for relative, expected in binding["store_roots"].items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("runtime_admission_store_path")
        directory = _safe_directory(state / path)
        if _identity(directory) != expected:
            raise ValueError("runtime_admission_store_replaced")


def _closure_proof(root: Path, binding_id: str) -> dict:
    proof = _read_json(root / "admission" / "closures" / (binding_id + ".json"))
    if (set(proof) != {"schema_version", "binding_id", "running_head", "receipt_sha256", "child_pid"}
            or proof["schema_version"] != 1 or proof["binding_id"] != binding_id
            or type(proof["running_head"]) is not dict
            or set(proof["running_head"]) != _HEAD_KEYS
            or proof["running_head"]["status"] != "running"):
        raise ValueError("runtime_admission_source_close_proof")
    running = proof["running_head"]
    receipt_path = root / "admission" / "receipts" / (running["run_id"] + ".json")
    receipt = _read_json(receipt_path)
    expected = {"schema_version": 1, "binding_id": running["binding_id"],
                "generation": running["generation"], "run_id": running["run_id"],
                "child_pid": proof["child_pid"], "status": "closed",
                "resources": dict.fromkeys(_CLOSED_RESOURCES, True)}
    if (receipt != expected
            or hashlib.sha256(receipt_path.read_bytes()).hexdigest() != proof["receipt_sha256"]):
        raise ValueError("runtime_admission_source_close_proof")
    return proof


def _delegation_records(root: Path, state: Path):
    directory = root / "admission" / "delegations"
    if not directory.exists():
        return
    _safe_directory(directory)
    for path in sorted(directory.glob("*.json")):
        record = _read_json(path)
        if path.stem != _digest(record):
            raise ValueError("runtime_admission_delegation_changed")
        if record.get("destination_state") == str(state):
            yield record


def _validate_delegated_target(root: Path, state: Path, target: dict, sources: list) -> None:
    for record in _delegation_records(root, state):
        if (set(record) != {"schema_version", "destination_state", "destination_identity",
                           "source_binding", "source_close_sha256", "targets"}
                or record["schema_version"] != 1 or type(record["targets"]) is not dict):
            raise ValueError("runtime_admission_delegation_schema")
        if record["destination_identity"] != _identity(state) or record["source_binding"] not in sources:
            continue
        if target["id"] not in record["targets"]:
            continue
        if record["targets"][target["id"]] != target:
            raise ValueError("runtime_admission_delegation_target_changed")
        source = read_binding(root, record["source_binding"])
        marker = _read_json(Path(source["state_root"]) / _MARKER)
        if marker.get("binding_id") != record["source_binding"]:
            continue
        if _binding_snapshot(root, source["slot"]) != source:
            continue
        proof = _closure_proof(root, record["source_binding"])
        if _digest(proof) != record["source_close_sha256"]:
            continue
        source_targets = _target_inventory(Path(source["state_root"]))
        if source_targets.get(target["id"]) != target:
            raise ValueError("runtime_admission_delegation_source_target_changed")
        for value in _descriptor_paths(target["descriptor"]):
            path = Path(value)
            if not path.is_relative_to(Path(source["state_root"])) or path.resolve() != path:
                raise ValueError("runtime_admission_delegation_foreign_source")
        return
    raise ValueError("runtime_admission_delegation_missing")


def validate_target_ownership(root: Path, binding: dict, *, allow_new_local: bool = False) -> None:
    """Reject foreign active paths; historical journals confer no execution grant."""
    root = _safe_directory(Path(root).resolve(strict=True))
    state = _safe_directory(Path(binding["state_root"]))
    targets = _target_inventory(state)
    path = state / ".kza-target-ownership.json"
    if binding["ownership_sha256"] is None:
        if path.exists() or path.is_symlink():
            raise ValueError("runtime_admission_ownership_changed")
        owned = set(targets)
    else:
        policy = _read_json(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding["ownership_sha256"]:
            raise ValueError("runtime_admission_ownership_changed")
        if (set(policy) != {"schema_version", "source_bindings", "owned_targets",
                           "delegated_targets", "historical_targets"}
                or policy["schema_version"] != 1
                or any(type(policy[key]) is not list for key in
                       ("source_bindings", "owned_targets", "delegated_targets", "historical_targets"))):
            raise ValueError("runtime_admission_ownership_schema")
        sources, owned_ids = policy["source_bindings"], policy["owned_targets"]
        if (any(type(item) is not str or re.fullmatch("[0-9a-f]{64}", item) is None for item in sources)
                or len(set(sources)) != len(sources)
                or any(type(item) is not str for item in owned_ids)
                or len(set(owned_ids)) != len(owned_ids)):
            raise ValueError("runtime_admission_ownership_schema")
        for source in sources:
            read_binding(root, source)
        delegated_ids = policy["delegated_targets"]
        if (any(type(item) is not str for item in delegated_ids)
                or len(set(delegated_ids)) != len(delegated_ids)):
            raise ValueError("runtime_admission_ownership_schema")
        delegated = set(delegated_ids)
        owned = set(owned_ids)
        if owned & delegated or not (owned | delegated) <= set(targets):
            raise ValueError("runtime_admission_active_inventory_mismatch")
        extras = set(targets) - owned - delegated
        if extras and not allow_new_local:
            raise ValueError("runtime_admission_active_inventory_mismatch")
        owned |= extras
        for identity in delegated:
            _validate_delegated_target(root, state, targets[identity], sources)
        seen = set()
        for item in policy["historical_targets"]:
            if (type(item) is not dict or set(item) != {"source_binding", "target_id", "archive_sha256"}
                    or item["source_binding"] not in sources or type(item["target_id"]) is not str
                    or type(item["archive_sha256"]) is not str
                    or re.fullmatch("[0-9a-f]{64}", item["archive_sha256"]) is None):
                raise ValueError("runtime_admission_history_schema")
            key = (item["source_binding"], item["target_id"], item["archive_sha256"])
            if key in seen:
                raise ValueError("runtime_admission_history_duplicate")
            seen.add(key)
            archive_path = root / "admission" / "history" / (item["archive_sha256"] + ".json")
            archive = _read_json(archive_path)
            if (hashlib.sha256(archive_path.read_bytes()).hexdigest() != item["archive_sha256"]
                    or archive.get("schema_version") != 1
                    or archive.get("source_binding") != item["source_binding"]
                    or type(archive.get("targets")) is not list
                    or not any(type(target) is dict and target.get("id") == item["target_id"]
                               for target in archive["targets"])):
                raise ValueError("runtime_admission_history_mismatch")
    for identity in owned:
        for value in _descriptor_paths(targets[identity]["descriptor"]):
            path = Path(value)
            if not path.is_relative_to(state) or path.resolve() != path:
                raise ValueError("runtime_admission_foreign_target")


def _manual_module():
    """Load the stdlib contract only in parent paths, outside V1 bridges."""
    global _manual_components_module
    if _manual_components_module is None:
        own = Path(__file__).resolve()
        suffix = ".pyc" if own.suffix == ".pyc" else ".py"
        path = own.parents[1] / ("src/memory/manual_components" + suffix)
        if path.is_symlink() or not path.is_file():
            raise ValueError("runtime_admission_manual_contract_missing")
        name = "_kza_manual_components_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError("runtime_admission_manual_contract_missing")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        _manual_components_module = module
    return _manual_components_module


def _manual_owner(root: Path, binding: dict, target: dict):
    module = _manual_module()
    state = Path(binding["state_root"])
    descriptor = target["descriptor"]
    directory = Path(descriptor["root"])
    owner = binding
    if not directory.is_relative_to(state):
        policy = _read_json(state / ".kza-target-ownership.json")
        if target["id"] not in policy["delegated_targets"]:
            raise ValueError("privacy_authority_stale")
        _validate_delegated_target(root, state, target, policy["source_bindings"])
        candidates = {}
        for record in _delegation_records(root, state):
            if (record["source_binding"] not in policy["source_bindings"]
                    or record["destination_identity"] != _identity(state)
                    or record["targets"].get(target["id"]) != target):
                continue
            source = read_binding(root, record["source_binding"])
            source_state = Path(source["state_root"])
            if (not directory.is_relative_to(source_state)
                    or _binding_snapshot(root, source["slot"]) != source
                    or _digest(_closure_proof(root, record["source_binding"])) != record["source_close_sha256"]
                    or _target_inventory(source_state).get(target["id"]) != target):
                continue
            if _read_json(source_state / _MARKER).get("binding_id") == record["source_binding"]:
                candidates[record["source_binding"]] = source
        if len(candidates) != 1:
            raise ValueError("runtime_admission_manual_owner_ambiguous")
        owner = next(iter(candidates.values()))
    return module.CollectorOwner(target["id"], descriptor["kind"], directory,
        Path(owner["state_root"]), owner["state_identity"], _digest(owner))


def _manual_capture(root: Path, binding: dict) -> dict:
    """Read typed collector rows and explicitly owned files, never SDK data."""
    module = _manual_module()
    validate_target_ownership(root, binding)
    state = Path(binding["state_root"])
    tables = ("wake_clips_v1", "stt_corrections_v1", "conversation_turns_v1",
              "conversation_export_parents_v1")
    stores = {}
    for relative, projection in binding["authority"]["projection"]["tables"].items():
        names = set(projection["schema_contract_v1"]["columns"])
        if not names.intersection(tables):
            continue
        if not set(tables) <= names:
            raise ValueError("runtime_admission_manual_schema")
        with _database_image(state / relative) as db:
            stores[relative] = {name: [dict(row) for row in db.execute("SELECT * FROM " + _quote(name))]
                                for name in tables}
    inventory = _target_inventory(state)
    used = set()
    for store in stores.values():
        protected = module.protected_collector_records(store)
        used.update(row["target_id"] for row in store["wake_clips_v1"] if row["id"] in protected)
    owners = {}
    for identity in used:
        target = inventory.get(identity)
        if (target is None or target["participant"] != "collectors" or target["version"] != 1
                or set(target["descriptor"]) != {"kind", "root"}
                or target["descriptor"]["kind"] not in {
                    "wake_clips", "stt_corrections", "conversations", "conversation_exports"}):
            raise ValueError("runtime_admission_manual_owner_missing")
        owners[identity] = _manual_owner(root, binding, target)
    return module.capture_manual_components(state, stores, owners)


def _manual_mapping(source: dict, candidate: dict) -> dict:
    """Derive actual target substitutions from two accredited captures."""
    if set(source["stores"]) != set(candidate["stores"]):
        raise ValueError("privacy_authority_stale")
    mapping = {}
    for relative, store in source["stores"].items():
        before, after = store["clips"], candidate["stores"][relative]["clips"]
        if set(before) != set(after):
            raise ValueError("privacy_authority_stale")
        for identity, row in before.items():
            old, new = row["locator"], after[identity]["locator"]
            if any(old[key] != new[key] for key in ("kind", "relative_root")):
                raise ValueError("privacy_authority_stale")
            if old["target_id"] != new["target_id"]:
                if old["target_id"] in mapping and mapping[old["target_id"]] != new["target_id"]:
                    raise ValueError("privacy_authority_stale")
                mapping[old["target_id"]] = new["target_id"]
    return mapping


def _require_manual_equivalent(source: dict, candidate: dict) -> None:
    module = _manual_module()
    try:
        equivalent = module.equivalent_manual_components(source, candidate,
            target_relocations=_manual_mapping(source, candidate))
    except module.ManualComponentError:
        raise ValueError("privacy_authority_stale") from None
    if not equivalent:
        raise ValueError("privacy_authority_stale")


def _manual_head_path(root: Path, head: dict) -> Path:
    return root / "admission" / "manual-components" / "heads" / (_digest(head) + ".json")


def _manual_head_shape(head: dict, status: str) -> bool:
    return (type(head) is dict and set(head) == _HEAD_KEYS
            and type(head["schema_version"]) is int and head["schema_version"] == 1
            and type(head["generation"]) is int and head["generation"] >= 1
            and type(head["binding_id"]) is str and re.fullmatch("[0-9a-f]{64}", head["binding_id"]) is not None
            and head["status"] == status
            and (head["run_id"] is None if status == "closed" else type(head["run_id"]) is str
                 and re.fullmatch("[0-9a-f]{32}", head["run_id"]) is not None))


def _manual_receipt_proof(root: Path, proof: dict) -> None:
    """Verify retained evidence without asserting a new wait or native close."""
    if (type(proof) is not dict
            or set(proof) != {"schema_version", "binding_id", "running_head", "receipt_sha256", "child_pid"}
            or type(proof["schema_version"]) is not int or proof["schema_version"] != 1
            or type(proof["child_pid"]) is not int or proof["child_pid"] <= 0):
        raise ValueError("runtime_admission_manual_close_proof")
    running = proof["running_head"]
    if not _manual_head_shape(running, "running") or running["generation"] < 2:
        raise ValueError("runtime_admission_manual_close_proof")
    path = root / "admission" / "receipts" / (running["run_id"] + ".json")
    expected = {"schema_version": 1, "binding_id": running["binding_id"],
                "generation": running["generation"], "run_id": running["run_id"],
                "child_pid": proof["child_pid"], "status": "closed",
                "resources": dict.fromkeys(_CLOSED_RESOURCES, True)}
    if (_canonical_json(_read_json(path)) != _canonical_json(expected)
            or hashlib.sha256(path.read_bytes()).hexdigest() != proof["receipt_sha256"]):
        raise ValueError("runtime_admission_manual_close_proof")


def _manual_closed_head(proof: dict) -> dict:
    return {"schema_version": 1, "generation": proof["running_head"]["generation"] + 1,
            "binding_id": proof["binding_id"], "status": "closed", "run_id": None}


def _read_manual_seal(root: Path, identity: str) -> dict:
    if type(identity) is not str or re.fullmatch("[0-9a-f]{64}", identity) is None:
        raise ValueError("runtime_admission_manual_seal_schema")
    record = _read_json(root / "admission" / "manual-components" / "seals" / (identity + ".json"))
    if (set(record) != {"schema_version", "binding_id", "closure_proof", "closed_head", "capture"}
            or type(record["schema_version"]) is not int or record["schema_version"] != 1
            or _digest(record) != identity
            or type(record["closure_proof"]) is not dict or type(record["capture"]) is not dict):
        raise ValueError("runtime_admission_manual_seal_schema")
    _manual_receipt_proof(root, record["closure_proof"])
    if (record["closure_proof"]["binding_id"] != record["binding_id"]
            or not _manual_head_shape(record["closed_head"], "closed")
            or _canonical_json(record["closed_head"]) != _canonical_json(_manual_closed_head(record["closure_proof"]))):
        raise ValueError("runtime_admission_manual_seal_schema")
    binding = read_binding(root, record["binding_id"])
    if (record["capture"].get("state_root") != binding["state_root"]
            or record["capture"].get("state_identity") != binding["state_identity"]):
        raise ValueError("runtime_admission_manual_seal_schema")
    return record


def _read_manual_head(root: Path, head: dict) -> dict:
    if not _manual_head_shape(head, "closed"):
        raise ValueError("runtime_admission_manual_companion_schema")
    try:
        record = _read_json(_manual_head_path(root, head))
    except FileNotFoundError:
        raise ValueError("runtime_admission_manual_legacy_unsealed") from None
    if (set(record) != {"schema_version", "head", "kind", "capture", "authority_seal", "previous"}
            or type(record["schema_version"]) is not int or record["schema_version"] != 1
            or _canonical_json(record["head"]) != _canonical_json(head)
            or head["status"] != "closed" or head["run_id"] is not None
            or type(record["capture"]) is not dict
            or record["kind"] not in {"bootstrap", "selection", "closure"}):
        raise ValueError("runtime_admission_manual_companion_schema")
    binding = read_binding(root, head["binding_id"])
    if (record["capture"].get("state_root") != binding["state_root"]
            or record["capture"].get("state_identity") != binding["state_identity"]):
        raise ValueError("runtime_admission_manual_companion_schema")
    if record["kind"] == "bootstrap":
        if (head["generation"] != 1 or record["previous"] is not None
                or record["authority_seal"] is not None or _has_authority(binding)
                or record["capture"].get("stores") != {}):
            raise ValueError("runtime_admission_manual_bootstrap_invalid")
        return record
    previous = record["previous"]
    if (type(previous) is not dict or set(previous) != {"head_sha256", "companion_sha256"}
            or any(type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None
                   for value in previous.values())):
        raise ValueError("runtime_admission_manual_companion_schema")
    path = root / "admission" / "manual-components" / "heads" / (previous["head_sha256"] + ".json")
    prior = _read_json(path)
    if (type(prior.get("schema_version")) is not int or prior["schema_version"] != 1
            or not _manual_head_shape(prior.get("head"), "closed")
            or _digest(prior) != previous["companion_sha256"] or _digest(prior["head"]) != previous["head_sha256"]):
        raise ValueError("runtime_admission_manual_companion_changed")
    if record["kind"] == "selection":
        if (prior["head"]["generation"] + 1 != head["generation"]
                or record["authority_seal"] != prior["authority_seal"]):
            raise ValueError("runtime_admission_manual_companion_changed")
        if record["authority_seal"] is None:
            if prior["kind"] not in {"bootstrap", "selection"} or _has_authority(binding):
                raise ValueError("runtime_admission_manual_bootstrap_invalid")
            _require_manual_equivalent(prior["capture"], record["capture"])
            return record
    seal = _read_manual_seal(root, record["authority_seal"])
    if record["kind"] == "closure":
        running = seal["closure_proof"]["running_head"]
        if (_canonical_json(seal["closed_head"]) != _canonical_json(head)
                or _canonical_json(seal["capture"]) != _canonical_json(record["capture"])
                or running["generation"] != prior["head"]["generation"] + 1
                or running["binding_id"] != prior["head"]["binding_id"]):
            raise ValueError("runtime_admission_manual_companion_changed")
    _require_manual_equivalent(seal["capture"], record["capture"])
    return record


def _verify_manual_head(root: Path, head: dict, binding: dict) -> dict:
    record = _read_manual_head(root, head)
    if _digest(binding) != head["binding_id"]:
        raise ValueError("privacy_authority_stale")
    module = _manual_module()
    try:
        module.verify_manual_components(_manual_capture(root, binding), record["capture"])
    except module.ManualComponentError:
        raise ValueError("privacy_authority_stale") from None
    if _binding_snapshot(root, binding["slot"]) != binding:
        raise ValueError("privacy_authority_stale")
    return record


def _prepare_manual_head(root: Path, expected: dict | None, binding: dict, *,
                         previous: dict | None, proof: dict | None = None) -> dict:
    """Write immutable metadata before its corresponding V1 CLOSED CAS."""
    if expected is not None and not _manual_head_shape(expected, "running" if proof is not None else "closed"):
        raise ValueError("runtime_admission_manual_companion_schema")
    head = {"schema_version": 1, "generation": 1 if expected is None else expected["generation"] + 1,
            "binding_id": _digest(binding), "status": "closed", "run_id": None}
    capture = _manual_capture(root, binding)
    if _binding_snapshot(root, binding["slot"]) != binding:
        raise ValueError("privacy_authority_stale")
    reference = None if previous is None else {
        "head_sha256": _digest(previous["head"]), "companion_sha256": _digest(previous)}
    if proof is not None:
        _manual_receipt_proof(root, proof)
        if (_canonical_json(head) != _canonical_json(_manual_closed_head(proof)) or previous is None
                or _canonical_json(expected) != _canonical_json(proof["running_head"])):
            raise ValueError("runtime_admission_manual_close_proof")
        seal = {"schema_version": 1, "binding_id": _digest(binding), "closure_proof": proof,
                "closed_head": head, "capture": capture}
        seal_id = _digest(seal)
        _atomic_record(root / "admission" / "manual-components" / "seals" / (seal_id + ".json"), seal, immutable=True)
        kind = "closure"
    elif previous is None:
        if expected is not None or _has_authority(binding) or capture["stores"]:
            raise ValueError("runtime_admission_manual_legacy_unsealed")
        kind, seal_id = "bootstrap", None
    else:
        if _canonical_json(expected) != _canonical_json(previous["head"]):
            raise ValueError("runtime_admission_head_changed")
        _require_manual_equivalent(previous["capture"], capture)
        kind, seal_id = "selection", previous["authority_seal"]
    record = {"schema_version": 1, "head": head, "kind": kind, "capture": capture,
              "authority_seal": seal_id, "previous": reference}
    _atomic_record(_manual_head_path(root, head), record, immutable=True)
    return record


def _publish_selection(root: Path, name: str) -> str | None:
    current = root / "current"
    previous = os.readlink(current) if current.is_symlink() else None
    with tempfile.TemporaryDirectory(prefix=".activate-", dir=root) as temporary:
        link = Path(temporary) / "current"
        link.symlink_to(Path("slots") / name / "runtime")
        os.replace(link, current)
    return previous


def _has_authority(binding: dict) -> bool:
    projection = binding["authority"]["projection"]
    return bool(projection["tables"] or projection["accounts"])


def _require_root_lease(root: Path) -> None:
    path = root / ".admission.lock"
    header = _read_json(path)
    if (set(header) != {"controller_pid", "lease_fd"} or header["controller_pid"] != os.getpid()
            or type(header["lease_fd"]) is not int):
        raise ValueError("runtime_admission_controller_lease_required")
    info = os.fstat(header["lease_fd"])
    if not stat.S_ISREG(info.st_mode) or {"device": info.st_dev, "inode": info.st_ino} != _identity(path):
        raise ValueError("runtime_admission_controller_lease_required")
    try:
        fcntl.flock(header["lease_fd"], fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError("runtime_admission_controller_lease_required") from None


def verify_closed_binding(root: Path, binding_id: str) -> dict:
    """Verify a closed source while the caller owns exclusive root admission.

    Historical bindings are permitted only while their physical roots, schemas,
    protected decisions and actual closure witness still match exactly.
    """
    root = _safe_directory(Path(root).resolve(strict=True))
    _require_root_lease(root)
    head = read_admission_head(root)
    if head is None or head["status"] != "closed":
        raise ValueError("runtime_admission_not_closed")
    binding = read_binding(root, binding_id)
    if _binding_snapshot(root, binding["slot"]) != binding:
        raise ValueError("privacy_authority_stale")
    marker = _read_json(Path(binding["state_root"]) / _MARKER)
    if marker != {"schema_version": 1, "admission_root": str(root),
                  "binding_id": binding_id, "state_root": binding["state_root"]}:
        raise ValueError("runtime_admission_marker_mismatch")
    _closure_proof(root, binding_id)
    validate_target_ownership(root, binding)
    _verify_manual_head(root, head, read_binding(root, head["binding_id"]))
    closed_head = _manual_closed_head(_closure_proof(root, binding_id))
    record = _verify_manual_head(root, closed_head, binding)
    if record["kind"] != "closure":
        raise ValueError("runtime_admission_manual_close_proof")
    return binding


def _seal_local_targets(root: Path, binding: dict) -> None:
    """Adopt only locally verified registrations after actual owned closure."""
    validate_target_ownership(root, binding, allow_new_local=True)
    if binding["ownership_sha256"] is None:
        return
    state = Path(binding["state_root"])
    path = state / ".kza-target-ownership.json"
    policy = _read_json(path)
    policy["owned_targets"] = sorted(set(_target_inventory(state)) - set(policy["delegated_targets"]))
    _atomic_record(path, policy)


def _authorize_target_delegation(root: Path, destination_state: Path, source_binding: str,
                                 target_ids: list[str]) -> str:
    head = read_admission_head(root)
    if head is None or head["status"] != "closed" or head["binding_id"] != source_binding:
        raise ValueError("runtime_admission_delegation_source_not_current")
    source = verify_closed_binding(root, source_binding)
    source_manual = _verify_manual_head(root, head, source)
    proof = _closure_proof(root, source_binding)
    state = _safe_directory(Path(destination_state).resolve(strict=True))
    if state.parent.parent != root / "slots" or state.name != "state" or str(state) == source["state_root"]:
        raise ValueError("runtime_admission_delegation_destination")
    _, checked_state, _ = _checked_slot(root, state.parent.name)
    if checked_state != state:
        raise ValueError("runtime_admission_delegation_destination")
    _assert_writers_closed(state)
    if (type(target_ids) not in {list, tuple} or not target_ids
            or any(type(item) is not str for item in target_ids)
            or len(set(target_ids)) != len(target_ids)):
        raise ValueError("runtime_admission_delegation_targets")
    source_targets = _target_inventory(Path(source["state_root"]))
    destination_targets = _target_inventory(state)
    targets = {}
    for identity in target_ids:
        target = source_targets.get(identity)
        if target is None or destination_targets.get(identity) != target:
            raise ValueError("runtime_admission_delegation_target_mismatch")
        for value in _descriptor_paths(target["descriptor"]):
            path = Path(value)
            if not path.is_relative_to(Path(source["state_root"])) or path.resolve() != path:
                raise ValueError("runtime_admission_delegation_foreign_source")
        targets[identity] = target
    policy_path = state / ".kza-target-ownership.json"
    if policy_path.exists() or policy_path.is_symlink():
        policy = _read_json(policy_path)
        if (set(policy) != {"schema_version", "source_bindings", "owned_targets",
                           "delegated_targets", "historical_targets"} or policy["schema_version"] != 1
                or any(type(policy[key]) is not list for key in
                       ("source_bindings", "owned_targets", "delegated_targets", "historical_targets"))):
            raise ValueError("runtime_admission_ownership_schema")
    else:
        policy = {"schema_version": 1, "source_bindings": [], "owned_targets": [],
                  "delegated_targets": [], "historical_targets": []}
    delegated = set(policy["delegated_targets"]) | set(targets)
    owned = set(destination_targets) - delegated
    for identity in owned:
        for value in _descriptor_paths(destination_targets[identity]["descriptor"]):
            path = Path(value)
            if not path.is_relative_to(state) or path.resolve() != path:
                raise ValueError("runtime_admission_foreign_target")
    record = {"schema_version": 1, "destination_state": str(state),
              "destination_identity": _identity(state), "source_binding": source_binding,
              "source_close_sha256": _digest(proof), "targets": targets}
    capability = _digest(record)
    _atomic_record(root / "admission" / "delegations" / (capability + ".json"), record, immutable=True)
    policy["source_bindings"] = sorted(set(policy["source_bindings"]) | {source_binding})
    policy["owned_targets"] = sorted(owned)
    policy["delegated_targets"] = sorted(delegated)
    _atomic_record(policy_path, policy)
    destination = _binding_snapshot(root, state.parent.name)
    validate_target_ownership(root, destination)
    _prepare_manual_head(root, head, source, previous=source_manual)
    _cas_head(root, head, source_binding, "closed")
    return capability


def _current_slot(root: Path) -> str | None:
    current = root / "current"
    if not current.is_symlink():
        if current.exists():
            raise ValueError("runtime_admission_current_not_link")
        return None
    resolved = current.resolve(strict=True)
    if resolved.parent.parent != root / "slots" or resolved.name != "runtime":
        raise ValueError("runtime_admission_current_redirected")
    return _slot_name(resolved.parent.name)


def _remapped_suggestion_projection(source_columns: list, source_rows: list,
                                    candidate_columns: list, candidate_rows: list,
                                    evidence_targets: dict, source_events: dict,
                                    candidate_events: dict) -> dict:
    """Pure exact comparison with one typed locator substitution, no policy edit."""
    if source_columns != candidate_columns:
        raise ValueError("privacy_authority_stale")
    if (type(evidence_targets) is not dict or any(type(key) is not str or type(value) is not str
            or re.fullmatch("[0-9a-f]{64}", key) is None or re.fullmatch("[0-9a-f]{64}", value) is None
            for key, value in evidence_targets.items())):
        raise ValueError("privacy_authority_stale")
    columns = set(item.split(":")[0] for item in _TABLE_COLUMNS["suggestions_policy_v1"].split())
    if set(source_columns) != columns:
        raise ValueError("privacy_authority_stale")
    source = {row["id"]: row for row in source_rows}
    candidate = {row["id"]: row for row in candidate_rows}
    if (len(source) != len(source_rows) or len(candidate) != len(candidate_rows)
            or set(source) != set(candidate)):
        raise ValueError("privacy_authority_stale")
    hashes = []
    for identity, original in source.items():
        current = candidate[identity]
        if set(original) != columns or set(current) != columns:
            raise ValueError("privacy_authority_stale")
        before, after = json.loads(original["evidence_json"]), json.loads(current["evidence_json"])
        if type(before) is not list or type(after) is not list or not before or len(before) != len(after):
            raise ValueError("privacy_authority_stale")
        remapped, seen, seen_events, seen_targets = [], set(), set(), set()
        for evidence in before:
            if (type(evidence) is not dict or set(evidence) != {"target_id", "event_id", "revision", "expires_at"}
                    or evidence["target_id"] not in evidence_targets
                    or type(evidence["event_id"]) is not int or not 0 < evidence["event_id"] <= 2**63 - 1
                    or type(evidence["revision"]) is not str
                    or re.fullmatch("[0-9a-f]{64}", evidence["revision"]) is None
                    or type(evidence["expires_at"]) not in (int, float)
                    or not math.isfinite(evidence["expires_at"])):
                raise ValueError("privacy_authority_stale")
            old_key = (evidence["target_id"], evidence["event_id"])
            new_target = evidence_targets[evidence["target_id"]]
            new_key = (new_target, evidence["event_id"])
            if (old_key in seen or evidence["event_id"] in seen_events
                    or old_key not in source_events or new_key not in candidate_events):
                raise ValueError("privacy_authority_stale")
            seen.add(old_key)
            seen_events.add(evidence["event_id"])
            seen_targets.add(evidence["target_id"])
            if _digest({key: _typed(value) for key, value in source_events[old_key].items()}) != _digest(
                    {key: _typed(value) for key, value in candidate_events[new_key].items()}):
                raise ValueError("privacy_authority_stale")
            remapped.append({**evidence, "target_id": new_target})
        if len(seen_targets) != 1 or _canonical_json(remapped) != _canonical_json(after):
            raise ValueError("privacy_authority_stale")
        expected = dict(original)
        # Syntax is accepted only after the complete typed JSON value matched.
        expected["evidence_json"] = current["evidence_json"]
        if _canonical_json(expected) != _canonical_json(current):
            raise ValueError("privacy_authority_stale")
        hashes.append(_digest([_typed(expected[column]) for column in source_columns]))
    return {"columns": source_columns, "row_hashes": sorted(hashes)}


def _protected_suggestions(binding: dict) -> dict:
    stores = {}
    state = Path(binding["state_root"])
    for relative, schema in binding["authority"]["projection"]["tables"].items():
        if "suggestions_policy_v1" not in schema:
            continue
        with _database_image(state / relative) as db:
            cursor = db.execute("SELECT * FROM suggestions_policy_v1 "
                                "WHERE status!='pending' OR snooze_until IS NOT NULL")
            stores[relative] = ([column[0] for column in cursor.description], [dict(row) for row in cursor])
    return stores


def _event_target_owner(root: Path, binding: dict, target: dict) -> dict:
    path = Path(target["descriptor"]["path"])
    if path.is_relative_to(Path(binding["state_root"])):
        return binding
    state = Path(binding["state_root"])
    policy = _read_json(state / ".kza-target-ownership.json")
    if target["id"] not in policy["delegated_targets"]:
        raise ValueError("privacy_authority_stale")
    _validate_delegated_target(root, state, target, policy["source_bindings"])
    for record in _delegation_records(root, state):
        if (record["targets"].get(target["id"]) == target
                and record["source_binding"] in policy["source_bindings"]
                and record["destination_identity"] == _identity(state)):
            owner = read_binding(root, record["source_binding"])
            marker = _read_json(Path(owner["state_root"]) / _MARKER)
            if (marker.get("binding_id") != record["source_binding"]
                    or not path.is_relative_to(Path(owner["state_root"]))
                    or _digest(_closure_proof(root, record["source_binding"])) != record["source_close_sha256"]):
                continue
            return verify_closed_binding(root, record["source_binding"])
    raise ValueError("privacy_authority_stale")


def _event_relocation_data(root: Path, source: dict, candidate: dict, target_relocations: dict) -> tuple:
    if (type(target_relocations) is not dict or not target_relocations
            or any(type(key) is not str or type(value) is not str
                   for key, value in target_relocations.items())):
        raise ValueError("privacy_authority_stale")
    source_targets = _target_inventory(Path(source["state_root"]))
    candidate_targets = _target_inventory(Path(candidate["state_root"]))
    evidence_targets, source_events, candidate_events = {}, {}, {}
    for old, new in target_relocations.items():
        before, after = source_targets.get(old), candidate_targets.get(new)
        if (before is None or after is None or before["participant"] != "events"
                or after["participant"] != "events" or before["version"] != 1 or after["version"] != 1
                or set(before["descriptor"]) != {"path", "table"}
                or set(after["descriptor"]) != {"path", "table"}
                or before["descriptor"]["table"] != "events_policy_v1"
                or after["descriptor"]["table"] != "events_policy_v1"):
            raise ValueError("privacy_authority_stale")
        owner = _event_target_owner(root, source, before)
        original, destination = Path(before["descriptor"]["path"]), Path(after["descriptor"]["path"])
        source_root, candidate_root = Path(owner["state_root"]), Path(candidate["state_root"])
        if (not destination.is_relative_to(candidate_root) or destination.resolve() != destination
                or original.resolve() != original or not original.is_relative_to(source_root)):
            raise ValueError("privacy_authority_stale")
        relative = original.relative_to(source_root).as_posix()
        if (destination.relative_to(candidate_root).as_posix() != relative
                or relative not in owner["schemas"] or relative not in candidate["schemas"]
                or owner["schemas"][relative] != candidate["schemas"][relative]
                or source["schemas"].get(relative) != candidate["schemas"][relative]):
            raise ValueError("privacy_authority_stale")
        latest_path = Path(source["state_root"]) / relative
        latest_descriptor = {"path": str(latest_path), "table": "events_policy_v1"}
        if not any(row["participant"] == "events" and row["version"] == 1
                   and row["descriptor"] == latest_descriptor for row in source_targets.values()):
            raise ValueError("privacy_authority_stale")
        old_evidence, new_evidence = _digest(before["descriptor"]), _digest(after["descriptor"])
        evidence_targets[old_evidence] = new_evidence
        rows = []
        for path in (original, latest_path, destination):
            with _database_image(path) as db:
                rows.append({row["id"]: dict(row) for row in db.execute("SELECT * FROM events_policy_v1")})
        for identity, row in rows[0].items():
            if rows[1].get(identity) == row:
                source_events[(old_evidence, identity)] = row
        for identity, row in rows[2].items():
            candidate_events[(new_evidence, identity)] = row
    return evidence_targets, source_events, candidate_events


def _derived_authority_equivalent(root: Path, source: dict, candidate: dict, target_relocations: dict) -> bool:
    mapping, before_events, after_events = _event_relocation_data(root, source, candidate, target_relocations)
    before, after = _protected_suggestions(source), _protected_suggestions(candidate)
    if set(before) != set(after):
        raise ValueError("privacy_authority_stale")
    expected = json.loads(_canonical_json(source["authority"]["projection"]))
    for relative, (columns, rows) in before.items():
        target_columns, target_rows = after[relative]
        projection = _remapped_suggestion_projection(columns, rows, target_columns, target_rows,
                                                     mapping, before_events, after_events)
        expected["tables"][relative]["suggestions_policy_v1"] = projection
    return candidate["authority"] == {"schema_version": 1, "digest": _digest(expected), "projection": expected}


def _derived_capability_record(root: Path, head: dict, source: dict, candidate: dict,
                                target_relocations: dict) -> dict:
    identity = _digest(candidate)
    closure = root / "admission" / "closures" / (head["binding_id"] + ".json")
    _closure_proof(root, head["binding_id"])
    return {"schema_version": 1, "admission_root": str(root), "source_head": head,
            "source_binding_id": head["binding_id"], "source_closure_sha256": hashlib.sha256(closure.read_bytes()).hexdigest(),
            "candidate_binding_id": identity, "candidate_state_root": candidate["state_root"],
            "candidate_state_identity": candidate["state_identity"], "target_relocations": target_relocations}


def _validate_derived_capability(root: Path, head: dict, source: dict, candidate: dict) -> bool:
    path = root / "admission" / "derived-authority" / (_digest(candidate) + ".json")
    try:
        _safe_directory(path.parent)
        record = _read_json(path)
        mapping = record["target_relocations"]
        if _canonical_json(record) != _canonical_json(_derived_capability_record(root, head, source, candidate, mapping)):
            return False
        if verify_closed_binding(root, head["binding_id"]) != source:
            return False
        if not _derived_authority_equivalent(root, source, candidate, mapping):
            return False
        return (_binding_snapshot(root, source["slot"]) == source
                and _binding_snapshot(root, candidate["slot"]) == candidate
                and read_admission_head(root) == head)
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return False


def _ticket_value(ticket: dict | str) -> dict:
    if type(ticket) is str:
        ticket = json.loads(ticket)
    if (type(ticket) is not dict or set(ticket) != _TICKET_KEYS or ticket["schema_version"] != 1
            or type(ticket["binding_id"]) is not str
            or re.fullmatch("[0-9a-f]{64}", ticket["binding_id"]) is None
            or type(ticket["run_id"]) is not str
            or re.fullmatch("[0-9a-f]{32}", ticket["run_id"]) is None
            or type(ticket["generation"]) is not int or ticket["generation"] < 1
            or type(ticket["controller_pid"]) is not int or ticket["controller_pid"] < 1
            or type(ticket["lease_fd"]) is not int or ticket["lease_fd"] < 0
            or type(ticket["lease_identity"]) is not dict
            or set(ticket["lease_identity"]) != {"device", "inode"}
            or any(type(value) is not int for value in ticket["lease_identity"].values())
            or type(ticket["admission_root"]) is not str):
        raise ValueError("runtime_admission_ticket_schema")
    return ticket


def validate_admission_ticket(state_root: Path, ticket: dict | str, *, check_state: bool = True) -> dict:
    """Validate a live inherited controller lease before creating any services."""
    ticket = _ticket_value(ticket)
    root = _safe_directory(Path(ticket["admission_root"]))
    state = _safe_directory(Path(state_root).resolve(strict=True))
    marker = _read_json(state / _MARKER)
    if marker != {"schema_version": 1, "admission_root": str(root),
                  "binding_id": ticket["binding_id"], "state_root": str(state)}:
        raise ValueError("runtime_admission_marker_mismatch")
    expected = {"schema_version": 1, "generation": ticket["generation"],
                "binding_id": ticket["binding_id"], "status": "running", "run_id": ticket["run_id"]}
    if read_admission_head(root) != expected or os.getppid() != ticket["controller_pid"]:
        raise ValueError("runtime_admission_claim_stale")
    descriptor = ticket["lease_fd"]
    info = os.fstat(descriptor)
    identity = {"device": info.st_dev, "inode": info.st_ino}
    if (not stat.S_ISREG(info.st_mode) or identity != ticket["lease_identity"]
            or identity != _identity(root / ".admission.lock")
            or (root / ".admission.lock").is_symlink()):
        raise ValueError("runtime_admission_lease_mismatch")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError("runtime_admission_lease_not_inherited") from None
    binding = read_binding(root, ticket["binding_id"])
    if binding["state_root"] != str(state):
        raise ValueError("runtime_admission_state_mismatch")
    _check_roots(binding)
    runtime, checked_state, _ = _checked_slot(root, binding["slot"])
    if str(runtime) != binding["runtime_root"] or checked_state != state:
        raise ValueError("runtime_admission_code_mismatch")
    if hashlib.sha256((runtime / "release.json").read_bytes()).hexdigest() != binding["release_sha256"]:
        raise ValueError("runtime_admission_release_changed")
    if check_state and _binding_snapshot(root, binding["slot"]) != binding:
        raise ValueError("privacy_authority_stale")
    validate_target_ownership(root, binding, allow_new_local=not check_state)
    return binding


def write_admission_receipt(ticket: dict | str, resources: dict) -> Path:
    """Record owned closure; the controller still requires an actual child wait."""
    ticket = _ticket_value(ticket)
    binding = read_binding(Path(ticket["admission_root"]), ticket["binding_id"])
    validate_admission_ticket(Path(binding["state_root"]), ticket, check_state=False)
    if type(resources) is not dict or set(resources) != _CLOSED_RESOURCES or any(
            value is not True for value in resources.values()):
        raise ValueError("runtime_admission_close_incomplete")
    receipt = {"schema_version": 1, "binding_id": ticket["binding_id"],
               "generation": ticket["generation"], "run_id": ticket["run_id"],
               "child_pid": os.getpid(), "status": "closed", "resources": resources}
    path = Path(ticket["admission_root"]) / "admission" / "receipts" / (ticket["run_id"] + ".json")
    _atomic_record(path, receipt, immutable=True)
    return path


class AdmissionController:
    """Own one admission root until each managed child is actually reaped.

    A closed head is immutable evidence of the last owned closure. Out-of-band
    changes cannot be adopted as a new authority merely by selecting that slot.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._descriptor = None

    def __enter__(self):
        _safe_directory(self.root, create=True)
        descriptor = os.open(self.root / ".admission.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("runtime_admission_unsafe_lease")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("runtime_admission_busy") from None
            head = read_admission_head(self.root)
            if head is not None and head["status"] != "closed":
                raise ValueError("runtime_admission_not_closed")
            os.ftruncate(descriptor, 0)
            os.write(descriptor, _canonical_json({"controller_pid": os.getpid(), "lease_fd": descriptor}))
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(self, *_exc):
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def _require_lease(self):
        if self._descriptor is None:
            raise ValueError("runtime_admission_controller_not_entered")

    def prepare(self, name: str, *, expected_head: dict | None = None) -> dict:
        """Persist a code/state witness without selecting it or advancing head."""
        self._require_lease()
        with _locked_root(self.root):
            if expected_head is not None and read_admission_head(self.root) != expected_head:
                raise ValueError("runtime_admission_head_changed")
            binding = _binding_snapshot(self.root, _slot_name(name))
            validate_target_ownership(self.root, binding)
            return {"binding_id": _store_binding(self.root, binding), "binding": binding}

    def authorize_target_delegation(self, destination_state: Path, source_binding: str,
                                    target_ids: list[str]) -> str:
        """Grant exact historical targets to one physical copied state."""
        self._require_lease()
        with _locked_root(self.root):
            return _authorize_target_delegation(self.root, destination_state, source_binding, target_ids)

    def authorize_derived_state(self, name: str, latest_binding_id: str, *,
                                target_relocations: dict[str, str]) -> str:
        """Authorize only exact protected event-evidence locator substitutions.

        Returns the candidate binding identity. The external capability changes
        no binding/projection format, pointer, actor, expiry or current head.
        """
        self._require_lease()
        with _locked_root(self.root):
            head = read_admission_head(self.root)
            if head is None or head["status"] != "closed" or head["binding_id"] != latest_binding_id:
                raise ValueError("privacy_authority_stale")
            source = verify_closed_binding(self.root, latest_binding_id)
            source_manual = _verify_manual_head(self.root, head, source)
            candidate = _binding_snapshot(self.root, _slot_name(name))
            if candidate["state_root"] == source["state_root"]:
                raise ValueError("privacy_authority_stale")
            validate_target_ownership(self.root, candidate)
            _require_manual_equivalent(source_manual["capture"], _manual_capture(self.root, candidate))
            if not _derived_authority_equivalent(self.root, source, candidate, target_relocations):
                raise ValueError("privacy_authority_stale")
            if (_binding_snapshot(self.root, source["slot"]) != source
                    or _binding_snapshot(self.root, candidate["slot"]) != candidate
                    or read_admission_head(self.root) != head):
                raise ValueError("privacy_authority_stale")
            identity = _store_binding(self.root, candidate)
            record = _derived_capability_record(self.root, head, source, candidate, target_relocations)
            _atomic_record(self.root / "admission" / "derived-authority" / (identity + ".json"), record)
            return identity

    def activate(self, name: str, *, expected_head: dict | None = None,
                 managed: bool = False) -> str | None:
        """Select a compatible closed state, rejecting stale authority first."""
        self._require_lease()
        name = _slot_name(name)
        with _locked_root(self.root):
            head = read_admission_head(self.root)
            if expected_head is not None and head != expected_head:
                raise ValueError("runtime_admission_head_changed")
            if head is not None and head["status"] != "closed":
                raise ValueError("runtime_admission_not_closed")
            binding = _binding_snapshot(self.root, name)
            validate_target_ownership(self.root, binding)
            if head is not None:
                latest = read_binding(self.root, head["binding_id"])
                if _binding_snapshot(self.root, latest["slot"]) != latest:
                    raise ValueError("privacy_authority_stale")
                latest_manual = _verify_manual_head(self.root, head, latest)
            else:
                current = _current_slot(self.root)
                latest = _binding_snapshot(self.root, current) if current is not None else binding
                latest_manual = None
            is_managed = managed or head is not None or _has_authority(binding) or _has_authority(latest)
            if is_managed:
                if latest["authority"]["digest"] != binding["authority"]["digest"]:
                    if head is None or not _validate_derived_capability(self.root, head, latest, binding):
                        raise ValueError("privacy_authority_stale")
                identity = _store_binding(self.root, binding)
                if head is None and _has_authority(latest):
                    raise ValueError("runtime_admission_manual_legacy_unsealed")
                _prepare_manual_head(self.root, head, binding, previous=latest_manual)
                _write_marker(self.root, identity, binding)
                _cas_head(self.root, head, identity, "closed")
            return _publish_selection(self.root, name)

    def run(self, name: str, argv: list[str] | None = None, *, timeout: float,
            env: dict | None = None, stdout=None, stderr=None) -> subprocess.CompletedProcess:
        """Run under the root lease; only a real closure receipt plus wait closes it.

        The process inherits the caller's process group. A timeout kills and
        reaps this direct child but records UNKNOWN regardless of its exit code.
        """
        self._require_lease()
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("runtime_admission_timeout")
        self.activate(name, managed=True)
        root = self.root
        with _locked_root(root):
            head = read_admission_head(root)
            binding = read_binding(root, head["binding_id"])
            validate_target_ownership(root, binding)
            previous_manual = _verify_manual_head(root, head, binding)
            run_id = uuid4().hex
            running = _cas_head(root, head, head["binding_id"], "running", run_id)
        ticket = {"schema_version": 1, "admission_root": str(root),
                  "binding_id": head["binding_id"], "generation": running["generation"],
                  "run_id": run_id, "lease_fd": self._descriptor,
                  "lease_identity": _identity(root / ".admission.lock"), "controller_pid": os.getpid()}
        child_env = os.environ.copy() if env is None else dict(env)
        child_env["KZA_RUNTIME_ADMISSION"] = _canonical_json(ticket).decode()
        command = list(argv) if argv is not None else [sys.executable, "-B", "-m", "src.main"]
        process = None
        try:
            process = subprocess.Popen(command, cwd=binding["runtime_root"], env=child_env,
                                       pass_fds=(self._descriptor,), stdout=stdout, stderr=stderr)
            output, error = process.communicate(timeout=timeout)
            returncode = process.wait()
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, command, output=output, stderr=error)
            receipt_path = root / "admission" / "receipts" / (run_id + ".json")
            try:
                receipt = _read_json(receipt_path)
            except FileNotFoundError:
                raise ValueError("runtime_admission_close_receipt_missing") from None
            expected = {"schema_version": 1, "binding_id": ticket["binding_id"],
                        "generation": running["generation"], "run_id": run_id,
                        "child_pid": process.pid, "status": "closed",
                        "resources": dict.fromkeys(_CLOSED_RESOURCES, True)}
            if receipt != expected:
                raise ValueError("runtime_admission_close_receipt_mismatch")
            with _locked_root(root):
                if read_admission_head(root) != running:
                    raise ValueError("runtime_admission_head_changed")
                _seal_local_targets(root, _binding_snapshot(root, name))
                closed = _binding_snapshot(root, name)
                validate_target_ownership(root, closed)
                identity = _store_binding(root, closed)
                proof = {"schema_version": 1, "binding_id": identity, "running_head": running,
                         "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                         "child_pid": process.pid}
                _atomic_record(root / "admission" / "closures" / (identity + ".json"), proof)
                _prepare_manual_head(root, running, closed, previous=previous_manual, proof=proof)
                _write_marker(root, identity, closed)
                _cas_head(root, running, identity, "closed")
            return subprocess.CompletedProcess(command, returncode, output, error)
        except BaseException as original:
            failures = []
            try:
                with _locked_root(root):
                    if read_admission_head(root) == running:
                        _cas_head(root, running, running["binding_id"], "unknown", run_id)
            except BaseException as error:
                failures.append(error)
            finally:
                if process is not None:
                    try:
                        try:
                            if process.poll() is None:
                                try:
                                    process.send_signal(signal.SIGTERM)
                                finally:
                                    try:
                                        process.communicate(timeout=2)
                                    except subprocess.TimeoutExpired:
                                        process.kill()
                                        process.communicate()
                        finally:
                            process.wait()
                    except BaseException as error:
                        failures.append(error)
            if failures:
                raise BaseExceptionGroup("runtime_admission_retirement_failed", [original, *failures]) from None
            raise


def prepare_binding(root: Path, name: str, *, expected_head: dict | None = None) -> dict:
    """Capture a closed slot under the admission lease without changing selection."""
    with AdmissionController(root) as controller:
        return controller.prepare(name, expected_head=expected_head)


def authorize_target_delegation(root: Path, destination_state: Path, source_binding: str,
                                 target_ids: list[str]) -> str:
    """Issue an exact delegation while exclusively owning the current closed head."""
    with AdmissionController(root) as controller:
        return controller.authorize_target_delegation(destination_state, source_binding, target_ids)


def activate_slot(root: Path, name: str) -> str | None:
    """Atomically select an integrity-checked slot with non-regressing authority."""
    with AdmissionController(root) as controller:
        return controller.activate(name)


def run_slot(root: Path, name: str, argv: list[str] | None = None, *, timeout: float,
             env: dict | None = None, stdout=None, stderr=None) -> subprocess.CompletedProcess:
    """Execute one managed child while retaining exclusive root admission."""
    with AdmissionController(root) as controller:
        return controller.run(name, argv, timeout=timeout, env=env, stdout=stdout, stderr=stderr)


def main() -> None:
    """Prepare or select a slot after the operator has stopped all state writers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", required=True,
                        help="Assert that all writers are stopped; this tool does not stop services")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("name")
    prepare.add_argument("--state-source", type=Path, required=True)
    activate = commands.add_parser("activate")
    activate.add_argument("name")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_slot(args.archive, args.root, args.name, args.state_source)
        print(json.dumps({"slot": args.name, "commit": manifest["git_commit"]}))
    else:
        previous = activate_slot(args.root, args.name)
        print(json.dumps({"slot": args.name, "previous": previous}))


if __name__ == "__main__":
    main()
