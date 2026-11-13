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
import threading
from uuid import UUID, uuid4


_HEAD_KEYS = {"schema_version", "generation", "binding_id", "status", "run_id"}
_BINDING_KEYS = {"schema_version", "slot", "runtime_root", "state_root", "release_sha256",
                 "code_commit", "python_magic_hex", "state_identity", "store_roots", "schemas",
                 "sdk_versions", "authority", "ownership_sha256"}
_TICKET_KEYS = {"schema_version", "admission_root", "binding_id", "generation", "run_id",
                "lease_fd", "lease_identity", "controller_pid"}
_MARKER = ".kza-state-binding.json"
_CLOSED_RESOURCES = {"graph_closed", "privacy_closed", "sdk_closed", "native_closed"}
# A managed candidate must carry the companion pair a managed child admits with
# (src/core/runtime_admission loads scripts/stage_runtime by path).
_PROBE_COMPANIONS = ("scripts/stage_runtime.pyc", "src/core/runtime_admission.pyc")
_PROBE_TIMEOUT_SECONDS = 60.0
_PROBE_TIMEOUT_BOUNDS = (1.0, 3600.0)
# The probe gets an allow-list, never the controller environment (secrets such
# as HOME_ASSISTANT_TOKEN must not reach candidate code that may be refused).
# _binding_snapshot reads no environment variable itself; these keep only what
# can change how it observes the same files: LANG/LC_ALL/LC_CTYPE fix the locale
# the controller decodes state file names with, and TMPDIR/SQLITE_TMPDIR keep any
# SQLite spill files of the in-memory backups where the controller would put them.
# PATH and HOME are not needed (no subprocess, no user files); -I already drops
# PYTHON* variables and the user site.
_PROBE_ENVIRONMENT = ("LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SQLITE_TMPDIR")
_PROBE_UNMANAGED_EXIT = 3
_PROBE_REJECTED_EXIT = 2
# Runs in a fresh "-I -B" interpreter: loads the candidate's own companion by
# path and prints the binding its code computes for the same root and slot,
# after its own validate_target_ownership accepted that binding (U3b-0b D2: the
# child re-snapshots every delegation owner with candidate code at start).
_PROBE_SOURCE = "\n".join([
    "import importlib.util, json, sys",
    "from pathlib import Path",
    "module_path, root, slot = sys.argv[1:4]",
    "spec = importlib.util.spec_from_file_location('_kza_candidate_stage_runtime', module_path)",
    "module = importlib.util.module_from_spec(spec)",
    "sys.modules[spec.name] = module",
    "spec.loader.exec_module(module)",
    "snapshot = getattr(module, '_binding_snapshot', None)",
    "validate = getattr(module, 'validate_target_ownership', None)",
    "if snapshot is None or validate is None:",
    "    sys.stdout.write(json.dumps({'probe': 'unmanaged'}))",
    "    raise SystemExit(" + str(_PROBE_UNMANAGED_EXIT) + ")",
    "try:",
    "    binding = snapshot(Path(root), slot)",
    "    validate(Path(root), binding)",
    "except ValueError as error:",
    "    sys.stdout.write(json.dumps({'probe': 'error', 'code': str(error)}))",
    "    raise SystemExit(" + str(_PROBE_REJECTED_EXIT) + ")",
    "sys.stdout.write(json.dumps({'probe': 'binding', 'binding': binding},",
    "                            sort_keys=True, separators=(',', ':'), allow_nan=False))",
])
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
    "fact_owners", "fact_owner_revisions",
    # U3b-4a-0: voice-attribution consents, attempts, misattribution events,
    # suspensions and the admin voice switch (G1) are authority, never payload.
    "voice_attribution_consents", "voice_verification_attempts", "voice_attribution_events",
    "voice_attribution_suspensions", "voice_attribution_control",
    # U3b-0d3: the content-free deletion ledger and its write-once salt. The
    # ledger is projected as a constant-size hash chain, never row by row.
    "deletion_ledger_v1", "deletion_ledger_salt_v1",
})

# Single-row version markers; each must hold exactly [(1,)].
_SCHEMA_MARKERS = frozenset({"voice_group_schema", "voice_person_schema", "voice_capture_review_schema",
                             "fact_owner_schema", "voice_attribution_schema"})

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
    "fact_owners": "fact_id:TEXT person_id:TEXT method:TEXT revision:INTEGER assigned_at:REAL assigned_by:TEXT evidence:TEXT",
    "fact_owner_revisions": "fact_id:TEXT revision:INTEGER last_change:TEXT confirmed_by_assignment:INTEGER changed_at:REAL",
    "fact_owner_schema": "version:INTEGER",
    "voice_attribution_consents": "person_id:TEXT state:TEXT revision:INTEGER verified_profile_id:TEXT verified_profile_revision:INTEGER verified_link_revision:INTEGER changed_at:REAL changed_by:TEXT",
    "voice_verification_attempts": "person_id:TEXT attempted_at:REAL outcome:TEXT",
    "voice_attribution_events": "id:TEXT profile_id:TEXT kind:TEXT score_bucket:INTEGER at:REAL",
    "voice_attribution_suspensions": "profile_id:TEXT revision:INTEGER suspended_at:REAL reason:TEXT",
    "voice_attribution_control": "id:INTEGER enabled:INTEGER revision:INTEGER changed_at:REAL changed_by:TEXT",
    "voice_attribution_schema": "version:INTEGER",
    "deletion_ledger_v1": "seq:INTEGER kind:TEXT subject_sha256:TEXT at:REAL prev_sha256:TEXT row_sha256:TEXT",
    "deletion_ledger_salt_v1": "id:INTEGER salt:TEXT",
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


# U3b-0d3 (design v3 N3): the chain src/memory/deletion_ledger.py appends.
# row_sha256 = sha256(prev_sha256 || canonical([seq, kind, subject_sha256, at])).
_LEDGER_TABLE = "deletion_ledger_v1"
_LEDGER_KINDS = frozenset({"conversation_erased", "fact_deleted", "fact_pruned_retired_owner",
                           "voice_profile_deleted", "user_retired", "credential_revoked",
                           "voice_consent_withdrawn"})
_LEDGER_GENESIS = "0" * 64
# Receipt schema 2 (design v2 U3b-0d3, v3 N2): the closed state the child observed,
# in a record beside the unchanged schema-1 receipt (read_receipt_state).
_RECEIPT_STATE_KEYS = ("closed_state_sha256", "state_manifest_sha256")
# Rewritten by the controller's own seal, so never part of the state manifest.
_MANIFEST_EXCLUDED = frozenset({_MARKER, ".kza-target-ownership.json"})
_CLOSED_STATE_KEYS = ("state_identity", "store_roots", "schemas", "sdk_versions", "ownership_sha256")

# U3b-0d (managed launcher, design v3 N7). build_runtime copies these three
# integers from this file's text into release.json ("launcher_contract",
# "format_epoch", "receipt_state", "controller_epoch"); the controller reads
# them as JSON from a verified release.json and never imports candidate code.
# A missing key counts as 0.
# LAUNCHER_CONTRACT: the child-side contract a slot implements (U3b-0d1 stop
#   and readiness, schema-2 receipt-state record). serve requires >= 3.
# FORMAT_EPOCH: shared formats written by the controller and read by slot code
#   (binding, head, ticket, manual head, delegation records). 1 adds the
#   'recovery' manual head kind. Slots below the controller's minimum are
#   refused at activation (runtime_admission_rollback_format_closed).
# RECEIPT_STATE_SCHEMA: 2 when the slot's child writes the receipt-state
#   record; the controller then refuses a missing record (review 0d3 K2).
# CONTROLLER_EPOCH: monotonic controller version (admission/controller.json).
LAUNCHER_CONTRACT = 3
FORMAT_EPOCH = 1
RECEIPT_STATE_SCHEMA = 2
CONTROLLER_EPOCH = 1
MIN_FORMAT_EPOCH = 1
_RELEASE_INTEGER_KEYS = ("launcher_contract", "format_epoch", "receipt_state", "controller_epoch")

_ADMISSION_VALIDATION_LOCK = threading.RLock()


class _AdmissionValidationSnapshot(tuple):
    """Immutable observations from a complete scan, never an execution grant.

    The installed RuntimeAdmission retains the exact issued object and binds it
    to its actual NativeExecution. A structurally similar object is not a
    substitute for that provenance. Nested metadata is canonical bytes.
    """

    __slots__ = ()

    def __new__(cls, ticket_json: bytes, binding_json: bytes, ownership_json: bytes,
                artifacts_json: bytes) -> "_AdmissionValidationSnapshot":
        if any(type(value) is not bytes for value in (ticket_json, binding_json, ownership_json, artifacts_json)):
            raise TypeError("runtime_admission_validation_snapshot")
        return tuple.__new__(cls, (ticket_json, binding_json, ownership_json, artifacts_json))

    @property
    def ticket_json(self) -> bytes:
        return self[0]

    @property
    def binding_json(self) -> bytes:
        return self[1]

    @property
    def ownership_json(self) -> bytes:
        return self[2]

    @property
    def artifacts_json(self) -> bytes:
        return self[3]


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


def _require_copy_not_invalidated(root: Path, state_root: Path) -> None:
    """Reject historical copy invalidations before opening any payload store.

    A fence records an earlier invalidation; its HEAD never grants execution
    and need not equal today's HEAD. The immutable binding format is unchanged.
    """
    directory = root / "admission/copy-cleanup/v1/fences"
    parent = root
    for component in ("admission", "copy-cleanup", "v1", "fences"):
        parent = parent / component
        if parent.is_symlink():
            raise ValueError("runtime_admission_copy_fence_unknown")
        if not parent.exists():
            return
        try:
            _safe_directory(parent)
        except (OSError, ValueError):
            raise ValueError("runtime_admission_copy_fence_unknown") from None
    keys = {"schema_version", "scope", "copy_id", "catalog_revision", "catalog_record_sha256",
            "admission_root", "admission_identity", "destination", "root_identity", "authority_head",
            "operation_id", "conversation_id", "layout_sha256", "claim_id", "status"}

    def physical(value):
        return (type(value) is dict and set(value) == {"device", "inode"}
                and type(value["device"]) is int and value["device"] >= 0
                and type(value["inode"]) is int and value["inode"] > 0)

    for path in sorted(directory.iterdir()):
        try:
            if not re.fullmatch(r"[0-9a-f]{32}\.json", path.name):
                raise ValueError
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 16384:
                    raise ValueError
                raw = stream.read()
            value = json.loads(raw)
            if type(value) is not dict or set(value) != keys or _canonical_json(value) + b"\n" != raw:
                raise ValueError
            if (type(value["schema_version"]) is not int or value["schema_version"] != 1
                    or value["scope"] != "collector_rows_files_v1" or value["status"] != "invalidated"
                    or value["copy_id"] != path.stem
                    or type(value["catalog_revision"]) is not int or value["catalog_revision"] < 1
                    or any(type(value[key]) is not str or not re.fullmatch("[0-9a-f]{64}", value[key])
                           for key in ("catalog_record_sha256", "layout_sha256"))
                    or type(value["claim_id"]) is not str or not re.fullmatch("[0-9a-f]{32}", value["claim_id"])
                    or value["admission_root"] != str(root)
                    or not physical(value["admission_identity"])
                    or value["admission_identity"] != _identity(root)
                    or not physical(value["root_identity"])):
                raise ValueError
            for key in ("operation_id", "conversation_id"):
                if type(value[key]) is not str or str(UUID(value[key])) != value[key]:
                    raise ValueError
            destination = value["destination"]
            if (type(destination) is not str or not Path(destination).is_absolute()
                    or Path(destination).resolve(strict=False) != Path(destination)):
                raise ValueError
            head = value["authority_head"]
            if (type(head) is not dict or set(head) != _HEAD_KEYS
                    or type(head["schema_version"]) is not int or head["schema_version"] != 1
                    or type(head["generation"]) is not int or head["generation"] < 1
                    or type(head["binding_id"]) is not str or not re.fullmatch("[0-9a-f]{64}", head["binding_id"])
                    or head["status"] != "closed" or head["run_id"] is not None):
                raise ValueError
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            raise ValueError("runtime_admission_copy_fence_unknown") from None
        # Replacing a directory at this locator must not evade invalidation.
        if destination == str(state_root):
            raise ValueError("runtime_admission_copy_invalidated")


def _read_copy_observation(directory: Path, root: Path, *, allow_orphans: bool = False) -> tuple[dict, dict, dict]:
    """Validate a historical observation, without granting any capability."""
    def require(condition):
        if not condition:
            raise ValueError("runtime_admission_copy_observation_unknown")

    def read(path):
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 16*1024*1024)
            raw = stream.read()
        value = json.loads(raw)
        require(type(value) is dict and _canonical_json(value) + b"\n" == raw)
        return value, hashlib.sha256(raw).hexdigest()

    def digest(value):
        return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None

    def physical(value):
        return (type(value) is dict and set(value) == {"device", "inode"}
                and type(value["device"]) is int and value["device"] >= 0
                and type(value["inode"]) is int and value["inode"] > 0)

    def relative(value):
        return (type(value) is str and bool(value) and not Path(value).is_absolute()
                and "\x00" not in value and ".." not in Path(value).parts and Path(value).as_posix() == value)

    _safe_directory(directory)
    require(re.fullmatch("[0-9a-f]{32}", directory.name) is not None)
    intent, intent_sha = read(directory / "intent.json")
    require(set(intent) == {"schema_version", "observation_id", "kind", "authority", "admission_root",
        "admission_identity", "observed_root", "parent_identity", "root_identity", "extents",
        "source_binding_ids", "existing_copy", "before_first_byte"})
    require(type(intent["schema_version"]) is int and intent["schema_version"] == 1
        and intent["observation_id"] == directory.name
        and type(intent["kind"]) is str and intent["kind"] in {"forensic_audit", "phase_checkpoint", "failed_work"}
        and intent["before_first_byte"] is False and intent["admission_root"] == str(root)
        and physical(intent["admission_identity"]) and intent["admission_identity"] == _identity(root)
        and physical(intent["root_identity"]) and physical(intent["parent_identity"]))
    observed = intent["observed_root"]
    require(type(observed) is str and Path(observed).is_absolute()
        and Path(observed).resolve(strict=False) == Path(observed))
    authority = intent["authority"]
    require(type(authority) is dict and set(authority) == _HEAD_KEYS
        and type(authority["schema_version"]) is int and authority["schema_version"] == 1
        and type(authority["generation"]) is int and authority["generation"] >= 1
        and digest(authority["binding_id"]) and authority["status"] == "closed" and authority["run_id"] is None)
    refs = intent["source_binding_ids"]
    require(type(refs) is list and len(set(refs)) == len(refs) and all(digest(v) for v in refs))
    existing = intent["existing_copy"]
    require(existing is None or type(existing) is dict and set(existing) == {"copy_id", "revision", "record_sha256"}
        and type(existing["copy_id"]) is str and re.fullmatch("[0-9a-f]{32}", existing["copy_id"])
        and type(existing["revision"]) is int and existing["revision"] >= 1 and digest(existing["record_sha256"]))
    extents = intent["extents"]
    require(type(extents) is list and bool(extents))
    families = {"managed_payload", "semantic_index", "ambient_capture", "collectors", "contexts",
        "events", "suggestions", "diagnostics", "voices", "streams", "history", "authority"}
    representations = {"raw", "state", "sqlite", "sdk_work", "blob", "logical_export", "manifest",
        "preparation", "history_archive", "derived_state"}
    for extent in extents:
        require(type(extent) is dict and set(extent) == {"relative", "family", "representation", "source_sha256"}
            and relative(extent["relative"])
            and type(extent["family"]) is str and extent["family"] in families
            and type(extent["representation"]) is str and extent["representation"] in representations
            and (extent["source_sha256"] is None or digest(extent["source_sha256"])))
    require(len({e["relative"] for e in extents}) == len(extents))
    head, _ = read(directory / "head.json")
    require(set(head) == {"schema_version", "observation_id", "revision", "sha256"}
        and type(head["schema_version"]) is int and head["schema_version"] == 1
        and head["observation_id"] == directory.name and type(head["revision"]) is int
        and head["revision"] in {1, 2} and digest(head["sha256"]))
    revision, expected, names, selected = head["revision"], head["sha256"], {"intent.json", "head.json"}, None
    while revision:
        name = f"{revision:012d}-{expected}.json"
        event, actual = read(directory / name)
        names.add(name)
        require(actual == expected and set(event) == {"schema_version", "observation_id", "revision",
            "previous_sha256", "intent_sha256", "phase", "inventory", "inventory_sha256"}
            and type(event["schema_version"]) is int and event["schema_version"] == 1
            and type(event["revision"]) is int and event["revision"] == revision
            and event["observation_id"] == directory.name and event["intent_sha256"] == intent_sha)
        if revision == 1:
            require(event["previous_sha256"] is None and event["phase"] == "intent"
                and event["inventory"] is None and event["inventory_sha256"] is None)
        else:
            inventory = event["inventory"]
            require(digest(event["previous_sha256"]) and event["phase"] == "observed"
                and type(inventory) is dict and bool(inventory)
                and event["inventory_sha256"] == hashlib.sha256(_canonical_json(inventory)).hexdigest())
            for name, item in inventory.items():
                require(relative(name) and name != "." and type(item) is dict
                    and set(item) == {"sha256", "size", "identity"} and digest(item["sha256"])
                    and type(item["size"]) is int and item["size"] >= 0 and physical(item["identity"]))
            covered = set()
            for extent in extents:
                path = extent["relative"]
                matches = {name for name in inventory if path == "." or name == path or Path(name).is_relative_to(path)}
                require(bool(matches))
                if extent["source_sha256"] is not None:
                    require(path in inventory and inventory[path]["sha256"] == extent["source_sha256"])
                covered.update(matches)
            require(covered == set(inventory))
        if selected is None:
            selected = event
        expected, revision = event["previous_sha256"], revision - 1
    actual_names = {p.name for p in directory.iterdir()}
    require(allow_orphans or actual_names == names)
    if allow_orphans:
        head = dict(head, uncommitted_entries=sorted(actual_names - names))
    return intent, selected, head


def _require_copy_not_observed(root: Path, state_root: Path) -> None:
    """Deny adoption of any historical observed locator before store access."""
    parent = root
    for part in ("admission", "copy-observations", "v1"):
        parent = parent / part
        try:
            if parent.is_symlink():
                raise ValueError
            if not parent.exists():
                return
            _safe_directory(parent)
        except (OSError, ValueError):
            raise ValueError("runtime_admission_copy_observation_unknown") from None
    for directory in sorted(parent.iterdir()):
        try:
            intent, _, _ = _read_copy_observation(directory, root)
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            raise ValueError("runtime_admission_copy_observation_unknown") from None
        observed = Path(intent["observed_root"])
        if state_root.is_relative_to(observed) or observed.is_relative_to(state_root):
            raise ValueError("runtime_admission_copy_observed")


def _durable_parents(directory: Path) -> None:
    """Create ``directory`` and fsync the parent of every level it created (review K6)."""
    missing = []
    current = directory
    while not current.exists():
        missing.append(current)
        current = current.parent
    _safe_directory(directory, create=True)
    for created in reversed(missing):
        descriptor = os.open(created.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_record(path: Path, value: dict, *, immutable: bool = False) -> None:
    _durable_parents(path.parent)
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


def _validate_binding_record(binding_id: str, binding: dict) -> dict:
    """Validate the original content-addressed binding after its caller reads it."""
    if set(binding) != _BINDING_KEYS or binding["schema_version"] != 1 or _digest(binding) != binding_id:
        raise ValueError("runtime_admission_binding_schema")
    return binding


def read_binding(root: Path, binding_id: str) -> dict:
    """Read an immutable canonical code/state descriptor by its content identity."""
    root = Path(root).resolve(strict=True)
    if type(binding_id) is not str or re.fullmatch("[0-9a-f]{64}", binding_id) is None:
        raise ValueError("runtime_admission_binding_id")
    path = root / "admission" / "bindings" / (binding_id + ".json")
    _safe_directory(path.parent)
    binding = _read_json(path)
    return _validate_binding_record(binding_id, binding)


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


def _ledger_row_sha256(prev: str, seq: int, kind: str, subject: str, at: float) -> str:
    return hashlib.sha256(prev.encode() + _canonical_json([seq, kind, subject, at])).hexdigest()


def _deletion_ledger_projection(db: sqlite3.Connection, *, upto: int | None = None) -> dict:
    """Constant-size chain projection; any break refuses (runtime_admission_ledger_chain_broken).

    Recomputes every row in seq order (contiguous from 1, known kind, 64-hex
    subject, finite REAL time, prev equal to the running head, row hash
    recomputed). Linear only in time: the binding holds count, max_seq and
    the head, never per-row digests (design v3 N3, review N3).
    """
    head, expected = _LEDGER_GENESIS, 1
    query = "SELECT seq,kind,subject_sha256,at,prev_sha256,row_sha256 FROM " + _LEDGER_TABLE
    parameters: tuple = ()
    if upto is not None:
        query, parameters = query + " WHERE seq<=?", (upto,)
    for seq, kind, subject, at, prev, row in db.execute(query + " ORDER BY seq", parameters):
        if (type(seq) is not int or seq != expected or kind not in _LEDGER_KINDS or not _hex64(subject)
                or type(at) is not float or not math.isfinite(at) or prev != head
                or row != _ledger_row_sha256(prev, seq, kind, subject, at)):
            raise ValueError("runtime_admission_ledger_chain_broken")
        head, expected = row, expected + 1
    return {"count": expected - 1, "max_seq": expected - 1, "chain_head": head, "chain_valid": True}


def _empty_protected_projection(name: str) -> dict:
    if name == _LEDGER_TABLE:
        return {"count": 0, "max_seq": 0, "chain_head": _LEDGER_GENESIS, "chain_valid": True}
    return {"columns": [item.split(":")[0] for item in _TABLE_COLUMNS[name].split()], "row_hashes": []}


def deletion_ledger_regressions(previous: dict, state_root: Path) -> list[str]:
    """Databases whose deletion ledger did not evolve forward from a binding (verification helper).

    Forward evolution (design v3 N3): current count >= the bound count, and the
    chain recomputed over seq <= the bound max_seq ends at the bound head. A
    missing database or table, or a broken chain, is a regression. Returns the
    regressed relative database paths, content-free; empty means forward.
    """
    state = _safe_directory(Path(state_root).resolve(strict=True))
    regressed = []
    for relative, tables in sorted(previous["authority"]["projection"]["tables"].items()):
        bound = tables.get(_LEDGER_TABLE)
        if bound is None:
            continue
        path = state / relative
        try:
            with _database_image(path) as db:
                current = _deletion_ledger_projection(db)
                prefix = _deletion_ledger_projection(db, upto=bound["max_seq"])
        except (ValueError, OSError, sqlite3.Error):
            regressed.append(relative)
            continue
        if (current["count"] < bound["count"] or prefix["max_seq"] != bound["max_seq"]
                or prefix["chain_head"] != bound["chain_head"]):
            regressed.append(relative)
    return regressed


def state_manifest_sha256(state_root: Path) -> str:
    """sha256 of the sorted (relative path, st_ino, st_size, st_mtime_ns) of every regular file.

    Covers every payload store (SQLite files with their -wal, Chroma,
    contexts, captures, collectors, voices, diagnostics) and excludes only
    the marker and the target-ownership policy the seal rewrites, plus every
    ``*-shm`` file: never authoritative (SQLite rebuilds it from the -wal) and
    touched by any read-only reader, so including it would turn an innocent
    reader between close and seal into UNKNOWN (review K1). lstat only:
    nothing is opened, so it must run before any database image (design v3 N2).
    """
    state = Path(state_root)
    if state.is_symlink() or not state.is_dir():
        raise ValueError("runtime_admission_unsafe_directory")
    entries, pending = [], [state]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            for entry in iterator:
                info = entry.stat(follow_symlinks=False)
                relative = Path(entry.path).relative_to(state).as_posix()
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError("runtime_admission_state_symlink")
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif (stat.S_ISREG(info.st_mode) and relative not in _MANIFEST_EXCLUDED
                        and not entry.name.endswith("-shm")):
                    entries.append([relative, info.st_ino, info.st_size, info.st_mtime_ns])
    return _digest(sorted(entries))


def closed_state_sha256(binding: dict) -> str:
    """Digest of the binding subset the seal does not change before its own snapshot."""
    value = {key: binding[key] for key in _CLOSED_STATE_KEYS}
    value["authority_digest"] = binding["authority"]["digest"]
    return _digest(value)


def _receipt_state_path(root: Path, run_id: str) -> Path:
    # Its own directory: readers and tests that list admission/receipts see only receipts.
    return Path(root) / "admission" / "receipt-states" / (run_id + ".json")


def read_receipt_state(root: Path, receipt: dict) -> dict | None:
    """The closed-state record a receipt-schema-2 child wrote beside its receipt, or None.

    The receipt file itself keeps its exact schema-1 shape: older slots read
    it when they validate delegation-chain hops and manual heads (probe and
    child), so changing it would close rollback (a shared format, design v3
    N7). The record names the same binding, generation, run and child pid and
    carries closed_state_sha256 and state_manifest_sha256. None means an
    older slot (no record), which run() still seals; a malformed or
    mismatching record raises runtime_admission_receipt_state.
    """
    path = _receipt_state_path(root, receipt["run_id"])
    if not path.exists() and not path.is_symlink():
        return None
    record = _read_json(path)
    identity = {key: receipt[key] for key in ("binding_id", "generation", "run_id", "child_pid")}
    if (set(record) != {"schema_version", *identity, *_RECEIPT_STATE_KEYS}
            or type(record["schema_version"]) is not int or record["schema_version"] != 2
            or {key: record[key] for key in identity} != identity
            or not all(_hex64(record[key]) for key in _RECEIPT_STATE_KEYS)):
        raise ValueError("runtime_admission_receipt_state")
    return record


def release_integers(binding: dict) -> dict:
    """The launcher integers of a binding's verified release.json (missing = 0).

    Read as JSON after the file's sha256 matched the binding's release hash;
    candidate code is never imported (design v3 N7).
    """
    raw = (Path(binding["runtime_root"]) / "release.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["release_sha256"]:
        raise ValueError("runtime_admission_release_changed")
    manifest = json.loads(raw)
    values = {}
    for key in _RELEASE_INTEGER_KEYS:
        value = manifest.get(key, 0)
        if type(value) is not int or value < 0:
            raise ValueError("runtime_admission_release_integer")
        values[key] = value
    return values


def _require_launcher_slot(binding: dict) -> None:
    """serve's slot integers (managed_launcher.check_slot): launcher contract and format epoch."""
    try:
        values = release_integers(binding)
    except (ValueError, OSError):
        raise ValueError("runtime_admission_launcher_contract_missing") from None
    if values["launcher_contract"] < LAUNCHER_CONTRACT:
        raise ValueError("runtime_admission_launcher_contract_missing")
    if values["format_epoch"] < MIN_FORMAT_EPOCH:
        raise ValueError("runtime_admission_rollback_format_closed")


def receipt_state_required(binding: dict) -> bool:
    """Whether the slot declares a schema-2 receipt-state record (review 0d3 K2).

    The declaration lives in release.json, covered by the binding's release
    hash, so deleting the record cannot downgrade the slot to 'older'.
    """
    values = release_integers(binding)
    return values["receipt_state"] >= 2 or values["launcher_contract"] >= 3


def verify_receipt_state(record: dict, state_root: Path, binding: dict | None = None) -> None:
    """Check a closed-state record against the state now (verification helper for recovery).

    The manifest is recomputed first, before any database image is opened;
    then, when a fresh pre-seal binding (captured after this manifest) is
    given, its closed-state digest. Never recompute the manifest after the
    binding: its read-only images may create or touch -shm files. Raises
    runtime_admission_state_manifest_mismatch or
    runtime_admission_closed_state_mismatch.
    """
    if type(record) is not dict or record.get("schema_version") != 2:
        raise ValueError("runtime_admission_receipt_state")
    if state_manifest_sha256(state_root) != record["state_manifest_sha256"]:
        raise ValueError("runtime_admission_state_manifest_mismatch")
    if binding is not None and closed_state_sha256(binding) != record["closed_state_sha256"]:
        raise ValueError("runtime_admission_closed_state_mismatch")


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
        if name in _SCHEMA_MARKERS:
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
                selected = {name: _row_projection(db, name)
                            for name in sorted(names & _PROTECTED_TABLES - {_LEDGER_TABLE})}
                if _LEDGER_TABLE in names:
                    selected[_LEDGER_TABLE] = _deletion_ledger_projection(db)
                selected["schema_contract_v1"] = contract
                if "facts" in names:
                    condition = " WHERE state IN ('confirmed','deleted','quarantined')"
                    if {"fact_owners", "fact_owner_revisions"} <= names:
                        # U3b R4: every owned or ever-assigned fact is pinned in any
                        # state, matching payload_derivation's never-cut set.
                        condition += (" OR id IN (SELECT fact_id FROM fact_owners"
                                      " UNION SELECT fact_id FROM fact_owner_revisions)")
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
    with _ADMISSION_VALIDATION_LOCK:
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


RUNTIME_FILE_MODE = 0o444
RUNTIME_DIRECTORY_MODE = 0o555


def make_runtime_read_only(runtime: Path) -> None:
    """Files 0444 and directories 0555, bottom-up; links are left as links.

    U3b-0e (design v3 2b, reviews F2 and N10): a write by the managed child
    into its own runtime (a cwd-relative default) fails loudly with EACCES at
    the first run instead of changing the artifact tree for the next start.
    """
    for _directory, _names, files, descriptor in os.fwalk(runtime, topdown=False, follow_symlinks=False):
        for name in files:
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                os.chmod(name, RUNTIME_FILE_MODE, dir_fd=descriptor)
            elif not stat.S_ISLNK(info.st_mode):
                raise ValueError("runtime_read_only_special_file")
        os.fchmod(descriptor, RUNTIME_DIRECTORY_MODE)


def _mount_points() -> list[str]:
    """Mount points from /proc/self/mountinfo (Linux); empty where it is absent.

    Same-filesystem bind mounts keep st_dev, so only the mount table sees them.
    """
    try:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="surrogateescape") as table:
            lines = table.read().splitlines()
    except OSError:
        return []
    points = []
    for line in lines:
        fields = line.split()
        if len(fields) > 4:
            points.append(fields[4].replace("\\040", " ").replace("\\011", "\t")
                          .replace("\\012", "\n").replace("\\134", "\\"))
    return points


def make_writable_then_remove(path: Path) -> None:
    """Remove a tree that contains a read-only runtime (abort, Q5 and 0c).

    Never follows a symlink (a link is removed as a link) and never crosses a
    mount. Before any mode changes it refuses when the path's device differs
    from its parent's, when any directory in the tree is a mount point
    (os.path.ismount) or on another device, or when the mount table lists a
    mount point at or under the path (same-filesystem bind mounts; review F7).
    Only directories gain owner rwx; the target of any link is untouched.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: ``make_writable_mount_crossing``.
    """
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        path.unlink()
        return
    absolute = Path(os.path.abspath(path))
    if absolute.parent.lstat().st_dev != info.st_dev or os.path.ismount(absolute):
        raise ValueError("make_writable_mount_crossing")
    real = os.path.realpath(absolute)
    for point in _mount_points():
        if point == real or point.startswith(real.rstrip("/") + "/"):
            raise ValueError("make_writable_mount_crossing")

    def fail(error: OSError) -> None:
        raise error
    for directory, _names, _files, descriptor in os.fwalk(path, follow_symlinks=False, onerror=fail):
        if os.fstat(descriptor).st_dev != info.st_dev or (directory != os.fspath(path) and os.path.ismount(directory)):
            raise ValueError("make_writable_mount_crossing")
    for _directory, _names, _files, descriptor in os.fwalk(path, follow_symlinks=False, onerror=fail):
        mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
        if mode & stat.S_IRWXU != stat.S_IRWXU:
            os.fchmod(descriptor, mode | stat.S_IRWXU)
    shutil.rmtree(path)


def prepare_slot(archive: Path, root: Path, name: str, state_source: Path) -> dict:
    """Verify an archive and copy offline config/data into a new private slot.

    Args:
        archive: Archive produced by build_runtime.py.
        root: Explicit staging root; no production path is assumed.
        name: New slot name, independent of the commit (allows repeated trials).
        state_source: Offline directory containing config/ and data/. Optional
            .env is copied privately. Models, logs and dependencies stay external.

    The runtime is made read-only (files 0444, directories 0555) before
    anything observes it; remove a slot with ``make_writable_then_remove``.

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
            # Before any artifact observation (activation binds the tree).
            make_runtime_read_only(runtime)
            os.rename(staging, target)
    return manifest


def _artifact_stat(info: os.stat_result, *, ancestor: bool = False) -> list[int]:
    """Observe identity; only code-tree objects also pin mutation metadata."""
    identity = [info.st_dev, info.st_ino, info.st_mode]
    return identity if ancestor else identity + [
        info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _artifact_ancestors(runtime: Path) -> dict:
    # Shared ancestors (including /private/tmp) legitimately gain siblings.
    # Their physical identity matters; their timestamps do not describe code.
    result = {}
    for parent in reversed(runtime.parents):
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("runtime_admission_artifact_changed")
        result[str(parent)] = _artifact_stat(info, ancestor=True)
    return result


def _artifact_tree(runtime: Path, expected: dict | None = None) -> dict:
    """Capture code metadata, or compare a pinned tree without listing it anew.

    Directory FDs keep every traversal on the observed physical tree. Symlink
    entries are lexical observations only: mutable state targets are not read.
    Parent ctime/mtime detects additions, including inside empty directories.
    """
    ancestors = _artifact_ancestors(runtime)
    entries = {}
    children = {}
    if expected is not None:
        if ancestors != expected["ancestors"]:
            raise ValueError("runtime_admission_artifact_changed")
        for relative in expected["entries"]:
            if relative != ".":
                parent = str(PurePosixPath(relative).parent)
                children.setdefault(parent, []).append(PurePosixPath(relative).name)

    def record(relative: str, info: os.stat_result, link: str | None = None) -> None:
        value = {"stat": _artifact_stat(info), "link": link}
        if expected is not None and expected["entries"].get(relative) != value:
            raise ValueError("runtime_admission_artifact_changed")
        entries[relative] = value

    def visit(descriptor: int, relative: str) -> None:
        before = os.fstat(descriptor)
        record(relative, before)
        names = os.listdir(descriptor) if expected is None else children.get(relative, [])
        for name in sorted(names):
            child = name if relative == "." else relative + "/" + name
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                nested = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                try:
                    if _artifact_stat(os.fstat(nested)) != _artifact_stat(info):
                        raise ValueError("runtime_admission_artifact_changed")
                    visit(nested, child)
                finally:
                    os.close(nested)
            elif stat.S_ISREG(info.st_mode):
                record(child, info)
            elif stat.S_ISLNK(info.st_mode) and child in {"config", "data", ".env"}:
                record(child, info, os.readlink(name, dir_fd=descriptor))
            else:
                raise ValueError("runtime_admission_artifact_changed")
        if _artifact_stat(os.fstat(descriptor)) != _artifact_stat(before):
            raise ValueError("runtime_admission_artifact_changed")

    descriptor = os.open(runtime, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        visit(descriptor, ".")
        if _artifact_stat(runtime.lstat()) != entries["."]["stat"]:
            raise ValueError("runtime_admission_artifact_changed")
    finally:
        os.close(descriptor)
    result = {"ancestors": ancestors, "entries": entries}
    if _artifact_ancestors(runtime) != ancestors or (expected is not None and result != expected):
        raise ValueError("runtime_admission_artifact_changed")
    return result


def _artifact_read(runtime: Path, relative: str, observation: dict) -> bytes:
    """Read the same nofollow leaf whose metadata brackets full validation."""
    descriptors = []
    directories = []
    try:
        descriptor = os.open(runtime, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(descriptor)
        directories.append((descriptor, "."))
        parent = PurePosixPath(".")
        for component in PurePosixPath(relative).parts[:-1]:
            descriptor = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            descriptors.append(descriptor)
            parent /= component
            directories.append((descriptor, str(parent)))
        for directory, name in directories:
            if _artifact_stat(os.fstat(directory)) != observation["entries"][name]["stat"]:
                raise ValueError("runtime_admission_artifact_changed")
        leaf = PurePosixPath(relative).name
        descriptor = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directories[-1][0])
        descriptors.append(descriptor)
        expected = observation["entries"][relative]["stat"]
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _artifact_stat(before) != expected:
            raise ValueError("runtime_admission_artifact_changed")
        if relative == "release.json" and before.st_size > 16 * 1024 * 1024:
            raise ValueError("runtime_admission_unsafe_record")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            value = stream.read()
        if (_artifact_stat(os.fstat(descriptor)) != expected
                or _artifact_stat(os.stat(leaf, dir_fd=directories[-1][0], follow_symlinks=False)) != expected):
            raise ValueError("runtime_admission_artifact_changed")
        for directory, name in directories:
            if _artifact_stat(os.fstat(directory)) != observation["entries"][name]["stat"]:
                raise ValueError("runtime_admission_artifact_changed")
        if (_artifact_stat(runtime.lstat()) != observation["entries"]["."]["stat"]
                or _artifact_ancestors(runtime) != observation["ancestors"]):
            raise ValueError("runtime_admission_artifact_changed")
        return value
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _checked_slot_live(root: Path, name: str, artifacts: dict) -> tuple[Path, Path, dict]:
    """Use only a prior full observation; changes never refresh it in place."""
    target = root / "slots" / _slot_name(name)
    runtime = target / "runtime"
    observation = artifacts.get(str(runtime))
    if observation is None:
        raise ValueError("runtime_admission_artifact_missing")
    _artifact_tree(runtime, observation)
    manifest = json.loads(_artifact_read(runtime, "release.json", observation))
    if type(manifest) is not dict:
        raise ValueError("runtime_admission_record_schema")
    for directory in ("config", "data"):
        if ((runtime / directory).resolve() != target / "state" / directory
                or not (runtime / directory).is_dir()):
            raise ValueError("Runtime and state do not belong to the same slot")
    if (runtime / ".env").is_symlink() and (runtime / ".env").resolve() != target / "state" / ".env":
        raise ValueError("Environment does not belong to the same slot")
    return runtime, _safe_directory(target / "state"), manifest


def _checked_slot(root: Path, name: str, *, artifacts: dict | None = None) -> tuple[Path, Path, dict]:
    """Validate installed code and same-slot roots without selecting the slot."""
    name = _slot_name(name)
    target = root / "slots" / name
    if target.resolve(strict=True) != target:
        raise ValueError("Slot must not redirect to another directory")
    runtime = target / "runtime"
    observation = _artifact_tree(runtime) if artifacts is not None else None
    manifest = (_read_json(runtime / "release.json") if observation is None
                else json.loads(_artifact_read(runtime, "release.json", observation)))
    if type(manifest) is not dict:
        raise ValueError("runtime_admission_record_schema")
    files = {}
    for filename in manifest["files"]:
        path = PurePosixPath(filename)
        if path.is_absolute() or ".." in path.parts or path.parts[0] not in {"src", "scripts", "tools"}:
            raise ValueError("Unsafe installed runtime path")
        installed = runtime / filename
        if installed.resolve() != installed or installed.suffix == ".py":
            raise ValueError("Installed runtime integrity mismatch")
        files[filename] = (installed.read_bytes() if observation is None
                           else _artifact_read(runtime, filename, observation))
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
    state = _safe_directory(target / "state")
    if observation is not None:
        _artifact_tree(runtime, observation)
        previous = artifacts.get(str(runtime))
        if previous is not None and previous != observation:
            raise ValueError("runtime_admission_artifact_changed")
        artifacts[str(runtime)] = observation
    return runtime, state, manifest


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


def _binding_snapshot(root: Path, name: str, *, artifacts: dict | None = None) -> dict:
    runtime, state, manifest = (_checked_slot(root, name) if artifacts is None
                                else _checked_slot(root, name, artifacts=artifacts))
    _require_copy_not_invalidated(root, state)
    _require_copy_not_observed(root, state)
    _require_copy_not_observed(root, runtime)
    _assert_writers_closed(state)
    roots, schemas, versions = _state_layout(state)
    ownership = state / ".kza-target-ownership.json"
    if ownership.is_symlink():
        raise ValueError("runtime_admission_ownership_redirected")
    return {"schema_version": 1, "slot": name, "runtime_root": str(runtime), "state_root": str(state),
            "release_sha256": hashlib.sha256((runtime / "release.json").read_bytes() if artifacts is None
                else _artifact_read(runtime, "release.json", artifacts[str(runtime)])).hexdigest(),
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


def _validate_closure_header(binding_id: str, proof: dict) -> dict:
    """Validate the original closure shape without reading external state."""
    if (set(proof) != {"schema_version", "binding_id", "running_head", "receipt_sha256", "child_pid"}
            or proof["schema_version"] != 1 or proof["binding_id"] != binding_id
            or type(proof["running_head"]) is not dict
            or set(proof["running_head"]) != _HEAD_KEYS
            or proof["running_head"]["status"] != "running"):
        raise ValueError("runtime_admission_source_close_proof")
    return proof["running_head"]


def _validate_closure_receipt(proof: dict, receipt: dict) -> None:
    """Keep the exact original closed-resource receipt comparison shared."""
    running = proof["running_head"]
    expected = {"schema_version": 1, "binding_id": running["binding_id"],
                "generation": running["generation"], "run_id": running["run_id"],
                "child_pid": proof["child_pid"], "status": "closed",
                "resources": dict.fromkeys(_CLOSED_RESOURCES, True)}
    if receipt != expected:
        raise ValueError("runtime_admission_source_close_proof")


def _validate_closure_digest(proof: dict, receipt_sha256: str) -> None:
    if receipt_sha256 != proof["receipt_sha256"]:
        raise ValueError("runtime_admission_source_close_proof")


def _closure_proof(root: Path, binding_id: str) -> dict:
    proof = _read_json(root / "admission" / "closures" / (binding_id + ".json"))
    running = _validate_closure_header(binding_id, proof)
    receipt_path = root / "admission" / "receipts" / (running["run_id"] + ".json")
    receipt = _read_json(receipt_path)
    _validate_closure_receipt(proof, receipt)
    _validate_closure_digest(proof, hashlib.sha256(receipt_path.read_bytes()).hexdigest())
    return proof


_RECORD_KEYS = frozenset({"schema_version", "destination_state", "destination_identity",
                          "source_binding", "source_close_sha256", "targets"})
# U3b-0b provenance sidecar, admission/delegation-chains/<record capability>.json.
# Records stay v1 and name the physical owner (flattened); the sidecar proves
# how the grantor came to hold each target: owner first, grantor last.
_CHAIN_KEYS = frozenset({"schema_version", "record_sha256", "destination_state", "destination_identity",
                         "grantor", "chains"})
_HOP_KEYS = frozenset({"binding_id", "close_sha256", "closure", "state_root", "state_identity", "via"})
# Every distinct owner state is re-snapshotted in full by every validation; past
# this many owners a grant is refused until compaction is decided (U3b-0b D1).
_DELEGATION_OWNER_LIMIT = 8


def _hex64(value: object) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _chain_path(root: Path, capability: str) -> Path:
    return root / "admission" / "delegation-chains" / (capability + ".json")


def _read_delegation_record(root: Path, capability: str) -> dict:
    """Read one content-addressed v1 delegation record named by a chain hop."""
    if not _hex64(capability):
        raise ValueError("runtime_admission_delegation_chain")
    path = root / "admission" / "delegations" / (capability + ".json")
    _safe_directory(path.parent)
    try:
        record = _read_json(path)
    except FileNotFoundError:
        raise ValueError("runtime_admission_delegation_chain") from None
    if (_digest(record) != capability or set(record) != _RECORD_KEYS or record["schema_version"] != 1
            or type(record["targets"]) is not dict):
        raise ValueError("runtime_admission_delegation_chain")
    return record


def _chain_shape(record: dict, chain: dict, hops: object) -> None:
    """Owner first, grantor last; hops pairwise distinct and none the destination."""
    if type(hops) is not list or not hops:
        raise ValueError("runtime_admission_delegation_chain")
    for index, hop in enumerate(hops):
        if (type(hop) is not dict or set(hop) != _HOP_KEYS or not _hex64(hop["binding_id"])
                or not _hex64(hop["close_sha256"]) or type(hop["closure"]) is not dict
                or type(hop["state_root"]) is not str or type(hop["state_identity"]) is not dict
                or set(hop["state_identity"]) != {"device", "inode"}
                or any(type(item) is not int for item in hop["state_identity"].values())
                or (hop["via"] is not None if index == 0 else not _hex64(hop["via"]))):
            raise ValueError("runtime_admission_delegation_chain")
    owner, grantor = hops[0], hops[-1]
    if (owner["binding_id"] != record["source_binding"]
            or owner["close_sha256"] != record["source_close_sha256"]
            or grantor["binding_id"] != chain["grantor"]["binding_id"]
            or grantor["close_sha256"] != chain["grantor"]["close_sha256"]):
        raise ValueError("runtime_admission_delegation_chain")
    states = [hop["state_root"] for hop in hops]
    identities = [_canonical_json(hop["state_identity"]) for hop in hops]
    if (len({hop["binding_id"] for hop in hops}) != len(hops) or len(set(states)) != len(hops)
            or len(set(identities)) != len(hops) or record["destination_state"] in states
            or _canonical_json(record["destination_identity"]) in identities):
        raise ValueError("runtime_admission_delegation_cycle")


def _read_chain(root: Path, record: dict) -> dict:
    """Read and shape-check the provenance sidecar of one delegation record."""
    capability = _digest(record)
    path = _chain_path(root, capability)
    if not (path.exists() or path.is_symlink()):
        raise ValueError("runtime_admission_delegation_chain_missing")
    _safe_directory(path.parent)
    chain = _read_json(path)
    if (set(chain) != _CHAIN_KEYS or chain["schema_version"] != 1 or chain["record_sha256"] != capability
            or chain["destination_state"] != record["destination_state"]
            or chain["destination_identity"] != record["destination_identity"]
            or type(chain["grantor"]) is not dict or set(chain["grantor"]) != {"binding_id", "close_sha256"}
            or type(chain["chains"]) is not dict or set(chain["chains"]) != set(record["targets"])):
        raise ValueError("runtime_admission_delegation_chain")
    for hops in chain["chains"].values():
        _chain_shape(record, chain, hops)
    return chain


def _verify_chain_hops(root: Path, target: dict, hops: list) -> dict:
    """Anchor every hop on immutable evidence and prove each later hop by its own grant.

    Intermediate hops are checked against their embedded closure proof and the
    immutable per-run receipt it names, never the mutable closures/<binding>
    file, so rerunning an intermediate does not revoke. Each later hop must be
    the destination of the record named by 'via', whose own sidecar carries
    exactly the preceding prefix. Returns the via records and sidecars.
    """
    vias = {}
    for index, hop in enumerate(hops):
        binding = read_binding(root, hop["binding_id"])
        if (binding["state_root"] != hop["state_root"] or binding["state_identity"] != hop["state_identity"]
                or _digest(hop["closure"]) != hop["close_sha256"]):
            raise ValueError("runtime_admission_delegation_chain")
        if index == 0:
            # The owner's closure is its live closures/<owner> proof, checked by the
            # caller against record.source_close_sha256 (0a semantics).
            continue
        running = _validate_closure_header(hop["binding_id"], hop["closure"])
        if type(running["run_id"]) is not str or re.fullmatch("[0-9a-f]{32}", running["run_id"]) is None:
            raise ValueError("runtime_admission_delegation_chain")
        run = read_binding(root, running["binding_id"])
        if (run["slot"] != binding["slot"] or run["state_root"] != hop["state_root"]
                or run["state_identity"] != hop["state_identity"]):
            # F1: the embedded closure must come from a run of this hop's own slot and
            # state, not another run's real closure relabelled with this binding id.
            raise ValueError("runtime_admission_delegation_chain")
        receipt_path = root / "admission" / "receipts" / (running["run_id"] + ".json")
        _validate_closure_receipt(hop["closure"], _read_json(receipt_path))
        _validate_closure_digest(hop["closure"], hashlib.sha256(receipt_path.read_bytes()).hexdigest())
        via = _read_delegation_record(root, hop["via"])
        if (via["destination_state"] != hop["state_root"]
                or via["destination_identity"] != hop["state_identity"]
                or via["source_binding"] != hops[0]["binding_id"]
                or via["source_close_sha256"] != hops[0]["close_sha256"]
                or via["targets"].get(target["id"]) != target):
            raise ValueError("runtime_admission_delegation_chain")
        via_chain = _read_chain(root, via)
        if via_chain["chains"].get(target["id"]) != hops[:index]:
            raise ValueError("runtime_admission_delegation_chain")
        vias[hop["via"]] = {"record": via, "chain": via_chain}
    return vias


def _owner_observation(root: Path, owner_id: str, *, artifacts: dict | None = None,
                       cache: dict | None = None) -> dict:
    """Observe one owner state once per validation pass (U3b-0b F2).

    The full owner snapshot and inventory are the expensive part of every
    delegated-target check; a pass validating several targets of the same owner
    reuses one observation instead of re-snapshotting per target.
    """
    if cache is not None and owner_id in cache:
        return cache[owner_id]
    source = read_binding(root, owner_id)
    source_state = Path(source["state_root"])
    value = {"source": source, "marker": _read_json(source_state / _MARKER), "live": False}
    if value["marker"].get("binding_id") == owner_id:
        observed = (_binding_snapshot(root, source["slot"]) if artifacts is None
                    else _binding_snapshot(root, source["slot"], artifacts=artifacts))
        if observed == source:
            proof = _closure_proof(root, owner_id)
            running = proof.get("running_head")
            if type(running) is not dict:
                raise ValueError("runtime_admission_delegation_chain")
            run = read_binding(root, running.get("binding_id"))
            if run["slot"] != source["slot"] or run["state_root"] != source["state_root"]:
                # F1: the owner's closure must belong to a run of the owner's own slot.
                raise ValueError("runtime_admission_delegation_chain")
            policy = None
            if source["ownership_sha256"] is not None:
                policy_path = source_state / ".kza-target-ownership.json"
                policy = _read_json(policy_path)
                if (hashlib.sha256(policy_path.read_bytes()).hexdigest() != source["ownership_sha256"]
                        or type(policy.get("delegated_targets")) is not list):
                    raise ValueError("runtime_admission_delegation_chain")
            value.update(live=True, proof=proof, inventory=_target_inventory(source_state), policy=policy)
    if cache is not None:
        cache[owner_id] = value
    return value


def _delegation_candidate(root: Path, record: dict, chain: dict, target: dict, sources: list,
                          *, artifacts: dict | None = None, cache: dict | None = None) -> dict | None:
    """Validate one record and its chain for one target; None when its owner is stale."""
    hops = chain["chains"][target["id"]]
    if any(hop["binding_id"] not in sources for hop in hops):
        raise ValueError("runtime_admission_delegation_chain")
    vias = _verify_chain_hops(root, target, hops)
    owner = _owner_observation(root, record["source_binding"], artifacts=artifacts, cache=cache)
    source, marker = owner["source"], owner["marker"]
    if not owner["live"] or _digest(owner["proof"]) != record["source_close_sha256"]:
        return None
    source_state = Path(source["state_root"])
    if owner["inventory"].get(target["id"]) != target:
        raise ValueError("runtime_admission_delegation_source_target_changed")
    for value in _descriptor_paths(target["descriptor"]):
        path = Path(value)
        if not path.is_relative_to(source_state) or path.resolve() != path:
            raise ValueError("runtime_admission_delegation_foreign_source")
    if owner["policy"] is not None and target["id"] in owner["policy"]["delegated_targets"]:
        # The owner must own the target, never merely hold it delegated.
        raise ValueError("runtime_admission_delegation_chain")
    return {"record": record, "source": source, "marker": marker, "proof": owner["proof"],
            "capability": _digest(record), "chain": chain, "vias": vias}


def _delegation_states(root: Path) -> tuple[set, set]:
    """State paths and identities that already served as a delegation owner or hop."""
    paths, identities = set(), set()
    directory = root / "admission" / "delegations"
    if not directory.exists():
        return paths, identities
    _safe_directory(directory)
    for path in sorted(directory.glob("*.json")):
        record = _read_json(path)
        if path.stem != _digest(record) or set(record) != _RECORD_KEYS:
            raise ValueError("runtime_admission_delegation_changed")
        owner = read_binding(root, record["source_binding"])
        paths.add(owner["state_root"])
        identities.add(_canonical_json(owner["state_identity"]))
        sidecar = _chain_path(root, path.stem)
        if sidecar.exists() or sidecar.is_symlink():
            for hops in _read_chain(root, record)["chains"].values():
                paths.update(hop["state_root"] for hop in hops)
                identities.update(_canonical_json(hop["state_identity"]) for hop in hops)
    return paths, identities


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


def _validate_delegated_target(root: Path, state: Path, target: dict, sources: list,
                               *, artifacts: dict | None = None, cache: dict | None = None) -> dict:
    """Select the record that grants one delegated target, with its proven provenance.

    Every honored record must carry a chain sidecar (U3b-0b). A record whose
    owner state is stale is skipped (0a live semantics); a structural defect in
    the record, its sidecar or any hop raises.
    """
    for record in _delegation_records(root, state):
        if (set(record) != _RECORD_KEYS or record["schema_version"] != 1 or type(record["targets"]) is not dict):
            raise ValueError("runtime_admission_delegation_schema")
        if record["destination_identity"] != _identity(state) or record["source_binding"] not in sources:
            continue
        if target["id"] not in record["targets"]:
            continue
        if record["targets"][target["id"]] != target:
            raise ValueError("runtime_admission_delegation_target_changed")
        selected = _delegation_candidate(root, record, _read_chain(root, record), target, sources,
                                         artifacts=artifacts, cache=cache)
        if selected is not None:
            return selected
    raise ValueError("runtime_admission_delegation_missing")


def _validate_ownership_history(root: Path, policy: dict) -> None:
    sources = policy["source_bindings"]
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


def _target_ownership_validation(
    root: Path, binding: dict, *, allow_new_local: bool = False, capture: bool = False,
    artifacts: dict | None = None,
) -> dict:
    """Reject foreign active paths; historical journals confer no execution grant."""
    root = _safe_directory(Path(root).resolve(strict=True))
    state = _safe_directory(Path(binding["state_root"]))
    targets = _target_inventory(state)
    policy = None
    delegations = {}
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
        owners = {}  # One observation per owner for this whole pass (U3b-0b F2).
        for identity in delegated:
            delegations[identity] = (_validate_delegated_target(root, state, targets[identity], sources,
                                                                cache=owners)
                if artifacts is None else _validate_delegated_target(
                    root, state, targets[identity], sources, artifacts=artifacts, cache=owners))
        _validate_ownership_history(root, policy)
    for identity in owned:
        for value in _descriptor_paths(targets[identity]["descriptor"]):
            path = Path(value)
            if not path.is_relative_to(state) or path.resolve() != path:
                raise ValueError("runtime_admission_foreign_target")
    return {"targets": targets, "policy": policy, "delegations": delegations,
            "delegation_records": list(_delegation_records(root, state)) if capture and delegations else []}


def validate_target_ownership(root: Path, binding: dict, *, allow_new_local: bool = False) -> None:
    """Reject foreign active paths; historical journals confer no execution grant."""
    with _ADMISSION_VALIDATION_LOCK:
        _target_ownership_validation(root, binding, allow_new_local=allow_new_local)


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


def _manual_owner(root: Path, binding: dict, target: dict, cache: dict | None = None):
    module = _manual_module()
    state = Path(binding["state_root"])
    descriptor = target["descriptor"]
    directory = Path(descriptor["root"])
    owner = binding
    if not directory.is_relative_to(state):
        policy = _read_json(state / ".kza-target-ownership.json")
        if target["id"] not in policy["delegated_targets"]:
            raise ValueError("privacy_authority_stale")
        _validate_delegated_target(root, state, target, policy["source_bindings"], cache=cache)
        candidates = {}
        for record in _delegation_records(root, state):
            if (record["source_binding"] not in policy["source_bindings"]
                    or record["destination_identity"] != _identity(state)
                    or record["targets"].get(target["id"]) != target):
                continue
            observed = _owner_observation(root, record["source_binding"], cache=cache)
            source = observed["source"]
            if (not directory.is_relative_to(Path(source["state_root"])) or not observed["live"]
                    or _digest(observed["proof"]) != record["source_close_sha256"]
                    or observed["inventory"].get(target["id"]) != target):
                continue
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
    cache = {}  # One owner observation per capture (U3b-0b F2).
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
        owners[identity] = _manual_owner(root, binding, target, cache)
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


_RECOVERY_KEYS = frozenset({"schema_version", "closed_head", "head_before", "run_id", "new_binding_id",
                            "previous", "mode", "class", "boot_id", "at", "checks", "confirmation"})


def recovery_record_path(root: Path, head: dict) -> Path:
    """Where the immutable recovery record of one adopted closed head lives."""
    return Path(root) / "admission" / "recoveries" / (_digest(head) + ".json")


def _read_recovery_record(root: Path, head: dict) -> dict:
    try:
        record = _read_json(recovery_record_path(root, head))
    except FileNotFoundError:
        raise ValueError("runtime_admission_recovery_record_missing") from None
    before = record.get("head_before")
    if (set(record) != _RECOVERY_KEYS or type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or _canonical_json(record["closed_head"]) != _canonical_json(head)
            or type(before) is not dict or before.get("status") not in {"running", "unknown"}
            or not _manual_head_shape(before, before["status"])
            or record["run_id"] != before["run_id"]
            or record["mode"] not in {"automatic", "operator"}
            or record["class"] not in {"crash", "boot", "operator"}
            or type(record["at"]) is not float or not math.isfinite(record["at"])
            or type(record["checks"]) is not dict
            or type(record["previous"]) is not dict):
        raise ValueError("runtime_admission_recovery_record_schema")
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
            or record["kind"] not in {"bootstrap", "selection", "closure", "recovery"}):
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
            if prior["kind"] not in {"bootstrap", "selection", "recovery"} or _has_authority(binding):
                raise ValueError("runtime_admission_manual_bootstrap_invalid")
            _require_manual_equivalent(prior["capture"], record["capture"])
            return record
    if record["kind"] == "recovery":
        # U3b-0d: an adopted state after a run without a usable receipt. The
        # immutable recovery record binds the open head it replaced; the
        # authority seal is the last real closure's, so the capture must stay
        # equivalent to it and delegation still needs a later clean close.
        recovery = _read_recovery_record(root, head)
        before = recovery["head_before"]
        distance = before["generation"] - prior["head"]["generation"]
        if (recovery["previous"] != previous or recovery["new_binding_id"] != head["binding_id"]
                or before["binding_id"] != prior["head"]["binding_id"]
                or before["generation"] + 1 != head["generation"]
                or distance != (1 if before["status"] == "running" else 2)
                or record["authority_seal"] != prior["authority_seal"]):
            raise ValueError("runtime_admission_manual_companion_changed")
        if record["authority_seal"] is None:
            if prior["kind"] not in {"bootstrap", "selection", "recovery"} or _has_authority(binding):
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
                         previous: dict | None, proof: dict | None = None,
                         recovery: dict | None = None) -> dict:
    """Write immutable metadata before its corresponding V1 CLOSED CAS."""
    if recovery is not None:
        if (proof is not None or previous is None or type(expected) is not dict
                or expected.get("status") not in {"running", "unknown"}
                or not _manual_head_shape(expected, expected["status"])
                or _canonical_json(recovery.get("head_before")) != _canonical_json(expected)):
            raise ValueError("runtime_admission_manual_companion_schema")
    elif expected is not None and not _manual_head_shape(expected, "running" if proof is not None else "closed"):
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
    elif recovery is not None:
        if _canonical_json(recovery["closed_head"]) != _canonical_json(head):
            raise ValueError("runtime_admission_manual_companion_schema")
        if previous["authority_seal"] is None:
            if _has_authority(binding):
                raise ValueError("runtime_admission_recovery_unsealed_authority")
            _require_manual_equivalent(previous["capture"], capture)
        else:
            _require_manual_equivalent(_read_manual_seal(root, previous["authority_seal"])["capture"], capture)
        kind, seal_id = "recovery", previous["authority_seal"]
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


def _seal_local_targets(root: Path, binding: dict, *, run_id: str | None = None) -> None:
    """Adopt only locally verified registrations after actual owned closure.

    With ``run_id`` an immutable intent (pre- and post-seal policy digests) is
    recorded before the policy is rewritten, so a controller killed after the
    rewrite can resume the seal from the receipt (U3b-0d R-a).
    """
    validate_target_ownership(root, binding, allow_new_local=True)
    if binding["ownership_sha256"] is None:
        return
    state = Path(binding["state_root"])
    path = state / ".kza-target-ownership.json"
    policy = _read_json(path)
    policy["owned_targets"] = sorted(set(_target_inventory(state)) - set(policy["delegated_targets"]))
    if run_id is not None:
        sealed = hashlib.sha256(_canonical_json(policy) + b"\n").hexdigest()
        if sealed != binding["ownership_sha256"]:
            _atomic_record(_seal_intent_path(root, run_id), {
                "schema_version": 1, "run_id": run_id, "pre_seal_ownership_sha256": binding["ownership_sha256"],
                "sealed_ownership_sha256": sealed}, immutable=True)
    _atomic_record(path, policy)


def _seal_intent_path(root: Path, run_id: str) -> Path:
    return Path(root) / "admission" / "seal-intents" / (run_id + ".json")


def _closed_state_matches(root: Path, run_id: str, pre_seal: dict, record: dict) -> bool:
    """The pre-seal binding still describes the closed state the child recorded.

    A resumed seal may find the ownership policy already rewritten by the
    interrupted controller; its intent record then restores the pre-seal digest.
    """
    if closed_state_sha256(pre_seal) == record["closed_state_sha256"]:
        return True
    path = _seal_intent_path(root, run_id)
    if not path.exists() and not path.is_symlink():
        return False
    intent = _read_json(path)
    if (set(intent) != {"schema_version", "run_id", "pre_seal_ownership_sha256", "sealed_ownership_sha256"}
            or intent["schema_version"] != 1 or intent["run_id"] != run_id
            or pre_seal["ownership_sha256"] != intent["sealed_ownership_sha256"]):
        return False
    return closed_state_sha256(dict(pre_seal, ownership_sha256=intent["pre_seal_ownership_sha256"])) \
        == record["closed_state_sha256"]


def expected_receipt(running: dict, child_pid: int) -> dict:
    """The exact schema-1 receipt a run's child writes."""
    return {"schema_version": 1, "binding_id": running["binding_id"], "generation": running["generation"],
            "run_id": running["run_id"], "child_pid": child_pid, "status": "closed",
            "resources": dict.fromkeys(_CLOSED_RESOURCES, True)}


def seal_admission_run(root: Path, running: dict, child_pid: int, previous_manual: dict) -> dict:
    """Seal one reaped run from its exact receipt (seal rule, design v2 M5 / v3 N2).

    The caller holds the root lease and has reaped the child. The seal needs
    the exact receipt, the receipt-state record when the slot declares one
    (review 0d3 K2), a state manifest recomputed under the root lock before
    any database image (K7, N2), and a closed-state digest equal to the
    record's. Every step is idempotent, so a controller killed mid-seal
    resumes here from the same receipt (R-a). Returns the closed head.
    """
    root = Path(root)
    run_id = running["run_id"]
    receipt_path = root / "admission" / "receipts" / (run_id + ".json")
    try:
        receipt = _read_json(receipt_path)
    except FileNotFoundError:
        raise ValueError("runtime_admission_close_receipt_missing") from None
    if receipt != expected_receipt(running, child_pid):
        raise ValueError("runtime_admission_close_receipt_mismatch")
    binding = read_binding(root, running["binding_id"])
    closed_record = read_receipt_state(root, receipt)
    if closed_record is None and receipt_state_required(binding):
        raise ValueError("runtime_admission_receipt_state")
    with _locked_root(root):
        if read_admission_head(root) != running:
            raise ValueError("runtime_admission_head_changed")
        if closed_record is not None:
            # Manifest first: lstat only, before any database image (N2, K7).
            verify_receipt_state(closed_record, Path(binding["state_root"]))
        pre_seal = _binding_snapshot(root, binding["slot"])
        if closed_record is not None and not _closed_state_matches(root, run_id, pre_seal, closed_record):
            raise ValueError("runtime_admission_closed_state_mismatch")
        _seal_local_targets(root, pre_seal, run_id=run_id)
        closed = _binding_snapshot(root, binding["slot"])
        validate_target_ownership(root, closed)
        identity = _store_binding(root, closed)
        proof = {"schema_version": 1, "binding_id": identity, "running_head": running,
                 "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                 "child_pid": child_pid}
        _atomic_record(root / "admission" / "closures" / (identity + ".json"), proof)
        _prepare_manual_head(root, running, closed, previous=previous_manual, proof=proof)
        _write_marker(root, identity, closed)
        return _cas_head(root, running, identity, "closed")


def manual_reference(previous: dict) -> dict:
    """The previous-companion reference a manual head record carries."""
    return {"head_sha256": _digest(previous["head"]), "companion_sha256": _digest(previous)}


def previous_closed_head(head: dict) -> dict:
    """The closed head an open (running or unknown) head was started from."""
    distance = 1 if head["status"] == "running" else 2
    if head["status"] not in {"running", "unknown"} or head["generation"] <= distance:
        raise ValueError("runtime_admission_head_schema")
    return {"schema_version": 1, "generation": head["generation"] - distance,
            "binding_id": head["binding_id"], "status": "closed", "run_id": None}


def read_manual_head(root: Path, head: dict) -> dict:
    """Read and validate one closed head's manual companion without re-snapshotting."""
    return _read_manual_head(Path(root), head)


def adopt_recovered_state(root: Path, head_before: dict, recovery: dict) -> dict:
    """Adopt the current state of an open head after recovery checks (R-b).

    The caller holds the lease and made the policy decision. ``recovery``
    carries every record field except closed_head, new_binding_id, previous
    and head_before, which are derived here under the root lock. Writes the
    binding, the immutable recovery record, a 'recovery' manual head, the
    marker and the CAS to closed. A recovered head has no closure proof, so
    delegation stays refused until a later clean close.
    """
    root = Path(root)
    with _locked_root(root):
        if read_admission_head(root) != head_before:
            raise ValueError("runtime_admission_head_changed")
        prior_head = previous_closed_head(head_before)
        previous = _read_manual_head(root, prior_head)
        slot = read_binding(root, head_before["binding_id"])["slot"]
        binding = _binding_snapshot(root, slot)
        validate_target_ownership(root, binding, allow_new_local=True)
        identity = _store_binding(root, binding)
        closed_head = {"schema_version": 1, "generation": head_before["generation"] + 1,
                       "binding_id": identity, "status": "closed", "run_id": None}
        record = dict(recovery, schema_version=1, closed_head=closed_head, head_before=head_before,
                      run_id=head_before["run_id"], new_binding_id=identity,
                      previous=manual_reference(previous))
        if set(record) != _RECOVERY_KEYS:
            raise ValueError("runtime_admission_recovery_record_schema")
        _atomic_record(recovery_record_path(root, closed_head), record, immutable=True)
        _prepare_manual_head(root, head_before, binding, previous=previous, recovery=record)
        _write_marker(root, identity, binding)
        return _cas_head(root, head_before, identity, "closed")


def _authorize_target_delegation(root: Path, destination_state: Path, source_binding: str,
                                 target_ids: list[str]) -> list[str]:
    """Grant exact held targets to one copied state, owner-flattened with provenance.

    Every check (source, destination, targets, cycles, owner limit, collisions
    and the full chain validation of the computed records) runs before the
    first write. Returns the sorted capabilities, one record per owner.
    """
    head = read_admission_head(root)
    if head is None or head["status"] != "closed" or head["binding_id"] != source_binding:
        raise ValueError("runtime_admission_delegation_source_not_current")
    source = verify_closed_binding(root, source_binding)
    # Inherited chains come only from the verified source's own selection.
    held = _target_ownership_validation(root, source)["delegations"]
    source_manual = _verify_manual_head(root, head, source)
    proof = _closure_proof(root, source_binding)
    source_state = Path(source["state_root"])
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
    if set(held) - set(target_ids):
        # A held delegation is never silently converted into destination ownership.
        raise ValueError("runtime_admission_delegation_incomplete")
    source_targets = _target_inventory(source_state)
    destination_targets = _target_inventory(state)
    grantor = {"binding_id": source_binding, "close_sha256": _digest(proof), "closure": proof,
               "state_root": source["state_root"], "state_identity": source["state_identity"]}
    grants = {}
    for identity in target_ids:
        target = source_targets.get(identity)
        if target is None or destination_targets.get(identity) != target:
            raise ValueError("runtime_admission_delegation_target_mismatch")
        if identity in held:
            selected = held[identity]
            hops = [*selected["chain"]["chains"][identity], dict(grantor, via=selected["capability"])]
        else:
            for value in _descriptor_paths(target["descriptor"]):
                path = Path(value)
                if not path.is_relative_to(source_state) or path.resolve() != path:
                    raise ValueError("runtime_admission_delegation_foreign_source")
            hops = [dict(grantor, via=None)]
        grants.setdefault((hops[0]["binding_id"], hops[0]["close_sha256"]), {})[identity] = (target, hops)
    destination_identity = _identity(state)
    used_paths, used_identities = _delegation_states(root)
    for items in grants.values():
        for _, hops in items.values():
            used_paths.update(hop["state_root"] for hop in hops)
            used_identities.update(_canonical_json(hop["state_identity"]) for hop in hops)
    if str(state) in used_paths or _canonical_json(destination_identity) in used_identities:
        raise ValueError("runtime_admission_delegation_cycle")
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
    existing = [record for record in _delegation_records(root, state)
                if record.get("destination_identity") == destination_identity]
    for identity in set(policy["delegated_targets"]) - set(target_ids):
        if not any(type(record.get("targets")) is dict and identity in record["targets"] for record in existing):
            raise ValueError("runtime_admission_delegation_incomplete")
    if len({owner for owner, _ in grants} | {record.get("source_binding") for record in existing}) \
            > _DELEGATION_OWNER_LIMIT:
        raise ValueError("runtime_admission_delegation_owner_limit")
    delegated = set(policy["delegated_targets"]) | set(target_ids)
    owned = set(destination_targets) - delegated
    for identity in owned:
        for value in _descriptor_paths(destination_targets[identity]["descriptor"]):
            path = Path(value)
            if not path.is_relative_to(state) or path.resolve() != path:
                raise ValueError("runtime_admission_foreign_target")
    sources = sorted(set(policy["source_bindings"])
                     | {hop["binding_id"] for items in grants.values() for _, hops in items.values()
                        for hop in hops})
    planned = []
    for (owner, close), items in sorted(grants.items()):
        record = {"schema_version": 1, "destination_state": str(state),
                  "destination_identity": destination_identity, "source_binding": owner,
                  "source_close_sha256": close,
                  "targets": {identity: target for identity, (target, _) in sorted(items.items())}}
        capability = _digest(record)
        chain = {"schema_version": 1, "record_sha256": capability, "destination_state": str(state),
                 "destination_identity": destination_identity,
                 "grantor": {"binding_id": source_binding, "close_sha256": grantor["close_sha256"]},
                 "chains": {identity: hops for identity, (_, hops) in sorted(items.items())}}
        for hops in chain["chains"].values():
            _chain_shape(record, chain, hops)
        for path, value in ((root / "admission" / "delegations" / (capability + ".json"), record),
                            (_chain_path(root, capability), chain)):
            if (path.exists() or path.is_symlink()) and _read_json(path) != value:
                raise ValueError("runtime_admission_delegation_conflict")
        planned.append((capability, record, chain))
    owners = {}
    for _, record, chain in planned:
        # The same validator later runs on the written files; nothing is written yet.
        for target in record["targets"].values():
            if _delegation_candidate(root, record, chain, target, sources, cache=owners) is None:
                raise ValueError("runtime_admission_delegation_source_stale")
    for identity in sorted(set(policy["delegated_targets"]) - set(target_ids)):
        # F6: entries carried over from an earlier grant into this destination are
        # validated before writing too, not only by the post-write self-check.
        if identity not in destination_targets:
            raise ValueError("runtime_admission_active_inventory_mismatch")
        _validate_delegated_target(root, state, destination_targets[identity], sources, cache=owners)
    for capability, _, chain in planned:
        _atomic_record(_chain_path(root, capability), chain, immutable=True)
    for capability, record, _ in planned:
        _atomic_record(root / "admission" / "delegations" / (capability + ".json"), record, immutable=True)
    policy["source_bindings"] = sources
    policy["owned_targets"] = sorted(owned)
    policy["delegated_targets"] = sorted(delegated)
    _atomic_record(policy_path, policy)
    destination = _binding_snapshot(root, state.parent.name)
    validate_target_ownership(root, destination)
    _prepare_manual_head(root, head, source, previous=source_manual)
    _cas_head(root, head, source_binding, "closed")
    return sorted(capability for capability, _, _ in planned)


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


# U3b-1a: the owner schema FactLedger.initialize migrates into a pre-owner ledger.
_FACT_OWNER_ADDITION = ("fact_owners", "fact_owner_revisions", "fact_owner_schema")
# U3b-4a-0: the voice-attribution schema FactLedger.initialize migrates into an older ledger.
_VOICE_ATTRIBUTION_ADDITION = ("voice_attribution_consents", "voice_verification_attempts",
                               "voice_attribution_events", "voice_attribution_suspensions",
                               "voice_attribution_control", "voice_attribution_schema")
# U3b-0d3: the deletion ledger FactLedger.initialize (and the dashboard authority) creates.
_DELETION_LEDGER_ADDITION = ("deletion_ledger_v1", "deletion_ledger_salt_v1")
# Each known additive schema group; a derivation may add a whole group, empty.
_SCHEMA_ADDITIONS = (_FACT_OWNER_ADDITION, _VOICE_ATTRIBUTION_ADDITION, _DELETION_LEDGER_ADDITION)


def _expect_empty_schema_additions(expected: dict, derived: dict) -> None:
    """Admit only the empty additive schema groups a derivation migrates into an older ledger.

    Materialization opens the destination ledger with current code, and
    FactLedger.initialize creates each known group (the U3b owner tables, the
    U3b-4a-0 voice-attribution tables) when the source image predates it. Per
    database and per group, that addition is expected only when the source
    contract has none of the group's tables and the derived contract has all
    of them; the expected projection then carries their exact _TABLE_COLUMNS
    contract and, for protected tables, no rows. Any row or partial group still
    mismatches, and an empty owner set pins no further fact, so every other
    projection must still match exactly.
    """
    for relative, tables in expected["tables"].items():
        contract = tables.get("schema_contract_v1")
        target = derived["tables"].get(relative)
        if contract is None or type(target) is not dict or "schema_contract_v1" not in target:
            continue
        for group in _SCHEMA_ADDITIONS:
            if (any(name in contract["columns"] for name in group)
                    or not all(name in target["schema_contract_v1"]["columns"] for name in group)):
                continue
            for name in group:
                contract["columns"][name] = dict(item.split(":") for item in _TABLE_COLUMNS[name].split())
                if name in _PROTECTED_TABLES:
                    tables[name] = _empty_protected_projection(name)


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
    _expect_empty_schema_additions(expected, candidate["authority"]["projection"])
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


def _admission_ticket_metadata(state_root: Path, ticket: dict | str) -> tuple[Path, Path, dict]:
    """Check actual mutable inherited authority and roots on every call."""
    ticket = _ticket_value(ticket)
    root = _safe_directory(Path(ticket["admission_root"]))
    state = _safe_directory(Path(state_root).resolve(strict=True))
    _require_copy_not_invalidated(root, state)
    _require_copy_not_observed(root, state)
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
    _require_copy_not_observed(root, Path(binding["runtime_root"]))
    _check_roots(binding)
    return root, state, binding


def _admission_ticket_core(state_root: Path, ticket: dict | str, *,
                           artifacts: dict | None = None) -> tuple[Path, Path, dict]:
    """Keep full code-byte validation for every new admission/invocation."""
    root, state, binding = _admission_ticket_metadata(state_root, ticket)
    runtime, checked_state, _ = (_checked_slot(root, binding["slot"]) if artifacts is None
        else _checked_slot(root, binding["slot"], artifacts=artifacts))
    if str(runtime) != binding["runtime_root"] or checked_state != state:
        raise ValueError("runtime_admission_code_mismatch")
    if hashlib.sha256((runtime / "release.json").read_bytes() if artifacts is None
                else _artifact_read(runtime, "release.json", artifacts[str(runtime)])).hexdigest() != binding["release_sha256"]:
        raise ValueError("runtime_admission_release_changed")
    return root, state, binding


def _admission_ticket_live_core(state_root: Path, ticket: dict | str,
                                artifacts: dict) -> tuple[Path, Path, dict]:
    """Join fresh mutable authority to code observed by this full validation."""
    root, state, binding = _admission_ticket_metadata(state_root, ticket)
    runtime, checked_state, _ = _checked_slot_live(root, binding["slot"], artifacts)
    if str(runtime) != binding["runtime_root"] or checked_state != state:
        raise ValueError("runtime_admission_code_mismatch")
    if hashlib.sha256(_artifact_read(runtime, "release.json", artifacts[str(runtime)])).hexdigest() != binding["release_sha256"]:
        raise ValueError("runtime_admission_release_changed")
    return root, state, binding


def _validate_source_binding_metadata(root: Path, source: dict, artifacts: dict) -> None:
    """Recheck immutable source identity without reopening its databases."""
    runtime, state, manifest = _checked_slot_live(root, source["slot"], artifacts)
    _require_copy_not_invalidated(root, state)
    _require_copy_not_observed(root, state)
    _require_copy_not_observed(root, runtime)
    if str(runtime) != source["runtime_root"] or str(state) != source["state_root"]:
        raise ValueError("runtime_admission_code_mismatch")
    _check_roots(source)
    if (hashlib.sha256(_artifact_read(runtime, "release.json", artifacts[str(runtime)])).hexdigest() != source["release_sha256"]
            or manifest["git_commit"] != source["code_commit"]
            or manifest["python_magic"] != source["python_magic_hex"]):
        raise ValueError("runtime_admission_release_changed")
    ownership = state / ".kza-target-ownership.json"
    if ownership.is_symlink():
        raise ValueError("runtime_admission_ownership_redirected")
    if source["ownership_sha256"] is None:
        if ownership.exists():
            raise ValueError("runtime_admission_ownership_changed")
    else:
        _read_json(ownership)
        if hashlib.sha256(ownership.read_bytes()).hexdigest() != source["ownership_sha256"]:
            raise ValueError("runtime_admission_ownership_changed")


def _validate_ownership_metadata(root: Path, binding: dict, ownership: dict, artifacts: dict) -> None:
    """Recheck pinned metadata; inventory content was scanned before native entry."""
    state = _safe_directory(Path(binding["state_root"]))
    path = state / ".kza-target-ownership.json"
    policy = ownership["policy"]
    targets, delegations = ownership["targets"], ownership["delegations"]
    if binding["ownership_sha256"] is None:
        if path.exists() or path.is_symlink() or policy is not None or delegations:
            raise ValueError("runtime_admission_ownership_changed")
    else:
        if (_read_json(path) != policy
                or hashlib.sha256(path.read_bytes()).hexdigest() != binding["ownership_sha256"]):
            raise ValueError("runtime_admission_ownership_changed")
        for identity in policy["source_bindings"]:
            read_binding(root, identity)
        _validate_ownership_history(root, policy)
    if delegations:
        # Do not select a different record or infer another grant during a live
        # operation. The selected records came from the complete validator.
        records = list(_delegation_records(root, state))
        if records != ownership["delegation_records"]:
            raise ValueError("runtime_admission_delegation_changed")
        checked_sources = set()
        for identity, selected in delegations.items():
            record = selected["record"]
            target = targets[identity]
            if (record not in records or record["destination_identity"] != _identity(state)
                    or record["source_binding"] not in policy["source_bindings"]
                    or record["targets"].get(identity) != target):
                raise ValueError("runtime_admission_delegation_target_changed")
            source_id = record["source_binding"]
            source = read_binding(root, source_id)
            if source != selected["source"]:
                raise ValueError("runtime_admission_delegation_changed")
            # U3b-0b: the provenance read by the full validation must be unchanged.
            if (selected["capability"] != _digest(record) or _read_chain(root, record) != selected["chain"]
                    or any(_read_delegation_record(root, capability) != via["record"]
                           or _read_chain(root, via["record"]) != via["chain"]
                           for capability, via in selected["vias"].items())):
                raise ValueError("runtime_admission_delegation_changed")
            if source_id not in checked_sources:
                marker = _read_json(Path(source["state_root"]) / _MARKER)
                if marker != selected["marker"] or marker.get("binding_id") != source_id:
                    raise ValueError("runtime_admission_marker_mismatch")
                _validate_source_binding_metadata(root, source, artifacts)
                proof = _closure_proof(root, source_id)
                if proof != selected["proof"] or _digest(proof) != record["source_close_sha256"]:
                    raise ValueError("runtime_admission_source_close_proof")
                checked_sources.add(source_id)
            elif _digest(selected["proof"]) != record["source_close_sha256"]:
                raise ValueError("runtime_admission_source_close_proof")
            for value in _descriptor_paths(target["descriptor"]):
                target_path = Path(value)
                if not target_path.is_relative_to(Path(source["state_root"])) or target_path.resolve() != target_path:
                    raise ValueError("runtime_admission_delegation_foreign_source")
    for identity in set(targets) - set(delegations):
        for value in _descriptor_paths(targets[identity]["descriptor"]):
            target_path = Path(value)
            if not target_path.is_relative_to(state) or target_path.resolve() != target_path:
                raise ValueError("runtime_admission_foreign_target")


def admission_validation_snapshot(state_root: Path, ticket: dict | str) -> _AdmissionValidationSnapshot:
    """Capture complete runtime validation before acquiring material locks.

    Args:
        state_root: State belonging to the inherited controller admission.
        ticket: The actual installed admission ticket.

    Returns:
        Immutable observations for the installed owner's retained operation.
        This value alone is not a capability or a substitute for full admission.
    """
    with _ADMISSION_VALIDATION_LOCK:
        value = _ticket_value(ticket)
        artifacts = {}
        root, state, binding = _admission_ticket_core(state_root, value, artifacts=artifacts)
        ownership = _target_ownership_validation(root, binding, allow_new_local=True, capture=True, artifacts=artifacts)
        snapshot = _AdmissionValidationSnapshot(
            _canonical_json(value), _canonical_json(binding), _canonical_json(ownership),
            _canonical_json(artifacts))
        # Join mutable HEAD and metadata observations again after the full scan.
        validate_admission_live(state, value, snapshot)
        return snapshot


def validate_admission_live(
    state_root: Path, ticket: dict | str, snapshot: _AdmissionValidationSnapshot,
) -> dict:
    """Recheck inherited authority and pinned metadata without SQLite or probes.

    Args:
        state_root: The exact admitted state.
        ticket: The original installed admission ticket.
        snapshot: The exact full-validation result retained by RuntimeAdmission
            and bound there to its original native execution. DTO shape alone
            grants nothing; this helper cannot admit a new native operation.

    Returns:
        The current immutable binding after checking its live metadata.
    """
    if type(snapshot) is not _AdmissionValidationSnapshot:
        raise ValueError("runtime_admission_validation_snapshot")
    value = _ticket_value(ticket)
    artifacts = json.loads(snapshot.artifacts_json)
    root, state, binding = _admission_ticket_live_core(state_root, value, artifacts)
    if (_canonical_json(value) != snapshot.ticket_json
            or _canonical_json(binding) != snapshot.binding_json):
        raise ValueError("runtime_admission_validation_changed")
    _validate_ownership_metadata(root, binding, json.loads(snapshot.ownership_json), artifacts)
    return binding


def validate_admission_ticket(state_root: Path, ticket: dict | str, *, check_state: bool = True) -> dict:
    """Validate a live inherited controller lease before creating any services."""
    with _ADMISSION_VALIDATION_LOCK:
        root, state, binding = _admission_ticket_core(state_root, ticket)
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
    # Receipt schema 2 (U3b-0d3): first the closed-state record, the closed
    # digest from a fresh snapshot and then the file manifest as the very last
    # state observation (the snapshot's read-only images may leave -shm
    # files, which the manifest must include). The receipt stays the commit point.
    root = Path(ticket["admission_root"])
    closed = _binding_snapshot(root, binding["slot"])
    _atomic_record(_receipt_state_path(root, ticket["run_id"]), {
        "schema_version": 2, "binding_id": ticket["binding_id"], "generation": ticket["generation"],
        "run_id": ticket["run_id"], "child_pid": os.getpid(),
        "closed_state_sha256": closed_state_sha256(closed),
        "state_manifest_sha256": state_manifest_sha256(Path(binding["state_root"]))}, immutable=True)
    path = Path(ticket["admission_root"]) / "admission" / "receipts" / (ticket["run_id"] + ".json")
    _atomic_record(path, receipt, immutable=True)
    return path


def _probe_timeout(value: object) -> float:
    """Return a probe timeout in [1, 3600] seconds or refuse with a coded error."""
    low, high = _PROBE_TIMEOUT_BOUNDS
    if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("runtime_admission_probe_timeout")
    return float(value)


def _probe_timeout_argument(text: str) -> float:
    try:
        return _probe_timeout(float(text))
    except ValueError:
        raise argparse.ArgumentTypeError("runtime_admission_probe_timeout: expected 1-3600 seconds") from None


def _probe_candidate(root: Path, name: str, binding: dict, *, timeout: float) -> None:
    """Have the candidate's own companion recompute its binding before any write.

    A managed child re-validates with its own code (validate_admission_ticket:
    full binding equality). Selecting a slot whose code disagrees would move the
    head and then wedge it at child start, so disagreement refuses here first.
    The probe is read-only by construction of the companion's code, not by an
    OS sandbox: database images open '?mode=ro' and writer checks take only
    transient non-blocking flocks on state *.lock/*.lease files. It runs with
    this interpreter (same site-packages, hence same sdk_versions), without the
    lease descriptor or an admission ticket, and with only _PROBE_ENVIRONMENT.
    """
    runtime = Path(binding["runtime_root"])
    release = (runtime / "release.json").read_bytes()
    if hashlib.sha256(release).hexdigest() != binding["release_sha256"]:
        raise ValueError("runtime_admission_release_changed")
    files = json.loads(release)["files"]
    module = runtime / "scripts" / "stage_runtime.pyc"
    if (any(companion not in files for companion in _PROBE_COMPANIONS)
            or module.is_symlink() or not module.is_file()):
        raise ValueError("runtime_admission_candidate_unmanaged")
    environment = {key: os.environ[key] for key in _PROBE_ENVIRONMENT if key in os.environ}
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", _PROBE_SOURCE, str(module), str(root), name],
            cwd=runtime, env=environment, stdin=subprocess.DEVNULL, capture_output=True,
            timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise ValueError("runtime_admission_candidate_rejected: probe_timeout") from None
    try:
        result = json.loads(completed.stdout)
    except (UnicodeDecodeError, ValueError):
        result = None
    if type(result) is not dict:
        raise ValueError("runtime_admission_candidate_rejected: probe_output exit="
                         + str(completed.returncode))
    if completed.returncode == _PROBE_UNMANAGED_EXIT and result == {"probe": "unmanaged"}:
        raise ValueError("runtime_admission_candidate_unmanaged")
    if (completed.returncode == _PROBE_REJECTED_EXIT and set(result) == {"probe", "code"}
            and result["probe"] == "error" and type(result["code"]) is str):
        raise ValueError("runtime_admission_candidate_rejected: " + result["code"][:200])
    if completed.returncode != 0 or set(result) != {"probe", "binding"} or result["probe"] != "binding":
        raise ValueError("runtime_admission_candidate_rejected: probe_output exit="
                         + str(completed.returncode))
    if _canonical_json(result["binding"]) != _canonical_json(binding):
        raise ValueError("runtime_admission_candidate_rejected: binding_mismatch")


class AdmissionController:
    """Own one admission root until each managed child is actually reaped.

    A closed head is immutable evidence of the last owned closure. Out-of-band
    changes cannot be adopted as a new authority merely by selecting that slot.
    """

    def __init__(self, root: Path, *, probe_timeout: float = _PROBE_TIMEOUT_SECONDS,
                 allow_open_head: bool = False):
        self.root = Path(root).resolve()
        self.probe_timeout = _probe_timeout(probe_timeout)
        # U3b-0d serve/adopt: enter over a running or unknown head so recovery
        # can run under the lease. Every other entry refuses an open head.
        self.allow_open_head = allow_open_head
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
            if head is not None and head["status"] != "closed" and not self.allow_open_head:
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
                                    target_ids: list[str]) -> list[str]:
        """Grant exact held targets to one physical copied state; one capability per owner."""
        self._require_lease()
        with _locked_root(self.root):
            return _authorize_target_delegation(self.root, destination_state, source_binding, target_ids)

    def delegate(self, name: str) -> dict:
        """Delegate what the closed head holds to one copied slot (any hop).

        Target ids: every target the head holds as delegated (path-less ones
        included, so a chain is never converted into ownership) plus every
        owned target whose descriptor carries absolute paths, all inside the
        head's state. Owned path-less targets stay owned by the destination.
        _authorize_target_delegation re-derives inherited chains from the
        verified head and records one owner-flattened record per owner.
        """
        self._require_lease()
        name = _slot_name(name)
        with _locked_root(self.root):
            head = read_admission_head(self.root)
            if head is None or head["status"] != "closed":
                raise ValueError("runtime_admission_delegation_source_not_current")
            source = verify_closed_binding(self.root, head["binding_id"])
            state = Path(source["state_root"])
            held = _target_ownership_validation(self.root, source)["delegations"]
            target_ids = []
            for identity, target in sorted(_target_inventory(state).items()):
                if identity in held:
                    target_ids.append(identity)
                    continue
                paths = [Path(value) for value in _descriptor_paths(target["descriptor"])]
                if not paths:
                    continue
                if any(not path.is_relative_to(state) or path.resolve() != path for path in paths):
                    raise ValueError("runtime_admission_delegation_foreign_source")
                target_ids.append(identity)
            capabilities = _authorize_target_delegation(
                self.root, self.root / "slots" / name / "state", head["binding_id"], target_ids)
        return {"capabilities": capabilities, "source_binding": head["binding_id"], "targets": target_ids}

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
                 managed: bool = False, launcher_checks: bool = True) -> str | None:
        """Select a compatible closed state, rejecting stale authority first.

        Once a managed launcher controller is installed (admission/
        controller.json), the slot must also pass serve's acceptance
        integers, so an operator activate cannot select a slot the next
        start refuses with 78 (review P10). ``launcher_checks=False`` is for
        rehearsals of that refusal only.
        """
        self._require_lease()
        name = _slot_name(name)
        with _locked_root(self.root):
            head = read_admission_head(self.root)
            if expected_head is not None and head != expected_head:
                raise ValueError("runtime_admission_head_changed")
            if head is not None and head["status"] != "closed":
                raise ValueError("runtime_admission_not_closed")
            binding = _binding_snapshot(self.root, name)
            if launcher_checks and (self.root / "admission" / "controller.json").exists():
                _require_launcher_slot(binding)
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
                # Before any admission record, marker, head CAS or selection.
                _probe_candidate(self.root, name, binding, timeout=self.probe_timeout)
                identity = _store_binding(self.root, binding)
                if head is None and _has_authority(latest):
                    raise ValueError("runtime_admission_manual_legacy_unsealed")
                _prepare_manual_head(self.root, head, binding, previous=latest_manual)
                _write_marker(self.root, identity, binding)
                _cas_head(self.root, head, identity, "closed")
            return _publish_selection(self.root, name)

    def begin_run(self) -> tuple[dict, dict, dict, dict]:
        """CAS the closed head to running and mint the child's admission ticket.

        Returns the head binding, the running head, the closed head's manual
        companion (verified now, before the child changes the state) and the
        ticket. The caller spawns the child with the lease descriptor.
        """
        self._require_lease()
        root = self.root
        with _locked_root(root):
            head = read_admission_head(root)
            if head is None or head["status"] != "closed":
                raise ValueError("runtime_admission_not_closed")
            binding = read_binding(root, head["binding_id"])
            validate_target_ownership(root, binding)
            previous_manual = _verify_manual_head(root, head, binding)
            run_id = uuid4().hex
            running = _cas_head(root, head, head["binding_id"], "running", run_id)
        ticket = {"schema_version": 1, "admission_root": str(root),
                  "binding_id": head["binding_id"], "generation": running["generation"],
                  "run_id": run_id, "lease_fd": self._descriptor,
                  "lease_identity": _identity(root / ".admission.lock"), "controller_pid": os.getpid()}
        return binding, running, previous_manual, ticket

    @property
    def lease_descriptor(self) -> int:
        """The held lease descriptor (inherited by the managed child)."""
        self._require_lease()
        return self._descriptor

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
        binding, running, previous_manual, ticket = self.begin_run()
        run_id = running["run_id"]
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
            # Reaped above. A slot that declares the receipt-state record must
            # have one (0d3 K2); one that does not is an older slot (v3 N7).
            seal_admission_run(root, running, process.pid, previous_manual)
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
                                 target_ids: list[str]) -> list[str]:
    """Issue an exact delegation while exclusively owning the current closed head."""
    with AdmissionController(root) as controller:
        return controller.authorize_target_delegation(destination_state, source_binding, target_ids)


def delegate_slot(root: Path, name: str) -> dict:
    """Delegate what the current closed head holds to one copied slot (multi-hop, U3b-0b)."""
    with AdmissionController(root) as controller:
        return controller.delegate(name)


def activate_slot(root: Path, name: str, *, probe_timeout: float = _PROBE_TIMEOUT_SECONDS,
                  launcher_checks: bool = True) -> str | None:
    """Atomically select an integrity-checked slot with non-regressing authority."""
    with AdmissionController(root, probe_timeout=probe_timeout) as controller:
        return controller.activate(name, launcher_checks=launcher_checks)


def run_slot(root: Path, name: str, argv: list[str] | None = None, *, timeout: float,
             env: dict | None = None, stdout=None, stderr=None) -> subprocess.CompletedProcess:
    """Execute one managed child while retaining exclusive root admission."""
    with AdmissionController(root) as controller:
        return controller.run(name, argv, timeout=timeout, env=env, stdout=stdout, stderr=stderr)


_launcher = None


def _launcher_module():
    """The managed launcher (U3b-0d) beside this file, bound to this module.

    Loaded by path like the manual-components contract: the controller runs
    as ``python -I -B .../scripts/stage_runtime.pyc`` with no package path.
    """
    global _launcher
    if _launcher is None:
        own = Path(__file__).resolve()
        suffix = ".pyc" if own.suffix == ".pyc" else ".py"
        path = own.parent / ("managed_launcher" + suffix)
        if path.is_symlink() or not path.is_file():
            raise ValueError("runtime_admission_launcher_missing")
        name = "_kza_managed_launcher_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError("runtime_admission_launcher_missing")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        module.bind_stage(sys.modules[__name__])
        _launcher = module
    return _launcher


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
    activate.add_argument("--probe-timeout", type=_probe_timeout_argument, default=_PROBE_TIMEOUT_SECONDS,
                          help="Seconds (1-3600) allowed for the candidate's own read-only validation")
    delegate = commands.add_parser(
        "delegate", help="Delegate the erasure targets the closed head holds (local and inherited) "
                         "to a copied slot")
    delegate.add_argument("name")
    launcher = _launcher_module()
    launcher.add_commands(commands)
    args = parser.parse_args()
    if args.command in launcher.COMMANDS:
        raise SystemExit(launcher.run_command(args))
    if args.command == "prepare":
        manifest = prepare_slot(args.archive, args.root, args.name, args.state_source)
        print(json.dumps({"slot": args.name, "commit": manifest["git_commit"]}))
    elif args.command == "delegate":
        result = delegate_slot(args.root, args.name)
        print(json.dumps({"slot": args.name, **result}))
    else:
        previous = activate_slot(args.root, args.name, probe_timeout=args.probe_timeout)
        print(json.dumps({"slot": args.name, "previous": previous}))


if __name__ == "__main__":
    main()
