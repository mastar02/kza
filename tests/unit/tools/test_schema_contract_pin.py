"""SC-2: the schema contract is pinned to SCHEMA_CONTRACT_EPOCH (sc-release-design v3 schema_rule).

The controller's contract (tables, columns, markers, protected set, pragma and
auth_meta rules, value vocabularies) is serialized by
stage_runtime.schema_contract_document(). Its sha256 is pinned here beside the
epoch: any change to the contract without a SCHEMA_CONTRACT_EPOCH bump (and a
new pin) fails, so release mode can later rely on "same epoch, same contract"
(SR-1/SR-2). The vocabulary test ties the controller's literal sets to the src
enums the stores write (SCv2-P5); the store test proves the real initializers
create nothing the contract does not list.
"""

from __future__ import annotations

import asyncio
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import types

import pytest

from scripts import build_runtime
from scripts import stage_runtime as stage

REPO = Path(__file__).resolve().parents[3]

# Bumping SCHEMA_CONTRACT_EPOCH is a schema transition (outside release mode, SC-D4): add the
# new epoch's digest here in the same change; never edit an existing entry once a release declares
# it. Epoch 1 was re-pinned twice before any archive declared it: SC-2 review SC2-R1/SC2-R2 (every
# src enum stored verbatim in a protected table and the reader's own database rules joined the
# document; the first pin was b5d785d1...), then AUTHZ-1 (the erasure-progress and attribution
# vocabularies of privacy.db; the second pin was f411ffec...). From SC-6a (R1 built) this entry
# is frozen.
PINNED_CONTRACTS = {
    1: "691c2272c3e4f5b4e3119b008396f340de6ee8ef4006abba659103c694f1e8d8",
}


def _digest() -> str:
    return hashlib.sha256(stage.schema_contract_document()).hexdigest()


def test_the_contract_document_is_pinned_to_its_epoch():
    assert stage.SCHEMA_CONTRACT_EPOCH == 1
    assert _digest() == PINNED_CONTRACTS[stage.SCHEMA_CONTRACT_EPOCH], (
        "the schema contract changed: bump SCHEMA_CONTRACT_EPOCH and pin the new digest (SC-D4)")


def test_the_document_is_canonical_json_of_every_contract_part():
    document = json.loads(stage.schema_contract_document())
    assert stage.schema_contract_document() == stage._canonical_json(document)
    assert document["tables"] == stage._TABLE_COLUMNS
    assert document["database_tables"] == stage._DATABASE_TABLE_COLUMNS
    assert document["markers"] == {"tables": sorted(stage._SCHEMA_MARKERS), "rows": [[1]]}
    assert document["protected"] == sorted(stage._PROTECTED_TABLES)
    assert document["database_rules"]["metrics"] == {"application_id": 1264206897, "user_version": [1, 2]}
    assert document["database_rules"]["auth_meta_schema_version"] == ["1", "2", "3"]
    assert document["database_rules"]["unbound_alone"] == ["memories"]
    assert document["database_rules"]["sdk_internals"] == {
        "tables": ["collections", "migrations", "segments"],
        "exempt_when": "every_sdk_internal_table_and_no_contract_table"}
    assert document["database_rules"]["refused_objects"] == ["trigger", "view"]
    assert document["database_rules"]["other"] == {"application_id": 0, "user_version": 0}
    assert document["vocabularies"] == {name: sorted(values)
                                        for name, values in stage._contract_vocabularies().items()}
    assert set(document["vocabularies"]) == set(SRC_VOCABULARIES)


def _frozen_plus(value):
    return frozenset(value) | {"sc2_new_value"}


# Each contract constant, changed: the document (and so the pin) must change with it (M-2.9, M-2.11).
CHANGES = {
    "_TABLE_COLUMNS": lambda value: {**value, "sc2_new_table": "id:TEXT"},
    "_DATABASE_TABLE_COLUMNS": lambda value: {**value, "data/sc2.db": {"t": "id:TEXT"}},
    "_SCHEMA_MARKERS": _frozen_plus,
    "_SCHEMA_MARKER_ROWS": lambda value: ((2,),),
    "_PROTECTED_TABLES": _frozen_plus,
    "_SQLITE_INTERNAL_TABLES": _frozen_plus,
    "_UNBOUND_ALONE_TABLES": _frozen_plus,
    "_SDK_INTERNAL_TABLES": _frozen_plus,
    "_SDK_INTERNALS_EXEMPT_WHEN": lambda value: value + "_or_any_sdk_table",
    "_REFUSED_SQLITE_OBJECTS": lambda value: frozenset(value) - {"trigger"},
    "_METRICS_APPLICATION_ID": lambda value: value + 1,
    "_METRICS_USER_VERSIONS": lambda value: frozenset(value) | {3},
    "_DEFAULT_APPLICATION_ID": lambda value: value + 1,
    "_DEFAULT_USER_VERSION": lambda value: value + 1,
    "_AUTH_META_SCHEMA_VERSIONS": lambda value: frozenset(value) | {"4"},
    "_LEDGER_KINDS": _frozen_plus,
    "_ERASURE_PARTICIPANTS": _frozen_plus,
    "_FACT_STATES": _frozen_plus,
    "_FACT_OWNER_CHANGES": _frozen_plus,
    "_FACT_OWNER_METHODS": _frozen_plus,
    "_VOICE_CONSENT_STATES": _frozen_plus,
    "_VOICE_VERIFICATION_OUTCOMES": _frozen_plus,
    "_VOICE_ATTRIBUTION_EVENT_KINDS": _frozen_plus,
    "_VOICE_SUSPENSION_REASONS": _frozen_plus,
    "_DASHBOARD_ROLES": _frozen_plus,
    "_CREDENTIAL_ACTIONS": _frozen_plus,
    "_CREDENTIAL_OUTCOMES": _frozen_plus,
    "_CREDENTIAL_REASONS": _frozen_plus,
    "_RECORDER_STATES": _frozen_plus,
    "_REMINDER_STATES": _frozen_plus,
    "_REMINDER_SCOPES": _frozen_plus,
    "_REMINDER_ACTION_STATUSES": _frozen_plus,
    "_ROUTINE_RUN_STATUSES": _frozen_plus,
    "_ROUTINE_ACTION_STATUSES": _frozen_plus,
    "_LIST_OWNER_TYPES": _frozen_plus,
    "_ERASURE_STATES": _frozen_plus,
    "_ERASURE_ERRORS": _frozen_plus,
    "_INDEX_CLEANUP_ERRORS": _frozen_plus,
    "_MEMORY_ORIGINS": _frozen_plus,
}


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_every_contract_constant_is_part_of_the_pinned_document(name, monkeypatch):
    pinned = _digest()
    monkeypatch.setattr(stage, name, CHANGES[name](getattr(stage, name)))
    assert _digest() != pinned


def test_a_changed_column_type_changes_the_document(monkeypatch):
    pinned = _digest()
    monkeypatch.setitem(stage._TABLE_COLUMNS, "auth_meta", "key:TEXT value:BLOB")
    assert _digest() != pinned


# Document key -> (stage literal, src module, src name, the table.column that stores it verbatim).
# SCv2-P5 named the first five; SC2-R1 added every other src enum (or named closed set) a
# protected table stores; AUTHZ-1 (Option A: privacy relevance, not the protected set) added
# every named src enum the erasure-progress and attribution tables of privacy.db store
# (PRIVACY_PROGRESS_TABLES, not protected). Inline literal sets in src have no importable name
# and are listed as excluded in DEPLOYMENT.md («Contrato de esquema»).
SRC_VOCABULARIES = {
    "deletion_kinds": ("_LEDGER_KINDS", "src.memory.deletion_ledger", "DeletionKind", "deletion_ledger_v1.kind"),
    "erasure_participants": ("_ERASURE_PARTICIPANTS", "src.memory.erasure_journal", "ErasureParticipant",
                             "erasure_targets.participant"),
    "fact_states": ("_FACT_STATES", "src.memory.fact_ledger", "FactState", "facts.state"),
    "fact_owner_changes": ("_FACT_OWNER_CHANGES", "src.memory.fact_ledger", "FactOwnerChange",
                           "fact_owner_revisions.last_change"),
    "fact_owner_methods": ("_FACT_OWNER_METHODS", "src.memory.fact_ledger", "FactOwnerMethod", "fact_owners.method"),
    "voice_consent_states": ("_VOICE_CONSENT_STATES", "src.memory.voice_attribution", "VoiceConsentState",
                             "voice_attribution_consents.state"),
    "voice_verification_outcomes": ("_VOICE_VERIFICATION_OUTCOMES", "src.memory.voice_attribution",
                                    "VoiceVerificationOutcome", "voice_verification_attempts.outcome"),
    "voice_attribution_event_kinds": ("_VOICE_ATTRIBUTION_EVENT_KINDS", "src.memory.voice_attribution",
                                      "VoiceAttributionEventKind", "voice_attribution_events.kind"),
    "voice_suspension_reasons": ("_VOICE_SUSPENSION_REASONS", "src.memory.voice_attribution",
                                 "VoiceSuspensionReason", "voice_attribution_suspensions.reason"),
    "dashboard_roles": ("_DASHBOARD_ROLES", "src.dashboard.authority", "DashboardRole", "credentials.role"),
    "credential_actions": ("_CREDENTIAL_ACTIONS", "src.dashboard.credential_journal", "CredentialAction",
                           "credential_attempts.kind"),
    "credential_outcomes": ("_CREDENTIAL_OUTCOMES", "src.dashboard.credential_journal", "CredentialOutcome",
                            "credential_attempts.outcome"),
    "credential_reasons": ("_CREDENTIAL_REASONS", "src.dashboard.credential_journal", "REASONS",
                           "credential_attempts.reason"),
    "recorder_states": ("_RECORDER_STATES", "src.memory.capture_recorders", "RecorderState",
                        "capture_browser_recorders.state"),
    "reminder_states": ("_REMINDER_STATES", "src.reminders.reminder_store", "ReminderState", "reminders.state"),
    "reminder_scopes": ("_REMINDER_SCOPES", "src.reminders.reminder_store", "ReminderScope", "reminders.scope"),
    "reminder_action_statuses": ("_REMINDER_ACTION_STATUSES", "src.reminders.reminder_store",
                                 "ReminderActionStatus", "reminder_home_actions.status"),
    "routine_run_statuses": ("_ROUTINE_RUN_STATUSES", "src.routines.models", "RunStatus", "routine_runs.status"),
    "routine_action_statuses": ("_ROUTINE_ACTION_STATUSES", "src.routines.models", "ActionStatus",
                                "routine_run_actions.status"),
    "list_owner_types": ("_LIST_OWNER_TYPES", "src.lists.list_store", "OwnerType", "lists.owner_type"),
    "erasure_states": ("_ERASURE_STATES", "src.memory.erasure_journal", "ErasureState", "erasure_steps.state"),
    "erasure_errors": ("_ERASURE_ERRORS", "src.memory.erasure_journal", "ErasureError",
                       "erasure_steps.last_error"),
    "index_cleanup_errors": ("_INDEX_CLEANUP_ERRORS", "src.memory.index_deletions", "IndexCleanupError",
                             "policy_index_deletions.last_error"),
    "memory_origins": ("_MEMORY_ORIGINS", "src.memory.fact_ledger", "MemoryOrigin", "fact_sources.origin"),
}

# AUTHZ-1: the privacy.db tables outside the protected set whose vocabularies are pinned because
# they govern erasure progress and attribution. Every column of each is classified below, so a new
# column forces the choice (pinned vocabulary, excluded inline literal set, or not a closed set).
PRIVACY_PROGRESS_TABLES = {
    "erasure_targets": {"participant": ("erasure_participants",)},
    "erasure_steps": {"participant": ("erasure_participants",), "state": ("erasure_states",),
                      # ErasureError, plus IndexCleanupError.RECHECK from IndexDeletionQueue.recover
                      "last_error": ("erasure_errors", "index_cleanup_errors")},
    "policy_index_deletions": {"state": "excluded", "last_error": ("index_cleanup_errors",)},
    "facts": {"state": ("fact_states",), "category": "excluded"},
    "fact_sources": {"origin": ("memory_origins",)},
}
_NOT_A_CLOSED_SET = {
    "erasure_targets": {"id", "version", "descriptor"},
    "erasure_steps": {"operation_id", "target_id", "version", "attempts", "next_attempt_at", "claim_token",
                      "cursor", "pending_count"},
    "policy_index_deletions": {"target_id", "fact_id", "revision", "attempts", "next_attempt_at", "claim_token"},
    "facts": {"id", "content", "confidence", "extracted_at", "expires_at", "revision", "confirmed_at",
              "confirmed_by"},
    "fact_sources": {"fact_id", "capture_id", "conversation_id", "captured_at", "expires_at", "dependencies"},
}


@pytest.mark.parametrize("key", sorted(SRC_VOCABULARIES))
def test_the_controller_literals_equal_the_src_vocabularies(key):
    """SCv2-P5 / SC2-R1: a new src value without the literal (and the epoch bump) fails here."""
    import importlib

    literal, module, name, column = SRC_VOCABULARIES[key]
    source = getattr(importlib.import_module(module), name)
    values = {item.value for item in source} if isinstance(source, type) else set(source)
    assert getattr(stage, literal) == values
    assert stage._contract_vocabularies()[key] is getattr(stage, literal)
    table, field = column.split(".")
    assert table in stage._TABLE_COLUMNS
    assert field in dict(item.split(":") for item in stage._TABLE_COLUMNS[table].split())
    assert table in stage._PROTECTED_TABLES or table in PRIVACY_PROGRESS_TABLES


def test_every_column_of_the_privacy_progress_tables_is_classified():
    """AUTHZ-1: each closed set an erasure-progress/attribution table stores is pinned or excluded."""
    for table, classified in PRIVACY_PROGRESS_TABLES.items():
        assert table not in stage._PROTECTED_TABLES, table
        columns = {item.split(":")[0] for item in stage._TABLE_COLUMNS[table].split()}
        assert columns == set(classified) | _NOT_A_CLOSED_SET[table], table
        assert not set(classified) & _NOT_A_CLOSED_SET[table], table
        for column, keys in classified.items():
            if keys == "excluded":
                continue
            for key in keys:
                assert key in stage._contract_vocabularies(), (table, column, key)
            assert any(SRC_VOCABULARIES[key][3] == f"{table}.{column}" for key in keys) \
                or table == "erasure_steps" and column == "participant", (table, column)


# AUTHZ3-2: every TEXT column of every protected table is classified, so a new closed set (or a
# diagnostic column turned selector) forces the choice. A tuple names pinned vocabularies; "excluded"
# is an inline literal set and "diagnostic" free text no code selects on, both named in DEPLOYMENT's
# exclusions paragraph; "rule" is governed by a database rule of the document. _PROTECTED_OPEN_TEXT
# holds identifiers, digests, timestamps, payload and JSON: not a closed set.
PROTECTED_CLOSED_TEXT = {
    "ambient_runtime_control": {},
    "audio_write_intents": {},
    "auth_meta": {"key": "excluded", "value": "rule"},
    "capture_browser_recorders": {"state": ("recorder_states",), "kind": "excluded"},
    "capture_runtime_control": {"mode": "excluded"},
    "conversation_erasures": {},
    "conversation_tombstones": {},
    "credential_attempts": {"kind": ("credential_actions",), "outcome": ("credential_outcomes",),
                            "reason": ("credential_reasons",)},
    "credentials": {"role": ("dashboard_roles",)},
    "deletion_ledger_salt_v1": {},
    "deletion_ledger_v1": {"kind": ("deletion_kinds",)},
    "fact_owner_revisions": {"last_change": ("fact_owner_changes",)},
    "fact_owners": {"method": ("fact_owner_methods",)},
    "legacy_quarantine_batches_v1": {},
    "legacy_quarantine_records_v1": {"state": "excluded"},
    "list_items": {},
    "lists": {"owner_type": ("list_owner_types",)},
    "live_deliveries": {},
    "live_delivery_lineage": {},
    "live_delivery_sources": {},
    "pending_audio_deletions": {"last_error": "diagnostic"},
    "quarantines": {},
    "reminder_home_actions": {"status": ("reminder_action_statuses",), "reason": "diagnostic"},
    "reminders": {"state": ("reminder_states",), "scope": ("reminder_scopes",), "recurrence": "excluded"},
    "retired_users": {},
    "routine_run_actions": {"status": ("routine_action_statuses",), "reason": "diagnostic"},
    "routine_runs": {"status": ("routine_run_statuses",), "source": "excluded", "reason": "diagnostic"},
    "routines": {},
    "sessions": {},
    "staged": {},
    "voice_attribution_consents": {"state": ("voice_consent_states",)},
    "voice_attribution_control": {},
    "voice_attribution_events": {"kind": ("voice_attribution_event_kinds",)},
    "voice_attribution_suspensions": {"reason": ("voice_suspension_reasons",)},
    "voice_group_request_lineage": {},
    "voice_group_requests": {},
    "voice_profile_people": {},
    "voice_profile_sources": {},
    "voice_profiles": {},
    "voice_review_requests": {},
    "voice_verification_attempts": {"outcome": ("voice_verification_outcomes",)},
}
_PROTECTED_OPEN_TEXT = {
    "ambient_runtime_control": set(),
    "audio_write_intents": {"path", "conversation_id"},
    "auth_meta": set(),
    "capture_browser_recorders": {"token_hash"},
    "capture_runtime_control": set(),
    "conversation_erasures": {"id", "conversation_id", "requested_by"},
    "conversation_tombstones": {"conversation_id", "deleted_by"},
    "credential_attempts": {"request_id", "actor_key", "command_digest", "credential_id"},
    "credentials": {"id", "digest", "subject", "user_id", "samples"},
    "deletion_ledger_salt_v1": {"salt"},
    "deletion_ledger_v1": {"subject_sha256", "prev_sha256", "row_sha256"},
    "fact_owner_revisions": {"fact_id"},
    "fact_owners": {"fact_id", "person_id", "assigned_by", "evidence"},
    "legacy_quarantine_batches_v1": {"id", "snapshot_sha256", "selection_sha256", "collection_id", "imported_at"},
    # encoding is copied verbatim from the vector SDK's embeddings_queue, not a src vocabulary.
    "legacy_quarantine_records_v1": {"batch_id", "record_id", "document", "metadata_json", "encoding", "digest"},
    "list_items": {"id", "list_id", "text", "added_by"},
    "lists": {"id", "name", "owner_id"},
    "live_deliveries": {"ticket", "client_id", "connection_id", "event_id", "nonce", "owner_namespace"},
    "live_delivery_lineage": {"ticket", "source_digest"},
    "live_delivery_sources": {"ticket", "conversation_id"},
    "pending_audio_deletions": {"path", "conversation_id"},
    "quarantines": {"source", "destination"},
    "reminder_home_actions": {"request_id", "body"},
    "reminders": {"id", "user_id", "text", "ha_actions"},
    "retired_users": {"user_id"},
    "routine_run_actions": {"run_id", "entity_id"},
    "routine_runs": {"run_id", "routine_id", "occurrence_key", "actor", "snapshot_json", "scheduled_for"},
    "routines": {"routine_id", "created_by", "created_at", "definition_json"},
    "sessions": {"digest", "credential", "legacy_digest"},
    "staged": {"name"},
    "voice_attribution_consents": {"person_id", "verified_profile_id", "changed_by"},
    "voice_attribution_control": {"changed_by"},
    "voice_attribution_events": {"id", "profile_id"},
    "voice_attribution_suspensions": {"profile_id"},
    "voice_group_request_lineage": {"actor", "request_id", "source_digest"},
    "voice_group_requests": {"actor", "request_id", "digest", "sources_json"},
    "voice_profile_people": {"profile_id", "user_id", "updated_by"},
    "voice_profile_sources": {"profile_id", "conversation_id"},
    "voice_profiles": {"id", "name", "model_key", "sources_json", "confirmed_by"},
    "voice_review_requests": {"actor", "request_id", "digest", "profile_id"},
    "voice_verification_attempts": {"person_id"},
}


def _text_columns(table: str) -> set[str]:
    return {name for name, kind in (item.split(":") for item in stage._TABLE_COLUMNS[table].split())
            if kind == "TEXT"}


def test_every_text_column_of_the_protected_tables_is_classified():
    """AUTHZ3-2: each TEXT column of a protected table is pinned, excluded, diagnostic, a rule or open."""
    assert set(PROTECTED_CLOSED_TEXT) == set(_PROTECTED_OPEN_TEXT) == set(stage._PROTECTED_TABLES)
    vocabularies = stage._contract_vocabularies()
    for table, classified in PROTECTED_CLOSED_TEXT.items():
        assert _text_columns(table) == set(classified) | _PROTECTED_OPEN_TEXT[table], table
        assert not set(classified) & _PROTECTED_OPEN_TEXT[table], table
        for column, keys in classified.items():
            if isinstance(keys, tuple):
                for key in keys:
                    assert key in vocabularies and SRC_VOCABULARIES[key][3] == f"{table}.{column}", (table, key)
            else:
                assert keys in {"excluded", "diagnostic", "rule"}, (table, column)
    assert PROTECTED_CLOSED_TEXT["auth_meta"]["value"] == "rule"
    document = json.loads(stage.schema_contract_document())
    assert document["database_rules"]["auth_meta_schema_version"] == sorted(stage._AUTH_META_SCHEMA_VERSIONS)
    # Every pinned vocabulary of a protected table is the classification of its column.
    for key, (_, _, _, column) in SRC_VOCABULARIES.items():
        table, field = column.split(".")
        if table in stage._PROTECTED_TABLES:
            assert key in PROTECTED_CLOSED_TEXT[table].get(field, ()), key


def _exclusions_paragraph() -> str:
    text = (REPO / "docs" / "architecture" / "DEPLOYMENT.md").read_text()
    blocks = [" ".join(block.split()) for block in re.split(r"\n\s*\n", text)]
    found = [block for block in blocks if block.startswith("Quedan fuera los conjuntos cerrados que `src` escribe")]
    assert len(found) == 1
    return found[0]


def test_every_excluded_or_diagnostic_column_is_named_in_deployment():
    """AUTHZ3-2 / AUTHZ-1: DEPLOYMENT's exclusions paragraph names each column the contract leaves out."""
    paragraph = _exclusions_paragraph()
    left_out = [(table, column) for tables in (PROTECTED_CLOSED_TEXT, PRIVACY_PROGRESS_TABLES)
                for table, classified in tables.items()
                for column, keys in classified.items() if keys in ("excluded", "diagnostic")]
    assert len(left_out) == 12
    for table, column in left_out:
        assert f"{table}.{column}" in paragraph, (table, column)
    assert "Tampoco son contrato las columnas de diagnóstico de tablas protegidas" in paragraph


def test_every_value_src_writes_to_erasure_steps_last_error_is_pinned():
    """AUTHZ-1: erasure_steps.last_error carries ErasureError plus the index recheck reason."""
    from src.core import adoption_schema
    from src.memory.erasure_journal import ErasureError
    from src.memory.index_deletions import IndexCleanupError

    assert adoption_schema.RESIDUE_REASON in stage._ERASURE_ERRORS
    assert {error.value for error in ErasureError} <= stage._ERASURE_ERRORS
    assert IndexCleanupError.RECHECK.value in stage._INDEX_CLEANUP_ERRORS


def test_every_contract_vocabulary_has_a_src_source():
    assert set(stage._contract_vocabularies()) == set(SRC_VOCABULARIES)


def test_the_contract_rules_are_the_ones_the_reader_enforces(tmp_path, monkeypatch):
    """_schema_contract reads the same constants the document serializes."""
    path = tmp_path / "auth.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE auth_meta (key TEXT, value TEXT)")
        db.execute("INSERT INTO auth_meta VALUES ('schema_version', '4')")
    with closing(sqlite3.connect(path)) as db:
        with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
            stage._schema_contract(db, {"auth_meta"})
        monkeypatch.setattr(stage, "_AUTH_META_SCHEMA_VERSIONS", frozenset({"4"}))
        assert stage._schema_contract(db, {"auth_meta"})["columns"] == {"auth_meta": {"key": "TEXT",
                                                                                      "value": "TEXT"}}
    with closing(sqlite3.connect(tmp_path / "marker.db")) as db, db:
        db.execute("CREATE TABLE fact_owner_schema (version INTEGER)")
        db.execute("INSERT INTO fact_owner_schema VALUES (2)")
        with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
            stage._schema_contract(db, {"fact_owner_schema"})
        monkeypatch.setattr(stage, "_SCHEMA_MARKER_ROWS", ((2,),))
        assert stage._schema_contract(db, {"fact_owner_schema"}) is not None


# --- the reader's database rules are the constants the document serializes (SC2-R2) --------------

def _db(path: Path, *statements: str, application: int = 0, version: int = 0) -> sqlite3.Connection:
    with closing(sqlite3.connect(path)) as db, db:
        for statement in statements:
            db.execute(statement)
        db.execute(f"PRAGMA application_id={application}")
        db.execute(f"PRAGMA user_version={version}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _names(db: sqlite3.Connection) -> set[str]:
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_the_unbound_alone_rule_is_the_constant(tmp_path, monkeypatch):
    with closing(_db(tmp_path / "m.db", "CREATE TABLE memories (id TEXT)")) as db:
        assert stage._schema_contract(db, _names(db)) is None
    with closing(_db(tmp_path / "o.db", "CREATE TABLE other_alone (id TEXT)")) as db:
        with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
            stage._schema_contract(db, _names(db))
        monkeypatch.setattr(stage, "_UNBOUND_ALONE_TABLES", frozenset({"other_alone"}))
        assert stage._schema_contract(db, _names(db)) is None
    with closing(_db(tmp_path / "mm.db", "CREATE TABLE memories (id TEXT)", "CREATE TABLE staged (name TEXT)")) as db:
        monkeypatch.setattr(stage, "_UNBOUND_ALONE_TABLES", frozenset({"memories"}))
        with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
            stage._schema_contract(db, _names(db))  # not alone: memories is not a contract table


def test_the_sdk_exemption_admits_exactly_what_its_descriptor_names(tmp_path):
    """every_sdk_internal_table_and_no_contract_table: all SDK tables, any extra non-contract table."""
    sdk = [f"CREATE TABLE {name} (id TEXT)" for name in sorted(stage._SDK_INTERNAL_TABLES)]
    with closing(_db(tmp_path / "a.db", *sdk, "CREATE TABLE embeddings (id TEXT)")) as db:
        assert stage._schema_contract(db, _names(db)) is None
    with closing(_db(tmp_path / "b.db", *sdk[:-1], "CREATE TABLE embeddings (id TEXT)")) as db:
        with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
            stage._schema_contract(db, _names(db))  # one SDK table missing
    with closing(_db(tmp_path / "c.db", *sdk, "CREATE TABLE staged (name TEXT)")) as db:
        with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
            stage._schema_contract(db, _names(db))  # a contract table beside them is read as a contract db


@pytest.mark.parametrize("kind", ["view", "trigger"])
def test_the_refused_objects_are_the_constant(tmp_path, monkeypatch, kind):
    statement = {"view": "CREATE VIEW v AS SELECT name FROM staged",
                 "trigger": "CREATE TRIGGER t AFTER INSERT ON staged BEGIN SELECT 1; END"}[kind]
    with closing(_db(tmp_path / "v.db", "CREATE TABLE staged (name TEXT)", statement)) as db:
        with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
            stage._schema_contract(db, _names(db))
        monkeypatch.setattr(stage, "_REFUSED_SQLITE_OBJECTS", stage._REFUSED_SQLITE_OBJECTS - {kind})
        assert stage._schema_contract(db, _names(db)) is not None


@pytest.mark.parametrize("pragma", ["application", "version"])
def test_the_default_pragmas_are_the_constants(tmp_path, monkeypatch, pragma):
    values = {"application": 7, "version": 0} if pragma == "application" else {"application": 0, "version": 7}
    with closing(_db(tmp_path / "p.db", "CREATE TABLE staged (name TEXT)", **values)) as db:
        with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
            stage._schema_contract(db, _names(db))
        name = "_DEFAULT_APPLICATION_ID" if pragma == "application" else "_DEFAULT_USER_VERSION"
        monkeypatch.setattr(stage, name, 7)
        assert stage._schema_contract(db, _names(db)) == {
            "columns": {"staged": {"name": "TEXT"}}, "application_id": values["application"],
            "user_version": values["version"]}


# --- release.json declares the contract (M-2.10) --------------------------------------------------

def test_build_runtime_declares_the_schema_contract_from_the_revision_text():
    assert build_runtime.RELEASE_INTEGERS["schema_contract"] == "SCHEMA_CONTRACT_EPOCH"
    declared = build_runtime.release_integers((REPO / "scripts" / "stage_runtime.py").read_bytes())
    assert declared["schema_contract"] == stage.SCHEMA_CONTRACT_EPOCH == 1
    assert declared["controller_epoch"] == stage.CONTROLLER_EPOCH == 4
    # A revision before SC-2 declares no contract: release_integers reads it as 0.
    assert "schema_contract" not in build_runtime.release_integers(b"CONTROLLER_EPOCH = 3\n")
    assert "schema_contract" in stage._RELEASE_INTEGER_KEYS


def test_a_release_without_the_key_reads_contract_zero(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    raw = json.dumps({"controller_epoch": 3, "launcher_contract": 3}).encode()
    (runtime / "release.json").write_bytes(raw)
    binding = {"runtime_root": str(runtime), "release_sha256": hashlib.sha256(raw).hexdigest()}
    assert stage.release_integers(binding)["schema_contract"] == 0


# --- the real store initializers create only contract tables and columns --------------------------

def _tables(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as db:
        return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _assert_contract(path: Path, relative: str) -> set[str]:
    """Every table the initializer created is a contract table with exactly the contract's columns.

    Checked table by table rather than through _schema_contract as a whole: a
    fresh ledger leaves voice_group_schema empty until VoiceGroups.initialize
    (the app's voice stage, not a store initializer) writes its version row.
    """
    names = _tables(path) - stage._SQLITE_INTERNAL_TABLES
    known = dict(stage._TABLE_COLUMNS)
    known.update(stage._DATABASE_TABLE_COLUMNS.get(relative, {}))
    assert names and names <= known.keys(), sorted(names - known.keys())
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        assert db.execute("SELECT 1 FROM sqlite_master WHERE type IN ('view','trigger')").fetchone() is None
        for name in names:
            actual = {row[1]: row[2].upper() for row in db.execute("PRAGMA table_info(" + stage._quote(name) + ")")}
            assert actual == dict(item.split(":") for item in known[name].split()), name
            if name in stage._SCHEMA_MARKERS:
                assert {tuple(row) for row in db.execute("SELECT version FROM " + stage._quote(name))} \
                    <= set(stage._SCHEMA_MARKER_ROWS), name
    return names


def test_every_store_initializer_creates_only_contract_tables(tmp_path, monkeypatch):
    """Fact ledger (with voice, erasure, deletion-ledger and live-receipt schemas), ambient store,
    event logger and dashboard authority, each on an empty file under data/."""
    from src.ambient.store import AmbientStore
    from src.analytics import event_logger
    from src.dashboard.authority import DashboardAuthority
    from src.memory import fact_ledger

    noop = lambda *args, **kwargs: None  # noqa: E731 - the managed output registry is not under test
    registry = types.SimpleNamespace(before_ledger_open=noop, ledger_connected=noop, borrowed_ledger=noop,
                                     ledger_closed=noop, ledger_close_failed=noop)
    monkeypatch.setattr(fact_ledger, "output_registration", registry)
    data = tmp_path / "data"
    data.mkdir()
    ledger = fact_ledger.FactLedger(str(data / "privacy.db"))
    ledger.initialize()
    ledger.close()
    privacy = _assert_contract(data / "privacy.db", "data/privacy.db")
    assert {"facts", "fact_owners", "voice_profiles", "voice_attribution_consents", "erasure_targets",
            "deletion_ledger_v1", "live_deliveries", "live_connections"} <= privacy

    async def stores():
        ambient = AmbientStore(db_path=str(data / "ambient.db"), retention_hours=12)
        await ambient.init()
        await ambient.close()
        authority = DashboardAuthority(data / "auth.db")
        try:
            await authority.initialize()
        finally:
            await authority.aclose()
        events = event_logger.EventLogger(str(data / "events.db"))
        try:
            await events.acall(events.initialize)
        finally:
            await events.aclose()

    asyncio.run(stores())
    assert {"utterances", "capture_conversations"} <= _assert_contract(data / "ambient.db", "data/ambient.db")
    assert {"auth_meta", "credentials", "sessions"} <= _assert_contract(data / "auth.db", "data/auth.db")
    assert "events_policy_v1" in _assert_contract(data / "events.db", "data/events.db")


def test_the_reminder_routine_and_list_stores_create_only_contract_tables(tmp_path):
    """SC-2 review R7: three more protected-table owners whose vocabularies joined the contract."""
    from src.lists.list_store import ListStore
    from src.reminders.reminder_store import ReminderStore
    from src.routines.routine_store import RoutineStore

    data = tmp_path / "data"
    data.mkdir()

    async def stores():
        for store in (ReminderStore(str(data / "reminders.db")), RoutineStore(str(data / "routines.db")),
                      ListStore(str(data / "lists.db"))):
            try:
                await store.initialize()
            finally:
                await store.close()

    asyncio.run(stores())
    assert {"reminders", "reminder_home_actions"} <= _assert_contract(data / "reminders.db", "data/reminders.db")
    assert {"routines", "routine_runs", "routine_run_actions"} <= _assert_contract(data / "routines.db",
                                                                                    "data/routines.db")
    assert {"lists", "list_items"} <= _assert_contract(data / "lists.db", "data/lists.db")
