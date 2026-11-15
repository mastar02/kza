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
import sqlite3
import types

import pytest

from scripts import build_runtime
from scripts import stage_runtime as stage

REPO = Path(__file__).resolve().parents[3]

# Bumping SCHEMA_CONTRACT_EPOCH is a schema transition (outside release mode, SC-D4): add the
# new epoch's digest here in the same change; never edit an existing entry.
PINNED_CONTRACTS = {
    1: "b5d785d185d1de676e79bed6b19712b5db90207516de3b79aca30bcf051ec923",
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
    assert set(document["vocabularies"]) == {"deletion_kinds", "erasure_participants", "fact_states",
                                             "fact_owner_changes", "fact_owner_methods"}


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
    "_SDK_INTERNAL_TABLES": _frozen_plus,
    "_METRICS_APPLICATION_ID": lambda value: value + 1,
    "_METRICS_USER_VERSIONS": lambda value: frozenset(value) | {3},
    "_AUTH_META_SCHEMA_VERSIONS": lambda value: frozenset(value) | {"4"},
    "_LEDGER_KINDS": _frozen_plus,
    "_ERASURE_PARTICIPANTS": _frozen_plus,
    "_FACT_STATES": _frozen_plus,
    "_FACT_OWNER_CHANGES": _frozen_plus,
    "_FACT_OWNER_METHODS": _frozen_plus,
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


def test_the_controller_literals_equal_the_src_vocabularies():
    """SCv2-P5: a new enum value without the literal (and the epoch bump) fails here."""
    from src.memory.deletion_ledger import DeletionKind
    from src.memory.erasure_journal import ErasureParticipant
    from src.memory.fact_ledger import FactOwnerChange, FactOwnerMethod, FactState

    assert stage._LEDGER_KINDS == {item.value for item in DeletionKind}
    assert stage._ERASURE_PARTICIPANTS == {item.value for item in ErasureParticipant}
    assert stage._FACT_STATES == {item.value for item in FactState}
    assert stage._FACT_OWNER_CHANGES == {item.value for item in FactOwnerChange}
    assert stage._FACT_OWNER_METHODS == {item.value for item in FactOwnerMethod}


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
