"""U3b-1a admission: a fresh ledger admits itself; pre-U3b digests are stable.

The reference controller is fixed and deterministic: this stage_runtime with
exactly the three owner registrations removed (1a-F1). It never depends on
what 'HEAD' is, so it stays valid after U3b-1a is committed.
"""

import copy
import sqlite3
from contextlib import contextmanager

import pytest

from scripts import stage_runtime as stage
from src.memory.fact_ledger import FactLedger, MemoryAssigner, MemoryOrigin, MemoryReviewer, MemorySource

OWNER_TABLES = {"fact_owners", "fact_owner_revisions", "fact_owner_schema"}
PRIVACY = "data/privacy.db"
NOW = 2_000_000_000.0


@contextmanager
def reference_controller(monkeypatch):
    """This controller without the U3b owner registrations: the pre-U3b admission contract."""
    with monkeypatch.context() as patch:
        patch.setattr(stage, "_TABLE_COLUMNS", {name: value for name, value in stage._TABLE_COLUMNS.items()
                                                if name not in OWNER_TABLES})
        patch.setattr(stage, "_PROTECTED_TABLES", stage._PROTECTED_TABLES - OWNER_TABLES)
        patch.setattr(stage, "_SCHEMA_MARKERS", stage._SCHEMA_MARKERS - {"fact_owner_schema"})
        yield


def test_reference_controller_removes_exactly_the_owner_registrations(monkeypatch):
    current = (dict(stage._TABLE_COLUMNS), stage._PROTECTED_TABLES, stage._SCHEMA_MARKERS)
    with reference_controller(monkeypatch):
        assert current[0].keys() - stage._TABLE_COLUMNS.keys() == OWNER_TABLES
        assert current[1] - stage._PROTECTED_TABLES == {"fact_owners", "fact_owner_revisions"}
        assert current[2] - stage._SCHEMA_MARKERS == {"fact_owner_schema"}
        assert {name: current[0][name] for name in stage._TABLE_COLUMNS} == stage._TABLE_COLUMNS
    assert (stage._TABLE_COLUMNS, stage._PROTECTED_TABLES, stage._SCHEMA_MARKERS) == current


def test_registration_strings_are_exact():
    assert stage._TABLE_COLUMNS["fact_owners"] == (
        "fact_id:TEXT person_id:TEXT method:TEXT revision:INTEGER assigned_at:REAL assigned_by:TEXT evidence:TEXT")
    assert stage._TABLE_COLUMNS["fact_owner_revisions"] == (
        "fact_id:TEXT revision:INTEGER last_change:TEXT confirmed_by_assignment:INTEGER changed_at:REAL")
    assert stage._TABLE_COLUMNS["fact_owner_schema"] == "version:INTEGER"
    assert {"fact_owners", "fact_owner_revisions"} <= stage._PROTECTED_TABLES
    assert "fact_owner_schema" not in stage._PROTECTED_TABLES
    assert stage._SCHEMA_MARKERS == {"voice_group_schema", "voice_person_schema", "voice_capture_review_schema",
                                     "fact_owner_schema", "voice_attribution_schema"}


def _source():
    return MemorySource("00000000-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000002",
                        NOW - 1, NOW + 3600, MemoryOrigin.VOICE)


def _state(tmp_path, *, pre_u3b):
    """A populated ledger in a state root; pre_u3b drops the owner objects (HEAD-era shape)."""
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    (state / "config").mkdir()
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    try:
        ledger.store("Synthetic automatic fact", "fact", .9, (_source(),))
        proposal = ledger.store("Synthetic reviewed fact", "fact", 1.0, (_source(),), proposed=True)
        ledger.confirm(proposal.id, MemoryReviewer("synthetic-reviewer"))
    finally:
        ledger.close()
    with sqlite3.connect(state / PRIVACY) as db:
        # The voice ledger owns this marker row; a bare FactLedger leaves it empty.
        db.execute("INSERT INTO voice_group_schema VALUES (1)")
    if pre_u3b:
        with sqlite3.connect(state / PRIVACY) as db:
            db.executescript("DROP TABLE fact_owners; DROP TABLE fact_owner_revisions; DROP TABLE fact_owner_schema;")
    return state


def test_fresh_ledger_admits_itself_and_its_authority_covers_both_owner_tables(tmp_path):
    state = _state(tmp_path, pre_u3b=False)
    with stage._database_image(state / PRIVACY) as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        contract = stage._schema_contract(db, names)
    assert OWNER_TABLES <= contract["columns"].keys()
    authority = stage.capture_state_authority(state)
    tables = authority["projection"]["tables"][PRIVACY]
    assert {"fact_owners", "fact_owner_revisions"} <= tables.keys()
    assert tables["fact_owners"]["row_hashes"] == tables["fact_owner_revisions"]["row_hashes"] == []
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    try:
        fact = ledger.list(scope="all")[0]
        ledger.assign_owner(fact.id, "user_synthetic", fact.revision, 0, MemoryAssigner("synthetic-operator"),
                            retired=lambda ids: False, active=lambda person: True)
    finally:
        ledger.close()
    owned = stage.capture_state_authority(state)
    assert owned["digest"] != authority["digest"]
    assert len(owned["projection"]["tables"][PRIVACY]["fact_owners"]["row_hashes"]) == 1
    assert len(owned["projection"]["tables"][PRIVACY]["fact_owner_revisions"]["row_hashes"]) == 1


def test_unmigrated_pre_u3b_state_has_the_same_digest_under_both_controllers(tmp_path, monkeypatch):
    """1a-F1 row (a)."""
    state = _state(tmp_path, pre_u3b=True)
    new = stage.capture_state_authority(state)
    with reference_controller(monkeypatch):
        reference = stage.capture_state_authority(state)
    assert new == reference


def test_initialisation_adds_only_two_empty_projections_and_three_contract_entries(tmp_path, monkeypatch):
    """1a-F1 row (b); the reference controller rejects the migrated state (rollback rejection point)."""
    state = _state(tmp_path, pre_u3b=True)
    before = copy.deepcopy(stage.capture_state_authority(state)["projection"])
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    ledger.close()
    after = stage.capture_state_authority(state)["projection"]
    added = after["tables"][PRIVACY]
    columns = added["schema_contract_v1"]["columns"]
    assert {name: columns[name] for name in OWNER_TABLES} == {
        "fact_owners": {"fact_id": "TEXT", "person_id": "TEXT", "method": "TEXT", "revision": "INTEGER",
                        "assigned_at": "REAL", "assigned_by": "TEXT", "evidence": "TEXT"},
        "fact_owner_revisions": {"fact_id": "TEXT", "revision": "INTEGER", "last_change": "TEXT",
                                 "confirmed_by_assignment": "INTEGER", "changed_at": "REAL"},
        "fact_owner_schema": {"version": "INTEGER"}}
    for name, expected in (("fact_owners", ["fact_id", "person_id", "method", "revision", "assigned_at",
                                            "assigned_by", "evidence"]),
                           ("fact_owner_revisions", ["fact_id", "revision", "last_change",
                                                     "confirmed_by_assignment", "changed_at"])):
        assert added[name] == {"columns": expected, "row_hashes": []}
    stripped = copy.deepcopy(after)
    for name in ("fact_owners", "fact_owner_revisions"):
        del stripped["tables"][PRIVACY][name]
    for name in OWNER_TABLES:
        del stripped["tables"][PRIVACY]["schema_contract_v1"]["columns"][name]
    assert stripped == before
    with reference_controller(monkeypatch):
        with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
            stage.capture_state_authority(state)


def test_derived_expectation_admits_only_the_empty_owner_schema_migrated_by_derivation(tmp_path):
    """A derivation from a pre-U3b image opens the destination ledger, which migrates in the owner schema.

    authorize_derived_state expects exactly that empty addition and nothing else: an owner row in the
    derived image, or an owner schema already present in the source, still mismatches.
    """
    state = _state(tmp_path, pre_u3b=True)
    source = stage.capture_state_authority(state)["projection"]
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    ledger.close()
    derived = stage.capture_state_authority(state)["projection"]
    assert source != derived
    expected = copy.deepcopy(source)
    stage._expect_empty_schema_additions(expected, derived)
    assert expected == derived

    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    try:
        fact = ledger.list(scope="all")[0]
        ledger.assign_owner(fact.id, "user_synthetic", fact.revision, 0, MemoryAssigner("synthetic-operator"),
                            retired=lambda ids: False, active=lambda person: True)
    finally:
        ledger.close()
    owned = stage.capture_state_authority(state)["projection"]
    expected = copy.deepcopy(source)
    stage._expect_empty_schema_additions(expected, owned)
    assert expected != owned

    unchanged = copy.deepcopy(derived)
    stage._expect_empty_schema_additions(unchanged, owned)
    assert unchanged == derived
