"""U3b-4a-0 admission: voice-attribution registration in the release controller.

The reference controller is fixed and deterministic (1a-F1): this
stage_runtime with exactly the voice-attribution registrations removed. It
never depends on what 'HEAD' is, so it stays valid after U3b-4a-0 is committed.
"""

import copy
from contextlib import contextmanager
import json
import sqlite3

import pytest

from scripts import stage_runtime as stage
from src.memory.fact_ledger import FactLedger, MemoryReviewer
from src.memory.voice_schema import VOICE_ATTRIBUTION_TABLES
from tests.mocks.memory_sources import memory_source
from tests.unit.memory.test_voice_attribution_schema import insert_deleted_profile, seed_voice_attribution_rows

PROTECTED = {"voice_attribution_consents", "voice_verification_attempts", "voice_attribution_events",
             "voice_attribution_suspensions", "voice_attribution_control"}
VOICE_TABLES = set(VOICE_ATTRIBUTION_TABLES)
PRIVACY = "data/privacy.db"
NOW = 2_000_000_000.0


@contextmanager
def reference_controller(monkeypatch):
    """This controller without the U3b-4a-0 registrations: the pre-4a admission contract."""
    with monkeypatch.context() as patch:
        patch.setattr(stage, "_TABLE_COLUMNS", {name: value for name, value in stage._TABLE_COLUMNS.items()
                                                if name not in VOICE_TABLES})
        patch.setattr(stage, "_PROTECTED_TABLES", stage._PROTECTED_TABLES - VOICE_TABLES)
        patch.setattr(stage, "_SCHEMA_MARKERS", stage._SCHEMA_MARKERS - {"voice_attribution_schema"})
        yield


def test_reference_controller_removes_exactly_the_voice_attribution_registrations(monkeypatch):
    current = (dict(stage._TABLE_COLUMNS), stage._PROTECTED_TABLES, stage._SCHEMA_MARKERS)
    with reference_controller(monkeypatch):
        assert current[0].keys() - stage._TABLE_COLUMNS.keys() == VOICE_TABLES
        assert current[1] - stage._PROTECTED_TABLES == PROTECTED
        assert current[2] - stage._SCHEMA_MARKERS == {"voice_attribution_schema"}
    assert (stage._TABLE_COLUMNS, stage._PROTECTED_TABLES, stage._SCHEMA_MARKERS) == current


def test_registration_strings_are_exact():
    assert {name: stage._TABLE_COLUMNS[name] for name in VOICE_TABLES} == {
        "voice_attribution_consents": "person_id:TEXT state:TEXT revision:INTEGER verified_profile_id:TEXT "
                                      "verified_profile_revision:INTEGER verified_link_revision:INTEGER "
                                      "changed_at:REAL changed_by:TEXT",
        "voice_verification_attempts": "person_id:TEXT attempted_at:REAL outcome:TEXT",
        "voice_attribution_events": "id:TEXT profile_id:TEXT kind:TEXT score_bucket:INTEGER at:REAL",
        "voice_attribution_suspensions": "profile_id:TEXT revision:INTEGER suspended_at:REAL reason:TEXT",
        "voice_attribution_control": "id:INTEGER enabled:INTEGER revision:INTEGER changed_at:REAL changed_by:TEXT",
        "voice_attribution_schema": "version:INTEGER",
    }
    assert PROTECTED <= stage._PROTECTED_TABLES and "voice_attribution_schema" not in stage._PROTECTED_TABLES
    assert "voice_attribution_schema" in stage._SCHEMA_MARKERS
    assert stage._VOICE_ATTRIBUTION_ADDITION == VOICE_ATTRIBUTION_TABLES
    assert stage._SCHEMA_ADDITIONS == (stage._FACT_OWNER_ADDITION, stage._VOICE_ATTRIBUTION_ADDITION,
                                       stage._DELETION_LEDGER_ADDITION)


def _state(tmp_path, *, pre_4a):
    """A populated state root; pre_4a drops the voice-attribution objects (the pre-4a shape)."""
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    (state / "config").mkdir()
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    try:
        ledger.store("Synthetic automatic fact", "fact", .9, (memory_source(now=NOW - 1),))
        proposal = ledger.store("Synthetic reviewed fact", "fact", 1.0, (memory_source(now=NOW - 1),), proposed=True)
        ledger.confirm(proposal.id, MemoryReviewer("synthetic-reviewer"))
    finally:
        ledger.close()
    with sqlite3.connect(state / PRIVACY) as db:
        # The voice ledger owns this marker row; a bare FactLedger leaves it empty.
        db.execute("INSERT INTO voice_group_schema VALUES (1)")
        if pre_4a:
            db.executescript(";".join("DROP TABLE " + table for table in VOICE_ATTRIBUTION_TABLES))
    return state


def test_fresh_ledger_admits_itself_and_its_authority_covers_the_protected_tables(tmp_path):
    state = _state(tmp_path, pre_4a=False)
    with stage._database_image(state / PRIVACY) as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        contract = stage._schema_contract(db, names)
    assert VOICE_TABLES <= contract["columns"].keys()
    authority = stage.capture_state_authority(state)
    tables = authority["projection"]["tables"][PRIVACY]
    assert all(tables[name]["row_hashes"] == [] for name in PROTECTED)
    with sqlite3.connect(state / PRIVACY) as db:
        seed_voice_attribution_rows(db, NOW)
    seeded = stage.capture_state_authority(state)
    assert seeded["digest"] != authority["digest"]
    assert {name: len(seeded["projection"]["tables"][PRIVACY][name]["row_hashes"]) for name in PROTECTED} == {
        "voice_attribution_consents": 1, "voice_verification_attempts": 2, "voice_attribution_events": 1,
        "voice_attribution_suspensions": 1, "voice_attribution_control": 1}


def test_a_control_switch_change_is_an_authority_change(tmp_path):
    """G1: the switch row is protected, so an offline flip is detected as a different authority."""
    state = _state(tmp_path, pre_4a=False)
    with sqlite3.connect(state / PRIVACY) as db:
        seed_voice_attribution_rows(db, NOW, enabled=1)
    on = stage.capture_state_authority(state)["digest"]
    with sqlite3.connect(state / PRIVACY) as db:
        db.execute("UPDATE voice_attribution_control SET enabled=0, revision=2")
    assert stage.capture_state_authority(state)["digest"] != on


def test_unmigrated_pre_4a_state_has_the_same_digest_under_both_controllers(tmp_path, monkeypatch):
    """1a-F1 row (a), for the voice-attribution group."""
    state = _state(tmp_path, pre_4a=True)
    new = stage.capture_state_authority(state)
    with reference_controller(monkeypatch):
        reference = stage.capture_state_authority(state)
    assert new == reference


def test_initialisation_adds_only_five_empty_projections_and_six_contract_entries(tmp_path, monkeypatch):
    """1a-F1 row (b); the reference controller rejects the migrated state (rollback rejection point)."""
    state = _state(tmp_path, pre_4a=True)
    before = copy.deepcopy(stage.capture_state_authority(state)["projection"])
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    ledger.close()
    after = stage.capture_state_authority(state)["projection"]
    added = after["tables"][PRIVACY]
    columns = added["schema_contract_v1"]["columns"]
    assert {name: columns[name] for name in VOICE_TABLES} == {
        name: dict(item.split(":") for item in stage._TABLE_COLUMNS[name].split()) for name in VOICE_TABLES}
    for name in PROTECTED:
        assert added[name] == {"columns": [item.split(":")[0] for item in stage._TABLE_COLUMNS[name].split()],
                               "row_hashes": []}
    stripped = copy.deepcopy(after)
    for name in PROTECTED:
        del stripped["tables"][PRIVACY][name]
    for name in VOICE_TABLES:
        del stripped["tables"][PRIVACY]["schema_contract_v1"]["columns"][name]
    assert stripped == before
    with reference_controller(monkeypatch):
        with pytest.raises(ValueError, match="runtime_admission_unknown_schema"):
            stage.capture_state_authority(state)


def test_derived_expectation_admits_only_the_empty_voice_attribution_group(tmp_path):
    """A derivation from a pre-4a image migrates in the empty group, and nothing else is admitted."""
    state = _state(tmp_path, pre_4a=True)
    source = stage.capture_state_authority(state)["projection"]
    ledger = FactLedger(str(state / PRIVACY), clock=lambda: NOW)
    ledger.initialize()
    ledger.close()
    derived = stage.capture_state_authority(state)["projection"]
    assert source != derived
    expected = copy.deepcopy(source)
    stage._expect_empty_schema_additions(expected, derived)
    assert expected == derived

    with sqlite3.connect(state / PRIVACY) as db:  # any row in the migrated group still mismatches
        seed_voice_attribution_rows(db, NOW)
    seeded = stage.capture_state_authority(state)["projection"]
    expected = copy.deepcopy(source)
    stage._expect_empty_schema_additions(expected, seeded)
    assert expected != seeded

    unchanged = copy.deepcopy(derived)  # a group already in the source is not re-expected
    stage._expect_empty_schema_additions(unchanged, seeded)
    assert unchanged == derived


def test_a_partial_voice_group_in_the_derived_state_is_not_admitted(tmp_path):
    state = _state(tmp_path, pre_4a=True)
    source = stage.capture_state_authority(state)["projection"]
    derived = copy.deepcopy(source)
    contract = derived["tables"][PRIVACY]["schema_contract_v1"]["columns"]
    for name in VOICE_ATTRIBUTION_TABLES[:-1]:  # every table except the marker
        contract[name] = dict(item.split(":") for item in stage._TABLE_COLUMNS[name].split())
    expected = copy.deepcopy(source)
    stage._expect_empty_schema_additions(expected, derived)
    assert expected == source != derived


def test_projection_size_under_a_maximal_synthetic_load_is_far_below_the_binding_cap(tmp_path):
    """R10: consents <= 256, attempts 12/person/day, events capped, suspensions <= 256, control <= 1.

    About 4.1k row hashes against B19's ~250k limit (scripts/stage_runtime.py:320-323).
    """
    state = _state(tmp_path, pre_4a=False)
    with sqlite3.connect(state / PRIVACY) as db:
        profiles = ["00000000-0000-4000-8000-%012x" % index for index in range(256)]
        for index, profile in enumerate(profiles):
            insert_deleted_profile(db, profile, NOW, index)
        db.executemany("INSERT INTO voice_attribution_consents VALUES (?,'active',1,?,1,1,?,'p')",
                       [("user_%03d" % index, profile, NOW) for index, profile in enumerate(profiles)])
        db.executemany("INSERT INTO voice_verification_attempts VALUES (?,?,'failed')",
                       [("user_%03d" % person, NOW + attempt) for person in range(256) for attempt in range(12)])
        db.executemany("INSERT INTO voice_attribution_events VALUES (?,?,'not_mine',0,?)",
                       [("evt_%d_%d" % (index, event), profile, NOW)
                        for index, profile in enumerate(profiles) for event in range(2)])
        db.executemany("INSERT INTO voice_attribution_suspensions VALUES (?,1,?,'not_mine_threshold')",
                       [(profile, NOW) for profile in profiles])
        db.execute("INSERT INTO voice_attribution_control VALUES (1,1,1,?,'o')", (NOW,))
    projection = stage.capture_state_authority(state)["projection"]["tables"][PRIVACY]
    voice = {name: projection[name] for name in PROTECTED}
    assert sum(len(table["row_hashes"]) for table in voice.values()) == 256 + 256 * 12 + 512 + 256 + 1
    assert len(json.dumps(voice)) < 16 * 1024 * 1024 // 50  # under 2% of the binding cap
