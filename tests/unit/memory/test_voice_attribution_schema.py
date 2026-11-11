"""U3b-4a-0: voice-attribution schema in the privacy ledger (design u3b-4 v4).

The initializer creates the exact DDL or verifies it, validates every value
against the closed enums, and never prunes: an aged closed state opens
byte-identical (review R1). Every failure leaves the file unchanged.
"""

import json
from pathlib import Path
import sqlite3
import time

import pytest

from src.memory.fact_ledger import FactLedger
from src.memory.voice_attribution import (
    VoiceAttributionEventKind,
    VoiceConsentState,
    VoiceSuspensionReason,
    VoiceVerificationOutcome,
)
from src.memory.voice_schema import VOICE_ATTRIBUTION_TABLES, voice_attribution_schema_objects

NOW = 2_000_000_000.0
DAY = 86400.0
PROFILE = "00000000-0000-4000-8000-0000000000a1"
PERSON = "user_synthetic_a"


def profile_sources(now: float, index: int = 0) -> str:
    """A valid, long-expired source list for a soft-deleted synthetic profile."""
    return json.dumps([{"capture_id": "00000000-0000-4000-8000-%012x" % (0xC000 + index),
                        "conversation_id": "00000000-0000-4000-8000-%012x" % (0xD000 + index),
                        "captured_at": now - 60 * DAY, "expires_at": now - 59 * DAY, "origin": "voice"}],
                      sort_keys=True)


def insert_deleted_profile(db: sqlite3.Connection, profile: str, now: float, index: int = 0) -> None:
    """A soft-deleted reviewed profile (as VoiceLedger._delete leaves it): FK target only."""
    db.execute("INSERT OR IGNORE INTO voice_profiles VALUES (?,NULL,'enc_synthetic',192,NULL,?,?,?,2,?)",
               (profile, profile_sources(now, index), now - 60 * DAY, "synthetic-reviewer", now - 50 * DAY))


def seed_voice_attribution_rows(db: sqlite3.Connection, now: float, *, enabled: int = 1) -> None:
    """One row of every kind, aged past every window; the profile row keeps the FKs valid."""
    insert_deleted_profile(db, PROFILE, now)
    db.execute("INSERT INTO voice_attribution_consents VALUES (?,?,3,?,1,1,?,?)",
               (PERSON, VoiceConsentState.ACTIVE.value, PROFILE, now - 40 * DAY, PERSON))
    db.execute("INSERT INTO voice_verification_attempts VALUES (?,?,?)",
               (PERSON, now - 40 * DAY, VoiceVerificationOutcome.WINDOW_OPENED.value))
    db.execute("INSERT INTO voice_verification_attempts VALUES (?,?,?)",
               (PERSON, now - 40 * DAY + 1, VoiceVerificationOutcome.PASSED.value))
    db.execute("INSERT INTO voice_attribution_events VALUES ('evt_synthetic',?,?,15,?)",
               (PROFILE, VoiceAttributionEventKind.NOT_MINE.value, now - 45 * DAY))
    db.execute("INSERT INTO voice_attribution_suspensions VALUES (?,1,?,?)",
               (PROFILE, now - 44 * DAY, VoiceSuspensionReason.NOT_MINE_THRESHOLD.value))
    db.execute("INSERT INTO voice_attribution_control VALUES (1,?,1,?,'synthetic-operator')",
               (enabled, now - 40 * DAY))


@pytest.fixture
def path(tmp_path):
    path = tmp_path / "privacy.db"
    ledger = FactLedger(str(path), clock=lambda: NOW)
    ledger.initialize()
    ledger.close()
    return path


def _objects(path: Path) -> list[tuple]:
    with sqlite3.connect(path) as db:
        return sorted(tuple(row) for row in db.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE tbl_name IN (%s) OR name IN (%s)"
            % ((",".join("?" * len(VOICE_ATTRIBUTION_TABLES)),) * 2), VOICE_ATTRIBUTION_TABLES * 2))


def test_fresh_ledger_has_exactly_the_ddl_objects_and_the_marker(path):
    expected = sorted((row["type"], row["name"], row["tbl_name"], row["sql"])
                      for row in voice_attribution_schema_objects())
    assert _objects(path) == expected
    assert len(expected) == 11  # 6 tables, 4 primary-key autoindexes, 1 index
    with sqlite3.connect(path) as db:
        assert [tuple(row) for row in db.execute("SELECT version FROM voice_attribution_schema")] == [(1,)]
        for table in VOICE_ATTRIBUTION_TABLES[:-1]:
            assert db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0] == 0  # nothing at open (G1)


@pytest.mark.parametrize("seeded", ["fixed_2033", "real_clock"])
def test_reopening_a_populated_ledger_changes_nothing_even_long_after_every_window(path, seeded):
    """R1: no time-based pruning at open; a clock years ahead leaves the file byte-identical.

    4a-0 review F2: the 'real_clock' seed is relative to time.time(), so the
    rows are already past every window by the wall clock too; an open-time
    prune by wall-clock time (mutant M3) deletes them and fails this test.
    """
    base = time.time() if seeded == "real_clock" else NOW
    with sqlite3.connect(path) as db:
        seed_voice_attribution_rows(db, base)
    before = path.read_bytes()
    for clock in (None, base, base + 400 * DAY):
        ledger = FactLedger(str(path), clock=None if clock is None else (lambda clock=clock: clock))
        ledger.initialize()
        assert ledger.authority_available
        ledger.close()
        assert path.read_bytes() == before


def test_every_enumerated_value_is_accepted(path):
    with sqlite3.connect(path) as db:
        insert_deleted_profile(db, PROFILE, NOW)
        for index, state in enumerate(VoiceConsentState):
            verified = (PROFILE, 1, 1) if state is VoiceConsentState.ACTIVE else (None, None, None)
            db.execute("INSERT INTO voice_attribution_consents VALUES (?,?,1,?,?,?,?,'p')",
                       ("user_" + str(index), state.value, *verified, NOW))
        for index, outcome in enumerate(VoiceVerificationOutcome):
            db.execute("INSERT INTO voice_verification_attempts VALUES (?,?,?)", (PERSON, NOW + index, outcome.value))
        for bucket in (-20, 0, 20):
            db.execute("INSERT INTO voice_attribution_events VALUES (?,?,?,?,?)",
                       ("evt_" + str(bucket), PROFILE, VoiceAttributionEventKind.NOT_MINE.value, bucket, NOW))
        db.execute("INSERT INTO voice_attribution_suspensions VALUES (?,1,?,?)",
                   (PROFILE, NOW, VoiceSuspensionReason.NOT_MINE_THRESHOLD.value))
        db.execute("INSERT INTO voice_attribution_control VALUES (1,0,1,?,'synthetic-operator')", (NOW,))
    ledger = FactLedger(str(path), clock=lambda: NOW)
    ledger.initialize()
    ledger.close()


@pytest.mark.parametrize("damage", [
    "DROP TABLE voice_attribution_consents",
    "DROP TABLE voice_verification_attempts",
    "DROP TABLE voice_attribution_events",
    "DROP TABLE voice_attribution_suspensions",
    "DROP TABLE voice_attribution_control",
    "DROP TABLE voice_attribution_schema",
    "DROP INDEX voice_attribution_events_profile",
    "CREATE INDEX voice_attribution_consents_extra ON voice_attribution_consents(state)",
    "DROP INDEX voice_attribution_events_profile; "
    "CREATE INDEX voice_attribution_events_profile ON voice_attribution_events(at)",
    "ALTER TABLE voice_attribution_control ADD COLUMN extra TEXT",
    "DROP TABLE voice_attribution_control; CREATE TABLE voice_attribution_control (id INTEGER PRIMARY KEY, "
    "enabled INTEGER NOT NULL CHECK(enabled IN (0,1)), revision INTEGER NOT NULL, changed_at REAL NOT NULL, "
    "changed_by TEXT NOT NULL)",
    "DROP TABLE voice_attribution_suspensions; CREATE TABLE voice_attribution_suspensions (profile_id TEXT "
    "PRIMARY KEY REFERENCES voice_profiles(id), revision INTEGER NOT NULL, suspended_at REAL NOT NULL, "
    "reason TEXT NOT NULL)",  # nullable text key (1a-F7)
    "UPDATE voice_attribution_schema SET version=2",
    "INSERT INTO voice_attribution_schema VALUES (2)",
    "DELETE FROM voice_attribution_schema",
    "DROP TABLE voice_attribution_schema; CREATE VIEW voice_attribution_schema AS SELECT 1 AS version",
])
def test_partial_or_foreign_shapes_fail_closed_with_the_file_unchanged(path, damage):
    with sqlite3.connect(path) as db:
        db.executescript(damage)
    before = path.read_bytes()
    ledger = FactLedger(str(path), clock=lambda: NOW)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="voice_attribution_schema_unavailable"):
            ledger.initialize()
        assert ledger._db is None and not ledger.authority_available
        assert path.read_bytes() == before


@pytest.mark.parametrize("statement,values", [
    ("INSERT INTO voice_attribution_consents VALUES (?,'enrolled',1,NULL,NULL,NULL,?,'p')", (PERSON, NOW)),
    ("INSERT INTO voice_attribution_consents VALUES (?,'active',1,NULL,NULL,NULL,?,'p')", (PERSON, NOW)),
    ("INSERT INTO voice_attribution_consents VALUES (?,'revoked',0,NULL,NULL,NULL,?,'p')", (PERSON, NOW)),
    ("INSERT INTO voice_attribution_consents VALUES (?,'revoked',1,?,NULL,1,?,'p')", (PERSON, PROFILE, NOW)),
    ("INSERT INTO voice_attribution_consents VALUES (?,'revoked',1,NULL,NULL,NULL,?,'')", (PERSON, NOW)),
    ("INSERT INTO voice_verification_attempts VALUES (?,?,'skipped')", (PERSON, NOW)),
    ("INSERT INTO voice_verification_attempts VALUES ('',?,'passed')", (NOW,)),
    ("INSERT INTO voice_attribution_events VALUES ('evt',?,'misheard',0,?)", (PROFILE, NOW)),
    ("INSERT INTO voice_attribution_events VALUES ('evt',?,'not_mine',21,?)", (PROFILE, NOW)),
    ("INSERT INTO voice_attribution_events VALUES ('evt',?,'not_mine',1.5,?)", (PROFILE, NOW)),
    ("INSERT INTO voice_attribution_suspensions VALUES (?,1,?,'manual')", (PROFILE, NOW)),
    ("INSERT INTO voice_attribution_suspensions VALUES (?,0,?,'not_mine_threshold')", (PROFILE, NOW)),
    ("INSERT INTO voice_attribution_control VALUES (2,0,1,?,'o')", (NOW,)),
    ("INSERT INTO voice_attribution_control VALUES (1,2,1,?,'o')", (NOW,)),
    ("INSERT INTO voice_attribution_control VALUES (1,1,0,?,'o')", (NOW,)),
    ("INSERT INTO voice_attribution_control VALUES (1,1,1,?,'')", (NOW,)),
    ("INSERT INTO voice_attribution_control VALUES (1,'on',1,?,'o')", (NOW,)),  # no integer affinity
])
def test_unknown_values_fail_closed_at_open_with_the_file_unchanged(path, statement, values):
    with sqlite3.connect(path) as db:
        db.execute(statement, values)
    before = path.read_bytes()
    ledger = FactLedger(str(path), clock=lambda: NOW)
    with pytest.raises(RuntimeError, match="voice_attribution_value_unknown"):
        ledger.initialize()
    assert ledger._db is None and path.read_bytes() == before


def test_a_second_control_row_fails_closed(path):
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO voice_attribution_control VALUES (1,0,1,?,'o')", (NOW,))
        db.execute("INSERT INTO voice_attribution_control VALUES (2,0,1,?,'o')", (NOW,))
    with pytest.raises(RuntimeError, match="voice_attribution_value_unknown"):
        FactLedger(str(path), clock=lambda: NOW).initialize()


def test_a_pre_4a_ledger_migrates_with_empty_tables_and_keeps_its_other_rows(tmp_path):
    path = tmp_path / "privacy.db"
    ledger = FactLedger(str(path), clock=lambda: NOW)
    ledger.initialize()
    ledger.close()
    with sqlite3.connect(path) as db:
        db.executescript(";".join("DROP TABLE " + table for table in VOICE_ATTRIBUTION_TABLES))
        others = sorted(tuple(row) for row in db.execute("SELECT type,name,sql FROM sqlite_master"))
    ledger = FactLedger(str(path), clock=lambda: NOW)
    ledger.initialize()
    ledger.close()
    with sqlite3.connect(path) as db:
        after = sorted(tuple(row) for row in db.execute("SELECT type,name,sql FROM sqlite_master"))
    added = sorted(set(after) - set(others))
    assert set(others) <= set(after)
    assert sorted(name for _kind, name, _sql in added) == sorted(
        row["name"] for row in voice_attribution_schema_objects())
