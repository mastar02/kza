"""Epoch 2 migrations and alert preferences on real SQLite connections."""

import asyncio
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import sys

import pytest

from scripts import adoption
from src.reminders.reminder_store import AlertZonesConflict, ReminderStore, TimerState, reminder_schema


REPO = Path(__file__).resolve().parents[3]
REMINDERS_DB = Path("data/reminders.db")
PLAN = "a" * 64


def _rows(path: Path, sql: str) -> list[tuple]:
    with closing(sqlite3.connect(path)) as db:
        return db.execute(sql).fetchall()


@pytest.fixture
def legacy_db(tmp_path):
    """Materialize the b0866e8 ReminderStore DDL in shallow clones too."""
    path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(path)) as db:
        db.executescript("""
            CREATE TABLE reminders (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
                trigger_at REAL NOT NULL, created_at REAL NOT NULL, recurrence TEXT,
                recurrence_end REAL, ha_actions TEXT, state TEXT NOT NULL DEFAULT 'active',
                last_fired_at REAL, fire_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX idx_reminders_trigger ON reminders (trigger_at) WHERE state = 'active';
            CREATE INDEX idx_reminders_user ON reminders (user_id, state);
            CREATE TABLE reminder_home_actions (
                request_id TEXT PRIMARY KEY, body TEXT NOT NULL, status TEXT NOT NULL, reason TEXT
            );
            INSERT INTO reminders (id, user_id, text, trigger_at, created_at, recurrence, recurrence_end)
            VALUES ('reminder-a', 'user-a', 'regar plantas', 100, 50, 'daily', 300);
            INSERT INTO reminder_home_actions (request_id, body, status)
            VALUES ('request-a', '{"entity_id":"light.hall"}', 'sending');
        """)
    return path


async def test_open_migrates_epoch1_reminders_and_keeps_existing_row(legacy_db):
    with closing(sqlite3.connect(legacy_db)) as db, db:
        db.execute("ALTER TABLE reminders ADD COLUMN scope TEXT NOT NULL DEFAULT 'private' "
                   "CHECK (scope IN ('private', 'shared'))")
        db.execute("UPDATE reminders SET scope = 'shared'")

    [old_row] = _rows(legacy_db, "SELECT * FROM reminders")
    store = ReminderStore(str(legacy_db))
    await store.initialize()
    try:
        previous = await store.get_by_id(old_row[0], shared_only=True)
        assert previous.delivery_room is None
        saved = await store.create("user-b", "probar la salida", 200.0, delivery_room="cocina")
        assert (await store.get_by_id(saved.id)).delivery_room == "cocina"
        assert _rows(legacy_db, "SELECT * FROM reminders ORDER BY rowid")[0] == (*old_row, None)
        assert _rows(legacy_db, "SELECT * FROM timers") == []
        assert _rows(legacy_db, "SELECT * FROM alert_preferences") == []
    finally:
        await store.close()


async def test_alert_zones_persist_and_guard_updates_and_reset(tmp_path):
    path = tmp_path / "reminders.db"
    store = ReminderStore(str(path))
    await store.initialize()
    try:
        assert await store.get_alert_zones() is None
        created = await store.set_alert_zones(["cocina", "salón"], expected_revision=0)
        assert created.zones == ("cocina", "salón") and created.revision == 1

        writes = await asyncio.gather(
            store.set_alert_zones(["pasillo"], expected_revision=1),
            store.set_alert_zones([], expected_revision=1),
            return_exceptions=True,
        )
        [winner] = [result for result in writes if not isinstance(result, BaseException)]
        [loser] = [result for result in writes if isinstance(result, BaseException)]
        assert winner.revision == 2 and isinstance(loser, AlertZonesConflict)
        assert await store.get_alert_zones() == winner

        empty = await store.set_alert_zones([])
        assert empty.zones == () and empty.revision == 3
        reset = await store.set_alert_zones(None, expected_revision=3)
        assert reset.zones is None and reset.revision == 4
        with pytest.raises(AlertZonesConflict, match="alert_zones_revision_conflict"):
            await store.set_alert_zones(["cocina"], expected_revision=0)
        with pytest.raises(ValueError, match="invalid_alert_zones"):
            await store.set_alert_zones([" "])
    finally:
        await store.close()

    assert _rows(path, "SELECT * FROM alert_preferences") == [("alert_zones", "null", reset.updated_at, 4)]
    await store.initialize()
    try:
        assert await store.get_alert_zones() == reset
        restored = await store.set_alert_zones(["cocina"], expected_revision=4)
        assert restored.revision == 5
    finally:
        await store.close()


def test_timer_ddl_is_idempotent_and_checks_states(tmp_path):
    with closing(sqlite3.connect(tmp_path / "reminders.db")) as db:
        reminder_schema(db)
        insert = ("INSERT INTO timers (id, label, actor, state, duration_s, created_at, updated_at) "
                  "VALUES (?, 'té', 'user-a', ?, 60, 100, 100)")
        db.executemany(insert, ((state, state) for state in TimerState))
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
            db.execute(insert, ("bad", "invalid"))
        schema = "SELECT type, name, sql FROM sqlite_master ORDER BY name"
        before = db.execute(schema).fetchall()
        reminder_schema(db)
        assert db.execute(schema).fetchall() == before
        assert db.execute("SELECT state, revision FROM timers ORDER BY rowid").fetchall() == [
            (state, 1) for state in TimerState]
        assert [row[2] for row in db.execute("PRAGMA index_info(idx_timers_state_ends_at)")] == ["state", "ends_at"]


def test_schema_only_adopts_b0866e8_copy_without_send_recovery(tmp_path, legacy_db):
    state = tmp_path / "copy"
    path = state / REMINDERS_DB
    path.parent.mkdir(parents=True)
    (state / "config").mkdir()
    (state / "config/settings.yaml").write_text("reminders:\n  db_path: ./data/reminders.db\n")
    source_bytes = legacy_db.read_bytes()
    shutil.copyfile(legacy_db, path)
    adoption.write_copy_marker(state, PLAN)
    before = adoption.table_snapshot(state)
    [old_row] = _rows(legacy_db, "SELECT * FROM reminders")

    assert adoption.run_child(sys.executable, REPO, state, "schema")["schema_only"]

    assert adoption.verify_contracts(state) == 1
    after = adoption.table_snapshot(state)
    assert adoption.verify_s3(before, after, adoption_time=500.0, plan_sha256=PLAN, residue={}) == {
        "column_additions": 2}
    assert _rows(path, "SELECT * FROM reminders") == [(*old_row, "private", None)]
    assert _rows(path, "SELECT status, reason FROM reminder_home_actions") == [("sending", None)]
    assert _rows(path, "SELECT * FROM timers") == []
    assert _rows(path, "SELECT * FROM alert_preferences") == []
    assert legacy_db.read_bytes() == source_bytes
