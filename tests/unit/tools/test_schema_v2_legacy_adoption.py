"""SCHEMA-V2 admits a closed legacy reminders database and keeps latest authority."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

import pytest

from scripts import stage_runtime as stage
from src.memory import payload_derivation as payload
from src.memory.state_snapshot import snapshot_closed_state
from src.reminders.reminder_store import ReminderStore, reminder_schema
from tests.mocks import compaction_world as cw
from tests.mocks import launcher_rehearsal as rehearsal
from tests.mocks.slot_reconciliation import fixture_layout


REMINDERS = "data/reminders.db"
GRAPH_REMINDERS = "data/graph/reminders.db"
PRIVACY = "data/graph/privacy.db"
PROTECTED = {"reminders", "reminder_home_actions", "timers", "alert_preferences"}
# Exact pre-SCHEMA-V2 store SQL from the historical managed slots (80b47f5).
LEGACY_DDL = """CREATE TABLE IF NOT EXISTS reminders (
                        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, text TEXT NOT NULL,
                        trigger_at REAL NOT NULL, created_at REAL NOT NULL,
                        recurrence TEXT, recurrence_end REAL, ha_actions TEXT,
                        state TEXT NOT NULL DEFAULT 'active', last_fired_at REAL,
                        fire_count INTEGER NOT NULL DEFAULT 0,
                        scope TEXT NOT NULL DEFAULT 'private' CHECK (scope IN ('private', 'shared'))
                    );
CREATE INDEX IF NOT EXISTS idx_reminders_trigger ON reminders (trigger_at) WHERE state = 'active';
CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders (user_id, state);
CREATE TABLE IF NOT EXISTS reminder_home_actions (
                        request_id TEXT PRIMARY KEY, body TEXT NOT NULL,
                        status TEXT NOT NULL, reason TEXT
                    );
"""
LEGACY_ROWS = """
INSERT INTO reminders (id, user_id, text, trigger_at, created_at, scope)
VALUES ('legacy', 'synthetic-user', 'Legacy reminder', 500, 20, 'shared');
INSERT INTO reminder_home_actions VALUES ('action', '{}', 'sent', NULL);
"""
LATEST_ROWS = """
UPDATE reminders SET text = 'Latest reminder', delivery_room = 'office';
UPDATE reminder_home_actions SET body = '{"latest":true}';
INSERT INTO timers (id, label, actor, state, duration_s, ends_at, created_at, updated_at)
VALUES ('timer', 'Test timer', 'synthetic-user', 'running', 300, 600, 300, 300);
INSERT INTO alert_preferences (key, value_json, updated_at)
VALUES ('alert_zones', '["office"]', 300);
"""


@pytest.fixture(scope="module")
def archive(tmp_path_factory) -> Path:
    return cw.build_x(tmp_path_factory.mktemp("schema-v2-runtime"))


def _script(path: Path, sql: str) -> None:
    with closing(sqlite3.connect(path)) as db:
        db.executescript(sql)


def _reminder_projection(root: Path) -> dict:
    binding = stage._binding_snapshot(root, "candidate")
    return binding["authority"]["projection"]["tables"][REMINDERS]


async def test_prepare_legacy_state_and_bind_it_before_and_after_real_migration(archive, tmp_path):
    source = rehearsal.offline_source(tmp_path)
    _script(source / REMINDERS, LEGACY_DDL + LEGACY_ROWS)
    root = (tmp_path / "staging").resolve()
    stage.prepare_slot(archive, root, "candidate", source)
    before = _reminder_projection(root)
    assert set(before["schema_contract_v1"]["columns"]) == {"reminders", "reminder_home_actions"}
    assert len(before["reminders"]["columns"]) == 12
    assert len(before["reminders"]["row_hashes"]) == 1

    path = root / "slots/candidate/state" / REMINDERS
    store = ReminderStore(str(path))
    try:
        await store.initialize()
    finally:
        await store.close()

    after = _reminder_projection(root)
    columns = after["schema_contract_v1"]["columns"]
    assert set(columns) == PROTECTED
    assert columns["reminders"] == {**before["schema_contract_v1"]["columns"]["reminders"], "delivery_room": "TEXT"}
    assert after["reminder_home_actions"] == before["reminder_home_actions"]
    with closing(sqlite3.connect(path)) as db:
        assert db.execute(
            "SELECT id, user_id, text, trigger_at, created_at, scope, state, delivery_room FROM reminders"
        ).fetchall() == [("legacy", "synthetic-user", "Legacy reminder", 500, 20, "shared", "active", None)]


@pytest.mark.parametrize("table", ["timers", "alert_preferences"])
def test_prepare_legacy_state_with_v2_table_refuses_binding(archive, tmp_path, table):
    source = rehearsal.offline_source(tmp_path)
    _script(source / REMINDERS, LEGACY_DDL + LEGACY_ROWS + rehearsal.table_ddl(table))
    root = (tmp_path / "staging").resolve()
    stage.prepare_slot(archive, root, "candidate", source)
    with pytest.raises(ValueError, match="^runtime_admission_unknown_schema$"):
        _reminder_projection(root)
    assert stage.read_admission_head(root) is None


@pytest.mark.parametrize("mutation", [
    "DROP INDEX idx_timers_state_ends_at",
    "CREATE INDEX unexpected_idx ON reminders (text)",
    "CREATE INDEX unexpected_idx ON reminder_home_actions (status)",
])
def test_cut_rejects_unrecognized_schema_objects(mutation):
    def schema(db):
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]

    with closing(sqlite3.connect(":memory:")) as db:
        db.executescript(LEGACY_DDL)
        ordinary = schema(db)
        reminder_schema(db)
        assert payload._reminder_schema_upgraded(ordinary, schema(db))
        db.execute(mutation)
        assert not payload._reminder_schema_upgraded(ordinary, schema(db))


def _empty_payload_table(name: str) -> str:
    columns = (" ".join(field.split(":")) for field in stage._TABLE_COLUMNS[name].split())
    first, *rest = columns
    return f"CREATE TABLE IF NOT EXISTS {name} ({', '.join([first + ' PRIMARY KEY', *rest])});"


def _seed_payload() -> str:
    # Only the reminders database carries data under test. An actual ledger and
    # closed snapshots carry the empty payload families through the public API.
    tables = (payload.PAYLOAD_TABLES | {"live_deliveries", "live_delivery_sources", "live_delivery_lineage"}
              ) - {"metrics", "lineage"}
    ddl = "\n".join(map(_empty_payload_table, sorted(tables)))
    return cw._LEDGER_OPEN.replace("data' / 'privacy.db", "data' / 'graph' / 'privacy.db") + "\n" + "\n".join([
        "from src.memory.erasure_journal import ErasureParticipant",
        "ledger.erasures.register_target(ErasureParticipant.SEMANTIC_INDEX, 1,",
        "    {'path': str(state / 'data/graph/chroma'), 'collection': 'unit_policy_v1'})",
        "ledger._db.execute('INSERT INTO voice_group_schema VALUES (1)')",
        "ledger.close()",
        f"with closing(sqlite3.connect(state / {PRIVACY!r})) as db:",
        f"    db.executescript({ddl!r})",
        "from src.obs.diagnostic_store import DiagnosticStore",
        "from src.obs.metrics_emitter import project_diagnostic",
        "metrics = DiagnosticStore(state / 'data/graph/metrics_policy_v1.jsonl', clock=lambda: 350,",
        "    retention_days=1, rotate_bytes=4096, project=project_diagnostic)",
        "try:",
        "    metrics.prune()",
        "finally:",
        "    metrics.close()",
    ])


def _execute_reminders(sql: str) -> str:
    return ("from contextlib import closing\nimport sqlite3\n"
            f"with closing(sqlite3.connect(state / {GRAPH_REMINDERS!r})) as db:\n"
            f"    db.executescript({sql!r})")


@pytest.mark.parametrize("migrate", [True, False], ids=["alter", "fresh"])
def test_cut_clones_latest_protected_reminders_end_to_end(archive, tmp_path, migrate):
    source = rehearsal.offline_source(tmp_path)
    (source / "data/graph").mkdir()
    root = (tmp_path / "staging").resolve()
    stage.prepare_slot(archive, root, "candidate", source)
    ordinary_id = cw.run(root, "candidate", _seed_payload(), _execute_reminders(LEGACY_DDL + LEGACY_ROWS))
    with stage.AdmissionController(root) as controller:
        ordinary = snapshot_closed_state(controller, ordinary_id, tmp_path / "ordinary")
    initialize = [
        "from contextlib import closing",
        "import sqlite3",
        "from src.reminders.reminder_store import reminder_schema",
        f"path = state / {GRAPH_REMINDERS!r}",
    ]
    if not migrate:
        initialize.append("path.unlink()")
    initialize.extend(["with closing(sqlite3.connect(path)) as db, db:", "    reminder_schema(db)"])
    latest_sql = LATEST_ROWS if migrate else LEGACY_ROWS + LATEST_ROWS
    latest_id = cw.run(root, "candidate", "\n".join(initialize), _execute_reminders(latest_sql))
    empty = tmp_path / "empty"
    (empty / "config").mkdir(parents=True)
    (empty / "data").mkdir()
    stage.prepare_slot(archive, root, "derived", empty)
    destination = root / "slots/derived/state"
    with stage.AdmissionController(root) as controller:
        latest = snapshot_closed_state(controller, latest_id, tmp_path / "latest")
        plan = payload.plan_payload_cut(ordinary, latest, now=350, layout=fixture_layout(ordinary, latest, destination))
        assert not (PROTECTED & {cut.table for cut in plan.tables})
        assert not (destination / GRAPH_REMINDERS).exists()
        derived = payload.materialize_derived_state(plan, destination, controller=controller)
        assert derived.serving is False
        assert stage.read_admission_head(root)["binding_id"] == latest_id
    assert (destination / GRAPH_REMINDERS).read_bytes() == (
        latest.snapshot_root / "state" / GRAPH_REMINDERS).read_bytes()
    tables = latest.manifest["databases"][GRAPH_REMINDERS]["tables"]
    assert set(tables) == PROTECTED
    assert all(table["rows"] for table in tables.values())
