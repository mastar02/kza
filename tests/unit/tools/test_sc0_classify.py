"""SC-0's writer classification is keyed by (file, function, verb and table), never by line (SCR-P-1).

The SC-R review found the R5V-4 inventory stale at d7a2528: line-pinned rules
had drifted, so DashboardAuthority's every-transaction purges read 'user', and
ReminderStore.purge_notice_rows (a recurring retention loop deleting rows of a
protected table) fell into reminder_store.py's catch-all 'user' rule. These
tests fail when a projected-table writer has no rule of its own, when two
rules claim it, or when a rule no longer names any writer.
"""
from pathlib import Path

import pytest

from tools.sc0 import classify
from tools.sc0.write_inventory import scan

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def rows():
    return scan(REPO)


def _row(rows, function: str, key: str, sql: str = "") -> list[dict]:
    found = [row for row in rows if row["function"] == function and row["verb"] + " " + row["table"] == key
             and sql in row["sql"]]
    assert found, (function, key, sql)
    return found


def test_every_rule_names_one_function_and_explicit_writes():
    """No catch-all can exist: each rule names an exact function and 'VERB table' keys."""
    for rule in classify.RULES:
        assert rule.file.startswith("src/") and rule.file.endswith(".py"), rule
        assert rule.function and ":" not in rule.function, rule
        assert rule.writes and all(" " in key and "*" not in key for key in rule.writes), rule
        assert rule.kind in {"boot_repair", "boot_init", "every_txn", "recurring", "close_path", "ingress",
                             "user", "controller", "upgrade", "boot_config", "follows_caller"}, rule


def test_every_projected_table_writer_has_exactly_one_keyed_rule(rows):
    """A new writer, or one moved to another function, fails here until a rule names it."""
    assert rows
    problems = []
    for row in rows:
        kind, _how = classify.rule_for(row)
        if kind in {"unclassified", "ambiguous"}:
            problems.append((row["file"], row["function"], row["verb"], row["table"], kind))
    assert problems == []


def test_a_writer_outside_every_rule_is_unclassified_not_caught_by_its_file(rows):
    """The purge_notice_rows shape: a new function of reminder_store.py deleting a protected table."""
    [purge] = _row(rows, "ReminderStore.purge_notice_rows.operation", "DELETE FROM reminder_home_actions")
    assert classify.rule_for({**purge, "function": "ReminderStore.new_writer.operation"})[0] == "unclassified"
    assert classify.rule_for({**purge, "table": "reminders"})[0] == "unclassified"
    assert classify.rule_for({**purge, "verb": "UPDATE"})[0] == "unclassified"
    assert classify.rule_for({**purge, "file": "src/reminders/other.py"})[0] == "unclassified"


def test_the_class_does_not_depend_on_the_line(rows):
    for row in rows:
        assert classify.rule_for({**row, "line": row["line"] + 7}) == classify.rule_for(row)


def test_no_rule_is_stale(rows):
    """Every rule key still names a writer (a renamed function cannot leave its old rule behind)."""
    present = {(row["file"], row["function"], row["verb"] + " " + row["table"]) for row in rows}
    stale = [(rule.file, rule.function, key) for rule in classify.RULES for key in rule.writes
             if (rule.file, rule.function, key) not in present]
    assert stale == []
    for rule in classify.RULES:
        if rule.sql:
            assert any(row["file"] == rule.file and row["function"] == rule.function and rule.sql in row["sql"]
                       for row in rows), rule


def test_the_rows_the_line_pins_lost_are_classified_again(rows):
    """SCR-P-1 (1): the every-transaction purges and the upgrade migrations; (2) purge_notice_rows."""
    transaction = "DashboardAuthority._transaction"
    every_txn = [("DELETE FROM sessions", "expires_at<=?"), ("DELETE FROM credentials", "expires_at<=?"),
                 ("DELETE FROM sessions", "legacy_digest IS NOT NULL"),
                 ("INSERT OR REPLACE INTO auth_meta", "VALUES (?,?)"),
                 ("DELETE FROM credentials", "IN (SELECT user_id FROM retired_users)"),
                 ("DELETE FROM sessions", "NOT IN (SELECT id FROM credentials)"),
                 ("INSERT OR REPLACE INTO auth_meta", "'clock_floor'")]
    for key, sql in every_txn:
        for row in _row(rows, transaction, key, sql):
            assert classify.rule_for(row)[0] == "every_txn", (key, sql)
    for row in _row(rows, transaction, "DELETE FROM sessions", "WHERE digest=?"):
        assert classify.rule_for(row)[0] == "user"
    for row in _row(rows, transaction, "INSERT INTO sessions"):
        assert classify.rule_for(row)[0] == "user"
    for function, key in (("FactLedger.initialize", "ALTER TABLE fact_sources"),
                          ("FactLedger.initialize", "UPDATE facts"),
                          ("CaptureRecorders._initialize", "ALTER TABLE capture_browser_recorders")):
        for row in _row(rows, function, key):
            assert classify.rule_for(row)[0] == "upgrade", (function, key)
    for row in _row(rows, "CaptureRecorders._initialize", "UPDATE capture_browser_recorders"):
        assert classify.rule_for(row)[0] == "boot_repair"
    [purge] = _row(rows, "ReminderStore.purge_notice_rows.operation", "DELETE FROM reminder_home_actions")
    assert purge["projection"] == "full"
    assert classify.rule_for(purge)[0] == "recurring"


def test_classify_without_logs_lists_every_writer(tmp_path):
    """The R5V-4 run with no logs: one row per static writer plus UNSCANNED, none unclassified."""
    found = classify.classify(tmp_path)
    assert len(found) == len(scan(REPO)) + len(classify.UNSCANNED)
    assert not [row for row in found if row["class"] in {"unclassified", "ambiguous"}]
