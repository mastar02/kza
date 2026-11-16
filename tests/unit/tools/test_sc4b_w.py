"""SC-4b: W1 classes and report, W0's transient-vs-drift mapping, the capture's golden digest.

sc-release-design v4 mechanism.w_abort_predicate (w0, w1, classes,
lists_note) and test_plan (the per-table directions of W_LOSS/W_GRANT, the
ledger prefix, the deleted or unreadable database, mode=rw without creating
a file, the unclassifiable value, objects only in K', the content-free report
with no clock input) and R3-OPS-2 (the selection split out of
capture_state_authority keeps its digest). Every state here is a fixed state
(tests/mocks/sc4b_states.py): K is the golden state, H a second copy of it,
K' a copy of K changed as a NEW (or storage damage) would change it.
"""

from __future__ import annotations

from contextlib import closing
import errno
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest

from scripts import compaction
from scripts import stage_runtime as stage
from tests.mocks import sc4b_states as states

# capture_state_authority(golden_state) before the selection split (f9f1d03, computed on the server).
GOLDEN_DIGEST = "ae3f67afbb57113f019d211ab28b43bc1ef9835a989e9e33206b3c18d23a9df7"

# One row of each W_LOSS/W_GRANT table: (database, table).
LISTED = sorted([("data/privacy.db", name) for name in ("conversation_tombstones", "conversation_erasures",
                                                        "fact_owner_revisions", "voice_attribution_suspensions")]
                + [("data/auth.db", "retired_users"), ("data/ambient.db", "pending_audio_deletions"),
                   ("data/auth.db", "credentials"), ("data/auth.db", "sessions")])


@pytest.fixture
def world(tmp_path):
    """K (authorized projection), H (its source) and K' (a copy of K to change)."""
    k = states.golden_state(tmp_path / "k")
    h = tmp_path / "h"
    shutil.copytree(k, h)
    k_prime = tmp_path / "k-prime"
    shutil.copytree(k, k_prime)
    return {"k": k, "h": h, "k_prime": k_prime, "projection": stage.capture_state_authority(k)["projection"]}


def _w1(world: dict, verify_branch: str = "yes") -> dict:
    return compaction.w1(world["k_prime"], world["h"], world["projection"], verify_branch=verify_branch)


def _class(world: dict) -> str:
    return _w1(world)["report"]["class"]


def _new_row(table: str) -> tuple:
    """A row no fixture holds, with the canary in every text column."""
    return tuple(states.CANARY + "-" + name if kind == "TEXT" else (77 if kind == "INTEGER" else 9.25)
                 for name, kind in states.columns(table))


def _assert_content_free(found: dict) -> None:
    text = json.dumps(found, sort_keys=True)
    assert states.CANARY not in text
    for value in ("Ana", "Beto", "retention", "KZA_EXAMPLE", "conversation_tombstones-conversation_id-0"):
        assert value not in text


# --- R3-OPS-2: the selection split keeps the capture's digest ---------------------------------------

def test_capture_digest_is_unchanged_across_the_selection_split(tmp_path):
    state = states.golden_state(tmp_path / "state")
    authority = stage.capture_state_authority(state)
    assert authority["digest"] == GOLDEN_DIGEST
    projection = authority["projection"]
    for relative, tables in projection["tables"].items():
        with stage._database_image(state / relative) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            selected = stage._authority_selection(db, names)
            assert {key: value for key, value in tables.items() if key != "schema_contract_v1"} == selected
            table_part = stage._authority_table_selection(db, names)
            assert set(table_part) <= set(selected) and stage._LEDGER_TABLE not in table_part
    for relative, users in projection["accounts"].items():
        assert stage._account_projection(state / relative) == users


# --- classes ------------------------------------------------------------------------------------------

def test_an_unchanged_k_prime_is_equal_and_the_report_has_no_clock_input(world):
    first, second = _w1(world), _w1(world)
    assert first == second and first["report"]["class"] == compaction.W_EQUAL
    assert first["sha256"] == stage._digest(first["report"])
    report = first["report"]
    assert set(report["databases"]) == set(states.DATABASES)
    assert all(found == {"status": "ok"} for found in report["databases"].values())
    assert report["ledgers"] == {"data/privacy.db": "equal"} and report["unclassified"] == {}
    assert report["verify_branch_reached"] == "yes"
    assert _w1(world, "no")["sha256"] != first["sha256"]  # the header is part of the digest
    _assert_content_free(first)


@pytest.mark.parametrize("relative, table", LISTED)
@pytest.mark.parametrize("direction", ["removed", "added"])
def test_each_listed_table_in_both_directions(world, relative, table, direction):
    """The 16 per-table cases: a change in the loss direction is loss_only (abort may return to H),
    in the other direction restrictive_or_unclear (safe stop)."""
    path = world["k_prime"] / relative
    if direction == "removed":
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(f"DELETE FROM {table} WHERE rowid = (SELECT MIN(rowid) FROM {table})")
    else:
        with closing(sqlite3.connect(path)) as db, db:
            states.insert(db, table, _new_row(table))
    found = _w1(world)
    loss = (direction == "removed") == (table in compaction.W_LOSS)
    assert table in compaction.W_LOSS | compaction.W_GRANT
    assert found["report"]["class"] == (compaction.W_LOSS_ONLY if loss else compaction.W_RESTRICTIVE)
    assert found["report"]["tables"][relative][table] == {"added": int(direction == "added"),
                                                          "removed": int(direction == "removed")}
    _assert_content_free(found)


def test_the_two_lists_are_the_reviewed_constants():
    assert compaction.W_LOSS == {"conversation_tombstones", "conversation_erasures", "retired_users",
                                 "fact_owner_revisions", "voice_attribution_suspensions", "pending_audio_deletions"}
    assert compaction.W_GRANT == {"credentials", "sessions"}
    assert compaction.W_LOSS <= stage._PROTECTED_TABLES and compaction.W_GRANT <= stage._PROTECTED_TABLES


@pytest.mark.parametrize("change", [
    ("data/privacy.db", "UPDATE conversation_tombstones SET deleted_by='x' WHERE rowid=1"),  # changed W_LOSS row
    ("data/auth.db", "UPDATE credentials SET revision=revision+1 WHERE rowid=1"),  # changed W_GRANT row
    ("data/privacy.db", "DELETE FROM voice_attribution_consents WHERE rowid=1"),  # neither list
    ("data/privacy.db", "UPDATE deletion_ledger_salt_v1 SET salt='" + "6b" * 32 + "'"),  # the salt: neither list
    ("data/ambient.db", "DELETE FROM audio_write_intents WHERE rowid=1"),
    ("data/privacy.db", "UPDATE facts SET state='deleted' WHERE state='confirmed'"),  # a filtered selection
])
def test_any_other_difference_is_restrictive(world, change):
    relative, statement = change
    states.execute(world["k_prime"] / relative, (statement, ()))
    assert _class(world) == compaction.W_RESTRICTIVE


def test_ledger_strict_prefix_is_loss_and_an_entry_h_lacks_is_restrictive(world):
    path = world["k_prime"] / "data" / "privacy.db"
    states.execute(path, ("DELETE FROM deletion_ledger_v1 WHERE seq=3", ()))
    found = _w1(world)
    assert found["report"]["ledgers"] == {"data/privacy.db": "strict_prefix"}
    assert found["report"]["class"] == compaction.W_LOSS_ONLY  # NEW truncated the ledger only
    # One entry more than H: an erasure H lacks.
    extra = states.ledger_rows(states.LEDGER + [("fact_deleted", "4d" * 32, states.NOW + 3.0)])[-1]
    states.execute(path, *[("INSERT INTO deletion_ledger_v1 VALUES (?,?,?,?,?,?)", row)
                           for row in states.ledger_rows(states.LEDGER)[2:]] + [
        ("INSERT INTO deletion_ledger_v1 VALUES (?,?,?,?,?,?)", extra)])
    found = _w1(world)
    assert found["report"]["ledgers"] == {"data/privacy.db": "other"}
    assert found["report"]["class"] == compaction.W_RESTRICTIVE


@pytest.mark.parametrize("damage", ["fork", "broken_chain", "dropped"])
def test_a_ledger_that_is_not_a_prefix_of_h_is_restrictive(world, damage):
    path = world["k_prime"] / "data" / "privacy.db"
    if damage == "fork":  # a two-entry chain that H does not have
        fork = states.ledger_rows([("user_retired", "9e" * 32, states.NOW)])
        states.execute(path, ("DELETE FROM deletion_ledger_v1", ()),
                       *[("INSERT INTO deletion_ledger_v1 VALUES (?,?,?,?,?,?)", row) for row in fork])
    elif damage == "broken_chain":
        states.execute(path, ("UPDATE deletion_ledger_v1 SET at=at+1 WHERE seq=2", ()))
    else:  # R3: a missing ledger is never the loss case
        states.execute(path, ("DROP TABLE deletion_ledger_v1", ()))
    found = _w1(world)
    assert found["report"]["ledgers"]["data/privacy.db"] == "other"
    assert found["report"]["class"] == compaction.W_RESTRICTIVE


@pytest.mark.parametrize("relative", ["data/privacy.db", "data/auth.db"])
def test_a_deleted_database_is_restrictive_and_w1_never_creates_it(world, relative):
    """R3-PRIV-3 (memory/privacy: ledger, salt, voice tables, fact owners, consents) and the authority
    database (credentials): a safe stop, never loss_only; R3-OPS-7: W1 does not create the file."""
    path = world["k_prime"] / relative
    path.unlink()
    found = _w1(world)
    assert found["report"]["databases"][relative] == {"status": "missing"}
    assert found["report"]["class"] == compaction.W_RESTRICTIVE
    assert not os.path.lexists(path)
    assert all(change["added"] == 0 for change in found["report"]["tables"][relative].values())


@pytest.mark.parametrize("damage", ["missing", "unreadable"])
def test_a_missing_database_holding_only_loss_tables_is_still_restrictive(tmp_path, damage):
    """SC4B-N3 (stricter): every row of it absent would read as loss_only; a missing or unreadable
    database never is (review RA4B-7 R13: 'unreadable' is not 'ok')."""
    k = tmp_path / "k"
    (k / "data").mkdir(parents=True)
    with closing(sqlite3.connect(k / "data" / "tombstones.db")) as db, db:
        db.execute(states.ddl("conversation_tombstones"))
        states.insert(db, "conversation_tombstones", _new_row("conversation_tombstones"))
    projection = stage.capture_state_authority(k)["projection"]
    assert set(projection["tables"]) == {"data/tombstones.db"}
    k_prime = tmp_path / "k-prime"
    shutil.copytree(k, k_prime)
    if damage == "missing":
        (k_prime / "data" / "tombstones.db").unlink()
    else:
        (k_prime / "data" / "tombstones.db").write_bytes(b"not a database" * 100)
    found = compaction.w1(k_prime, k, projection, verify_branch="yes")["report"]
    assert found["databases"]["data/tombstones.db"]["status"] == damage
    assert found["tables"]["data/tombstones.db"]["conversation_tombstones"] == {"added": 0, "removed": 1}
    assert found["class"] == compaction.W_RESTRICTIVE


def test_an_unreadable_database_is_shown_and_restrictive(world):
    """R4-OPS-9: storage damage after a crash, told apart from a write by the per-database status."""
    path = world["k_prime"] / "data" / "privacy.db"
    path.write_bytes(b"not a database" * 100)
    found = _w1(world)
    assert found["report"]["databases"]["data/privacy.db"]["status"] == "unreadable"
    assert found["report"]["databases"]["data/privacy.db"]["error"].startswith("sqlite:")
    assert found["report"]["class"] == compaction.W_RESTRICTIVE
    path.unlink()
    path.symlink_to(world["k"] / "data" / "privacy.db")  # a redirected database is not read
    assert _w1(world)["report"]["databases"]["data/privacy.db"]["status"] == "unreadable"


def test_an_unclassifiable_value_counts_as_a_changed_row(world):
    states.execute(world["k_prime"] / "data" / "privacy.db",
                   ("UPDATE wake_clips_v1 SET sources_json='[]' WHERE rowid=1", ()))
    found = _w1(world)
    assert "wake_deadline_fences" in found["report"]["unclassified"]["data/privacy.db"]
    assert found["report"]["class"] == compaction.W_RESTRICTIVE
    assert found["report"]["tables"]["data/privacy.db"]["conversation_tombstones"] == {"added": 0, "removed": 0}


# Review RA4B-1: values a defective NEW (or an isolation breach) writes during V that the selection cannot
# classify by TypeError or AttributeError, not only ValueError (probe2 of the review).
UNCLASSIFIABLE = {
    "wake_clip_sources_null": ("data/privacy.db", "UPDATE wake_clips_v1 SET sources_json=NULL WHERE rowid=1",
                               "wake_deadline_fences"),
    "wake_clip_expires_text": ("data/privacy.db",
                               "UPDATE wake_clips_v1 SET expires_at='2026-01-01' WHERE rowid=1",
                               "wake_deadline_fences"),
    "voice_fingerprint_expires_text": ("data/privacy.db",
                                       "UPDATE voice_fingerprints SET expires_at='soon' WHERE rowid=1",
                                       "voice_deadline_fences"),
    "fact_source_expires_text": ("data/privacy.db", "UPDATE fact_sources SET expires_at='soon'",
                                 "fact_deadline_fences"),
    "suggestion_sources_number": ("data/suggestions.db",
                                  "UPDATE suggestions_policy_v1 SET sources_json=7 WHERE rowid=1",
                                  "suggestion_deadline_fences"),
}


@pytest.mark.parametrize("case", sorted(UNCLASSIFIABLE))
def test_a_value_of_another_type_is_unclassified_never_an_uncoded_error(world, case):
    relative, statement, key = UNCLASSIFIABLE[case]
    states.execute(world["k_prime"] / relative, (statement, ()))
    found = _w1(world)
    assert key in found["report"]["unclassified"][relative]
    assert found["report"]["databases"][relative] == {"status": "ok"}
    assert found["report"]["class"] == compaction.W_RESTRICTIVE
    _assert_content_free(found)


def test_an_account_user_that_is_not_an_object_is_unclassifiable(world):
    """Review RA4B-1: users [1] raised AttributeError out of W1 before."""
    accounts = world["k_prime"] / "data" / "home_accounts_v1.json"
    accounts.write_text(json.dumps({"schema": "kza-users-accounts-v1", "users": [1]}))
    found = _w1(world)
    assert found["report"]["accounts"]["data/home_accounts_v1.json"] == {"status": "unclassifiable", "added": 0,
                                                                         "removed": 2}
    assert found["report"]["class"] == compaction.W_RESTRICTIVE


def test_any_other_failure_reading_a_k_prime_database_is_unreadable_with_a_code(world, monkeypatch):
    """Review RA4B-1: W1 never raises an uncoded error; the code names only the exception type."""
    def broken(db, names):
        raise RuntimeError(states.CANARY)

    monkeypatch.setattr(stage, "_authority_table_selection", broken)
    found = _w1(world)
    assert found["report"]["databases"]["data/privacy.db"] == {"status": "unreadable",
                                                               "error": "python:RuntimeError"}
    assert found["report"]["class"] == compaction.W_RESTRICTIVE
    _assert_content_free(found)


def _wal_pending(path: Path) -> None:
    """Leave committed frames in ``path``'s WAL that a read-write open would checkpoint."""
    db = sqlite3.connect(path, isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE wal_scratch (x INTEGER)")
    db.execute("INSERT INTO wal_scratch VALUES (1)")
    db.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)
    db.close()
    wal = path.with_name(path.name + "-wal")
    assert wal.exists() and wal.stat().st_size > 0


def _tree(path: Path) -> dict:
    return {item.relative_to(path).as_posix(): (item.read_bytes(), item.stat().st_mtime_ns)
            for item in sorted(path.rglob("*")) if item.is_file()}


@pytest.mark.parametrize("link", ["directory", "file"])
def test_w1_never_opens_a_k_prime_entry_through_a_link(world, link):
    """Review RA4B-4: K'/data (or one database) a link into H: W1 writes nothing outside K' (H's pending
    WAL is not checkpointed, no mtime moves) and reports the entries as not read: restrictive."""
    h_data, k_data = world["h"] / "data", world["k_prime"] / "data"
    _wal_pending(h_data / "privacy.db")
    before = _tree(world["h"])
    if link == "directory":
        shutil.rmtree(k_data)
        k_data.symlink_to(h_data)
        shutil.rmtree(world["k_prime"] / "config")
        (world["k_prime"] / "config").symlink_to(world["h"] / "config")
    else:
        (k_data / "privacy.db").unlink()
        (k_data / "privacy.db").symlink_to(h_data / "privacy.db")
    found = _w1(world)
    assert _tree(world["h"]) == before
    report = found["report"]
    assert report["class"] == compaction.W_RESTRICTIVE
    redirected = {"status": "unreadable", "error": "runtime_admission_redirected_database"}
    assert report["databases"]["data/privacy.db"] == redirected
    if link == "directory":
        assert all(found == redirected for found in report["databases"].values())
        assert report["accounts"]["data/home_accounts_v1.json"]["status"] == "unclassifiable"
        assert report["configuration"]["config/policy.yaml"] == "not_regular"
    else:
        assert report["databases"]["data/auth.db"] == {"status": "ok"}


def test_a_k_prime_database_a_live_holder_locks_is_a_transient_refusal(world):
    """Review RA4B-5/RB-7: an operator's sqlite3 session holding a lock on K' is not storage damage: a
    transient refusal (nothing written, content-free), and once released the same K' is 'equal'."""
    holder = sqlite3.connect(world["k_prime"] / "data" / "auth.db", isolation_level=None)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        with pytest.raises(compaction.CompactionRefusal) as caught:
            _w1(world)
    finally:
        holder.close()
    assert str(caught.value) == compaction.K_TRANSIENT
    assert caught.value.report == {"w1": "transient", "state": "k_prime", "database": "data/auth.db",
                                   "sqlite": "SQLITE_BUSY"}
    assert _class(world) == compaction.W_EQUAL


@pytest.mark.parametrize("number, transient", [(errno.EMFILE, True), (errno.ENOMEM, True), (errno.EIO, False),
                                               (errno.ENOENT, False)])
def test_a_k_prime_errno_is_transient_only_for_a_resource_that_clears(world, monkeypatch, number, transient):
    """SC4B-N19: EIO on K' persists with the damage, so it stays 'unreadable' (SS and its exit), never a
    transient refusal that would hold the only exit shut."""
    def failing(path):
        raise OSError(number, "x")

    monkeypatch.setattr(compaction, "_w1_recover", failing)
    if transient:
        with pytest.raises(compaction.CompactionRefusal) as caught:
            _w1(world)
        assert str(caught.value) == compaction.K_TRANSIENT
        assert caught.value.report["errno"] == errno.errorcode[number]
    else:
        found = _w1(world)
        assert found["report"]["databases"]["data/privacy.db"] == {"status": "unreadable",
                                                                   "error": "errno:" + errno.errorcode[number]}
        assert found["report"]["class"] == compaction.W_RESTRICTIVE


def test_k_transient_lists_are_the_w0_lists_without_persistent_faults():
    assert compaction.K_TRANSIENT_ERRNOS == compaction.W0_TRANSIENT_ERRNOS - {errno.EIO}
    assert compaction.K_TRANSIENT_SQLITE == {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_NOMEM,
                                             sqlite3.SQLITE_INTERRUPT}


def test_unverifiable_k_prime_locks_are_restrictive(world):
    """Review RB-1: W0 could not open a K' lock entry; W1 is restrictive_or_unclear (SS) and says why."""
    found = compaction.w1(world["k_prime"], world["h"], world["projection"], verify_branch="yes",
                          k_writers={"status": "unverifiable", "error": "errno:ELOOP"})
    assert found["report"]["class"] == compaction.W_RESTRICTIVE
    assert found["report"]["k_writers"] == {"status": "unverifiable", "error": "errno:ELOOP"}
    assert compaction._w_counts(found["report"])["k_writers"] == {"status": "unverifiable", "error": "errno:ELOOP"}
    assert _w1(world)["report"]["k_writers"] == {"status": "closed"}


def test_objects_only_in_k_prime_are_ignored(world):
    """PA-2: NEW creating an unknown table and database during V; they are destroyed with K'."""
    states.execute(world["k_prime"] / "data" / "privacy.db", ("CREATE TABLE new_thing (a TEXT)", ()),
                   ("INSERT INTO new_thing VALUES ('x')", ()))
    with closing(sqlite3.connect(world["k_prime"] / "data" / "new.db")) as db, db:
        db.execute(states.ddl("conversation_tombstones"))
        states.insert(db, "conversation_tombstones", _new_row("conversation_tombstones"))
    assert _class(world) == compaction.W_EQUAL


@pytest.mark.parametrize("change", ["account_added", "account_file_deleted", "configuration_changed", "env_deleted",
                                    "account_unreadable"])
def test_accounts_and_configuration_differences_are_restrictive(world, change):
    k_prime = world["k_prime"]
    accounts = k_prime / "data" / "home_accounts_v1.json"
    if change == "account_added":
        value = json.loads(accounts.read_text())
        value["users"].append({"id": "u3", "name": states.CANARY, "role": "home"})
        accounts.write_text(json.dumps(value))
    elif change == "account_file_deleted":
        accounts.unlink()
    elif change == "account_unreadable":
        accounts.write_text("{}")
    elif change == "configuration_changed":
        (k_prime / "config" / "policy.yaml").write_text("retention: " + states.CANARY + "\n")
    else:
        (k_prime / ".env").unlink()
    found = _w1(world)
    assert found["report"]["class"] == compaction.W_RESTRICTIVE
    _assert_content_free(found)


def test_a_last_seen_change_is_not_a_difference(world):
    accounts = world["k_prime"] / "data" / "home_accounts_v1.json"
    value = json.loads(accounts.read_text())
    value["users"][0]["last_seen"] = 99.0
    accounts.write_text(json.dumps(value))
    assert _class(world) == compaction.W_EQUAL


_KILL_IN_TRANSACTION = """
import os, sqlite3, sys
db = sqlite3.connect(sys.argv[1], isolation_level=None)
db.execute("PRAGMA journal_mode=DELETE")
db.execute("PRAGMA cache_size=1")
db.execute("BEGIN IMMEDIATE")
for index in range(400):
    db.execute("INSERT INTO conversation_tombstones VALUES (?, 1.0, ?)", ("kill-%d" % index, "x" * 200))
os._exit(9)
"""


def test_a_kill_inside_a_write_transaction_is_recovered_in_place_and_equal(world, tmp_path):
    """CR-3: V killed inside a write transaction leaves a hot journal; W1's mode=rw open rolls it
    back in place (as R-b's quick check would) and K' equals K."""
    path = world["k_prime"] / "data" / "privacy.db"
    script = tmp_path / "kill.py"
    script.write_text(_KILL_IN_TRANSACTION)
    assert subprocess.run([sys.executable, "-I", str(script), str(path)], timeout=60).returncode == 9
    journal = path.with_name(path.name + "-journal")
    assert journal.exists() and journal.stat().st_size > 0
    found = _w1(world)
    assert found["report"]["class"] == compaction.W_EQUAL
    assert found["report"]["databases"]["data/privacy.db"] == {"status": "ok"}
    assert not journal.exists() or journal.stat().st_size == 0


def test_w_counts_keep_only_the_differences(world):
    states.execute(world["k_prime"] / "data" / "auth.db",
                   ("INSERT INTO credentials VALUES (?,?,?,?,?,?,?,?,?)", _new_row("credentials")))
    report = _w1(world)["report"]
    counts = compaction._w_counts(report)
    assert counts["tables"] == {"data/auth.db": {"credentials": {"added": 1, "removed": 0}}}
    assert counts["databases"] == {} and counts["ledgers"] == {} and counts["accounts"] == {}


# --- W0: transient vs drift (R3-OPS-5, R4-PRIV-1, R4-OPS-3) -----------------------------------------

def _sqlite_error(tmp_path: Path, kind: str) -> sqlite3.Error:
    path = tmp_path / (kind + ".db")
    try:
        if kind == "busy":
            sqlite3.connect(path).close()
            holder = sqlite3.connect(path, isolation_level=None)
            holder.execute("BEGIN EXCLUSIVE")
            try:
                sqlite3.connect(path, timeout=0).execute("SELECT * FROM sqlite_master").fetchall()
            finally:
                holder.close()
        elif kind == "notadb":
            path.write_bytes(b"garbage" * 1000)
            sqlite3.connect(path).execute("SELECT * FROM sqlite_master").fetchall()
        else:
            sqlite3.connect(path).execute("SELECT * FROM missing_table").fetchall()
    except sqlite3.Error as error:
        return error
    raise AssertionError("no error")


@pytest.mark.parametrize("error, code, transient", [
    (ValueError("runtime_admission_writer_active"), "runtime_admission_writer_active", True),
    (ValueError("runtime_admission_busy"), "runtime_admission_busy", True),
    (ValueError("runtime_admission_sdk_version_unavailable"), "runtime_admission_sdk_version_unavailable", True),
    (OSError(errno.EMFILE, "x"), compaction.H_TRANSIENT, True),
    (OSError(errno.EIO, "x"), compaction.H_TRANSIENT, True),
    (OSError(errno.EBUSY, "x"), compaction.H_TRANSIENT, True),
    (FileNotFoundError(errno.ENOENT, "x"), compaction.H_CHANGED, False),
    (OSError(errno.ELOOP, "x"), compaction.H_CHANGED, False),
    (OSError(errno.ENOTDIR, "x"), compaction.H_CHANGED, False),
    (ValueError("privacy_authority_stale"), compaction.H_CHANGED, False),
    (ValueError("runtime_admission_source_close_proof"), compaction.H_CHANGED, False),
    (ValueError("runtime_admission_marker_mismatch"), compaction.H_CHANGED, False),
    (KeyError("binding"), compaction.H_CHANGED, False),
])
def test_w0_maps_each_failure_reading_h(error, code, transient):
    refusal = compaction._w0_refusal(error)
    assert str(refusal) == code
    assert refusal.report["w0"] == ("transient" if transient else "drift")


@pytest.mark.parametrize("kind, code, name", [("busy", compaction.H_TRANSIENT, "SQLITE_BUSY"),
                                              ("notadb", compaction.H_CHANGED, "SQLITE_NOTADB"),
                                              ("no_such_table", compaction.H_CHANGED, "SQLITE_ERROR")])
def test_w0_classifies_sqlite_errors_by_code_not_by_class(tmp_path, kind, code, name):
    """'no such table' is an OperationalError like 'database is locked': only the code tells them apart."""
    error = _sqlite_error(tmp_path, kind)
    refusal = compaction._w0_refusal(error)
    assert str(refusal) == code
    assert name in json.dumps(refusal.report)


def test_w0_transient_lists_are_the_design_lists():
    assert compaction.W0_TRANSIENT_ERRNOS == {errno.EMFILE, errno.ENFILE, errno.EIO, errno.EAGAIN, errno.EINTR,
                                              errno.ENOMEM, errno.EBUSY}
    assert compaction.W0_TRANSIENT_SQLITE == {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR,
                                              sqlite3.SQLITE_NOMEM, sqlite3.SQLITE_INTERRUPT}
    assert compaction.W0_TRANSIENT_CODES == {"runtime_admission_writer_active", "runtime_admission_busy",
                                             "runtime_admission_sdk_version_unavailable"}


def test_the_cli_passes_discard_k_changes_and_prints_the_w_report(monkeypatch, capsys):
    """`compact abort <id> --to-h --confirm-h <prefix> --discard-k-changes <digest prefix>` reaches
    compact_abort; a safe-stop refusal prints its code, the content-free report and its digest (78)."""
    import argparse
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    compaction.add_commands(commands)
    args = parser.parse_args(["compact", "abort", "0" * 32, "--to-h", "--confirm-h", "a" * 12,
                              "--discard-k-changes", "b" * 12])
    args.root = Path("/nonexistent")
    seen = {}

    def abort(root, compaction_id, **options):
        seen.update(options)
        raise compaction.CompactionRefusal(compaction.ABORT_PROTECTED_CHANGED,
                                           {"w": {"class": compaction.W_RESTRICTIVE}, "w_sha256": "c" * 64})

    monkeypatch.setattr(compaction, "compact_abort", abort)
    assert compaction.run_command(args) == 78
    assert seen["discard_k_changes"] == "b" * 12 and seen["to_h"] is True and seen["confirm_h"] == "a" * 12
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"refused": compaction.ABORT_PROTECTED_CHANGED,
                       "report": {"w": {"class": compaction.W_RESTRICTIVE}, "w_sha256": "c" * 64}}
