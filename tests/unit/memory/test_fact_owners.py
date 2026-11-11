"""U3b-1a ledger core: owner schema, confirm-on-assign, 'No es mío', scopes and pins.

Every case runs on a real FactLedger file. Refusals compare the database file
bytes before and after, so 'nothing changed' is literal.
"""

import ast
from dataclasses import replace
from pathlib import Path
import sqlite3
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.memory.fact_ledger import (
    FACT_OWNER_SCHEMA, AssignmentReceipt, FactDependency, FactLedger, FactOwnerChange, FactOwnerMethod,
    MemoryAssigner, MemoryOrigin, MemoryOwnerActor, MemoryOwnerDependents, MemoryOwnerReader, MemoryReviewer,
    MemoryScope,
    MemorySource, OwnedCounts,
)
from src.memory.long_term import LongTermMemory

ROOT = Path(__file__).resolve().parents[3]
NOW = 2_000_000_000.0
OPERATOR = MemoryAssigner("synthetic-operator")
REVIEWER = MemoryReviewer("synthetic-reviewer")
ALLOW = {"retired": lambda ids: False, "active": lambda person: True}
PERSON, OTHER = "user_" + "a" * 32, "user_" + "b" * 32


class Clock:
    def __init__(self):
        self.value = NOW

    def __call__(self):
        return self.value


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def ledger(tmp_path, clock):
    value = FactLedger(str(tmp_path / "privacy.db"), clock=clock)
    value.initialize()
    try:
        yield value
    finally:
        value.close()


def _source(clock, *, conversation=None, dependencies=()):
    return MemorySource(str(uuid4()), conversation or str(uuid4()), clock() - 1, clock() + 3600,
                        MemoryOrigin.VOICE, tuple(dependencies))


def _automatic(ledger, clock, content="Synthetic automatic fact", **kwargs):
    return ledger.store(content, "fact", .9, (_source(clock, **kwargs),))


def _proposed(ledger, clock, content="Synthetic proposal", **kwargs):
    return ledger.store(content, "fact", 1.0, (_source(clock, **kwargs),), proposed=True)


def _confirmed(ledger, clock, content="Synthetic confirmed fact", **kwargs):
    return ledger.confirm(_proposed(ledger, clock, content, **kwargs).id, REVIEWER)


def _derived(ledger, clock, parent, content="Synthetic derived fact"):
    dependency = FactDependency(parent.id, parent.revision, parent.expires_at)
    return ledger.store(content, "fact", .9, (_source(clock, dependencies=(dependency,)),))


def _bytes(ledger):
    return Path(ledger.path).read_bytes()


def _row(ledger, table, fact_id):
    row = ledger._db.execute("SELECT * FROM " + table + " WHERE " + ("id" if table == "facts" else "fact_id")
                             + "=?", (fact_id,)).fetchone()
    return None if row is None else dict(row)


def _assign(ledger, fact, person=PERSON, owner_revision=0, actor=OPERATOR, **kwargs):
    return ledger.assign_owner(fact.id, person, fact.revision, owner_revision, actor, **{**ALLOW, **kwargs})


def _refused(ledger, error, match, action):
    before = _bytes(ledger)
    with pytest.raises(error, match=match):
        action()
    assert _bytes(ledger) == before


# --- schema --------------------------------------------------------------------------


def _owner_objects(path):
    with sqlite3.connect(path) as db:
        return sorted(db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master "
                                 "WHERE tbl_name IN ('fact_owners','fact_owner_revisions','fact_owner_schema')"))


def test_fresh_ledger_has_the_exact_owner_schema_and_marker(ledger):
    assert [tuple(r) for r in ledger._db.execute("SELECT version FROM fact_owner_schema")] == [(1,)]
    columns = {name: [(r[1], r[2], r[3], r[5]) for r in ledger._db.execute("PRAGMA table_info(" + name + ")")]
               for name in ("fact_owners", "fact_owner_revisions")}
    assert columns["fact_owners"][0] == ("fact_id", "TEXT", 1, 1)  # NOT NULL primary key (1a-F7)
    assert columns["fact_owner_revisions"][0] == ("fact_id", "TEXT", 1, 1)
    with pytest.raises(sqlite3.IntegrityError):
        ledger._db.execute("INSERT INTO fact_owner_revisions VALUES (NULL,1,'assigned',0,1)")


def test_fresh_and_migrated_ledgers_have_identical_owner_objects(tmp_path, clock):
    fresh = FactLedger(str(tmp_path / "fresh.db"), clock=clock)
    fresh.initialize()
    fresh.close()
    migrated = FactLedger(str(tmp_path / "migrated.db"), clock=clock)
    migrated.initialize()
    _automatic(migrated, clock)
    migrated.close()
    with sqlite3.connect(tmp_path / "migrated.db") as db:
        db.executescript("DROP TABLE fact_owners; DROP TABLE fact_owner_revisions; DROP TABLE fact_owner_schema;")
    migrated.initialize()
    migrated.close()
    assert _owner_objects(tmp_path / "fresh.db") == _owner_objects(tmp_path / "migrated.db")
    script = sqlite3.connect(":memory:")
    script.executescript("CREATE TABLE facts (id TEXT PRIMARY KEY);" + FACT_OWNER_SCHEMA)
    assert sorted(script.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name!='facts' "
                                 "AND tbl_name!='facts'")) == _owner_objects(tmp_path / "fresh.db")


def test_populated_pre_u3b_ledger_initialises_with_no_owner_rows_or_index_jobs(tmp_path, clock):
    path = tmp_path / "privacy.db"
    ledger = FactLedger(str(path), clock=clock)
    ledger.initialize()
    _automatic(ledger, clock)
    _confirmed(ledger, clock)
    _proposed(ledger, clock, "Another proposal")
    ledger.close()
    with sqlite3.connect(path) as db:
        db.executescript("DROP TABLE fact_owners; DROP TABLE fact_owner_revisions; DROP TABLE fact_owner_schema;")
        jobs = sorted(db.execute("SELECT * FROM index_jobs"))
        deletions = sorted(db.execute("SELECT * FROM policy_index_deletions"))
        facts = sorted(db.execute("SELECT * FROM facts"))
    ledger.initialize()
    try:
        assert ledger._db.execute("SELECT COUNT(*) FROM fact_owners").fetchone()[0] == 0
        assert ledger._db.execute("SELECT COUNT(*) FROM fact_owner_revisions").fetchone()[0] == 0
        assert sorted(map(tuple, ledger._db.execute("SELECT * FROM index_jobs"))) == jobs
        assert sorted(map(tuple, ledger._db.execute("SELECT * FROM policy_index_deletions"))) == deletions
        assert sorted(map(tuple, ledger._db.execute("SELECT * FROM facts"))) == facts
    finally:
        ledger.close()


_RECREATE_NULLABLE = ("DROP TABLE fact_owners; CREATE TABLE fact_owners (fact_id TEXT PRIMARY KEY REFERENCES "
                      "facts(id), person_id TEXT NOT NULL, method TEXT NOT NULL, revision INTEGER NOT NULL, "
                      "assigned_at REAL NOT NULL, assigned_by TEXT NOT NULL, evidence TEXT NOT NULL DEFAULT '{}');"
                      "CREATE INDEX fact_owners_person ON fact_owners(person_id);")


def _recreate(table, old, new):
    """Recreate one owner table from FACT_OWNER_SCHEMA with one textual change (plus its index)."""
    statements = [part.strip() for part in FACT_OWNER_SCHEMA.split(";") if part.strip()]
    create = next(item for item in statements if item.startswith("CREATE TABLE " + table + " "))
    assert create.count(old) == 1
    index = [item for item in statements if item.startswith("CREATE INDEX") and " ON " + table + "(" in item]
    return ";".join(["DROP TABLE " + table, create.replace(old, new), *index]) + ";"


_FK = "fact_id TEXT NOT NULL PRIMARY KEY REFERENCES facts(id)"
_LAST = "evidence TEXT NOT NULL DEFAULT '{}')"


@pytest.mark.parametrize("damage", [
    # R3: foreign key missing, retargeted or with an action; columns/indexes otherwise exact.
    pytest.param(_recreate("fact_owners", _FK, "fact_id TEXT NOT NULL PRIMARY KEY"), id='fk_missing_owners'),
    pytest.param(_recreate("fact_owner_revisions", _FK, "fact_id TEXT NOT NULL PRIMARY KEY"), id='fk_missing_revisions'),
    pytest.param(_recreate("fact_owners", _FK, "fact_id TEXT NOT NULL PRIMARY KEY REFERENCES conversation_tombstones(conversation_id)"), id='fk_retargeted_owners'),
    pytest.param(_recreate("fact_owner_revisions", _FK, _FK + " ON DELETE CASCADE"), id='fk_on_delete_cascade_revisions'),
    # R2: shapes PRAGMA table_info/index_list cannot see; refused by the stored-DDL comparison.
    pytest.param(_recreate("fact_owners", "person_id TEXT NOT NULL,", "person_id TEXT NOT NULL COLLATE NOCASE,"), id='ddl_collate_nocase'),
    pytest.param(_recreate("fact_owners", "person_id TEXT NOT NULL,", "person_id TEXT NOT NULL CHECK (length(person_id) < 3),"), id='ddl_check_person'),
    pytest.param(_recreate("fact_owner_revisions", "last_change TEXT NOT NULL,",
              "last_change TEXT NOT NULL CHECK (last_change != 'owner_returned'),"), id='ddl_check_last_change'),
    pytest.param(_recreate("fact_owners", _LAST, _LAST + " STRICT"), id='ddl_strict'),
    pytest.param(_recreate("fact_owners", _LAST, "evidence TEXT NOT NULL DEFAULT '{}', shadow TEXT GENERATED ALWAYS AS (person_id) VIRTUAL)"), id='ddl_generated_column'),
    pytest.param(_recreate("fact_owners", _LAST, _LAST + " WITHOUT ROWID"), id='ddl_without_rowid'),
    pytest.param(_recreate("fact_owners", "assigned_by TEXT NOT NULL,", "assigned_by TEXT NOT NULL UNIQUE,"), id='ddl_unique'),
    pytest.param(_recreate("fact_owners", "fact_owners (", "fact_owners  ("), id='ddl_text_drift'),  # any DDL text drift
    "DROP TABLE fact_owners",
    "DROP TABLE fact_owner_revisions",
    "DROP TABLE fact_owner_schema",
    "DROP INDEX fact_owners_person",
    "DROP INDEX fact_owner_revisions_change",
    "CREATE INDEX fact_owners_extra ON fact_owners(method)",
    "CREATE INDEX fact_owner_revisions_extra ON fact_owner_revisions(changed_at)",
    "DROP INDEX fact_owners_person; CREATE INDEX fact_owners_person ON fact_owners(method)",
    "UPDATE fact_owner_schema SET version=2",
    "INSERT INTO fact_owner_schema VALUES (2)",
    "DELETE FROM fact_owner_schema",
    "ALTER TABLE fact_owners ADD COLUMN extra TEXT",
    _RECREATE_NULLABLE,  # the v5 DDL without NOT NULL on fact_id (1a-F7)
    "CREATE TRIGGER fact_owners_audit AFTER INSERT ON fact_owners BEGIN SELECT 1; END",
    "DROP TABLE fact_owner_schema; CREATE VIEW fact_owner_schema AS SELECT 1 AS version",
])
def test_partial_owner_shapes_fail_closed_with_the_file_unchanged(tmp_path, clock, damage):
    path = tmp_path / "privacy.db"
    ledger = FactLedger(str(path), clock=clock)
    ledger.initialize()
    _automatic(ledger, clock)
    ledger.close()
    with sqlite3.connect(path) as db:
        db.executescript(damage)
    before = path.read_bytes()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="fact_owner_schema_unavailable"):
            ledger.initialize()
        assert ledger._db is None and not ledger.authority_available
        assert path.read_bytes() == before


def _fact_ids(ledger, clock):
    return _confirmed(ledger, clock).id, _confirmed(ledger, clock, "Second confirmed").id


@pytest.mark.parametrize("rows,code", [
    ([("o", "{f}", PERSON, "admin_magic", 1), ("r", "{f}", 1, "assigned", 1)], "fact_owner_value_unknown"),
    ([("o", "{f}", PERSON, "operator_assignment", 1), ("r", "{f}", 1, "stolen", 1)], "fact_owner_value_unknown"),
    ([("r", "{f}", 1, "assigned", 2)], "fact_owner_value_unknown"),
    ([("r", "{f}", 1, "assigned", "x")], "fact_owner_value_unknown"),
    ([("r", "{f}", 0, "assigned", 1)], "fact_owner_value_unknown"),
    ([("r", "{f}", 1.5, "assigned", 1)], "fact_owner_value_unknown"),
    ([("o", "{f}", "", "operator_assignment", 1), ("r", "{f}", 1, "assigned", 1)], "fact_owner_value_unknown"),
    ([("o", "{f}", PERSON, "operator_assignment", 2), ("r", "{f}", 1, "assigned", 1)],
     "fact_owner_revision_mismatch"),
    ([("o", "{f}", PERSON, "operator_assignment", 1)], "fact_owner_revision_mismatch"),
    ([("o", "{f}", PERSON, "operator_assignment", 1, "not json"), ("r", "{f}", 1, "assigned", 1)],
     "fact_owner_value_unknown"),
])
def test_unknown_owner_values_fail_closed_at_open_and_on_every_owner_read(tmp_path, clock, rows, code):
    path = tmp_path / "privacy.db"
    ledger = FactLedger(str(path), clock=clock)
    ledger.initialize()
    target, _ = _fact_ids(ledger, clock)
    for row in rows:
        kind, values = row[0], tuple(target if v == "{f}" else v for v in row[1:])
        if kind == "o":
            evidence = values[4] if len(values) > 4 else "{}"
            ledger._db.execute("INSERT INTO fact_owners VALUES (?,?,?,?,?,?,?)",
                               (values[0], values[1], values[2], values[3], NOW, "sql", evidence))
        else:
            ledger._db.execute("INSERT INTO fact_owner_revisions VALUES (?,?,?,?,?)",
                               (values[0], values[1], values[2], values[3], NOW))
    fact = ledger.get(target)
    # Row-materialising methods refuse on the open ledger, nothing written.
    _refused(ledger, RuntimeError, code, lambda: _assign(ledger, fact, OTHER, owner_revision=1))
    _refused(ledger, RuntimeError, code, lambda: ledger.unassign_owner(target, 1, OPERATOR))
    _refused(ledger, RuntimeError, code, lambda: ledger.owned_counts(OPERATOR))
    _refused(ledger, RuntimeError, code, lambda: ledger.eligible_ids(scope="household"))
    _refused(ledger, RuntimeError, code, lambda: ledger.list(scope="person", person_id=PERSON))
    if any(row[0] == "o" for row in rows):
        _refused(ledger, RuntimeError, code, lambda: ledger.return_owned(target, PERSON,
                                                                            MemoryOwnerActor("s", PERSON)))
    ledger.close()
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match=code):
        ledger.initialize()
    assert path.read_bytes() == before and ledger._db is None


# --- assign_owner outcomes -------------------------------------------------------------


def test_automatic_target_is_confirmed_and_owned(ledger, clock):
    fact = _automatic(ledger, clock)
    receipt = _assign(ledger, fact)
    assert receipt == AssignmentReceipt(fact.revision + 1, 1, 0)
    row = _row(ledger, "facts", fact.id)
    assert (row["state"], row["expires_at"], row["revision"], row["confirmed_by"]) == (
        "confirmed", None, fact.revision + 1, OPERATOR.subject)
    assert _row(ledger, "index_jobs", fact.id)["revision"] == fact.revision + 1
    assert _row(ledger, "index_jobs", fact.id)["operation"] == "upsert"
    owner = _row(ledger, "fact_owners", fact.id)
    assert (owner["person_id"], owner["method"], owner["revision"], owner["assigned_by"], owner["evidence"]) == (
        PERSON, "operator_assignment", 1, OPERATOR.subject, "{}")
    record = _row(ledger, "fact_owner_revisions", fact.id)
    assert (record["revision"], record["last_change"], record["confirmed_by_assignment"]) == (1, "assigned", 1)
    assert ledger.facts_available((FactDependency(fact.id, fact.revision + 1, None),))


def test_proposed_target_is_confirmed_and_owned(ledger, clock):
    fact = _proposed(ledger, clock)
    assert _assign(ledger, fact) == AssignmentReceipt(fact.revision + 1, 1, 0)
    assert _row(ledger, "facts", fact.id)["state"] == "confirmed"
    assert _row(ledger, "facts", fact.id)["expires_at"] is None
    assert _row(ledger, "fact_owner_revisions", fact.id)["confirmed_by_assignment"] == 1


def test_confirmed_target_gets_an_owner_only(ledger, clock):
    fact = _confirmed(ledger, clock)
    row, jobs = _row(ledger, "facts", fact.id), _row(ledger, "index_jobs", fact.id)
    assert _assign(ledger, fact) == AssignmentReceipt(fact.revision, 1, 0)
    assert _row(ledger, "facts", fact.id) == row and _row(ledger, "index_jobs", fact.id) == jobs
    assert _row(ledger, "fact_owner_revisions", fact.id)["confirmed_by_assignment"] == 0


def test_same_person_with_matching_cas_is_a_no_op(ledger, clock):
    fact = _automatic(ledger, clock)
    _assign(ledger, fact)
    fact = ledger.get(fact.id)
    before = _bytes(ledger)
    assert _assign(ledger, fact, owner_revision=1) == AssignmentReceipt(fact.revision, 1, 0)
    assert _bytes(ledger) == before


def test_reassign_bumps_the_owner_revision_and_keeps_confirmation_provenance(ledger, clock):
    fact = _automatic(ledger, clock)
    _assign(ledger, fact)
    fact = ledger.get(fact.id)
    assert _assign(ledger, fact, OTHER, owner_revision=1) == AssignmentReceipt(fact.revision, 2, 0)
    record = _row(ledger, "fact_owner_revisions", fact.id)
    assert (record["revision"], record["last_change"], record["confirmed_by_assignment"]) == (2, "reassigned", 1)
    assert _row(ledger, "fact_owners", fact.id)["person_id"] == OTHER


# --- exceptions (contracts_1a.exceptions), ledger byte-identical ----------------------------


def test_unavailable_targets_are_lookup_errors(ledger, clock):
    missing = replace(_confirmed(ledger, clock), id="fact_" + "0" * 64)
    _refused(ledger, LookupError, "memory_fact_unavailable", lambda: _assign(ledger, missing))
    deleted = _confirmed(ledger, clock, "To delete")
    ledger.delete_fact(deleted.id, REVIEWER)
    _refused(ledger, LookupError, "memory_fact_unavailable", lambda: _assign(ledger, deleted))
    quarantined = _automatic(ledger, clock, "To quarantine")
    ledger._db.execute("UPDATE facts SET state='quarantined' WHERE id=?", (quarantined.id,))
    _refused(ledger, LookupError, "memory_fact_unavailable", lambda: _assign(ledger, quarantined))
    tombstoned = _confirmed(ledger, clock, "Tombstoned")
    ledger._db.execute("INSERT INTO conversation_tombstones VALUES (?,?,?)",
                       (tombstoned.sources[0].conversation_id, NOW, "sql"))
    _refused(ledger, LookupError, "memory_fact_unavailable", lambda: _assign(ledger, tombstoned))
    expired = _automatic(ledger, clock, "Will expire")
    clock.value = expired.expires_at
    _refused(ledger, LookupError, "memory_fact_unavailable", lambda: _assign(ledger, expired))


def test_revision_conflicts(ledger, clock):
    fact = _automatic(ledger, clock)
    _refused(ledger, ValueError, "memory_owner_conflict",
             lambda: ledger.assign_owner(fact.id, PERSON, fact.revision + 1, 0, OPERATOR, **ALLOW))
    _refused(ledger, ValueError, "memory_owner_conflict", lambda: _assign(ledger, fact, owner_revision=1))
    _assign(ledger, fact)
    fact = ledger.get(fact.id)
    _refused(ledger, ValueError, "memory_owner_conflict", lambda: _assign(ledger, fact, OTHER, owner_revision=0))
    _refused(ledger, ValueError, "memory_owner_conflict", lambda: _assign(ledger, fact, OTHER, owner_revision=2))
    for bad in (True, "1", None, -1):
        _refused(ledger, ValueError, "memory_owner_conflict", lambda: _assign(ledger, fact, OTHER, owner_revision=bad))


def test_derived_target_is_refused(ledger, clock):
    derived = _derived(ledger, clock, _confirmed(ledger, clock))
    _refused(ledger, ValueError, "memory_owner_derived_fact", lambda: _assign(ledger, derived))


@pytest.mark.parametrize("person", ["", "bad\x00id", "x" * 129, None, 7])
def test_invalid_person_shape(ledger, clock, person):
    fact = _automatic(ledger, clock)
    _refused(ledger, ValueError, "memory_owner_invalid_person", lambda: _assign(ledger, fact, person))


def test_wrong_actor_or_stale_session_is_denied(ledger, clock):
    fact = _automatic(ledger, clock)
    for actor in (REVIEWER, MemoryOwnerActor("s", PERSON), None, MemoryAssigner("")):
        _refused(ledger, PermissionError, "memory_assign_denied", lambda: _assign(ledger, fact, actor=actor))
    for current in (lambda: False, lambda: None, lambda: 1 / 0):
        actor = MemoryAssigner("synthetic-operator", current)
        _refused(ledger, PermissionError, "memory_assign_denied", lambda: _assign(ledger, fact, actor=actor))
    for method in (lambda: ledger.unassign_owner(fact.id, 0, REVIEWER), lambda: ledger.owned_counts(REVIEWER)):
        _refused(ledger, PermissionError, "memory_assign_denied", method)


def _no_reads(ledger, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("ledger read before the authority refusal")
    monkeypatch.setattr(ledger, "_fact", forbidden)
    monkeypatch.setattr(ledger, "initialize", forbidden)


@pytest.mark.parametrize("missing", [{"retired": None}, {"active": None}, {"retired": "no"}, {"active": 1}])
def test_missing_account_callables_refuse_before_any_read(ledger, clock, monkeypatch, missing):
    fact = _automatic(ledger, clock)
    _no_reads(ledger, monkeypatch)
    _refused(ledger, PermissionError, "memory_owner_authority_missing",
             lambda: _assign(ledger, fact, **missing))


def _raises(*_args):
    raise OSError("account provider down")


@pytest.mark.parametrize("callables", [
    {"retired": _raises}, {"active": _raises},
    {"retired": lambda ids: None}, {"retired": lambda ids: 0}, {"retired": lambda ids: "False"},
    {"active": lambda person: 1}, {"active": lambda person: None}, {"active": lambda person: MagicMock()},
])
def test_raising_or_non_bool_callables_fail_closed_before_any_read(ledger, clock, monkeypatch, callables):
    fact = _automatic(ledger, clock)
    _no_reads(ledger, monkeypatch)
    _refused(ledger, PermissionError, "memory_owner_authority_unavailable",
             lambda: _assign(ledger, fact, **callables))


def test_retired_and_inactive_people_are_refused(ledger, clock):
    fact = _automatic(ledger, clock)
    calls = []
    _refused(ledger, PermissionError, "memory_owner_retired",
             lambda: _assign(ledger, fact, retired=lambda ids: calls.append(ids) or True))
    assert calls == [frozenset({PERSON})]
    _refused(ledger, PermissionError, "memory_owner_inactive", lambda: _assign(ledger, fact, active=lambda p: False))


@pytest.mark.parametrize("which", ["retired", "active"])
@pytest.mark.parametrize("flip_at", [2, 3, 4])
def test_account_flip_between_guard_calls_refuses_inside_the_transaction(ledger, clock, which, flip_at):
    """Guard order: before initialize (1), after initialize (2), after the body (3), before commit (4)."""
    fact = _automatic(ledger, clock)
    calls = []

    def flipping(*_args):
        calls.append(1)
        allowed = len(calls) < flip_at
        return (not allowed) if which == "retired" else allowed

    code = "memory_owner_retired" if which == "retired" else "memory_owner_inactive"
    _refused(ledger, PermissionError, code, lambda: _assign(ledger, fact, **{which: flipping}))
    assert len(calls) == flip_at
    assert _row(ledger, "fact_owners", fact.id) is None and ledger.get(fact.id).state.value == "automatic"


# --- dependents rule (AUTOMATIC -> CONFIRMED only) ---------------------------------------


def test_automatic_target_with_dependents_lists_each_state_and_deletes_only_with_the_flag(ledger, clock):
    parent = _automatic(ledger, clock)
    child = _derived(ledger, clock, parent, "Derived child")
    grandchild = _derived(ledger, clock, child, "Derived grandchild")
    confirmed_child = _derived(ledger, clock, parent, "Derived child later confirmed by SQL")
    ledger._db.execute("UPDATE facts SET state='confirmed', expires_at=NULL WHERE id=?", (confirmed_child.id,))
    expected = sorted([(child.id, "automatic"), (grandchild.id, "automatic"), (confirmed_child.id, "confirmed")])
    before = _bytes(ledger)
    with pytest.raises(MemoryOwnerDependents) as raised:
        _assign(ledger, parent)
    assert str(raised.value) == "memory_owner_has_dependents" and isinstance(raised.value, ValueError)
    assert raised.value.code == "memory_owner_has_dependents" and raised.value.dependents == expected
    assert _bytes(ledger) == before
    receipt = _assign(ledger, parent, delete_dependents=True)
    assert receipt == AssignmentReceipt(parent.revision + 1, 1, 3, tuple(identity for identity, _ in expected))
    assert len(receipt.deleted_dependent_ids) == receipt.deleted_dependents
    assert {_row(ledger, "facts", identity)["state"] for identity, _ in expected} == {"deleted"}
    assert _row(ledger, "facts", parent.id)["state"] == "confirmed"


def test_confirmed_target_with_dependents_reassigns_and_leaves_them(ledger, clock):
    parent = _confirmed(ledger, clock)
    child = _derived(ledger, clock, parent)
    child_row = _row(ledger, "facts", child.id)
    _assign(ledger, parent)
    _assign(ledger, ledger.get(parent.id), OTHER, owner_revision=1)
    assert _row(ledger, "facts", child.id) == child_row
    for target in (parent, _proposed(ledger, clock)):
        current = ledger.get(target.id, review=True)
        revision = _row(ledger, "fact_owner_revisions", target.id)
        _refused(ledger, ValueError, "memory_owner_dependents_not_applicable",
                 lambda: _assign(ledger, current, OTHER, 0 if revision is None else revision["revision"],
                                 delete_dependents=True))


def test_assignment_does_not_apply_the_dependents_rule_to_proposals(ledger, clock):
    proposal = _proposed(ledger, clock)
    # Impossible through the API (proposals are never receipts); inserted by SQL.
    child = _automatic(ledger, clock, "SQL child of a proposal")
    ledger._db.execute("INSERT INTO fact_dependencies VALUES (?,?)", (child.id, proposal.id))
    assert _assign(ledger, proposal) == AssignmentReceipt(proposal.revision + 1, 1, 0)
    assert _row(ledger, "facts", child.id)["state"] == "automatic"


# --- 'No es mío' and operator unassign ------------------------------------------------------


def test_owner_return_goes_to_devueltos_and_can_be_reassigned(ledger, clock):
    fact = _automatic(ledger, clock)
    _assign(ledger, fact)
    row = _row(ledger, "facts", fact.id)
    actor = MemoryOwnerActor("synthetic-home", PERSON)
    assert ledger.return_owned(fact.id, PERSON, actor).returned is True
    assert _row(ledger, "facts", fact.id) == row and row["state"] == "confirmed" and row["expires_at"] is None
    assert _row(ledger, "fact_owners", fact.id) is None
    record = _row(ledger, "fact_owner_revisions", fact.id)
    assert (record["revision"], record["last_change"], record["confirmed_by_assignment"]) == (
        2, "owner_returned", 1)
    assert ledger.owned_counts(OPERATOR) == OwnedCounts(0, 1, 1, 0)
    assert ledger.return_owned(fact.id, PERSON, actor).returned is False
    fact = ledger.get(fact.id)
    _refused(ledger, ValueError, "memory_owner_conflict", lambda: _assign(ledger, fact, OTHER, owner_revision=1))
    assert _assign(ledger, fact, OTHER, owner_revision=2) == AssignmentReceipt(fact.revision, 3, 0)
    assert _row(ledger, "fact_owner_revisions", fact.id)["last_change"] == "assigned"
    assert ledger.owned_counts(OPERATOR) == OwnedCounts(1, 0, 1, 0)


def test_return_of_a_fact_not_owned_by_the_actor_is_indistinguishable_from_missing(ledger, clock):
    fact = _confirmed(ledger, clock)
    _assign(ledger, fact)
    other = MemoryOwnerActor("synthetic-home", OTHER)
    before = _bytes(ledger)
    assert ledger.return_owned(fact.id, OTHER, other).returned is False
    assert ledger.return_owned("fact_" + "0" * 64, OTHER, other).returned is False
    assert _bytes(ledger) == before
    for actor, person in ((other, PERSON), (OPERATOR, PERSON), (MemoryOwnerActor("s", PERSON, lambda: False), PERSON)):
        _refused(ledger, PermissionError, "memory_owner_actor_denied",
                 lambda: ledger.return_owned(fact.id, person, actor))


def test_operator_unassign_keeps_the_fact_and_is_not_devueltos(ledger, clock):
    fact = _automatic(ledger, clock)
    _assign(ledger, fact)
    row = _row(ledger, "facts", fact.id)
    _refused(ledger, ValueError, "memory_owner_conflict", lambda: ledger.unassign_owner(fact.id, 0, OPERATOR))
    assert ledger.unassign_owner(fact.id, 1, OPERATOR) == AssignmentReceipt(row["revision"], 2)
    assert _row(ledger, "facts", fact.id) == row and _row(ledger, "fact_owners", fact.id) is None
    assert _row(ledger, "fact_owner_revisions", fact.id)["last_change"] == "operator_unassigned"
    assert ledger.owned_counts(OPERATOR) == OwnedCounts(0, 0, 1, 0)
    before = _bytes(ledger)
    assert ledger.unassign_owner(fact.id, 2, OPERATOR) == AssignmentReceipt(row["revision"], 2)
    assert _bytes(ledger) == before
    _refused(ledger, LookupError, "memory_fact_unavailable",
             lambda: ledger.unassign_owner("fact_" + "0" * 64, 0, OPERATOR))


# --- owned_counts matrix ------------------------------------------------------------------

_STATES = ("automatic", "proposed", "confirmed", "deleted")
_CHANGES = (None, "assigned", "reassigned", "owner_returned", "operator_unassigned")
_MATRIX = [(state, owned, change) for state in _STATES for owned in (True, False) for change in _CHANGES]


def _insert_cell(db, index, state, owned, change):
    """Rows impossible through the API (owner on automatic, deleted with owner...) go in by SQL."""
    fact_id = "fact_" + format(index, "064x")
    db.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,NULL,NULL)",
               (fact_id, None if state == "deleted" else "matrix", "fact", .9, state, NOW,
                None if state in ("confirmed", "deleted") else NOW + 60, 1))
    if change is not None:
        db.execute("INSERT INTO fact_owner_revisions VALUES (?,?,?,?,?)", (fact_id, 3, change, 0, NOW))
    if owned:
        db.execute("INSERT INTO fact_owners VALUES (?,?,?,?,?,?,'{}')",
                   (fact_id, PERSON, "operator_assignment", 3, NOW, "sql"))


def _oracle(cells):
    live = [cell for cell in cells if cell[0] != "deleted"]
    return OwnedCounts(
        currently_owned=sum(1 for _, owned, _ in live if owned),
        returned=sum(1 for _, owned, change in live if not owned and change == "owner_returned"),
        ever_assigned_live=sum(1 for _, _, change in live if change is not None),
        owned_proposals=sum(1 for state, owned, _ in live if owned and state == "proposed"))


@pytest.mark.parametrize("cell", _MATRIX, ids=lambda cell: "-".join(map(str, cell)))
def test_owned_counts_cell(clock, cell):
    ledger = FactLedger(":memory:", clock=clock)
    ledger.initialize()
    try:
        _insert_cell(ledger._db, 1, *cell)
        if cell[1] and cell[2] is None:
            # An owner row without its revision row breaks the mirror invariant.
            with pytest.raises(RuntimeError, match="fact_owner_revision_mismatch"):
                ledger.owned_counts(OPERATOR)
        else:
            assert ledger.owned_counts(OPERATOR) == _oracle([cell])
    finally:
        ledger.close()


def test_owned_counts_over_the_full_valid_matrix(clock):
    ledger = FactLedger(":memory:", clock=clock)
    ledger.initialize()
    try:
        cells = [cell for cell in _MATRIX if not (cell[1] and cell[2] is None)]
        for index, cell in enumerate(cells):
            _insert_cell(ledger._db, index, *cell)
        counts = ledger.owned_counts(OPERATOR)
        assert counts == _oracle(cells) == OwnedCounts(12, 3, 24, 4)
        assert counts.to_dict() == {"currently_owned": 12, "returned": 3, "ever_assigned_live": 24,
                                    "owned_proposals": 4}
    finally:
        ledger.close()


# --- scopes (W5) ---------------------------------------------------------------------------


@pytest.fixture
def scoped(ledger, clock):
    household = _confirmed(ledger, clock, "Household")
    automatic = _automatic(ledger, clock, "Household automatic")
    owned, other = _confirmed(ledger, clock, "Owned"), _automatic(ledger, clock, "Other's")
    returned, unassigned = _confirmed(ledger, clock, "Returned"), _confirmed(ledger, clock, "Unassigned")
    for fact in (owned, returned, unassigned):
        _assign(ledger, fact)
    _assign(ledger, other, OTHER)
    ledger.return_owned(returned.id, PERSON, MemoryOwnerActor("home", PERSON))
    ledger.unassign_owner(unassigned.id, 1, OPERATOR)
    return {"household": {household.id, automatic.id}, PERSON: {owned.id}, OTHER: {other.id},
            "all": {household.id, automatic.id, owned.id, other.id, returned.id, unassigned.id}}


def test_scopes_partition_owned_household_and_returned_facts(ledger, scoped):
    assert set(ledger.eligible_ids(scope="household")) == scoped["household"]
    assert set(ledger.eligible_ids(scope=MemoryScope.HOUSEHOLD)) == scoped["household"]
    assert set(ledger.eligible_ids(scope="person", person_id=PERSON)) == scoped[PERSON]
    assert set(ledger.eligible_ids(scope="person", person_id=OTHER)) == scoped[OTHER]
    assert set(ledger.eligible_ids(scope="person", person_id="user_nobody")) == set()
    assert set(ledger.eligible_ids(scope="all")) == scoped["all"]
    assert {f.id for f in ledger.list(scope="household")} == scoped["household"]
    assert {f.id for f in ledger.list(scope="person", person_id=PERSON)} == scoped[PERSON]
    assert {f.id for f in ledger.list(scope="all")} == scoped["all"]
    assert set(ledger.eligible_ids(category="fact", scope="person", person_id=OTHER)) == scoped[OTHER]


@pytest.mark.parametrize("arguments", [
    {"scope": "mine"}, {"scope": None}, {"scope": 1}, {"scope": "person"}, {"scope": "person", "person_id": ""},
    {"scope": "household", "person_id": PERSON}, {"scope": "all", "person_id": PERSON},
])
def test_invalid_scopes_are_refused(ledger, arguments):
    with pytest.raises(ValueError, match="invalid_memory_scope"):
        ledger.eligible_ids(**arguments)
    with pytest.raises(ValueError, match="invalid_memory_scope"):
        ledger.list(**arguments)


def test_scope_is_a_required_keyword_on_fact_readers(ledger):
    memory = LongTermMemory(MagicMock(), ledger_path=":memory:")
    # U3b-2 (A16 R7): FactLedger.list lost its 'all' default with the operator route.
    for call in (lambda: ledger.eligible_ids(), lambda: ledger.eligible_ids(0, category="fact"),
                 lambda: ledger.list(), lambda: ledger.list(limit=1),
                 lambda: memory.search_relevant("q"), lambda: memory.get_by_category("fact")):
        with pytest.raises(TypeError, match="scope"):
            call()


async def test_async_search_requires_a_scope(tmp_path):
    memory = LongTermMemory(MagicMock(), ledger_path=str(tmp_path / "privacy.db"))
    try:
        with pytest.raises(TypeError, match="scope"):
            await memory.asearch_relevant("q")
    finally:
        await memory.aclose()


def test_long_term_search_resolves_index_candidates_inside_the_scope(tmp_path, clock, scoped, ledger):
    client = MagicMock()
    memory = LongTermMemory(client, ledger_path=":memory:")
    memory.ledger = ledger
    collection = client.get_or_create_collection.return_value
    collection.count.return_value = 6
    # The index holds every candidate; it is only narrowed to them.
    collection.get.side_effect = lambda ids, include: {"ids": list(ids)}
    collection.query.return_value = {"ids": [[]], "distances": [[]]}
    memory.search_relevant("q", scope="person", person_id=PERSON)
    assert set(collection.get.call_args.kwargs["ids"]) == scoped[PERSON]
    assert set(collection.query.call_args.kwargs["ids"]) == scoped[PERSON]
    memory.search_relevant("q", scope="household")
    assert set(collection.get.call_args.kwargs["ids"]) == scoped["household"]
    assert set(collection.query.call_args.kwargs["ids"]) == scoped["household"]
    assert {fact["id"] for fact in memory.get_by_category("fact", 10, scope="all")} == scoped["all"]


def test_operator_stats_are_unchanged_by_assignments(ledger, clock):
    memory = LongTermMemory(MagicMock(), ledger_path=":memory:")
    memory.ledger = ledger
    facts = [_confirmed(ledger, clock, "Stat " + str(i)) for i in range(3)]
    preference = ledger.store("color: azul", "preference", .9, (_source(clock),))
    before = (memory.get_stats(), ledger.eligible_ids(category="preference", scope="all"))
    _assign(ledger, facts[0])
    _assign(ledger, facts[1], OTHER)
    ledger.return_owned(facts[0].id, PERSON, MemoryOwnerActor("home", PERSON))
    assert (memory.get_stats(), ledger.eligible_ids(category="preference", scope="all")) == before
    assert before[1] == [preference.id]


# --- pruning --------------------------------------------------------------------------------


def _owner_rows(ledger, fact_id):
    return (_row(ledger, "fact_owners", fact_id), _row(ledger, "fact_owner_revisions", fact_id))


def test_delete_prunes_owner_and_revision_rows_for_every_affected_id(ledger, clock):
    parent = _confirmed(ledger, clock)
    child = _derived(ledger, clock, parent)
    _assign(ledger, parent)
    # Owner rows on a derived fact are impossible through the API; SQL covers _delete's closure.
    ledger._db.execute("INSERT INTO fact_owner_revisions VALUES (?,1,'owner_returned',0,?)", (child.id, NOW))
    assert ledger.delete_fact(parent.id, REVIEWER)
    assert _owner_rows(ledger, parent.id) == _owner_rows(ledger, child.id) == (None, None)


def test_expire_prunes_owner_rows(ledger, clock):
    proposal = _proposed(ledger, clock)
    automatic = _automatic(ledger, clock, "Expiring automatic")
    ledger._db.execute("INSERT INTO fact_owner_revisions VALUES (?,1,'voice_proposed',0,?)", (proposal.id, NOW))
    ledger._db.execute("INSERT INTO fact_owner_revisions VALUES (?,1,'assigned',0,?)", (automatic.id, NOW))
    ledger._db.execute("INSERT INTO fact_owners VALUES (?,?,'voice_proposal',1,?,'sql','{}')",
                       (automatic.id, PERSON, NOW))
    clock.value = automatic.expires_at + 1
    assert ledger.expire() == 2
    assert _owner_rows(ledger, proposal.id) == _owner_rows(ledger, automatic.id) == (None, None)


def test_conversation_tombstone_prunes_owner_rows(ledger, clock):
    conversation = str(uuid4())
    fact = _confirmed(ledger, clock, conversation=conversation)
    _assign(ledger, fact)
    assert ledger.delete_conversation(conversation, REVIEWER) == 1
    assert _owner_rows(ledger, fact.id) == (None, None)


def test_initializer_prunes_rows_whose_fact_is_deleted(tmp_path, clock):
    path = tmp_path / "privacy.db"
    ledger = FactLedger(str(path), clock=clock)
    ledger.initialize()
    kept, gone = _confirmed(ledger, clock), _confirmed(ledger, clock, "Deleted by SQL")
    _assign(ledger, kept)
    _assign(ledger, gone)
    ledger.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE facts SET state='deleted', content=NULL WHERE id=?", (gone.id,))
    ledger.initialize()
    try:
        assert _owner_rows(ledger, gone.id) == (None, None)
        assert _row(ledger, "fact_owners", kept.id)["person_id"] == PERSON
    finally:
        ledger.close()


# --- enums and pins ---------------------------------------------------------------------------


def test_owner_enums_are_closed_and_include_the_voice_stage():
    assert [m.value for m in FactOwnerMethod] == ["operator_assignment", "person_session", "voice_proposal"]
    assert [c.value for c in FactOwnerChange] == ["assigned", "reassigned", "owner_returned", "operator_unassigned",
                                                  "voice_proposed", "owner_confirmed", "operator_confirmed"]
    assert [s.value for s in MemoryScope] == ["household", "person", "all"]


def test_owner_permissions_are_granted_only_to_their_roles():
    """U3b-2 replaced the 1a 'no grant' pin: memory:assign is operator-only and
    memory:own:* belongs only to a household credential bound to a person."""
    from src.dashboard.authority import DashboardRole, _permissions
    own = {"memory:own:read", "memory:own:write"}
    assert "memory:assign" in _permissions(DashboardRole.OPERATOR, None)
    assert not own & _permissions(DashboardRole.OPERATOR, None)
    assert own <= _permissions(DashboardRole.HOME, PERSON)
    assert "memory:assign" not in _permissions(DashboardRole.HOME, PERSON)
    assert not (own | {"memory:assign"}) & _permissions(DashboardRole.HOME, None)
    assert not (own | {"memory:assign"}) & _permissions(DashboardRole.EVALUATOR, None)


def test_minting_requires_the_exact_permission():
    from src.dashboard.authority import DashboardPrincipal, DashboardRole, _permissions
    operator = _permissions(DashboardRole.OPERATOR, None)
    assert MemoryAssigner.from_dashboard(DashboardPrincipal("operator", operator)).subject == "operator"
    with pytest.raises(PermissionError, match="memory_assign_denied"):
        MemoryAssigner.from_dashboard(DashboardPrincipal("home", _permissions(DashboardRole.HOME, PERSON),
                                                         DashboardRole.HOME, PERSON))
    with pytest.raises(PermissionError, match="memory_owner_actor_denied"):
        MemoryOwnerActor.from_dashboard(DashboardPrincipal("operator", operator))
    # A principal that somehow held the strings without the HOME role or a person is still refused.
    everything = operator | {"memory:own:read", "memory:own:write"}
    with pytest.raises(PermissionError, match="memory_owner_actor_denied"):
        MemoryOwnerActor.from_dashboard(DashboardPrincipal("operator", everything, DashboardRole.OPERATOR, PERSON))
    with pytest.raises(PermissionError, match="memory_owner_actor_denied"):
        MemoryOwnerActor.from_dashboard(DashboardPrincipal("home", frozenset({"memory:own:write"}),
                                                           DashboardRole.HOME, None))
    with pytest.raises(PermissionError, match="memory_owner_actor_denied"):
        MemoryOwnerActor.from_dashboard(DashboardPrincipal("home", frozenset({"memory:own:read"}),
                                                           DashboardRole.HOME, PERSON))
    with pytest.raises(PermissionError, match="memory_owner_actor_denied"):
        MemoryOwnerReader.from_dashboard(DashboardPrincipal("home", frozenset({"memory:own:write"}),
                                                            DashboardRole.HOME, PERSON))
    assert MemoryAssigner.from_dashboard(DashboardPrincipal("op", frozenset({"memory:assign"}))).subject == "op"
    actor = MemoryOwnerActor.from_dashboard(DashboardPrincipal("home", frozenset({"memory:own:write"}),
                                                               DashboardRole.HOME, PERSON))
    assert (actor.subject, actor.person_id) == ("home", PERSON)
    reader = MemoryOwnerReader.from_dashboard(DashboardPrincipal("home", frozenset({"memory:own:read"}),
                                                                 DashboardRole.HOME, PERSON))
    assert (reader.subject, reader.person_id) == ("home", PERSON)
    for invalid in (object(), None):
        with pytest.raises(PermissionError):
            MemoryAssigner.from_dashboard(invalid)
        with pytest.raises(PermissionError):
            MemoryOwnerReader.from_dashboard(invalid)


_ASSIGNMENT_NAMES = frozenset({"assign_owner", "return_owned", "unassign_owner", "MemoryAssigner",
                               "MemoryOwnerActor", "memory:assign", "memory:own:write"})


def _references(source, names):
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in names:
            found.append(f"{node.lineno}:{node.attr}")
        elif isinstance(node, ast.Name) and node.id in names:
            found.append(f"{node.lineno}:{node.id}")
        elif isinstance(node, ast.Constant) and node.value in names:
            found.append(f"{node.lineno}:{node.value!r}")
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            found += [f"{node.lineno}:import {alias.name}" for alias in node.names
                      if alias.name.rsplit(".", 1)[-1] in names]
    return found


# U3b-2 replaced the 1a zero-callers pin: these are the only production references.
_ASSIGNMENT_REFERENCES = {
    "src/dashboard/authority.py": {"'memory:assign'", "'memory:own:write'"},
    "src/dashboard/memory_api.py": {"MemoryAssigner", "MemoryOwnerActor", "import MemoryAssigner",
                                    "import MemoryOwnerActor", "assign_owner", "unassign_owner", "return_owned",
                                    "'memory:assign'", "'memory:own:write'"},
    "src/memory/long_term.py": {"import MemoryOwnerActor", "MemoryOwnerActor"},
    # U3b-4a-2: operator voice routes (consent reopen, resume) mint MemoryAssigner from the session.
    "src/dashboard/memory_voice_api.py": {"MemoryAssigner", "import MemoryAssigner", "'memory:assign'"},
}


def test_assignment_api_has_exactly_the_u3b2_production_references():
    found = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path == ROOT / "src/memory/fact_ledger.py":
            continue
        uses = {use.split(":", 1)[1] for use in _references(path.read_text(encoding="utf-8"), _ASSIGNMENT_NAMES)}
        if uses:
            found[str(path.relative_to(ROOT))] = uses
    assert found == _ASSIGNMENT_REFERENCES


def test_the_assignment_scan_detects_each_name():
    for name in _ASSIGNMENT_NAMES - {"memory:assign", "memory:own:write"}:
        assert _references(f"ledger.{name}(1)\n", {name})
        assert _references(f"from src.memory.fact_ledger import {name}\n", {name})
        assert _references(f"getattr(ledger, {name!r})\n", {name})
    assert _references("x = {'memory:assign'}\n", {"memory:assign"})


_READERS = {"eligible_ids", "search_relevant", "asearch_relevant", "get_by_category", "list"}
# (module, reader, scope expression). 'scope' is LongTermMemory forwarding its
# own required keyword to the ledger; every caller that decides passes 'all'.
_SCOPED_CALLS = [
    ("src/memory/long_term.py", "eligible_ids", "scope"),        # search_relevant -> ledger
    ("src/memory/long_term.py", "list", "scope"),                # get_by_category -> ledger
    ("src/memory/long_term.py", "list", "'all'"),                # proposals (review queue)
    ("src/memory/long_term.py", "eligible_ids", "'all'"),        # get_stats
    ("src/memory/memory_manager.py", "search_relevant", "'all'"),  # build_context (prompt path)
    ("src/memory/memory_manager.py", "get_by_category", "'all'"),  # build_context preferences
    ("src/memory/memory_manager.py", "eligible_ids", "'all'"),     # get_stats preferences_count
]


def _reader_calls():
    calls = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = str(path.relative_to(ROOT))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _READERS):
                continue
            receiver = ast.unparse(node.func.value)
            if node.func.attr == "list" and not receiver.endswith("ledger"):
                continue
            scope = next((ast.unparse(k.value) for k in node.keywords if k.arg == "scope"), None)
            calls.append((relative, node.func.attr, scope))
    return calls


def test_every_existing_fact_reader_call_passes_scope_all():
    """1a-F5: no behaviour change in U3b-1a; the prompt-scope change belongs to U3b-5."""
    assert sorted(_reader_calls()) == sorted(_SCOPED_CALLS)


def test_no_reader_relies_on_a_list_default():
    """U3b-2 replaced the 1a list-default pin (A5/A16 R7): the keyword is required everywhere.

    Signature-level, so an aliased receiver cannot escape it.
    """
    import inspect
    from src.memory.fact_ledger import FactLedger
    parameter = inspect.signature(FactLedger.list).parameters["scope"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is inspect.Parameter.empty
    calls = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            # Direct calls and worker hand-offs such as service.acall(service.ledger.list, ...).
            targets = [node.func, *node.args]
            if any(isinstance(t, ast.Attribute) and t.attr == "list" and ast.unparse(t.value).endswith("ledger")
                   for t in targets):
                scope = next((ast.unparse(k.value) for k in node.keywords if k.arg == "scope"), None)
                calls.append((str(path.relative_to(ROOT)), scope))
    assert sorted(calls) == [("src/dashboard/memory_api.py", "MemoryScope.ALL"),
                             ("src/memory/long_term.py", "'all'"), ("src/memory/long_term.py", "scope")]


# Review R0: the scope-required readers, scanned everywhere a caller can live, including
# code embedded in string literals (scripts run against compiled slots).
_SCOPE_READERS = {"list", "eligible_ids", "search_relevant", "asearch_relevant", "get_by_category"}
_UNSCOPED_ALLOWED = {
    # LongTermMemory.asearch_relevant forwards the caller's kwargs (scope included) to its worker.
    ("src/memory/long_term.py", "self._run(self.search_relevant, args, kwargs)"),
    # Deliberate negatives asserting the TypeError.
    ("tests/unit/memory/test_fact_owners.py", "ledger.eligible_ids()"),
    ("tests/unit/memory/test_fact_owners.py", "ledger.eligible_ids(0, category='fact')"),
    ("tests/unit/memory/test_fact_owners.py", "ledger.list()"),
    ("tests/unit/memory/test_fact_owners.py", "ledger.list(limit=1)"),
    ("tests/unit/memory/test_fact_owners.py", "memory.search_relevant('q')"),
    ("tests/unit/memory/test_fact_owners.py", "memory.get_by_category('fact')"),
    ("tests/unit/memory/test_fact_owners.py", "memory.asearch_relevant('q')"),
    # A synthetic source fed to the caller scanner of test_fact_reader_callers.
    ("tests/unit/memory/test_fact_reader_callers.py", "memory.long_term.acall(memory.long_term.search_relevant, 'q')"),
}


def _parse_all(text):
    return compile(text, "<scan>", "exec", ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def _unscoped_reader_calls(source, *, embedded=True):
    """Calls or worker hand-offs of a scope-required reader without scope= (or **kwargs)."""
    tree = _parse_all(source)
    trees = [tree]
    for node in ast.walk(tree) if embedded else ():
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and any(name in node.value for name in _SCOPE_READERS):
            try:
                trees.append(_parse_all(node.value.strip()))
            except SyntaxError:
                pass
    found = []
    for candidate in trees:
        for node in ast.walk(candidate):
            if not isinstance(node, ast.Call):
                continue
            for target in (node.func, *node.args):
                if not (isinstance(target, ast.Attribute) and target.attr in _SCOPE_READERS):
                    continue
                if target.attr == "list" and not ast.unparse(target.value).endswith("ledger"):
                    continue
                if not any(k.arg in ("scope", None) for k in node.keywords):
                    found.append(ast.unparse(node))
    return found


def test_every_scope_required_reader_call_passes_a_scope_in_src_tests_scripts_and_tools():
    found = set()
    for base in ("src", "tests", "scripts", "tools"):
        for path in sorted((ROOT / base).rglob("*.py")):
            relative = str(path.relative_to(ROOT))
            # This file's string literals are scanner fixtures; its real calls are still scanned.
            embedded = path != Path(__file__).resolve()
            found |= {(relative, call) for call in _unscoped_reader_calls(path.read_text(encoding="utf-8"),
                                                                         embedded=embedded)}
    assert found == _UNSCOPED_ALLOWED


def test_the_scope_scan_sees_hand_offs_and_embedded_scripts():
    assert _unscoped_reader_calls("x = await privacy.acall(privacy.ledger.list)\n")
    assert not _unscoped_reader_calls("x = await privacy.acall(privacy.ledger.list, scope='all')\n")
    embedded = 'script = "            facts = await policy.acall(policy.ledger.list)"\n'
    assert _unscoped_reader_calls(embedded)
    assert not _unscoped_reader_calls(embedded.replace("ledger.list)", "ledger.list, scope='all')"))
    assert _unscoped_reader_calls("memory.search_relevant('q')\n")
    assert not _unscoped_reader_calls("ids = ledger.eligible_ids(**query)\n")