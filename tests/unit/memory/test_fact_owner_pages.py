"""U3b-2 ledger side: keyset pages, the person's own view, owner delete/return and F1.

Synthetic ledgers only. Every refusal compares the database file bytes.
"""

from pathlib import Path
import re
import sqlite3
from uuid import uuid4

import pytest

from src.memory.fact_ledger import (
    MEMORY_PAGE_LIMIT, FactDependency, FactLedger, FactOwnerChange, FactPageCursor, MemoryAssigner,
    MemoryOrigin, MemoryOwnerActor, MemoryOwnerReader, MemoryReviewer, MemorySource,
)

NOW = 2_000_000_000.0
OPERATOR = MemoryAssigner("synthetic-operator")
REVIEWER = MemoryReviewer("synthetic-reviewer")
ALLOW = {"retired": lambda ids: False, "active": lambda person: True}
PERSON, OTHER = "user_" + "a" * 32, "user_" + "b" * 32
MINE = MemoryOwnerActor("synthetic-home-a", PERSON)
READ_MINE = MemoryOwnerReader("synthetic-home-a", PERSON)
READ_OTHER = MemoryOwnerReader("synthetic-home-b", OTHER)


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


def _source(clock, dependencies=()):
    return MemorySource(str(uuid4()), str(uuid4()), clock() - 1, clock() + 3600, MemoryOrigin.VOICE,
                        tuple(dependencies))


def _automatic(ledger, clock, content="Synthetic automatic fact"):
    return ledger.store(content, "fact", .9, (_source(clock),))


def _proposed(ledger, clock, content="Synthetic proposal"):
    return ledger.store(content, "fact", 1.0, (_source(clock),), proposed=True)


def _confirmed(ledger, clock, content="Synthetic confirmed fact"):
    return ledger.confirm(_proposed(ledger, clock, content).id, REVIEWER)


def _derived(ledger, clock, parent, content="Synthetic derived fact"):
    return ledger.store(content, "fact", .9,
                        (_source(clock, (FactDependency(parent.id, parent.revision, parent.expires_at),)),))


def _assign(ledger, fact, person=PERSON, owner_revision=0, **kwargs):
    return ledger.assign_owner(fact.id, person, fact.revision, owner_revision, OPERATOR, **{**ALLOW, **kwargs})


def _bytes(ledger):
    return Path(ledger.path).read_bytes()


def _owned(ledger, clock, count, person=PERSON, step=1.0, prefix="Synthetic owned"):
    facts = []
    for index in range(count):
        clock.value += step
        fact = _confirmed(ledger, clock, f"{prefix} {index}")
        _assign(ledger, fact, person)
        facts.append(fact.id)
    return facts


def _ids(page):
    return [item["id"] for item in page.items]


def _walk(read, limit):
    seen, cursor = [], None
    while True:
        page = read(after=cursor, limit=limit)
        assert not set(_ids(page)) & set(seen), "a page repeated an item"  # also bounds the walk
        seen += _ids(page)
        if page.next_cursor is None:
            return seen
        cursor = FactPageCursor.decode(page.next())


# --- cursor ------------------------------------------------------------------------------------


def test_cursor_is_exact_canonical_and_content_free():
    fact_id = "fact_" + "c" * 64
    for value in (0.0, NOW, NOW + 1e-6, 1.5, 2.0 ** -1074):
        cursor = FactPageCursor(value, fact_id)
        encoded = cursor.encode()
        assert re.fullmatch(r"k1\.[0-9a-f]{16}\.fact_[0-9a-f]{64}", encoded)
        assert FactPageCursor.decode(encoded) == cursor


@pytest.mark.parametrize("tampered", [
    "", "k1", "k2.41ddcd6500000000.fact_" + "c" * 64, "k1.41ddcd650000000.fact_" + "c" * 64,
    "k1.41DDCD6500000000.fact_" + "c" * 64, "k1.41ddcd6500000000.fact_" + "C" * 64,
    "k1.41ddcd6500000000.fact_" + "c" * 63, "k1.7ff0000000000000.fact_" + "c" * 64,   # +inf
    "k1.7ff8000000000000.fact_" + "c" * 64,                                             # NaN
    "k1.41ddcd6500000000.fact_" + "c" * 64 + "\n", " k1.41ddcd6500000000.fact_" + "c" * 64,
    "k1.41ddcd6500000000.fact_" + "c" * 64 + ".x", None, 12, b"k1",
])
def test_tampered_cursor_is_refused(tampered):
    with pytest.raises(ValueError, match="memory_cursor_invalid"):
        FactPageCursor.decode(tampered)


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.0, "5", None])
def test_page_arguments_are_validated_before_any_read(ledger, limit):
    before = _bytes(ledger)
    for call in (lambda: ledger.owned_page(READ_MINE, limit=limit),
                 lambda: ledger.returned_page(OPERATOR, limit=limit),
                 lambda: ledger.review_page(limit=limit)):
        with pytest.raises(ValueError, match="memory_page_invalid"):
            call()
    with pytest.raises(ValueError, match="memory_page_invalid"):
        ledger.owned_page(READ_MINE, after="k1.41ddcd6500000000.fact_" + "c" * 64)
    assert _bytes(ledger) == before


# --- keyset (V13) ------------------------------------------------------------------------------


def test_page_boundary_at_equal_extracted_at_neither_skips_nor_repeats(ledger, clock):
    clock.value += 10
    same = [_confirmed(ledger, clock, f"Synthetic same instant {index}") for index in range(5)]
    for fact in same:
        _assign(ledger, fact)
    older = _owned(ledger, clock, 2, step=-3)
    expected = sorted(fact.id for fact in same) + older  # extracted_at DESC, id ASC
    for limit in (1, 2, 3, 4, 7):
        assert _walk(lambda **kw: ledger.owned_page(READ_MINE, **kw), limit) == expected
    first = ledger.owned_page(READ_MINE, limit=2)
    assert first.next_cursor == FactPageCursor(same[0].extracted_at, sorted(f.id for f in same)[1])


def test_owner_filter_is_in_sql_so_pages_do_not_depend_on_other_people(ledger, clock):
    mine = _owned(ledger, clock, 3)
    # Interleave many newer facts of another person and of the household.
    for index in range(30):
        clock.value += 1
        _assign(ledger, _confirmed(ledger, clock, f"Synthetic other {index}"), OTHER)
        _automatic(ledger, clock, f"Synthetic household {index}")
    statements = []
    ledger._db.set_trace_callback(statements.append)
    try:
        page = ledger.owned_page(READ_MINE, limit=2)
    finally:
        ledger._db.set_trace_callback(None)
    assert _ids(page) == list(reversed(mine))[:2]
    # Only facts of this person are ever materialised: no read of another person's fact row.
    materialised = {m for s in statements for m in re.findall(r"SELECT \* FROM facts WHERE id='(fact_[0-9a-f]{64})'", s)}
    assert materialised <= set(mine)
    assert len(_walk(lambda **kw: ledger.owned_page(READ_OTHER, **kw), 7)) == 30


def test_assignment_during_paging_has_no_snapshot_guarantee(ledger, clock):
    """An assignment behind the cursor is missed by later pages; one ahead of it appears (V13)."""
    old = _confirmed(ledger, clock, "Synthetic oldest, assigned later")
    mine = _owned(ledger, clock, 4)
    clock.value += 1
    newest = _confirmed(ledger, clock, "Synthetic newest, assigned later")
    first = ledger.owned_page(READ_MINE, limit=2)
    assert _ids(first) == [mine[3], mine[2]]
    _assign(ledger, newest)
    _assign(ledger, old)
    rest, cursor = [], FactPageCursor.decode(first.next())
    while cursor is not None:
        page = ledger.owned_page(READ_MINE, after=cursor, limit=2)
        assert not set(_ids(page)) & set(rest), "a page repeated an item"
        rest += _ids(page)
        cursor = page.next_cursor
    assert rest == [mine[1], mine[0], old.id] and newest.id not in rest
    assert _ids(ledger.owned_page(READ_MINE, limit=100))[0] == newest.id


def test_owned_proposals_share_the_same_ordering(ledger, clock):
    confirmed = _owned(ledger, clock, 1)
    clock.value += 1
    proposal = _proposed(ledger, clock, "Synthetic owned proposal")
    # No 1a/2 API owns a proposal (voice stage); insert the rows by SQL as 1a tests do.
    ledger._db.execute("INSERT INTO fact_owners VALUES (?,?, 'voice_proposal', 1, ?, 'synthetic', '{}')",
                       (proposal.id, PERSON, clock()))
    ledger._db.execute("INSERT INTO fact_owner_revisions VALUES (?,1,'voice_proposed',0,?)", (proposal.id, clock()))
    ledger._db.commit()
    page = ledger.owned_page(READ_MINE)
    assert _ids(page) == [proposal.id, confirmed[0]]
    assert page.items[0]["state"] == "proposed" and page.items[0]["owner_method"] == "voice_proposal"


def test_person_projection_carries_no_sources_subject_counts_or_foreign_ids(ledger, clock):
    parent = _automatic(ledger, clock, "Synthetic parent")
    child = _derived(ledger, clock, parent, "Synthetic hidden child")
    with pytest.raises(ValueError, match="memory_owner_has_dependents"):
        _assign(ledger, parent)
    confirmed = _confirmed(ledger, clock, "Synthetic owned parent")
    hidden = _derived(ledger, clock, confirmed, "Synthetic derived from owned")
    _assign(ledger, confirmed)
    page = ledger.owned_page(READ_MINE)
    assert [item["id"] for item in page.items] == [confirmed.id]
    item = page.items[0]
    assert set(item) == {"id", "content", "category", "state", "created_at", "expires_at", "owner_method",
                         "owner_assigned_at", "owned_dependents", "has_hidden_dependents"}
    assert item["owned_dependents"] == [] and item["has_hidden_dependents"] is True
    text = repr(page.items)
    for foreign in (hidden.id, child.id, parent.id, "synthetic-operator", "synthetic-reviewer", "capture", "conversation"):
        assert foreign not in text


def test_expired_tombstoned_and_deleted_owned_facts_leave_the_view(ledger, clock):
    kept, gone = _owned(ledger, clock, 2)
    ledger.delete_fact(gone, REVIEWER)
    assert _ids(ledger.owned_page(READ_MINE)) == [kept]


# --- owner delete / return: indistinguishable outcomes ------------------------------------------


def _trace(ledger, action):
    statements = []
    ledger._db.set_trace_callback(statements.append)
    try:
        result = action()
    finally:
        ledger._db.set_trace_callback(None)
    # The requested id is the only literal allowed to differ between outcomes.
    return result, [re.sub(r"fact_id='[^']*'", "fact_id='<id>'", re.sub(r"fact_[0-9a-f]{64}", "<id>", s))
                    for s in statements]


@pytest.mark.parametrize("method", ["delete_owned", "return_owned"])
def test_foreign_unassigned_and_missing_ids_run_identical_statements(ledger, clock, method):
    foreign = _confirmed(ledger, clock, "Synthetic fact of B")
    _assign(ledger, foreign, OTHER)
    unassigned = _confirmed(ledger, clock, "Synthetic household fact")
    returned = _confirmed(ledger, clock, "Synthetic returned by B")
    _assign(ledger, returned, OTHER)
    assert ledger.return_owned(returned.id, OTHER, MemoryOwnerActor("synthetic-home-b", OTHER))
    call = (lambda fid: ledger.delete_owned(fid, MINE)) if method == "delete_owned" else \
        (lambda fid: ledger.return_owned(fid, PERSON, MINE))
    before = _bytes(ledger)
    outcomes = {name: _trace(ledger, lambda fid=fid: call(fid)) for name, fid in
                (("missing", "fact_" + "0" * 64), ("foreign", foreign.id), ("unassigned", unassigned.id),
                 ("returned", returned.id), ("malformed", "not-a-fact"))}
    assert {name: result for name, (result, _) in outcomes.items()} == dict.fromkeys(outcomes, False)
    reference = outcomes["missing"][1]
    assert all(statements == reference for _, statements in outcomes.values())
    assert not any("facts WHERE id" in s for s in reference)  # nothing but the ownership probe
    assert _bytes(ledger) == before


def test_owner_delete_cascades_and_prunes_owner_rows(ledger, clock):
    fact = _confirmed(ledger, clock, "Synthetic owned with child")
    child = _derived(ledger, clock, fact)
    _assign(ledger, fact)
    assert ledger.delete_owned(fact.id, MINE) is True
    assert ledger.get(fact.id) is None and ledger.get(child.id) is None
    assert ledger._db.execute("SELECT COUNT(*) FROM fact_owners").fetchone()[0] == 0
    assert ledger._db.execute("SELECT COUNT(*) FROM fact_owner_revisions").fetchone()[0] == 0
    assert ledger.delete_owned(fact.id, MINE) is False
    assert ledger.owner_refs_cleared(frozenset({PERSON}))


def test_return_leaves_the_fact_in_devueltos_and_out_of_the_person_view(ledger, clock):
    fact = _owned(ledger, clock, 1)[0]
    assert ledger.return_owned(fact, PERSON, MINE) is True
    assert _ids(ledger.owned_page(READ_MINE)) == []
    page = ledger.returned_page(OPERATOR)
    assert _ids(page) == [fact] and page.items[0]["owner"] is None
    assert page.items[0]["owner_change"] == FactOwnerChange.OWNER_RETURNED.value
    assert page.items[0]["owner_revision"] == 2
    assert ledger.owned_counts(OPERATOR).returned == 1


@pytest.mark.parametrize("actor", [READ_MINE, MemoryReviewer("x"), OPERATOR, object(),
                                   MemoryOwnerActor("", PERSON), MemoryOwnerActor("s", "")])
def test_owner_writes_require_the_owner_actor(ledger, clock, actor):
    fact = _owned(ledger, clock, 1)[0]
    before = _bytes(ledger)
    with pytest.raises(PermissionError, match="memory_owner_actor_denied"):
        ledger.delete_owned(fact, actor)
    assert _bytes(ledger) == before


def test_retired_session_cannot_delete_or_read(ledger, clock):
    fact = _owned(ledger, clock, 1)[0]
    stale = MemoryOwnerActor("synthetic-home-a", PERSON, lambda: False)
    before = _bytes(ledger)
    with pytest.raises(PermissionError, match="memory_owner_actor_denied"):
        ledger.delete_owned(fact, stale)
    with pytest.raises(PermissionError, match="memory_owner_actor_denied"):
        ledger.owned_page(MemoryOwnerReader("synthetic-home-a", PERSON, lambda: False))
    assert _bytes(ledger) == before


# --- operator pages ---------------------------------------------------------------------------


def test_review_page_first_page_equals_the_u3_snapshot_lists(ledger, clock):
    for index in range(130):
        clock.value += 1
        (_automatic if index % 3 else _confirmed)(ledger, clock, f"Synthetic fact {index}")
        if index % 5 == 0:
            _proposed(ledger, clock, f"Synthetic proposal {index}")
    snapshot = ledger.snapshot()
    page = ledger.review_page()
    assert [f["id"] for f in page["facts"]] == [f["id"] for f in snapshot["facts"]]
    assert [f["id"] for f in page["proposals"]] == [f["id"] for f in snapshot["proposals"]]
    for key in ("observed_at", "legacy_policy", "pending_index_cleanup", "pending_index"):
        assert page[key] == snapshot[key]
    assert page["facts_next"] is not None and page["proposals_next"] is None


def test_review_pages_reach_every_fact_including_old_assigned_ones(ledger, clock):
    old = _owned(ledger, clock, 1, prefix="Synthetic old assigned")[0]
    for index in range(MEMORY_PAGE_LIMIT + 20):
        clock.value += 1
        _automatic(ledger, clock, f"Synthetic newer {index}")
    first = ledger.review_page()
    assert old not in [f["id"] for f in first["facts"]]
    seen, cursor = [f["id"] for f in first["facts"]], first["facts_next"]
    while cursor:
        page = ledger.review_page(facts_after=FactPageCursor.decode(cursor), limit=7)
        assert not {f["id"] for f in page["facts"]} & set(seen), "a page repeated an item"
        seen += [f["id"] for f in page["facts"]]
        cursor = page["facts_next"]
    assert len(seen) == len(set(seen)) == MEMORY_PAGE_LIMIT + 21 and seen[-1] == old
    owned = next(f for f in ledger.review_page(facts_after=FactPageCursor.decode(first["facts_next"]))["facts"]
                 if f["id"] == old)
    assert owned["owner"]["person_id"] == PERSON and owned["owner_revision"] == 1


def test_devueltos_queue_is_paginated_without_a_cap(ledger, clock):
    facts = _owned(ledger, clock, MEMORY_PAGE_LIMIT + 5)
    for fact in facts:
        assert ledger.return_owned(fact, PERSON, MINE)
    unassigned = _owned(ledger, clock, 1, prefix="Synthetic operator-unassigned")[0]
    ledger.unassign_owner(unassigned, 1, OPERATOR)
    seen = _walk(lambda **kw: ledger.returned_page(OPERATOR, **kw), MEMORY_PAGE_LIMIT)
    assert seen == list(reversed(facts)) and unassigned not in seen
    assert ledger.owned_counts(OPERATOR).returned == len(seen)
    with pytest.raises(PermissionError, match="memory_assign_denied"):
        ledger.returned_page(REVIEWER)


# --- F1: the post-failure proof rebuild never reads an aborted transaction ----------------------


class _AbortedRollback:
    """A connection whose rollback fails and leaves it inside the transaction."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        return getattr(self._real, name)

    @property
    def in_transaction(self):
        return True

    def rollback(self):
        raise sqlite3.OperationalError("synthetic rollback failure")


def test_f1_failed_rollback_never_publishes_cleared_from_uncommitted_state(ledger, clock):
    fact = _owned(ledger, clock, 1)[0]
    assert not ledger.owner_refs_cleared(frozenset({PERSON}))
    calls = []

    def retired(identities):
        # True for the pre-transaction checks; the post-body guard refuses after _delete ran.
        calls.append(identities)
        return len(calls) <= 3

    real = ledger._db
    ledger._db = _AbortedRollback(real)
    try:
        with pytest.raises((PermissionError, sqlite3.OperationalError)):
            ledger.clear_user_facts(frozenset({PERSON}), retired)
        assert len(calls) == 4
        # Inside the aborted transaction the owned fact is already deleted ...
        assert real.execute("SELECT COUNT(*) FROM fact_owners").fetchone()[0] == 0
        # ... but the proof must not be rebuilt from it: committed SQL still has the row.
        assert ledger.owner_refs_cleared(frozenset({PERSON})) is False
    finally:
        ledger._db = real
        real.rollback()
    with sqlite3.connect(ledger.path) as other:
        assert other.execute("SELECT person_id FROM fact_owners WHERE fact_id=?", (fact,)).fetchone() == (PERSON,)
    assert ledger.owner_refs_cleared(frozenset({PERSON})) is False
