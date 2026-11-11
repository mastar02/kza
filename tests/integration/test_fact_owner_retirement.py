"""U3b-1b: account removal deletes the person's assigned memories (D7).

Real DashboardAuthority, UserManager, voice registry and privacy FactLedger
(one SQLite file), driven through DELETE /api/admin/users/{id}.
"""

import asyncio
import sqlite3
import threading

import pytest
import pytest_asyncio

from src.dashboard.authority import DashboardAuthority
from src.main import _prune_retired_memory_owners
from src.memory.fact_ledger import MemoryAssigner, MemoryReviewer
from tests.integration.test_voice_person_api import people_web as _people_web
from tests.integration.test_voice_person_links import people_graph as _people_graph
from tests.mocks.memory_sources import memory_source

people_graph = _people_graph
people_web = _people_web
OPERATOR = MemoryAssigner("synthetic-operator")
REVIEWER = MemoryReviewer("synthetic-reviewer")


def production_proof(g):
    """The src/main.py retirement_proof lambda over this graph's owners."""
    return lambda identities: (g.authority.users_retired(identities)
                               and g.links.user_links_cleared(identities)
                               and g.privacy.ledger.owner_refs_cleared(identities))


@pytest_asyncio.fixture
async def owners(people_web):
    g = people_web
    g.ledger = g.privacy.ledger
    g.manager._retirement_proof = production_proof(g)
    g.wired = {"retired": g.authority.memory_owner_retired, "active": g.authority.memory_owner_active}
    return g


async def confirmed(g, content):
    def create():
        proposal = g.ledger.store(content, "fact", 1.0, (memory_source(now=g.clock.now, ttl=3600),), proposed=True)
        return g.ledger.confirm(proposal.id, REVIEWER)
    return await g.privacy.acall(create)


async def assign(g, fact, person, owner_revision=0, **callables):
    return await g.privacy.acall(lambda: g.ledger.assign_owner(
        fact.id, person, fact.revision, owner_revision, OPERATOR, **{**g.wired, **callables}))


def rows(g, sql, *args):
    with sqlite3.connect(g.database) as db:
        return db.execute(sql, args).fetchall()


def owner_of(g, fact_id):
    found = rows(g, "SELECT person_id FROM fact_owners WHERE fact_id=?", fact_id)
    return found[0][0] if found else None


def state_of(g, fact_id):
    return rows(g, "SELECT state, content IS NULL FROM facts WHERE id=?", fact_id)[0]


def hold_authority(monkeypatch, authority):
    entered, release = threading.Event(), threading.Event()
    transaction = authority._transaction

    def held(action, args, guard):
        if action == "retire_user":
            entered.set()
            assert release.wait(5)
        return transaction(action, args, guard)

    monkeypatch.setattr(authority, "_transaction", held)
    return entered, release


# --- wired account callables ----------------------------------------------------------------


async def test_wired_callables_are_exact_for_each_account_state(owners, monkeypatch):
    g = owners
    a, b, unknown = frozenset({g.a}), frozenset({g.b}), "user_" + "f" * 32
    assert (g.authority.memory_owner_retired(a), g.authority.memory_owner_active(g.a)) == (False, True)
    assert (g.authority.memory_owner_retired(frozenset({unknown})), g.authority.memory_owner_active(unknown)) == (
        False, False)
    for invalid in ({g.a}, [g.a], frozenset()):
        assert g.authority.memory_owner_retired(invalid) is None
    assert g.authority.memory_owner_active(None) is None
    entered, release = hold_authority(monkeypatch, g.authority)
    pending = asyncio.create_task(g.authority.retire_user(g.root, g.a))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        # In progress: users_retired still says False (A16); the wired pair refuses.
        assert g.authority.users_retired(a) is False
        assert (g.authority.memory_owner_retired(a), g.authority.memory_owner_active(g.a)) == (True, False)
        assert (g.authority.memory_owner_retired(b), g.authority.memory_owner_active(g.b)) == (False, True)
    finally:
        release.set()
    assert await pending
    assert g.authority.users_retired(a)
    assert (g.authority.memory_owner_retired(a), g.authority.memory_owner_active(g.a)) == (True, False)


async def test_unusable_authority_is_unknown_never_not_retired(tmp_path):
    authority = DashboardAuthority(tmp_path / "auth.db", expected_token=lambda: "synthetic-root")
    person = frozenset({"user_" + "a" * 32})
    assert authority.users_retired(person) is False
    assert authority.memory_owner_retired(person) is None
    assert authority.memory_owner_active("user_" + "a" * 32) is None
    await authority.initialize()
    assert authority.memory_owner_retired(person) is False
    await authority.aclose()
    assert authority.memory_owner_retired(person) is None
    assert authority.memory_owner_active("user_" + "a" * 32) is None


async def test_assignment_with_wired_callables_refuses_unusable_retiring_and_unknown(owners, monkeypatch):
    g = owners
    fact = await confirmed(g, "Synthetic household fact")
    before = rows(g, "SELECT * FROM fact_owners"), rows(g, "SELECT * FROM fact_owner_revisions")
    with pytest.raises(PermissionError, match="memory_owner_inactive"):
        await assign(g, fact, "user_" + "f" * 32)
    with monkeypatch.context() as changes:
        changes.setattr(g.authority, "_ready", False)
        with pytest.raises(PermissionError, match="memory_owner_authority_unavailable"):
            await assign(g, fact, g.a)
    entered, release = hold_authority(monkeypatch, g.authority)
    pending = asyncio.create_task(g.authority.retire_user(g.root, g.a))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        with pytest.raises(PermissionError, match="memory_owner_retired"):
            await assign(g, fact, g.a)
    finally:
        release.set()
    await pending
    assert (rows(g, "SELECT * FROM fact_owners"), rows(g, "SELECT * FROM fact_owner_revisions")) == before
    assert (await assign(g, fact, g.b)).owner_revision == 1


async def test_wired_callables_never_reenter_the_ledger_or_wait_for_the_catalog(owners):
    g = owners
    fact = await confirmed(g, "Synthetic fact")
    inside, statements = [], []

    def traced(name):
        original = g.wired[name]

        def call(value):
            inside.append(name)
            count = len(statements)
            try:
                return original(value)
            finally:
                assert len(statements) == count, "account callable executed SQL on the ledger"
        return call

    entered, release = threading.Event(), threading.Event()

    def hold_catalog():
        with g.manager._lock:
            entered.set()
            assert release.wait(5)

    holder = asyncio.create_task(asyncio.to_thread(hold_catalog))
    try:
        assert await asyncio.to_thread(entered.wait, 3)

        def run():
            g.ledger._db.set_trace_callback(statements.append)
            try:
                return g.ledger.assign_owner(fact.id, g.a, fact.revision, 0, OPERATOR,
                                             retired=traced("retired"), active=traced("active"))
            finally:
                g.ledger._db.set_trace_callback(None)
        receipt = await asyncio.wait_for(g.privacy.acall(run), 3)
    finally:
        release.set()
        await holder
    assert receipt.owner_revision == 1 and inside.count("retired") == inside.count("active") >= 3
    assert statements


# --- retirement cascade through the admin route ------------------------------------------------


async def test_account_delete_deletes_the_persons_assigned_memories(owners):
    g = owners
    mine = [await confirmed(g, "Synthetic fact of A %d" % n) for n in range(3)]
    other = await confirmed(g, "Synthetic fact of B")
    household = await confirmed(g, "Synthetic household fact")
    for fact in mine:
        await assign(g, fact, g.a)
    await assign(g, other, g.b)
    assert not g.ledger.owner_refs_cleared(frozenset({g.a}))
    response = await g.client.delete("/api/admin/users/" + g.a)
    assert response.status_code == 200
    assert g.manager.get_user(g.a) is None
    for fact in mine:
        assert state_of(g, fact.id) == ("deleted", 1) and owner_of(g, fact.id) is None
        assert rows(g, "SELECT * FROM fact_owner_revisions WHERE fact_id=?", fact.id) == []
    assert owner_of(g, other.id) == g.b and state_of(g, other.id) == ("confirmed", 0)
    assert state_of(g, household.id) == ("confirmed", 0)
    assert rows(g, "SELECT COUNT(*) FROM fact_owners WHERE person_id=?", g.a) == [(0,)]
    assert {fact.id for fact in mine} <= {r[0] for r in rows(g, "SELECT fact_id FROM index_jobs WHERE operation='delete'")}
    assert g.ledger.owner_refs_cleared(frozenset({g.a})) and not g.ledger.owner_refs_cleared(frozenset({g.b}))


async def test_catalog_removal_is_refused_while_owner_rows_or_proof_remain(owners):
    g = owners
    fact = await confirmed(g, "Synthetic fact of A")
    await assign(g, fact, g.a)
    await g.authority.retire_user(g.root, g.a)
    before = g.manager.data_path.read_bytes(), g.manager._version
    with pytest.raises(PermissionError, match="account_retirement_required"):
        await asyncio.to_thread(g.manager.remove_user, g.a)
    assert (g.manager.data_path.read_bytes(), g.manager._version) == before
    assert await g.privacy.acall(g.ledger.clear_user_facts, frozenset({g.a}), g.authority.users_retired) == 1
    g.ledger._fact_owner_proof.begin()  # an owner writer in flight withdraws the proof for everyone
    try:
        with pytest.raises(PermissionError, match="account_retirement_required"):
            await asyncio.to_thread(g.manager.remove_user, g.a)
        assert g.manager.get_user(g.a) is not None
    finally:
        g.ledger._fact_owner_proof.finish(True, frozenset())
    assert (await asyncio.to_thread(g.manager.remove_user, g.a))[0]


@pytest.mark.parametrize("stage", ["cleanup_before", "cleanup_after", "catalog_replace"])
async def test_partial_removal_reports_retired_and_retry_completes(owners, monkeypatch, stage):
    g = owners
    fact = await confirmed(g, "Synthetic fact of A")
    await assign(g, fact, g.a)
    original = g.ledger.clear_user_facts

    def clear(*args):
        if stage == "cleanup_before":
            raise OSError("synthetic cleanup unavailable before commit")
        original(*args)
        raise OSError("synthetic cleanup result unavailable after commit")

    before = g.manager.data_path.read_bytes(), g.manager._version
    with monkeypatch.context() as changes:
        if stage.startswith("cleanup"):
            changes.setattr(g.ledger, "clear_user_facts", clear)
        else:
            def replace(*args):
                raise OSError("synthetic catalog replace unavailable")
            changes.setattr("src.users.user_manager.os.replace", replace)
        response = await g.client.delete("/api/admin/users/" + g.a)
    assert response.status_code == 503 and response.json()["detail"]["credentials_retired"] is True
    assert g.authority.users_retired(frozenset({g.a})) and g.manager.get_user(g.a) is not None
    assert (g.manager.data_path.read_bytes(), g.manager._version) == before
    assert (owner_of(g, fact.id) == g.a) is (stage == "cleanup_before")
    assert (await g.client.delete("/api/admin/users/" + g.a)).status_code == 200
    assert g.manager.get_user(g.a) is None and owner_of(g, fact.id) is None
    assert state_of(g, fact.id) == ("deleted", 1)


async def test_unproven_retirement_maps_to_unavailable_and_deletes_nothing(owners, monkeypatch):
    g = owners
    fact = await confirmed(g, "Synthetic fact of A")
    await assign(g, fact, g.a)
    original = g.ledger.clear_user_facts
    monkeypatch.setattr(g.ledger, "clear_user_facts", lambda ids, retired: original(ids, lambda ids: False))
    response = await g.client.delete("/api/admin/users/" + g.a)
    assert response.status_code == 503
    assert response.json()["detail"] == {"code": "user_catalog_unavailable", "credentials_retired": True}
    assert owner_of(g, fact.id) == g.a and g.manager.get_user(g.a) is not None
    monkeypatch.undo()
    assert (await g.client.delete("/api/admin/users/" + g.a)).status_code == 200


async def test_crash_after_retirement_is_completed_by_the_startup_prune(owners):
    """Retired durably, then the process died before any cleanup ran."""
    g = owners
    fact, other = await confirmed(g, "Synthetic fact of A"), await confirmed(g, "Synthetic fact of B")
    await assign(g, fact, g.a)
    await assign(g, other, g.b)
    await g.authority.retire_user(g.root, g.a)
    assert await _prune_retired_memory_owners(g.privacy, g.authority) == 1
    assert owner_of(g, fact.id) is None and state_of(g, fact.id) == ("deleted", 1)
    assert owner_of(g, other.id) == g.b
    assert await _prune_retired_memory_owners(g.privacy, g.authority) == 0
    assert (await g.client.delete("/api/admin/users/" + g.a)).status_code == 200


async def test_startup_prune_with_unusable_authority_keeps_rows(owners, tmp_path):
    g = owners
    fact = await confirmed(g, "Synthetic fact of A")
    await assign(g, fact, g.a)
    await g.authority.retire_user(g.root, g.a)
    unusable = DashboardAuthority(tmp_path / "other-auth.db", expected_token=lambda: "synthetic-root")
    assert await _prune_retired_memory_owners(g.privacy, unusable) == 0
    assert owner_of(g, fact.id) == g.a


async def test_startup_prune_failure_is_logged_and_keeps_rows(owners, monkeypatch, caplog):
    g = owners
    fact = await confirmed(g, "Synthetic fact of A")
    await assign(g, fact, g.a)
    await g.authority.retire_user(g.root, g.a)

    def broken(retired, *, probe=None):  # U3b-2 R1: the boot prune also passes the tri-state probe
        raise sqlite3.OperationalError("synthetic private detail")

    monkeypatch.setattr(g.ledger, "prune_retired_owners", broken)
    assert await _prune_retired_memory_owners(g.privacy, g.authority) == 0
    assert owner_of(g, fact.id) == g.a
    assert "OperationalError" in caplog.text and "synthetic private detail" not in caplog.text


# --- assignment vs retirement, both orders -------------------------------------------------


async def test_race_assignment_commits_before_removal(owners):
    g = owners
    fact = await confirmed(g, "Synthetic fact")
    await assign(g, fact, g.a)
    assert (await g.client.delete("/api/admin/users/" + g.a)).status_code == 200
    assert owner_of(g, fact.id) is None and state_of(g, fact.id) == ("deleted", 1)


async def test_race_removal_starts_before_assignment(owners, monkeypatch):
    g = owners
    fact = await confirmed(g, "Synthetic fact")
    entered, release = hold_authority(monkeypatch, g.authority)
    removal = asyncio.create_task(g.client.delete("/api/admin/users/" + g.a))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        with pytest.raises(PermissionError, match="memory_owner_retired"):
            await assign(g, fact, g.a)
    finally:
        release.set()
    assert (await removal).status_code == 200
    assert owner_of(g, fact.id) is None and state_of(g, fact.id) == ("confirmed", 0)
    with pytest.raises(PermissionError, match="memory_owner_retired"):
        await assign(g, fact, g.a)
    assert rows(g, "SELECT COUNT(*) FROM fact_owners") == [(0,)]


async def test_race_assignment_in_flight_when_removal_retires(owners, monkeypatch):
    """The assignment passed its first guards; retirement lands before its commit."""
    g = owners
    fact = await confirmed(g, "Synthetic fact")
    inside, release = threading.Event(), threading.Event()
    original = g.ledger._owner_state

    def held(db, fact_id):
        inside.set()
        assert release.wait(5)
        return original(db, fact_id)

    monkeypatch.setattr(g.ledger, "_owner_state", held)
    assignment = asyncio.create_task(assign(g, fact, g.a))
    removal = None
    try:
        assert await asyncio.to_thread(inside.wait, 3)
        removal = asyncio.create_task(g.client.delete("/api/admin/users/" + g.a))
        async with asyncio.timeout(3):
            while g.authority.memory_owner_retired(frozenset({g.a})) is not True:
                await asyncio.sleep(.001)
    finally:
        release.set()
    with pytest.raises(PermissionError, match="memory_owner_retired"):
        await assignment
    assert (await removal).status_code == 200
    assert g.manager.get_user(g.a) is None
    assert owner_of(g, fact.id) is None and state_of(g, fact.id) == ("confirmed", 0)
    assert rows(g, "SELECT COUNT(*) FROM fact_owner_revisions") == [(0,)]
    assert g.ledger.owner_refs_cleared(frozenset({g.a}))
