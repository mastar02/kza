"""U3b-1b ledger side: owner proof, D7 retirement cascade, startup prune.

Every case runs on a real FactLedger file. Refusals compare the database file
bytes before and after, so 'nothing changed' is literal.
"""

import ast
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from uuid import uuid4

import pytest

from src.memory.fact_ledger import (
    FactDependency, FactLedger, MemoryAssigner, MemoryOrigin, MemoryOwnerActor, MemoryReviewer, MemorySource,
)

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


def _source(clock, *, dependencies=()):
    return MemorySource(str(uuid4()), str(uuid4()), clock() - 1, clock() + 3600, MemoryOrigin.VOICE,
                        tuple(dependencies))


def _automatic(ledger, clock, content="Synthetic automatic fact", **kwargs):
    return ledger.store(content, "fact", .9, (_source(clock, **kwargs),))


def _confirmed(ledger, clock, content="Synthetic confirmed fact"):
    return ledger.confirm(ledger.store(content, "fact", 1.0, (_source(clock),), proposed=True).id, REVIEWER)


def _derived(ledger, clock, parent, content="Synthetic derived fact"):
    return _automatic(ledger, clock, content,
                      dependencies=(FactDependency(parent.id, parent.revision, parent.expires_at),))


def _assign(ledger, fact, person=PERSON, owner_revision=0, **kwargs):
    return ledger.assign_owner(fact.id, person, fact.revision, owner_revision, OPERATOR, **{**ALLOW, **kwargs})


def _bytes(ledger):
    return Path(ledger.path).read_bytes()


def _row(ledger, table, fact_id):
    row = ledger._db.execute("SELECT * FROM " + table + " WHERE " + ("id" if table == "facts" else "fact_id")
                             + "=?", (fact_id,)).fetchone()
    return None if row is None else dict(row)


def _retired_only(*people):
    targets = frozenset(people)
    return lambda identities: type(identities) is frozenset and bool(identities) and identities <= targets


def _owned_world(ledger, clock):
    """P owns a confirmed fact with a derived dependent and an assigned automatic fact;
    OTHER owns one; one fact was returned by P ('Devueltos'); one is household."""
    mine = _confirmed(ledger, clock, "Synthetic fact of P")
    dependent = _derived(ledger, clock, mine, "Synthetic dependent of P's fact")
    automatic = _automatic(ledger, clock, "Synthetic automatic of P")
    others = _confirmed(ledger, clock, "Synthetic fact of OTHER")
    returned = _confirmed(ledger, clock, "Synthetic returned fact")
    household = _confirmed(ledger, clock, "Synthetic household fact")
    _assign(ledger, mine)
    _assign(ledger, automatic)
    _assign(ledger, others, OTHER)
    _assign(ledger, returned)
    assert ledger.return_owned(returned.id, PERSON, MemoryOwnerActor("synthetic-home", PERSON))
    return {"mine": mine.id, "dependent": dependent.id, "automatic": automatic.id, "others": others.id,
            "returned": returned.id, "household": household.id}


# --- owner proof ----------------------------------------------------------------------------


def test_proof_is_published_from_sql_at_open_and_withdrawn_on_close(tmp_path, clock):
    path = tmp_path / "privacy.db"
    ledger = FactLedger(str(path), clock=clock)
    assert not ledger.owner_refs_cleared(frozenset({PERSON}))
    ledger.initialize()
    assert ledger.owner_refs_cleared(frozenset({PERSON, OTHER}))
    _assign(ledger, _confirmed(ledger, clock))
    assert not ledger.owner_refs_cleared(frozenset({PERSON}))
    assert ledger.owner_refs_cleared(frozenset({OTHER}))
    ledger.close()
    assert not ledger.owner_refs_cleared(frozenset({OTHER}))
    reopened = FactLedger(str(path), clock=clock)
    reopened.initialize()
    try:
        assert not reopened.owner_refs_cleared(frozenset({PERSON}))
        assert reopened.owner_refs_cleared(frozenset({OTHER}))
        assert reopened.owner_refs_cleared(frozenset())
    finally:
        reopened.close()


@pytest.mark.parametrize("identities", [{PERSON}, [PERSON], frozenset({""}), frozenset({"x\n"}),
                                        frozenset({"user_%d" % n for n in range(257)})])
def test_proof_refuses_malformed_identity_sets(ledger, identities):
    assert ledger.owner_refs_cleared(identities) is False


def test_assignment_withdraws_the_proof_while_in_flight(ledger, clock):
    fact = _confirmed(ledger, clock)
    seen = []

    def active(person):
        seen.append(ledger._fact_owner_proof.snapshot.available)
        return True

    _assign(ledger, fact, active=active)
    assert seen and not any(seen)
    assert ledger._fact_owner_proof.snapshot.available


def test_refused_assignment_republishes_the_actual_sql_state(ledger, clock):
    fact = _confirmed(ledger, clock)
    with pytest.raises(ValueError, match="memory_owner_conflict"):
        _assign(ledger, fact, owner_revision=3)
    assert ledger.owner_refs_cleared(frozenset({PERSON}))
    with pytest.raises(PermissionError, match="memory_owner_retired"):
        _assign(ledger, fact, retired=lambda ids: True)
    assert ledger.owner_refs_cleared(frozenset({PERSON}))


@pytest.mark.parametrize("stage", ["before", "after"])
def test_uncertain_commit_is_resolved_from_actual_sql(ledger, clock, monkeypatch, stage):
    fact = _confirmed(ledger, clock)
    original = ledger._transaction

    @contextmanager
    def transaction(guard=None):
        with original(guard=guard) as db:
            yield db
            if stage == "before":
                raise OSError("synthetic error before actual commit")
        raise OSError("synthetic error after actual commit")

    monkeypatch.setattr(ledger, "_transaction", transaction)
    with pytest.raises(OSError):
        _assign(ledger, fact)
    monkeypatch.undo()
    owned = stage == "after"
    assert (_row(ledger, "fact_owners", fact.id) is not None) is owned
    assert ledger.owner_refs_cleared(frozenset({PERSON})) is not owned


def test_proof_stays_unavailable_when_the_state_cannot_be_read(ledger, clock, monkeypatch):
    fact = _confirmed(ledger, clock)
    import src.memory.fact_ledger as module

    def broken(db):
        raise sqlite3.OperationalError("synthetic unreadable owner rows")

    monkeypatch.setattr(module, "_owner_refs", broken)
    with pytest.raises(sqlite3.OperationalError):
        _assign(ledger, fact)
    monkeypatch.undo()
    assert not ledger.owner_refs_cleared(frozenset({OTHER}))
    assert _row(ledger, "fact_owners", fact.id) is None
    assert ledger.prune_retired_owners(_retired_only()) == 0
    assert ledger.owner_refs_cleared(frozenset({OTHER, PERSON}))


# --- retirement cascade ---------------------------------------------------------------------


def test_cascade_deletes_only_the_retired_persons_owned_facts(ledger, clock):
    ids = _owned_world(ledger, clock)
    ledger._db.execute("DELETE FROM index_jobs")
    assert ledger.clear_user_facts(frozenset({PERSON}), _retired_only(PERSON)) == 3
    for key in ("mine", "dependent", "automatic"):
        row = _row(ledger, "facts", ids[key])
        assert (row["state"], row["content"]) == ("deleted", None)
        assert _row(ledger, "fact_owners", ids[key]) is None
        assert _row(ledger, "fact_owner_revisions", ids[key]) is None
        assert _row(ledger, "fact_sources", ids[key]) is None
    for key in ("others", "returned", "household"):
        assert _row(ledger, "facts", ids[key])["state"] == "confirmed"
    assert _row(ledger, "fact_owners", ids["others"])["person_id"] == OTHER
    assert _row(ledger, "fact_owner_revisions", ids["returned"])["last_change"] == "owner_returned"
    jobs = {(job["fact_id"], job["operation"]) for job in ledger.jobs(100)}
    assert jobs == {(ids[key], "delete") for key in ("mine", "dependent", "automatic")}
    assert ledger.owner_refs_cleared(frozenset({PERSON}))
    assert not ledger.owner_refs_cleared(frozenset({OTHER}))


def test_cascade_is_idempotent(ledger, clock):
    _owned_world(ledger, clock)
    assert ledger.clear_user_facts(frozenset({PERSON}), _retired_only(PERSON)) == 3
    before = _bytes(ledger)
    assert ledger.clear_user_facts(frozenset({PERSON}), _retired_only(PERSON)) == 0
    assert _bytes(ledger) == before
    assert ledger.owner_refs_cleared(frozenset({PERSON}))


def test_cascade_of_a_person_without_rows_changes_nothing(ledger, clock):
    _owned_world(ledger, clock)
    before = _bytes(ledger)
    assert ledger.clear_user_facts(frozenset({"user_" + "c" * 32}), lambda ids: True) == 0
    assert _bytes(ledger) == before


@pytest.mark.parametrize("retired", [
    lambda ids: False, lambda ids: None, lambda ids: 1, lambda ids: "True", "not callable", None,
    lambda ids: (_ for _ in ()).throw(RuntimeError("synthetic authority failure")),
], ids=["false", "none", "one", "string", "non_callable", "missing", "raises"])
def test_cascade_requires_an_exact_durable_retirement_proof(ledger, clock, retired):
    _owned_world(ledger, clock)
    before = _bytes(ledger)
    with pytest.raises(PermissionError, match="memory_owner_retirement_required"):
        ledger.clear_user_facts(frozenset({PERSON}), retired)
    assert _bytes(ledger) == before
    assert not ledger.owner_refs_cleared(frozenset({PERSON}))


@pytest.mark.parametrize("flip_at", [2, 3, 4, 5])
def test_cascade_rechecks_retirement_inside_the_transaction(ledger, clock, flip_at):
    _owned_world(ledger, clock)
    calls = []

    def retired(identities):
        calls.append(identities)
        return len(calls) < flip_at

    before = _bytes(ledger)
    with pytest.raises(PermissionError, match="memory_owner_retirement_required"):
        ledger.clear_user_facts(frozenset({PERSON}), retired)
    assert _bytes(ledger) == before
    assert len(calls) == flip_at


@pytest.mark.parametrize("targets, error", [
    (frozenset(), ValueError), ({PERSON}, ValueError), (frozenset({"user_%d" % n for n in range(257)}), ValueError),
    (frozenset({"bad\x00id"}), ValueError),
])
def test_cascade_rejects_malformed_targets(ledger, clock, targets, error):
    _owned_world(ledger, clock)
    before = _bytes(ledger)
    with pytest.raises(error):
        ledger.clear_user_facts(targets, lambda ids: True)
    assert _bytes(ledger) == before


def test_crash_inside_the_cascade_rolls_back_every_deletion_and_rerun_completes(ledger, clock, monkeypatch):
    ids = _owned_world(ledger, clock)
    original, calls = ledger._delete, []

    def delete(db, fact_id):
        calls.append(fact_id)
        if len(calls) == 2:
            raise OSError("synthetic crash between cascade steps")
        return original(db, fact_id)

    before = _bytes(ledger)
    monkeypatch.setattr(ledger, "_delete", delete)
    with pytest.raises(OSError):
        ledger.clear_user_facts(frozenset({PERSON}), _retired_only(PERSON))
    monkeypatch.undo()
    assert len(calls) == 2 and _bytes(ledger) == before
    assert _row(ledger, "facts", ids["mine"])["state"] == "confirmed"
    assert not ledger.owner_refs_cleared(frozenset({PERSON}))
    assert ledger.clear_user_facts(frozenset({PERSON}), _retired_only(PERSON)) == 3
    assert ledger.owner_refs_cleared(frozenset({PERSON}))


def test_cascade_refuses_unknown_owner_values(ledger, clock):
    fact = _confirmed(ledger, clock)
    _assign(ledger, fact)
    ledger._db.execute("UPDATE fact_owners SET method='synthetic' WHERE fact_id=?", (fact.id,))
    before = _bytes(ledger)
    with pytest.raises(RuntimeError, match="fact_owner_value_unknown"):
        ledger.clear_user_facts(frozenset({PERSON}), lambda ids: True)
    assert _bytes(ledger) == before


# --- startup prune --------------------------------------------------------------------------


def test_startup_prune_deletes_only_durably_retired_owners_and_is_idempotent(tmp_path, clock):
    path = tmp_path / "privacy.db"
    ledger = FactLedger(str(path), clock=clock)
    ledger.initialize()
    ids = _owned_world(ledger, clock)
    ledger.close()
    restarted = FactLedger(str(path), clock=clock)
    restarted.initialize()
    try:
        asked = []

        def retired(identities):
            asked.append(identities)
            return identities == frozenset({PERSON})

        assert restarted.prune_retired_owners(retired) == 3
        assert frozenset({PERSON}) in asked and frozenset({OTHER}) in asked
        assert _row(restarted, "facts", ids["mine"])["state"] == "deleted"
        assert _row(restarted, "fact_owners", ids["others"])["person_id"] == OTHER
        assert _row(restarted, "fact_owner_revisions", ids["returned"]) is not None
        assert restarted.owner_refs_cleared(frozenset({PERSON}))
        before = _bytes(restarted)
        assert restarted.prune_retired_owners(retired) == 0
        assert _bytes(restarted) == before
        assert not set(dict(restarted._db.execute("SELECT person_id, 1 FROM fact_owners"))) & {PERSON}
    finally:
        restarted.close()


@pytest.mark.parametrize("retired", [
    lambda ids: False, lambda ids: None,
    lambda ids: (_ for _ in ()).throw(RuntimeError("synthetic authority unusable")),
], ids=["not_retired", "unknown", "raises"])
def test_startup_prune_never_guesses(ledger, clock, retired):
    _owned_world(ledger, clock)
    before = _bytes(ledger)
    assert ledger.prune_retired_owners(retired) == 0
    assert _bytes(ledger) == before


def test_startup_prune_requires_a_callable(ledger):
    with pytest.raises(PermissionError, match="memory_owner_retirement_required"):
        ledger.prune_retired_owners(None)


def test_restored_owner_rows_of_a_retired_person_are_pruned_at_startup(tmp_path, clock):
    """A ledger image from before the retirement comes back with P's rows."""
    path = tmp_path / "privacy.db"
    ledger = FactLedger(str(path), clock=clock)
    ledger.initialize()
    _owned_world(ledger, clock)
    ledger.close()
    image = path.read_bytes()
    ledger.initialize()
    assert ledger.clear_user_facts(frozenset({PERSON}), _retired_only(PERSON)) == 3
    ledger.close()
    path.write_bytes(image)
    ledger.initialize()
    try:
        assert not ledger.owner_refs_cleared(frozenset({PERSON}))
        assert ledger.prune_retired_owners(_retired_only(PERSON)) == 3
        assert ledger.owner_refs_cleared(frozenset({PERSON}))
    finally:
        ledger.close()


# --- assignment vs retirement at ledger level ------------------------------------------------


def test_assignment_after_cascade_is_refused_for_the_retired_person(ledger, clock):
    _owned_world(ledger, clock)
    retired = _retired_only(PERSON)
    assert ledger.clear_user_facts(frozenset({PERSON}), retired) == 3
    fact = _confirmed(ledger, clock, "Synthetic late fact")
    before = _bytes(ledger)
    with pytest.raises(PermissionError, match="memory_owner_retired"):
        _assign(ledger, fact, retired=retired)
    assert _bytes(ledger) == before
    assert ledger.owner_refs_cleared(frozenset({PERSON}))


# --- production wiring pins -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
_PARTICIPANT = {"clear_user_facts", "prune_retired_owners", "owner_refs_cleared",
                "memory_owner_retired", "memory_owner_active"}


def _attribute_uses(names):
    found = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path in (ROOT / "src/memory/fact_ledger.py", ROOT / "src/dashboard/authority.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr in names:
                found.setdefault(str(path.relative_to(ROOT)), []).append(node.attr)
    return found


def test_retirement_participant_has_exactly_the_wired_production_callers():
    assert _attribute_uses(_PARTICIPANT) == {
        "src/dashboard/admin.py": ["clear_user_facts"],
        "src/dashboard/memory_api.py": ["memory_owner_retired", "memory_owner_active"],  # U3b-2 assign route
        # U3b-2 R1: the boot prune passes memory_owner_retired as the tri-state probe.
        "src/main.py": ["memory_owner_retired", "prune_retired_owners", "memory_owner_retired", "owner_refs_cleared"],
        "src/memory/privacy_services.py": ["prune_retired_owners"],  # F2 maintenance-tick retry
    }


def _assign_calls_with_unwired_callables(source):
    bad = []
    for node in ast.walk(ast.parse(source)):
        # Direct calls and worker hand-offs such as service.acall(ledger.assign_owner, ..., retired=..., active=...).
        if isinstance(node, ast.Call) and any(isinstance(target, ast.Attribute) and target.attr == "assign_owner"
                                              for target in (node.func, *node.args)):
            keywords = {k.arg: k.value for k in node.keywords}
            wired = all(isinstance(keywords.get(name), ast.Attribute) and keywords[name].attr == "memory_owner_" + name
                        for name in ("retired", "active"))
            if not wired:
                bad.append(node.lineno)
    return bad


def test_any_production_assignment_passes_the_wired_account_callables():
    """Forward pin for U3b-2: 1b adds no caller, but any caller must use the authority pair."""
    offenders = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path == ROOT / "src/memory/fact_ledger.py":
            continue
        bad = _assign_calls_with_unwired_callables(path.read_text(encoding="utf-8"))
        if bad:
            offenders[str(path.relative_to(ROOT))] = bad
    assert offenders == {}


def test_the_wiring_scan_detects_unwired_callables():
    wired = "ledger.assign_owner(f, p, 1, 0, a, retired=auth.memory_owner_retired, active=auth.memory_owner_active)\n"
    assert _assign_calls_with_unwired_callables(wired) == []
    assert _assign_calls_with_unwired_callables(wired.replace("auth.memory_owner_retired", "auth.users_retired"))
    assert _assign_calls_with_unwired_callables(wired.replace("auth.memory_owner_active", "lambda p: True"))
    assert _assign_calls_with_unwired_callables("ledger.assign_owner(f, p, 1, 0, a, **checks)\n")
    handoff = "await run(svc.ledger.assign_owner, f, p, 1, 0, a, retired=auth.memory_owner_retired, active=auth.memory_owner_active)\n"
    assert _assign_calls_with_unwired_callables(handoff) == []
    assert _assign_calls_with_unwired_callables(handoff.replace("auth.memory_owner_active", "auth.users_linkable"))


def test_the_assign_route_is_the_only_production_assignment_and_it_is_wired():
    """U3b-2: the forward rule is no longer vacuous; exactly one assignment call exists."""
    calls = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        if path == ROOT / "src/memory/fact_ledger.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and any(isinstance(t, ast.Attribute) and t.attr == "assign_owner"
                                                  for t in (node.func, *node.args)):
                calls.append(str(path.relative_to(ROOT)))
    assert calls == ["src/dashboard/memory_api.py"]


def test_account_delete_orders_retirement_memories_links_then_catalog():
    """G3/N3 (U3b-4a-1): memories (pending voice proposals included) go before the voice links."""
    tree = ast.parse((ROOT / "src/dashboard/admin.py").read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "_delete_account")
    steps = {"retire_user", "clear_user_links", "clear_user_facts", "remove_user"}
    lines = {node.attr: node.lineno for node in ast.walk(function)
             if isinstance(node, ast.Attribute) and node.attr in steps}
    assert sorted(lines, key=lines.get) == ["retire_user", "clear_user_facts", "clear_user_links", "remove_user"]
    assert function.args.args and len(function.args.args) == 7  # output_registration_v2 _delete_step shape


def test_boot_prune_runs_right_after_the_authority_is_loaded():
    tree = ast.parse((ROOT / "src/main.py").read_text(encoding="utf-8"))
    build = next(node for node in ast.walk(tree)
                 if isinstance(node, ast.AsyncFunctionDef) and node.name == "build_application")
    awaited = [ast.unparse(node.value) for node in sorted(
        (node for node in ast.walk(build) if isinstance(node, ast.Await)), key=lambda n: (n.lineno, n.col_offset))]
    position = awaited.index("dashboard_authority.initialize()")
    assert awaited[position + 1] == "_prune_retired_memory_owners(privacy_services, dashboard_authority)"
