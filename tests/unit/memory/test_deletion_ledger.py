"""U3b-0d3: the content-free, append-only deletion ledger (design v3 N3, reviews D3/E3).

Every case runs on real SQLite files. The coverage test enumerates the
deletion entry points from the source (AST) and proves, by behaviour, that
each one appends exactly the rows its request creates and nothing on replay.
"""

import ast
import asyncio
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import pytest

from scripts import stage_runtime as stage
from src.memory import deletion_ledger as ledger_module
from src.memory.deletion_ledger import (
    DeletionKind, DeletionLedgerError, append_deletion, initialize_deletion_ledger,
    resurrected, row_sha256, subject_digest, verify_chain,
)
from src.memory.fact_ledger import (
    FactDependency, FactLedger, MemoryAssigner, MemoryOrigin, MemoryOwnerActor, MemoryReviewer, MemorySource,
    MemoryVoiceConsentActor,
)

REPO = Path(__file__).resolve().parents[3]
NOW = 2_000_000_000.0
REVIEWER = MemoryReviewer("synthetic-reviewer")
OPERATOR = MemoryAssigner("synthetic-operator")
ALLOW = {"retired": lambda ids: False, "active": lambda person: True}
PERSON = "user_" + "a" * 32
PROFILE = "00000000-0000-4000-8000-0000000000d1"


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


def _source(clock, conversation=None, *, dependencies=()):
    return MemorySource(str(uuid4()), conversation or str(uuid4()), clock() - 1, clock() + 3600,
                        MemoryOrigin.VOICE, tuple(dependencies))


def _automatic(ledger, clock, content="Synthetic automatic fact", conversation=None, **kwargs):
    return ledger.store(content, "fact", .9, (_source(clock, conversation, **kwargs),))


def _rows(db):
    return [tuple(row) for row in db.execute("SELECT * FROM deletion_ledger_v1 ORDER BY seq")]


def _kinds(db):
    return [row[1] for row in _rows(db)]


def _salt(db):
    return db.execute("SELECT salt FROM deletion_ledger_salt_v1").fetchone()[0]


def _memory_db():
    db = sqlite3.connect(":memory:", isolation_level=None)
    initialize_deletion_ledger(db)
    return db


def _append(db, kind, subject, at=NOW):
    db.execute("BEGIN IMMEDIATE")
    try:
        result = append_deletion(db, kind, subject, at)
        db.commit()
        return result
    except BaseException:
        db.rollback()
        raise


# --- the table itself -----------------------------------------------------------------------------

def test_kinds_are_exactly_the_seven_user_requested_deletions():
    """Design v3 N3's six kinds plus voice_consent_withdrawn (user decision 2026-11-12)."""
    assert {kind.value for kind in DeletionKind} == {
        "conversation_erased", "fact_deleted", "fact_pruned_retired_owner", "voice_profile_deleted",
        "user_retired", "credential_revoked", "voice_consent_withdrawn"} == stage._LEDGER_KINDS


def test_subject_digests_are_salted():
    """Review K4/Z2: fact ids derive from content, so an unsalted digest would be guessable."""
    import hashlib
    first, second = _memory_db(), _memory_db()
    for db in (first, second):
        _append(db, DeletionKind.FACT_DELETED, "fact-1")
    stored = [_rows(db)[0][2] for db in (first, second)]
    assert _salt(first) != _salt(second) and stored[0] != stored[1]
    assert hashlib.sha256(b"fact-1").hexdigest() not in stored
    assert subject_digest("a" * 64, "fact-1") != subject_digest("b" * 64, "fact-1")


def test_append_is_idempotent_per_kind_and_subject_and_chains_from_genesis():
    db = _memory_db()
    assert _append(db, DeletionKind.FACT_DELETED, "fact-1") is True
    assert _append(db, DeletionKind.FACT_DELETED, "fact-1", NOW + 5) is False  # insert-if-absent
    assert _append(db, DeletionKind.FACT_PRUNED_RETIRED_OWNER, "fact-1") is True  # the key includes the kind
    assert _append(db, DeletionKind.CONVERSATION_ERASED, "conversation-1") is True
    rows = _rows(db)
    assert [row[0] for row in rows] == [1, 2, 3]
    assert rows[0][4] == "0" * 64
    for previous, row in zip([None, *rows], rows):
        assert row[4] == (previous[5] if previous else "0" * 64)
        assert row[5] == row_sha256(row[4], row[0], row[1], row[2], row[3])
    chain = verify_chain(db)
    assert (chain.count, chain.max_seq, chain.chain_head, chain.chain_valid) == (3, 3, rows[-1][5], True)
    # Content-free: no subject id or salt appears in any ledger row.
    assert all("fact-1" not in json.dumps(row) and "conversation-1" not in json.dumps(row) for row in rows)
    assert rows[0][2] == subject_digest(_salt(db), "fact-1")


def test_stage_runtime_and_src_compute_the_same_chain():
    db = _memory_db()
    for index in range(5):
        _append(db, DeletionKind.CREDENTIAL_REVOKED, "credential-%d" % index, NOW + index / 3)
    expected = verify_chain(db).to_dict()
    assert stage._deletion_ledger_projection(db) == expected
    assert all(stage._ledger_row_sha256(*row[4:5], *row[:4]) == row[5] for row in _rows(db))


def test_salt_is_write_once_and_created_by_the_first_append():
    db = _memory_db()
    assert db.execute("SELECT count(*) FROM deletion_ledger_salt_v1").fetchone()[0] == 0  # nothing at open
    _append(db, DeletionKind.USER_RETIRED, "synthetic-user")
    salt = _salt(db)
    _append(db, DeletionKind.USER_RETIRED, "other-user")
    assert _salt(db) == salt
    with pytest.raises(sqlite3.IntegrityError):  # a second salt row is impossible (id = 1, PRIMARY KEY)
        db.execute("INSERT INTO deletion_ledger_salt_v1 VALUES (1, ?)", ("f" * 64,))
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO deletion_ledger_salt_v1 VALUES (2, ?)", ("f" * 64,))
    db.execute("DELETE FROM deletion_ledger_salt_v1")  # out of band
    with pytest.raises(DeletionLedgerError, match="deletion_ledger_salt_missing"):
        initialize_deletion_ledger(db)
    with pytest.raises(DeletionLedgerError, match="deletion_ledger_salt_missing"):
        _append(db, DeletionKind.USER_RETIRED, "third-user")  # never re-salted under existing rows


def test_a_changed_salt_changes_the_admission_authority(tmp_path):
    state = tmp_path / "state"
    (state / "data").mkdir(parents=True)
    path = state / "data" / "ledger.db"
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        initialize_deletion_ledger(db)
        _append(db, DeletionKind.USER_RETIRED, "synthetic-user")
    before = stage.capture_state_authority(state)["digest"]
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE deletion_ledger_salt_v1 SET salt=?", ("e" * 64,))
    assert stage.capture_state_authority(state)["digest"] != before


@pytest.mark.parametrize("statement", [
    "UPDATE deletion_ledger_v1 SET at=at+1 WHERE seq=1",
    "UPDATE deletion_ledger_v1 SET kind='fact_deleted' WHERE seq=2",
    "DELETE FROM deletion_ledger_v1 WHERE seq=1",
    "DELETE FROM deletion_ledger_v1 WHERE seq=2",
    "UPDATE deletion_ledger_v1 SET seq=9 WHERE seq=3",
])
def test_update_or_delete_breaks_the_chain(statement):
    db = _memory_db()
    for index, kind in enumerate((DeletionKind.USER_RETIRED, DeletionKind.CREDENTIAL_REVOKED,
                                  DeletionKind.USER_RETIRED)):
        _append(db, kind, "subject-%d" % index)
    db.execute(statement)
    with pytest.raises(DeletionLedgerError, match="deletion_ledger_chain_broken"):
        initialize_deletion_ledger(db)
    with pytest.raises(ValueError, match="runtime_admission_ledger_chain_broken"):
        stage._deletion_ledger_projection(db)


def test_a_self_consistent_row_on_a_forged_prev_breaks_the_chain():
    """Replacing a row with one whose own hash is valid but chains from elsewhere is refused."""
    db = _memory_db()
    for index in range(3):
        _append(db, DeletionKind.USER_RETIRED, "subject-%d" % index)
    seq, kind, subject, at, _prev, _row = _rows(db)[1]
    forged = "f" * 64
    db.execute("UPDATE deletion_ledger_v1 SET prev_sha256=?, row_sha256=? WHERE seq=?",
               (forged, row_sha256(forged, seq, kind, subject, at), seq))
    with pytest.raises(DeletionLedgerError, match="deletion_ledger_chain_broken"):
        verify_chain(db)
    with pytest.raises(ValueError, match="runtime_admission_ledger_chain_broken"):
        stage._deletion_ledger_projection(db)


@pytest.mark.parametrize("statement", [
    "DROP TABLE deletion_ledger_salt_v1",
    "CREATE INDEX extra ON deletion_ledger_v1(at)",
    "ALTER TABLE deletion_ledger_v1 ADD COLUMN extra TEXT",
])
def test_foreign_or_partial_shape_fails_closed(statement):
    db = _memory_db()
    db.execute(statement)
    with pytest.raises(DeletionLedgerError, match="deletion_ledger_schema_unavailable"):
        initialize_deletion_ledger(db)


def test_append_outside_a_transaction_or_with_an_unknown_kind_is_refused():
    db = _memory_db()
    with pytest.raises(DeletionLedgerError, match="deletion_ledger_transaction_required"):
        append_deletion(db, DeletionKind.USER_RETIRED, "synthetic-user", NOW)
    db.execute("BEGIN")
    with pytest.raises(DeletionLedgerError, match="deletion_ledger_kind_invalid"):
        append_deletion(db, "user_retired", "synthetic-user", NOW)
    with pytest.raises(DeletionLedgerError, match="deletion_ledger_time_invalid"):
        append_deletion(db, DeletionKind.USER_RETIRED, "synthetic-user", float("nan"))
    db.rollback()
    assert _rows(db) == []


def test_resurrection_count_is_content_free():
    db = _memory_db()
    assert resurrected(db, frozenset({DeletionKind.FACT_DELETED}), ["fact-1"]) == 0  # no salt yet
    _append(db, DeletionKind.FACT_DELETED, "fact-1")
    _append(db, DeletionKind.CONVERSATION_ERASED, "conversation-1")
    assert resurrected(db, frozenset({DeletionKind.FACT_DELETED}), ["fact-1", "fact-2", "conversation-1"]) == 1
    assert resurrected(db, frozenset(DeletionKind), ["fact-1", "conversation-1"]) == 2


# --- privacy ledger entry points ------------------------------------------------------------------

def test_conversation_erasure_appends_once_and_replay_never_appends(ledger, clock):
    conversation = str(uuid4())
    parent = _automatic(ledger, clock, conversation=conversation)
    _automatic(ledger, clock, "Synthetic dependent", dependencies=(FactDependency(
        parent.id, parent.revision, parent.expires_at),))
    ledger.erasures.request(conversation, REVIEWER)
    assert sorted(_kinds(ledger._db)) == ["conversation_erased", "fact_deleted", "fact_deleted"]
    before = _rows(ledger._db)
    # Erasure recovery (twice), a repeated request and a reopen are replays: nothing is appended.
    ledger.erasures.recover()
    ledger.erasures.recover()
    clock.value += 10
    ledger.erasures.request(conversation, REVIEWER)
    ledger.delete_conversation(conversation, REVIEWER)
    ledger.close()
    ledger.initialize()
    ledger.erasures.recover()
    assert _rows(ledger._db) == before


def test_explicit_fact_deletions_append_every_deleted_fact(ledger, clock):
    deleted = _automatic(ledger, clock)
    dependent = _automatic(ledger, clock, "Synthetic dependent", dependencies=(FactDependency(
        deleted.id, deleted.revision, deleted.expires_at),))
    assert ledger.delete_fact(deleted.id, REVIEWER) is True
    proposal = ledger.store("Synthetic proposal", "fact", 1.0, (_source(clock),), proposed=True)
    assert ledger.reject_proposal(proposal.id, REVIEWER) is True
    salt = _salt(ledger._db)
    digests = {row[2] for row in _rows(ledger._db)}
    assert digests == {subject_digest(salt, identity) for identity in (deleted.id, dependent.id, proposal.id)}
    assert set(_kinds(ledger._db)) == {"fact_deleted"}
    assert ledger.delete_fact(deleted.id, REVIEWER) is False  # already deleted: no new row
    assert len(_rows(ledger._db)) == 3


def test_owner_deletion_and_confirm_with_dependents_append(ledger, clock):
    owned = _automatic(ledger, clock, "Synthetic owned")
    ledger.assign_owner(owned.id, PERSON, owned.revision, 0, OPERATOR, **ALLOW)
    owned = ledger.get(owned.id)
    assert ledger.delete_owned(owned.id, MemoryOwnerActor("synthetic-home", PERSON)) is True
    parent = _automatic(ledger, clock, "Synthetic parent")
    child = _automatic(ledger, clock, "Synthetic child", dependencies=(FactDependency(
        parent.id, parent.revision, parent.expires_at),))
    ledger.assign_owner(parent.id, PERSON, parent.revision, 0, OPERATOR, delete_dependents=True, **ALLOW)
    salt = _salt(ledger._db)
    assert {row[2] for row in _rows(ledger._db)} == {subject_digest(salt, owned.id), subject_digest(salt, child.id)}


def test_retirement_cascade_appends_pruned_owner_rows_once(ledger, clock):
    fact = _automatic(ledger, clock, "Synthetic retired-owner fact")
    ledger.assign_owner(fact.id, PERSON, fact.revision, 0, OPERATOR, **ALLOW)
    assert ledger.clear_user_facts(frozenset({PERSON}), lambda ids: True) == 1
    assert _kinds(ledger._db) == ["fact_pruned_retired_owner"]
    before = _rows(ledger._db)
    assert ledger.prune_retired_owners(lambda ids: True) == 0  # the boot re-run finds nothing new
    assert ledger.clear_user_facts(frozenset({PERSON}), lambda ids: True) == 0
    assert _rows(ledger._db) == before


def test_expiry_and_voice_prune_append_nothing(ledger, clock):
    _automatic(ledger, clock)
    clock.value += 31 * 86400  # past the 30-day automatic retention
    assert ledger.expire() == 1
    from src.users.voice_ledger import VoiceLedger
    VoiceLedger(ledger).prune()
    assert _rows(ledger._db) == []


def _seed_profile(ledger, clock):
    sources = json.dumps([_source(clock).to_dict()], sort_keys=True)
    with ledger._transaction() as db:
        db.execute("INSERT INTO voice_profiles VALUES (?,'Synthetic','enc_synthetic',2,?,?,?,'synthetic-reviewer',1,NULL)",
                   (PROFILE, bytes(8), sources, clock()))


def test_voice_profile_deletion_appends_once(ledger, clock):
    from src.users.voice_ledger import VoiceLedger, VoiceReviewer
    _seed_profile(ledger, clock)
    voices = VoiceLedger(ledger)
    assert voices.delete_profile(PROFILE, 1, VoiceReviewer("synthetic-voice-operator")) is True
    assert voices.delete_profile(PROFILE, 2, VoiceReviewer("synthetic-voice-operator")) is False
    rows = _rows(ledger._db)
    assert [row[1] for row in rows] == ["voice_profile_deleted"]
    assert rows[0][2] == subject_digest(_salt(ledger._db), PROFILE)


# --- dashboard authority entry points -------------------------------------------------------------

def _auth_rows(service):
    with closing(sqlite3.connect(service.path)) as db:
        return [tuple(row) for row in db.execute("SELECT * FROM deletion_ledger_v1 ORDER BY seq")]


def test_revocation_and_retirement_append_and_logout_does_not(tmp_path):
    from src.core.account_metadata import ActiveAccounts
    from src.dashboard.authority import DashboardAuthority, DashboardRole
    from src.dashboard.credential_journal import CredentialAction, CredentialCommand, CredentialGrantSpec

    async def scenario():
        now, root = [1_700_000_000.0], "synthetic-administrative-token-A"
        service = DashboardAuthority(tmp_path / "private" / "auth.db", session_ttl_s=60, clock=lambda: now[0],
                                     expected_token=lambda: root,
                                     account_provider=lambda: ActiveAccounts(True, frozenset({"synthetic-user"})))
        await service.initialize()
        try:
            assert _auth_rows(service) == []
            admin = service.resolve(root)
            issued = await service.issue_credential(admin, subject="synthetic-a", role=DashboardRole.HOME,
                                                    expires_at=now[0] + 120)
            session = await service.login(issued.token)
            assert await service.logout(session.lease) is True
            assert _auth_rows(service) == []  # logout and session expiry are never recorded
            assert await service.revoke_credential(admin, issued.identity) is True
            assert await service.revoke_credential(admin, issued.identity) is False
            command = CredentialCommand(str(uuid4()), CredentialAction.ISSUE,
                                        grant=CredentialGrantSpec(DashboardRole.HOME, now[0] + 90, subject="synthetic-b"))
            await service.commands.execute(admin, command)
            revoke = CredentialCommand(str(uuid4()), CredentialAction.REVOKE, command.request_id, 1)
            assert (await service.commands.execute(admin, revoke)).applied
            assert await service.retire_user(admin, "synthetic-user") is True
            await service.retire_user(admin, "synthetic-user")
            rows = _auth_rows(service)
            assert [row[1] for row in rows] == ["credential_revoked", "credential_revoked", "user_retired"]
            now[0] += 300  # every remaining credential and session expires: nothing appended
            await service.synchronize()
            assert _auth_rows(service) == rows
        finally:
            await service.aclose()
        reopened = DashboardAuthority(tmp_path / "private" / "auth.db", clock=lambda: now[0],
                                      expected_token=lambda: root)
        await reopened.initialize()
        try:
            assert _auth_rows(reopened) == rows  # migration validates, never appends
        finally:
            await reopened.aclose()

    asyncio.run(scenario())


# --- coverage: every deletion entry point is recorded or explicitly excluded ----------------------

# (file, function) -> the kind its request appends, or None for an explicit exclusion.
_ENTRY_POINTS = {
    ("src/memory/fact_ledger.py", "revoke_voice_consent"): "VOICE_CONSENT_WITHDRAWN",
    ("src/memory/fact_ledger.py", "request_voice_consent"): None,  # grants consent, never withdraws
    ("src/memory/fact_ledger.py", "reopen_voice_consent"): None,  # an operator reopen, not a person's withdrawal
    ("src/memory/fact_ledger.py", "delete_fact"): "FACT_DELETED",
    ("src/memory/fact_ledger.py", "reject_proposal"): "FACT_DELETED",
    ("src/memory/fact_ledger.py", "assign_owner"): "FACT_DELETED",
    ("src/memory/fact_ledger.py", "delete_owned"): "FACT_DELETED",
    ("src/memory/fact_ledger.py", "_release_voice_proposals"): "FACT_PRUNED_RETIRED_OWNER",
    ("src/memory/fact_ledger.py", "clear_user_facts"): "FACT_PRUNED_RETIRED_OWNER",
    ("src/memory/fact_ledger.py", "delete_conversation"): "CONVERSATION_ERASED",
    ("src/memory/fact_ledger.py", "expire"): None,  # automatic expiry is never recorded
    ("src/users/voice_ledger.py", "delete_profile"): "VOICE_PROFILE_DELETED",
    ("src/users/voice_ledger.py", "prune"): None,  # committed-invalid profiles follow an erasure
    ("src/dashboard/authority.py", "_transaction"): "USER_RETIRED+CREDENTIAL_REVOKED",
    ("src/dashboard/credential_commands.py", "_apply_sql"): "CREDENTIAL_REVOKED",
}


def _functions_with(path, predicate):
    tree = ast.parse((REPO / path).read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if predicate(child):
                    found.setdefault(node.name, set()).add(child)
    return found


def _deletion_call(node):
    """A call of a deletion primitive: FactLedger._delete, VoiceLedger._delete, or credential/retirement SQL."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_delete":
        return True
    return (isinstance(node, ast.Constant) and type(node.value) is str
            and any(text in node.value for text in ("DELETE FROM credentials WHERE id=", "INTO retired_users")))


def _innermost(path):
    """Map each deletion primitive to the innermost function that contains it."""
    tree = ast.parse((REPO / path).read_text())
    owners = {}

    def visit(node, function):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
            if _deletion_call(child):
                owners.setdefault(inner, []).append(child)
            visit(child, inner)

    visit(tree, None)
    return owners


def _appends(path, function):
    tree = ast.parse((REPO / path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            kinds = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) \
                        and child.value.id == "DeletionKind":
                    kinds.add(child.attr)
            return kinds
    raise AssertionError((path, function))


def _consent_revokers():
    """Functions that write a consent row and can land it in VoiceConsentState.REVOKED."""
    tree = ast.parse((REPO / "src/memory/fact_ledger.py").read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes = list(ast.walk(node))
            writes = any(isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                         and child.func.attr == "_write_consent" for child in nodes)
            revoked = any(isinstance(child, ast.Attribute) and child.attr == "REVOKED"
                          and isinstance(child.value, ast.Name) and child.value.id == "VoiceConsentState"
                          for child in nodes)
            if writes and revoked:
                found.add(node.name)
    return found


def test_every_deletion_entry_point_is_enumerated_and_recorded():
    """Enumerate the deletion primitives' callers from the source; each is mapped and appends its kind."""
    discovered = set()
    for path in ("src/memory/fact_ledger.py", "src/users/voice_ledger.py", "src/dashboard/authority.py",
                 "src/dashboard/credential_commands.py"):
        for function in _innermost(path):
            if function in {"_delete"}:
                continue  # the primitives themselves (FactLedger._delete, VoiceLedger._delete)
            discovered.add((path, function))
    discovered |= {("src/memory/fact_ledger.py", function) for function in _consent_revokers()}
    assert discovered == set(_ENTRY_POINTS), sorted(discovered ^ set(_ENTRY_POINTS))
    for (path, function), expected in _ENTRY_POINTS.items():
        kinds = _appends(path, function)
        if expected is None:
            assert kinds == set(), (path, function)
        else:
            assert set(expected.split("+")) <= kinds, (path, function, kinds)
    # The primitive appends only what its caller passes; it names no kind of its own.
    assert _appends("src/memory/fact_ledger.py", "_delete") == set()
    assert "append_deletion" not in (REPO / "src/memory/erasure_journal.py").read_text()
    assert "append_deletion" not in (REPO / "src/memory/payload_derivation.py").read_text()
    assert "append_deletion" not in (REPO / "src/memory/state_journal_derivation.py").read_text()
    assert ledger_module.append_deletion.__name__ == "append_deletion"


# --- voice consent withdrawal (seventh kind) -------------------------------------------------------

def _seed_consent(ledger, clock, state="pending_verification", revision=1):
    with ledger._transaction() as db:
        db.execute("INSERT INTO voice_attribution_consents VALUES (?,?,?,NULL,NULL,NULL,?,?) ON CONFLICT(person_id) "
                   "DO UPDATE SET state=excluded.state, revision=excluded.revision",
                   (PERSON, state, revision, clock(), PERSON))


def test_each_voice_consent_withdrawal_appends_one_row(ledger, clock):
    from src.memory.deletion_ledger import consent_withdrawal_subject
    actor = MemoryVoiceConsentActor("synthetic-home", PERSON)
    _seed_consent(ledger, clock)
    assert ledger.revoke_voice_consent(actor, 1).revision == 2
    with pytest.raises(ValueError, match="memory_voice_consent_conflict"):  # a replayed request
        ledger.revoke_voice_consent(actor, 1)
    _seed_consent(ledger, clock, revision=3)  # consent requested again, then withdrawn again
    ledger.revoke_voice_consent(actor, 3)
    salt = _salt(ledger._db)
    assert [(row[1], row[2]) for row in _rows(ledger._db)] == [
        ("voice_consent_withdrawn", subject_digest(salt, consent_withdrawal_subject(PERSON, 2))),
        ("voice_consent_withdrawn", subject_digest(salt, consent_withdrawal_subject(PERSON, 4)))]


def test_operator_reopen_appends_nothing(ledger, clock):
    _seed_consent(ledger, clock, state="verification_locked", revision=2)
    ledger.reopen_voice_consent(PERSON, 2, OPERATOR)
    assert _rows(ledger._db) == []


# --- K3: every explicit append is in its deletion's own transaction --------------------------------

class _Injected(RuntimeError):
    pass


def _failing_append(*_args, **_kwargs):
    raise _Injected("synthetic append failure")


def test_conversation_erasure_and_its_append_share_one_transaction(ledger, clock, monkeypatch):
    from src.memory import fact_ledger as module
    bare = str(uuid4())  # no derived facts: the conversation append is the only append
    monkeypatch.setattr(module, "append_deletion", _failing_append)
    with pytest.raises(_Injected):
        ledger.delete_conversation(bare, REVIEWER)
    monkeypatch.undo()
    assert ledger._db.execute("SELECT count(*) FROM conversation_tombstones").fetchone()[0] == 0
    assert ledger._db.execute("SELECT count(*) FROM conversation_erasures").fetchone()[0] == 0
    conversation = str(uuid4())
    fact = _automatic(ledger, clock, conversation=conversation)
    monkeypatch.setattr(module, "append_deletion", _failing_append)
    with pytest.raises(_Injected):
        ledger.delete_conversation(conversation, REVIEWER)
    monkeypatch.undo()
    assert ledger._db.execute("SELECT count(*) FROM conversation_tombstones").fetchone()[0] == 0
    assert ledger.get(fact.id).state.value == "automatic" and _rows(ledger._db) == []
    # A failure after the append, before commit, rolls the append back with the erasure.
    monkeypatch.setattr(ledger.erasures, "ensure_operation", _failing_append)
    with pytest.raises(_Injected):
        ledger.delete_conversation(conversation, REVIEWER)
    monkeypatch.undo()
    assert ledger._db.execute("SELECT count(*) FROM conversation_tombstones").fetchone()[0] == 0
    assert _rows(ledger._db) == []


def test_voice_profile_deletion_and_its_append_share_one_transaction(ledger, clock, monkeypatch):
    from src.users import voice_ledger as module
    _seed_profile(ledger, clock)
    voices = module.VoiceLedger(ledger)
    monkeypatch.setattr(module, "append_deletion", _failing_append)
    with pytest.raises(_Injected):
        voices.delete_profile(PROFILE, 1, module.VoiceReviewer("synthetic-voice-operator"))
    monkeypatch.undo()
    row = ledger._db.execute("SELECT deleted_at, revision FROM voice_profiles WHERE id=?", (PROFILE,)).fetchone()
    assert tuple(row) == (None, 1) and _rows(ledger._db) == []


def test_consent_withdrawal_and_its_append_share_one_transaction(ledger, clock, monkeypatch):
    from src.memory import fact_ledger as module
    _seed_consent(ledger, clock)
    monkeypatch.setattr(module, "append_deletion", _failing_append)
    with pytest.raises(_Injected):
        ledger.revoke_voice_consent(MemoryVoiceConsentActor("synthetic-home", PERSON), 1)
    monkeypatch.undo()
    assert tuple(ledger._db.execute("SELECT state, revision FROM voice_attribution_consents").fetchone()) == (
        "pending_verification", 1)


def test_dashboard_retirement_and_revocations_share_one_transaction(tmp_path, monkeypatch):
    from src.core.account_metadata import ActiveAccounts
    from src.dashboard import authority as authority_module
    from src.dashboard import credential_commands as commands_module
    from src.dashboard.authority import DashboardAuthority, DashboardRole
    from src.dashboard.credential_journal import CredentialAction, CredentialCommand, CredentialGrantSpec

    def table(service, statement):
        with closing(sqlite3.connect(service.path)) as db:
            return db.execute(statement).fetchall()

    async def scenario():
        now, root = [1_700_000_000.0], "synthetic-administrative-token-A"
        service = DashboardAuthority(tmp_path / "private" / "auth.db", session_ttl_s=60, clock=lambda: now[0],
                                     expected_token=lambda: root,
                                     account_provider=lambda: ActiveAccounts(True, frozenset({"synthetic-user"})))
        await service.initialize()
        try:
            admin = service.resolve(root)
            issued = await service.issue_credential(admin, subject="synthetic-a", role=DashboardRole.HOME,
                                                    expires_at=now[0] + 120)
            command = CredentialCommand(str(uuid4()), CredentialAction.ISSUE,
                                        grant=CredentialGrantSpec(DashboardRole.HOME, now[0] + 90, subject="synthetic-b"))
            await service.commands.execute(admin, command)
            credentials = table(service, "SELECT id FROM credentials ORDER BY id")
            monkeypatch.setattr(authority_module, "append_deletion", _failing_append)
            monkeypatch.setattr(commands_module, "append_deletion", _failing_append)
            with pytest.raises(Exception):
                await service.revoke_credential(admin, issued.identity)
            revoke = CredentialCommand(str(uuid4()), CredentialAction.REVOKE, command.request_id, 1)
            with pytest.raises(Exception):
                await service.commands.execute(admin, revoke)
            with pytest.raises(Exception):
                await service.retire_user(admin, "synthetic-user")
            monkeypatch.undo()
            assert table(service, "SELECT id FROM credentials ORDER BY id") == credentials
            assert table(service, "SELECT * FROM retired_users") == []
            assert table(service, "SELECT * FROM deletion_ledger_v1") == []
        finally:
            await service.aclose()

    asyncio.run(scenario())


# --- K5: a revoked credential id is never reissued -------------------------------------------------

def test_a_legacy_revoked_id_cannot_be_reissued_through_a_client_chosen_issue_target(tmp_path):
    from src.dashboard.authority import DashboardAuthority, DashboardRole
    from src.dashboard.credential_journal import (
        CredentialAction, CredentialCommand, CredentialGrantSpec, CredentialOutcome,
    )

    async def scenario():
        now, root = [1_700_000_000.0], "synthetic-administrative-token-A"
        service = DashboardAuthority(tmp_path / "private" / "auth.db", session_ttl_s=60, clock=lambda: now[0],
                                     expected_token=lambda: root)
        await service.initialize()
        try:
            admin = service.resolve(root)
            issued = await service.issue_credential(admin, subject="synthetic-a", role=DashboardRole.HOME,
                                                    expires_at=now[0] + 120)
            assert await service.revoke_credential(admin, issued.identity) is True  # legacy: no attempt row
            reissue = CredentialCommand(issued.identity, CredentialAction.ISSUE,  # target = request_id
                                        grant=CredentialGrantSpec(DashboardRole.HOME, now[0] + 90, subject="synthetic-z"))
            result = await service.commands.execute(admin, reissue)
            assert not result.applied and result.token is None
            assert result.receipt.outcome is CredentialOutcome.NO_EFFECT
            assert result.receipt.reason == "revision_conflict"
            with closing(sqlite3.connect(service.path)) as db:
                assert db.execute("SELECT count(*) FROM credentials WHERE id=?", (issued.identity,)).fetchone()[0] == 0
                assert [row[0] for row in db.execute("SELECT kind FROM deletion_ledger_v1")] == ["credential_revoked"]
        finally:
            await service.aclose()

    asyncio.run(scenario())
