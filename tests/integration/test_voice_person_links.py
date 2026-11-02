"""Real SQLite relationship CAS and pure account-removal proofs with fake audio."""

import asyncio
from contextlib import contextmanager
import hashlib
import sqlite3
import threading
from types import SimpleNamespace
from uuid import uuid4
from weakref import ref

import pytest
import pytest_asyncio

from src.dashboard.authority import DashboardAuthority
from src.dashboard.live_retention_types import VoicePersonFence
from src.memory.privacy_services import PrivacyServices
from src.users.user_manager import PermissionLevel, UserManager
from src.users.voice_ledger import VoiceLedger, VoiceReviewer
from src.users.voice_people import VoicePersonLinks
from src.users.voice_registry import VoiceRegistry
from tests.mocks.voice_registry import ACTOR, graph, seed


@pytest_asyncio.fixture
async def people_graph(tmp_path, monkeypatch):
    # Exercise the real native owners; keep periodic maintenance out of the
    # deterministic commit/publication barriers used by these tests.
    monkeypatch.setattr(VoiceRegistry, 'start', lambda self: None)
    manager = voices = clock = None
    authority = DashboardAuthority(tmp_path/'auth.db', clock=lambda: clock.now if clock is not None else 2_000_000_000.,
        expected_token=lambda: 'synthetic-person-root',
        account_provider=lambda: manager.active_accounts() if manager is not None else None)
    await authority.initialize()
    try:
        manager = await asyncio.to_thread(UserManager, str(tmp_path/'accounts.json'),
            retirement_proof=lambda identities: authority.users_retired(identities)
                and voices is not None and voices.ledger.people.user_links_cleared(identities))
        a, _ = await asyncio.to_thread(manager.add_user, 'Synthetic A', PermissionLevel.ADMIN)
        b, _ = await asyncio.to_thread(manager.add_user, 'Synthetic B', PermissionLevel.ADMIN)
        async with graph(tmp_path) as (privacy, owner, voices, clock):
            sources, identities = await seed(voices, clock)
            profile = await voices.acall(voices.ledger.confirm, identities, 'Synthetic reviewed voice', str(uuid4()), ACTOR)
            root = authority.resolve('synthetic-person-root')
            session = await authority.login('synthetic-person-root')
            actor = VoiceReviewer.from_dashboard(session.lease.principal,
                lambda: session.lease.current(frozenset({'voices:review', 'admin:manage'})), person_authority=authority)
            yield SimpleNamespace(privacy=privacy, owner=owner, voices=voices, links=voices.ledger.people,
                clock=clock, sources=sources, identities=identities, profile=profile, authority=authority,
                root=root, session=session, actor=actor, manager=manager, a=a.user_id, b=b.user_id,
                database=tmp_path/'privacy.db')
    finally:
        await authority.aclose()


async def metadata(graph, links=None):
    def read():
        with graph.privacy.ledger._transaction() as db:
            return (links or graph.links).project_rows(db, (graph.profile['id'],), graph.actor)
    return await graph.voices.acall(read)


async def profile_fingerprint(graph):
    def read():
        with graph.privacy.ledger._transaction() as db:
            row = db.execute('SELECT revision,embedding,name FROM voice_profiles WHERE id=?', (graph.profile['id'],)).fetchone()
            return row['revision'], hashlib.sha256(row['embedding']).hexdigest(), row['name']
    return await graph.voices.acall(read)


async def test_link_unlink_persistence_and_independent_revisions(people_graph):
    g = people_graph
    before = await profile_fingerprint(g)
    assert g.links.user_links_cleared(frozenset({g.a}))
    receipt = await g.voices.acall(g.links.set_person, g.profile['id'], g.a, 1, 0, g.actor)
    assert receipt.link_revision == 1 and not g.links.user_links_cleared(frozenset({g.a}))
    rows, fence = await metadata(g)
    assert rows == [{'user_id': g.a, 'revision': 1}] and fence.current()
    restored = VoicePersonLinks(g.voices.ledger)
    assert restored.user_links_cleared(frozenset({g.b}))
    assert not restored.user_links_cleared(frozenset({g.a}))
    await g.voices.acall(restored.reconcile)
    assert (await metadata(g, restored))[0] == rows
    assert restored.user_links_cleared(frozenset({g.b}))
    receipt = await g.voices.acall(g.links.set_person, g.profile['id'], None, 1, 1, g.actor)
    assert receipt.link_revision == 2
    assert (await metadata(g))[0] == [{'user_id': None, 'revision': 2}]
    with pytest.raises(ValueError, match='link_conflict'):
        await g.voices.acall(g.links.set_person, g.profile['id'], g.b, 1, 0, g.actor)
    with pytest.raises(ValueError, match='profile_conflict'):
        await g.voices.acall(g.links.set_person, g.profile['id'], g.b, 2, 2, g.actor)
    await g.voices.acall(g.links.set_person, g.profile['id'], g.b, 1, 2, g.actor)
    assert (await metadata(g))[0] == [{'user_id': g.b, 'revision': 3}]
    assert await profile_fingerprint(g) == before
    ordinary = await g.voices.acall(g.voices.ledger.project, 'profiles', None, ACTOR)
    assert set(ordinary.rows[0]) == {'id','name','revision','confirmed_at','confirmed_by','expires_at'}


async def test_link_metadata_never_reads_the_acoustic_vector(people_graph):
    g = people_graph
    statements = []
    def write():
        g.privacy.ledger._db.set_trace_callback(statements.append)
        try:
            return g.links.set_person(g.profile['id'], g.a, 1, 0, g.actor)
        finally:
            g.privacy.ledger._db.set_trace_callback(None)
    assert (await g.voices.acall(write)).link_revision == 1
    assert not any('embedding' in statement.lower() for statement in statements)


async def test_review_alone_cannot_link_and_unavailable_target_is_not_faked_as_unlinked(people_graph):
    g = people_graph
    with pytest.raises(PermissionError):
        await g.voices.acall(g.links.set_person, g.profile['id'], g.a, 1, 0, ACTOR)
    await g.voices.acall(g.links.set_person, g.profile['id'], g.a, 1, 0, g.actor)
    await g.authority.retire_user(g.root, g.a)
    with pytest.raises(LookupError):
        await metadata(g)
    with pytest.raises(LookupError):
        await g.voices.acall(g.links.set_person, g.profile['id'], g.a, 1, 1, g.actor)
    assert (await g.voices.acall(g.links.set_person, g.profile['id'], None, 1, 1, g.actor)).link_revision == 2


@pytest.mark.parametrize('change', ['actor', 'target'])
async def test_final_guard_rolls_back_provisional_link_after_exit_reads(people_graph, monkeypatch, change):
    g = people_graph
    original = g.privacy.ledger._read_fact_receipts
    entered, release = threading.Event(), threading.Event()
    def read(db):
        value = original(db)
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return value
    monkeypatch.setattr(g.privacy.ledger, '_read_fact_receipts', read)
    g.privacy.ledger._pending_fact_ids = frozenset({'synthetic-exit-read-recovery'})
    pending = asyncio.create_task(g.voices.acall(g.links.set_person, g.profile['id'], g.a, 1, 0, g.actor))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        if change == 'actor':
            assert await g.authority.logout(g.session.lease)
        else:
            assert await g.authority.retire_user(g.root, g.a)
    finally:
        release.set()
    with pytest.raises(PermissionError if change == 'actor' else LookupError):
        await pending
    with sqlite3.connect(g.database) as db:
        assert db.execute('SELECT count(*) FROM voice_profile_people').fetchone()[0] == 0
        assert db.execute('SELECT revision FROM voice_profiles').fetchone()[0] == 1
    assert not g.links.user_links_cleared(frozenset({g.a}))
    await g.voices.acall(g.links.reconcile)
    assert g.links.user_links_cleared(frozenset({g.a}))


async def test_committed_link_cannot_authorize_json_deletion_before_proof_publication(people_graph, monkeypatch):
    g = people_graph
    original = g.links._finish
    entered, release = threading.Event(), threading.Event()
    def finish(committed, users):
        if committed and not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(committed, users)
    monkeypatch.setattr(g.links, '_finish', finish)
    pending = asyncio.create_task(g.voices.acall(g.links.set_person, g.profile['id'], g.a, 1, 0, g.actor))
    before = g.manager.data_path.read_bytes(), g.manager._version
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        with sqlite3.connect(g.database) as db:
            assert db.execute('SELECT user_id FROM voice_profile_people').fetchone()[0] == g.a
        await g.authority.retire_user(g.root, g.a)
        assert not g.links.user_links_cleared(frozenset({g.a}))
        with pytest.raises(PermissionError):
            await asyncio.wait_for(asyncio.to_thread(g.manager.remove_user, g.a), 1)
        assert (g.manager.data_path.read_bytes(), g.manager._version) == before
        assert g.manager.get_user(g.a) is not None
    finally:
        release.set()
        await pending
    assert await g.voices.acall(g.links.clear_user_links, frozenset({g.a}), g.authority.users_retired) == 1
    assert g.links.user_links_cleared(frozenset({g.a}))
    assert (await asyncio.to_thread(g.manager.remove_user, g.a))[0]
    assert (await metadata(g))[0] == [{'user_id': None, 'revision': 2}]


@pytest.mark.parametrize('stage', ['before', 'after'])
@pytest.mark.parametrize('operation', ['link', 'clear'])
async def test_unknown_commit_stays_unavailable_until_real_sql_reconciliation(people_graph, monkeypatch, stage, operation):
    g = people_graph
    if operation == 'clear':
        await g.voices.acall(g.links.set_person, g.profile['id'], g.a, 1, 0, g.actor)
        await g.authority.retire_user(g.root, g.a)
    original = g.privacy.ledger._transaction
    @contextmanager
    def transaction(guard=None):
        with original(guard=guard) as db:
            yield db
            if guard is not None and stage == 'before':
                raise OSError('synthetic error before actual commit')
        if guard is not None and stage == 'after':
            raise OSError('synthetic error after actual commit')
    with monkeypatch.context() as changes:
        changes.setattr(g.privacy.ledger, '_transaction', transaction)
        with pytest.raises(OSError):
            if operation == 'link':
                await g.voices.acall(g.links.set_person, g.profile['id'], g.a, 1, 0, g.actor)
            else:
                await g.voices.acall(g.links.clear_user_links, frozenset({g.a}), g.authority.users_retired)
    assert not g.links.user_links_cleared(frozenset({g.a}))
    await g.voices.acall(g.links.reconcile)
    linked = (operation == 'link') == (stage == 'after')
    assert g.links.user_links_cleared(frozenset({g.a})) is not linked
    if operation == 'clear':
        count = await g.voices.acall(g.links.clear_user_links, frozenset({g.a}), g.authority.users_retired)
        assert count == (1 if stage == 'before' else 0)
        assert (await metadata(g))[0] == [{'user_id': None, 'revision': 2}]


async def test_profile_deletion_clears_only_relationship_and_keeps_unlink_revision(people_graph):
    g = people_graph
    await g.voices.acall(g.links.set_person, g.profile['id'], g.a, 1, 0, g.actor)
    assert await g.voices.acall(g.voices.ledger.delete_profile, g.profile['id'], 1, ACTOR)
    await g.voices.acall(g.links.reconcile)
    assert g.links.user_links_cleared(frozenset({g.a}))
    with sqlite3.connect(g.database) as db:
        assert db.execute('SELECT user_id,revision FROM voice_profile_people').fetchone() == (None, 2)


async def test_canonical_batch_and_fence_do_not_wait_for_user_catalog_lock(people_graph):
    g = people_graph
    entered, release = threading.Event(), threading.Event()
    def hold():
        with g.manager._lock:
            entered.set()
            assert release.wait(5)
    holder = asyncio.create_task(asyncio.to_thread(hold))
    fence = VoicePersonFence(ref(g.authority), tuple(sorted((g.a, g.b))))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert await asyncio.wait_for(asyncio.to_thread(g.authority.users_linkable, frozenset({g.a, g.b})), 1)
        assert await asyncio.wait_for(asyncio.to_thread(fence.current), 1)
        assert not release.is_set()
    finally:
        release.set()
        await holder


async def test_later_writer_cannot_be_overwritten_by_older_proof_publication(people_graph, monkeypatch):
    g = people_graph
    original = g.links._finish
    entered, release = threading.Event(), threading.Event()
    def finish(committed, users):
        if committed and not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(committed, users)
    monkeypatch.setattr(g.links, '_finish', finish)
    first = asyncio.create_task(g.voices.acall(g.links.set_person, g.profile['id'], g.a, 1, 0, g.actor))
    second = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        second = asyncio.create_task(g.voices.acall(g.links.set_person, g.profile['id'], g.b, 1, 1, g.actor))
        async with asyncio.timeout(3):
            while g.links._proof.pending < 2:
                await asyncio.sleep(0)
        assert not g.links.user_links_cleared(frozenset({g.a, g.b}))
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second is not None else []))
    assert g.links.user_links_cleared(frozenset({g.a}))
    assert not g.links.user_links_cleared(frozenset({g.b}))
    assert (await metadata(g))[0] == [{'user_id': g.b, 'revision': 2}]


async def test_read_only_reconciliation_does_not_withdraw_a_valid_empty_proof(people_graph, monkeypatch):
    g = people_graph
    original = g.links._users
    entered, release = threading.Event(), threading.Event()
    def read(db):
        value = original(db)
        entered.set()
        assert release.wait(5)
        return value
    monkeypatch.setattr(g.links, '_users', read)
    pending = asyncio.create_task(g.voices.acall(g.links.reconcile))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert g.links.user_links_cleared(frozenset({g.a}))
    finally:
        release.set()
        await pending


async def test_all_helpers_share_pending_admission_and_the_same_published_proof(people_graph,monkeypatch):
    g = people_graph
    other = VoiceLedger(g.privacy.ledger).people
    await g.voices.acall(other.reconcile)
    assert other.user_links_cleared(frozenset({g.a})) and g.links.user_links_cleared(frozenset({g.a}))
    original = other._finish
    entered, release = threading.Event(), threading.Event()
    def finish(committed, users):
        if committed and not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(committed, users)
    monkeypatch.setattr(other,'_finish',finish)
    pending = asyncio.create_task(g.voices.acall(other.set_person,g.profile['id'],g.a,1,0,g.actor))
    try:
        assert await asyncio.to_thread(entered.wait,3)
        assert not other.user_links_cleared(frozenset({g.a}))
        assert not g.links.user_links_cleared(frozenset({g.a}))
        await g.authority.retire_user(g.root,g.a)
        with pytest.raises(PermissionError):
            await asyncio.wait_for(asyncio.to_thread(g.manager.remove_user,g.a),1)
        assert g.manager.get_user(g.a) is not None
    finally:
        release.set()
        await pending
    assert not other.user_links_cleared(frozenset({g.a})) and not g.links.user_links_cleared(frozenset({g.a}))
    await g.voices.acall(g.links.clear_user_links,frozenset({g.a}),g.authority.users_retired)
    assert other.user_links_cleared(frozenset({g.a})) and g.links.user_links_cleared(frozenset({g.a}))
    assert (await asyncio.to_thread(g.manager.remove_user,g.a))[0]


async def test_fresh_ledger_recovers_only_the_actual_persisted_relationship(people_graph):
    g = people_graph
    await g.voices.acall(g.links.set_person,g.profile['id'],g.a,1,0,g.actor)
    await g.voices.aclose()
    await g.owner.aclose()
    await g.privacy.aclose()
    reopened = PrivacyServices(str(g.database),clock=g.clock)
    people = VoiceLedger(reopened.ledger).people
    try:
        assert not people.user_links_cleared(frozenset({g.b}))
        await reopened.acall(reopened.initialize)
        await reopened.acall(people.reconcile)
        assert not people.user_links_cleared(frozenset({g.a}))
        assert people.user_links_cleared(frozenset({g.b}))
        def read():
            with reopened.ledger._transaction() as db:
                return people.project_rows(db,(g.profile['id'],),g.actor)
        rows,fence = await reopened.acall(read)
        assert rows==[{'user_id':g.a,'revision':1}] and fence.current()
    finally:
        await reopened.aclose()
