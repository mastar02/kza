"""Canonical person metadata fences never authenticate acoustic profiles."""

import asyncio
import gc
import threading
from types import SimpleNamespace
from weakref import ref

import pytest
import pytest_asyncio

from src.core.account_metadata import ActiveAccounts
from src.dashboard.authority import DashboardAuthority
from src.dashboard.live_retention_types import RevisionSet, RevisionStamp, SnapshotMetadata, VoicePersonFence


@pytest_asyncio.fixture
async def people_authority(tmp_path):
    snapshots = [ActiveAccounts(True, frozenset({'synthetic-a', 'synthetic-b'}))]
    calls = []
    def provider():
        calls.append(True)
        return snapshots[0]
    authority = DashboardAuthority(tmp_path/'auth.db', expected_token=lambda: 'synthetic-person-root', account_provider=provider)
    await authority.initialize()
    try:
        yield SimpleNamespace(authority=authority, snapshots=snapshots, calls=calls)
    finally:
        await authority.aclose()


async def test_batch_checks_one_canonical_snapshot_and_all_targets(people_authority):
    graph = people_authority
    assert graph.authority.users_linkable(frozenset({'synthetic-a', 'synthetic-b'}))
    assert len(graph.calls) == 1
    graph.snapshots[0] = ActiveAccounts(True, frozenset({'synthetic-a'}))
    assert not graph.authority.users_linkable(frozenset({'synthetic-a', 'synthetic-b'}))
    assert graph.authority.users_linkable(frozenset({'synthetic-a'}))
    graph.snapshots[0] = ActiveAccounts(False, frozenset())
    assert not graph.authority.users_linkable(frozenset({'synthetic-a'}))
    graph.snapshots[0] = None
    assert not graph.authority.users_linkable(frozenset({'synthetic-a'}))


async def test_batch_never_treats_an_empty_or_unknown_identity_as_linkable(people_authority):
    authority = people_authority.authority
    for identities in [None, 'synthetic-a', ['synthetic-a'], {'synthetic-a'}, (), frozenset(),
                       frozenset({None}), frozenset({''}), frozenset({'bad\nidentity'}),
                       frozenset({'a'*129}), frozenset(str(index) for index in range(257))]:
        assert not authority.users_linkable(identities)
    assert people_authority.calls == []


async def test_closed_fence_shape_contains_only_weak_authority_and_canonical_ids(people_authority):
    authority = people_authority.authority
    fence = VoicePersonFence(ref(authority), ('synthetic-a', 'synthetic-b'))
    assert fence.current() and 'synthetic-' not in repr(fence)
    for identities in [(), ['synthetic-a'], ('synthetic-b', 'synthetic-a'), ('synthetic-a', 'synthetic-a'),
                       ('',), (None,), ('bad\x7fidentity',), ('a'*129,), tuple(str(index) for index in range(51))]:
        with pytest.raises(ValueError):
            VoicePersonFence(ref(authority), identities)
    with pytest.raises(ValueError):
        VoicePersonFence(lambda: authority, ('synthetic-a',))
    with pytest.raises(ValueError):
        RevisionSet([fence])
    with pytest.raises(ValueError):
        RevisionSet((SimpleNamespace(current=lambda: True),))


async def test_pending_retirement_invalidates_person_fence_before_sqlite_commit(people_authority, monkeypatch):
    graph = people_authority
    fence = VoicePersonFence(ref(graph.authority), ('synthetic-a',))
    original = graph.authority._transaction
    entered, release = threading.Event(), threading.Event()
    def transaction(action, args, guard):
        if action == 'retire_user':
            entered.set()
            assert release.wait(5)
        return original(action, args, guard)
    monkeypatch.setattr(graph.authority, '_transaction', transaction)
    actor = graph.authority.resolve('synthetic-person-root')
    pending = asyncio.create_task(graph.authority.retire_user(actor, 'synthetic-a'))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert 'synthetic-a' in graph.snapshots[0].active_ids
        assert not fence.current()
        assert not graph.authority.users_retired(frozenset({'synthetic-a'}))
        assert graph.authority.users_linkable(frozenset({'synthetic-b'}))
    finally:
        release.set()
        await pending
    assert graph.authority.users_retired(frozenset({'synthetic-a'}))
    assert not fence.current()


async def test_projection_revision_and_person_availability_are_independent(people_authority):
    class Producer:
        projection_revision = 1
    producer = Producer()
    fence = VoicePersonFence(ref(people_authority.authority), ('synthetic-a',))
    combined = RevisionSet((RevisionStamp(ref(producer), 1), fence))
    class Policy:
        def sources_available(self, sources):
            return True
    metadata = SnapshotMetadata('synthetic-token', (), revision=combined)
    assert metadata.available(Policy(), 1)
    producer.projection_revision = 2
    assert not metadata.available(Policy(), 1)
    producer.projection_revision = 1
    people_authority.snapshots[0] = ActiveAccounts(True, frozenset({'synthetic-b'}))
    assert not metadata.available(Policy(), 1)


async def test_missing_provider_failure_or_closed_authority_cannot_keep_person_metadata(people_authority, monkeypatch):
    authority = people_authority.authority
    fence = VoicePersonFence(ref(authority), ('synthetic-a',))
    def failed():
        raise RuntimeError('synthetic unavailable canonical source')
    with monkeypatch.context() as changes:
        changes.setattr(authority, '_account_provider', failed)
        assert not fence.current()
    with monkeypatch.context() as changes:
        changes.setattr(authority, '_account_provider', None)
        assert not fence.current()
    with monkeypatch.context() as changes:
        changes.setattr(authority, '_usable', lambda: False)
        assert not fence.current()
    await authority.aclose()
    assert not fence.current()


async def test_person_fence_does_not_keep_its_authority_alive(tmp_path):
    authority = DashboardAuthority(tmp_path/'weak.db', expected_token=lambda: 'synthetic-weak-root',
        account_provider=lambda: ActiveAccounts(True, frozenset({'synthetic-a'})))
    await authority.initialize()
    pointer = ref(authority)
    fence = VoicePersonFence(pointer, ('synthetic-a',))
    assert fence.current()
    await authority.aclose()
    del authority
    await asyncio.sleep(0)
    gc.collect()
    assert pointer() is None
    assert not fence.current()
