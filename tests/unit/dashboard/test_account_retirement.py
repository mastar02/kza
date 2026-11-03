"""Synthetic canonical accounts and durable retirement, without voice or household IO."""

import asyncio
from contextlib import AsyncExitStack, closing
import os
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from src.core.account_metadata import ActiveAccounts
from src.dashboard.authority import DashboardAuthority, DashboardRole as Role
from src.users.user_manager import PermissionLevel, UserManager


@pytest.fixture
async def graph(tmp_path):
    users = UserManager(str(tmp_path / 'users.json'))
    target, _ = users.add_user('Synthetic target', PermissionLevel.ADULT)
    other, _ = users.add_user('Synthetic other', PermissionLevel.ADULT)
    now, token = [1_700_000_000.], 'synthetic-retirement-root'
    async with AsyncExitStack() as stack:
        service = DashboardAuthority(tmp_path / 'auth.db', clock=lambda: now[0],
                                     expected_token=lambda: token, account_provider=users.active_accounts)
        users._retirement_proof = service.users_retired
        stack.push_async_callback(service.aclose)
        await service.initialize()
        yield SimpleNamespace(users=users, target=target.user_id, other=other.user_id,
                              service=service, root=service.resolve(token), now=now, token=token)


async def issue(graph, identity):
    return await graph.service.issue_credential(graph.root, subject='synthetic', role=Role.HOME,
                                                expires_at=graph.now[0] + 120, user_id=identity)


async def wait(event):
    assert await asyncio.to_thread(event.wait, 3)


async def test_retirement_fences_all_bindings_before_sql_and_survives_observer_cancel(graph, monkeypatch):
    service = graph.service
    first, second, other = await issue(graph, graph.target), await issue(graph, graph.target), await issue(graph, graph.other)
    sessions = [await service.login(first.token), await service.login(first.token), await service.login(second.token)]
    leases = [service.resolve(first.token), service.resolve(second.token), *(item.lease for item in sessions)]
    entered, release = threading.Event(), threading.Event()
    transaction = service._transaction

    def held(action, args, guard):
        if action == 'retire_user':
            entered.set()
            assert release.wait(5)
        return transaction(action, args, guard)

    monkeypatch.setattr(service, '_transaction', held)
    pending = asyncio.create_task(service.retire_user(graph.root, graph.target))
    try:
        await wait(entered)
        assert all(not lease.current() for lease in leases)
        assert service.resolve(other.token).current()
        assert not service.users_retired(frozenset({graph.target}))
        with pytest.raises(ValueError, match='personal_binding'):
            await issue(graph, graph.target)
        with pytest.raises(ValueError, match='personal_binding'):
            await service.revise_credential(graph.root, other.identity, role=Role.HOME,
                                            expires_at=graph.now[0] + 100, user_id=graph.target)
        retained = tuple(service._tasks)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    finally:
        release.set()
    await asyncio.gather(*retained)
    assert service.users_retired(frozenset({graph.target}))
    assert not service.users_retired(frozenset({graph.other}))
    assert service.resolve(other.token).current()
    assert all(service.resolve(item.token) is None for item in (first, second, *sessions))
    assert graph.users.remove_user(graph.target)[0]
    await service.aclose()
    replacement = DashboardAuthority(service.path, clock=lambda: graph.now[0], expected_token=lambda: graph.token,
                                      account_provider=graph.users.active_accounts)
    try:
        await replacement.initialize()
        assert replacement.users_retired(frozenset({graph.target}))
        assert replacement.resolve(first.token) is None
        assert replacement.resolve(other.token).current()
    finally:
        await replacement.aclose()


async def test_retirement_captures_binding_revised_away_in_commit_publication_gap(graph, monkeypatch):
    service = graph.service
    moving, unrelated = await issue(graph, graph.target), await issue(graph, graph.other)
    entered, release = threading.Event(), threading.Event()
    transaction = service._transaction

    def held(action, args, guard):
        result = transaction(action, args, guard)
        if action == 'revise':
            entered.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(service, '_transaction', held)
    revision = asyncio.create_task(service.revise_credential(graph.root, moving.identity, role=Role.HOME,
                                   expires_at=graph.now[0] + 100, user_id=graph.other))
    retirement = None
    try:
        await wait(entered)
        retirement = asyncio.create_task(service.retire_user(graph.root, graph.target))
        await asyncio.sleep(0)
        assert service.resolve(moving.token) is None
    finally:
        release.set()
        await revision
        if retirement is not None:
            await retirement
    assert service.resolve(moving.token) is None
    assert service.resolve(unrelated.token).current()


@pytest.mark.parametrize('operation', ['issue', 'revise'])
async def test_canonical_membership_is_checked_again_before_native_commit(graph, monkeypatch, operation):
    service = graph.service
    credential = await issue(graph, graph.other)
    snapshot = [graph.users.active_accounts()]
    service._account_provider = lambda: snapshot[0]
    entered, release = threading.Event(), threading.Event()
    transaction = service._transaction

    def held(action, args, guard):
        if action == operation:
            entered.set()
            assert release.wait(5)
        return transaction(action, args, guard)

    monkeypatch.setattr(service, '_transaction', held)
    if operation == 'issue':
        pending = asyncio.create_task(issue(graph, graph.target))
    else:
        pending = asyncio.create_task(service.revise_credential(graph.root, credential.identity, role=Role.HOME,
                                    expires_at=graph.now[0] + 100, user_id=graph.target))
    try:
        await wait(entered)
        snapshot[0] = ActiveAccounts(True, frozenset({graph.other}), snapshot[0].revision + 1)
    finally:
        release.set()
    with pytest.raises(PermissionError):
        await pending
    with closing(sqlite3.connect(service.path)) as db:
        assert db.execute('SELECT COUNT(*) FROM credentials WHERE user_id=?', (graph.target,)).fetchone()[0] == 0


async def test_retirement_removes_new_binding_committed_before_ram_publication(graph, monkeypatch):
    service = graph.service
    entered, release = threading.Event(), threading.Event()
    transaction = service._transaction

    def held(action, args, guard):
        result = transaction(action, args, guard)
        if action == 'issue':
            entered.set()
            assert release.wait(5)
        return result

    monkeypatch.setattr(service, '_transaction', held)
    issuing = asyncio.create_task(issue(graph, graph.target))
    retirement = None
    try:
        await wait(entered)
        retirement = asyncio.create_task(service.retire_user(graph.root, graph.target))
        await asyncio.sleep(0)
    finally:
        release.set()
    with pytest.raises(PermissionError, match='dashboard_grant_expired'):
        await issuing
    if retirement is not None:
        await retirement
    with closing(sqlite3.connect(service.path)) as db:
        assert db.execute('SELECT COUNT(*) FROM credentials WHERE user_id=?', (graph.target,)).fetchone()[0] == 0


async def test_actor_logout_before_retirement_commit_leaves_no_durable_proof(graph, monkeypatch):
    service = graph.service
    credential = await service.issue_credential(graph.root, subject='synthetic-operator', role=Role.OPERATOR,
                                               expires_at=graph.now[0] + 120)
    session = await service.login(credential.token)
    entered, release = threading.Event(), threading.Event()
    transaction = service._transaction

    def held(action, args, guard):
        if action == 'retire_user':
            entered.set()
            assert release.wait(5)
        return transaction(action, args, guard)

    monkeypatch.setattr(service, '_transaction', held)
    pending = asyncio.create_task(service.retire_user(session.lease, graph.target))
    logout = None
    try:
        await wait(entered)
        logout = asyncio.create_task(service.logout(session.lease))
        await asyncio.sleep(0)
        assert not session.lease.current()
    finally:
        release.set()
    with pytest.raises(PermissionError):
        await pending
    if logout is not None:
        await logout
    assert not service.users_retired(frozenset({graph.target}))
    with pytest.raises(PermissionError, match='account_retirement_required'):
        graph.users.remove_user(graph.target)
    assert graph.users.get_user(graph.target) is not None
    assert await service.retire_user(graph.root, graph.target)
    assert graph.users.remove_user(graph.target)[0]


async def test_restoring_old_credential_rows_cannot_override_durable_retirement(graph):
    service = graph.service
    credential = await issue(graph, graph.target)
    session = await service.login(credential.token)
    with closing(sqlite3.connect(service.path)) as db:
        row = db.execute('SELECT * FROM credentials WHERE user_id=?', (graph.target,)).fetchone()
        session_row = db.execute('SELECT * FROM sessions').fetchone()
    await service.retire_user(graph.root, graph.target)
    assert graph.users.get_user(graph.target) is not None
    await service.aclose()
    with closing(sqlite3.connect(service.path)) as db, db:
        db.execute('INSERT INTO credentials VALUES (' + ','.join('?' for _ in row) + ')', row)
        db.execute('INSERT INTO sessions VALUES (' + ','.join('?' for _ in session_row) + ')', session_row)
    replacement = DashboardAuthority(service.path, clock=lambda: graph.now[0], expected_token=lambda: graph.token,
                                      account_provider=graph.users.active_accounts)
    try:
        await replacement.initialize()
        assert replacement.resolve(credential.token) is None
        assert replacement.resolve(None, session.token) is None
        assert replacement.users_retired(frozenset({graph.target}))
    finally:
        await replacement.aclose()


async def test_personal_lease_guard_does_not_wait_for_catalog_lock(graph):
    credential = await issue(graph, graph.target)
    lease = graph.service.resolve(credential.token)
    entered, release = threading.Event(), threading.Event()

    def hold():
        with graph.users._lock:
            entered.set()
            assert release.wait(5)

    blocker = asyncio.create_task(asyncio.to_thread(hold))
    try:
        await wait(entered)
        assert await asyncio.wait_for(asyncio.to_thread(lease.current), .5)
    finally:
        release.set()
        await blocker


@pytest.mark.parametrize('unavailable', [None, 'wrong-type', ActiveAccounts(False, frozenset()),
                                       ActiveAccounts(True, frozenset()), 'raises'])
async def test_unknown_or_inactive_canonical_account_retires_personal_lease_only(graph, unavailable):
    credential = await issue(graph, graph.target)
    session = await graph.service.login(credential.token)

    def provider():
        if unavailable == 'raises':
            raise RuntimeError('synthetic canonical failure')
        return unavailable

    graph.service._account_provider = provider
    assert not session.lease.current()
    assert graph.service.resolve(credential.token) is None
    assert graph.root.current()
    with pytest.raises(ValueError, match='personal_binding'):
        await issue(graph, graph.target)


async def test_durable_retirement_allows_retry_after_catalog_replace_failure(graph, monkeypatch):
    import src.users.user_manager as catalog

    credential = await issue(graph, graph.target)
    before, snapshot = graph.users.data_path.read_bytes(), graph.users.active_accounts()
    await graph.service.retire_user(graph.root, graph.target)
    replace = catalog.os.replace

    def fail(*args):
        raise OSError('synthetic private path must not escape')

    monkeypatch.setattr(catalog.os, 'replace', fail)
    with pytest.raises(OSError):
        graph.users.remove_user(graph.target)
    assert graph.users.data_path.read_bytes() == before
    assert graph.users.active_accounts() is snapshot
    assert graph.service.resolve(credential.token) is None
    with pytest.raises(ValueError, match='personal_binding'):
        await issue(graph, graph.target)
    monkeypatch.setattr(catalog.os, 'replace', replace)
    assert graph.users.remove_user(graph.target)[0]
    assert graph.target not in graph.users.active_accounts().active_ids


@pytest.mark.parametrize('version', ['1', '99'])
async def test_invalid_schema_is_rejected_before_creating_tables(tmp_path, version):
    path = tmp_path / 'auth.db'
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE auth_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        db.execute('INSERT INTO auth_meta VALUES (?,?)', ('schema_version', version))
    os.chmod(path, 0o600)
    before = path.read_bytes()
    service = DashboardAuthority(path, expected_token=lambda: 'synthetic-root')
    try:
        reason = 'unsupported_dashboard_authority_schema' if version=='99' else 'invalid_dashboard_authority_table'
        with pytest.raises(ValueError,match=reason):
            await service.initialize()
        assert path.read_bytes()==before
        assert not service.available
    finally:
        await service.aclose()


async def test_current_schema_missing_retirements_is_rejected_without_repair(graph):
    await graph.service.aclose()
    with closing(sqlite3.connect(graph.service.path)) as db, db:
        db.execute('DROP TABLE retired_users')
    before = graph.service.path.read_bytes()
    service = DashboardAuthority(graph.service.path, expected_token=lambda: graph.token)
    try:
        with pytest.raises(ValueError, match='invalid_dashboard_authority_table'):
            await service.initialize()
        assert graph.service.path.read_bytes() == before
        assert not service.available
    finally:
        await service.aclose()


async def test_schema_one_migration_preserves_valid_roles_and_sessions(graph):
    credential = await issue(graph, graph.target)
    session = await graph.service.login(credential.token)
    await graph.service.aclose()
    with closing(sqlite3.connect(graph.service.path)) as db, db:
        db.execute("UPDATE auth_meta SET value='1' WHERE key='schema_version'")
        db.execute('DROP TABLE retired_users')
        db.execute('DROP TABLE credential_attempts')
    service = DashboardAuthority(graph.service.path, clock=lambda: graph.now[0], expected_token=lambda: graph.token,
                                  account_provider=graph.users.active_accounts)
    try:
        await service.initialize()
        lease = service.resolve(None, session.token)
        assert lease.current()
        assert lease.principal.role is Role.HOME
        assert lease.principal.user_id == graph.target
        assert service.resolve(credential.token).current()
        with closing(sqlite3.connect(service.path)) as db:
            assert db.execute("SELECT value FROM auth_meta WHERE key='schema_version'").fetchone()[0]=='3'
            assert db.execute('SELECT COUNT(*) FROM credential_attempts').fetchone()[0]==0
            assert db.execute('SELECT COUNT(*) FROM retired_users').fetchone()[0]==0
    finally:
        await service.aclose()
