"""Real credential journals, one-time secrets and shared writer reservations."""

import asyncio
from dataclasses import replace
import json
import sqlite3
import threading
from uuid import uuid4

import pytest

from src.dashboard.authority import AuthorityRejected, DashboardAuthority, DashboardRole
from src.dashboard.credential_journal import (
    CredentialAction as Action, CredentialCommand as Command, CredentialGrantSpec as Grant,
    CredentialOutcome as Outcome,
)
from tests.unit.dashboard import test_authority as authority_fixtures

pytestmark = pytest.mark.asyncio
authority = authority_fixtures.authority


def creation(now,*,role=DashboardRole.HOME,ttl=90.,subject='synthetic-command'):
    return Command(str(uuid4()),Action.ISSUE,grant=Grant(role,now+ttl,subject=subject))


async def rows(service,statement,args=()):
    def read():
        with sqlite3.connect(service.path) as db:
            return db.execute(statement,args).fetchall()
    return await service._work.run(read)


async def test_parallel_duplicate_creates_one_secret_and_returns_detached_metadata(authority,monkeypatch):
    from src.dashboard import credential_commands as module
    service,now,root = authority
    actor = service.resolve(root[0])
    command = creation(now[0])
    calls = []
    original = module.secrets.token_urlsafe
    def counted(size):
        calls.append(size)
        return original(size)
    monkeypatch.setattr(module.secrets,'token_urlsafe',counted)
    first,second = await asyncio.gather(service.commands.execute(actor,command),service.commands.execute(actor,command))
    assert first.applied and first.token and not second.applied and second.token is None
    assert first.receipt==second.receipt and calls==[32]
    assert service.resolve(first.token) is not None
    catalog = service.commands.catalog(actor)
    attempt = service.commands.attempt(actor,command.request_id)
    assert len(catalog['rows'])==1 and catalog['rows'][0]['id']==command.request_id
    document = json.dumps([catalog,attempt,first.receipt.public()])
    assert first.token not in document and 'digest' not in document and 'actor_key' not in document
    catalog['rows'][0]['subject'] = 'client mutation'
    assert service.commands.catalog(actor)['rows'][0]['subject']=='synthetic-command'
    assert first.token not in repr(first) and first.token not in repr(first.receipt)


@pytest.mark.parametrize('remove',['revoke','expiry'])
async def test_receipt_survives_credential_removal_and_never_recreates_secret(authority,remove):
    service,now,root = authority
    actor = service.resolve(root[0])
    command = creation(now[0],ttl=2. if remove=='expiry' else 90.)
    result = await service.commands.execute(actor,command)
    if remove=='revoke':
        revoke = Command(str(uuid4()),Action.REVOKE,command.request_id,1)
        assert (await service.commands.execute(actor,revoke)).receipt.outcome is Outcome.COMMITTED
    else:
        now[0]+=3
        await service.synchronize()
    assert service.resolve(result.token) is None
    replay = await service.commands.execute(service.resolve(root[0]),command)
    assert replay.receipt==result.receipt and replay.token is None and not replay.applied
    assert service.commands.catalog(service.resolve(root[0]))['rows']==[]
    assert await rows(service,'SELECT COUNT(*) FROM credential_attempts')==[(2 if remove=='revoke' else 1,)]


async def test_same_intent_cannot_change_command_or_original_login(authority):
    service,now,root = authority
    actor = service.resolve(root[0])
    command = creation(now[0])
    await service.commands.execute(actor,command)
    with pytest.raises(AuthorityRejected,match='attempt_conflict'):
        await service.commands.execute(actor,replace(command,grant=replace(command.grant,expires_at=now[0]+80)))
    session = await service.login(root[0])
    with pytest.raises(AuthorityRejected,match='attempt_conflict'):
        await service.commands.execute(session.lease,command)
    assert service.available and len(service.commands.catalog(actor)['rows'])==1


async def test_replay_survives_restart_after_legacy_revoke(authority):
    service,now,root = authority
    actor = service.resolve(root[0])
    command = creation(now[0])
    result = await service.commands.execute(actor,command)
    assert await service.revoke_credential(actor,command.request_id)
    await service.aclose()
    restored = DashboardAuthority(service.path,clock=lambda:now[0],expected_token=lambda:root[0])
    await restored.initialize()
    try:
        replay = await restored.commands.execute(restored.resolve(root[0]),command)
        assert replay.receipt==result.receipt and not replay.applied and replay.token is None
        assert restored.commands.catalog(restored.resolve(root[0]))['rows']==[]
    finally:
        await restored.aclose()


async def test_cas_loser_never_seals_winning_revision_or_revocation(authority):
    service,now,root = authority
    actor = service.resolve(root[0])
    created = await service.commands.execute(actor,creation(now[0]))
    target = created.receipt.credential_id
    revise = Command(str(uuid4()),Action.REVISE,target,1,Grant(DashboardRole.OPERATOR,now[0]+90))
    winner = await service.commands.execute(actor,revise)
    current = service.resolve(created.token)
    assert current is not None and current.principal.revision==2
    losing_revoke = Command(str(uuid4()),Action.REVOKE,target,1)
    loser = await service.commands.execute(actor,losing_revoke)
    assert loser.receipt.outcome is Outcome.NO_EFFECT and loser.receipt.reason=='revision_conflict'
    assert current.current() and service.resolve(created.token) is not None
    assert not service._credential_reservations and target not in service._credential_revoking
    assert target not in service._revision_admission
    replay = await service.commands.execute(actor,losing_revoke)
    assert replay.receipt==loser.receipt and current.current()
    third = await service.commands.execute(actor,Command(str(uuid4()),Action.REVISE,target,2,
        Grant(DashboardRole.HOME,now[0]+90)))
    assert winner.receipt.result_revision==2 and third.receipt.result_revision==3
    assert not current.current() and service.available


@pytest.mark.parametrize('direction',['legacy_first','command_first'])
async def test_legacy_and_cas_share_target_ownership_until_publication(authority,monkeypatch,direction):
    service,now,root = authority
    actor = service.resolve(root[0])
    created = await service.commands.execute(actor,creation(now[0]))
    target = created.receipt.credential_id
    entered,release = threading.Event(),threading.Event()
    original = service._transaction
    action = 'revise' if direction=='legacy_first' else 'credential_command'
    def held(kind,args,guard):
        if kind==action:
            entered.set()
            assert release.wait(5)
        return original(kind,args,guard)
    monkeypatch.setattr(service,'_transaction',held)
    command = Command(str(uuid4()),Action.REVISE,target,1,Grant(DashboardRole.OPERATOR,now[0]+90))
    first = asyncio.create_task(service.revise_credential(actor,target,role=DashboardRole.OPERATOR,expires_at=now[0]+90)
        if direction=='legacy_first' else service.commands.execute(actor,command))
    second = None
    try:
        assert await asyncio.to_thread(entered.wait,3)
        if direction=='command_first':
            with pytest.raises(AuthorityRejected,match='writer_busy'):
                await service.revoke_credential(actor,target)
        else:
            second = asyncio.create_task(service.commands.execute(actor,command))
        release.set()
        await first
        if second is not None:
            loser = await second
            assert loser.receipt.outcome is Outcome.NO_EFFECT
        assert service.resolve(created.token).principal.revision==2
        assert not service._credential_reservations and not service._credential_revoking
        assert target not in service._revision_admission
    finally:
        release.set()
        await asyncio.gather(*[task for task in (first,second) if task is not None],return_exceptions=True)


@pytest.mark.parametrize('gate',['snapshot','receipt'])
async def test_final_guard_rolls_back_effect_and_persists_no_effect_snapshot(authority,monkeypatch,gate):
    service,now,root = authority
    session = await service.login(root[0])
    actor = session.lease
    command = creation(now[0])
    entered,release = threading.Event(),threading.Event()
    from src.dashboard import credential_journal as module
    owner,name = (service,'_read') if gate=='snapshot' else (module,'read_attempts')
    original = getattr(owner,name)
    reads = 0
    def held(db):
        nonlocal reads
        result = original(db)
        reads += 1
        if reads==(1 if gate=='snapshot' else 2):
            entered.set()
            assert release.wait(5)
        return result
    monkeypatch.setattr(owner,name,held)
    pending = asyncio.create_task(service.commands.execute(actor,command))
    logout = None
    try:
        assert await asyncio.to_thread(entered.wait,3)
        logout = asyncio.create_task(service.logout(actor))
        await asyncio.sleep(.01)
        assert not actor.current()
        release.set()
        result = await pending
        assert result.receipt.outcome is Outcome.NO_EFFECT and result.token is None
        await logout
        root_actor = service.resolve(root[0])
        assert service.commands.catalog(root_actor)['rows']==[] and service.available
        assert service.commands.attempt(root_actor,command.request_id)['receipt']['outcome']=='no_effect'
        assert await rows(service,'SELECT COUNT(*) FROM credentials')==[(0,)]
    finally:
        release.set()
        await asyncio.gather(*[task for task in (pending,logout) if task is not None],return_exceptions=True)


async def test_cancelled_first_observer_retains_work_and_replay_has_no_secret(authority,monkeypatch):
    service,now,root = authority
    actor = service.resolve(root[0])
    command = creation(now[0])
    entered,release = threading.Event(),threading.Event()
    original = service._transaction
    def held(kind,args,guard):
        if kind=='credential_command':
            entered.set()
            assert release.wait(5)
        return original(kind,args,guard)
    monkeypatch.setattr(service,'_transaction',held)
    pending = asyncio.create_task(service.commands.execute(actor,command))
    try:
        assert await asyncio.to_thread(entered.wait,3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert service._tasks and service.commands.attempt(actor,command.request_id)['state']=='pending'
        release.set()
        async with asyncio.timeout(3):
            while service._tasks:
                await asyncio.sleep(.001)
        replay = await service.commands.execute(actor,command)
        assert replay.receipt.outcome is Outcome.COMMITTED and replay.token is None and not replay.applied
        assert len(service.commands.catalog(actor)['rows'])==1
    finally:
        release.set()
        await asyncio.gather(pending,return_exceptions=True)


@pytest.mark.parametrize('failure',[OSError,PermissionError,AuthorityRejected])
async def test_postcommit_exception_is_unknown_until_reopen_proves_receipt(authority,monkeypatch,failure):
    service,now,root = authority
    actor = service.resolve(root[0])
    command = creation(now[0])
    original = service._transaction
    def uncertain(kind,args,guard):
        result = original(kind,args,guard)
        if kind=='credential_command':
            raise failure('synthetic failure after actual commit')
        return result
    monkeypatch.setattr(service,'_transaction',uncertain)
    expected = OSError if failure is OSError else RuntimeError
    with pytest.raises(expected):
        await service.commands.execute(actor,command)
    assert not service.available
    with pytest.raises(PermissionError):
        service.commands.attempt(actor,command.request_id)
    await service.aclose()
    restored = DashboardAuthority(service.path,clock=lambda:now[0],expected_token=lambda:root[0])
    await restored.initialize()
    try:
        result = await restored.commands.execute(restored.resolve(root[0]),command)
        assert result.receipt.outcome is Outcome.COMMITTED and result.token is None and not result.applied
    finally:
        await restored.aclose()


async def test_schema3_missing_journal_is_never_repaired(authority):
    service,now,root = authority
    await service.aclose()
    with sqlite3.connect(service.path) as db:
        db.execute('DROP TABLE credential_attempts')
    reopened = DashboardAuthority(service.path,clock=lambda:now[0],expected_token=lambda:root[0])
    try:
        with pytest.raises(ValueError):
            await reopened.initialize()
        assert not reopened.available
        with sqlite3.connect(service.path) as db:
            assert db.execute("SELECT 1 FROM sqlite_master WHERE name='credential_attempts'").fetchone() is None
    finally:
        await reopened.aclose()


@pytest.mark.parametrize('fail',[False,True])
async def test_schema2_upgrade_is_atomic(authority,monkeypatch,fail):
    from src.dashboard import credential_journal as module
    service,now,root = authority
    await service.aclose()
    with sqlite3.connect(service.path) as db:
        db.execute('DROP TABLE credential_attempts')
        db.execute("UPDATE auth_meta SET value='2' WHERE key='schema_version'")
    if fail:
        original = module.validate_table
        def failing(db,name):
            original(db,name)
            if name=='credential_attempts':
                raise OSError('synthetic DDL boundary failure')
        monkeypatch.setattr(module,'validate_table',failing)
    reopened = DashboardAuthority(service.path,clock=lambda:now[0],expected_token=lambda:root[0])
    try:
        if fail:
            with pytest.raises(OSError):
                await reopened.initialize()
        else:
            await reopened.initialize()
            assert reopened.available
        with sqlite3.connect(service.path) as db:
            assert db.execute("SELECT value FROM auth_meta WHERE key='schema_version'").fetchone()[0]==('2' if fail else '3')
            assert bool(db.execute("SELECT 1 FROM sqlite_master WHERE name='credential_attempts'").fetchone()) is not fail
    finally:
        await reopened.aclose()
