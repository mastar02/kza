"""Durable authority failures never restore an older permission snapshot."""

import asyncio
import sqlite3
import threading
from uuid import uuid4

import pytest

from src.dashboard.authority import AuthorityRejected, DashboardAuthority, DashboardRole
from src.dashboard.credential_journal import CredentialAction as Action, CredentialCommand as Command, CredentialGrantSpec as Grant
from tests.unit.dashboard import test_credential_commands as fixtures

pytestmark = pytest.mark.asyncio
authority = fixtures.authority


@pytest.mark.parametrize('phase',['transaction','native_commit','publication'])
@pytest.mark.parametrize('failure',[PermissionError,AuthorityRejected])
async def test_committed_revision_keeps_ownership_when_publication_fails(authority,monkeypatch,phase,failure):
    service,now,root = authority
    actor = service.resolve(root[0])
    created = await service.commands.execute(actor,fixtures.creation(now[0]))
    target = created.receipt.credential_id
    command = Command(str(uuid4()),Action.REVISE,target,1,Grant(DashboardRole.OPERATOR,now[0]+90))
    if phase=='transaction':
        original = service._transaction
        def fail_transaction(kind,args,guard):
            state = original(kind,args,guard)
            if kind=='credential_command':
                raise failure('synthetic postcommit failure')
            return state
        monkeypatch.setattr(service,'_transaction',fail_transaction)
    elif phase=='native_commit':
        fail_once = [True]
        class FailingCommit(sqlite3.Connection):
            def commit(self):
                super().commit()
                if fail_once[0]:
                    fail_once[0] = False
                    raise failure('synthetic failure inside actual commit')
        original_connect = sqlite3.connect
        def connect(*args,**kwargs):
            return original_connect(*args,**kwargs,factory=FailingCommit)
        monkeypatch.setattr(sqlite3,'connect',connect)
    else:
        original = service._publish_state
        def fail_publication(state):
            original(state)
            raise failure('synthetic publication failure')
        monkeypatch.setattr(service,'_publish_state',fail_publication)
    with pytest.raises(RuntimeError,match='outcome_unknown'):
        await service.commands.execute(actor,command)
    assert not service.available
    assert target in service._credential_reservations
    assert service._revision_admission[target]==2
    assert service.resolve(created.token) is None
    assert await fixtures.rows(service,'SELECT revision FROM credentials WHERE id=?',(target,))==[(2,)]
    await service.aclose()
    restored = DashboardAuthority(service.path,clock=lambda:now[0],expected_token=lambda:root[0])
    await restored.initialize()
    try:
        replay = await restored.commands.execute(restored.resolve(root[0]),command)
        assert replay.receipt.result_revision==2 and not replay.applied and replay.token is None
    finally:
        await restored.aclose()


@pytest.mark.parametrize('key,value',[
    ('clock_floor','nan'),('clock_floor','inf'),('clock_floor','-1'),
    ('clock_floor',None),('legacy_epoch',None),('legacy_digest',None),
    ('legacy_epoch','-1'),('legacy_epoch','1.5'),('legacy_digest','invalid'),
])
async def test_corrupt_temporal_metadata_is_not_repaired(authority,key,value):
    service,now,root = authority
    await service.aclose()
    with sqlite3.connect(service.path) as db:
        if value is None:
            db.execute('DELETE FROM auth_meta WHERE key=?',(key,))
        else:
            db.execute('UPDATE auth_meta SET value=? WHERE key=?',(value,key))
    restored = DashboardAuthority(service.path,clock=lambda:now[0],expected_token=lambda:root[0])
    try:
        with pytest.raises(ValueError):
            await restored.initialize()
        assert not restored.available
        with sqlite3.connect(service.path) as db:
            observed = db.execute('SELECT value FROM auth_meta WHERE key=?',(key,)).fetchone()
            assert observed==(None if value is None else (value,))
    finally:
        await restored.aclose()


async def test_missing_unique_digest_constraint_is_not_repaired(authority):
    service,now,root = authority
    await service.aclose()
    with sqlite3.connect(service.path) as db:
        db.execute('ALTER TABLE credentials RENAME TO old_credentials')
        db.execute('''CREATE TABLE credentials (
            id TEXT PRIMARY KEY,digest TEXT NOT NULL,subject TEXT NOT NULL,role TEXT NOT NULL,
            user_id TEXT,issued_at REAL NOT NULL,expires_at REAL NOT NULL,revision INTEGER NOT NULL,
            samples TEXT NOT NULL)''')
        db.execute('INSERT INTO credentials SELECT * FROM old_credentials')
        db.execute('DROP TABLE old_credentials')
    restored = DashboardAuthority(service.path,clock=lambda:now[0],expected_token=lambda:root[0])
    try:
        with pytest.raises(ValueError,match='digest_index'):
            await restored.initialize()
        assert not restored.available
    finally:
        await restored.aclose()


@pytest.mark.parametrize('action',[Action.REVISE,Action.REVOKE])
async def test_partial_restore_cannot_resurrect_retired_permissions(authority,action):
    service,now,root = authority
    actor = service.resolve(root[0])
    created = await service.issue_credential(actor,subject='synthetic-restore',
        role=DashboardRole.HOME,expires_at=now[0]+90)
    original = (await fixtures.rows(service,'SELECT * FROM credentials WHERE id=?',(created.identity,)))[0]
    command = Command(str(uuid4()),action,created.identity,1,
        Grant(DashboardRole.OPERATOR,now[0]+90) if action is Action.REVISE else None)
    await service.commands.execute(actor,command)
    await service.aclose()
    with sqlite3.connect(service.path) as db:
        db.execute('INSERT OR REPLACE INTO credentials VALUES (?,?,?,?,?,?,?,?,?)',original)
    restored = DashboardAuthority(service.path,clock=lambda:now[0],expected_token=lambda:root[0])
    try:
        with pytest.raises(ValueError,match='credential_history'):
            await restored.initialize()
        assert not restored.available
        with sqlite3.connect(service.path) as db:
            assert db.execute('SELECT revision FROM credentials WHERE id=?',(created.identity,)).fetchone()==(1,)
    finally:
        await restored.aclose()


@pytest.mark.parametrize('column,value',[
    ('result_revision',2),('expected_revision',1),('reason','target_missing'),
    ('outcome','no_effect'),('credential_id','different-target'),
])
async def test_impossible_receipt_is_not_repaired(authority,column,value):
    service,now,root = authority
    await service.commands.execute(service.resolve(root[0]),fixtures.creation(now[0]))
    await service.aclose()
    assert column in {'result_revision','expected_revision','reason','outcome','credential_id'}
    with sqlite3.connect(service.path) as db:
        db.execute('UPDATE credential_attempts SET '+column+'=?',(value,))
    restored = DashboardAuthority(service.path,clock=lambda:now[0],expected_token=lambda:root[0])
    try:
        with pytest.raises(ValueError,match='credential_receipt'):
            await restored.initialize()
        assert not restored.available
    finally:
        await restored.aclose()


async def test_capacity_retains_old_receipts_and_catalog_requires_management(authority,monkeypatch):
    from src.dashboard import credential_commands, credential_journal
    monkeypatch.setattr(credential_commands,'MAX_CREDENTIAL_ATTEMPTS',2)
    monkeypatch.setattr(credential_journal,'MAX_CREDENTIAL_ATTEMPTS',2)
    service,now,root = authority
    actor = service.resolve(root[0])
    commands = [fixtures.creation(now[0]) for _ in range(3)]
    first,second = [await service.commands.execute(actor,command) for command in commands[:2]]
    with pytest.raises(AuthorityRejected,match='attempt_capacity'):
        await service.commands.execute(actor,commands[2])
    replay = await service.commands.execute(actor,commands[0])
    assert replay.receipt==first.receipt and replay.token is None and not replay.applied
    home = service.resolve(second.token)
    for operation in (lambda:service.commands.catalog(home),lambda:service.commands.attempt(home,commands[0].request_id)):
        with pytest.raises(PermissionError):
            operation()
    page = service.commands.catalog(actor,limit=1)
    other = service.commands.catalog(actor,after=page['next_cursor'],limit=1)
    assert len(page['rows'])==len(other['rows'])==1 and other['next_cursor'] is None
    assert page['rows'][0]['id']<other['rows'][0]['id']
    page['rows'][0]['subject']='detached'
    assert service.commands.catalog(actor,limit=1)['rows'][0]['subject']!='detached'


@pytest.mark.parametrize('stage',['before_start','queued'])
async def test_cancelled_owner_closes_authority_and_keeps_reservation(authority,monkeypatch,stage):
    service,now,root = authority
    actor = service.resolve(root[0])
    created = await service.issue_credential(actor,subject='synthetic-owner-cancellation',
        role=DashboardRole.HOME,expires_at=now[0]+90)
    if stage=='before_start':
        original = service._admit
        def cancel_before_start(*args,**kwargs):
            task = original(*args,**kwargs)
            task.cancel()
            return task
        monkeypatch.setattr(service,'_admit',cancel_before_start)
    else:
        await service._serial.acquire()
    observer = asyncio.create_task(service.revise_credential(actor,created.identity,
        role=DashboardRole.OPERATOR,expires_at=now[0]+90))
    try:
        if stage=='queued':
            for _ in range(3):
                await asyncio.sleep(0)
            owner = next(task for task,reservation in service._task_mutations.items()
                if reservation[0]==created.identity)
            owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        await asyncio.sleep(0)
        assert not service.available and created.identity in service._credential_reservations
        assert service.resolve(created.token) is None
    finally:
        if stage=='queued':
            service._serial.release()
        await asyncio.gather(observer,return_exceptions=True)


@pytest.mark.parametrize('table,column',[
    ('credentials','issued_at'),('sessions','issued_at'),
    ('retired_users','retired_at'),('credential_attempts','completed_at'),
])
async def test_persisted_event_after_raw_clock_floor_is_not_repaired(authority,table,column):
    service,now,root = authority
    actor = service.resolve(root[0])
    created = await service.commands.execute(actor,fixtures.creation(now[0]))
    await service.login(created.token)
    await service.retire_user(actor,'unrelated-synthetic-user')
    await service.aclose()
    with sqlite3.connect(service.path) as db:
        db.execute('UPDATE '+table+' SET '+column+'=?',(now[0]+1,))
    now[0] += 2
    restored = DashboardAuthority(service.path,clock=lambda:now[0],expected_token=lambda:root[0])
    try:
        with pytest.raises(ValueError,match='history_clock'):
            await restored.initialize()
        assert not restored.available
    finally:
        await restored.aclose()


async def test_revoked_identity_cannot_be_reused_by_legacy_or_command_issue(authority,monkeypatch):
    from src.dashboard import authority as module
    service,now,root = authority
    actor = service.resolve(root[0])
    created = await service.issue_credential(actor,subject='synthetic-tombstone',
        role=DashboardRole.HOME,expires_at=now[0]+90)
    revoke = Command(str(uuid4()),Action.REVOKE,created.identity,1)
    await service.commands.execute(actor,revoke)
    monkeypatch.setattr(module,'uuid4',lambda:created.identity)
    with pytest.raises(AuthorityRejected,match='credential_retired'):
        await service.issue_credential(actor,subject='synthetic-reuse',role=DashboardRole.HOME,expires_at=now[0]+90)
    intent = Command(created.identity,Action.ISSUE,grant=Grant(DashboardRole.HOME,now[0]+90,subject='synthetic-reuse'))
    result = await service.commands.execute(actor,intent)
    assert not result.applied and result.receipt.reason=='revision_conflict' and result.token is None
    assert service.available and service.commands.catalog(actor)['rows']==[]


@pytest.mark.parametrize('operation',['issue','revise'])
async def test_account_retirement_blocks_new_home_binding_at_final_commit(authority,monkeypatch,operation):
    from src.dashboard import credential_journal as journal
    service,now,root = authority
    actor = service.resolve(root[0])
    created = None
    if operation=='revise':
        created = await service.issue_credential(actor,subject='synthetic-prior-independent-home',
            role=DashboardRole.HOME,expires_at=now[0]+90)
    grant = Grant(DashboardRole.HOME,now[0]+90,user_id='synthetic-user',
        subject='synthetic-new-home' if created is None else None)
    intent = Command(str(uuid4()),Action.ISSUE if created is None else Action.REVISE,
        None if created is None else created.identity,None if created is None else 1,grant)
    entered,release = threading.Event(),threading.Event()
    original = journal.read_attempts
    reads = 0
    def held(db):
        nonlocal reads
        result = original(db)
        reads += 1
        if reads==2:
            entered.set()
            assert release.wait(5)
        return result
    monkeypatch.setattr(journal,'read_attempts',held)
    pending = asyncio.create_task(service.commands.execute(actor,intent))
    retirement = None
    try:
        assert await asyncio.to_thread(entered.wait,3)
        retirement = asyncio.create_task(service.retire_user(actor,'synthetic-user'))
        await asyncio.sleep(0)
        assert 'synthetic-user' in service._retiring_users
        release.set()
        result = await pending
        await retirement
        assert not result.applied and result.token is None and result.receipt.outcome.value=='no_effect'
        page = service.commands.catalog(actor)['rows']
        assert len(page)==(0 if created is None else 1)
        if created is not None:
            assert page[0]['revision']==1 and page[0]['user_id'] is None
            assert service.resolve(created.token).current()
        assert 'synthetic-user' in service._retired_users and service.available
    finally:
        release.set()
        await asyncio.gather(*[task for task in (pending,retirement) if task is not None],return_exceptions=True)


@pytest.mark.parametrize('action',[Action.REVISE,Action.REVOKE])
@pytest.mark.parametrize('revision',[True,0,1<<63])
async def test_invalid_revision_is_rejected_before_native_admission(authority,action,revision):
    service,now,_ = authority
    with pytest.raises(ValueError,match='credential_revision'):
        Command(str(uuid4()),action,str(uuid4()),revision,
            Grant(DashboardRole.HOME,now[0]+90) if action is Action.REVISE else None)
    assert service.available and not service._tasks and not service._credential_reservations
