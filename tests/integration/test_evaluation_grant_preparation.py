"""Native real-source grant preparation, exact issuer and final commit fences."""

import asyncio
from dataclasses import replace
import json
import sqlite3
import threading

import pytest

from src.dashboard.authority import DashboardRole, EvaluationModality
from src.dashboard.evaluation_actor import EvaluationReadActor
from src.dashboard.evaluation_grants import EvaluationGrantSelection
from src.memory.fact_ledger import FactLedger
from src.users.voice_evaluation import VoiceEvaluationReader
from src.users.voice_ledger import VoiceLedger
from tests.integration import test_voice_evaluation_reader as native_fixtures
from tests.integration.test_capture_producer_privacy import until
from tests.integration.test_voice_evaluation_reader import sql
from tests.mocks.voice_registry import VECTOR

pytestmark = pytest.mark.asyncio
evaluation_graph = native_fixtures.evaluation_graph


async def prepare(g,*,issuer=None,selection_ttl=20.,credential_ttl=90.,modes=None):
    issuer = g.root if issuer is None else issuer
    selection = EvaluationGrantSelection(g.identity,frozenset(EvaluationModality) if modes is None else modes,
        g.clock.now+selection_ttl)
    return await g.voices.acall(g.voices.evaluation.prepare_grant,issuer,(selection,),g.clock.now+credential_ttl)


async def issue(g,scope,*,actor=None,**changes):
    values = dict(subject='synthetic-prepared-evaluator',role=DashboardRole.EVALUATOR,
        expires_at=scope.credential_expires_at,samples=scope.samples,prepared_scope=scope)
    values.update(changes)
    return await g.authority.issue_credential(scope.issuer if actor is None else actor,**values)


async def operator_session(g,*,session_ttl=None):
    credential = await g.authority.issue_credential(g.root,subject='synthetic-proof-issuer',
        role=DashboardRole.OPERATOR,expires_at=g.clock.now+90)
    previous = g.authority.session_ttl_s
    if session_ttl is not None:
        g.authority.session_ttl_s = session_ttl
    try:
        session = await g.authority.login(credential.token)
    finally:
        g.authority.session_ttl_s = previous
    return credential,session


async def test_metadata_preparation_never_reads_either_payload_or_operator_extras(evaluation_graph):
    g = evaluation_graph
    await sql(g,'UPDATE voice_capture_reviews SET audio_wav=?,transcript=CAST(? AS TEXT) WHERE fingerprint_id=?',
        (b'corrupt unselected audio',b'private-invalid-text'+bytes([255]),g.identity))
    allowed = {'voice_fingerprints':{'id','sources_json','expires_at'},
        'voice_capture_reviews':{'fingerprint_id'},'facts':{'id','revision','expires_at','state'},
        'conversation_tombstones':{'conversation_id','deleted_at'}}
    reads = []
    def authorize(operation,table,column,*_):
        if operation==sqlite3.SQLITE_READ:
            reads.append((table,column))
            if table not in allowed or column not in allowed[table]:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    selection = EvaluationGrantSelection(g.identity,frozenset(EvaluationModality),g.clock.now+20)
    def native():
        g.privacy.ledger._db.set_authorizer(authorize)
        try:
            return g.voices.evaluation.prepare_grant(g.root,(selection,),g.clock.now+90)
        finally:
            g.privacy.ledger._db.set_authorizer(None)
    scope = await g.voices.acall(native)
    assert scope.samples[0].source_digests==g.digests
    assert scope.samples[0].modalities==frozenset(EvaluationModality)
    assert ('voice_capture_reviews','fingerprint_id') in reads
    assert not any(column in {'audio_wav','transcript','room_id','model_key','embedding'} for _,column in reads)
    assert scope.current(g.authority._now())


@pytest.mark.parametrize('modality',list(EvaluationModality))
async def test_prepared_actual_scope_issues_a_readable_exact_grant(evaluation_graph,modality):
    g = evaluation_graph
    scope = await prepare(g,modes=frozenset({modality}))
    credential = await issue(g,scope)
    session = await g.authority.login(credential.token)
    actor = EvaluationReadActor(session.lease,g.identity,modality)
    projection = await g.voices.acall(g.voices.evaluation.read,actor)
    assert projection.rows[0]['modality']==modality.value
    assert projection.sources==scope.records[0].sources


async def test_prepared_source_hashes_bind_complete_stored_declarations(evaluation_graph):
    from src.dashboard.authority import source_digest
    g = evaluation_graph
    values = [source.to_dict() for source in g.sources]
    values[0]['original_extra_metadata'] = 'synthetic-declaration'
    await sql(g,'UPDATE voice_fingerprints SET sources_json=? WHERE id=?',(json.dumps(values),g.identity))
    scope = await prepare(g)
    assert scope.samples[0].source_digests==frozenset(source_digest(value) for value in values)
    assert scope.samples[0].source_digests!=g.digests


@pytest.mark.parametrize('bound',['credential','selection','fingerprint','source'])
async def test_preparation_caps_exact_effective_deadline(evaluation_graph,bound):
    g = evaluation_graph
    credential_ttl,selection_ttl = (3.25,80.) if bound=='credential' else (90.,4.5) if bound=='selection' else (90.,80.)
    if bound=='fingerprint':
        await sql(g,'UPDATE voice_fingerprints SET expires_at=? WHERE id=?',(g.clock.now+5.75,g.identity))
    scope = await prepare(g,credential_ttl=credential_ttl,selection_ttl=selection_ttl)
    expected = {'credential':3.25,'selection':4.5,'fingerprint':5.75,'source':60.}[bound]
    assert scope.samples[0].expires_at==g.clock.now+expected
    g.clock.now+=expected
    assert not scope.current(g.authority._now())


async def test_issuer_session_expiry_does_not_cap_an_authorized_issued_grant(evaluation_graph):
    g = evaluation_graph
    _,session = await operator_session(g,session_ttl=1.)
    scope = await prepare(g,issuer=session.lease)
    assert scope.samples[0].expires_at==g.clock.now+20
    credential = await issue(g,scope)
    g.clock.now+=2
    assert not session.lease.current() and scope.current(g.authority._now())
    assert g.authority.resolve(credential.token) is not None


@pytest.mark.parametrize('operation',['issue','revise'])
@pytest.mark.parametrize('change',['issuer','role','samples','expiry','invalid_object'])
async def test_prepared_proof_never_binds_another_issuer_or_command(evaluation_graph,change,operation):
    g = evaluation_graph
    scope = await prepare(g)
    options = {}
    actor = scope.issuer
    if change=='issuer':
        _,session = await operator_session(g)
        actor = session.lease
    elif change=='role':
        options.update(role=DashboardRole.OPERATOR,samples=())
    elif change=='samples':
        options['samples'] = (replace(scope.samples[0],expires_at=scope.samples[0].expires_at-1),)
    elif change=='expiry':
        options['expires_at'] = scope.credential_expires_at-1
    else:
        options['prepared_scope'] = False
    target = session = None
    if operation=='revise':
        target = await g.authority.issue_credential(g.root,subject='synthetic-mismatch-home',
            role=DashboardRole.HOME,expires_at=g.clock.now+90)
        session = await g.authority.login(target.token)
    before = len(g.authority._credentials)
    with pytest.raises(PermissionError,match='scope_mismatch'):
        if operation=='issue':
            await issue(g,scope,actor=actor,**options)
        else:
            values = dict(role=DashboardRole.EVALUATOR,expires_at=scope.credential_expires_at,
                samples=scope.samples,prepared_scope=scope)
            values.update(options)
            await g.authority.revise_credential(actor,target.identity,**values)
    assert g.authority.available and len(g.authority._credentials)==before
    if session is not None:
        assert session.lease.current() and g.authority.resolve(target.token) is not None
        assert target.identity not in g.authority._revision_admission


async def test_invalid_scope_before_revision_admission_keeps_old_lease_current(evaluation_graph):
    g = evaluation_graph
    target = await g.authority.issue_credential(g.root,subject='synthetic-old-home',role=DashboardRole.HOME,
        expires_at=g.clock.now+90)
    session = await g.authority.login(target.token)
    scope = await prepare(g)
    g.privacy.ledger._pending_conversations = frozenset({g.sources[0].conversation_id})
    try:
        with pytest.raises(PermissionError,match='scope_retired'):
            await g.authority.revise_credential(g.root,target.identity,role=DashboardRole.EVALUATOR,
                expires_at=scope.credential_expires_at,samples=scope.samples,prepared_scope=scope)
    finally:
        g.privacy.ledger._pending_conversations = frozenset()
    assert session.lease.current() and g.authority.resolve(target.token) is not None
    assert target.identity not in g.authority._revision_admission and g.authority.available


async def test_self_revision_uses_native_issuer_guard_and_data_only_scope(evaluation_graph):
    g = evaluation_graph
    credential,session = await operator_session(g)
    scope = await prepare(g,issuer=session.lease)
    assert await g.authority.revise_credential(session.lease,credential.identity,role=DashboardRole.EVALUATOR,
        expires_at=scope.credential_expires_at,samples=scope.samples,prepared_scope=scope)
    assert not session.lease.current() and scope.current(g.authority._now())
    current = g.authority.resolve(credential.token)
    assert current is not None and current.principal.role is DashboardRole.EVALUATOR
    assert current.principal.samples==scope.samples and g.authority.available


@pytest.mark.parametrize('operation',['issue','revise'])
@pytest.mark.parametrize('change',['sample','source'])
async def test_scope_is_rechecked_at_actual_final_commit(evaluation_graph,monkeypatch,operation,change):
    g = evaluation_graph
    target = None
    if operation=='revise':
        target = await g.authority.issue_credential(g.root,subject='synthetic-existing-home',
            role=DashboardRole.HOME,expires_at=g.clock.now+90)
    scope = await prepare(g)
    before = len(g.authority._credentials)
    entered,release = threading.Event(),threading.Event()
    original = g.authority._read
    def held(db):
        result = original(db)
        entered.set()
        assert release.wait(5)
        return result
    monkeypatch.setattr(g.authority,'_read',held)
    pending = None
    try:
        if operation=='issue':
            pending = asyncio.create_task(issue(g,scope))
        else:
            pending = asyncio.create_task(g.authority.revise_credential(g.root,target.identity,role=DashboardRole.EVALUATOR,
                expires_at=scope.credential_expires_at,samples=scope.samples,prepared_scope=scope))
        assert await asyncio.to_thread(entered.wait,3)
        if change=='sample':
            g.clock.now = scope.samples[0].expires_at
        else:
            g.privacy.ledger._pending_conversations = frozenset({g.sources[0].conversation_id})
        release.set()
        with pytest.raises(PermissionError):
            await pending
    finally:
        release.set()
        g.privacy.ledger._pending_conversations = frozenset()
        if pending is not None:
            await asyncio.gather(pending,return_exceptions=True)
    assert g.authority.available and len(g.authority._credentials)==before
    def rows():
        with sqlite3.connect(g.authority.path) as db:
            return db.execute('SELECT id,role,revision FROM credentials').fetchall()
    stored = await g.voices.acall(rows)
    assert len(stored)==before
    if target is not None:
        assert stored==[(target.identity,'home',1)]


async def test_closed_and_reopened_memory_ledger_never_revives_prepared_scope(evaluation_graph):
    g = evaluation_graph
    ledger = FactLedger(':memory:',clock=lambda:g.clock.now)
    voice = VoiceLedger(ledger)
    try:
        identity = await g.voices.acall(voice.put,'synthetic-model',VECTOR.copy(),3.,g.sources,g.review)
        reader = VoiceEvaluationReader(ledger)
        selection = EvaluationGrantSelection(identity,frozenset(EvaluationModality),g.clock.now+20)
        scope = await g.voices.acall(reader.prepare_grant,g.root,(selection,),g.clock.now+90)
        assert scope.current(g.authority._now())
        opening = ledger.lifecycle_generation
        await g.voices.acall(ledger.close)
        assert not scope.current(g.authority._now())
        await g.voices.acall(ledger.initialize)
        assert ledger.lifecycle_generation>opening
        assert ledger.sources_available(scope.records[0].sources)
        assert not scope.current(g.authority._now())
        with pytest.raises(PermissionError,match='scope_retired'):
            await issue(g,scope)
    finally:
        await g.voices.acall(ledger.close)


async def test_observed_authority_clock_floor_cannot_reopen_sample_deadline(evaluation_graph):
    g = evaluation_graph
    scope = await prepare(g,selection_ttl=3.25)
    g.clock.now = scope.samples[0].expires_at
    assert not scope.current(g.authority._now())
    g.clock.now-=2
    assert not scope.current(g.clock.now)


async def test_cancelled_observer_keeps_actual_preparation_until_native_exit(evaluation_graph,monkeypatch):
    import src.users.voice_evaluation as module
    g = evaluation_graph
    entered,release,denied = threading.Event(),threading.Event(),threading.Event()
    original = module._source_proof
    def held(encoded):
        value = original(encoded)
        entered.set()
        assert release.wait(5)
        return value
    monkeypatch.setattr(module,'_source_proof',held)
    selection = EvaluationGrantSelection(g.identity,frozenset(EvaluationModality),g.clock.now+20)
    def native():
        try:
            g.voices.evaluation.prepare_grant(g.root,(selection,),g.clock.now+90)
        except LookupError:
            denied.set()
    pending = asyncio.create_task(g.voices.acall(native))
    try:
        assert await asyncio.to_thread(entered.wait,3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert g.voices._io_tasks and not denied.is_set()
        g.privacy.ledger._pending_conversations = frozenset({g.sources[0].conversation_id})
        release.set()
        await until(lambda:not g.voices._io_tasks)
        assert denied.is_set() and g.authority.available
    finally:
        release.set()
        await asyncio.gather(pending,return_exceptions=True)
        await until(lambda:not g.voices._io_tasks)
        g.privacy.ledger._pending_conversations = frozenset()


async def test_cancelled_issuer_observer_waits_for_retained_final_guard_rollback(evaluation_graph,monkeypatch):
    g = evaluation_graph
    scope = await prepare(g)
    entered,release = threading.Event(),threading.Event()
    original = g.authority._read
    def held(db):
        result = original(db)
        entered.set()
        assert release.wait(5)
        return result
    monkeypatch.setattr(g.authority,'_read',held)
    pending = asyncio.create_task(issue(g,scope))
    try:
        assert await asyncio.to_thread(entered.wait,3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert g.authority._tasks
        g.privacy.ledger._pending_conversations = frozenset({g.sources[0].conversation_id})
        release.set()
        await until(lambda:not g.authority._tasks)
        assert g.authority.available and len(g.authority._credentials)==0
    finally:
        release.set()
        await asyncio.gather(pending,return_exceptions=True)
        await until(lambda:not g.authority._tasks)
        g.privacy.ledger._pending_conversations = frozenset()


async def test_malformed_native_binding_is_rejected_before_revision_seal(evaluation_graph):
    g = evaluation_graph
    credential,session = await operator_session(g)
    malformed = replace(session.lease,binding='invalid')
    assert malformed.current(frozenset({'auth:manage'}))
    scope = await prepare(g,issuer=malformed)
    with pytest.raises(PermissionError,match='invalid_dashboard_binding'):
        await g.authority.revise_credential(malformed,credential.identity,role=DashboardRole.EVALUATOR,
            expires_at=scope.credential_expires_at,samples=scope.samples,prepared_scope=scope)
    assert session.lease.current() and g.authority.resolve(credential.token) is not None
    assert credential.identity not in g.authority._revision_admission and g.authority.available


async def test_missing_review_row_cannot_be_granted_from_fingerprint_metadata_alone(evaluation_graph):
    g = evaluation_graph
    await sql(g,'DELETE FROM voice_capture_reviews WHERE fingerprint_id=?',(g.identity,))
    with pytest.raises(LookupError,match='evaluation_grant_unavailable'):
        await prepare(g)
