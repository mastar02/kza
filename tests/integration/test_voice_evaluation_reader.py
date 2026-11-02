"""Real SQLite modality isolation and original evaluation-grant lifetimes."""

import asyncio
import base64
from dataclasses import replace
import gc
import json
import sqlite3
import threading
from types import SimpleNamespace
from uuid import uuid4
from weakref import ref

import numpy as np
import pytest
import pytest_asyncio

from src.dashboard.authority import (
    DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample, source_digest,
)
from src.dashboard.evaluation_actor import EvaluationReadActor
from src.dashboard.live_retention_types import EvaluationScopeFence, SnapshotMetadata
from src.memory.fact_ledger import MemorySource
from src.users.voice_capture import build_review_capture
from src.users.voice_registry import VoiceRegistry
from tests.mocks.memory_sources import memory_source
from tests.mocks.voice_registry import AUDIO, VECTOR, graph


@pytest_asyncio.fixture
async def evaluation_graph(tmp_path,monkeypatch):
    monkeypatch.setattr(VoiceRegistry,'start',lambda self:None)
    async with graph(tmp_path) as (privacy,owner,voices,clock):
        authority=DashboardAuthority(tmp_path/'evaluation-auth.db',clock=lambda:clock.now,
            expected_token=lambda:'synthetic-evaluation-root',session_ttl_s=120)
        await authority.initialize()
        try:
            root=authority.resolve('synthetic-evaluation-root')
            sources=tuple(memory_source(now=clock.now,ttl=60) for _ in range(2))
            review=build_review_capture(np.tile(AUDIO,3),'Synthetic selected evaluation text','Synthetic private room')
            identity=await voices.acall(voices.ledger.put,'synthetic-model',VECTOR.copy(),3.,sources,review)
            other_source=memory_source(now=clock.now,ttl=60)
            other_review=build_review_capture(np.tile(AUDIO,3),'Unapproved other sample','Other private room')
            other=await voices.acall(voices.ledger.put,'synthetic-model',VECTOR.copy(),3.,(other_source,),other_review)
            digests=frozenset(source_digest(source.to_dict()) for source in sources)
            async def issue(modality,*,modes=None,approved=None,sample_ttl=20.,credential_ttl=90.,session_ttl=None):
                sample=EvaluationSample(identity,digests if approved is None else approved,
                    frozenset({modality}) if modes is None else modes,clock.now+sample_ttl)
                grant=await authority.issue_credential(root,subject='synthetic-evaluator-'+str(uuid4()),
                    role=DashboardRole.EVALUATOR,expires_at=clock.now+credential_ttl,samples=(sample,))
                previous=authority.session_ttl_s
                if session_ttl is not None:
                    authority.session_ttl_s=session_ttl
                try:
                    session=await authority.login(grant.token)
                finally:
                    authority.session_ttl_s=previous
                return SimpleNamespace(actor=EvaluationReadActor(session.lease,identity,modality),
                    sample=sample,grant=grant,session=session)
            yield SimpleNamespace(privacy=privacy,owner=owner,voices=voices,clock=clock,authority=authority,
                root=root,sources=sources,identity=identity,other=other,review=review,digests=digests,issue=issue)
        finally:
            await authority.aclose()


async def sql(g,statement,parameters=()):
    def execute():
        with g.privacy.ledger._transaction() as db:
            return db.execute(statement,parameters).fetchall()
    return await g.voices.acall(execute)


async def authorized_read(g,actor,reads):
    allowed={'facts':{'id','revision','expires_at','state'},
        'conversation_tombstones':{'conversation_id','deleted_at'},
        'voice_fingerprints':{'id','sources_json','expires_at'},
        'voice_capture_reviews':{'fingerprint_id'}|({'audio_wav','clip_duration_s'}
            if actor.modality is EvaluationModality.AUDIO_PREVIEW else {'transcript'})}
    def authorize(operation,table,column,*_):
        if operation==sqlite3.SQLITE_READ:
            reads.append((table,column))
            if table not in allowed or column not in allowed[table]:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    def read():
        g.privacy.ledger._db.set_authorizer(authorize)
        try:
            return g.voices.evaluation.read(actor)
        finally:
            g.privacy.ledger._db.set_authorizer(None)
    return await g.voices.acall(read)


@pytest.mark.parametrize('modality',list(EvaluationModality))
async def test_exact_modality_never_reads_other_payload_or_operator_metadata(evaluation_graph,modality):
    g=evaluation_graph
    issued=await g.issue(modality,modes=frozenset(EvaluationModality))
    if modality is EvaluationModality.AUDIO_PREVIEW:
        await sql(g,'UPDATE voice_capture_reviews SET transcript=? WHERE fingerprint_id=?',('x'*9000,g.identity))
    else:
        await sql(g,'UPDATE voice_capture_reviews SET audio_wav=? WHERE fingerprint_id=?',(b'invalid unapproved audio',g.identity))
    reads=[]
    projection=await authorized_read(g,issued.actor,reads)
    row=projection.rows[0]
    assert row['id']==g.identity and row['modality']==modality.value
    if modality is EvaluationModality.AUDIO_PREVIEW:
        assert set(row)=={'id','modality','audio_mime','audio_base64','clip_duration_s'}
        assert base64.b64decode(row['audio_base64'])==g.review.audio_wav
        assert ('voice_capture_reviews','transcript') not in reads
    else:
        assert set(row)=={'id','modality','transcript_scope','transcript'}
        assert row['transcript']==g.review.transcript
        assert ('voice_capture_reviews','audio_wav') not in reads
    assert frozenset(projection.sources)==frozenset(g.sources) and projection.source_digests==g.digests
    assert projection.revision.current()


@pytest.mark.parametrize('change',['extra_field','missing_field','additional_source','missing_source','changed_time'])
async def test_complete_original_source_declarations_are_required_before_payload(evaluation_graph,change):
    g=evaluation_graph
    declarations=[source.to_dict() for source in g.sources]
    approved=g.digests
    if change=='missing_field':
        # Parsing either declaration produces the same MemorySource; the grant
        # must still bind the complete original declaration before parsing it.
        declarations[0]['fact_dependencies']=[]
        approved=frozenset(source_digest(source) for source in declarations)
        declarations[0].pop('fact_dependencies')
    elif change=='extra_field':
        declarations[0]['unexpected_provenance']='must not normalize away'
    elif change=='additional_source':
        declarations.append(memory_source(now=g.clock.now,ttl=60).to_dict())
    elif change=='missing_source':
        declarations.pop()
    else:
        declarations[0]['expires_at']-=1
    issued=await g.issue(EvaluationModality.AUDIO_PREVIEW,approved=approved)
    await sql(g,'UPDATE voice_fingerprints SET sources_json=? WHERE id=?',(json.dumps(declarations),g.identity))
    reads=[]
    with pytest.raises((PermissionError,LookupError)):
        await authorized_read(g,issued.actor,reads)
    assert not any(table=='voice_capture_reviews' for table,_ in reads)


async def test_unapproved_fingerprint_wrong_mode_and_operator_have_no_reader(evaluation_graph):
    g=evaluation_graph
    issued=await g.issue(EvaluationModality.AUDIO_PREVIEW)
    with pytest.raises(PermissionError):
        EvaluationReadActor(issued.session.lease,g.other,EvaluationModality.AUDIO_PREVIEW)
    with pytest.raises(PermissionError):
        EvaluationReadActor(issued.session.lease,g.identity,EvaluationModality.TRANSCRIPT_FULL)
    assert 'evaluation:read' in g.root.principal.permissions
    with pytest.raises(PermissionError):
        EvaluationReadActor(g.root,g.identity,EvaluationModality.AUDIO_PREVIEW)


@pytest.mark.parametrize('bound',['sample','credential','session','fingerprint','source'])
async def test_deadline_uses_exact_original_minimum_without_rounding(evaluation_graph,bound):
    g=evaluation_graph
    arguments={'sample_ttl':90.,'credential_ttl':120.}
    expected=g.clock.now+60.
    if bound=='sample':
        arguments['sample_ttl']=3.25;expected=g.clock.now+3.25
    elif bound=='credential':
        arguments['credential_ttl']=4.5;expected=g.clock.now+4.5
    elif bound=='session':
        arguments['session_ttl']=5.75;expected=g.clock.now+5.75
    elif bound=='fingerprint':
        expected=g.clock.now+6.125
        await sql(g,'UPDATE voice_fingerprints SET expires_at=? WHERE id=?',(expected,g.identity))
    else:
        await sql(g,'UPDATE voice_fingerprints SET expires_at=? WHERE id=?',(g.clock.now+300,g.identity))
    issued=await g.issue(EvaluationModality.TRANSCRIPT_FULL,**arguments)
    projection=await g.voices.acall(g.voices.evaluation.read,issued.actor)
    assert projection.expires_at==expected
    with pytest.raises(ValueError):
        replace(projection,expires_at=expected+1000)


@pytest.mark.parametrize('change',['sample','pending_source','logout','revise','replacement'])
async def test_actual_encoder_completion_cannot_restore_expired_or_retired_authority(evaluation_graph,monkeypatch,change):
    import src.users.voice_evaluation as module
    g=evaluation_graph
    issued=await g.issue(EvaluationModality.AUDIO_PREVIEW)
    entered,release=threading.Event(),threading.Event()
    original=module.b64encode
    def held(value):
        entered.set()
        assert release.wait(5)
        return original(value)
    monkeypatch.setattr(module,'b64encode',held)
    pending=asyncio.create_task(g.voices.acall(g.voices.evaluation.read,issued.actor))
    try:
        async with asyncio.timeout(3):
            while not entered.is_set():
                await asyncio.sleep(.001)
        if change=='sample':
            g.clock.now=issued.sample.expires_at
            assert issued.session.lease.current()
        elif change=='pending_source':
            g.privacy.ledger._pending_conversations=frozenset({g.sources[0].conversation_id})
        elif change=='logout':
            await g.authority.logout(issued.session.lease)
        elif change=='revise':
            assert await g.authority.revise_credential(g.root,issued.grant.identity,role=DashboardRole.EVALUATOR,
                expires_at=g.clock.now+90,samples=(issued.sample,))
        else:
            replacement=await g.authority.login(issued.grant.token,replacing=issued.session.lease)
            assert replacement.lease.current()
        release.set()
        with pytest.raises((PermissionError,LookupError)):
            await pending
    finally:
        release.set()
        await asyncio.gather(pending,return_exceptions=True)
        g.privacy.ledger._pending_conversations=frozenset()


async def test_delivered_fence_keeps_only_scope_metadata_and_expires_before_session(evaluation_graph):
    g=evaluation_graph
    issued=await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    projection=await g.voices.acall(g.voices.evaluation.read,issued.actor)
    actor_ref=ref(issued.actor)
    metadata=SnapshotMetadata('synthetic-evaluation-event',projection.sources,projection.expires_at,
        'evaluation:sample',revision=projection.revision)
    projection.rows.clear();issued.actor=None;del projection;gc.collect()
    assert actor_ref() is not None and metadata.available(g.privacy.ledger,g.clock.now)
    g.clock.now=issued.sample.expires_at
    assert issued.session.lease.current()
    assert not metadata.revision.current() and not metadata.available(g.privacy.ledger,g.clock.now)
    g.clock.now-=10
    assert not metadata.revision.current()


async def test_invalid_wav_error_does_not_retain_validator_audio_context(evaluation_graph):
    g=evaluation_graph
    issued=await g.issue(EvaluationModality.AUDIO_PREVIEW)
    canary=b'SYNTHETIC_PRIVATE_WAV_CANARY'+b'x'*64
    await sql(g,'UPDATE voice_capture_reviews SET audio_wav=? WHERE fingerprint_id=?',(canary,g.identity))
    def failure():
        try:
            g.voices.evaluation.read(issued.actor)
        except LookupError as error:
            return error
        raise AssertionError('invalid audio was accepted')
    error=await g.voices.acall(failure)
    assert error.__context__ is None and error.__cause__ is None and 'CANARY' not in str(error)
    trace=error.__traceback__
    while trace is not None:
        assert trace.tb_frame.f_code.co_name!='validate_voice_preview'
        if trace.tb_frame.f_code.co_name=='read':
            assert all(trace.tb_frame.f_locals[name] is None for name in ['metadata','payload','data','proof','transcript'])
        trace=trace.tb_next


@pytest.mark.parametrize('modality',list(EvaluationModality))
async def test_actual_projection_constructor_retirement_drops_private_traceback(evaluation_graph,monkeypatch,modality):
    import src.users.voice_evaluation as module
    g=evaluation_graph
    issued=await g.issue(modality,sample_ttl=2.25)
    original=module.EvaluationProjection
    entered=False
    def retire_then_construct(*args,**kwargs):
        nonlocal entered
        entered=True
        g.clock.now=issued.sample.expires_at
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'EvaluationProjection',retire_then_construct)
    def failure():
        try:
            g.voices.evaluation.read(issued.actor)
        except PermissionError as exc:
            return exc
        raise AssertionError('expired constructor returned a sample')
    error=await g.voices.acall(failure)
    assert entered and error.__context__ is None and error.__cause__ is None
    assert str(error)=='evaluation_scope_unavailable'
    frames=[]
    current=error.__traceback__
    while current is not None:
        frames.append(current.tb_frame)
        current=current.tb_next
    assert not any(frame.f_code.co_name in {'retire_then_construct','__post_init__'} for frame in frames)
    reader=next(frame for frame in frames if frame.f_code.co_name=='read')
    for name in ('proof','metadata','payload','data','sources','digests','result','transcript'):
        assert reader.f_locals[name] is None


@pytest.mark.parametrize('transcript',['x'*8193,'\x00'+'x'*40000,'é'*8193])
async def test_selected_transcript_limits_include_bytes_after_nul(evaluation_graph,transcript):
    g=evaluation_graph
    issued=await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    await sql(g,'UPDATE voice_capture_reviews SET transcript=? WHERE fingerprint_id=?',(transcript,g.identity))
    with pytest.raises(LookupError,match='evaluation_sample_unavailable'):
        await authorized_read(g,issued.actor,[])


async def test_nested_source_json_drops_parser_context_before_payload(evaluation_graph):
    g=evaluation_graph
    issued=await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    encoded='['*1500+'"private-parser-canary"'+']'*1500
    await sql(g,'UPDATE voice_fingerprints SET sources_json=? WHERE id=?',(encoded,g.identity))
    reads=[]
    def authorize(operation,table,column,*_):
        if operation==sqlite3.SQLITE_READ:
            reads.append((table,column))
        return sqlite3.SQLITE_OK
    def failure():
        g.privacy.ledger._db.set_authorizer(authorize)
        try:
            g.voices.evaluation.read(issued.actor)
        except LookupError as exc:
            return exc
        finally:
            g.privacy.ledger._db.set_authorizer(None)
        raise AssertionError('deeply nested source proof accepted')
    error=await g.voices.acall(failure)
    assert error.__context__ is None and error.__cause__ is None
    assert str(error)=='evaluation_sample_unavailable'
    assert not any(table=='voice_capture_reviews' for table,_ in reads)
    current=error.__traceback__
    while current is not None:
        assert current.tb_frame.f_code.co_name not in {'_source_proof','loads','raw_decode','decode'}
        if current.tb_frame.f_code.co_name=='read':
            assert current.tb_frame.f_locals['metadata'] is None
        current=current.tb_next


@pytest.mark.parametrize('column',['sources_json','transcript'])
async def test_sqlite_invalid_text_drops_private_database_error(evaluation_graph,column):
    g=evaluation_graph
    issued=await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    canary=b'sqlite-private-canary'+bytes([255])
    if column=='sources_json':
        await sql(g,'UPDATE voice_fingerprints SET sources_json=CAST(? AS TEXT) WHERE id=?',(canary,g.identity))
    else:
        await sql(g,'UPDATE voice_capture_reviews SET transcript=CAST(? AS TEXT) WHERE fingerprint_id=?',(canary,g.identity))
    def failure():
        try:
            g.voices.evaluation.read(issued.actor)
        except LookupError as exc:
            return exc
        raise AssertionError('invalid SQLite text accepted')
    error=await g.voices.acall(failure)
    assert str(error)=='evaluation_sample_unavailable'
    assert error.__context__ is None and error.__cause__ is None
    current=error.__traceback__
    while current is not None:
        frame=current.tb_frame
        assert frame.f_code.co_name not in {'_source_proof','__post_init__'}
        if frame.f_code.co_name=='read':
            for name in ('proof','metadata','payload','data','sources','digests','result','transcript'):
                assert frame.f_locals[name] is None
        current=current.tb_next


async def test_duplicate_source_keys_cannot_normalize_to_approved_proof(evaluation_graph):
    g=evaluation_graph
    issued=await g.issue(EvaluationModality.AUDIO_PREVIEW)
    declarations=[source.to_dict() for source in g.sources]
    encoded=json.dumps(declarations)
    approved='"conversation_id": '+json.dumps(declarations[0]['conversation_id'])
    duplicate='"conversation_id": '+json.dumps(str(uuid4()))+', '+approved
    assert approved in encoded
    encoded=encoded.replace(approved,duplicate,1)
    # The standard parser silently discards the unapproved first declaration.
    assert json.loads(encoded)==declarations
    await sql(g,'UPDATE voice_fingerprints SET sources_json=? WHERE id=?',(encoded,g.identity))
    reads=[]
    with pytest.raises(LookupError,match='evaluation_sample_unavailable'):
        await authorized_read(g,issued.actor,reads)
    assert not any(table=='voice_capture_reviews' for table,_ in reads)


@pytest.mark.parametrize('modality',list(EvaluationModality))
async def test_native_proof_cannot_rebind_mutable_projection_to_other_stored_sources(evaluation_graph,modality):
    from src.users.voice_evaluation import EvaluationProjection
    g=evaluation_graph
    issued=await g.issue(modality)
    projection=await g.voices.acall(g.voices.evaluation.read,issued.actor)
    proof=projection.revision.stamps[0]
    assert proof.sources is projection.sources
    rows=await sql(g,'SELECT sources_json FROM voice_fingerprints WHERE id=?',(g.other,))
    other_sources=tuple(MemorySource.from_dict(value) for value in json.loads(rows[0]['sources_json']))
    projection.sources=other_sources
    with pytest.raises(ValueError,match='invalid_evaluation_fence'):
        EvaluationProjection.__post_init__(projection)
    assert projection.revision.stamps[0] is proof
    assert proof.sources!=other_sources and frozenset(proof.sources)==frozenset(g.sources)


async def test_evaluation_fence_rejects_arbitrary_callbacks_and_extended_grants(evaluation_graph):
    g=evaluation_graph
    issued=await g.issue(EvaluationModality.AUDIO_PREVIEW)
    with pytest.raises(ValueError):
        EvaluationScopeFence(SimpleNamespace(current=lambda *_:True),g.digests,issued.sample.expires_at,g.sources)
    with pytest.raises(ValueError):
        EvaluationScopeFence(issued.actor,g.digests,issued.actor.grant_deadline()+1,g.sources)
