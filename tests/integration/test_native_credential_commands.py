"""Credential intentions use actual metadata proofs and retained native work."""

import asyncio
import threading
from uuid import uuid4

import pytest

from src.dashboard.authority import DashboardRole, EvaluationModality
from src.dashboard.credential_journal import CredentialAction as Action, CredentialCommand as Command, CredentialGrantSpec as Grant
from src.dashboard.evaluation_actor import EvaluationReadActor
from src.dashboard.evaluation_grants import EvaluationGrantSelection
from tests.integration import test_voice_evaluation_reader as fixtures

pytestmark = pytest.mark.asyncio
evaluation_graph = fixtures.evaluation_graph


def command(g,modality,*,target=None):
    grant = Grant(DashboardRole.EVALUATOR,g.clock.now+90,selections=(
        EvaluationGrantSelection(g.identity,frozenset({modality}),g.clock.now+20),),
        subject='synthetic-native-evaluator' if target is None else None)
    return Command(str(uuid4()),Action.ISSUE if target is None else Action.REVISE,
        target,1 if target is not None else None,grant)


def provider(g):
    async def prepare(actor,grant):
        return await g.voices.acall(g.voices.evaluation.prepare_grant,actor,grant.selections,grant.expires_at)
    return prepare


@pytest.mark.parametrize('modality',list(EvaluationModality))
async def test_issued_exact_modality_and_replay_after_source_removal(evaluation_graph,modality):
    g = evaluation_graph
    intent = command(g,modality)
    result = await g.authority.commands.execute(g.root,intent,prepare_scope=provider(g))
    session = await g.authority.login(result.token)
    projection = await g.voices.acall(g.voices.evaluation.read,EvaluationReadActor(session.lease,g.identity,modality))
    assert projection.rows[0]['modality']==modality.value
    g.privacy.ledger._pending_conversations = frozenset({g.sources[0].conversation_id})
    await g.authority.revoke_credential(g.root,result.receipt.credential_id)
    async def forbidden(actor,grant):
        pytest.fail('replay attempted a new metadata read')
    replay = await g.authority.commands.execute(g.root,intent,prepare_scope=forbidden)
    assert replay.receipt==result.receipt and replay.token is None and not replay.applied
    assert g.authority.commands.catalog(g.root)['rows']==[]


async def test_known_preparation_failure_is_durable_without_retrying_sources(evaluation_graph):
    g = evaluation_graph
    intent = command(g,EvaluationModality.TRANSCRIPT_FULL)
    g.privacy.ledger._pending_conversations = frozenset({g.sources[0].conversation_id})
    first = await g.authority.commands.execute(g.root,intent,prepare_scope=provider(g))
    assert first.receipt.reason=='scope_unavailable' and first.token is None
    g.privacy.ledger._pending_conversations = frozenset()
    async def forbidden(actor,grant):
        pytest.fail('terminal no-effect replay attempted preparation')
    replay = await g.authority.commands.execute(g.root,intent,prepare_scope=forbidden)
    assert replay.receipt==first.receipt and not replay.applied
    assert g.authority.commands.catalog(g.root)['rows']==[] and g.authority.available


@pytest.mark.parametrize('operation',['issue','revise'])
async def test_source_retirement_during_last_receipt_read_rolls_back_effect(evaluation_graph,monkeypatch,operation):
    from src.dashboard import credential_journal as journal
    g = evaluation_graph
    target = None
    if operation=='revise':
        target = await g.authority.issue_credential(g.root,subject='synthetic-prior-home',
            role=DashboardRole.HOME,expires_at=g.clock.now+90)
    intent = command(g,EvaluationModality.TRANSCRIPT_FULL,target=None if target is None else target.identity)
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
    pending = asyncio.create_task(g.authority.commands.execute(g.root,intent,prepare_scope=provider(g)))
    try:
        assert await asyncio.to_thread(entered.wait,3)
        g.privacy.ledger._pending_conversations = frozenset({g.sources[0].conversation_id})
        release.set()
        result = await pending
        assert not result.applied and result.token is None and result.receipt.outcome.value=='no_effect'
        rows = g.authority.commands.catalog(g.root)['rows']
        assert len(rows)==(0 if target is None else 1)
        if target is not None:
            assert rows[0]['role']=='home' and rows[0]['revision']==1
            assert g.authority.resolve(target.token).current()
        assert not g.authority._credential_reservations and g.authority.available
    finally:
        release.set()
        await asyncio.gather(pending,return_exceptions=True)


async def test_self_revision_keeps_original_issuer_and_separate_evaluator_permissions(evaluation_graph):
    g = evaluation_graph
    created = await g.authority.issue_credential(g.root,subject='synthetic-self-operator',
        role=DashboardRole.OPERATOR,expires_at=g.clock.now+90)
    original = g.authority.resolve(created.token)
    intent = command(g,EvaluationModality.TRANSCRIPT_FULL,target=created.identity)
    result = await g.authority.commands.execute(original,intent,prepare_scope=provider(g))
    assert result.applied and result.receipt.result_revision==2 and not original.current()
    evaluator = g.authority.resolve(created.token)
    assert evaluator.current(frozenset({'evaluation:read'}))
    assert not evaluator.current(frozenset({'auth:manage'}))
    with pytest.raises(PermissionError):
        g.authority.commands.catalog(evaluator)
    projection = await g.voices.acall(g.voices.evaluation.read,
        EvaluationReadActor(evaluator,g.identity,EvaluationModality.TRANSCRIPT_FULL))
    assert projection.rows[0]['modality']=='transcript_full'
