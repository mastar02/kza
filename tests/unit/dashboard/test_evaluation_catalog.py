"""The evaluator's selector is an exact, expiring authorization projection."""

from dataclasses import FrozenInstanceError, replace
import sqlite3

import pytest

from src.dashboard.authority import DashboardAuthority, DashboardRole, EvaluationModality, EvaluationSample
from tests.unit.dashboard import test_authority as fixtures

pytestmark = pytest.mark.asyncio
authority = fixtures.authority


async def evaluator(service,now,root,deadlines=(3.,7.)):
    samples = tuple(EvaluationSample('vf_'+format(index,'064x'),frozenset({'a'*64}),
        frozenset(EvaluationModality),now+offset) for index,offset in enumerate(deadlines))
    credential = await service.issue_credential(service.resolve(root),subject='synthetic-catalog',
        role=DashboardRole.EVALUATOR,expires_at=now+90,samples=samples)
    session = await service.login(credential.token)
    return session.lease


async def test_catalog_transitions_to_empty_without_reopening_expired_choices(authority):
    service,now,root = authority
    start = now[0]
    actor = await evaluator(service,start,root[0])
    first = service.evaluation_grants(actor)
    assert first.expires_at==start+3 and len(first.grants)==2
    now[0]=start+3
    assert not first.current()
    second = service.evaluation_grants(actor)
    assert len(second.grants)==1 and second.expires_at==start+7
    now[0]=start+7
    empty = service.evaluation_grants(actor)
    assert empty.public()=={'observed_at':now[0],'expires_at':actor.principal.expires_at,'grants':[]}
    now[0]=start+1
    assert empty.current() and service.evaluation_grants(actor).grants==()
    assert not first.current() and not second.current()
    now[0]=actor.principal.expires_at
    assert not empty.current()
    with pytest.raises(PermissionError):
        service.evaluation_grants(actor)


async def test_native_catalog_uses_no_sqlite_or_owned_io(authority,monkeypatch):
    service,now,root = authority
    actor = await evaluator(service,now[0],root[0])
    def forbidden(*args,**kwargs):
        pytest.fail('pure authorization projection performed I/O')
    with monkeypatch.context() as patch:
        patch.setattr(sqlite3,'connect',forbidden)
        patch.setattr(service._work,'run',forbidden)
        catalog = service.evaluation_grants(actor)
        assert catalog.current() and len(catalog.public()['grants'])==2


@pytest.mark.parametrize('role',[DashboardRole.HOME,DashboardRole.OPERATOR])
async def test_non_evaluator_has_no_catalog_even_with_operator_read_permission(authority,role):
    service,now,root = authority
    issued = await service.issue_credential(service.resolve(root[0]),subject='synthetic-other-role',
        role=role,expires_at=now[0]+90)
    actor = service.resolve(issued.token)
    if role is DashboardRole.OPERATOR:
        assert actor.current(frozenset({'evaluation:read'}))
    with pytest.raises(PermissionError):
        service.evaluation_grants(actor)


async def test_foreign_authority_cannot_project_original_lease(authority,tmp_path):
    service,now,root = authority
    actor = await evaluator(service,now[0],root[0])
    other = DashboardAuthority(tmp_path/'other'/'auth.db',clock=lambda:now[0])
    try:
        with pytest.raises(PermissionError):
            other.evaluation_grants(actor)
    finally:
        await other.aclose()


async def test_session_deadline_caps_every_grant_and_catalog(authority):
    service,now,root = authority
    service.session_ttl_s=2.5
    actor = await evaluator(service,now[0],root[0],(20.,30.))
    catalog = service.evaluation_grants(actor)
    assert catalog.expires_at==now[0]+2.5
    assert {grant.expires_at for grant in catalog.grants}=={catalog.expires_at}
    now[0]+=2.5
    assert not catalog.current()


async def test_maximum_catalog_is_exact_frozen_and_contains_no_proofs(authority):
    service,now,root = authority
    actor = await evaluator(service,now[0],root[0],(10.,)*64)
    catalog = service.evaluation_grants(actor)
    public = catalog.public()
    assert len(public['grants'])==64
    assert all(set(row)=={'fingerprint_id','modalities','expires_at'} for row in public['grants'])
    assert set(public)=={'observed_at','expires_at','grants'}
    assert 'a'*64 not in str(public) and 'vf_' not in repr(catalog)
    public['grants'].clear()
    assert len(catalog.grants)==64
    with pytest.raises(FrozenInstanceError):
        catalog.grants=()


@pytest.mark.parametrize('change',['omit','deadline','mode','sample'])
async def test_forged_catalog_cannot_rebind_original_grants(authority,change):
    service,now,root = authority
    actor = await evaluator(service,now[0],root[0])
    catalog = service.evaluation_grants(actor)
    with pytest.raises(ValueError):
        if change=='omit':
            replace(catalog,grants=(),samples=())
        elif change=='deadline':
            replace(catalog,grants=(replace(catalog.grants[0],expires_at=now[0]+100),catalog.grants[1]))
        elif change=='mode':
            replace(catalog,grants=(replace(catalog.grants[0],modalities=frozenset({EvaluationModality.TRANSCRIPT_FULL})),catalog.grants[1]))
        else:
            replace(catalog,samples=(replace(catalog.samples[0]),catalog.samples[1]))


@pytest.mark.parametrize('empty',[False,True])
async def test_original_session_logout_retires_catalog_including_empty(authority,empty):
    service,now,root = authority
    actor = await evaluator(service,now[0],root[0])
    if empty:
        now[0]+=7
    catalog = service.evaluation_grants(actor)
    await service.logout(actor)
    assert not catalog.current()
    with pytest.raises(PermissionError):
        service.evaluation_grants(actor)
