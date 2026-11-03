"""Deliver only the original evaluator's authorized choices, without payload I/O."""

import asyncio

import httpx
import pytest
from starlette.responses import StreamingResponse

from src.dashboard.authority import DashboardRole, EvaluationModality
from tests.integration import test_voice_evaluation_delivery as fixtures
from tests.integration.test_capture_producer_privacy import until

pytestmark = pytest.mark.asyncio
evaluation_graph = fixtures.evaluation_graph
evaluation_http = fixtures.evaluation_http
PATH = '/api/evaluation/grants'


@pytest.mark.parametrize('modality',list(EvaluationModality))
async def test_catalog_has_exact_safe_fields_without_voice_io(evaluation_http,monkeypatch,modality):
    g = evaluation_http
    issued = await g.issue(modality)
    def forbidden(*args,**kwargs):
        pytest.fail('grant metadata consulted voice data or payload')
    with monkeypatch.context() as patch:
        patch.setattr(g.voices,'acall',forbidden)
        patch.setattr(g.voices.evaluation,'read',forbidden)
        patch.setattr(g.voices.evaluation,'prepare_grant',forbidden)
        async with fixtures.client_for(g,issued) as client:
            response = await client.get(PATH)
    assert response.status_code==200 and response.headers['cache-control']=='no-store'
    assert response.json()=={'observed_at':g.clock.now,'expires_at':issued.sample.expires_at,
        'grants':[{'fingerprint_id':g.identity,'modalities':[modality.value],
            'expires_at':issued.sample.expires_at}]}
    assert all(digest not in response.text for digest in g.digests)
    assert g.review.transcript not in response.text
    assert not g.bus._retention._states


@pytest.mark.parametrize('role',[DashboardRole.OPERATOR,DashboardRole.HOME])
async def test_other_roles_cannot_list_evaluator_grants(evaluation_http,role):
    g = evaluation_http
    credential = await g.authority.issue_credential(g.root,subject='synthetic-catalog-other-role',
        role=role,expires_at=g.clock.now+60)
    session = await g.authority.login(credential.token)
    if role is DashboardRole.OPERATOR:
        assert session.lease.current(frozenset({'evaluation:read'}))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=g.api.app),base_url='http://testserver',
            cookies={'kza_dashboard_token':session.token}) as client:
        response = await client.get(PATH)
    assert response.status_code==403 and response.headers['cache-control']=='no-store'
    assert 'grants' not in response.json()


@pytest.mark.parametrize('query',['?modality=transcript_full','?limit=1','?fingerprint_id=bad'])
async def test_catalog_rejects_query_expansion(evaluation_http,query):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    async with fixtures.client_for(g,issued) as client:
        response = await client.get(PATH+query)
    assert response.status_code==422 and response.headers['cache-control']=='no-store'


async def test_http_expiry_observation_does_not_reopen_with_clock_rollback(evaluation_http):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL,sample_ttl=3.25)
    async with fixtures.client_for(g,issued) as client:
        assert len((await client.get(PATH)).json()['grants'])==1
        original = g.clock.now
        g.clock.now = issued.sample.expires_at
        empty = (await client.get(PATH)).json()
        assert empty['grants']==[] and empty['expires_at']==issued.session.lease.principal.expires_at
        g.clock.now = original
        assert (await client.get(PATH)).json()['grants']==[]


@pytest.mark.parametrize('stage',['headers','buffer'])
@pytest.mark.parametrize('empty,change',[
    (False,'sample'),(False,'logout'),(True,'logout'),(False,'revision'),
    (True,'revision'),(False,'owner'),(True,'owner'),(True,'session'),
])
async def test_retired_catalog_never_becomes_an_empty_success(evaluation_http,stage,empty,change):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL,sample_ttl=3.25)
    if empty:
        g.clock.now = issued.sample.expires_at
    entered,release = asyncio.Event(),asyncio.Event()
    messages = []
    if stage=='buffer':
        @g.api.app.middleware('http')
        async def buffered(request,call_next):
            response = await call_next(request)
            body = b''.join([part async for part in response.body_iterator])
            entered.set()
            await release.wait()
            async def chunks():
                yield body
            return StreamingResponse(chunks(),status_code=response.status_code,headers=dict(response.headers))
    async def application(scope,receive,send):
        async def actual(message):
            if stage=='headers' and message['type']=='http.response.start':
                entered.set()
                await release.wait()
            messages.append(message.copy())
            await send(message)
        await g.api.app(scope,receive,actual)
    closing = None
    async with fixtures.client_for(g,issued,app=application) as client:
        pending = asyncio.create_task(client.get(PATH))
        try:
            await until(entered.is_set)
            if change=='sample':
                g.clock.now = issued.sample.expires_at
            elif change=='logout':
                await g.authority.logout(issued.session.lease)
            elif change=='revision':
                await g.authority.revise_credential(g.root,issued.grant.identity,
                    role=DashboardRole.HOME,expires_at=g.clock.now+90)
            elif change=='session':
                g.clock.now = issued.session.lease.principal.expires_at
            else:
                closing = g.bus._retention.begin_close()
            release.set()
            if stage=='headers':
                with pytest.raises(AssertionError):
                    await pending
                assert [message['type'] for message in messages]==['http.response.start']
            else:
                response = await pending
                assert response.status_code==403 and 'grants' not in response.json()
            assert g.identity.encode() not in b''.join(message.get('body',b'') for message in messages)
        finally:
            release.set()
            await asyncio.gather(pending,return_exceptions=True)
            if closing is not None:
                await closing


async def test_retired_source_still_lists_grant_but_cannot_be_read(evaluation_http):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    g.privacy.ledger._pending_conversations=frozenset({g.sources[0].conversation_id})
    async with fixtures.client_for(g,issued) as client:
        response = await client.get(PATH)
    assert response.status_code==200 and response.json()['grants'][0]['fingerprint_id']==g.identity
    with pytest.raises(LookupError,match='evaluation_sample_unavailable'):
        await g.voices.acall(g.voices.evaluation.read,issued.actor)


@pytest.mark.parametrize('owner',['missing','invalid'])
async def test_missing_owner_cannot_fall_back_to_unowned_catalog(evaluation_http,owner):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    g.api.app.state.rest_delivery_owner = None if owner=='missing' else object()
    try:
        async with fixtures.client_for(g,issued) as client:
            response = await client.get(PATH)
        assert response.status_code==503 and 'grants' not in response.json()
        assert response.headers['cache-control']=='no-store'
    finally:
        g.api.app.state.rest_delivery_owner = g.bus._retention.rest


async def test_native_catalog_of_another_login_cannot_be_substituted(evaluation_http,monkeypatch):
    g = evaluation_http
    issued = await g.issue(EvaluationModality.TRANSCRIPT_FULL)
    other = await g.issue(EvaluationModality.AUDIO_PREVIEW)
    foreign = g.authority.evaluation_grants(other.session.lease)
    monkeypatch.setattr(g.authority,'evaluation_grants',lambda _:foreign)
    async with fixtures.client_for(g,issued) as client:
        response = await client.get(PATH)
    assert response.status_code==503 and g.identity not in response.text
