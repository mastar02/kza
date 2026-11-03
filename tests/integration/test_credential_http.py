"""Real authority commands behind retained HTTP delivery, using synthetic data."""

import asyncio
import json
import threading
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from starlette.responses import StreamingResponse

from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import DashboardRole, EvaluationModality
from src.dashboard.evaluation_actor import EvaluationReadActor
from src.dashboard.rest_delivery import RestOperation
from tests.integration import test_voice_evaluation_delivery as fixtures
from tests.integration.test_capture_producer_privacy import until

pytestmark = pytest.mark.asyncio
evaluation_graph = fixtures.evaluation_graph
evaluation_http = fixtures.evaluation_http
BASE = '/api/admin/auth/credentials'
ATTEMPTS = '/api/admin/auth/credential-attempts/'


@pytest_asyncio.fixture
async def credential_http(evaluation_http):
    g = evaluation_http
    session = await g.authority.login('synthetic-evaluation-root')
    g.issuer,g.issuer_cookie = session.lease,session.token
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=g.api.app),
            base_url='http://testserver',cookies={COOKIE_NAME:session.token}) as client:
        yield g,client


def issue_body(g):
    return {'request_id':str(uuid4()),'subject':'synthetic-http-subject',
        'role':'home','expires_at':g.clock.now+90}


async def test_real_issue_catalog_cas_revoke_and_replay_without_secret(credential_http):
    g,client = credential_http
    intent = issue_body(g)
    first = await client.post(BASE,json=intent)
    assert first.status_code==200 and first.headers['cache-control']=='no-store'
    created = first.json()
    assert created['applied'] and created['token'].startswith('kza_')
    identity = created['receipt']['credential_id']
    catalog = (await client.get(BASE)).json()
    assert len(catalog['rows'])==1 and catalog['rows'][0]['id']==identity
    assert catalog['rows'][0]['revision']==1 and catalog['rows'][0]['active']
    assert 'digest' not in json.dumps(catalog) and created['token'] not in json.dumps(catalog)
    attempt = (await client.get(ATTEMPTS+intent['request_id'])).json()
    assert attempt=={'state':'terminal','receipt':created['receipt']}
    replay = (await client.post(BASE,json=intent)).json()
    assert replay=={'receipt':created['receipt'],'applied':False}
    edit = {'request_id':str(uuid4()),'expected_revision':1,'role':'operator','expires_at':g.clock.now+80}
    winner = (await client.put(BASE+'/'+identity,json=edit)).json()
    assert winner['applied'] and winner['receipt']['result_revision']==2 and 'token' not in winner
    edit['request_id'] = str(uuid4())
    stale = (await client.put(BASE+'/'+identity,json=edit)).json()
    assert not stale['applied'] and stale['receipt']['reason']=='revision_conflict'
    assert g.authority.resolve(created['token']).principal.role is DashboardRole.OPERATOR
    deletion = await client.request('DELETE',BASE+'/'+identity,
        json={'request_id':str(uuid4()),'expected_revision':2})
    assert deletion.status_code==200 and deletion.json()['applied']
    assert (await client.get(BASE)).json()['rows']==[]
    assert (await client.post(BASE,json=intent)).json()==replay


async def test_concurrent_duplicate_observers_receive_only_one_secret(credential_http):
    g,client = credential_http
    intent = issue_body(g)
    responses = await asyncio.gather(*(client.post(BASE,json=intent) for _ in range(3)))
    assert all(response.status_code==200 for response in responses)
    values = [response.json() for response in responses]
    assert sum('token' in value for value in values)==1
    assert sum(value['applied'] for value in values)==1
    assert len({json.dumps(value['receipt'],sort_keys=True) for value in values})==1


@pytest.mark.parametrize('role',['home','evaluator'])
@pytest.mark.parametrize('method,path',[
    ('GET',BASE),('GET',ATTEMPTS+'11111111-1111-4111-8111-111111111111'),
    ('POST',BASE),('PUT',BASE+'/synthetic-target'),('DELETE',BASE+'/synthetic-target'),
])
async def test_non_admin_roles_cannot_reach_credential_management(credential_http,role,method,path):
    g,_ = credential_http
    if role=='evaluator':
        session = (await g.issue(EvaluationModality.TRANSCRIPT_FULL)).session
    else:
        credential = await g.authority.issue_credential(g.root,subject='synthetic-home',
            role=DashboardRole.HOME,expires_at=g.clock.now+60)
        session = await g.authority.login(credential.token)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=g.api.app),
            base_url='http://testserver',cookies={COOKIE_NAME:session.token}) as client:
        response = await client.request(method,path,json={})
    assert response.status_code==403 and response.headers['cache-control']=='no-store'
    assert not g.authority._attempts


@pytest.mark.parametrize('change',['authority_absent','owner_absent','owner_invalid'])
async def test_required_native_owners_never_fall_back(credential_http,monkeypatch,change):
    g,client = credential_http
    if change=='authority_absent':
        monkeypatch.setenv('KZA_DASHBOARD_TOKEN','synthetic-unrelated-legacy-token')
        g.api.app.state.dashboard_authority = None
    else:
        g.api.app.state.rest_delivery_owner = None if change=='owner_absent' else object()
    try:
        response = await client.post(BASE,json=issue_body(g))
        assert response.status_code==(401 if change=='authority_absent' else 503)
        assert response.headers['cache-control']=='no-store'
        assert not g.authority._attempts
    finally:
        g.api.app.state.dashboard_authority = g.authority
        g.api.app.state.rest_delivery_owner = g.bus._retention.rest


@pytest.mark.parametrize('change',[
    'unknown','role','timestamp','boolean_revision','duplicate_selection',
    'duplicate_modes','source_hash','missing','oversize','duplicate_json','nested',
    'surrogate_subject','surrogate_user',
])
async def test_closed_bounded_input_errors_never_echo_private_input(credential_http,change):
    g,client = credential_http
    value = issue_body(g)
    marker = 'synthetic-private-canary'
    path,method = BASE,'POST'
    if change=='unknown':
        value['token']=marker
    elif change=='role':
        value['role']=marker
    elif change=='timestamp':
        value['expires_at']=marker
    elif change=='boolean_revision':
        value.pop('subject');value['expected_revision']=True;method='PUT';path+='/'+str(uuid4())
    elif change in {'duplicate_selection','duplicate_modes','source_hash'}:
        value['role']='evaluator'
        selected = {'fingerprint_id':g.identity,'modalities':['transcript_full'],'expires_at':g.clock.now+10}
        if change=='duplicate_modes':
            selected['modalities']*=2
        if change=='source_hash':
            selected['source_digests']=[marker]
        value['selections']=[selected,selected] if change=='duplicate_selection' else [selected]
    elif change=='missing':
        value.pop('subject')
    elif change=='surrogate_subject':
        value['subject']='\ud800'
    elif change=='surrogate_user':
        value['user_id']='\ud800'
    raw = json.dumps(value)
    if change=='oversize':
        raw=json.dumps({'payload':marker*6000})
    elif change=='duplicate_json':
        raw=raw[:-1]+',"role":"'+marker+'"}'
    elif change=='nested':
        raw='['*1100+'"'+marker+'"'+']'*1100
    response = await client.request(method,path,content=raw,headers={'Content-Type':'application/json'})
    assert response.status_code==422 and marker not in response.text
    assert response.json()=={'detail':'invalid_credential_request'}
    assert response.headers['cache-control']=='no-store' and not g.authority._attempts
    assert g.authority.available


@pytest.mark.parametrize('suffix',['?limit=0','?limit=1&limit=2','?unknown=canary','?after='])
async def test_catalog_query_is_closed(credential_http,suffix):
    _,client = credential_http
    response = await client.get(BASE+suffix)
    assert response.status_code==422 and response.json()=={'detail':'invalid_credential_request'}


@pytest.mark.parametrize('change',['issuer_logout','target_revoke','target_revision','target_expiry','owner_close'])
async def test_completed_response_keeps_guards_through_outer_buffer(credential_http,change):
    g,client = credential_http
    entered,release = asyncio.Event(),asyncio.Event()
    @g.api.app.middleware('http')
    async def buffered(request,call_next):
        response = await call_next(request)
        body = b''.join([part async for part in response.body_iterator])
        entered.set()
        await release.wait()
        async def chunks():
            yield body
        return StreamingResponse(chunks(),status_code=response.status_code,headers=dict(response.headers))
    intent = issue_body(g)
    pending = asyncio.create_task(client.post(BASE,json=intent))
    closing = None
    try:
        await until(entered.is_set)
        await until(lambda:g.api.admin_work.status()['operations']==0)
        target = intent['request_id']
        if change=='issuer_logout':
            await g.authority.logout(g.issuer)
        elif change=='target_revoke':
            await g.authority.revoke_credential(g.root,target)
        elif change=='target_revision':
            await g.authority.revise_credential(g.root,target,role=DashboardRole.HOME,expires_at=g.clock.now+80)
        elif change=='target_expiry':
            g.clock.now+=91
        else:
            closing = g.bus._retention.begin_close()
        release.set()
        response = await pending
        assert response.status_code==403 and 'token' not in response.text
        assert response.json()=={'detail':'dashboard_session_retired'}
        assert g.authority._attempts[intent['request_id']].outcome.value=='committed'
    finally:
        release.set()
        await asyncio.gather(pending,return_exceptions=True)
        if closing is not None:
            await closing


@pytest.mark.parametrize('modality',list(EvaluationModality))
async def test_evaluation_grant_uses_actual_scope_and_replay_needs_no_sources(credential_http,modality):
    g,client = credential_http
    intent = issue_body(g)
    intent.update(role='evaluator',selections=[{'fingerprint_id':g.identity,
        'modalities':[modality.value],'expires_at':g.clock.now+20}])
    response = await client.post(BASE,json=intent)
    assert response.status_code==200 and response.json()['applied']
    created = response.json()
    session = await g.authority.login(created['token'])
    selected = EvaluationReadActor(session.lease,g.identity,modality)
    projection = await g.voices.acall(g.voices.evaluation.read,selected)
    assert projection.rows[0]['modality']==modality.value
    assert not session.lease.current(frozenset({'auth:manage','voices:review'}))
    g.privacy.ledger._pending_conversations = frozenset({g.sources[0].conversation_id})
    unavailable = dict(intent,request_id=str(uuid4()))
    denied = (await client.post(BASE,json=unavailable)).json()
    assert not denied['applied'] and denied['receipt']['reason']=='scope_unavailable' and 'token' not in denied
    await g.authority.revoke_credential(g.root,created['receipt']['credential_id'])
    replay = (await client.post(BASE,json=intent)).json()
    assert replay=={'receipt':created['receipt'],'applied':False}


async def test_terminal_replay_survives_requested_expiry(credential_http):
    g,client = credential_http
    intent = issue_body(g)
    first = (await client.post(BASE,json=intent)).json()
    g.clock.now+=91
    replay = await client.post(BASE,json=intent)
    assert replay.status_code==200 and replay.json()=={'receipt':first['receipt'],'applied':False}
    assert (await client.get(BASE)).json()['rows']==[]


@pytest.mark.parametrize('change',['revoke','revise'])
async def test_target_change_before_issued_lease_lookup_never_returns_secret(credential_http,monkeypatch,change):
    g,client = credential_http
    intent = issue_body(g)
    original = g.authority.commands.execute
    tokens = []
    async def changed(*args,**kwargs):
        result = await original(*args,**kwargs)
        tokens.append(result.token)
        if change=='revoke':
            await g.authority.revoke_credential(g.root,result.receipt.credential_id)
        else:
            await g.authority.revise_credential(g.root,result.receipt.credential_id,
                role=DashboardRole.HOME,expires_at=g.clock.now+80)
        return result
    monkeypatch.setattr(g.authority.commands,'execute',changed)
    response = await client.post(BASE,json=intent)
    assert response.status_code==403 and tokens[0] not in response.text
    assert g.authority._attempts[intent['request_id']].outcome.value=='committed'


@pytest.mark.parametrize('change',['issuer','target'])
async def test_retirement_while_actual_headers_wait_aborts_private_body(credential_http,change):
    g,_ = credential_http
    entered,release = asyncio.Event(),asyncio.Event()
    messages = []
    async def gated(scope,receive,send):
        async def actual(message):
            if message['type']=='http.response.start':
                entered.set()
                await release.wait()
            messages.append(message.copy())
            await send(message)
        await g.api.app(scope,receive,actual)
    intent = issue_body(g)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gated),base_url='http://testserver',
            cookies={COOKIE_NAME:g.issuer_cookie}) as client:
        pending = asyncio.create_task(client.post(BASE,json=intent))
        try:
            await until(entered.is_set)
            if change=='issuer':
                await g.authority.logout(g.issuer)
            else:
                await g.authority.revoke_credential(g.root,intent['request_id'])
            release.set()
            with pytest.raises(AssertionError):
                await pending
            assert [item['type'] for item in messages]==['http.response.start']
            assert messages[0]['status']==200
        finally:
            release.set()
            await asyncio.gather(pending,return_exceptions=True)


async def test_cancelled_http_observer_keeps_real_commit_and_cannot_recover_secret(credential_http,monkeypatch):
    g,client = credential_http
    entered,release = threading.Event(),threading.Event()
    original = g.authority._transaction
    def held(kind,args,guard):
        if kind=='credential_command':
            entered.set()
            assert release.wait(5)
        return original(kind,args,guard)
    monkeypatch.setattr(g.authority,'_transaction',held)
    intent = issue_body(g)
    pending = asyncio.create_task(client.post(BASE,json=intent))
    try:
        assert await asyncio.to_thread(entered.wait,3)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert g.api.admin_work.status()['operations']==1
        assert g.bus._retention.rest._scopes
        release.set()
        await until(lambda:g.api.admin_work.status()['operations']==0 and not g.bus._retention.rest._scopes)
        assert g.authority.commands.attempt(g.root,intent['request_id'])['receipt']['outcome']=='committed'
        replay = await client.post(BASE,json=intent)
        assert replay.status_code==200 and 'token' not in replay.json()
    finally:
        release.set()
        await asyncio.gather(pending,return_exceptions=True)


async def test_transport_exception_does_not_keep_secret_in_own_traceback_frames(credential_http,monkeypatch):
    from src.dashboard import credential_commands
    g,_ = credential_http
    canary = 'kza_synthetic-http-secret-canary'
    monkeypatch.setattr(credential_commands.secrets,'token_urlsafe',lambda _:canary[4:])
    errors = []
    async def failing(scope,receive,send):
        async def actual(message):
            if message['type']=='http.response.body' and canary.encode() in message.get('body',b''):
                error = RuntimeError('synthetic transport error')
                errors.append(error)
                raise error
            await send(message)
        await g.api.app(scope,receive,actual)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=failing),base_url='http://testserver',
            cookies={COOKIE_NAME:g.issuer_cookie}) as client:
        with pytest.raises(AssertionError):
            await client.post(BASE,json=issue_body(g))
    assert len(errors)==1
    own_frames = []
    tb = errors[0].__traceback__
    while tb is not None:
        if '/src/dashboard/' in tb.tb_frame.f_code.co_filename:
            own_frames.append(tb.tb_frame)
            assert all(not isinstance(value,(str,bytes)) or canary not in str(value)
                for value in tb.tb_frame.f_locals.values())
            if 'message' in tb.tb_frame.f_locals:
                assert tb.tb_frame.f_locals['message'] is None
        tb = tb.tb_next
    assert len(own_frames)>=2


async def test_original_login_and_command_conflicts_do_not_reissue(credential_http):
    g,client = credential_http
    intent = issue_body(g)
    first = (await client.post(BASE,json=intent)).json()
    changed = await client.post(BASE,json=dict(intent,role='operator'))
    assert changed.status_code==409 and 'token' not in changed.json()
    second_session = await g.authority.login('synthetic-evaluation-root')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=g.api.app),base_url='http://testserver',
            cookies={COOKIE_NAME:second_session.token}) as other:
        conflict = await other.post(BASE,json=intent)
        assert conflict.status_code==409 and first['token'] not in conflict.text
        receipt = await other.get(ATTEMPTS+intent['request_id'])
        assert receipt.status_code==200 and receipt.json()['receipt']==first['receipt']
    assert len(g.authority._credentials)==1


async def test_concurrent_http_cas_loser_keeps_the_winning_credential_current(credential_http):
    g,client = credential_http
    created = (await client.post(BASE,json=issue_body(g))).json()
    identity = created['receipt']['credential_id']
    edits = [{'request_id':str(uuid4()),'expected_revision':1,'role':role,'expires_at':g.clock.now+70}
        for role in ('home','operator')]
    responses = await asyncio.gather(*(client.put(BASE+'/'+identity,json=value) for value in edits))
    assert all(response.status_code==200 for response in responses)
    winners = [index for index,response in enumerate(responses) if response.json()['applied']]
    assert len(winners)==1
    lease = g.authority.resolve(created['token'])
    assert lease.current() and lease.principal.revision==2
    assert lease.principal.role.value==edits[winners[0]]['role']
    assert not g.authority._credential_reservations and not g.authority._credential_revoking


async def test_send_capacity_exception_clears_undelivered_secret_message(credential_http):
    g,_ = credential_http
    owner = g.bus._retention.rest
    release = asyncio.Event()
    canary = b'synthetic-undelivered-secret-canary'
    error = None
    async def application(scope,receive,send):
        nonlocal error
        pending = [asyncio.create_task(send({'type':'http.response.body','body':canary})) for _ in range(3)]
        try:
            await until(lambda:pending[2].done())
            error = pending[2].exception()
        finally:
            release.set()
            await asyncio.gather(*pending,return_exceptions=True)
    async def actual(message):
        await release.wait()
    async def receive():
        return {'type':'http.disconnect'}
    await owner._scope(RestOperation(sources=()),application,{'type':'http'},receive,actual)
    assert isinstance(error,RuntimeError) and str(error)=='rest_transport_capacity'
    tb = error.__traceback__
    checked = False
    while tb is not None:
        if tb.tb_frame.f_code.co_name=='owned_send':
            assert tb.tb_frame.f_locals['message'] is None
            checked = True
        tb = tb.tb_next
    assert checked


@pytest.mark.parametrize('kind',['missing','wrong','foreign_origin'])
@pytest.mark.parametrize('method,path',[
    ('GET',BASE),('GET',ATTEMPTS+'11111111-1111-4111-8111-111111111111'),
    ('POST',BASE),('PUT',BASE+'/synthetic-target'),('DELETE',BASE+'/synthetic-target'),
])
async def test_real_authority_authentication_precedes_input_and_commands(credential_http,monkeypatch,kind,method,path):
    from src.dashboard import credential_api
    g,_ = credential_http
    async def forbidden(*args,**kwargs):
        pytest.fail('authentication allowed private input or native command')
    monkeypatch.setattr(credential_api,'_body',forbidden)
    monkeypatch.setattr(g.authority.commands,'execute',forbidden)
    headers,cookies = {},{}
    if kind=='wrong':
        headers={'Authorization':'Bearer synthetic-wrong-credential'}
    elif kind=='foreign_origin':
        headers={'Origin':'https://foreign.invalid'}
        cookies={COOKIE_NAME:g.issuer_cookie}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=g.api.app),base_url='http://testserver',
            headers=headers,cookies=cookies) as client:
        response = await client.request(method,path,content=b'not JSON')
    assert response.status_code==(403 if kind=='foreign_origin' else 401)
    assert response.headers['cache-control']=='no-store' and not g.authority._attempts


async def test_legacy_bearer_without_native_authority_never_reaches_body_or_effect(credential_http,monkeypatch):
    from src.dashboard import credential_api
    g,_ = credential_http
    token='synthetic-legacy-only-token'
    monkeypatch.setenv('KZA_DASHBOARD_TOKEN',token)
    async def forbidden(*args,**kwargs):
        pytest.fail('legacy authentication reached native credential work')
    monkeypatch.setattr(credential_api,'_body',forbidden)
    monkeypatch.setattr(g.authority.commands,'execute',forbidden)
    g.api.app.state.dashboard_authority = None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=g.api.app),base_url='http://testserver',
                headers={'Authorization':'Bearer '+token}) as client:
            response = await client.post(BASE,content=b'not JSON')
        assert response.status_code==503 and response.headers['cache-control']=='no-store'
        assert not g.authority._attempts
    finally:
        g.api.app.state.dashboard_authority = g.authority
