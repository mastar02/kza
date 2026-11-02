"""Original session authority governs actual private HTTP body delivery."""

import asyncio

from httpx import ASGITransport, AsyncClient
import pytest
from starlette.responses import StreamingResponse

from src.dashboard.api import DashboardAPI
from src.dashboard.authority import DashboardRole
from tests.integration.test_memory_session_authority import proposal, review_graph as memory_review_graph

pytestmark = pytest.mark.asyncio
review_graph = memory_review_graph


@pytest.mark.parametrize('path', ['/api/memory', '/api/memory/facts', '/api/memory/proposals'])
@pytest.mark.parametrize('change', ['logout', 'expire', 'rotate', 'revision'])
@pytest.mark.parametrize('stage', ['read', 'headers'])
@pytest.mark.parametrize('buffered', [False, True])
async def test_original_session_retirement_prevents_private_body(
    review_graph, monkeypatch, dashboard_auth, path, change, stage, buffered,
):
    graph = review_graph
    private = 'Synthetic preference restricted to this original session.'
    await proposal(graph, content=private)
    api = DashboardAPI(authority=graph.authority, memory_manager=graph.manager)
    graph.shutdown.push_async_callback(api.stop)
    if buffered:
        @api.app.middleware('http')
        async def buffering(request, call_next):
            return await call_next(request)
    if change == 'revision':
        root = graph.authority.resolve(dashboard_auth['Authorization'])
        grant = await graph.authority.issue_credential(root, subject='synthetic-reviewer',
            role=DashboardRole.OPERATOR, expires_at=graph.clock.now + 120)
        assert (await graph.client.post('/api/admin/auth/login', json={'token': grant.token})).status_code == 200
    entered, release = asyncio.Event(), asyncio.Event()
    messages = []
    original = graph.service.acall

    async def held_read(function, *args, **kwargs):
        value = await original(function, *args, **kwargs)
        if stage == 'read' and function.__name__ in {'snapshot', 'list', 'proposals'}:
            entered.set()
            await release.wait()
        return value

    monkeypatch.setattr(graph.service, 'acall', held_read)

    async def gated(scope, receive, send):
        async def actual(message):
            if stage == 'headers' and message['type'] == 'http.response.start':
                entered.set()
                await release.wait()
            messages.append(message.copy())
            await send(message)
        await api.app(scope, receive, actual)

    async with AsyncClient(transport=ASGITransport(app=gated), base_url='http://testserver',
                           cookies=graph.client.cookies) as client:
        pending = asyncio.create_task(client.get(path))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if change == 'logout':
                assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
                token = dashboard_auth['Authorization'].split(' ', 1)[1]
                assert (await graph.client.post('/api/admin/auth/login', json={'token': token})).status_code == 200
            elif change == 'expire':
                graph.clock.now += 60
            elif change == 'revision':
                assert await graph.authority.revise_credential(root, grant.identity,
                    role=DashboardRole.HOME, expires_at=graph.clock.now + 90)
            else:
                monkeypatch.setenv('KZA_DASHBOARD_TOKEN', 'synthetic-replacement-administrator')
            release.set()
            if stage == 'headers':
                # ASGITransport reports the deliberately incomplete response. A real
                # server closes its transport; no body or successful replacement exists.
                with pytest.raises(AssertionError):
                    await pending
                assert [m['type'] for m in messages] == ['http.response.start']
                assert messages[0]['status'] == 200
            else:
                response = await pending
                assert response.status_code == 403
                assert response.json() == {'detail': 'dashboard_session_retired'}
                assert response.headers['cache-control'] == 'no-store'
                assert int(response.headers['content-length']) == len(response.content)
            assert private.encode() not in b''.join(m.get('body', b'') for m in messages)
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize('buffered', [False, True])
async def test_stream_retirement_preserves_sent_fragment_and_aborts_remaining_body(review_graph, buffered):
    graph = review_graph
    api = DashboardAPI(authority=graph.authority)
    graph.shutdown.push_async_callback(api.stop)
    if buffered:
        @api.app.middleware('http')
        async def buffering(request, call_next):
            return await call_next(request)
    entered, release = asyncio.Event(), asyncio.Event()
    messages = []
    admitted = b'Synthetic fragment admitted before retirement.'
    withheld = b'Synthetic private fragment after retirement.'

    async def chunks():
        yield admitted
        await release.wait()
        yield withheld

    @api.app.get('/api/private-stream')
    async def stream():
        return StreamingResponse(chunks(), media_type='text/plain')
    # This synthetic route is registered after the dashboard's catch-all asset mount.
    api.app.router.routes.insert(0, api.app.router.routes.pop())

    async def gated(scope, receive, send):
        async def actual(message):
            messages.append(message.copy())
            await send(message)
            if message.get('body') == admitted:
                entered.set()
        await api.app(scope, receive, actual)

    async with AsyncClient(transport=ASGITransport(app=gated), base_url='http://testserver',
                           cookies=graph.client.cookies) as client:
        pending = asyncio.create_task(client.get('/api/private-stream'))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
            release.set()
            with pytest.raises(AssertionError):
                await pending
            bodies = [m for m in messages if m['type'] == 'http.response.body']
            assert b''.join(m.get('body', b'') for m in bodies) == admitted
            assert all(m.get('more_body') is True for m in bodies)
            assert [m['status'] for m in messages if m['type'] == 'http.response.start'] == [200]
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize('change', ['logout', 'expire', 'rotate'])
async def test_login_checks_new_session_before_publishing_cookie(
    review_graph, dashboard_auth, monkeypatch, change,
):
    graph = review_graph
    api = DashboardAPI(authority=graph.authority)
    graph.shutdown.push_async_callback(api.stop)
    entered, release = asyncio.Event(), asyncio.Event()
    issued = []
    original = graph.authority.login

    async def login(token):
        result = await original(token)
        issued.append(result.lease)
        return result

    monkeypatch.setattr(graph.authority, 'login', login)

    @api.app.middleware('http')
    async def held_response(request, call_next):
        response = await call_next(request)
        entered.set()
        await release.wait()
        return response

    async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://testserver') as client:
        token = dashboard_auth['Authorization'].split(' ', 1)[1]
        pending = asyncio.create_task(client.post('/api/admin/auth/login', json={'token': token}))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert len(issued) == 1
            if change == 'logout':
                assert await graph.authority.logout(issued[0])
            elif change == 'expire':
                graph.clock.now += 60
            else:
                monkeypatch.setenv('KZA_DASHBOARD_TOKEN', 'synthetic-new-login-root')
            release.set()
            response = await asyncio.wait_for(pending, 3)
            assert response.status_code == 403
            assert response.json() == {'detail': 'dashboard_session_retired'}
            assert 'set-cookie' not in response.headers
            assert not client.cookies
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize('buffered', [False, True])
async def test_denied_stream_completes_without_waiting_for_original_first_chunk(review_graph, buffered):
    graph = review_graph
    api = DashboardAPI(authority=graph.authority)
    graph.shutdown.push_async_callback(api.stop)
    if buffered:
        @api.app.middleware('http')
        async def buffering(request, call_next):
            return await call_next(request)
    entered, release, body_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    body_started, body_finished = [], []

    async def chunks():
        body_started.append(True)
        try:
            await body_release.wait()
            yield b'Synthetic private stream content.'
        finally:
            body_finished.append(True)

    class HeldStream(StreamingResponse):
        async def __call__(self, scope, receive, send):
            entered.set()
            await release.wait()
            await super().__call__(scope, receive, send)

    @api.app.get('/api/private-stream')
    async def stream():
        return HeldStream(chunks(), media_type='text/plain')
    api.app.router.routes.insert(0, api.app.router.routes.pop())

    async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://testserver',
                           cookies=graph.client.cookies) as client:
        pending = asyncio.create_task(client.get('/api/private-stream'))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert (await graph.client.post('/api/admin/auth/logout')).json()['session_revoked']
            release.set()
            response = await asyncio.wait_for(pending, 3)
            assert not body_release.is_set()
            assert response.status_code == 403
            assert response.json() == {'detail': 'dashboard_session_retired'}
            assert body_finished == body_started
        finally:
            release.set()
            body_release.set()
            await asyncio.gather(pending, return_exceptions=True)
