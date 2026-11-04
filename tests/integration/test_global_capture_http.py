"""Global capture authority through the real application and HTTP sessions."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
import threading
import time
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
import pytest

from src.dashboard.auth import COOKIE_NAME
from src.dashboard.authority import DashboardRole, EvaluationModality, EvaluationSample
from src.main import build_application
from src.memory.capture_control import CaptureControl, CaptureController
from src.memory.producer_context import producer_context
from tests.integration.test_capability_application import configuration, leaves as capability_leaves
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.application_graph import OfflineSTT
from tests.mocks.capture_models import capture_factories, capture_stt
from tests.mocks.task_transport import TaskTransport
from tests.mocks.voice_registry import AUDIO

pytestmark = pytest.mark.asyncio
URL = '/api/capture/control'
FIELDS = {
    'revision', 'target_muted', 'persisted_muted', 'phase', 'ingress_sealed',
    'supported', 'pending_operations', 'total_mute_confirmed',
    'pending_browser_recorders', 'error_code',
}


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    return capture_factories(capability_leaves.__wrapped__(monkeypatch, tmp_path))


@asynccontextmanager
async def application(tmp_path, leaves):
    async with TaskTransport().serve() as url:
        config, _ = configuration(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            yield graph


@pytest.fixture
async def capture_graph(tmp_path, leaves, dashboard_auth):
    async with application(tmp_path, leaves) as graph:
        yield graph


async def test_factory_foreign_model_cannot_claim_global_mute_or_become_owned(tmp_path, leaves, dashboard_auth):
    foreign = OfflineSTT()
    close_calls = []
    foreign.close = lambda: close_calls.append("close")
    async with application(tmp_path, replace(leaves, stt=lambda _: foreign)) as graph:
        assert graph.stt_executors._capture_unreviewed
        assert id(foreign) not in graph.stt_executors._models
        async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
            await login(client, dashboard_auth['Authorization'].split(' ', 1)[1])
            before = (await client.get(URL)).json()
            assert before['supported'] is False and not before['total_mute_confirmed']
            response = await client.post(URL, json={'muted': True, 'expected_revision': before['revision']})
            assert response.status_code == 503
            after = (await client.get(URL)).json()
            assert after['supported'] is False and not after['total_mute_confirmed']
            assert after['revision'] == before['revision']
    assert close_calls == []


async def login(client, token):
    response = await client.post('/api/admin/auth/login', json={'token': token})
    assert response.status_code == 200
    assert client.cookies[COOKIE_NAME]


async def settled(client, phase):
    async with asyncio.timeout(4):
        while True:
            response = await client.get(URL)
            assert response.status_code == 200
            state = response.json()
            if state['phase'] == phase:
                return state
            await asyncio.sleep(.001)


async def test_factory_installs_real_control_and_opaque_operator_cookie(capture_graph, dashboard_auth):
    graph = capture_graph
    assert isinstance(graph.capture_control, CaptureControl)
    assert graph.dashboard.capture_control is graph.capture_control
    async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
        assert (await client.get(URL)).status_code == 401
        token = dashboard_auth['Authorization'].split(' ', 1)[1]
        await login(client, token)
        assert client.cookies[COOKIE_NAME] != token
        before = (await client.get(URL)).json()
        assert set(before) == FIELDS
        assert before['phase'] == 'running' and not before['total_mute_confirmed']
        assert (await client.post(URL, json={'muted': True, 'expected_revision': before['revision']})).status_code == 202
        muted = await settled(client, 'muted')
        assert muted['persisted_muted'] and muted['target_muted']
        assert muted['total_mute_confirmed'] and muted['ingress_sealed']
        assert muted['pending_operations'] == 0 and set(muted) == FIELDS
        assert (await client.post(URL, json={'muted': False, 'expected_revision': muted['revision']})).status_code == 202
        running = await settled(client, 'running')
        assert not running['target_muted'] and not running['persisted_muted']
        assert not running['total_mute_confirmed']


@pytest.mark.parametrize('role', [DashboardRole.HOME, DashboardRole.EVALUATOR])
async def test_other_real_roles_cannot_read_or_change_global_policy(capture_graph, dashboard_auth, role):
    graph = capture_graph
    before = graph.capture_control.snapshot()
    now = time.time()
    samples = (EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}),
        frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 60),) if role is DashboardRole.EVALUATOR else ()
    grant = await graph.dashboard_authority.issue_credential(
        graph.dashboard_authority.resolve(dashboard_auth['Authorization']),
        subject='synthetic-capture-member', role=role, expires_at=now + 120, samples=samples,
    )
    async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
        response = await client.post('/api/admin/auth/login', json={
            'token': grant.token, 'role': 'operator', 'permissions': ['admin:manage'],
            'user_id': 'synthetic-untrusted-person',
        })
        assert response.status_code == 200
        assert (await client.get(URL)).status_code == 403
        assert (await client.post(URL, json={'muted': True, 'expected_revision': before['revision']})).status_code == 403
    assert graph.capture_control.snapshot() == before


@pytest.mark.parametrize('payload', [
    {}, {'muted': True}, {'expected_revision': 0},
    {'muted': 1, 'expected_revision': 0}, {'muted': 'true', 'expected_revision': 0},
    {'muted': None, 'expected_revision': 0}, {'muted': True, 'expected_revision': True},
    {'muted': True, 'expected_revision': -1}, {'muted': True, 'expected_revision': 0.0},
    {'muted': True, 'expected_revision': '0'},
    {'muted': True, 'expected_revision': 0, 'origin': 'voice'},
    {'muted': True, 'expected_revision': 0, 'actor': 'operator'},
    {'muted': True, 'expected_revision': 0, 'user_id': 'synthetic-account'},
])
async def test_strict_body_rejects_untrusted_claims_before_mutation(capture_graph, dashboard_auth, payload):
    before = capture_graph.capture_control.snapshot()
    async with AsyncClient(transport=ASGITransport(app=capture_graph.dashboard.app), base_url='http://testserver') as client:
        assert (await client.post(URL, headers=dashboard_auth, json=payload)).status_code == 422
    assert capture_graph.capture_control.snapshot() == before


@pytest.mark.parametrize('body', [
    '{"muted":true,"muted":false,"expected_revision":0}',
    '{"muted":true,"expected_revision":0,"expected_revision":1}', '{', 'null', '[]',
])
async def test_duplicate_or_malformed_json_cannot_advance_revision(capture_graph, dashboard_auth, body):
    before = capture_graph.capture_control.snapshot()
    async with AsyncClient(transport=ASGITransport(app=capture_graph.dashboard.app), base_url='http://testserver') as client:
        response = await client.post(URL, headers={**dashboard_auth, 'Content-Type': 'application/json'}, content=body)
        assert response.status_code == 422
    assert capture_graph.capture_control.snapshot() == before


@pytest.mark.parametrize('origin', ['http://foreign.invalid', 'null', 'https://testserver',
    'http://testserver:81', 'http://testserver/path', 'http://user@testserver'])
async def test_foreign_or_malformed_origin_cannot_use_valid_operator_cookie(capture_graph, dashboard_auth, origin):
    before = capture_graph.capture_control.snapshot()
    async with AsyncClient(transport=ASGITransport(app=capture_graph.dashboard.app), base_url='http://testserver') as client:
        await login(client, dashboard_auth['Authorization'].split(' ', 1)[1])
        response = await client.post(URL, headers={'Origin': origin},
            json={'muted': True, 'expected_revision': before['revision']})
        assert response.status_code == 403
    assert capture_graph.capture_control.snapshot() == before


async def test_no_origin_and_exact_origin_preserve_revision_conflicts(capture_graph, dashboard_auth):
    async with AsyncClient(transport=ASGITransport(app=capture_graph.dashboard.app), base_url='http://testserver') as client:
        await login(client, dashboard_auth['Authorization'].split(' ', 1)[1])
        initial = (await client.get(URL)).json()
        assert (await client.post(URL, json={'muted': True, 'expected_revision': initial['revision']})).status_code == 202
        muted = await settled(client, 'muted')
        assert (await client.post(URL, json={'muted': False, 'expected_revision': initial['revision']})).status_code == 409
        assert capture_graph.capture_control.snapshot() == muted
        assert (await client.post(URL, headers={'Origin': 'http://testserver:80'},
            json={'muted': False, 'expected_revision': muted['revision']})).status_code == 202
        await settled(client, 'running')


async def test_non_json_request_cannot_mutate_global_state(capture_graph, dashboard_auth):
    before = capture_graph.capture_control.snapshot()
    async with AsyncClient(transport=ASGITransport(app=capture_graph.dashboard.app), base_url='http://testserver') as client:
        assert (await client.post(URL, headers=dashboard_auth, content='muted=true&expected_revision=0')).status_code == 415
    assert capture_graph.capture_control.snapshot() == before


@pytest.mark.parametrize('revocation', ['logout', 'role_reduction', 'expiration'])
async def test_original_lease_cannot_be_replaced_after_revocation(capture_graph, dashboard_auth, monkeypatch, revocation):
    graph = capture_graph
    authority = graph.dashboard_authority
    now = [time.time()]
    monkeypatch.setattr(authority, '_clock', lambda: now[0])
    root = authority.resolve(dashboard_auth['Authorization'])
    grant = await authority.issue_credential(root, subject='synthetic-revoked-controller',
        role=DashboardRole.OPERATOR, expires_at=now[0] + 60)
    async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
        await login(client, grant.token)
        lease = authority.resolve(None, client.cookies[COOKIE_NAME])
        original = CaptureController.from_dashboard(lease.principal, lease.current)
        assert (await client.post(URL, json={'muted': True,
            'expected_revision': graph.capture_control.snapshot()['revision']})).status_code == 202
        muted = await settled(client, 'muted')
        if revocation == 'logout':
            assert (await client.post('/api/admin/auth/logout')).status_code == 200
        elif revocation == 'role_reduction':
            assert await authority.revise_credential(root, grant.identity,
                role=DashboardRole.HOME, expires_at=now[0] + 50)
        else:
            now[0] += 61
        with pytest.raises(PermissionError):
            original.require_current()
        with pytest.raises(PermissionError):
            graph.capture_control.request(False, muted['revision'], original)
        assert (await client.post(URL, json={'muted': False, 'expected_revision': muted['revision']})).status_code == 401
        assert graph.capture_control.snapshot() == muted
        if revocation == 'role_reduction':
            assert (await client.post(URL, headers={'Authorization': 'Bearer ' + grant.token},
                json={'muted': False, 'expected_revision': muted['revision']})).status_code == 403


async def test_queued_http_resume_keeps_original_cookie_lease_until_transaction(capture_graph, dashboard_auth, monkeypatch):
    graph = capture_graph
    control = graph.capture_control
    entered, release = threading.Event(), threading.Event()
    write = control._write_mode

    def held(mode, revision, controller):
        if mode == 'resuming':
            entered.set()
            assert release.wait(4)
        return write(mode, revision, controller)

    async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
        await login(client, dashboard_auth['Authorization'].split(' ', 1)[1])
        assert (await client.post(URL, json={'muted': True,
            'expected_revision': control.snapshot()['revision']})).status_code == 202
        muted = await settled(client, 'muted')
        monkeypatch.setattr(control, '_write_mode', held)
        try:
            assert (await client.post(URL, json={'muted': False,
                'expected_revision': muted['revision']})).status_code == 202
            await until(entered.is_set)
            assert (await client.post('/api/admin/auth/logout')).status_code == 200
            assert control.snapshot()['ingress_sealed']
        finally:
            release.set()
        await until(lambda: control.snapshot()['phase'] == 'failed')
        assert (await graph.privacy.acall(control._load_policy))[0] == 'muted'
        assert control.snapshot()['persisted_muted'] and control.snapshot()['ingress_sealed']
        assert (await client.get(URL)).status_code == 401


async def test_202_keeps_real_factory_native_command_owned_until_return(tmp_path, leaves, dashboard_auth):
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def recognize(audio, *, language):
        entered.set()
        assert release.wait(4)
        try:
            return 'enciende la luz'
        finally:
            returned.set()

    stt = capture_stt(recognize=recognize)
    async with application(tmp_path, replace(leaves, stt=lambda _: stt)) as graph:
        source = graph.privacy.source_for_capture(str(uuid4()), time.time(), 'synthetic-command')
        with producer_context('synthetic-command', (source,)):
            command = asyncio.create_task(graph.pipeline.command_processor.process_command(AUDIO.copy()))
        try:
            await until(entered.is_set)
            command.cancel()
            await asyncio.gather(command, return_exceptions=True)
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
                await login(client, dashboard_auth['Authorization'].split(' ', 1)[1])
                before = (await client.get(URL)).json()
                assert (await client.post(URL, json={'muted': True, 'expected_revision': before['revision']})).status_code == 202
                pending = (await client.get(URL)).json()
                assert pending['phase'] == 'muting' and pending['ingress_sealed']
                assert pending['pending_operations'] > 0 and not pending['total_mute_confirmed']
                assert not returned.is_set()
                release.set()
                muted = await settled(client, 'muted')
                assert returned.is_set() and muted['total_mute_confirmed']
                assert muted['pending_operations'] == 0
        finally:
            release.set()
            await asyncio.gather(command, return_exceptions=True)


async def test_factory_restart_restores_durable_mute_before_any_run(tmp_path, leaves, dashboard_auth):
    async with application(tmp_path, leaves) as graph:
        async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
            assert (await client.post(URL, headers=dashboard_auth,
                json={'muted': True, 'expected_revision': graph.capture_control.snapshot()['revision']})).status_code == 202
            await until(lambda: graph.capture_control.snapshot()['phase'] == 'muted')
            old = graph.capture_control.snapshot()
    async with application(tmp_path, leaves) as graph:
        restoring = graph.capture_control.snapshot()
        assert restoring['revision'] >= old['revision']
        assert restoring['persisted_muted'] and restoring['target_muted']
        assert restoring['ingress_sealed'] and not graph._run_entered
        with pytest.raises(PermissionError):
            graph.privacy.source_for_capture(str(uuid4()), time.time(), 'synthetic-restart')
        await until(lambda: graph.capture_control.snapshot()['phase'] == 'muted')
        restored = graph.capture_control.snapshot()
        assert restored['total_mute_confirmed'] and restored['ingress_sealed']
        async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
            assert (await client.post(URL, headers=dashboard_auth,
                json={'muted': False, 'expected_revision': restored['revision']})).status_code == 202
            await until(lambda: graph.capture_control.snapshot()['phase'] == 'running')
        source = graph.privacy.source_for_capture(str(uuid4()), time.time(), 'synthetic-restart')
        assert graph.privacy.sources_available((source,))
