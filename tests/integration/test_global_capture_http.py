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
from src.dashboard.authority import DashboardLease, DashboardRole, EvaluationModality, EvaluationSample
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
    'pending_browser_recorders', 'pending_native_clients', 'error_code',
    'live_capture_owners', 'carried_capture_owners',
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


FORGET = URL + '/forget-carried'


@asynccontextmanager
async def carried_application(tmp_path, leaves, count=2):
    """R1-B2: a first boot issues ``count`` browser owners; the second carries them."""
    from tests.integration.test_global_capture_control import actor
    async with application(tmp_path, leaves) as graph:
        recorders, principal = graph.capture_control.recorders, actor()
        for _ in range(count):
            token = (await recorders.register(principal))['token']
            await recorders.activate(token, principal)
    async with application(tmp_path, leaves) as graph:
        assert graph.capture_control.recorders.carried() == count
        yield graph


async def test_operator_forgets_lost_recordings_with_the_count_shown(tmp_path, leaves, dashboard_auth, caplog):
    async with carried_application(tmp_path, leaves) as graph:
        async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
            await login(client, dashboard_auth['Authorization'].split(' ', 1)[1])
            before = (await client.get(URL)).json()
            assert before['carried_capture_owners'] == 2 and before['live_capture_owners'] == 0
            assert (await client.post(URL, json={'muted': True, 'expected_revision': before['revision']})).status_code == 202
            muted = await settled(client, 'muted')
            assert muted['total_mute_confirmed'] is False
            # A stale page that showed another count forgets nothing.
            for stale in (0, 1, 3):
                response = await client.post(FORGET, json={'expected_carried': stale, 'no_capture_running': True})
                assert response.status_code == 409
                assert response.json() == {'detail': 'capture_carried_count_conflict'}
                assert response.headers['cache-control'] == 'no-store'
            assert (await client.get(URL)).json() == muted
            with caplog.at_level('WARNING'):
                response = await client.post(FORGET, json={'expected_carried': 2, 'no_capture_running': True})
            assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
            forgotten = response.json()
            assert set(forgotten) == FIELDS | {'forgotten_capture_owners'}
            assert forgotten['forgotten_capture_owners'] == 2
            assert forgotten['carried_capture_owners'] == forgotten['pending_browser_recorders'] == 0
            assert forgotten['total_mute_confirmed'] is True and forgotten['revision'] == muted['revision']
            assert [r.getMessage() for r in caplog.records if 'forgot' in r.getMessage()] == [
                '[CaptureRecorders] an administrator forgot 2 carried owner(s), affirming none is still recording']
            # Idempotent: the repeated request forgets nothing more.
            again = await client.post(FORGET, json={'expected_carried': 2, 'no_capture_running': True})
            assert again.status_code == 200 and again.json()['forgotten_capture_owners'] == 0


@pytest.mark.parametrize('payload', [
    {}, {'expected_carried': 2}, {'no_capture_running': True},
    {'expected_carried': 2, 'no_capture_running': False},
    {'expected_carried': 2, 'no_capture_running': 1},
    {'expected_carried': 2, 'no_capture_running': 'true'},
    {'expected_carried': 2, 'no_capture_running': None},
    {'expected_carried': True, 'no_capture_running': True},
    {'expected_carried': -1, 'no_capture_running': True},
    {'expected_carried': 2.0, 'no_capture_running': True},
    {'expected_carried': '2', 'no_capture_running': True},
    {'expected_carried': 2, 'no_capture_running': True, 'user_id': 'synthetic-account'},
    {'expected_carried': 2, 'no_capture_running': True, 'live': True},
])
async def test_forget_body_requires_the_explicit_affirmation_and_count(tmp_path, leaves, dashboard_auth, payload):
    async with carried_application(tmp_path, leaves) as graph:
        async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
            assert (await client.post(FORGET, headers=dashboard_auth, json=payload)).status_code == 422
            assert (await client.post(FORGET, headers={**dashboard_auth, 'Content-Type': 'application/json'},
                content='{"expected_carried":2,"expected_carried":2,"no_capture_running":true}')).status_code == 422
            assert (await client.post(FORGET, headers=dashboard_auth,
                content='expected_carried=2&no_capture_running=true')).status_code == 415
        assert graph.capture_control.recorders.carried() == 2


@pytest.mark.parametrize('role', [DashboardRole.HOME, DashboardRole.EVALUATOR])
async def test_household_and_evaluator_roles_cannot_forget(tmp_path, leaves, dashboard_auth, role):
    async with carried_application(tmp_path, leaves) as graph:
        now = time.time()
        samples = (EvaluationSample('vf_' + 'a' * 64, frozenset({'b' * 64}),
            frozenset({EvaluationModality.AUDIO_PREVIEW}), now + 60),) if role is DashboardRole.EVALUATOR else ()
        grant = await graph.dashboard_authority.issue_credential(
            graph.dashboard_authority.resolve(dashboard_auth['Authorization']),
            subject='synthetic-forget-member', role=role, expires_at=now + 120, samples=samples,
        )
        async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
            await login(client, grant.token)
            response = await client.post(FORGET, json={'expected_carried': 2, 'no_capture_running': True})
            assert response.status_code == 403
        async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
            assert (await client.post(FORGET, json={'expected_carried': 2, 'no_capture_running': True})).status_code == 401
        assert graph.capture_control.recorders.carried() == 2


async def test_session_lapsing_inside_the_forget_transaction_is_a_403(tmp_path, leaves, dashboard_auth, monkeypatch, caplog):
    """R1B2-1: the HTTP wiring rechecks the session inside the transaction.

    The operator's lease answers True for the route, the controller and the
    entry checks; it lapses only once the transaction selected the carried
    rows, so only the in-transaction recheck can refuse the forget.
    """
    async with carried_application(tmp_path, leaves) as graph:
        recorders = graph.capture_control.recorders
        async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
            await login(client, dashboard_auth['Authorization'].split(' ', 1)[1])
            before = (await client.get(URL)).json()
            original = DashboardLease.current
            checks = []

            def lapsing(self, permissions=frozenset()):
                checks.append(bool(recorders._forgetting))
                return not recorders._forgetting and original(self, permissions)

            with monkeypatch.context() as patch, caplog.at_level('WARNING'):
                patch.setattr(DashboardLease, 'current', lapsing)
                response = await client.post(FORGET, json={'expected_carried': 2, 'no_capture_running': True})
            assert response.status_code == 403 and response.json() == {'detail': 'capture_control_denied'}
            assert response.headers['cache-control'] == 'no-store'
            # Refused by a check made while the rows were selected, not at entry.
            assert checks[0] is False and True in checks
            assert not [r for r in caplog.records if 'forgot' in r.getMessage()]
            assert (await client.get(URL)).json() == before
        assert recorders.carried() == recorders.pending() == 2 and not recorders._fault


async def test_operator_session_from_a_foreign_origin_or_form_body_cannot_forget(tmp_path, leaves, dashboard_auth, caplog):
    """R1B2-4: a valid operator cookie is not enough for a cross-site forget.

    The session holds admin:manage, so only the Origin check (403) and the JSON
    content-type requirement (415) refuse these requests.
    """
    async with carried_application(tmp_path, leaves) as graph:
        async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
            await login(client, dashboard_auth['Authorization'].split(' ', 1)[1])
            before = (await client.get(URL)).json()
            assert before['carried_capture_owners'] == 2
            with caplog.at_level('WARNING'):
                for origin in ('http://foreign.invalid', 'null', 'http://testserver:8080', 'https://testserver'):
                    response = await client.post(FORGET, headers={'Origin': origin},
                        json={'expected_carried': 2, 'no_capture_running': True})
                    assert response.status_code == 403, origin
                for content_type in ('text/plain', 'application/x-www-form-urlencoded', 'multipart/form-data; boundary=x'):
                    response = await client.post(FORGET, headers={'Content-Type': content_type},
                        content='{"expected_carried":2,"no_capture_running":true}')
                    assert response.status_code == 415, content_type
                    assert response.json() == {'detail': 'capture_control_json_required'}
            assert not [r for r in caplog.records if 'forgot' in r.getMessage()]
            assert (await client.get(URL)).json() == before
            # The same session, same-origin and JSON, is admitted.
            response = await client.post(FORGET, headers={'Origin': 'http://testserver'},
                json={'expected_carried': 2, 'no_capture_running': True})
            assert response.status_code == 200 and response.json()['forgotten_capture_owners'] == 2
        assert graph.capture_control.recorders.carried() == 0
