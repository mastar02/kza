"""Real HA errors and native probes cannot become fresh empty lists or demo rows."""

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
from threading import Event

from httpx import ASGITransport, AsyncClient
import pytest

from src.core.task_outcome import CURRENT_REQUEST_ID
from src.dashboard import system_monitor
from src.dashboard.api import DashboardAPI
from src.dashboard.observation_http import ObservationHeaders
from src.main import build_application
from tests.integration.test_capability_application import configuration, leaves as capability_leaves
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    return capability_leaves.__wrapped__(monkeypatch, tmp_path)


@pytest.mark.parametrize("demo", [None, False, True], ids=["omitted", "false", "explicit-demo"])
async def test_factory_requires_explicit_demo_and_preserves_real_empty(
    leaves, tmp_path, dashboard_auth, monkeypatch, demo,
):
    transport = TaskTransport()
    transport.states.clear()
    monkeypatch.setattr(system_monitor, 'gpu_snapshot', lambda: [])
    monkeypatch.setattr(system_monitor, 'services_snapshot', lambda: [])
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        config["dashboard"].pop("observability_use_mocks")
        if demo is not None:
            config["dashboard"]["observability_use_mocks"] = demo
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
                for path, source in (('/api/ha/entities', 'ha-rest-states'), ('/api/system/gpus', 'nvidia-smi'),
                                     ('/api/system/services', 'service-probes')):
                    response = await client.get(path, headers=dashboard_auth)
                    assert response.status_code == 200
                    if demo is True:
                        assert response.json()
                        assert response.headers['X-KZA-Source'] == 'mock'
                        assert response.headers['X-KZA-Observation-State'] == 'demo'
                        assert 'X-KZA-Observed-At' not in response.headers
                        assert 'X-KZA-Expires-At' not in response.headers
                    else:
                        assert response.json() == []
                        assert response.headers['X-KZA-Source'] == 'real'
                        assert response.headers['X-KZA-Source-Id'] == source
                        assert 'X-KZA-Observed-At' in response.headers and 'X-KZA-Expires-At' in response.headers


@pytest.mark.parametrize('status', [401, 403, 503, 'timeout'])
async def test_real_ha_failure_is_not_a_fresh_empty_inventory(leaves, tmp_path, dashboard_auth, monkeypatch, status):
    transport = TaskTransport()
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
                first = await client.get('/api/ha/entities', headers=dashboard_auth)
                assert first.headers['X-KZA-Source'] == 'real' and first.json()
                if status == 'timeout':
                    import src.home_assistant.ha_client as ha_module
                    monkeypatch.setattr(ha_module, 'REST_TIMEOUT_BULK', .03)
                    transport.states_hold = True
                else:
                    transport.states_status = status
                try:
                    failed = await client.get('/api/ha/entities', headers=dashboard_auth)
                    assert failed.json() == [] and failed.headers['X-KZA-Source'] == 'degraded'
                    assert failed.headers['X-KZA-Observation-State'] == ('denied' if status in (401, 403) else 'unavailable')
                    assert failed.headers['X-KZA-Observed-At'] == first.headers['X-KZA-Observed-At']
                    assert failed.headers['X-KZA-Expires-At'] == first.headers['X-KZA-Expires-At']
                finally:
                    transport.states_release.set()


@pytest.mark.parametrize('revoke', [False, True])
async def test_native_probe_is_off_loop_and_owned_after_cancel_or_session_revocation(
    leaves, tmp_path, dashboard_auth, monkeypatch, revoke,
):
    transport = TaskTransport()
    entered, release = Event(), Event()
    contexts = []

    def probe():
        contexts.append(CURRENT_REQUEST_ID.get())
        entered.set()
        release.wait()
        return [{'id': 'synthetic-gpu'}]

    monkeypatch.setattr(system_monitor, 'gpu_snapshot', probe)
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
                token = CURRENT_REQUEST_ID.set('synthetic-private-request')
                try:
                    request = asyncio.create_task(client.get('/api/system/gpus', headers=dashboard_auth))
                finally:
                    CURRENT_REQUEST_ID.reset(token)
                closing = None
                try:
                    await until(entered.is_set)
                    progress = asyncio.Event()
                    asyncio.get_running_loop().call_soon(progress.set)
                    await asyncio.wait_for(progress.wait(), 1)
                    assert contexts == [None]
                    if revoke:
                        monkeypatch.delenv('KZA_DASHBOARD_TOKEN')
                    else:
                        request.cancel()
                        await asyncio.gather(request, return_exceptions=True)
                        closing = asyncio.create_task(graph.dashboard.stop())
                        await until(lambda: graph.dashboard.app.state.observability_monitors._closed)
                        assert not closing.done()
                    release.set()
                    if revoke:
                        response = await request
                        assert response.status_code == 401 and response.headers['Cache-Control'] == 'no-store'
                        assert 'synthetic-gpu' not in response.text
                    else:
                        await asyncio.wait_for(closing, 2)
                        assert graph.dashboard.app.state.observability_monitors._tasks == {}
                finally:
                    release.set()
                    await asyncio.gather(request, *(tuple([closing]) if closing is not None else ()), return_exceptions=True)


async def test_shared_native_read_keeps_one_deadline_for_a_delayed_http_waiter(leaves, tmp_path, dashboard_auth, monkeypatch):
    offset, count = [0], [0]
    entered, release = Event(), Event()
    late, both = asyncio.Event(), asyncio.Event()
    original_init = ObservationHeaders.__init__

    def initialize(value):
        original_init(value, clock=lambda: datetime.now(UTC)+timedelta(seconds=offset[0]))

    def probe():
        count[0] += 1
        entered.set()
        release.wait()
        return []

    monkeypatch.setattr(ObservationHeaders, '__init__', initialize)
    monkeypatch.setattr(system_monitor, 'gpu_snapshot', probe)
    transport = TaskTransport()
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            monitors = graph.dashboard.app.state.observability_monitors
            original = monitors.sample
            callers = 0

            async def sample(component):
                nonlocal callers
                callers += 1
                index = callers
                if index == 2:
                    both.set()
                result = await original(component)
                if index == 2:
                    await late.wait()
                return result

            monitors.sample = sample
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
                first = asyncio.create_task(client.get('/api/system/gpus', headers=dashboard_auth))
                second = asyncio.create_task(client.get('/api/system/gpus', headers=dashboard_auth))
                try:
                    await asyncio.wait_for(both.wait(), 2)
                    await until(entered.is_set)
                    release.set()
                    fresh = await asyncio.wait_for(first, 2)
                    offset[0] = 60
                    late.set()
                    delayed = await asyncio.wait_for(second, 2)
                    assert count[0] == 1
                    assert delayed.headers['X-KZA-Observed-At'] == fresh.headers['X-KZA-Observed-At']
                    assert delayed.headers['X-KZA-Expires-At'] == fresh.headers['X-KZA-Expires-At']
                    assert delayed.headers['X-KZA-Observation-State'] == 'stale'
                finally:
                    release.set()
                    late.set()
                    await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.parametrize('fault', ['serve', 'catalog'])
async def test_dashboard_stop_drains_native_monitor_after_an_earlier_close_failure(dashboard_auth, monkeypatch, fault):
    entered, release = Event(), Event()

    def probe():
        entered.set()
        release.wait()
        return []

    async def fail():
        raise RuntimeError('synthetic shutdown fault')

    monkeypatch.setattr(system_monitor, 'gpu_snapshot', probe)
    api = DashboardAPI()
    original_catalog_close = api.home_catalog.aclose
    async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://testserver') as client:
        request = asyncio.create_task(client.get('/api/system/gpus', headers=dashboard_auth))
        closing = None
        try:
            await until(entered.is_set)
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            if fault == 'serve':
                api._serve_task = asyncio.create_task(fail())
                await asyncio.gather(api._serve_task, return_exceptions=True)
            else:
                api.home_catalog.aclose = fail
            closing = asyncio.create_task(api.stop())
            await until(lambda: api.app.state.observability_monitors._closed)
            assert not closing.done()
            release.set()
            with pytest.raises(RuntimeError, match='synthetic shutdown fault'):
                await closing
            assert api.app.state.observability_monitors._tasks == {}
        finally:
            release.set()
            await asyncio.gather(request, *(tuple([closing]) if closing is not None else ()), return_exceptions=True)
            api._serve_task = None
            api.home_catalog.aclose = original_catalog_close
            await api.stop()
