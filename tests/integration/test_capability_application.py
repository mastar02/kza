"""Voice and authenticated Casa share the actual application's real HA reader."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict
from datetime import datetime
import json

from httpx import ASGITransport, AsyncClient
import pytest

from src.application import ApplicationFactories
from src.home_assistant.ha_client import HomeAssistantClient
from src.main import build_application
from src.nlu.home_command import HomeCommand
from tests.integration.test_capture_producer_privacy import until
from tests.mocks.application_graph import OfflineEncoder, OfflineSTT, OfflineTTS, offline_config
from tests.mocks.audio_output import AudioBackend
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    import src.tts.piper_tts as audio
    import src.vectordb.chroma_sync as index
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(index, 'SentenceTransformer', lambda *a, **kw: OfflineEncoder())
    monkeypatch.setattr(audio, 'sd', AudioBackend())
    return ApplicationFactories(stt=lambda _: OfflineSTT(), tts=lambda _: OfflineTTS(), preflight=lambda: None)


def configuration(root, url):
    config = offline_config(root, url)
    path = root/'selection.json'
    path.write_text(json.dumps({'selection_approved': True, 'live_api_verified': True, 'allowed_entities': [{
        'entity_id': 'light.living', 'operations': ['turn_on', 'turn_off', 'brightness'],
        'parameters': {'brightness_pct': {'min': 20, 'max': 60}},
        'live_verified': True, 'indirect_effects_verified': True}]}))
    config['home_assistant']['control_selection_path'] = str(path)
    return config, path


async def test_actual_factory_voice_and_casa_share_one_observation_and_limits(leaves, tmp_path, dashboard_auth):
    transport = TaskTransport()
    transport.states['light.living']['attributes']['supported_color_modes'] = ['brightness']
    async with transport.serve() as url:
        config, selection = configuration(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            reader = graph.home_capabilities
            resolver = graph.pipeline.request_router.home_voice_resolver
            assert reader is resolver.capabilities is graph.home_services.catalog.capabilities
            original = reader.snapshot
            calls, both_entered = [], asyncio.Event()

            async def observed():
                calls.append(True)
                if len(calls) == 2:
                    both_entered.set()
                return await original()

            reader.snapshot = observed
            transport.states_hold = True
            before = transport.state_reads
            command = HomeCommand('light', '', 'brightness', {'brightness_pct': 30})
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
                voice = asyncio.create_task(resolver.resolve(command, room_id=None))
                web = asyncio.create_task(client.get('/api/home/catalog', headers=dashboard_auth))
                await asyncio.wait_for(both_entered.wait(), 2)
                await asyncio.wait_for(transport.states_entered.wait(), 2)
                assert transport.state_reads == before+1
                transport.states_release.set()
                target, response = await asyncio.wait_for(asyncio.gather(voice, web), 3)
                assert response.status_code == 200 and target.entity_id == 'light.living'
                entity = next(value for value in response.json()['entities'] if value['entity_id'] == target.entity_id)
                assert entity['capabilities'] == list(target.capability.effective_operations(reader.now()))
                assert entity['control_limits'] == json.loads(json.dumps(asdict(target.capability.limits)))
                observation = entity['observation']
                for key in ('observed_at', 'expires_at', 'attempted_at'):
                    if observation[key] is not None:
                        observation[key] = observation[key].replace('Z', '+00:00')
                assert observation == json.loads(json.dumps(target.observation.to_dict(reader.now())))
                payload = json.loads(selection.read_text()); payload['allowed_entities'] = []
                selection.write_text(json.dumps(payload))
                target = await resolver.resolve(command, room_id=None)
                response = await client.get('/api/home/catalog', headers=dashboard_auth)
                assert target.entity_id is None and target.reason == 'no_selected_target'
                assert all(not value['capabilities'] for value in response.json()['entities'])
                assert transport.sends == []


@pytest.mark.parametrize('status', [401, 403, 503])
async def test_failed_real_rest_read_preserves_original_time_and_disables_both_consumers(
    leaves, tmp_path, dashboard_auth, status,
):
    transport = TaskTransport()
    transport.states['light.living']['attributes']['supported_color_modes'] = ['brightness']
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            first = await graph.home_capabilities.snapshot()
            transport.states_status = status
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
                response = await client.get('/api/home/catalog', headers=dashboard_auth)
                assert response.status_code == 200
                value = response.json()
                assert value['stale'] and datetime.fromisoformat(value['observation']['observed_at']) == first.observation.observed_at
                assert datetime.fromisoformat(value['observation']['expires_at']) == first.observation.expires_at
                assert value['observation']['state'] == ('denied' if status in (401, 403) else 'unavailable')
                assert all(not entity['available'] and not entity['capabilities'] for entity in value['entities'])
            target = await graph.pipeline.request_router.home_voice_resolver.resolve(HomeCommand('light', '', 'brightness', {'brightness_pct': 30}), room_id=None)
            assert target.entity_id is None and target.reason == 'capability_unavailable'
            assert transport.sends == []


async def test_graph_close_waits_for_cancelled_consumers_real_rest_read_before_closing_ha(leaves, tmp_path, monkeypatch):
    transport = TaskTransport()
    closed, order = asyncio.Event(), []
    original = HomeAssistantClient.close
    reader = None

    async def close(ha):
        order.append(reader._refresh_task.done())
        closed.set()
        await original(ha)

    monkeypatch.setattr(HomeAssistantClient, 'close', close)
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        stack = AsyncExitStack()
        graph = await build_application(config, shutdown=stack, factories=leaves)
        reader = graph.home_capabilities
        transport.states_hold = True
        waiter = asyncio.create_task(reader.snapshot())
        await asyncio.wait_for(transport.states_entered.wait(), 2)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        closing = asyncio.create_task(stack.aclose())
        await until(lambda: reader._closed)
        assert not closed.is_set() and not closing.done()
        transport.states_release.set()
        await asyncio.wait_for(closing, 3)
        assert closed.is_set() and order == [True]
        with pytest.raises(RuntimeError, match='capabilities_closed'):
            await reader.snapshot()
