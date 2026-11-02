"""The production weather route reads the same source as Casa, with real transport."""

import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from httpx import ASGITransport, AsyncClient
import pytest

from src.main import build_application
from src.world.weather import NO_DATA, NO_FORECAST
from tests.integration.test_capability_application import configuration, leaves as capability_leaves
from tests.integration.test_task_outcomes import event
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio
WEATHER = 'weather.synthetic'


@pytest.fixture
def leaves(monkeypatch, tmp_path):
    return capability_leaves.__wrapped__(monkeypatch, tmp_path)


def weather(transport):
    transport.states[WEATHER] = {'entity_id': WEATHER, 'state': 'sunny', 'attributes': {'temperature': 22}}


async def test_current_weather_and_casa_share_the_real_read_not_an_unclocked_cache(leaves, tmp_path, dashboard_auth):
    transport = TaskTransport()
    weather(transport)
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        config['home_assistant']['weather_entity'] = WEATHER
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            reader = graph.home_capabilities
            assert graph.pipeline._orchestrator.home_capabilities is reader
            assert graph.pipeline._orchestrator.dispatcher.home_capabilities is reader
            graph.ha_client.get_entity_state_cached = MagicMock(side_effect=AssertionError('unclocked cache used'))
            graph.models.stt.text = 'qué tiempo hace hoy'
            original = reader.snapshot
            callers, both = 0, asyncio.Event()

            async def snapshot():
                nonlocal callers
                callers += 1
                if callers == 2:
                    both.set()
                return await original()

            reader.snapshot = snapshot
            transport.states_hold = True
            async with AsyncClient(transport=ASGITransport(app=graph.dashboard.app), base_url='http://testserver') as client:
                voice = asyncio.create_task(graph.pipeline.process_command(event()))
                web = asyncio.create_task(client.get('/api/home/catalog', headers=dashboard_auth))
                try:
                    await asyncio.wait_for(both.wait(), 3)
                    transport.states_release.set()
                    result, response = await asyncio.wait_for(asyncio.gather(voice, web), 3)
                    assert result['intent'] == 'weather' and result['task_completed']
                    assert '22 grados' in result['response']
                    observed = datetime.fromisoformat(result['source_observation']['observed_at'])
                    assert observed == datetime.fromisoformat(response.json()['observation']['observed_at'])
                    assert result['source_observation']['source'] == 'ha-rest-states'
                    graph.ha_client.get_entity_state_cached.assert_not_called()
                    assert transport.sends == []
                finally:
                    transport.states_release.set()
                    await asyncio.gather(voice, web, return_exceptions=True)


@pytest.mark.parametrize('missing', [False, True])
async def test_missing_exact_weather_or_lost_observation_cannot_fall_back_to_cache_or_another_entity(leaves, tmp_path, missing):
    transport = TaskTransport()
    weather(transport)
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        config['home_assistant']['weather_entity'] = 'weather.missing' if missing else WEATHER
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            await graph.home_capabilities.snapshot()
            graph.ha_client.get_entity_state_cached = MagicMock(return_value={'state': 'sunny', 'attributes': {'temperature': 88}})
            if not missing:
                transport.states_status = 503
            graph.models.stt.text = 'qué tiempo hace hoy'
            result = await graph.pipeline.process_command(event())
            assert result['intent'] == 'weather' and result['response'] == NO_DATA
            assert not result['task_completed'] and not result['success']
            assert '88' not in result['response']
            graph.ha_client.get_entity_state_cached.assert_not_called()


@pytest.mark.parametrize('mode', ['valid', 'empty', 'invalid', 'failed'])
async def test_forecast_retrieval_is_separate_from_issue_time_and_task_completion(leaves, tmp_path, mode):
    transport = TaskTransport()
    tomorrow = (datetime.now().date()+timedelta(days=1)).isoformat()+'T12:00:00+00:00'
    today = datetime.now().date().isoformat()+'T12:00:00+00:00'
    forecast = [{'datetime': today, 'condition': 'sunny', 'temperature': 22},
                {'datetime': tomorrow, 'condition': 'sunny', 'temperature': 25}]
    if mode == 'empty':
        forecast = []
    elif mode == 'invalid':
        forecast = 'invalid'
    transport.forecast_payload = {'service_response': {WEATHER: {'forecast': forecast}}}
    if mode == 'failed':
        transport.forecast_status = 503
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        config['home_assistant']['weather_entity'] = WEATHER
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            graph.models.stt.text = 'qué tiempo hace mañana'
            result = await graph.pipeline.process_command(event())
            assert result['intent'] == 'weather'
            assert result['source_observation']['source'] == 'ha-weather-forecast'
            assert result['task_completed'] == (mode == 'valid')
            if mode in {'valid', 'empty'}:
                assert result['source_observation']['observed_at'] is not None
                assert result['source_observation']['unobserved_fields'] == ('issued_at', 'forecast_date_alignment')
            else:
                assert result['source_observation']['observed_at'] is None
            if mode != 'valid':
                assert result['response'] in (NO_DATA, NO_FORECAST) and not result['success']
            assert len(transport.forecast_reads) == 1 and transport.sends == []


async def test_cancelled_forecast_query_has_one_attempt_and_no_detached_read(leaves, tmp_path):
    transport = TaskTransport(forecast_hold=True)
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        config['home_assistant']['weather_entity'] = WEATHER
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            graph.models.stt.text = 'qué tiempo hace mañana'
            request = asyncio.create_task(graph.pipeline.process_command(event()))
            try:
                await asyncio.wait_for(transport.forecast_entered.wait(), 3)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                assert len(transport.forecast_reads) == 1 and transport.sends == []
            finally:
                transport.forecast_release.set()
                await asyncio.gather(request, return_exceptions=True)
