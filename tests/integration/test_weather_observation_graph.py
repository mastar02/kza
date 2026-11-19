"""The production weather route reads the same source as Casa, with real transport."""

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from unittest.mock import MagicMock

from httpx import ASGITransport, AsyncClient
import pytest

from src.main import build_application
from src.world.weather import NO_DATA, NO_FORECAST, NO_FORECAST_HORIZON
from tests.integration.test_capability_application import configuration, leaves as capability_leaves
from tests.integration.test_task_outcomes import event
from tests.mocks.task_transport import TaskTransport

pytestmark = pytest.mark.asyncio
WEATHER = 'weather.synthetic'
# Injected turn clock (T037-U1): 02:30Z on the 5th is 23:30 on the 4th in the
# household zone (UTC-3), so "mañana" is the 5th even though UTC says the 5th.
FROZEN = datetime(2026, 8, 5, 2, 30, tzinfo=UTC)
TODAY = {'datetime': '2026-08-04T15:00:00+00:00', 'condition': 'sunny', 'temperature': 22}
TOMORROW = {'datetime': '2026-08-05T15:00:00+00:00', 'condition': 'rainy', 'temperature': 25}
AFTER = {'datetime': '2026-08-06T15:00:00+00:00', 'condition': 'cloudy', 'temperature': 19}
YESTERDAY = {'datetime': '2026-08-03T15:00:00+00:00', 'condition': 'fog', 'temperature': 11}


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
async def test_forecast_retrieval_is_separate_from_issue_time_and_task_completion(leaves, tmp_path, monkeypatch, mode):
    transport = TaskTransport()
    forecast = [TODAY, TOMORROW]
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
            monkeypatch.setattr(graph.home_capabilities, '_clock', lambda: FROZEN)
            graph.models.stt.text = 'qué tiempo hace mañana'
            result = await graph.pipeline.process_command(event())
            assert result['intent'] == 'weather'
            assert result['source_observation']['source'] == 'ha-weather-forecast'
            assert result['task_completed'] == (mode == 'valid')
            if mode in {'valid', 'empty'}:
                assert result['source_observation']['observed_at'] == FROZEN.isoformat()
                assert result['source_observation']['unobserved_fields'] == (
                    ('issued_at',) if mode == 'valid' else ('issued_at', 'forecast_date_alignment'))
            else:
                assert result['source_observation']['observed_at'] is None
            if mode == 'valid':
                assert result['response'] == 'Mañana: lluvioso, 25 grados.'
            else:
                assert result['response'] in (NO_DATA, NO_FORECAST) and not result['success']
            assert len(transport.forecast_reads) == 1 and transport.sends == []


@pytest.mark.parametrize(('forecast', 'spoken', 'invalid'), [
    pytest.param([AFTER, TOMORROW, YESTERDAY, TODAY], 'Mañana: lluvioso, 25 grados.', False, id='reordered'),
    pytest.param([YESTERDAY, TODAY, TOMORROW], 'Mañana: lluvioso, 25 grados.', False, id='first-entry-yesterday'),
    pytest.param([TODAY, AFTER], None, False, id='missing-day'),
    pytest.param([TODAY, {**TOMORROW, 'datetime': '2026-08-05T00:00:00+00:00'}], None, False, id='utc-midnight-is-yesterday-local'),
    pytest.param([TODAY, {**AFTER, 'datetime': '2026-08-05T18:00:00+00:00'}, TOMORROW], None, False, id='conflicting-duplicates'),
    pytest.param([TODAY, {**TOMORROW, 'datetime': '2026-08-05T15:00:00'}], None, True, id='naive-date'),
    pytest.param([TODAY, {**TOMORROW, 'datetime': 'mañana'}], None, True, id='unparseable-date'),
])
async def test_forecast_day_is_selected_by_household_date_through_the_application(
        leaves, tmp_path, monkeypatch, forecast, spoken, invalid):
    transport = TaskTransport()
    transport.forecast_payload = {'service_response': {WEATHER: {'forecast': forecast}}}
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        config['home_assistant']['weather_entity'] = WEATHER
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            monkeypatch.setattr(graph.home_capabilities, '_clock', lambda: FROZEN)
            graph.models.stt.text = 'va a llover mañana'
            result = await graph.pipeline.process_command(event())
            assert result['intent'] == 'weather'
            observation = result['source_observation']
            assert observation['source'] == 'ha-weather-forecast' and observation['attempted_at'] == FROZEN.isoformat()
            assert result['task_completed'] is (spoken is not None)
            if spoken is not None:
                assert result['response'] == spoken
                assert observation['unobserved_fields'] == ('issued_at',)
            else:
                # PQ-T037-1: a missing or conflicting day names it; invalid payloads stay generic.
                assert result['response'] == (NO_FORECAST if invalid else 'No tengo el pronóstico para mañana.')
                assert not result['success'] and 'lluvioso' not in result['response']
                if invalid:
                    assert observation['reason'] == 'source_invalid' and observation['observed_at'] is None
                else:
                    assert observation['unobserved_fields'] == ('issued_at', 'forecast_date_alignment')
                    assert observation['observed_at'] == FROZEN.isoformat()
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


# T037-U2: one real-factory case per horizon (FROZEN is Tuesday 4 August, 23:30 local).
SATURDAY = {'datetime': '2026-08-08T15:00:00+00:00', 'condition': 'pouring', 'temperature': 12}


@pytest.mark.parametrize(('text', 'spoken', 'forecast_reads', 'completed'), [
    pytest.param('va a llover hoy', 'Hoy: soleado, 22 grados.', 1, True, id='hoy'),
    pytest.param('va a llover mañana', 'Mañana: lluvioso, 25 grados.', 1, True, id='manana'),
    pytest.param('va a llover esta mañana', 'Hoy: soleado, 22 grados.', 1, True, id='esta-manana'),
    pytest.param('va a llover por la mañana', 'Hoy: soleado, 22 grados.', 1, True, id='por-la-manana'),
    pytest.param('qué tiempo hace pasado mañana', 'Pasado mañana: nublado, 19 grados.', 1, True, id='pasado-manana'),
    pytest.param('va a llover el sábado', 'El sábado: lloviendo fuerte, 12 grados.', 1, True, id='el-sabado'),
    pytest.param('va a llover el viernes', 'No tengo el pronóstico para el viernes.', 1, False, id='weekday-missing'),
    pytest.param('va a llover la semana que viene', NO_FORECAST_HORIZON, 0, False, id='la-semana-que-viene'),
])
async def test_each_horizon_through_the_application(leaves, tmp_path, monkeypatch, text, spoken, forecast_reads,
                                                    completed):
    transport = TaskTransport()
    weather(transport)
    transport.forecast_payload = {'service_response': {WEATHER: {'forecast': [SATURDAY, AFTER, TOMORROW, TODAY]}}}
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        config['home_assistant']['weather_entity'] = WEATHER
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            monkeypatch.setattr(graph.home_capabilities, '_clock', lambda: FROZEN)
            graph.models.stt.text = text
            reads_before = transport.state_reads
            result = await graph.pipeline.process_command(event())
            assert result['intent'] == 'weather' and result['path'] == 'fast_weather'
            assert result['response'] == spoken
            assert result['task_completed'] is completed
            assert len(transport.forecast_reads) == forecast_reads and transport.sends == []
            # A forecast horizon never falls back to the current conditions.
            assert transport.state_reads == reads_before
            if forecast_reads:
                assert result['source_observation']['source'] == 'ha-weather-forecast'
            else:
                assert 'source_observation' not in result


async def test_present_tense_this_morning_reads_current_conditions_through_the_application(leaves, tmp_path,
                                                                                          monkeypatch):
    transport = TaskTransport()
    weather(transport)
    transport.forecast_payload = {'service_response': {WEATHER: {'forecast': [TODAY, TOMORROW]}}}
    async with transport.serve() as url:
        config, _ = configuration(tmp_path, url)
        config['home_assistant']['weather_entity'] = WEATHER
        async with AsyncExitStack() as stack:
            graph = await build_application(config, shutdown=stack, factories=leaves)
            monkeypatch.setattr(graph.home_capabilities, '_clock', lambda: FROZEN)
            graph.models.stt.text = 'qué tiempo hace esta mañana'
            result = await graph.pipeline.process_command(event())
            assert result['intent'] == 'weather' and '22 grados' in result['response']
            assert result['source_observation']['source'] == 'ha-rest-states'
            assert transport.forecast_reads == [] and transport.sends == []
