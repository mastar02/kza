"""Native complete() wire, validation, privacy admission and resource ownership."""

import asyncio
import json
from unittest.mock import MagicMock

import httpx
import pytest

from src.core.egress import EgressDenied, EgressPolicy
from src.llm.idle_watchdog import IdleTimeoutError
from tests.mocks.llm_http import SSEBody, event
from tests.unit.llm.test_http_async_stream import setup

pytestmark = pytest.mark.asyncio


def reply(text=' <think>hidden</think>Respuesta completa. ', *, style='chat', finish='stop', usage=7):
    choice = {'index': 0, 'finish_reason': finish}
    choice.update({'message': {'content': text}} if style == 'chat' else {'text': text})
    return {'choices': [choice], 'usage': {'completion_tokens': usage}}


@pytest.mark.parametrize('style', ['chat', 'completions'])
async def test_nonstream_wire_and_provider_metrics_without_sync_load(monkeypatch, style):
    reasoner, endpoint = setup(monkeypatch, style=style)
    endpoint.completion_payload = reply(style=style)
    reasoner.load = MagicMock(side_effect=AssertionError('sync load must never run'))
    reasoner._metrics_tracker = MagicMock()
    reasoner._endpoint_id = 'synthetic'
    assert await reasoner.complete('consulta', max_tokens=123, temperature=.2) == 'Respuesta completa.'
    payload = json.loads(endpoint.requests[-1].content)
    assert endpoint.requests[-1].url.path == ('/v1/chat/completions' if style == 'chat' else '/v1/completions')
    assert payload == {'model': 'configured-model', 'max_tokens': 123, 'temperature': .2, 'stream': False,
                       **({'messages': [{'role': 'user', 'content': 'consulta'}]} if style == 'chat' else {'prompt': 'consulta'})}
    assert reasoner._last_metrics['tokens'] == 7 and reasoner._last_metrics['ms'] >= 0
    reasoner._metrics_tracker.record.assert_called_once_with('synthetic', 7, reasoner._last_metrics['ms'])
    assert endpoint.client.is_closed()
    reasoner.load.assert_not_called()


@pytest.mark.parametrize('usage,expected', [(0, 0), (None, None), (-1, None), (True, None), ('7', None)])
async def test_invalid_usage_never_reuses_old_metrics_or_guesses_tokens(monkeypatch, usage, expected):
    reasoner, endpoint = setup(monkeypatch)
    endpoint.completion_payload = reply(usage=usage)
    reasoner._last_metrics = {'tokens': 999, 'ms': 1}
    assert await reasoner.complete('consulta') == 'Respuesta completa.'
    if expected is None:
        assert reasoner._last_metrics is None
    else:
        assert reasoner._last_metrics['tokens'] == expected


@pytest.mark.parametrize('payload', [
    [], {}, {'error': {'message': 'private provider detail'}}, {'choices': []},
    {'choices': [None]}, {'choices': [{}, {}]},
    {'choices': [{'finish_reason': 'stop', 'message': []}]},
    reply(finish='length'), reply(finish=None), reply(finish='tool_calls'),
    reply(text=None), reply(text=[]), reply(text='\ud800'),
    reply(text='x'*65537),
])
async def test_invalid_or_truncated_response_never_returns_text(monkeypatch, payload):
    reasoner, endpoint = setup(monkeypatch)
    endpoint.completion_payload = payload
    with pytest.raises((ValueError, RuntimeError)):
        await reasoner.complete('consulta')
    assert reasoner._last_metrics is None
    assert endpoint.client.is_closed()


@pytest.mark.parametrize('mutate', ['consent', 'model'])
async def test_complete_rechecks_admission_after_discovery(monkeypatch, mutate):
    reasoner, endpoint = setup(monkeypatch)
    def change():
        if mutate == 'consent':
            reasoner.policy = EgressPolicy.for_http(reasoner.base_url, 'cloud', False)
        else:
            reasoner.model = 'other'
    endpoint.on_models = change
    with pytest.raises((EgressDenied, RuntimeError)):
        await reasoner.complete('must not send')
    assert [request.method for request in endpoint.requests] == ['GET']
    assert endpoint.client.is_closed()


async def test_nonstream_cancellation_waits_actual_body_close(monkeypatch):
    reasoner, endpoint = setup(monkeypatch)
    body = SSEBody([asyncio.Event()])
    closing, release = asyncio.Event(), asyncio.Event()
    original_close = body.aclose
    async def close():
        closing.set()
        await release.wait()
        await original_close()
    body.aclose = close
    handle = endpoint.handle
    async def stalled(request):
        if request.method == 'POST':
            endpoint.requests.append(request)
            return httpx.Response(200, headers={'content-type': 'application/json'}, stream=body)
        return await handle(request)
    monkeypatch.setattr(endpoint, 'handle', stalled)
    task = asyncio.create_task(reasoner.complete('consulta'))
    await asyncio.wait_for(body.waiting.wait(), 1)
    task.cancel()
    await asyncio.wait_for(closing.wait(), 1)
    task.cancel()
    await asyncio.sleep(.02)
    assert not task.done() and not body.closed and not endpoint.client.is_closed()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert body.closed and endpoint.client.is_closed()


async def test_nonstream_wire_size_limit_closes_body(monkeypatch):
    reasoner, endpoint = setup(monkeypatch)
    body = SSEBody([b'x'*800000])
    handle = endpoint.handle
    async def oversized(request):
        if request.method == 'POST':
            return httpx.Response(200, stream=body)
        return await handle(request)
    monkeypatch.setattr(endpoint, 'handle', oversized)
    with pytest.raises(ValueError, match='generation_metadata_limit'):
        await reasoner.complete('consulta')
    assert body.closed and endpoint.client.is_closed()


@pytest.mark.parametrize('timeout', [False, True])
async def test_idle_complete_requires_stop_and_closes_stalled_or_incomplete_stream(monkeypatch, timeout):
    parts = [event('partial'), asyncio.Event()] if timeout else [event('partial')]
    reasoner, endpoint = setup(monkeypatch, parts, idle_timeout_s=.02)
    with pytest.raises(IdleTimeoutError if timeout else RuntimeError):
        await reasoner.complete('consulta')
    assert endpoint.body.closed and endpoint.client.is_closed()
