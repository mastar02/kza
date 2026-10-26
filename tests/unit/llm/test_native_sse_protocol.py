"""Native SSE framing, bounded data and explicit response/client cleanup."""

from contextlib import aclosing
import json

import httpx
import pytest

from src.llm.http_generation import HTTPGenerationClient
from tests.mocks.llm_http import event
from tests.unit.llm.test_http_async_stream import collect, setup

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('separator', [b'\n', b'\r\n', b'\r'])
async def test_fragmented_utf8_and_sse_line_endings_preserve_all_text(monkeypatch, separator):
    text = 'Una respuesta con ñ, música y 🌿.'
    payload = {'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]}
    first = b'\xef\xbb\xbf: comentario'+separator+b'id: synthetic'+separator
    first += b'data: '+json.dumps(payload, ensure_ascii=False).encode()+separator+separator
    final = event(finish='stop').replace(b'\n', separator)
    reasoner, endpoint = setup(monkeypatch, [bytes([byte]) for byte in first+final])
    assert ''.join(chunk['token'] for chunk in await collect(reasoner)) == text
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_multiline_data_comments_and_unknown_fields(monkeypatch):
    packet = b': keepalive\nretry: 100\nevent: message\ndata: {"choices":\ndata: [{"index":0,"delta":{"content":"Texto completo."}}]}\n\n'
    reasoner, endpoint = setup(monkeypatch, [packet, event(finish='stop'), b': final comment'])
    assert (await collect(reasoner))[-1]['text'] == 'Texto completo.'
    assert endpoint.body.closed and endpoint.client.is_closed()


@pytest.mark.parametrize('payload', [
    [], {'error': {'message': 'provider details must not become output'}},
    {}, {'choices': None}, {'choices': [None]}, {'choices': [{}, {}]},
    {'choices': [{'index': 1}]}, {'choices': [{'index': True}]},
    {'choices': [{'delta': []}]}, {'choices': [{'delta': {'content': {}}}]},
    {'choices': [{'delta': {'content': 23}}]},
    {'choices': [{'delta': {'content': '\ud800'}}]},
    {'choices': [{'finish_reason': 23}]},
])
async def test_invalid_provider_payload_closes_every_resource(monkeypatch, payload):
    reasoner, endpoint = setup(monkeypatch, [b'data: '+json.dumps(payload).encode()+b'\n\n'])
    with pytest.raises((ValueError, RuntimeError)):
        await collect(reasoner)
    assert endpoint.body.closed and endpoint.client.is_closed()
    assert sum(request.method == 'POST' for request in endpoint.requests) == 1


@pytest.mark.parametrize('packet', [b'data: {invalid json}\n\n', b'data: \xff\n\n', b'data: {"choices": []}\n'])
async def test_bad_json_utf8_or_unterminated_event_is_not_success(monkeypatch, packet):
    reasoner, endpoint = setup(monkeypatch, [packet])
    with pytest.raises(ValueError):
        await collect(reasoner)
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_done_without_terminal_stop_is_incomplete(monkeypatch):
    reasoner, endpoint = setup(monkeypatch, [event('Un texto con tail.'), b'data: [DONE]\n\n'])
    with pytest.raises(RuntimeError, match='generation_incomplete'):
        await collect(reasoner)
    assert endpoint.body.closed and endpoint.client.is_closed()


@pytest.mark.parametrize('finish', ['stop', 'length'])
async def test_data_after_terminal_reason_cannot_replace_or_extend_it(monkeypatch, finish):
    reasoner, endpoint = setup(monkeypatch, [event('Texto completo.'), event(finish=finish),
        event('texto posterior'), event(finish='stop')])
    with pytest.raises(ValueError, match='generation_after_finish'):
        await collect(reasoner)
    assert endpoint.body.closed and endpoint.client.is_closed()


@pytest.mark.parametrize('prefix', [b'data: ', b':'])
async def test_oversized_line_or_comment_is_bounded(monkeypatch, prefix):
    reasoner, endpoint = setup(monkeypatch, [prefix+b'x'*65537])
    with pytest.raises(ValueError, match='generation_event_limit'):
        await collect(reasoner, max_chars=10)
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_many_small_data_lines_cannot_bypass_event_limit(monkeypatch):
    reasoner, endpoint = setup(monkeypatch, [b'data: x\n'*10000])
    with pytest.raises(ValueError, match='generation_event_limit'):
        await collect(reasoner, max_chars=10)
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_unicode_json_escaping_does_not_shrink_the_text_budget(monkeypatch):
    text = '🌿'*6000
    reasoner, endpoint = setup(monkeypatch, [event(text), event(finish='stop')])
    assert (await collect(reasoner, max_chars=len(text)))[-1]['text'] == text
    assert endpoint.body.closed and endpoint.client.is_closed()


@pytest.mark.parametrize('headers', [
    {'content-type': 'application/json'},
    {'content-type': 'text/event-stream', 'content-encoding': 'gzip'},
])
async def test_unsupported_payload_encoding_or_mime_closes_stream(monkeypatch, headers):
    reasoner, endpoint = setup(monkeypatch)
    handle = endpoint.handle

    async def altered(request):
        if request.method == 'POST':
            endpoint.requests.append(request)
            return httpx.Response(200, headers=headers, stream=endpoint.body)
        return await handle(request)

    monkeypatch.setattr(endpoint, 'handle', altered)
    with pytest.raises(ValueError):
        await collect(reasoner)
    assert endpoint.body.closed and endpoint.client.is_closed()


@pytest.mark.parametrize('body', [b'x'*262145, b'{"data":[{"id":12}]}', b'{"data":null}'])
async def test_bad_or_oversized_model_discovery_never_sends_prompt(monkeypatch, body):
    reasoner, endpoint = setup(monkeypatch)

    async def discovery(request):
        endpoint.requests.append(request)
        return httpx.Response(200, content=body, headers={'content-type': 'application/json'})

    monkeypatch.setattr(endpoint, 'handle', discovery)
    with pytest.raises(ValueError):
        await collect(reasoner)
    assert [request.method for request in endpoint.requests] == ['GET']
    assert endpoint.client.is_closed()


async def test_request_headers_and_pinned_paths(monkeypatch):
    reasoner, endpoint = setup(monkeypatch)
    await collect(reasoner)
    assert all(request.headers['Authorization'] == 'Bearer synthetic-key' for request in endpoint.requests)
    assert all(request.headers['Accept-Encoding'] == 'identity' for request in endpoint.requests)
    assert endpoint.requests[-1].headers['Accept'] == 'text/event-stream'
    assert endpoint.requests[-1].url.path == '/v1/chat/completions'
    assert isinstance(endpoint.client, HTTPGenerationClient)


async def test_malformed_trailing_event_does_not_flush_unverified_tail(monkeypatch):
    reasoner, endpoint = setup(monkeypatch, [event('Una respuesta completa.'), event(finish='stop'), b'data: {'])
    chunks = []
    with pytest.raises(ValueError, match='generation_sse_incomplete'):
        async with aclosing(reasoner.astream('synthetic')) as stream:
            async for item in stream:
                chunks.append(item)
    assert ''.join(chunk['token'] for chunk in chunks) == 'Una respuesta completa.'[:-8]
    assert endpoint.body.closed and endpoint.client.is_closed()
