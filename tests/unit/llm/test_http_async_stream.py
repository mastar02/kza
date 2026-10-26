"""Exercise native HTTP/SSE generation over an in-memory transport."""

import asyncio
import json
import threading
import time
from contextlib import aclosing

import httpx
import pytest

from src.core.egress import EgressDenied, EgressPolicy
from src.llm.idle_watchdog import IdleTimeoutError
from src.llm.reasoner import HttpReasoner
from tests.mocks.llm_http import Endpoint, SSEBody, event

pytestmark = pytest.mark.asyncio
ENDPOINT = "http://127.0.0.1:8200/v1"


def setup(monkeypatch, parts=None, *, style="chat", models=("configured-model",), model="configured-model",
          status=200, on_models=None, **kwargs):
    body = SSEBody(parts if parts is not None else [event("Hola"), event(finish="stop")])
    endpoint = Endpoint(body, models=models, status=status, on_models=on_models)
    reasoner = HttpReasoner(base_url=ENDPOINT, model=model, api_style=style, cloud_consent=True, **kwargs)
    monkeypatch.setattr(reasoner, "_new_async_client", endpoint.build)
    return reasoner, endpoint


async def collect(reasoner, **kwargs):
    async with aclosing(reasoner.astream("consulta sintética", **kwargs)) as stream:
        return [chunk async for chunk in stream]


@pytest.mark.parametrize("style", ["chat", "completions"])
async def test_wire_contract_and_owned_resources(monkeypatch, style):
    reasoner, endpoint = setup(monkeypatch, [
        event("Una respuesta ", style=style), event("completa.", style=style),
        event(finish="stop", style=style), event(choices=False, style=style), b"data: [DONE]\n\n",
    ], style=style)
    chunks = await collect(reasoner, max_tokens=321, temperature=0.2)
    assert "".join(chunk["token"] for chunk in chunks) == "Una respuesta completa."
    assert chunks[-1]["text"] == "Una respuesta completa."
    assert [chunk["token_count"] for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert [req.method for req in endpoint.requests] == ["GET", "POST"]
    payload = json.loads(endpoint.requests[-1].content)
    assert payload["model"] == "configured-model"
    assert payload["max_tokens"] == 321 and payload["temperature"] == 0.2 and payload["stream"] is True
    if style == "chat":
        assert endpoint.requests[-1].url.path == "/v1/chat/completions"
        assert payload["messages"] == [{"role": "user", "content": "consulta sintética"}]
    else:
        assert endpoint.requests[-1].url.path == "/v1/completions"
        assert payload["prompt"] == "consulta sintética"
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_thinking_tags_across_every_character_never_escape(monkeypatch):
    text = "<think>private reasoning</think>Respuesta visible completa."
    reasoner, endpoint = setup(monkeypatch, [*(event(char) for char in text), event(finish="stop")])
    chunks = await collect(reasoner)
    assert "".join(chunk["token"] for chunk in chunks) == "Respuesta visible completa."
    assert all("private" not in chunk["text"] and "think" not in chunk["text"] for chunk in chunks)
    assert endpoint.body.closed


@pytest.mark.parametrize("models,model,expected", [
    (("configured-model", "other"), None, "configured-model"),
    (("other", "configured-model"), "configured-model", "configured-model"),
])
async def test_model_discovery(monkeypatch, models, model, expected):
    reasoner, endpoint = setup(monkeypatch, models=models, model=model)
    await collect(reasoner)
    assert json.loads(endpoint.requests[-1].content)["model"] == expected


@pytest.mark.parametrize("models,expected", [(("other",), "configured_model_unavailable"), ((), "endpoint_has_no_model")])
async def test_unavailable_model_does_not_send_prompt(monkeypatch, models, expected):
    reasoner, endpoint = setup(monkeypatch, models=models, model="configured-model" if models else None)
    with pytest.raises(RuntimeError, match=expected):
        await collect(reasoner)
    assert len(endpoint.requests) == 1 and endpoint.client.is_closed()


async def test_stale_sync_fallback_cannot_replace_explicit_model(monkeypatch):
    reasoner, endpoint = setup(monkeypatch)
    reasoner._resolved_base_url = ENDPOINT
    reasoner._resolved_model = "old-fallback"
    await collect(reasoner)
    assert [req.method for req in endpoint.requests] == ["GET", "POST"]
    assert json.loads(endpoint.requests[-1].content)["model"] == "configured-model"


async def test_matching_cached_model_avoids_discovery(monkeypatch):
    reasoner, endpoint = setup(monkeypatch)
    reasoner._resolved_base_url = ENDPOINT
    reasoner._resolved_model = "configured-model"
    await collect(reasoner)
    assert [req.method for req in endpoint.requests] == ["POST"]


async def test_consent_denied_before_factory(monkeypatch):
    reasoner, endpoint = setup(monkeypatch)
    reasoner.policy = EgressPolicy.for_http(ENDPOINT, cloud_consent=False)
    with pytest.raises(EgressDenied):
        await collect(reasoner)
    assert endpoint.client is None and endpoint.requests == []


async def test_consent_revoked_during_discovery_blocks_prompt(monkeypatch):
    reasoner, endpoint = setup(monkeypatch)
    endpoint.on_models = lambda: setattr(reasoner, "policy", EgressPolicy.for_http(ENDPOINT))
    with pytest.raises(EgressDenied):
        await collect(reasoner)
    assert len(endpoint.requests) == 1 and endpoint.client.is_closed()


@pytest.mark.parametrize("attribute,value", [("base_url", "http://127.0.0.1:8101/v1"), ("api_style", "completions"), ("model", "other")])
async def test_configuration_change_during_discovery_blocks_prompt(monkeypatch, attribute, value):
    reasoner, endpoint = setup(monkeypatch)
    endpoint.on_models = lambda: setattr(reasoner, attribute, value)
    with pytest.raises(RuntimeError, match="reasoner_configuration_changed"):
        await collect(reasoner)
    assert len(endpoint.requests) == 1 and endpoint.client.is_closed()


@pytest.mark.parametrize("finish,expected", [(None, "generation_incomplete"), ("length", "generation_truncated"), ("content_filter", "generation_incomplete")])
async def test_incomplete_answer_never_flushes_tail_or_retries(monkeypatch, finish, expected):
    reasoner, endpoint = setup(monkeypatch, [event("Respuesta larga con tail"), event(finish=finish)])
    emitted = []
    with pytest.raises(RuntimeError, match=expected):
        async with aclosing(reasoner.astream("consulta")) as stream:
            async for chunk in stream:
                emitted.append(chunk["token"])
    assert "".join(emitted) == "Respuesta larga con tail"[:-8]
    assert endpoint.body.closed and endpoint.client.is_closed()
    assert sum(req.method == "POST" for req in endpoint.requests) == 1


async def test_text_limit_closes_without_flushing(monkeypatch):
    reasoner, endpoint = setup(monkeypatch, [event("longer than limit"), event(finish="stop")])
    with pytest.raises(ValueError, match="generation_text_limit"):
        await collect(reasoner, max_chars=5)
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_idle_timeout_closes_pending_read(monkeypatch):
    reasoner, endpoint = setup(monkeypatch, [event("pending tail"), asyncio.Event()], idle_timeout_s=0.02)
    with pytest.raises(IdleTimeoutError) as error:
        await collect(reasoner)
    assert error.value.chunks_received == 1
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_cancel_pending_read_closes_without_tail(monkeypatch):
    reasoner, endpoint = setup(monkeypatch, [event("pending tail"), asyncio.Event()])
    task = asyncio.create_task(collect(reasoner))
    await asyncio.wait_for(endpoint.body.waiting.wait(), 1)
    started = time.monotonic()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert time.monotonic() - started < 0.1
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_early_consumer_close_releases_stream_and_client(monkeypatch):
    reasoner, endpoint = setup(monkeypatch, [event("long enough to emit text"), asyncio.Event()])
    async with aclosing(reasoner.astream("consulta")) as stream:
        assert (await anext(stream))["token"]
    assert endpoint.body.closed and endpoint.client.is_closed()


async def test_transport_error_is_single_attempt_and_closes(monkeypatch):
    reasoner, endpoint = setup(monkeypatch, [event("pending tail"), httpx.ReadError("synthetic")])
    with pytest.raises(httpx.ReadError):
        await collect(reasoner)
    assert endpoint.body.closed and endpoint.client.is_closed()
    assert sum(req.method == "POST" for req in endpoint.requests) == 1


@pytest.mark.parametrize("status", [307, 429, 500])
async def test_http_failure_never_redirects_or_retries(monkeypatch, status):
    reasoner, endpoint = setup(monkeypatch, status=status)
    with pytest.raises(httpx.HTTPStatusError):
        await collect(reasoner)
    assert endpoint.client.is_closed()
    assert len(endpoint.requests) == 2
    assert all(str(req.url).startswith(ENDPOINT) for req in endpoint.requests)


async def test_repeated_cancellation_during_constructor_cleans_resource(monkeypatch):
    reasoner, endpoint = setup(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def build(base_url):
        entered.set()
        assert release.wait(1)
        return endpoint.build(base_url)

    monkeypatch.setattr(reasoner, "_new_async_client", build)
    task = asyncio.create_task(collect(reasoner))
    while not entered.is_set():
        await asyncio.sleep(0.001)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert endpoint.client.is_closed() and endpoint.requests == []


@pytest.mark.parametrize("fail", [False, True])
async def test_transport_preparation_keeps_loop_available_and_closes_on_error(monkeypatch, fail):
    reasoner, endpoint = setup(monkeypatch)
    client = endpoint.build(ENDPOINT)

    class LazyClient:
        def prepare(self, style):
            time.sleep(0.2)
            if fail:
                raise RuntimeError("synthetic transport initialization failure")
            client.prepare(style)

        async def models(self):
            return await client.models()

        async def stream(self, *args, **kwargs):
            return await client.stream(*args, **kwargs)

        async def close(self):
            await client.close()

    monkeypatch.setattr(reasoner, "_new_async_client", lambda _: LazyClient())
    task = asyncio.create_task(collect(reasoner))
    beats = 0
    while not task.done():
        await asyncio.sleep(0.005)
        beats += 1
    if fail:
        with pytest.raises(RuntimeError, match="synthetic transport initialization failure"):
            await task
        assert endpoint.requests == []
    else:
        assert (await task)[-1]["text"] == "Hola"
        assert endpoint.body.closed
    assert beats >= 20 and client.is_closed()


async def test_stalled_network_keeps_five_ms_heartbeat_available(monkeypatch):
    """The H2 predeclared 200 ms stall, repeated ten times without network."""
    all_lags = []
    for _ in range(10):
        reasoner, endpoint = setup(monkeypatch, [0.2, event("respuesta completa"), event(finish="stop")])
        task = asyncio.create_task(collect(reasoner))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 0.005
        lags = []
        while not task.done():
            await asyncio.sleep(max(0, deadline - loop.time()))
            lags.append(max(0, loop.time() - deadline))
            deadline = loop.time() + 0.005
        assert (await task)[-1]["text"] == "respuesta completa"
        assert endpoint.body.closed and endpoint.client.is_closed()
        assert len(lags) >= 20
        all_lags.extend(lags)
    ordered = sorted(all_lags)
    assert ordered[int(0.95 * (len(ordered) - 1))] <= 0.025
    assert max(ordered) <= 0.075
